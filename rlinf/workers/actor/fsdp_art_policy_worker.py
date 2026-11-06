# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
)
from rlinf.utils.nested_dict_process import (
    process_nested_dict_for_train,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

_ART_LOSS_SPACES = ("velocity_linear_sigma", "velocity_uniform")
_ART_MASK_METRICS = ("sde_kl", "fixed_mse")
_ART_OUTPUT_GRADIENT_STATS = (
    "output_guidance_grad_sq",
    "output_restoration_grad_sq",
    "output_grad_inner",
)
_ART_DRIFT_REGIONS = ("full", "exec", "unexec", "future", "padding")


class EmbodiedARTFSDPPolicy(EmbodiedFSDPActor):
    """FSDP policy worker for DiffusionART."""

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, float]:
        """Report rollout metrics; ART builds guidance directly from rewards."""
        metrics = compute_rollout_metrics(self.rollout_batch)
        metrics.update(self._group_success_metrics)
        return metrics

    @Worker.timer("run_training")
    def run_training(self) -> dict[str, float]:
        """Prepare frozen ART targets and apply prompt-balanced updates."""
        generator = torch.Generator().manual_seed(
            self.cfg.actor.seed + self.version + self._rank
        )
        action_trajectories = self.cfg.actor.model.model_type == "openpi"
        if action_trajectories:
            preparation_metrics, has_guidance, windows = (
                self._prepare_art_trajectory_training_inputs(generator)
            )
            self._select_art_training_timesteps(generator)
        else:
            endpoints = self.rollout_batch["forward_inputs"]["art_endpoint"]
            rollout_size = endpoints.shape[0] * endpoints.shape[1]
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, torch.arange(rollout_size)
            )
            preparation_metrics, has_guidance = self._prepare_art_training_inputs(
                generator
            )
        error_keys = ("actor/branch_state_error_max", "actor/branch_image_error_max")
        if error_keys[0] in preparation_metrics:
            errors = torch.tensor(
                [preparation_metrics[key] for key in error_keys], device=self.device
            )
            torch.distributed.all_reduce(errors, op=torch.distributed.ReduceOp.MAX)
            preparation_metrics.update(zip(error_keys, errors.tolist()))
        active = torch.tensor(float(has_guidance), device=self.device)
        torch.distributed.all_reduce(active, op=torch.distributed.ReduceOp.MAX)
        has_guidance = bool(active.item())
        preparation_metrics["actor/skipped_zero_guidance"] = float(not has_guidance)
        replay_diagnostics = action_trajectories and self.cfg.algorithm.get(
            "art_replay_diagnostics", False
        )
        if self.is_weight_offloaded and (has_guidance or replay_diagnostics):
            self.load_param_and_grad(self.device)
        if replay_diagnostics:
            preparation_metrics.update(self._art_replay_metrics())
        if not has_guidance:
            clear_memory()
            return all_reduce_dict(
                preparation_metrics,
                op=torch.distributed.ReduceOp.AVG,
            )
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
        actor_reference = (
            self.cfg.algorithm.get("art_reference_source", "rollout") == "actor"
        )
        rollout_reference = self.rollout_batch["forward_inputs"].get("art_old_velocity")
        if actor_reference or rollout_reference is None:
            self._precompute_old_velocities()
        if replay_diagnostics:
            preparation_metrics.update(self._art_replay_metrics("actor/reference_pre"))

        if not action_trajectories:
            row_windows = _prompt_disjoint_windows(
                torch.arange(rollout_size) // int(self.cfg.algorithm.group_size),
                updates=int(self.cfg.algorithm.get("art_updates_per_rollout", 2)),
                order=torch.randperm(rollout_size, generator=generator).tolist(),
            )
            windows = tuple(
                torch.tensor(rows, dtype=torch.long) for rows in row_windows
            )

        self.model.train()
        metrics: dict[str, list[float]] = {}
        mask_counts = torch.zeros(4, device=self.device, dtype=torch.float64)
        for key, value in preparation_metrics.items():
            append_to_dict(metrics, {key: value})

        for update_index, window_rows in enumerate(windows, start=1):
            window_batch = self._index_nested(self.rollout_batch, window_rows)
            window_size = window_rows.numel()
            micro_batch_size = int(self.cfg.actor.micro_batch_size)
            expected_window_size = self.cfg.actor.global_batch_size // self._world_size
            assert window_size == expected_window_size, (
                f"ART window has {window_size} rows on this rank; expected "
                f"{expected_window_size} from actor.global_batch_size."
            )
            assert window_size % micro_batch_size == 0, (
                f"ART window size {window_size} must be divisible by "
                f"actor.micro_batch_size {micro_batch_size}."
            )
            loss_scale = 1.0
            if action_trajectories:
                valid_count = (
                    window_batch["forward_inputs"]["art_coordinate_mask"]
                    .flatten(1)
                    .any(1)
                    .sum()
                    .to(device=self.device, dtype=torch.float32)
                )
                torch.distributed.all_reduce(valid_count)
                if valid_count.item() == 0:
                    continue
                loss_scale = self._world_size * window_size / valid_count.item()
            micro_batches = split_dict_to_chunk(
                window_batch,
                window_size // micro_batch_size,
            )
            num_train_timesteps = window_batch["forward_inputs"]["art_states"].shape[1]
            self.gradient_accumulation = len(micro_batches) * num_train_timesteps

            self.optimizer.zero_grad()
            update_mask_counts = torch.zeros_like(mask_counts)
            backward_index = 0
            for micro_batch in micro_batches:
                forward_inputs = micro_batch["forward_inputs"]
                model_inputs = put_tensor_device(
                    self._model_forward_inputs(forward_inputs),
                    self.device,
                )
                guidance_direction = forward_inputs["art_guidance_direction"].to(
                    self.device
                )
                guidance_magnitude = forward_inputs["art_guidance_magnitude"].to(
                    self.device
                )
                for timestep_index in range(num_train_timesteps):
                    backward_index += 1
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(
                            backward_index == self.gradient_accumulation
                        ),
                    )
                    loss, step_metrics = self._art_forward_and_loss(
                        forward_inputs,
                        model_inputs,
                        guidance_direction,
                        guidance_magnitude,
                        timestep_index,
                    )
                    update_mask_counts += torch.tensor(
                        [
                            step_metrics.pop(f"actor/{key}_count")
                            for key in (
                                "active",
                                "masked",
                                "outward",
                                "movement_score_sum",
                            )
                        ],
                        device=self.device,
                    )
                    scaled_loss = loss * loss_scale / self.gradient_accumulation
                    if action_trajectories:
                        step_metrics["actor/art_loss"] *= loss_scale
                        step_metrics["actor/total_loss"] *= loss_scale
                    if replay_diagnostics:
                        for key in _ART_OUTPUT_GRADIENT_STATS:
                            step_metrics[f"actor/art_{key}"] *= (
                                loss_scale / self.gradient_accumulation
                            ) ** 2
                    with backward_ctx:
                        self.grad_scaler.scale(scaled_loss).backward()
                    append_to_dict(metrics, step_metrics)
                    del loss, scaled_loss, step_metrics

            self.torch_platform.empty_cache()
            grad_norm, lr_list = self.optimizer_step()
            self.lr_scheduler.step()
            mask_counts += update_mask_counts
            counts = update_mask_counts.clone()
            torch.distributed.all_reduce(counts)
            append_to_dict(
                metrics, _art_mask_count_metrics(counts, f"actor/u{update_index}")
            )
            if replay_diagnostics:
                append_to_dict(
                    metrics,
                    self._art_replay_metrics(
                        f"actor/drift_u{update_index}", reference=rollout_reference
                    ),
                )
                if actor_reference:
                    append_to_dict(
                        metrics,
                        self._art_replay_metrics(f"actor/update_u{update_index}"),
                    )
            append_to_dict(
                metrics,
                {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                },
            )

        self.optimizer.zero_grad()
        clear_memory()
        torch.distributed.all_reduce(mask_counts)
        append_to_dict(metrics, _art_mask_count_metrics(mask_counts, "actor"))
        mean_metrics = {key: np.mean(value) for key, value in metrics.items()}
        mean_metrics = all_reduce_dict(mean_metrics, op=torch.distributed.ReduceOp.AVG)
        if replay_diagnostics:
            guidance_sq, restoration_sq, inner = (
                mean_metrics.pop(f"actor/art_{key}")
                for key in _ART_OUTPUT_GRADIENT_STATS
            )
            mean_metrics["actor/art_output_restore_ratio"] = math.sqrt(
                restoration_sq / max(guidance_sq, 1e-30)
            )
            mean_metrics["actor/art_output_restore_cosine"] = inner / math.sqrt(
                max(guidance_sq * restoration_sq, 1e-30)
            )
        return mean_metrics

    def _art_covariance_options(self) -> dict:
        defaults = {
            "art_covariance_reduction": "unbiased",
            "art_direction_mode": "historical",
            "art_scale_mode": "unit_normalized_covariance",
            "art_guidance_gain": 1.0,
            "art_zero_signal_mode": "exact",
            "art_zero_signal_atol": 0.0,
            "art_eps": 1e-5,
        }
        return {
            (
                "reduction"
                if key == "art_covariance_reduction"
                else key.removeprefix("art_")
            ): self.cfg.algorithm.get(key, default)
            for key, default in defaults.items()
        }

    @torch.no_grad()
    def _prepare_art_trajectory_training_inputs(
        self, generator: torch.Generator
    ) -> tuple[dict[str, float], bool, tuple[torch.Tensor, ...]]:
        """Flatten the trained decisions and attach their frozen guidance.

        Returned windows index rows of the flattened ``self.rollout_batch``.
        """
        batch = self.rollout_batch
        forward_inputs = batch["forward_inputs"]
        endpoints = forward_inputs["art_endpoint"]
        group_size = int(self.cfg.algorithm.group_size)
        action_mask, _ = compute_loss_mask(batch["dones"])
        if action_mask.shape != endpoints.shape[:3]:
            raise ValueError(
                "ART termination masks must match the executed action chunks"
            )
        episode_returns = torch.where(action_mask, batch["rewards"], 0.0).sum((0, 2))
        if batch.get("loss_mask") is not None:
            action_mask = action_mask & batch["loss_mask"]
        state_sampling = self.cfg.algorithm.get("art_state_sampling", False)
        branch_index = (
            0
            if state_sampling
            else self.cfg.algorithm.get("art_branch_chunk_index", None)
        )
        preparation_metrics = {}
        if branch_index is not None:
            first_decision, decisions = branch_index, 1
            episode_returns = torch.where(
                action_mask[branch_index:], batch["rewards"][branch_index:], 0.0
            ).sum((0, 2))
            group_valid = (
                (
                    forward_inputs["art_branch_valid"][branch_index]
                    & action_mask[branch_index, :, 0]
                )
                .reshape(-1, group_size)
                .all(1)
            )
            selected_mask = (
                action_mask[branch_index : branch_index + 1]
                & (group_valid.repeat_interleave(group_size)[None, :, None])
            )
            preparation_metrics = {
                "actor/branch_chunk_index": float(branch_index),
                "actor/branch_valid_group_frac": group_valid.float().mean().item(),
                "actor/branch_state_error_max": forward_inputs[
                    "art_branch_state_error"
                ][branch_index]
                .max()
                .item(),
                "actor/branch_image_error_max": forward_inputs[
                    "art_branch_image_error"
                ][branch_index]
                .max()
                .item(),
            }
        else:
            first_decision = 0
            decisions = (
                1
                if self.cfg.algorithm.get("art_first_chunk_only", False)
                else endpoints.shape[0]
            )
            selected_mask = action_mask[:decisions]
        condition_ids = forward_inputs["art_state_id"][0] if state_sampling else None
        guidance = _art_trajectory_guidance(
            endpoints[first_decision : first_decision + decisions],
            episode_returns,
            selected_mask,
            group_size,
            condition_ids=condition_ids,
            **self._art_covariance_options(),
        )
        episodes = endpoints.shape[1]
        episode_windows = _prompt_disjoint_windows(
            torch.arange(episodes) // group_size
            if condition_ids is None
            else condition_ids.cpu(),
            updates=int(self.cfg.algorithm.get("art_updates_per_rollout", 2)),
            order=torch.randperm(episodes, generator=generator).tolist(),
        )
        windows = []
        for rows in episode_windows:
            indices = (
                torch.arange(decisions)[:, None] * episodes + torch.tensor(rows)[None]
            ).reshape(-1)
            windows.append(indices[torch.randperm(len(indices), generator=generator)])

        self.rollout_batch = process_nested_dict_for_train(
            batch, first_decision * episodes + torch.arange(decisions * episodes)
        )
        self.rollout_batch["forward_inputs"].update(
            art_guidance_direction=guidance["directions"].flatten(0, 1),
            art_guidance_magnitude=guidance["magnitudes"].flatten(0, 1),
            art_coordinate_mask=guidance["coordinate_mask"].flatten(0, 1),
        )
        return (
            {
                **preparation_metrics,
                **_art_guidance_metrics(
                    guidance["stats"],
                    guidance["directions"].flatten(0, 1),
                    guidance["magnitudes"].flatten(0, 1),
                    episode_returns,
                    guidance["coordinate_mask"].flatten(0, 1),
                ),
                "actor/valid_chunk_frac": selected_mask.any(-1).float().mean().item(),
                "actor/valid_action_frac": selected_mask.float().mean().item(),
            },
            bool(guidance["stats"]["group_active"].any()),
            tuple(windows),
        )

    @torch.no_grad()
    def _prepare_art_training_inputs(
        self,
        generator: torch.Generator,
    ) -> tuple[dict[str, float], bool]:
        forward_inputs = self.rollout_batch["forward_inputs"]
        endpoints = forward_inputs["art_endpoint"]
        row_count = endpoints.shape[0]
        group_size = int(self.cfg.algorithm.group_size)
        group_count = row_count // group_size
        groups = torch.arange(row_count, device=endpoints.device) // group_size

        rewards = self.rollout_batch["rewards"].float().reshape(row_count, -1)
        reward_reduction = self.cfg.algorithm.get("art_reward_reduction", "mean")
        rewards = (
            rewards.sum(dim=1) if reward_reduction == "sum" else rewards.mean(dim=1)
        )
        guidance_stats = _art_scalar_covariance(
            endpoints, rewards, groups, group_count, **self._art_covariance_options()
        )
        eps = float(self.cfg.algorithm.get("art_eps", 1e-5))
        group_directions = guidance_stats["group_directions"]
        group_direction_rms = (
            group_directions.float().flatten(1).square().mean(1).sqrt()
        )
        normalized_group_directions = group_directions / group_direction_rms.clamp_min(
            eps
        ).reshape((-1,) + (1,) * (group_directions.ndim - 1))
        forward_inputs["art_guidance_direction"] = normalized_group_directions[groups]
        forward_inputs["art_guidance_magnitude"] = guidance_stats[
            "group_guidance_magnitudes"
        ][groups]

        self._select_art_training_timesteps(generator)

        return (
            _art_guidance_metrics(
                guidance_stats,
                forward_inputs["art_guidance_direction"],
                forward_inputs["art_guidance_magnitude"],
                rewards,
            ),
            bool(guidance_stats["group_active"].any()),
        )

    @torch.no_grad()
    def _select_art_training_timesteps(self, generator: torch.Generator) -> None:
        forward_inputs = self.rollout_batch["forward_inputs"]
        states = forward_inputs["art_states"]
        sigmas = forward_inputs["art_sigmas"]
        sigmas_next = forward_inputs["art_sigmas_next"]
        _validate_art_rollout(
            states, sigmas, sigmas_next, forward_inputs["art_guidance_magnitude"]
        )
        row_count, timestep_pool = states.shape[:2]
        if (
            self.cfg.algorithm.get("art_exclude_final_step", False)
            and timestep_pool > 1
        ):
            timestep_pool -= 1
        fraction = float(self.cfg.algorithm.get("art_timestep_fraction", 1.0))
        train_timestep_count = max(1, int(timestep_pool * fraction))
        selected = torch.stack(
            [
                torch.randperm(timestep_pool, generator=generator)[
                    :train_timestep_count
                ]
                for _ in range(row_count)
            ]
        )
        for key in ("art_states", "art_old_velocity"):
            if key not in forward_inputs:
                continue
            value = forward_inputs[key]
            if value.shape[:2] != states.shape[:2]:
                raise ValueError(
                    f"ART {key} must have matching sample and timestep axes"
                )
            indices = (
                selected.to(value.device)
                .view(row_count, train_timestep_count, *([1] * (value.ndim - 2)))
                .expand(-1, -1, *value.shape[2:])
            )
            forward_inputs[key] = value.gather(1, indices)
        forward_inputs["art_sigmas"] = sigmas.gather(1, selected.to(sigmas.device))
        forward_inputs["art_sigmas_next"] = sigmas_next.gather(
            1, selected.to(sigmas_next.device)
        )
        del forward_inputs["art_endpoint"]

    @torch.no_grad()
    def _art_replay_metrics(
        self,
        prefix: str = "actor/replay",
        *,
        reference: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Measure full-output drift on fixed rollout states, pooled across ranks.

        The default prefix reports pre-update replay error. Post-update probes
        reuse the same rows and denoising timesteps, including zero-guidance and
        corrective-masked samples. Rows without valid executed actions are omitted.
        """
        forward_inputs = self.rollout_batch["forward_inputs"]
        if "art_old_velocity" not in forward_inputs and reference is None:
            raise ValueError(
                "ART replay diagnostics require cached rollout velocities."
            )
        old_velocities = (
            forward_inputs["art_old_velocity"] if reference is None else reference
        )
        action_chunk = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        coordinate_mask = forward_inputs.get("art_coordinate_mask")
        stats = torch.zeros(len(_ART_DRIFT_REGIONS), 2, device=self.device)
        max_error = torch.zeros((), device=self.device)
        for rows, timestep_index, velocity in self._replay_velocities(forward_inputs):
            batch_stats, batch_max = _art_velocity_drift_stats(
                velocity,
                old_velocities[rows, timestep_index].to(self.device),
                action_chunk,
                action_dim,
                None
                if coordinate_mask is None
                else coordinate_mask[rows].to(self.device),
            )
            stats += batch_stats
            max_error = torch.maximum(max_error, batch_max)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
            torch.distributed.all_reduce(max_error, op=torch.distributed.ReduceOp.MAX)
        mean_square = stats[:, 0] / stats[:, 1].clamp_min(1)
        return {
            f"{prefix}_mse": mean_square[1].item(),
            f"{prefix}_max_abs": max_error.item(),
            **{
                f"{prefix}_{region}_rms": rms
                for region, rms in zip(_ART_DRIFT_REGIONS, mean_square.sqrt().tolist())
            },
        }

    @torch.no_grad()
    def _precompute_old_velocities(self) -> None:
        forward_inputs = self.rollout_batch["forward_inputs"]
        old_velocities = torch.empty_like(forward_inputs["art_states"])
        for rows, timestep_index, velocity in self._replay_velocities(forward_inputs):
            old_velocities[rows, timestep_index] = velocity.to(old_velocities.device)
        forward_inputs["art_old_velocity"] = old_velocities
        self.torch_platform.empty_cache()

    @torch.no_grad()
    def _replay_velocities(self, forward_inputs: dict):
        """Yield eval-mode velocities at cached states per microbatch and timestep.

        Yields ``(rows, timestep_index, velocity)`` with ``rows`` a slice into
        ``forward_inputs``; the model's training mode is restored afterwards.
        """
        states = forward_inputs["art_states"]
        sigmas = forward_inputs["art_sigmas"]
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        row_count, timestep_count = states.shape[:2]
        was_training = self.model.training
        self.model.eval()
        try:
            for start in range(0, row_count, micro_batch_size):
                rows = slice(start, min(start + micro_batch_size, row_count))
                model_inputs = put_tensor_device(
                    self._model_forward_inputs(
                        forward_inputs, torch.arange(rows.start, rows.stop)
                    ),
                    self.device,
                )
                for timestep_index in range(timestep_count):
                    with self.amp_context:
                        output = self.model(
                            forward_type=ForwardType.NFT,
                            forward_inputs=model_inputs,
                            nft_inputs={
                                "x_t": states[rows, timestep_index].to(self.device),
                                "timesteps": sigmas[rows, timestep_index].to(
                                    self.device
                                ),
                            },
                            compute_values=False,
                        )
                    yield rows, timestep_index, output["v_theta"].detach()
        finally:
            self.model.train(was_training)

    def _art_forward_and_loss(
        self,
        forward_inputs: dict[str, torch.Tensor],
        model_inputs: dict[str, torch.Tensor],
        guidance_direction: torch.Tensor,
        guidance_magnitude: torch.Tensor,
        timestep_index: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        states = forward_inputs["art_states"][:, timestep_index].to(self.device)
        sigmas = forward_inputs["art_sigmas"][:, timestep_index].to(self.device)
        sigmas_next = forward_inputs["art_sigmas_next"][:, timestep_index].to(
            self.device
        )
        old_velocity = forward_inputs["art_old_velocity"][:, timestep_index].to(
            self.device
        )

        with self.amp_context:
            output = self.model(
                forward_type=ForwardType.NFT,
                forward_inputs=model_inputs,
                nft_inputs={"x_t": states, "timesteps": sigmas},
                compute_values=False,
            )
        coordinate_mask = forward_inputs.get("art_coordinate_mask")
        velocity = output["v_theta"]
        if coordinate_mask is not None:
            coordinate_mask = coordinate_mask.to(self.device)
            chunk, dims = guidance_direction.shape[1:]
            velocity = velocity[:, :chunk, :dims]
            old_velocity = old_velocity[:, :chunk, :dims]
        output_gradient_stats = coordinate_mask is not None and bool(
            self.cfg.algorithm.get("art_replay_diagnostics", False)
        )
        terms = _art_loss(
            velocity,
            old_velocity,
            sigmas,
            sigmas_next,
            guidance_direction,
            guidance_magnitude,
            loss_space=self.cfg.algorithm.get(
                "art_loss_space", "velocity_linear_sigma"
            ),
            mask_metric=self.cfg.algorithm.get("art_mask_metric", "sde_kl"),
            mask_threshold=float(self.cfg.algorithm.get("art_mask_threshold", 1e-5)),
            reference_noise_level=float(
                self.cfg.algorithm.get("art_reference_noise_level", 0.8)
            ),
            corrective_mask_enabled=bool(
                self.cfg.algorithm.get("art_corrective_mask_enabled", True)
            ),
            coordinate_mask=coordinate_mask,
            output_gradient_stats=output_gradient_stats,
        )
        loss = terms["per_item_loss"].mean()
        active = terms["active"]
        active_count = active.sum().clamp_min(1)
        metrics = {
            "actor/active_count": active.sum().item(),
            "actor/masked_count": terms["masked"].sum().item(),
            "actor/outward_count": ((terms["outward_alignment"] > 0) & active)
            .sum()
            .item(),
            "actor/movement_score_sum_count": (terms["movement_score"] * active)
            .sum()
            .item(),
            "actor/art_loss": loss.detach().item(),
            "actor/total_loss": loss.detach().item(),
            "actor/unmasked_loss": (
                (terms["per_item_unmasked_loss"] * active).sum() / active_count
            ).item(),
            "actor/mask_frac": (terms["masked"].float().sum() / active_count).item(),
            "actor/outward_frac": (
                ((terms["outward_alignment"] > 0) & active).float().sum() / active_count
            ).item(),
            "actor/old_deviate": terms["movement_energy"].mean().item(),
            "actor/movement_score": terms["movement_score"].mean().item(),
        }
        if output_gradient_stats:
            metrics.update(
                {
                    f"actor/art_{key}": terms[key].item()
                    for key in _ART_OUTPUT_GRADIENT_STATS
                }
            )
        return loss, metrics

    @classmethod
    def _index_nested(cls, value, indices: torch.Tensor):
        if isinstance(value, torch.Tensor):
            return value.index_select(0, indices.to(value.device))
        if isinstance(value, dict):
            return {
                key: cls._index_nested(nested_value, indices)
                for key, nested_value in value.items()
            }
        return value

    @classmethod
    def _model_forward_inputs(
        cls,
        forward_inputs: dict,
        indices: torch.Tensor | None = None,
    ) -> dict:
        model_input_keys = {
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "latent_condition",
            "first_frame_mask",
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "obs_state",
        }
        inputs = {
            key: value
            for key, value in forward_inputs.items()
            if key in model_input_keys
            or key.startswith(("observation/", "obs_image__", "obs_image_mask__"))
        }
        return inputs if indices is None else cls._index_nested(inputs, indices)


@torch.no_grad()
def _art_velocity_drift_stats(
    velocity: torch.Tensor,
    old_velocity: torch.Tensor,
    action_chunk: int,
    action_dim: int,
    coordinate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return squared-error sums/counts per output region and executed max error.

    Unexecuted coordinates include future action tokens, padding dimensions and
    any unexecuted tail of a terminated chunk. Entirely invalid rows are omitted.
    """
    _validate_art_same_shape(velocity, old_velocity=old_velocity)
    error = velocity.detach().float() - old_velocity.detach().float()
    executed = torch.zeros_like(error, dtype=torch.bool)
    executed[:, :action_chunk, :action_dim] = (
        True if coordinate_mask is None else coordinate_mask
    )
    valid_rows = executed.flatten(1).any(1)[:, None, None]
    full = valid_rows.expand_as(error)
    future = torch.zeros_like(executed)
    future[:, action_chunk:, :action_dim] = True
    padding = torch.zeros_like(executed)
    padding[:, :, action_dim:] = True
    masks = (full, executed, full & ~executed, future & full, padding & full)
    squared_error = error.square()
    stats = torch.stack(
        [
            torch.stack(
                (
                    torch.where(mask, squared_error, 0.0).sum(),
                    mask.sum().to(error.dtype),
                )
            )
            for mask in masks
        ]
    )
    max_error = torch.where(executed, error.abs(), 0.0).amax()
    return stats, max_error


@torch.no_grad()
def _validate_art_rollout(
    states: torch.Tensor,
    sigmas: torch.Tensor,
    sigmas_next: torch.Tensor,
    guidance_magnitude: torch.Tensor,
) -> None:
    """Validate schedule values once before ART micro-batch training starts."""
    _validate_art_batch(states, "states")
    expected_schedule_shape = states.shape[:2]
    if sigmas.shape != expected_schedule_shape:
        raise ValueError(
            f"sigmas must have shape {tuple(expected_schedule_shape)}, "
            f"got {tuple(sigmas.shape)}"
        )
    if sigmas_next.shape != expected_schedule_shape:
        raise ValueError(
            f"sigmas_next must have shape {tuple(expected_schedule_shape)}, "
            f"got {tuple(sigmas_next.shape)}"
        )
    magnitude = _art_vector(
        guidance_magnitude,
        states,
        "guidance_magnitude",
    ).float()
    sigma = sigmas.detach().to(device=states.device, dtype=torch.float32)
    sigma_next = sigmas_next.detach().to(device=states.device, dtype=torch.float32)

    finite = (
        torch.isfinite(sigma).all()
        & torch.isfinite(sigma_next).all()
        & torch.isfinite(magnitude).all()
    )
    in_range = (
        ((sigma >= 0) & (sigma <= 1)).all()
        & ((sigma_next >= 0) & (sigma_next <= 1)).all()
        & (magnitude >= 0).all()
    )
    ordered = (sigma_next < sigma).all()
    if bool(finite & in_range & ordered):
        return
    if not bool(finite):
        raise ValueError("ART sigmas and guidance magnitudes must be finite")
    if not bool(in_range):
        raise ValueError(
            "ART sigmas must be in [0, 1] and guidance magnitudes must be nonnegative"
        )
    raise ValueError("ART requires sigma_next < sigma for every ODE step")


def _art_loss(
    current_velocity: torch.Tensor,
    old_velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    guidance_direction: torch.Tensor,
    guidance_magnitude: torch.Tensor,
    *,
    loss_space: str = "velocity_linear_sigma",
    mask_metric: str = "sde_kl",
    mask_threshold: float = 1e-5,
    reference_noise_level: float = 0.8,
    corrective_mask_enabled: bool = True,
    coordinate_mask: torch.Tensor | None = None,
    output_gradient_stats: bool = False,
) -> dict[str, torch.Tensor]:
    """Return the paper's masked, frozen-target velocity loss per item.

    ``output_gradient_stats`` requires ``coordinate_mask`` and adds the
    guidance/restoration split of the mean loss's output gradient.
    """
    _validate_art_batch(current_velocity, "current_velocity")
    _validate_art_same_shape(
        current_velocity,
        old_velocity=old_velocity,
        guidance_direction=guidance_direction,
    )
    if loss_space not in _ART_LOSS_SPACES:
        raise ValueError(f"loss_space must be one of {_ART_LOSS_SPACES}")
    if mask_metric not in _ART_MASK_METRICS:
        raise ValueError(f"mask_metric must be one of {_ART_MASK_METRICS}")
    threshold = _validate_art_scalar(mask_threshold, "mask_threshold", positive=True)
    noise = _validate_art_scalar(
        reference_noise_level, "reference_noise_level", positive=True
    )
    if type(corrective_mask_enabled) is not bool:
        raise ValueError("corrective_mask_enabled must be boolean")

    velocity = current_velocity.float()
    old = old_velocity.detach().to(velocity)
    direction = guidance_direction.detach().to(velocity)
    if coordinate_mask is not None:
        coordinate_mask = torch.broadcast_to(
            coordinate_mask.to(device=velocity.device, dtype=torch.bool), velocity.shape
        )
        velocity = torch.where(coordinate_mask, velocity, 0.0)
        old = torch.where(coordinate_mask, old, 0.0)
        direction = torch.where(coordinate_mask, direction, 0.0)
    magnitude = _art_vector(
        guidance_magnitude,
        velocity,
        "guidance_magnitude",
    ).float()
    sigma = _art_vector(sigma, velocity, "sigma").float()
    sigma_next = _art_vector(sigma_next, velocity, "sigma_next").float()

    displacement = _expand_art_vector(magnitude, velocity) * direction
    target = (old - displacement).detach()
    residual = target - velocity
    weight = sigma if loss_space == "velocity_linear_sigma" else torch.ones_like(sigma)
    unmasked = weight * _art_coordinate_mean(residual.square(), coordinate_mask)

    movement = velocity - old
    movement_energy = _art_coordinate_mean(movement.square(), coordinate_mask)
    outward_alignment = _art_coordinate_mean(
        movement * residual.detach(), coordinate_mask
    )
    if mask_metric == "sde_kl":
        eps = 1e-5
        step = sigma - sigma_next
        initial = torch.isclose(sigma, torch.ones_like(sigma), atol=eps, rtol=0)
        variance = torch.where(
            initial,
            torch.full_like(sigma, noise**2),
            (noise**2 * sigma / (1 - sigma).clamp_min(eps) * step).clamp_min(0),
        )
        coefficient = torch.where(
            initial,
            step,
            step * (1 + 0.5 * noise**2),
        )
        movement_score = (
            coefficient.square() * movement_energy / (2 * variance.clamp_min(eps**2))
        )
    else:
        movement_score = movement_energy

    direction_rms = _art_coordinate_mean(direction.square(), coordinate_mask).sqrt()
    active = (magnitude > 0) & (direction_rms > 0) & (sigma > 0)
    masked = (
        active
        & corrective_mask_enabled
        & (movement_score >= threshold)
        & (outward_alignment > 0)
    ).detach()
    keep = (active & ~masked).detach()
    loss = torch.where(keep, unmasked, torch.zeros_like(unmasked))
    terms = {
        "per_item_loss": loss,
        "per_item_unmasked_loss": unmasked,
        "target": target,
        "active": active,
        "keep_mask": keep,
        "masked": masked,
        "movement_energy": movement_energy.detach(),
        "movement_score": movement_score.detach(),
        "outward_alignment": outward_alignment.detach(),
    }
    if output_gradient_stats:
        # These are gradients of the mean loss with respect to velocity outputs.
        with torch.no_grad():
            coordinate_count = coordinate_mask.flatten(1).sum(1).clamp_min(1)
            scale = _expand_art_vector(
                2 * weight * keep / (velocity.shape[0] * coordinate_count), velocity
            )
            guidance_gradient = scale * displacement
            restoration_gradient = scale * movement
            terms.update(
                output_guidance_grad_sq=guidance_gradient.square().sum(),
                output_restoration_grad_sq=restoration_gradient.square().sum(),
                output_grad_inner=(guidance_gradient * restoration_gradient).sum(),
            )
    return terms


def _art_coordinate_mean(
    value: torch.Tensor, coordinate_mask: torch.Tensor | None
) -> torch.Tensor:
    if coordinate_mask is None:
        return value.flatten(1).mean(1)
    mask = torch.broadcast_to(coordinate_mask, value.shape)
    return torch.where(mask, value, 0.0).flatten(1).sum(1) / mask.flatten(1).sum(
        1
    ).clamp_min(1)


def _prompt_disjoint_windows(groups, updates: int = 2, *, order=None):
    """Split every complete prompt group into disjoint row windows."""
    if (
        groups.ndim != 1
        or not groups.numel()
        or groups.dtype not in (torch.int32, torch.int64)
        or (groups < 0).any()
        or type(updates) is not int
        or updates <= 0
    ):
        raise ValueError(
            "prompt windows require nonempty integer groups and a positive update count"
        )
    ids = groups.detach().cpu().tolist()
    order = list(range(len(ids))) if order is None else list(order)
    if any(type(row) is not int for row in order) or sorted(order) != list(
        range(len(ids))
    ):
        raise ValueError("order must be a permutation of all rollout rows")
    members = {}
    for row in order:
        members.setdefault(ids[row], []).append(row)
    if any(len(rows) % updates for rows in members.values()):
        raise ValueError("each prompt group must divide evenly into update windows")
    assignments = [0] * len(ids)
    for rows in members.values():
        size = len(rows) // updates
        for position, row in enumerate(rows):
            assignments[row] = position // size
    return tuple(
        tuple(row for row in order if assignments[row] == window)
        for window in range(updates)
    )


@torch.no_grad()
def _art_scalar_covariance(
    endpoints: torch.Tensor,
    rewards: torch.Tensor,
    row_group_indices: torch.Tensor,
    group_count: int,
    *,
    reduction: str = "unbiased",
    direction_mode: str = "historical",
    scale_mode: str = "unit_normalized_covariance",
    guidance_gain: float = 1.0,
    zero_signal_mode: str = "historical",
    zero_signal_atol: float = 0.0,
    eps: float = 1e-5,
    coordinate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Complete-group covariance, never a microbatch estimator.

    Historical direction reduction divides covariance by reward MAD before
    RMS normalization with an epsilon floor. ``normalized_aggregate`` instead
    normalizes the aggregate covariance to exact unit RMS when active.
    ``exact`` clears roundoff from exactly constant rewards without a cutoff.
    ``threshold`` zero handling explicitly gates raw covariance and reward range;
    it is not silently applied to the historical fp32 computation.
    """
    _validate_art_batch(endpoints, "endpoints")
    eps = _validate_art_scalar(eps, "eps", positive=True)
    gain = _validate_art_scalar(guidance_gain, "guidance_gain")
    atol = _validate_art_scalar(zero_signal_atol, "zero_signal_atol")
    if reduction not in ("unbiased", "population"):
        raise ValueError("reduction must be unbiased or population")
    if direction_mode not in ("historical", "normalized_aggregate"):
        raise ValueError("unknown direction_mode")
    if zero_signal_mode not in ("historical", "exact", "threshold"):
        raise ValueError("unknown zero_signal_mode")
    if (zero_signal_mode in ("historical", "exact") and atol != 0) or (
        zero_signal_mode == "threshold" and atol <= 0
    ):
        raise ValueError(
            "threshold mode requires positive atol; historical/exact require zero"
        )
    if scale_mode not in (
        "reward_mad",
        "raw_covariance",
        "reward_std_normalized_covariance",
        "unit_normalized_covariance",
    ):
        raise ValueError("unknown scale_mode")
    if (
        isinstance(group_count, bool)
        or not isinstance(group_count, int)
        or group_count < 1
    ):
        raise ValueError("group_count must be a positive integer")
    if row_group_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("row_group_indices must be integer indices")
    x = endpoints.detach().float().flatten(1)
    r = rewards.detach().to(device=x.device, dtype=torch.float32).reshape(-1)
    mask = None
    valid_rows = torch.ones(len(x), device=x.device, dtype=torch.bool)
    if coordinate_mask is not None:
        mask = torch.broadcast_to(
            coordinate_mask.to(device=x.device, dtype=torch.bool), endpoints.shape
        ).flatten(1)
        valid_rows = mask.any(1)
        x = torch.where(mask, x, 0.0)
        r = torch.where(valid_rows, r, 0.0)
    groups = row_group_indices.to(device=x.device, dtype=torch.long).reshape(-1)
    if r.numel() != len(x) or groups.numel() != len(x):
        raise ValueError("rewards and groups must match the endpoint count")
    if (groups < 0).any() or (groups >= group_count).any():
        raise ValueError("group index out of range")
    if not torch.isfinite(x).all() or not torch.isfinite(r).all():
        raise ValueError("endpoints and rewards must be finite")
    counts = x.new_zeros(group_count).index_add_(0, groups, valid_rows.float())
    sums = x.new_zeros(group_count).index_add_(0, groups, r)
    square_sums = x.new_zeros(group_count).index_add_(0, groups, r.square())
    means = sums / counts.clamp_min(1)
    stds = (square_sums / counts.clamp_min(1) - means.square()).clamp_min(0).sqrt()
    centered = torch.where(valid_rows, r - means[groups], 0.0)
    if zero_signal_mode == "exact":
        maxima = r.new_full((group_count,), -torch.inf).scatter_reduce_(
            0, groups, torch.where(valid_rows, r, -torch.inf), reduce="amax"
        )
        minima = r.new_full((group_count,), torch.inf).scatter_reduce_(
            0, groups, torch.where(valid_rows, r, torch.inf), reduce="amin"
        )
        constant = maxima == minima
        # Summing identical FP32 rewards can round their mean away from the reward.
        centered = torch.where(constant[groups], 0.0, centered)
        means = torch.where(constant, maxima, means)
        stds = torch.where(constant, 0.0, stds)
    mad = x.new_zeros(group_count).index_add_(
        0, groups, centered.abs()
    ) / counts.clamp_min(1)
    if mask is None:
        numerator = x.new_zeros(group_count, x.shape[1]).index_add_(
            0, groups, centered[:, None] * x
        )
        denominator = (counts - int(reduction == "unbiased")).clamp_min(1)
        covariance = numerator / denominator[:, None]
        covariance = torch.where((counts > 1)[:, None], covariance, 0.0)
        coordinate_counts = counts[:, None].expand_as(covariance)
        supported_coordinates = None
    else:
        coordinate_counts = x.new_zeros(group_count, x.shape[1]).index_add_(
            0, groups, mask.float()
        )
        coordinate_reward_sums = torch.zeros_like(coordinate_counts).index_add_(
            0, groups, r[:, None] * mask
        )
        coordinate_means = coordinate_reward_sums / coordinate_counts.clamp_min(1)
        coordinate_centered = torch.where(
            mask, r[:, None] - coordinate_means[groups], 0.0
        )
        coordinate_groups = groups[:, None].expand_as(x)
        maxima = torch.full_like(coordinate_counts, -torch.inf).scatter_reduce_(
            0,
            coordinate_groups,
            torch.where(mask, r[:, None], -torch.inf),
            reduce="amax",
        )
        minima = torch.full_like(coordinate_counts, torch.inf).scatter_reduce_(
            0,
            coordinate_groups,
            torch.where(mask, r[:, None], torch.inf),
            reduce="amin",
        )
        coordinate_centered = torch.where(
            (maxima == minima)[groups], 0.0, coordinate_centered
        )
        numerator = torch.zeros_like(coordinate_counts).index_add_(
            0, groups, coordinate_centered * x
        )
        denominator = (coordinate_counts - int(reduction == "unbiased")).clamp_min(1)
        supported_coordinates = coordinate_counts > 1
        covariance = torch.where(supported_coordinates, numerator / denominator, 0.0)
    rms = _art_coordinate_mean(covariance.square(), supported_coordinates).sqrt()
    magnitudes = {
        "reward_mad": mad,
        "raw_covariance": rms,
        "reward_std_normalized_covariance": rms / (stds + eps),
        "unit_normalized_covariance": torch.ones_like(rms),
    }[scale_mode] * gain
    signal = covariance / torch.where(mad > 0, mad, 1.0)[:, None]
    signal_rms = _art_coordinate_mean(signal.square(), supported_coordinates).sqrt()
    active = (counts > 1) & (mad > 0) & (signal_rms > 0) & (magnitudes > 0)
    if zero_signal_mode == "exact":
        magnitudes = torch.where(active, magnitudes, 0.0)
    if zero_signal_mode == "threshold":
        maxima = r.new_full((group_count,), -torch.inf).scatter_reduce_(
            0, groups, r, reduce="amax"
        )
        minima = r.new_full((group_count,), torch.inf).scatter_reduce_(
            0, groups, r, reduce="amin"
        )
        active &= (maxima - minima > atol) & (rms > atol)
        magnitudes = torch.where(active, magnitudes, 0.0)
    if direction_mode == "historical":
        directions = signal / signal_rms.clamp_min(eps)[:, None]
    else:
        # Float64 norm avoids underflow for tiny but nonzero aggregates.
        norm = _art_coordinate_mean(
            covariance.double().square(), supported_coordinates
        ).sqrt()
        directions = (
            covariance.double() / torch.where(norm > 0, norm, 1.0)[:, None]
        ).float()
    directions = torch.where(active[:, None], directions, 0.0)
    shape = (group_count, *endpoints.shape[1:])
    return {
        "group_covariance": covariance.reshape(shape),
        "group_directions": directions.reshape(shape),
        "group_guidance_magnitudes": magnitudes,
        "group_active": active,
        "group_counts": counts,
        "group_reward_means": means,
        "group_reward_stds": stds,
        "group_reward_abs_means": mad,
        "group_covariance_rms": rms,
        "group_coordinate_counts": coordinate_counts.reshape(shape),
    }


@torch.no_grad()
def _art_trajectory_guidance(
    endpoints: torch.Tensor,
    episode_returns: torch.Tensor,
    action_mask: torch.Tensor,
    group_size: int,
    *,
    condition_ids: torch.Tensor | None = None,
    **covariance_options,
) -> dict:
    """Aggregate same-reset trajectories at each decision, before flattening."""
    if endpoints.ndim != 4 or action_mask.shape != endpoints.shape[:3]:
        raise ValueError(
            "ART actions must be [decisions, episodes, actions, dims] with a matching action mask"
        )
    decisions, episodes = endpoints.shape[:2]
    if group_size < 2 or episodes % group_size:
        raise ValueError(
            f"ART needs complete groups of at least two trajectories, got {episodes=} and {group_size=}"
        )
    if episode_returns.shape != (episodes,):
        raise ValueError(
            f"ART episode returns must have shape {(episodes,)}, got {tuple(episode_returns.shape)}"
        )
    device = endpoints.device
    if condition_ids is None:
        condition_groups = torch.arange(episodes, device=device) // group_size
    else:
        _, condition_groups = torch.unique(
            condition_ids.to(device), sorted=True, return_inverse=True
        )
    decision_ids = torch.arange(decisions, device=device)[:, None]
    groups = decision_ids * (episodes // group_size) + condition_groups[None]
    stats = _art_scalar_covariance(
        endpoints.flatten(0, 1),
        episode_returns[None].expand(decisions, -1).reshape(-1),
        groups.reshape(-1),
        decisions * (episodes // group_size),
        coordinate_mask=action_mask.flatten(0, 1)[..., None],
        **covariance_options,
    )
    coordinate_mask = action_mask[..., None] & (
        stats["group_coordinate_counts"][groups] > 1
    )
    directions = torch.where(coordinate_mask, stats["group_directions"][groups], 0.0)
    return {
        "directions": directions,
        "magnitudes": stats["group_guidance_magnitudes"][groups],
        "coordinate_mask": coordinate_mask,
        "stats": stats,
    }


def _art_guidance_metrics(
    stats: dict[str, torch.Tensor],
    directions: torch.Tensor,
    magnitudes: torch.Tensor,
    rewards: torch.Tensor,
    coordinate_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize per-row guidance and the group statistics it came from."""
    active = stats["group_active"]
    covariance = stats["group_covariance_rms"][active]
    direction_rms = _art_coordinate_mean(
        directions.float().square(), coordinate_mask
    ).sqrt()
    return {
        "actor/active_group_frac": active.float().mean().item(),
        "actor/covariance_rms": covariance.mean().item() if covariance.numel() else 0.0,
        "actor/guidance_rms": (direction_rms * magnitudes.float()).mean().item(),
        "actor/art_reward_mean": rewards.mean().item(),
        "actor/art_reward_std": rewards.std(correction=0).item(),
    }


def _validate_art_scalar(value: float, name: str, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return value


def _validate_art_batch(value: torch.Tensor, name: str) -> None:
    if value.ndim < 2 or value.shape[0] == 0 or value.numel() == 0:
        raise ValueError(f"{name} must have nonempty batch and feature dimensions")


def _validate_art_same_shape(reference: torch.Tensor, **values: torch.Tensor) -> None:
    for name, value in values.items():
        if value.shape != reference.shape:
            raise ValueError(f"{name} must have shape {tuple(reference.shape)}")


def _art_vector(
    value: torch.Tensor, reference: torch.Tensor, name: str
) -> torch.Tensor:
    """Accept one value per item, including explicit singleton feature axes."""
    if (
        value.ndim == 0
        or value.numel() != reference.shape[0]
        or value.shape[0] != reference.shape[0]
    ):
        raise ValueError(f"{name} must contain one value per item")
    return value.detach().to(device=reference.device).reshape(-1)


def _expand_art_vector(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.reshape((-1,) + (1,) * (reference.ndim - 1))


def _art_mask_count_metrics(counts: torch.Tensor, prefix: str) -> dict[str, float]:
    """Reduce ratios after pooling active, masked, and outward item counts."""
    active, masked, outward, movement_score = counts.tolist()
    return {
        f"{prefix}/active_count": active,
        f"{prefix}/mask_frac_global": masked / max(active, 1),
        f"{prefix}/outward_frac_global": outward / max(active, 1),
        f"{prefix}/movement_score_active": movement_score / max(active, 1),
    }
