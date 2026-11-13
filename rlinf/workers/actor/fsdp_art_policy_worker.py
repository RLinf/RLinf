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

from rlinf.algorithms.diffusion.art import (
    art_guidance_metrics,
    art_trajectory_guidance,
    compute_art_loss,
    prepare_art_training_inputs,
    prompt_disjoint_windows,
    validate_art_rollout,
)
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
            preparation_metrics, has_guidance = prepare_art_training_inputs(
                self.rollout_batch["forward_inputs"],
                self.rollout_batch["rewards"],
                config=self.cfg.algorithm,
                generator=generator,
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
        if not action_trajectories or actor_reference or rollout_reference is None:
            self._precompute_old_velocities()
        if replay_diagnostics:
            preparation_metrics.update(self._art_replay_metrics("actor/reference_pre"))

        if not action_trajectories:
            row_windows = prompt_disjoint_windows(
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
                            step_metrics[f"actor/{key}_count"]
                            if key == "active"
                            else step_metrics.pop(f"actor/{key}_count")
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
            if action_trajectories:
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

        if not action_trajectories:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        torch.distributed.all_reduce(mask_counts)
        mean_metrics = self._reduce_training_metrics(metrics)
        mean_metrics.update(_art_mask_count_metrics(mask_counts, "actor"))
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

    @staticmethod
    def _reduce_training_metrics(metrics: dict[str, list[float]]) -> dict[str, float]:
        """Weight active-only metrics across micro-batches and actor ranks."""
        active_counts = metrics.pop("actor/active_count")
        active_metric_sums = {
            key: sum(
                value * count
                for value, count in zip(metrics.pop(key), active_counts, strict=True)
            )
            for key in (
                "actor/unmasked_loss",
                "actor/mask_frac",
                "actor/outward_frac",
            )
        }
        local_metrics = {
            key: sum(values) if key.endswith("_count") else np.mean(values)
            for key, values in metrics.items()
        }
        local_metrics.update(active_metric_sums)
        local_metrics["actor/active_count"] = sum(active_counts)
        reduced_metrics = all_reduce_dict(
            local_metrics, op=torch.distributed.ReduceOp.SUM
        )
        active_count = max(reduced_metrics.pop("actor/active_count"), 1)
        world_size = torch.distributed.get_world_size()
        return {
            key: value / (active_count if key in active_metric_sums else world_size)
            for key, value in reduced_metrics.items()
        }

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
        guidance = art_trajectory_guidance(
            endpoints[first_decision : first_decision + decisions],
            episode_returns,
            selected_mask,
            group_size,
            condition_ids=condition_ids,
            **self._art_covariance_options(),
        )
        episodes = endpoints.shape[1]
        episode_windows = prompt_disjoint_windows(
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
                **art_guidance_metrics(
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
    def _select_art_training_timesteps(self, generator: torch.Generator) -> None:
        forward_inputs = self.rollout_batch["forward_inputs"]
        states = forward_inputs["art_states"]
        sigmas = forward_inputs["art_sigmas"]
        sigmas_next = forward_inputs["art_sigmas_next"]
        validate_art_rollout(
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
        return compute_art_loss(
            velocity,
            old_velocity,
            sigmas,
            sigmas_next,
            guidance_direction,
            guidance_magnitude,
            config=self.cfg.algorithm,
            coordinate_mask=coordinate_mask,
            output_gradient_stats=output_gradient_stats,
        )

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
    if old_velocity.shape != velocity.shape:
        raise ValueError(f"old_velocity must have shape {tuple(velocity.shape)}")
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


def _art_mask_count_metrics(counts: torch.Tensor, prefix: str) -> dict[str, float]:
    """Reduce ratios after pooling active, masked, and outward item counts."""
    active, masked, outward, movement_score = counts.tolist()
    return {
        f"{prefix}/active_count": active,
        f"{prefix}/mask_frac_global": masked / max(active, 1),
        f"{prefix}/outward_frac_global": outward / max(active, 1),
        f"{prefix}/movement_score_active": movement_score / max(active, 1),
    }
