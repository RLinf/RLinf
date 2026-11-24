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

import numpy as np
import torch

from rlinf.algorithms.diffusion.art import (
    art_guidance_metrics,
    art_state_clusters,
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


class EmbodiedARTFSDPPolicy(EmbodiedFSDPActor):
    """FSDP policy worker for DiffusionART."""

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, float]:
        """Report rollout metrics; ART builds guidance directly from rewards."""
        (self.rollout_batch,) = self.env_batches.values()
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
        active = torch.tensor(float(has_guidance), device=self.device)
        torch.distributed.all_reduce(active, op=torch.distributed.ReduceOp.MAX)
        has_guidance = bool(active.item())
        preparation_metrics["actor/skipped_zero_guidance"] = float(not has_guidance)
        if self.is_weight_offloaded and has_guidance:
            self.load_param_and_grad(self.device)
        if not has_guidance:
            clear_memory()
            return all_reduce_dict(
                preparation_metrics,
                op=torch.distributed.ReduceOp.AVG,
            )
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
        # Replay on the actor so the frozen reference uses the same numerics.
        self._precompute_old_velocities()

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
                    with backward_ctx:
                        self.grad_scaler.scale(scaled_loss).backward()
                    append_to_dict(metrics, step_metrics)
                    del loss, scaled_loss, step_metrics

            self.torch_platform.empty_cache()
            grad_norm, lr_list = self.optimizer_step()
            mask_counts += update_mask_counts
            counts = update_mask_counts.clone()
            torch.distributed.all_reduce(counts)
            append_to_dict(
                metrics, _art_mask_count_metrics(counts, f"actor/u{update_index}")
            )
            append_to_dict(
                metrics,
                {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                },
            )

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        torch.distributed.all_reduce(mask_counts)
        mean_metrics = self._reduce_training_metrics(metrics)
        mean_metrics.update(_art_mask_count_metrics(mask_counts, "actor"))
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
        decisions, episodes = endpoints.shape[:2]
        anchors = forward_inputs.pop("art_anchor")
        preparation_metrics = {}
        if self.cfg.algorithm.get("art_grouping", "index") == "cluster":
            returns_to_go = (
                torch.where(action_mask, batch["rewards"], 0.0)
                .sum(-1)
                .flip(0)
                .cumsum(0)
                .flip(0)
            )
            guidance, preparation_metrics = self._art_cluster_guidance(
                endpoints, anchors, returns_to_go, action_mask, forward_inputs
            )
        else:
            guidance = art_trajectory_guidance(
                endpoints - anchors.to(endpoints),
                episode_returns,
                action_mask,
                group_size,
                **self._art_covariance_options(),
            )
        episode_windows = prompt_disjoint_windows(
            torch.arange(episodes) // group_size,
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
            batch, torch.arange(decisions * episodes)
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
                "actor/valid_chunk_frac": action_mask.any(-1).float().mean().item(),
                "actor/valid_action_frac": action_mask.float().mean().item(),
            },
            bool(guidance["stats"]["group_active"].any()),
            tuple(windows),
        )

    @torch.no_grad()
    def _art_cluster_guidance(
        self,
        endpoints: torch.Tensor,
        anchors: torch.Tensor,
        returns_to_go: torch.Tensor,
        action_mask: torch.Tensor,
        forward_inputs: dict,
    ) -> tuple[dict, dict[str, float]]:
        """Group each live chunk with nearby states from every rank's episodes.

        Decision 0 keeps its exact reset groups. Later chunks join
        ``art_state_clusters`` balls over the episodes of all ranks, so every
        rank derives identical groups and guidance; each keeps its own
        episodes. Chunks outside a cluster receive no guidance.

        Returns:
            Local guidance in the ``art_trajectory_guidance`` layout, with
            statistics of the reset groups and clusters only, and global
            cluster metrics.
        """
        algorithm = self.cfg.algorithm
        group_size = int(algorithm.group_size)
        local_episodes = endpoints.shape[1]
        endpoints = self._gather_episodes(endpoints)
        returns_to_go = self._gather_episodes(returns_to_go)
        action_mask = self._gather_episodes(action_mask)
        features = self._gather_episodes(forward_inputs["art_cluster_features"])
        buckets = self._gather_episodes(forward_inputs["art_cluster_bucket"])
        anchors = self._gather_episodes(anchors.to(endpoints))
        decisions, episodes = returns_to_go.shape
        eligible = action_mask[..., 0].clone()
        eligible[0] = False
        trajectories = torch.arange(episodes).expand(decisions, -1)
        labels, seeds = art_state_clusters(
            features.flatten(0, 1),
            buckets.flatten(0, 1),
            trajectories.flatten(),
            eligible.flatten(),
            radius=float(algorithm.art_cluster_radius),
            min_size=int(algorithm.art_cluster_min_size),
            max_size=group_size,
        )
        reset_groups = episodes // group_size
        kept_groups = reset_groups + len(seeds)
        groups = labels.reshape(decisions, episodes) + reset_groups
        groups[0] = torch.arange(episodes) // group_size
        unclustered = labels.reshape(decisions, episodes) < 0
        unclustered[0] = False
        # A single masked group excludes discarded rows without allocating a
        # covariance group for every unused chunk.
        groups[unclustered] = kept_groups
        action_mask = action_mask & ~unclustered[..., None]
        guidance = art_trajectory_guidance(
            endpoints - anchors,
            returns_to_go,
            action_mask,
            group_size,
            groups=groups,
            **self._art_covariance_options(),
        )
        metrics = _art_cluster_metrics(
            labels,
            seeds,
            features.flatten(0, 1),
            trajectories.flatten(),
            torch.arange(decisions).repeat_interleave(episodes),
            eligible.flatten(),
            group_size,
            guidance["stats"]["group_active"][reset_groups:kept_groups],
        )
        local = slice(self._rank * local_episodes, (self._rank + 1) * local_episodes)
        return (
            {
                "directions": guidance["directions"][:, local],
                "magnitudes": guidance["magnitudes"][:, local],
                "coordinate_mask": guidance["coordinate_mask"][:, local],
                "stats": {
                    key: value[:kept_groups] for key, value in guidance["stats"].items()
                },
            },
            metrics,
        )

    def _gather_episodes(self, value: torch.Tensor) -> torch.Tensor:
        """Concatenate every actor rank's episodes along dimension 1 in rank order."""
        if self._world_size == 1:
            return value.detach().cpu()
        wire_dtype = torch.uint8 if value.dtype == torch.bool else value.dtype
        local = value.to(device=self.device, dtype=wire_dtype).contiguous()
        parts = [torch.empty_like(local) for _ in range(self._world_size)]
        torch.distributed.all_gather(parts, local)
        return torch.cat(parts, dim=1).to(device=value.device, dtype=value.dtype)

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
        indices = (
            selected.to(states.device)
            .view(row_count, train_timestep_count, *([1] * (states.ndim - 2)))
            .expand(-1, -1, *states.shape[2:])
        )
        forward_inputs["art_states"] = states.gather(1, indices)
        forward_inputs["art_sigmas"] = sigmas.gather(1, selected.to(sigmas.device))
        forward_inputs["art_sigmas_next"] = sigmas_next.gather(
            1, selected.to(sigmas_next.device)
        )
        del forward_inputs["art_endpoint"]

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
                            forward_type=ForwardType.ART,
                            forward_inputs=model_inputs,
                            art_inputs={
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
                forward_type=ForwardType.ART,
                forward_inputs=model_inputs,
                art_inputs={"x_t": states, "timesteps": sigmas},
                compute_values=False,
            )
        coordinate_mask = forward_inputs.get("art_coordinate_mask")
        velocity = output["v_theta"]
        if coordinate_mask is not None:
            coordinate_mask = coordinate_mask.to(self.device)
            chunk, dims = guidance_direction.shape[1:]
            velocity = velocity[:, :chunk, :dims]
            old_velocity = old_velocity[:, :chunk, :dims]
        return compute_art_loss(
            velocity,
            old_velocity,
            sigmas,
            sigmas_next,
            guidance_direction,
            guidance_magnitude,
            config=self.cfg.algorithm,
            coordinate_mask=coordinate_mask,
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


def _art_mask_count_metrics(counts: torch.Tensor, prefix: str) -> dict[str, float]:
    """Reduce ratios after pooling active, masked, and outward item counts."""
    active, masked, outward, movement_score = counts.tolist()
    return {
        f"{prefix}/active_count": active,
        f"{prefix}/mask_frac_global": masked / max(active, 1),
        f"{prefix}/outward_frac_global": outward / max(active, 1),
        f"{prefix}/movement_score_active": movement_score / max(active, 1),
    }


@torch.no_grad()
def _art_cluster_metrics(
    labels: torch.Tensor,
    seeds: torch.Tensor,
    features: torch.Tensor,
    episodes: torch.Tensor,
    decisions: torch.Tensor,
    eligible: torch.Tensor,
    group_size: int,
    active: torch.Tensor,
) -> dict[str, float]:
    """Describe clusters from one global clustering; every rank reports the same values.

    Args:
        labels: ``[rows]`` cluster of each row or -1.
        seeds: ``[clusters]`` seed row of each cluster.
        features: ``[rows, features]`` cluster coordinates.
        episodes: ``[rows]`` global episode of each row.
        decisions: ``[rows]`` decision of each row.
        eligible: ``[rows]`` rows that could join a cluster.
        group_size: Episodes per reset group.
        active: ``[clusters]`` whether each cluster produced guidance.
    """
    clustered = labels >= 0
    count = len(seeds)
    metrics = {
        "actor/cluster_count": float(count),
        "actor/cluster_row_frac": clustered.sum().item()
        / max(eligible.sum().item(), 1),
    }
    if not count:
        return metrics
    members = labels[clustered]

    def extremes(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = values[clustered]
        low = values.new_full((count,), torch.iinfo(values.dtype).max)
        high = values.new_full((count,), torch.iinfo(values.dtype).min)
        return (
            low.scatter_reduce_(0, members, values, reduce="amin"),
            high.scatter_reduce_(0, members, values, reduce="amax"),
        )

    first_decision, last_decision = extremes(decisions)
    first_reset, last_reset = extremes(episodes // group_size)
    metrics.update(
        {
            "actor/cluster_size_mean": torch.bincount(members, minlength=count)
            .float()
            .mean()
            .item(),
            "actor/cluster_seed_distance_mean": (
                features[clustered] - features[seeds[members]]
            )
            .norm(dim=1)
            .mean()
            .item(),
            "actor/cluster_decision_span_mean": (last_decision - first_decision)
            .float()
            .mean()
            .item(),
            "actor/cluster_cross_reset_frac": (last_reset != first_reset)
            .float()
            .mean()
            .item(),
            "actor/cluster_active_frac": active.float().mean().item(),
        }
    )
    return metrics
