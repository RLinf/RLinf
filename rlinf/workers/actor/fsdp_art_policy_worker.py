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
from rlinf.utils.metric_utils import append_to_dict, compute_rollout_metrics
from rlinf.utils.nested_dict_process import (
    process_nested_dict_for_train,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

_ART_LOSS_SPACES = ("velocity_linear_sigma", "velocity_uniform")
_ART_MASK_METRICS = ("sde_kl", "fixed_mse")


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
        endpoints = self.rollout_batch["forward_inputs"]["art_endpoint"]
        rollout_size = endpoints.shape[0] * endpoints.shape[1]
        self.rollout_batch = process_nested_dict_for_train(
            self.rollout_batch,
            torch.arange(rollout_size),
        )

        generator = torch.Generator().manual_seed(
            self.cfg.actor.seed + self.version + self._rank
        )
        preparation_metrics, has_guidance = self._prepare_art_training_inputs(generator)
        active = torch.tensor(float(has_guidance), device=self.device)
        torch.distributed.all_reduce(active, op=torch.distributed.ReduceOp.MAX)
        has_guidance = bool(active.item())
        preparation_metrics["actor/skipped_zero_guidance"] = float(not has_guidance)
        if not has_guidance:
            clear_memory()
            return all_reduce_dict(
                preparation_metrics,
                op=torch.distributed.ReduceOp.AVG,
            )

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
        self._precompute_old_velocities()

        group_size = int(self.cfg.algorithm.group_size)
        updates = int(self.cfg.algorithm.get("art_updates_per_rollout", 2))
        groups = torch.arange(rollout_size) // group_size
        order = torch.randperm(rollout_size, generator=generator).tolist()
        row_windows = _prompt_disjoint_windows(
            groups,
            updates=updates,
            order=order,
        )
        windows = tuple(torch.tensor(rows, dtype=torch.long) for rows in row_windows)

        self.model.train()
        metrics: dict[str, list[float]] = {}
        for key, value in preparation_metrics.items():
            append_to_dict(metrics, {key: value})

        for window_rows in windows:
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
            micro_batches = split_dict_to_chunk(
                window_batch,
                window_size // micro_batch_size,
            )
            num_train_timesteps = window_batch["forward_inputs"]["art_states"].shape[1]
            self.gradient_accumulation = len(micro_batches) * num_train_timesteps

            self.optimizer.zero_grad()
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
                    scaled_loss = loss / self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(scaled_loss).backward()
                    append_to_dict(metrics, step_metrics)
                    del loss, scaled_loss, step_metrics

            self.torch_platform.empty_cache()
            grad_norm, lr_list = self.optimizer_step()
            self.lr_scheduler.step()
            append_to_dict(
                metrics,
                {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                },
            )

        self.optimizer.zero_grad()
        clear_memory()
        mean_metrics = {key: np.mean(value) for key, value in metrics.items()}
        return all_reduce_dict(mean_metrics, op=torch.distributed.ReduceOp.AVG)

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
            endpoints,
            rewards,
            groups,
            group_count,
            reduction=self.cfg.algorithm.get(
                "art_covariance_reduction",
                "unbiased",
            ),
            direction_mode=self.cfg.algorithm.get(
                "art_direction_mode",
                "historical",
            ),
            scale_mode=self.cfg.algorithm.get(
                "art_scale_mode",
                "unit_normalized_covariance",
            ),
            guidance_gain=float(self.cfg.algorithm.get("art_guidance_gain", 1.0)),
            zero_signal_mode=self.cfg.algorithm.get(
                "art_zero_signal_mode",
                "exact",
            ),
            zero_signal_atol=float(self.cfg.algorithm.get("art_zero_signal_atol", 0.0)),
            eps=float(self.cfg.algorithm.get("art_eps", 1e-5)),
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

        states = forward_inputs["art_states"]
        sigmas = forward_inputs["art_sigmas"]
        sigmas_next = forward_inputs["art_sigmas_next"]
        _validate_art_rollout(
            states,
            sigmas,
            sigmas_next,
            forward_inputs["art_guidance_magnitude"],
        )
        timestep_pool = states.shape[1]
        if (
            self.cfg.algorithm.get("art_exclude_final_step", False)
            and timestep_pool > 1
        ):
            timestep_pool -= 1
        fraction = float(
            self.cfg.algorithm.get(
                "art_timestep_fraction",
                1.0,
            )
        )
        train_timestep_count = max(1, int(timestep_pool * fraction))
        selected = torch.stack(
            [
                torch.randperm(timestep_pool, generator=generator)[
                    :train_timestep_count
                ]
                for _ in range(row_count)
            ]
        )
        latent_indices = selected.view(
            row_count,
            train_timestep_count,
            *([1] * (states.ndim - 2)),
        ).expand(-1, -1, *states.shape[2:])
        forward_inputs["art_states"] = states.gather(1, latent_indices)
        forward_inputs["art_sigmas"] = sigmas.gather(1, selected)
        forward_inputs["art_sigmas_next"] = sigmas_next.gather(1, selected)
        del forward_inputs["art_endpoint"]

        active = guidance_stats["group_active"]
        covariance_rms = guidance_stats["group_covariance_rms"]
        active_covariance = covariance_rms[active]
        direction_rms = (
            forward_inputs["art_guidance_direction"]
            .float()
            .flatten(1)
            .square()
            .mean(1)
            .sqrt()
        )
        guidance_rms = direction_rms * forward_inputs["art_guidance_magnitude"].float()
        return (
            {
                "actor/active_group_frac": active.float().mean().item(),
                "actor/covariance_rms": (
                    active_covariance.mean().item()
                    if active_covariance.numel()
                    else 0.0
                ),
                "actor/guidance_rms": guidance_rms.mean().item(),
                "actor/art_reward_mean": rewards.mean().item(),
                "actor/art_reward_std": rewards.std(correction=0).item(),
            },
            bool(active.any()),
        )

    @torch.no_grad()
    def _precompute_old_velocities(self) -> None:
        forward_inputs = self.rollout_batch["forward_inputs"]
        states = forward_inputs["art_states"]
        sigmas = forward_inputs["art_sigmas"]
        old_velocities = torch.empty_like(states)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        row_count, timestep_count = states.shape[:2]

        self.model.eval()
        for start in range(0, row_count, micro_batch_size):
            end = min(start + micro_batch_size, row_count)
            rows = torch.arange(start, end)
            model_inputs = put_tensor_device(
                self._model_forward_inputs(forward_inputs, rows),
                self.device,
            )
            for timestep_index in range(timestep_count):
                with self.amp_context:
                    output = self.model(
                        forward_type=ForwardType.NFT,
                        forward_inputs=model_inputs,
                        nft_inputs={
                            "x_t": states[start:end, timestep_index].to(self.device),
                            "timesteps": sigmas[start:end, timestep_index].to(
                                self.device
                            ),
                        },
                        compute_values=False,
                    )
                old_velocities[start:end, timestep_index] = (
                    output["v_theta"].detach().to(old_velocities.device)
                )
        forward_inputs["art_old_velocity"] = old_velocities
        self.torch_platform.empty_cache()

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
        terms = _art_loss(
            output["v_theta"],
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
        )
        loss = terms["per_item_loss"].mean()
        active = terms["active"]
        active_count = active.sum().clamp_min(1)
        metrics = {
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
        }
        inputs = {
            key: value
            for key, value in forward_inputs.items()
            if key in model_input_keys
        }
        return inputs if indices is None else cls._index_nested(inputs, indices)


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
) -> dict[str, torch.Tensor]:
    """Return the paper's masked, frozen-target velocity loss per item."""
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
    unmasked = weight * residual.square().flatten(1).mean(1)

    movement = velocity - old
    movement_energy = movement.square().flatten(1).mean(1)
    outward_alignment = (movement * residual.detach()).flatten(1).mean(1)
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

    direction_rms = direction.square().flatten(1).mean(1).sqrt()
    active = (magnitude > 0) & (direction_rms > 0) & (sigma > 0)
    masked = (
        active
        & corrective_mask_enabled
        & (movement_score >= threshold)
        & (outward_alignment > 0)
    ).detach()
    keep = (active & ~masked).detach()
    loss = torch.where(keep, unmasked, torch.zeros_like(unmasked))
    return {
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
    groups = row_group_indices.to(device=x.device, dtype=torch.long).reshape(-1)
    if r.numel() != len(x) or groups.numel() != len(x):
        raise ValueError("rewards and groups must match the endpoint count")
    if (groups < 0).any() or (groups >= group_count).any():
        raise ValueError("group index out of range")
    if not torch.isfinite(x).all() or not torch.isfinite(r).all():
        raise ValueError("endpoints and rewards must be finite")
    counts = torch.bincount(groups, minlength=group_count).float()
    sums = x.new_zeros(group_count).index_add_(0, groups, r)
    square_sums = x.new_zeros(group_count).index_add_(0, groups, r.square())
    means = sums / counts.clamp_min(1)
    stds = (square_sums / counts.clamp_min(1) - means.square()).clamp_min(0).sqrt()
    centered = r - means[groups]
    if zero_signal_mode == "exact":
        maxima = r.new_full((group_count,), -torch.inf).scatter_reduce_(
            0, groups, r, reduce="amax"
        )
        minima = r.new_full((group_count,), torch.inf).scatter_reduce_(
            0, groups, r, reduce="amin"
        )
        constant = maxima == minima
        # Summing identical FP32 rewards can round their mean away from the reward.
        centered = torch.where(constant[groups], 0.0, centered)
        means = torch.where(constant, maxima, means)
        stds = torch.where(constant, 0.0, stds)
    mad = x.new_zeros(group_count).index_add_(
        0, groups, centered.abs()
    ) / counts.clamp_min(1)
    numerator = x.new_zeros(group_count, x.shape[1]).index_add_(
        0, groups, centered[:, None] * x
    )
    denominator = (counts - int(reduction == "unbiased")).clamp_min(1)
    covariance = numerator / denominator[:, None]
    covariance = torch.where((counts > 1)[:, None], covariance, 0.0)
    rms = covariance.square().mean(1).sqrt()
    magnitudes = {
        "reward_mad": mad,
        "raw_covariance": rms,
        "reward_std_normalized_covariance": rms / (stds + eps),
        "unit_normalized_covariance": torch.ones_like(rms),
    }[scale_mode] * gain
    signal = covariance / torch.where(mad > 0, mad, 1.0)[:, None]
    signal_rms = signal.square().mean(1).sqrt()
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
        norm = covariance.double().square().mean(1).sqrt()
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
