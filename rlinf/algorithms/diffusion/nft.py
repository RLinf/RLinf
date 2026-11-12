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

"""Tensor objectives, schedules, and EMA updates for NFT."""

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from rlinf.utils.utils import masked_mean


def get_nft_tau(
    tau: float | list[float] | tuple[float, ...] | ListConfig, step: int = 0
) -> float:
    """Evaluate a scalar tau or a linear [start, end, start_step, end_step] schedule."""
    if isinstance(tau, (list, tuple, ListConfig)):
        start_tau, end_tau = float(tau[0]), float(tau[1])
        start_step, end_step = int(tau[2]), int(tau[3])
        if step <= start_step:
            return start_tau
        if step >= end_step:
            return end_tau
        ratio = (
            0.0
            if start_step == end_step
            else ((step - start_step) / float(end_step - start_step))
        )
        return start_tau + ratio * (end_tau - start_tau)
    return float(tau)


def nft_requires_ema(tau: float | list[float] | tuple[float, ...] | ListConfig) -> bool:
    """Return whether tau needs lagged rollout weights at any scheduled step."""
    if isinstance(tau, (list, tuple, ListConfig)):
        return min(float(tau[0]), float(tau[1])) < 1.0
    return float(tau) < 1.0


@torch.no_grad()
def update_nft_ema(
    rollout_state: dict[str, Any], current_state: Mapping[str, Any], tau: float
) -> None:
    """Update rollout weights in place, copying non-floating state from the actor."""
    for key, rollout_tensor in rollout_state.items():
        src = current_state[key]
        if torch.is_tensor(rollout_tensor) and rollout_tensor.is_floating_point():
            if tau >= 1.0:
                rollout_tensor.copy_(src)
            else:
                rollout_tensor.lerp_(src.to(rollout_tensor.dtype), tau)
        elif torch.is_tensor(rollout_tensor) and torch.is_tensor(src):
            rollout_tensor.copy_(src)
        else:
            rollout_state[key] = src


def build_nft_schedule(
    step_indices: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    num_steps: int,
    ndim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the uniform 1-to-0 schedule and selected, optionally broadcast timesteps."""
    schedule = torch.linspace(1, 0, num_steps + 1, device=device, dtype=dtype)
    t = schedule[step_indices.long()]
    if ndim is not None:
        t = t.view(-1, *([1] * (ndim - 1)))
    return schedule, t


def resample_nft_inputs(
    x0: torch.Tensor, num_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample step indices and interpolate clean samples with fresh Gaussian noise."""
    step_indices = torch.randint(0, num_steps, (x0.shape[0],), device=x0.device)
    _, t = build_nft_schedule(
        step_indices, x0.device, x0.dtype, num_steps=num_steps, ndim=x0.ndim
    )
    xcur = (1 - t) * x0 + t * torch.randn_like(x0)
    return xcur, step_indices


def compute_nft_loss(
    v_theta: torch.Tensor,
    forward_inputs: Mapping[str, torch.Tensor],
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    schedule: torch.Tensor,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute NFT energies, the configured masked objective, and local metrics.

    Args:
        v_theta: Current actor velocities, retaining their autograd graph.
        forward_inputs: Rollout states, old velocities, step indices, and noise levels.
        advantages: Raw rewards or centered advantages before NFT rescaling.
        loss_mask: Valid items or action positions.
        schedule: Flow timesteps, including the terminal zero.
        config: The algorithm configuration with ``nft_*`` objective settings.

    Returns:
        The differentiable scalar loss and detached actor metrics.
    """
    target_space = config.get("nft_target_space", "xnext")
    step_indices = forward_inputs["nft_step_index"]
    gather_type = config.get("nft_gather_type", "action_level")
    chunk = v_theta.shape[1]
    v_theta = v_theta[:, :chunk, ...]
    x_t = forward_inputs["nft_xcur"][:, :chunk, ...]
    v_old = forward_inputs["nft_v"][:, :chunk, ...].detach()
    batch_size = x_t.shape[0]
    delta_v, v_pos, v_neg = _compute_delta_v_candidates(v_theta, v_old, config=config)
    t_bc, dt_bc, sigma_i, std_t_det = _build_schedule_params(
        schedule, step_indices, forward_inputs["nft_noise_level"], x_t
    )
    target, pred_pos = _compute_nft_target_and_pred(
        forward_inputs, target_space, x_t, v_pos, t_bc, dt_bc, sigma_i
    )
    _, pred_neg = _compute_nft_target_and_pred(
        forward_inputs, target_space, x_t, v_neg, t_bc, dt_bc, sigma_i
    )
    noise_level = forward_inputs["nft_noise_level"]
    weight_mode = config.get("nft_weight_mode", "auto")
    weight_kwargs = {
        "config": config,
        "weight_mode": weight_mode,
        "t_bc": t_bc,
        "std_t_det": std_t_det,
        "noise_level": noise_level,
        "target": target,
        "gather_type": gather_type,
    }
    w_pos = _compute_nft_weight(**weight_kwargs, pred=pred_pos, sample_type="pos")
    w_neg = _compute_nft_weight(**weight_kwargs, pred=pred_neg, sample_type="neg")
    energy_reduction = str(config.get("nft_energy_reduction", "mean"))
    e_pos = _reduce_nft_tensor(
        (pred_pos - target) ** 2 * w_pos,
        gather_type,
        energy_reduction,
    )
    e_neg = _reduce_nft_tensor(
        (pred_neg - target) ** 2 * w_neg,
        gather_type,
        energy_reduction,
    )
    loss_mask = _align_nft_signal(loss_mask, e_pos, batch_size)
    advantages = _align_nft_signal(advantages, e_pos, batch_size)
    advantages = _postprocess_advantages(advantages, config=config)
    delta_e = e_pos - e_neg
    clip_frac = 0.0
    e_pos_clip = None
    e_neg_clip = None
    clip_ratio = config.get("nft_clip_ratio", None)
    loss_form = config.get("nft_loss_form", "dpo")
    if loss_form == "mse" and clip_ratio is not None:
        e_pos_clip, e_neg_clip, clip_frac = _compute_nft_clipped_energies(
            config=config,
            forward_inputs=forward_inputs,
            target_space=target_space,
            x_t=x_t,
            delta_v=delta_v,
            v_old=v_old,
            t_bc=t_bc,
            dt_bc=dt_bc,
            sigma_i=sigma_i,
            target=target,
            w_pos=w_pos,
            w_neg=w_neg,
            gather_type=gather_type,
            energy_reduction=energy_reduction,
            clip_ratio=float(clip_ratio),
        )
    loss, clip_loss_frac = _aggregate_nft_loss(
        e_pos,
        e_neg,
        delta_e,
        advantages,
        loss_mask,
        config=config,
        e_pos_clip=e_pos_clip,
        e_neg_clip=e_neg_clip,
    )
    with torch.no_grad():
        delta_v_norm = _reduce_nft_tensor(delta_v, gather_type, "norm")
        delta_v_norm = delta_v_norm.mean().item()
        metrics_data = {
            "actor/nft_loss": loss.item(),
            "actor/delta_v_norm": delta_v_norm,
            "actor/clip_frac": clip_frac,
            "actor/E_pos_mean": e_pos.mean().item(),
            "actor/E_neg_mean": e_neg.mean().item(),
            "actor/E_pos_mean_pos_only": masked_mean(
                e_pos, (advantages > 0.5) & loss_mask.bool()
            ).item(),
            "actor/E_neg_mean_neg_only": masked_mean(
                e_neg, (advantages < 0.5) & loss_mask.bool()
            ).item(),
            "actor/delta_E_mean": delta_e.mean().item(),
        }
        if clip_loss_frac is not None:
            metrics_data["actor/clip_loss_frac"] = clip_loss_frac
    return loss, metrics_data


def _compute_nft_clipped_energies(
    *,
    config: DictConfig,
    forward_inputs: Mapping[str, torch.Tensor],
    target_space: str,
    x_t: torch.Tensor,
    delta_v: torch.Tensor,
    v_old: torch.Tensor,
    t_bc: torch.Tensor,
    dt_bc: torch.Tensor,
    sigma_i: torch.Tensor,
    target: torch.Tensor,
    w_pos: torch.Tensor | float,
    w_neg: torch.Tensor | float,
    gather_type: str,
    energy_reduction: str,
    clip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute clipped candidate energies for the NFT MSE objective."""
    delta_norm = _reduce_nft_tensor(
        delta_v, gather_type, "norm", keepdim=True
    ).clamp_min(1e-8)
    old_norm = _reduce_nft_tensor(v_old, gather_type, "norm", keepdim=True).clamp_min(
        1e-8
    )
    clip_coef = (clip_ratio * old_norm / delta_norm).clamp(max=1.0)
    clip_frac = (clip_coef < 1.0).float().mean().item()
    delta_v_clip = torch.where(
        clip_coef < 1.0,
        (delta_v * clip_coef).detach(),
        delta_v,
    )
    beta = float(config.get("nft_beta", 1.0))
    v_pos_clip = v_old + beta * delta_v_clip
    v_neg_clip = v_old - beta * delta_v_clip
    _, pred_pos_clip = _compute_nft_target_and_pred(
        forward_inputs,
        target_space,
        x_t,
        v_pos_clip,
        t_bc,
        dt_bc,
        sigma_i,
    )
    _, pred_neg_clip = _compute_nft_target_and_pred(
        forward_inputs,
        target_space,
        x_t,
        v_neg_clip,
        t_bc,
        dt_bc,
        sigma_i,
    )
    e_pos_clip = _reduce_nft_tensor(
        (pred_pos_clip - target) ** 2 * w_pos,
        gather_type,
        energy_reduction,
    )
    e_neg_clip = _reduce_nft_tensor(
        (pred_neg_clip - target) ** 2 * w_neg,
        gather_type,
        energy_reduction,
    )
    return e_pos_clip, e_neg_clip, clip_frac


def _aggregate_nft_loss(
    e_pos: torch.Tensor,
    e_neg: torch.Tensor,
    delta_e: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    config: DictConfig,
    e_pos_clip: torch.Tensor | None = None,
    e_neg_clip: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float | None]:
    """Aggregate NFT energies into the configured objective."""
    loss_form = config.get("nft_loss_form", "dpo")
    if loss_form == "dpo":
        dpo_beta = float(config.get("dpo_beta", 1.0))
        y = advantages * 2.0 - 1.0
        logit = (dpo_beta / 2.0) * y * delta_e
        return masked_mean(F.softplus(logit), loss_mask), None

    if loss_form == "mse":
        beta = float(config.get("nft_beta", 1.0))
        loss_scale = float(config.get("adv_clip_max", 1.0)) / beta
        loss_elem = (advantages * e_pos + (1.0 - advantages) * e_neg) * loss_scale
        clip_loss_frac = None
        if e_pos_clip is not None or e_neg_clip is not None:
            if e_pos_clip is None or e_neg_clip is None:
                raise ValueError(
                    "Both positive and negative clipped NFT energies are required."
                )
            clip_loss_elem = (
                advantages * e_pos_clip + (1.0 - advantages) * e_neg_clip
            ) * loss_scale
            clip_loss_frac = masked_mean(
                (loss_elem < clip_loss_elem).float(),
                loss_mask,
            ).item()
            loss_elem = torch.maximum(loss_elem, clip_loss_elem)
        return masked_mean(loss_elem, loss_mask), clip_loss_frac

    raise ValueError(f"Unsupported nft_loss_form: {loss_form}")


def _postprocess_advantages(
    advantages: torch.Tensor, *, config: DictConfig
) -> torch.Tensor:
    """Map centered advantages to [0, 1], leaving raw rewards unchanged."""
    adv_type = config.get("adv_type", "raw")
    if adv_type == "raw":
        return advantages
    adv_clip_max = float(config.get("adv_clip_max", 1.0))
    advantages = advantages.clamp(-adv_clip_max, adv_clip_max)
    advantages = (advantages + adv_clip_max) / (2.0 * adv_clip_max)
    return advantages


def _compute_delta_v_candidates(
    v_theta: torch.Tensor,
    v_old: torch.Tensor,
    *,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute raw delta_v and pos/neg candidate velocities."""
    delta_v = v_theta - v_old
    beta = float(config.get("nft_beta", 1.0))
    v_pos = v_old + beta * delta_v
    v_neg = v_old - beta * delta_v
    return delta_v, v_pos, v_neg


def _align_nft_signal(
    signal: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    signal = signal.reshape(batch_size, -1)
    if target.ndim == 1:
        return signal[:, 0]
    target_shape = target.shape
    target_size = math.prod(target_shape[1:])
    if signal.shape[1] == target_size:
        return signal.reshape(target_shape)
    if signal.shape[1] == 1:
        return signal.expand(target_shape)
    raise ValueError(
        f"Cannot align NFT signal shape {tuple(signal.shape)} "
        f"to target shape {tuple(target_shape)}."
    )


def _get_nft_reduce_dims(gather_type: str, ndim: int) -> tuple[int, ...]:
    if gather_type == "action_level":
        return tuple(range(2, ndim))
    if gather_type == "chunk_level":
        return tuple(range(1, ndim))
    raise ValueError(f"Unsupported nft_gather_type: {gather_type}")


def _reduce_nft_tensor(
    tensor: torch.Tensor,
    gather_type: str,
    reduction: str,
    *,
    keepdim: bool = False,
) -> torch.Tensor:
    reduce_dims = _get_nft_reduce_dims(gather_type, tensor.ndim)
    if not reduce_dims:
        return tensor.abs() if reduction == "norm" else tensor
    if reduction == "sum":
        return tensor.sum(dim=reduce_dims, keepdim=keepdim)
    if reduction == "mean":
        return tensor.mean(dim=reduce_dims, keepdim=keepdim)
    if reduction == "norm":
        return torch.linalg.vector_norm(tensor, dim=reduce_dims, keepdim=keepdim)
    raise ValueError(f"Unsupported NFT reduction: {reduction}")


def _build_schedule_params(
    schedule: torch.Tensor,  # [num_steps+1] linspace 1->0
    step_indices: torch.Tensor,  # [B]
    noise_level: torch.Tensor | float,
    x_t: torch.Tensor,  # reference tensor for ndim/device/dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute timestep & noise params, broadcast to [B, 1, ..., 1] for x_t.ndim.

    Returns: (t_bc, dt_bc, sigma_i, std_t_det)
    """
    ndim = x_t.ndim
    idx = step_indices.long()

    def pad(x):
        return x.view(-1, *([1] * (ndim - 1)))

    t_bc = pad(schedule[idx])
    dt_bc = pad(schedule[idx] - schedule[idx + 1])
    # SDE noise scale: sigma_i = sqrt(t / (1-t)) * noise_level
    safe_schedule = schedule.clone()
    safe_schedule[0] = safe_schedule[1]  # avoid div-by-zero at t=1
    sigma_i = pad(torch.sqrt(schedule[:-1] / (1 - safe_schedule[:-1]))[idx])
    nl = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
    sigma_i = sigma_i * (pad(nl) if nl.ndim > 0 else nl)
    std_t_det = (torch.sqrt(dt_bc.clamp_min(0)) * sigma_i).detach()
    return t_bc, dt_bc, sigma_i, std_t_det


def _compute_nft_target_and_pred(
    forward_inputs: Mapping[str, torch.Tensor],
    target_space: str,
    x_t: torch.Tensor,
    vel: torch.Tensor,
    t_bc: torch.Tensor,
    dt_bc: torch.Tensor,
    sigma_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build target and predicted state for the given NFT target space."""
    x0_pred = x_t - vel * t_bc
    x1_pred = x_t + vel * (1 - t_bc)
    if target_space == "x0":
        target = _slice_like(forward_inputs["nft_x0"], x_t)
        pred = x0_pred - (sigma_i**2 / 2) * x1_pred
    elif target_space == "xnext":
        target = _slice_like(forward_inputs["nft_xnext"], x_t)
        w0 = 1.0 - (t_bc - dt_bc)
        w1 = t_bc - dt_bc - sigma_i**2 * dt_bc / (2 * t_bc)
        pred = x0_pred * w0 + x1_pred * w1
    else:
        raise ValueError(f"Unsupported nft_target_space: {target_space}")
    return target, pred


def _slice_like(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    slices = (slice(None),) + tuple(slice(0, size) for size in reference.shape[1:])
    return target[slices]


def _compute_nft_weight(
    weight_mode: str,
    t_bc: torch.Tensor,
    std_t_det: torch.Tensor,
    noise_level: torch.Tensor,
    target: torch.Tensor,
    gather_type: str,
    *,
    config: DictConfig,
    pred: torch.Tensor,
    sample_type: str,
) -> torch.Tensor | float:
    """Compute per-element multiplicative weight for NFT energy.

    Supported weight_mode values (set via ``nft_weight_mode`` in config):
        "constant"  -- fixed scalar ``nft_weight_scale`` (default 1.0).
        "t"        -- t^2, scaled by ``nft_weight_scale``.
        "1-t"      -- (1-t)^2, scaled by ``nft_weight_scale``.
        "inv_t2"   -- 1 / (t^2 + eps), compensates x0 target gradient ∝ t^2.
        "inv_t3"   -- 1 / (t^3 + eps), overcompensates to bias toward low-t.
        "sigma"    -- 1 / (std_t_det^2 + eps).
        "adaptive" -- 1 / abs-error-mean (stop-grad), a la DiffusionNFT.
        "auto"     -- "adaptive" when noise_level==0 (ODE), otherwise "sigma" (SDE).
    """
    if weight_mode == "auto":
        if torch.all(noise_level == 0):
            weight_mode = "adaptive"
        else:
            weight_mode = "sigma"
    weight = config.get("nft_weight_scale", 1.0)
    if isinstance(weight, ListConfig):
        weight = list(weight)
    if isinstance(weight, (list, tuple)):
        if len(weight) != 2:
            raise ValueError("nft_weight_scale list must be [pos_scale, neg_scale].")
        weight = float(weight[0] if sample_type == "pos" else weight[1])
    else:
        weight = float(weight)
    if weight_mode == "constant":
        pass
    elif weight_mode == "t":
        weight *= t_bc**2
    elif weight_mode == "1-t":
        weight *= (1 - t_bc) ** 2
    elif weight_mode == "inv_t2":
        weight /= t_bc**2 + 1e-4
    elif weight_mode == "inv_t3":
        weight /= t_bc**3 + 1e-4
    elif weight_mode == "sigma":
        # loss in pi-step-nft paper
        weight /= std_t_det**2 + 1e-4
    elif weight_mode == "adaptive":
        with torch.no_grad():
            w = _reduce_nft_tensor(
                torch.abs(pred.double() - target.double()),
                gather_type,
                "mean",
                keepdim=True,
            )
            w = w.clamp(min=1e-5).to(pred.dtype)
        weight /= w
    return weight
