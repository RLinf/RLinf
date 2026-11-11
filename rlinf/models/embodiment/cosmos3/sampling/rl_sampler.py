# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flow-SDE transitions and Gaussian scores for Cosmos3 PPO."""

from __future__ import annotations

import math
from typing import Literal

import torch

TimeConvention = Literal["noise_at_one", "noise_at_zero"]


def get_timesteps(
    num_steps: int,
    device: torch.device | str,
    *,
    time_convention: TimeConvention = "noise_at_one",
) -> torch.Tensor:
    """Return an endpoint-inclusive denoising grid in the chosen convention."""
    if num_steps < 1:
        raise ValueError(f"Expected num_steps >= 1, got {num_steps}.")
    if time_convention == "noise_at_one":
        return torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    if time_convention == "noise_at_zero":
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    raise ValueError(f"Unsupported time_convention={time_convention!r}.")


def sample_mean_var(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    idx: torch.Tensor,
    *,
    noise_method: Literal["flow_ode", "flow_sde"],
    noise_level: float,
    num_steps: int | None = None,
    timesteps: torch.Tensor | None = None,
    time_convention: TimeConvention = "noise_at_one",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a transition mean and standard deviation on a fixed time grid.

    ``v_t`` is the derivative with respect to the grid's time coordinate.
    ``idx`` is scalar or contains one index per batch item. An explicit grid
    permits the same native Cosmos time schedule during rollout and recompute.
    """
    if timesteps is None:
        if num_steps is None:
            raise ValueError("Expected num_steps or an explicit timesteps tensor.")
        timesteps = get_timesteps(
            num_steps, x_t.device, time_convention=time_convention
        )
    if timesteps.ndim != 1 or timesteps.numel() < 2:
        raise ValueError("Expected a one-dimensional grid with at least two times.")
    if noise_method == "flow_sde":
        if timesteps.numel() < 3:
            raise ValueError("Flow-SDE requires at least two denoising steps.")
        if not math.isfinite(noise_level) or noise_level <= 0:
            raise ValueError(
                f"Expected a finite positive Flow-SDE noise_level, got {noise_level}."
            )
    elif noise_method != "flow_ode":
        raise ValueError(f"Unsupported noise_method={noise_method!r}.")

    dtype = torch.float64 if torch.float64 in (x_t.dtype, v_t.dtype) else torch.float32
    x_t, v_t = x_t.to(dtype), v_t.to(dtype)
    timesteps = timesteps.to(device=x_t.device, dtype=dtype)
    idx = torch.as_tensor(idx, device=x_t.device, dtype=torch.long)
    if idx.ndim > 1 or (idx.ndim == 1 and idx.numel() not in (1, x_t.shape[0])):
        raise ValueError("Expected a scalar step index or one index per batch item.")
    t = timesteps[idx]
    next_t = timesteps[idx + 1]
    if t.ndim == 1:
        shape = (t.shape[0],) + (1,) * (x_t.ndim - 1)
        t, next_t = t.reshape(shape), next_t.reshape(shape)

    if time_convention == "noise_at_one":
        delta = t - next_t
        noise_prediction = x_t + (1 - t) * v_t
        distance_to_data = t
        safe_t = torch.where(t == 1, next_t, t)
        sigma_ratio = t / (1 - safe_t)
    elif time_convention == "noise_at_zero":
        delta = next_t - t
        noise_prediction = x_t - t * v_t
        distance_to_data = 1 - t
        safe_t = torch.where(t == 0, next_t, t)
        sigma_ratio = (1 - t) / safe_t
    else:
        raise ValueError(f"Unsupported time_convention={time_convention!r}.")

    mean = x_t + (next_t - t) * v_t
    if noise_method == "flow_ode":
        return mean, torch.zeros_like(x_t)
    sigma_squared = noise_level**2 * sigma_ratio
    mean = mean - (sigma_squared * delta / (2 * distance_to_data)) * noise_prediction
    std = torch.sqrt(sigma_squared * delta).expand_as(x_t)
    return mean, std


def gaussian_logprob(
    sample: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-coordinate scores, excluding masked or deterministic entries.

    Zero-variance entries are a scoring convention for ODE/conditioned slots,
    not a Gaussian density. PPO must select a positive-variance SDE transition.
    """
    active = sigma > 0
    if mask is not None:
        active = active & mask.to(device=sigma.device, dtype=torch.bool)
    safe_sigma = torch.where(active, sigma, torch.ones_like(sigma))
    residual = torch.where(active, sample - mu, torch.zeros_like(sample))
    score = (
        -torch.log(safe_sigma)
        - 0.5 * math.log(2 * math.pi)
        - 0.5 * (residual / safe_sigma) ** 2
    )
    return torch.where(active, score, torch.zeros_like(score))


def gaussian_entropy(
    sigma: torch.Tensor, *, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Return per-coordinate entropy for unmasked stochastic entries."""
    active = sigma > 0
    if mask is not None:
        active = active & mask.to(device=sigma.device, dtype=torch.bool)
    safe_sigma = torch.where(active, sigma, torch.ones_like(sigma))
    entropy = 0.5 * math.log(2 * math.pi * math.e) + torch.log(safe_sigma)
    return torch.where(active, entropy, torch.zeros_like(entropy))
