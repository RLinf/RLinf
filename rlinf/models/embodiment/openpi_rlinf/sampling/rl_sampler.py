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

"""Stochastic flow-matching helpers for the vendored OpenPI PyTorch RL path."""

from __future__ import annotations

import math

import torch


def get_timesteps(num_steps: int, device: torch.device | str) -> torch.Tensor:
    """Return ``[1, (N-1)/N, ..., 1/N, 0]`` of length ``num_steps + 1``."""
    timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device)
    return torch.cat([timesteps, torch.zeros(1, device=device)])


def sample_mean_var(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    idx: torch.Tensor,
    *,
    noise_method: str,
    noise_level: float,
    num_steps: int,
    noise_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the ``(x_t_mean, x_t_std)`` pair for one denoise step.

    Matches ``OpenPi0ForRLActionPrediction.sample_mean_var_val``.
    ``noise_std`` is required when ``noise_method='flow_noise'`` (from ExploreNoiseNet).
    """
    device = x_t.device
    timesteps = get_timesteps(num_steps, device).to(x_t.dtype)
    t_input = timesteps[idx][:, None, None].expand_as(x_t)
    delta = (timesteps[idx] - timesteps[idx + 1])[:, None, None].expand_as(x_t)
    x0_pred = x_t - v_t * t_input
    x1_pred = x_t + v_t * (1 - t_input)

    if noise_method == "flow_ode":
        x0_weight = 1 - (t_input - delta)
        x1_weight = t_input - delta
        x_t_std = torch.zeros_like(t_input)
    elif noise_method == "flow_sde":
        denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
        sigma_ratio = timesteps / (1 - denom_timesteps)
        sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
        sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
        x0_weight = torch.ones_like(t_input) - (t_input - delta)
        x1_weight = (t_input - delta) - sigma_i * sigma_i * delta / (2 * t_input)
        x_t_std = torch.sqrt(delta) * sigma_i
    elif noise_method == "flow_cps":
        pi = torch.pi
        level = torch.as_tensor(noise_level, device=device, dtype=x_t.dtype)
        cos_term = torch.cos(pi * level / 2)
        sin_term = torch.sin(pi * level / 2)
        x0_weight = torch.ones_like(t_input) - (t_input - delta)
        x1_weight = (t_input - delta) * cos_term
        x_t_std = (t_input - delta) * sin_term
    elif noise_method == "flow_noise":
        if noise_std is None:
            raise ValueError(
                "noise_method='flow_noise' requires noise_std from ExploreNoiseNet."
            )
        x0_weight = 1 - (t_input - delta)
        x1_weight = t_input - delta
        x_t_std = noise_std.to(dtype=x_t.dtype)
    else:
        raise NotImplementedError(
            f"noise_method={noise_method!r} is not implemented. "
            "Supported: 'flow_ode', 'flow_sde', 'flow_cps', 'flow_noise'."
        )

    x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
    return x_t_mean, x_t_std


def gaussian_logprob(
    sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Per-element Gaussian log-probability; zero wherever ``sigma == 0``."""
    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    log_two_pi = math.log(2 * math.pi)
    log_prob = (
        -torch.log(sigma_safe)
        - 0.5 * log_two_pi
        - 0.5 * ((sample - mu) / sigma_safe) ** 2
    )
    return torch.where(mask, torch.zeros_like(log_prob), log_prob)


def gaussian_entropy(sigma: torch.Tensor) -> torch.Tensor:
    """Per-element Gaussian entropy; zero wherever ``sigma == 0``."""
    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
    return torch.where(mask, torch.zeros_like(entropy), entropy)


def value_from_prefix(
    value_head: torch.nn.Module,
    prefix_out: torch.Tensor,
    prefix_mask: torch.Tensor,
    *,
    mode: str = "mean_token",
) -> torch.Tensor:
    """Pool ``prefix_out`` with ``prefix_mask`` and run ``value_head`` → ``[B]``."""
    mask_f = prefix_mask.to(prefix_out.dtype)
    if mode == "mean_token":
        mask_exp = mask_f.unsqueeze(-1)
        summed = (prefix_out * mask_exp).sum(dim=1)
        denom = mask_exp.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
    elif mode == "first_token":
        first_idx = mask_f.to(torch.long).argmax(dim=1)
        pooled = prefix_out[
            torch.arange(prefix_out.shape[0], device=prefix_out.device), first_idx
        ]
    elif mode == "last_token":
        rev = torch.flip(mask_f, dims=[1])
        last_from_end = rev.to(torch.long).argmax(dim=1)
        last_idx = mask_f.shape[1] - 1 - last_from_end
        pooled = prefix_out[
            torch.arange(prefix_out.shape[0], device=prefix_out.device), last_idx
        ]
    else:
        raise NotImplementedError(
            f"value_vlm_mode={mode!r} is not implemented. "
            "Supported: 'mean_token', 'first_token', 'last_token'."
        )
    return value_head(pooled)[:, 0].to(torch.float32)


def value_from_suffix(
    value_head: torch.nn.Module,
    suffix_out: torch.Tensor,
    *,
    action_chunk: int | None = None,
    chunk_critic_input: bool = False,
    detach_critic_input: bool = False,
) -> torch.Tensor:
    """Mean-pool suffix hidden states and run ``value_head`` → ``[B]``."""
    if chunk_critic_input and action_chunk is not None:
        pooled = torch.mean(suffix_out[:, :action_chunk], dim=1)
    else:
        pooled = torch.mean(suffix_out, dim=1)
    if detach_critic_input:
        pooled = pooled.detach()
    return value_head(pooled)[:, 0].to(torch.float32)
