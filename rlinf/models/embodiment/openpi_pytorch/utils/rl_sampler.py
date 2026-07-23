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

"""Stochastic flow-matching helpers for the vendored OpenPI PyTorch RL path.

These reproduce the chain-collecting SDE sampler used by the original RL
implementation in :mod:`rlinf.models.embodiment.openpi.openpi_action_model`, but
as plain functions that operate on the vendored ``Pi0`` model's
``build_prefix_cache`` / ``run_suffix`` / ``velocity_from_suffix`` primitives.

Scope (only what BEHAVIOR pi05 PPO needs):
- ``noise_method``: ``flow_ode`` (deterministic Euler step) and ``flow_sde``
  (Euler + per-step Gaussian noise). ``flow_cps`` / ``flow_noise`` are
  rejected so the failure is loud.
- ``joint_logprob = False`` only — a single random denoise step is the
  stochastic one; every other step is a deterministic flow_ode step.
- ``value_after_vlm = True`` with ``mean_token`` pooling (the only combination
  the BEHAVIOR pi05 config uses).
"""

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the ``(x_t_mean, x_t_std)`` pair for one denoise step.

    Matches ``OpenPi0ForRLActionPrediction.sample_mean_var_val`` (the
    ``v_t`` + ``value_t`` plumbing is owned by the caller here).

    Args:
        x_t:           (B, H, A) current latent.
        v_t:           (B, H, A) velocity prediction (already in fp32).
        idx:           (B,) long denoise-step index in ``[0, num_steps - 1]``.
        noise_method:  ``"flow_ode"`` or ``"flow_sde"``.
        noise_level:   scalar sigma scale (the YAML's ``openpi.noise_level``).
        num_steps:     total denoise steps.

    Returns:
        ``(x_t_mean, x_t_std)``, both shape ``(B, H, A)``.
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
        # Reproduces openpi/openpi_action_model.py:809-816 exactly.
        denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
        sigma_ratio = timesteps / (1 - denom_timesteps)
        sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
        sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
        x0_weight = torch.ones_like(t_input) - (t_input - delta)
        x1_weight = (t_input - delta) - sigma_i * sigma_i * delta / (2 * t_input)
        x_t_std = torch.sqrt(delta) * sigma_i
    else:
        raise NotImplementedError(
            f"noise_method={noise_method!r} is not implemented in the PyTorch "
            "OpenPI RL port. Supported: 'flow_ode', 'flow_sde'."
        )

    x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
    return x_t_mean, x_t_std


def gaussian_logprob(
    sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Per-element Gaussian log-probability; zero wherever ``sigma == 0``.

    The zero-fallback at ``sigma == 0`` is needed because flow_ode steps
    produce a deterministic update and would otherwise blow up under the
    standard ``-log sigma`` term. Matches the non-``safe_get_logprob`` branch
    of ``OpenPi0ForRLActionPrediction.get_logprob_norm``.
    """
    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    log_two_pi = math.log(2 * math.pi)
    log_prob = (
        -torch.log(sigma_safe)
        - 0.5 * log_two_pi
        - 0.5 * ((sample - mu) / sigma_safe) ** 2
    )
    return torch.where(mask, torch.zeros_like(log_prob), log_prob)


def value_from_prefix(
    value_head: torch.nn.Module,
    prefix_out: torch.Tensor,
    prefix_mask: torch.Tensor,
    *,
    mode: str = "mean_token",
) -> torch.Tensor:
    """Pool ``prefix_out`` with ``prefix_mask`` and run ``value_head`` → ``[B]``.

    ``mean_token`` averages over all valid prefix tokens (images + language),
    which is the BEHAVIOR pi05 configuration. Other pooling modes from the
    original code (``first_token``/``last_token``) are not wired through
    here — they raise ``NotImplementedError`` so a config typo fails loudly
    rather than silently producing a wrong value.
    """
    if mode != "mean_token":
        raise NotImplementedError(
            f"value_vlm_mode={mode!r} is not implemented in the PyTorch OpenPI "
            "RL port. Supported: 'mean_token'."
        )
    mask_f = prefix_mask.to(prefix_out.dtype).unsqueeze(-1)
    summed = (prefix_out * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    # Keep the pooled features in the prefix dtype so the value head's bf16/fp32
    # weights line up; promote the scalar output to fp32 for the PPO critic loss
    # (which assert-checks float32 alignment with returns / prev_values).
    pooled = summed / denom
    return value_head(pooled)[:, 0].to(torch.float32)
