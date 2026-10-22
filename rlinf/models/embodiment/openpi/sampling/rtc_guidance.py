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

"""RTC overlap guidance for the vendored Pi0 eval sampler.

``RTCGuidanceContext`` and ``build_rtc_target_and_mask`` are the shared
protocol used by both ``openpi`` and the leftover OpenPI PyTorch
sampler; the Euler loop below is the vendored-Pi0 implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
from rlinf.models.embodiment.openpi.sampling import rl_sampler


@dataclass
class RTCGuidanceContext:
    """RTC context passed from rollout runtime to the OpenPI sampler."""

    prev_model_actions: torch.Tensor | None = None  # [B, H, A_model]
    executed_horizon: int = 0
    delay_steps: int = 0

    def get_prev_remaining(self) -> torch.Tensor | None:
        if self.prev_model_actions is None:
            return None
        executed_horizon = int(max(self.executed_horizon, 0))
        if executed_horizon >= self.prev_model_actions.shape[1]:
            return None
        return self.prev_model_actions[:, executed_horizon:, :]


def build_rtc_target_and_mask(
    prev_remaining: torch.Tensor | None,
    horizon: int,
    action_dim: int,
    delay_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build overlap target and soft mask in model action space."""
    batch_size = 1 if prev_remaining is None else prev_remaining.shape[0]
    target = torch.zeros((batch_size, horizon, action_dim), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, horizon, 1), device=device, dtype=dtype)

    if prev_remaining is None or prev_remaining.numel() == 0:
        return target, mask

    overlap = min(prev_remaining.shape[1], horizon)
    if overlap <= 0:
        return target, mask

    target[:, :overlap] = prev_remaining[:, :overlap].to(device=device, dtype=dtype)
    hard_end = min(max(int(delay_steps), 0), overlap)
    if hard_end > 0:
        # Actions that are likely to be executed before the new chunk arrives
        # are treated as hard constraints.
        mask[:, :hard_end, 0] = 1.0
    if hard_end < overlap:
        # Later overlap positions are softly guided so the new plan can still
        # adapt to the latest camera observation.
        i = torch.arange(hard_end, overlap, device=device, dtype=dtype)
        denom = max(float(overlap - hard_end + 1), 1.0)
        c_i = (overlap - i) / denom
        soft = c_i * (torch.expm1(c_i) / (math.e - 1.0))
        mask[:, hard_end:overlap, 0] = soft
    return target, mask


@torch.no_grad()
def sample_actions_with_rtc_guidance(
    pi0_model,
    observation,
    rtc_context: RTCGuidanceContext,
    *,
    num_steps: int,
    noise: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
    guidance_clip: float = 5.0,
) -> torch.Tensor:
    """Euler ODE sampling with overlap guidance against the previous chunk."""
    observation = preprocess_observation(observation, train=False)
    B = observation.state.shape[0]
    device = observation.state.device
    if noise is None:
        noise = torch.randn(
            B,
            pi0_model.action_horizon,
            pi0_model.action_dim,
            device=device,
            dtype=torch.float32,
            generator=rng,
        )
    else:
        noise = noise.to(device=device, dtype=torch.float32)

    prefix_out, prefix_mask, kv_cache = pi0_model.build_prefix_cache(observation)
    del prefix_out

    prev_remaining = (
        rtc_context.get_prev_remaining() if rtc_context is not None else None
    )
    target, mask = build_rtc_target_and_mask(
        prev_remaining=prev_remaining,
        horizon=pi0_model.action_horizon,
        action_dim=pi0_model.action_dim,
        delay_steps=0 if rtc_context is None else rtc_context.delay_steps,
        device=device,
        dtype=noise.dtype,
    )

    x_t = noise
    paper_tau = torch.tensor(0.0, device=device, dtype=noise.dtype)
    dt = torch.tensor(1.0 / num_steps, device=device, dtype=noise.dtype)
    idx_step = torch.empty(B, device=device, dtype=torch.long)

    for idx in range(num_steps):
        t_val = float(rl_sampler.get_timesteps(num_steps, device)[idx].item())
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        suffix_act = pi0_model.run_suffix(
            observation, x_t, t_tensor, kv_cache, prefix_mask
        )
        v_t = pi0_model.velocity_from_suffix(suffix_act).to(torch.float32)
        idx_step.fill_(idx)
        x_t_mean, x_t_std = rl_sampler.sample_mean_var(
            x_t.to(torch.float32),
            v_t,
            idx_step,
            noise_method="flow_ode",
            noise_level=0.0,
            num_steps=num_steps,
        )
        if prev_remaining is not None and prev_remaining.numel() > 0:
            guidance_term = (target - x_t_mean) * mask
            guidance_scale = torch.clamp(
                (1.0 - paper_tau) / torch.clamp(paper_tau + 1e-4, min=1e-4),
                max=float(guidance_clip),
            )
            x_t_mean = x_t_mean + guidance_scale * guidance_term
        step_noise = torch.randn(
            x_t.shape, device=device, dtype=torch.float32, generator=rng
        )
        x_t = x_t_mean + step_noise * x_t_std
        paper_tau = paper_tau + dt
    return x_t
