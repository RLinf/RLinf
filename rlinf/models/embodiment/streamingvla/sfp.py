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

"""Pure PyTorch primitives for the Streaming Flow Policy objective."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def create_sinusoidal_pos_embedding(
    time: Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> Tensor:
    """Create the reference sine/cosine embedding for scalar timesteps."""
    if dimension % 2:
        raise ValueError(f"dimension must be divisible by 2, got {dimension}.")
    if time.ndim != 1:
        raise ValueError(
            f"time must have shape (batch_size,), got {tuple(time.shape)}."
        )

    fraction = torch.linspace(
        0.0,
        1.0,
        dimension // 2,
        dtype=torch.float64,
        device=time.device,
    )
    period = min_period * (max_period / min_period) ** fraction
    sinusoid = (2.0 * math.pi / period)[None, :] * time[:, None]
    return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=1)


def sample_beta(
    alpha: float, beta: float, batch_size: int, device: torch.device
) -> Tensor:
    """Sample a beta distribution with the reference fp32 semantics."""
    alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_tensor = torch.as_tensor(beta, dtype=torch.float32, device=device)
    return torch.distributions.Beta(alpha_tensor, beta_tensor).sample((batch_size,))


def make_attention_masks_and_position_ids(
    pad_masks: Tensor, attention_categories: Tensor
) -> tuple[Tensor, Tensor]:
    """Build the prefix-LM mask and position ids used by StreamingVLA."""
    if pad_masks.ndim != 2 or attention_categories.ndim != 2:
        raise ValueError(
            "pad_masks and attention_categories must both be rank-2 tensors."
        )
    cumulative = torch.cumsum(attention_categories, dim=1)
    attention = cumulative[:, None, :] <= cumulative[:, :, None]
    padding = pad_masks[:, None, :] & pad_masks[:, :, None]
    positions = torch.cumsum(pad_masks, dim=1) - 1
    return attention & padding, positions


def compute_sfp_flow_targets(
    actions: Tensor,
    action_states: Tensor | None,
    time: Tensor,
    noise: Tensor | None,
    *,
    action_horizon: int,
    sigma: float = 0.16,
    noise_decay: float = 4.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute StreamingVLA interpolation and velocity targets in fp32.

    Args:
        actions: Normalized action deltas with shape ``(B, H, D)``.
        action_states: Normalized initial action state with shape ``(B, D)``.
        time: Explicit flow timestep with shape ``(B,)``.
        noise: Optional explicit Gaussian noise with shape ``(B, 1, D)``.
        action_horizon: Number of action deltas in each trajectory.
        sigma: Initial noise scale.
        noise_decay: Exponential noise decay coefficient.

    Returns:
        ``scaled_t``, ``index``, ``alpha``, noisy ``x_t``, target ``u_t``, and
        the injected noise. All floating-point outputs are fp32.
    """
    actions = actions.to(dtype=torch.float32)
    if actions.ndim != 3 or actions.shape[1] != action_horizon:
        raise ValueError(
            "actions must have shape (batch, action_horizon, action_dim); "
            f"got {tuple(actions.shape)} with horizon={action_horizon}."
        )
    time = time.to(device=actions.device, dtype=torch.float32)
    if time.shape != (actions.shape[0],):
        raise ValueError(
            f"time must have shape {(actions.shape[0],)}, got {tuple(time.shape)}."
        )

    if action_states is None:
        initial_state = torch.zeros(
            actions.shape[0],
            1,
            actions.shape[2],
            device=actions.device,
            dtype=torch.float32,
        )
    else:
        expected_shape = (actions.shape[0], actions.shape[2])
        if action_states.shape != expected_shape:
            raise ValueError(
                f"action_states must have shape {expected_shape}, "
                f"got {tuple(action_states.shape)}."
            )
        initial_state = action_states.to(
            device=actions.device, dtype=torch.float32
        ).unsqueeze(1)

    trajectory = torch.cumsum(torch.cat([initial_state, actions], dim=1), dim=1)
    scaled_t = time * action_horizon
    index = torch.clamp(scaled_t.floor().long(), 0, action_horizon - 1)
    alpha = scaled_t - index.float()
    batch_index = torch.arange(actions.shape[0], device=actions.device)
    current = trajectory[batch_index, index].unsqueeze(1)
    following = trajectory[batch_index, index + 1].unsqueeze(1)
    interpolated = current + alpha[:, None, None] * (following - current)
    velocity = (following - current) * action_horizon

    if noise is None:
        noise = torch.randn_like(interpolated)
    elif noise.shape != interpolated.shape:
        raise ValueError(
            f"noise must have shape {tuple(interpolated.shape)}, "
            f"got {tuple(noise.shape)}."
        )
    noise = noise.to(device=actions.device, dtype=torch.float32)
    added_noise = sigma * torch.exp(-noise_decay * time)[:, None, None] * noise
    return (
        scaled_t,
        index,
        alpha,
        interpolated + added_noise,
        velocity - noise_decay * added_noise,
        added_noise,
    )
