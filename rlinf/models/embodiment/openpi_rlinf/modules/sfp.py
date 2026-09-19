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

"""Streaming Flow Policy targets for the Pi0.5 action expert.

Ordinary flow matching denoises a whole action chunk at once, and its flow time
is a denoising coordinate unrelated to robot time: nothing is executable until
every denoising step has run. SFP integrates the chunk into a trajectory and
maps the flow time onto it instead, so ``t=0`` is the action state the robot is
at now and ``t=1`` is the end of the chunk. A policy trained this way can be
integrated forward from wherever it currently is, which is what lets a caller
overlap action generation with execution.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_sfp_flow_targets(
    actions: Tensor,
    action_states: Tensor | None,
    time: Tensor,
    noise: Tensor,
    *,
    action_horizon: int,
    sigma: float = 0.16,
    noise_decay: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """Locate ``time`` on the action trajectory and return the velocity there.

    Args:
        actions: Normalized action deltas, ``(B, action_horizon, action_dim)``.
        action_states: Normalized cumulative action state at the start of the
            chunk, ``(B, action_dim)``. ``None`` starts the trajectory at zero.
        time: Flow time in ``(0, 1]``, shape ``(B,)``.
        noise: Gaussian noise, shape ``(B, 1, action_dim)``.
        action_horizon: Number of action deltas in one chunk.
        sigma: Scale of the noise injected at ``t=0``.
        noise_decay: Rate at which that noise decays along the trajectory.

    Returns:
        The noised position ``x_t`` and the velocity target ``u_t``, both
        ``(B, 1, action_dim)`` in float32.

    The trajectory is a cumulative sum over ``action_states`` and ``actions``,
    so the two are only commensurable when normalized on a common scale. Use
    :class:`~rlinf.models.embodiment.openpi.dataconfig.sfp_transforms.SfpNormalize`,
    whose pure scaling commutes with the cumulative sum; the affine quantile
    normalization openpi applies elsewhere does not.
    """
    if actions.ndim != 3 or actions.shape[1] != action_horizon:
        raise ValueError(
            "actions must have shape (batch, action_horizon, action_dim); got "
            f"{tuple(actions.shape)} with action_horizon={action_horizon}."
        )
    batch_size, _, action_dim = actions.shape
    if time.shape != (batch_size,):
        raise ValueError(
            f"time must have shape {(batch_size,)}, got {tuple(time.shape)}."
        )
    if noise.shape != (batch_size, 1, action_dim):
        raise ValueError(
            f"noise must have shape {(batch_size, 1, action_dim)}, "
            f"got {tuple(noise.shape)}."
        )

    device = actions.device
    actions = actions.to(dtype=torch.float32)
    time = time.to(device=device, dtype=torch.float32)
    noise = noise.to(device=device, dtype=torch.float32)

    if action_states is None:
        start = actions.new_zeros((batch_size, 1, action_dim))
    else:
        if action_states.shape != (batch_size, action_dim):
            raise ValueError(
                f"action_states must have shape {(batch_size, action_dim)}, "
                f"got {tuple(action_states.shape)}."
            )
        start = action_states.to(device=device, dtype=torch.float32).unsqueeze(1)

    trajectory = torch.cumsum(torch.cat([start, actions], dim=1), dim=1)

    scaled_time = time * action_horizon
    index = scaled_time.floor().long().clamp_(0, action_horizon - 1)
    alpha = (scaled_time - index)[:, None, None]
    row = torch.arange(batch_size, device=device)
    current = trajectory[row, index].unsqueeze(1)
    segment = trajectory[row, index + 1].unsqueeze(1) - current

    injected = sigma * torch.exp(-noise_decay * time)[:, None, None] * noise
    x_t = current + alpha * segment + injected
    u_t = segment * action_horizon - noise_decay * injected
    return x_t, u_t
