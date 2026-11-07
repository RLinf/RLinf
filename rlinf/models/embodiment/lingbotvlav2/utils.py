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

"""LingBot-VLA 2.0 RL helpers that do not need the V2 package.

A training rollout follows the flow ODE except at one random denoising step per
sample, where it draws from the flow-SDE transition. PPO scores only that step,
as RLinf's LingBot-VLA and OpenPI policies do.
"""

from typing import Callable, Optional

import torch

_TRAINABLE_PARTS = (
    ".qwen_expert.",
    ".state_proj.",
    ".action_in_proj.",
    ".action_out_proj.",
    ".action_time_mlp_",
    ".time_mlp_",
)
# MoE routers stay frozen: a small router change can flip the discrete expert
# choice for inputs far from the training batch.
_FROZEN_PARTS = (".mlp.gate.",)


def is_rl_trainable(name: str) -> bool:
    """Return whether RL updates the V2 policy parameter ``name``."""
    return any(part in name for part in _TRAINABLE_PARTS) and not any(
        part in name for part in _FROZEN_PARTS
    )


# Maps a latent [batch, horizon, dim] and per-sample times [batch] to a velocity.
VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def flow_sde_transition(
    x: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
    delta: torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FP32 Gaussian transition used by sampling and PPO replay.

    At t=1 the diffusion denominator uses the next time, as in RLinf's
    LingBot-VLA/OpenPI flow-SDE. Every step has positive variance, including
    the final step.
    """
    x, velocity = x.float(), velocity.float()
    denominator = torch.where(time == 1, delta, 1 - time)
    sigma_squared = noise_level**2 * time / denominator
    mean = (
        x
        - delta * velocity
        - sigma_squared * delta / (2 * time) * (x + (1 - time) * velocity)
    )
    std = torch.sqrt(delta * sigma_squared)
    return mean, std


def gaussian_logprob(
    sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Score a detached continuous sample without reducing latent dimensions."""
    return torch.distributions.Normal(mean.float(), std.float()).log_prob(
        sample.float()
    )


def _denoise_step(
    x: torch.Tensor,
    velocity_fn: VelocityFn,
    times: torch.Tensor,
    steps: torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the ODE update and the SDE mean and std at grid indices ``steps``.

    Sampling and replay both go through here, so a replayed step repeats the
    sampled arithmetic exactly.
    """
    time = times[steps]
    delta = (time - times[steps + 1]).view(-1, 1, 1)
    velocity = velocity_fn(x, time).float()
    mean, std = flow_sde_transition(
        x, velocity, time.view(-1, 1, 1), delta, noise_level
    )
    return x - delta * velocity, mean, std


def sample_with_one_sde_step(
    noise: torch.Tensor,
    velocity_fn: VelocityFn,
    times: torch.Tensor,
    noise_level: float,
    sde_steps: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Denoise ``noise`` along ``times``, drawing from the SDE at ``sde_steps``.

    Args:
        noise: Latent at ``times[0] == 1``, shaped ``[batch, horizon, dim]``.
        velocity_fn: Velocity for a latent and per-sample times.
        times: Decreasing time grid from 1 to 0, shaped ``[num_steps + 1]``.
        noise_level: Flow-SDE noise scale.
        sde_steps: Index of each sample's stochastic step, shaped ``[batch]``;
            None follows the ODE at every step.

    Returns:
        The final latent; the latent before and after each sample's stochastic
        step, shaped ``[batch, 2, horizon, dim]``; and that step's log density
        per coordinate, shaped ``[batch, horizon, dim]``. The last two are zero
        when ``sde_steps`` is None.
    """
    x = noise.float()
    states = x.new_zeros((len(x), 2, *x.shape[1:]))
    logprobs = torch.zeros_like(x)
    for step in range(len(times) - 1):
        steps = torch.full((len(x),), step, device=x.device, dtype=torch.long)
        ode, mean, std = _denoise_step(x, velocity_fn, times, steps, noise_level)
        if sde_steps is None:
            x = ode
            continue
        chosen = (sde_steps == step).view(-1, 1, 1)
        sample = mean + std * torch.randn_like(mean)
        states[:, 0] = torch.where(chosen, x, states[:, 0])
        states[:, 1] = torch.where(chosen, sample, states[:, 1])
        logprobs = torch.where(chosen, gaussian_logprob(sample, mean, std), logprobs)
        x = torch.where(chosen, sample, ode)
    return x, states, logprobs


def score_sde_step(
    states: torch.Tensor,
    velocity_fn: VelocityFn,
    times: torch.Tensor,
    sde_steps: torch.Tensor,
    noise_level: float,
) -> torch.Tensor:
    """Return the log density of each recorded stochastic step per coordinate.

    ``states`` and ``sde_steps`` are what :func:`sample_with_one_sde_step`
    recorded. The density is differentiable through ``velocity_fn``.
    """
    _, mean, std = _denoise_step(
        states[:, 0], velocity_fn, times, sde_steps, noise_level
    )
    return gaussian_logprob(states[:, 1], mean, std)
