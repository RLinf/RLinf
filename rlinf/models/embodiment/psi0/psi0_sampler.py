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

"""Psi0-specific stochastic flow transition used only for RL sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Psi0Transition:
    """One sampled transition and the information needed to recompute it."""

    x: torch.Tensor
    next_x: torch.Tensor
    timestep: torch.Tensor
    sigma: torch.Tensor
    sigma_next: torch.Tensor
    sample_mask: torch.Tensor
    logprobs: torch.Tensor


def gaussian_logprob(
    sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Return elementwise Gaussian log-probabilities."""
    if torch.any(std <= 0):
        raise ValueError("Psi0 stochastic transition requires positive std.")
    return (
        -torch.log(std)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((sample - mean) / std).square()
    )


def transition_mean_std(
    x: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    *,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match diffusers' stochastic FlowMatch Euler transition."""
    if noise_scale <= 0:
        raise ValueError("Psi0 stochastic noise_scale must be positive.")
    sigma = sigma.to(device=x.device, dtype=torch.float32)
    sigma_next = sigma_next.to(device=x.device, dtype=torch.float32)
    while sigma.ndim < x.ndim:
        sigma = sigma.unsqueeze(-1)
        sigma_next = sigma_next.unsqueeze(-1)
    x0 = x.float() - sigma * velocity.float()
    mean = (1.0 - sigma_next) * x0
    std = torch.ones_like(mean) * sigma_next * noise_scale
    return mean, std


class Psi0StochasticTransitionSampler:
    """Use the checkpoint scheduler grid with one stochastic denoise step."""

    def __init__(self, *, noise_scale: float = 1.0) -> None:
        if noise_scale <= 0:
            raise ValueError("Psi0 stochastic noise_scale must be positive.")
        self.noise_scale = float(noise_scale)

    def sample(
        self,
        *,
        scheduler,
        num_inference_steps: int,
        initial_noise: torch.Tensor,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        condition_actions: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        transition_index: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, Psi0Transition]:
        """Sample a plan and retain one finite-density transition."""
        device = initial_noise.device
        batch_size, horizon, _ = initial_noise.shape
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps.to(device=device)
        sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)
        if len(timesteps) < 3 or len(sigmas) != len(timesteps) + 1:
            raise ValueError("Unexpected Psi0 FlowMatch scheduler grid.")

        if transition_index is None:
            transition_index = int(
                torch.randint(
                    0,
                    len(timesteps) - 2,
                    (1,),
                    device=device,
                    generator=generator,
                ).item()
            )
        if not 0 <= transition_index < len(timesteps) - 2:
            raise ValueError(
                "Psi0 stochastic transition must exclude the terminal low-variance steps."
            )

        if condition_mask is None:
            condition_mask = torch.zeros(
                (batch_size, horizon), dtype=torch.bool, device=device
            )
        else:
            condition_mask = condition_mask.to(device=device, dtype=torch.bool)
        if condition_mask.shape != (batch_size, horizon):
            raise ValueError("Psi0 RTC condition_mask has an invalid shape.")
        if condition_mask.any():
            if (
                condition_actions is None
                or condition_actions.shape != initial_noise.shape
            ):
                raise ValueError(
                    "Psi0 RTC conditions require full-shape condition_actions."
                )
            condition_actions = condition_actions.to(device=device, dtype=torch.float32)

        x = initial_noise.float()
        selected: Psi0Transition | None = None
        for index, timestep in enumerate(timesteps):
            if condition_mask.any():
                x = torch.where(condition_mask[..., None], condition_actions, x)
            model_timestep = torch.full(
                (batch_size, horizon),
                float(timestep.item()),
                device=device,
                dtype=torch.float32,
            )
            model_timestep = torch.where(
                condition_mask, torch.zeros_like(model_timestep), model_timestep
            )
            velocity = velocity_fn(x, model_timestep).float()
            sigma = sigmas[index].expand(batch_size)
            sigma_next = sigmas[index + 1].expand(batch_size)

            if index == transition_index:
                mean, std = transition_mean_std(
                    x,
                    velocity,
                    sigma,
                    sigma_next,
                    noise_scale=self.noise_scale,
                )
                noise = torch.randn(
                    x.shape,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                next_x = mean + std * noise
                sample_mask = ~condition_mask
                logprobs = gaussian_logprob(next_x, mean, std)
                logprobs = logprobs * sample_mask[..., None]
                selected = Psi0Transition(
                    x=x,
                    next_x=next_x,
                    timestep=model_timestep,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    sample_mask=sample_mask,
                    logprobs=logprobs,
                )
            else:
                dt = (sigma_next - sigma).view(batch_size, 1, 1)
                next_x = x + dt * velocity
            x = next_x

        if selected is None:
            raise RuntimeError("Psi0 stochastic transition was not sampled.")
        return x, selected

    def recompute(
        self,
        *,
        transition_x: torch.Tensor,
        transition_next: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute log-probability and entropy for a saved transition."""
        mean, std = transition_mean_std(
            transition_x,
            velocity,
            sigma,
            sigma_next,
            noise_scale=self.noise_scale,
        )
        mask = sample_mask.to(dtype=torch.bool, device=mean.device)[..., None]
        logprobs = gaussian_logprob(transition_next.float(), mean, std)
        entropy = torch.log(std) + 0.5 * (1.0 + math.log(2.0 * math.pi))
        return logprobs * mask, entropy * mask
