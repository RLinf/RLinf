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

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        half_dim = embed_dim // 2
        self.register_buffer(
            "frequencies",
            torch.exp(
                torch.arange(half_dim, dtype=torch.float32)
                * -(math.log(10000.0) / (half_dim - 1))
            ),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time * self.frequencies.to(dtype=time.dtype)
        return self.projection(torch.cat((angles.sin(), angles.cos()), dim=-1))


class OGPOFlowActor(nn.Module):
    """MLP flow used by the ManiSkill OGPO recipe."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        denoising_steps: int,
    ):
        super().__init__()
        self.register_buffer("action_scale", torch.ones(action_dim))
        self.register_buffer("action_bias", torch.zeros(action_dim))
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.time_embedding = SinusoidalTimeEmbedding()
        input_dim = obs_dim + action_dim + 32
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.GELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.velocity_net = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _velocity(
        self, obs: torch.Tensor, actions: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        return self.velocity_net(
            torch.cat((obs, actions, self.time_embedding(time)), dim=-1)
        )

    def _step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        time: torch.Tensor,
        noise_std: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        velocity = self._velocity(obs, actions, time)
        score = (time * velocity - actions) / (1.0 - time)
        drift = velocity + 0.5 * noise_std**2 * score
        mean = (actions + drift / self.denoising_steps).clamp(-1.0, 1.0)
        return velocity, drift, mean

    @staticmethod
    def _expand_obs(obs: torch.Tensor, num_samples: int) -> torch.Tensor:
        return (
            obs.unsqueeze(0)
            .expand(num_samples, *obs.shape)
            .reshape(num_samples * obs.shape[0], -1)
        )

    def sample_sde(
        self, obs: torch.Tensor, num_samples: int, noise_std: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        expanded_obs = self._expand_obs(obs, num_samples)
        actions = torch.randn(
            num_samples * batch_size,
            self.action_dim,
            device=obs.device,
            dtype=obs.dtype,
        )
        chains = [actions]
        log_prob = (
            Normal(torch.zeros_like(actions), torch.ones_like(actions))
            .log_prob(actions)
            .sum(dim=-1)
        )
        for step in range(self.denoising_steps):
            time = torch.full_like(actions[:, :1], step / self.denoising_steps)
            _, _, mean = self._step(expanded_obs, actions, time, noise_std)
            actions = mean + noise_std * torch.randn_like(actions).clamp(-3.0, 3.0)
            if step == self.denoising_steps - 1:
                actions = actions.clamp(-1.0, 1.0)
            log_prob += Normal(mean, noise_std).log_prob(actions).sum(dim=-1)
            chains.append(actions)

        log_prob /= (self.denoising_steps + 1) * self.action_dim
        return (
            actions.reshape(num_samples, batch_size, self.action_dim),
            torch.stack(chains, dim=1).reshape(
                num_samples,
                batch_size,
                self.denoising_steps + 1,
                self.action_dim,
            ),
            log_prob.reshape(num_samples, batch_size),
        )

    def sample_ode(self, obs: torch.Tensor) -> torch.Tensor:
        actions = torch.randn(
            obs.shape[0], self.action_dim, device=obs.device, dtype=obs.dtype
        )
        for step in range(self.denoising_steps):
            time = torch.full_like(actions[:, :1], step / self.denoising_steps)
            actions += self._velocity(obs, actions, time) / self.denoising_steps
        return actions.clamp(-1.0, 1.0)

    def log_prob(
        self, obs: torch.Tensor, chains: torch.Tensor, noise_std: float
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        num_samples, batch_size, chain_length, action_dim = chains.shape
        if chain_length != self.denoising_steps + 1 or action_dim != self.action_dim:
            raise ValueError("Chain shape does not match the OGPO flow actor")
        obs = self._expand_obs(obs, num_samples)
        chains = chains.reshape(num_samples * batch_size, chain_length, action_dim)
        initial = chains[:, 0]
        log_prob = (
            Normal(torch.zeros_like(initial), torch.ones_like(initial))
            .log_prob(initial)
            .sum(dim=-1)
        )
        velocity_means = []
        drift_means = []
        for step in range(self.denoising_steps):
            actions = chains[:, step]
            time = torch.full_like(actions[:, :1], step / self.denoising_steps)
            velocity, drift, mean = self._step(obs, actions, time, noise_std)
            log_prob += (
                Normal(mean, noise_std).log_prob(chains[:, step + 1]).sum(dim=-1)
            )
            velocity_means.append(velocity.mean())
            drift_means.append(drift.mean())
        log_prob /= (self.denoising_steps + 1) * self.action_dim
        return log_prob.reshape(num_samples, batch_size), {
            "velocity_mean": torch.stack(velocity_means).mean().detach(),
            "drift_mean": torch.stack(drift_means).mean().detach(),
        }

    def flow_matching(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = actions.clamp(-1.0, 1.0)
        noise = torch.randn_like(actions)
        time = torch.rand(
            actions.shape[0], 1, device=actions.device, dtype=actions.dtype
        )
        interpolated = (1.0 - time) * noise + time * actions
        target = actions - noise
        return self._velocity(obs, interpolated, time), target
