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

"""SiLU MLPs for SuperMimic. Not registered in EMBODIED_MODEL."""

from __future__ import annotations

import torch
from torch import nn

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    HISTORY_LEN,
    LATENT_DIM,
    OBS_DIM,
    RESIDUAL_DIM,
    STATE_DIM,
    XI_DIM,
)
from rlinf.envs.sim.g1_mimic.features import heading_invariant_features


def _mlp(sizes: list[int], last_gain: float | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        if i < len(sizes) - 2:
            layers.extend((linear, nn.SiLU()))
        else:
            if last_gain is not None:
                nn.init.xavier_uniform_(linear.weight, gain=last_gain)
                nn.init.zeros_(linear.bias)
            layers.append(linear)
    return nn.Sequential(*layers)


class DynamicsNet(nn.Module):
    """Paper Table A.4: MLP hidden widths 2048, 2048, 2048. Last layer is zero."""

    def __init__(self):
        super().__init__()
        self.net = _mlp(
            [STATE_DIM + ACTION_DIM + XI_DIM, 2048, 2048, 2048, RESIDUAL_DIM]
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor):
        feat = heading_invariant_features(state)
        return self.net(torch.cat((feat, action, xi), dim=-1))


class MotionEncoder(nn.Module):
    def __init__(self, goal_dim: int):
        super().__init__()
        self.net = _mlp([goal_dim, 2048, 1024, 512, 256, LATENT_DIM])

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(goal))


class MotionDecoder(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = _mlp(
            [LATENT_DIM + HISTORY_LEN * obs_dim, 2048, 1024, 512, 256, ACTION_DIM],
            last_gain=0.01,
        )

    def forward(self, latent: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((latent, history), dim=-1))


class CriticNet(nn.Module):
    def __init__(self, goal_dim: int):
        super().__init__()
        in_dim = STATE_DIM + goal_dim + XI_DIM + 4
        self.net = _mlp([in_dim, 256, 128, 1])

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        xi: torch.Tensor,
        root_err: torch.Tensor,
    ) -> torch.Tensor:
        feat = heading_invariant_features(state)
        return self.net(torch.cat((feat, goal, xi, root_err), dim=-1)).squeeze(-1)


class TerminationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _mlp([4, 64, 64, 1])

    def forward(self, term_feat: torch.Tensor) -> torch.Tensor:
        return self.net(term_feat).squeeze(-1)


class VelocityEncoder(nn.Module):
    """MLP(256, 256) on the normalized command and the 10-frame observation."""

    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = _mlp([3 + HISTORY_LEN * obs_dim, 256, 256, LATENT_DIM])

    def forward(self, command: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat((command, history), dim=-1)))


class VelocityCritic(nn.Module):
    """One critic for the velocity interface: heading state, command, and ξ."""

    def __init__(self):
        super().__init__()
        self.net = _mlp([STATE_DIM + 3 + XI_DIM, 256, 128, 1])

    def forward(
        self, state: torch.Tensor, command: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        feat = heading_invariant_features(state)
        return self.net(torch.cat((feat, command, xi), dim=-1)).squeeze(-1)


class SuperMimicModels(nn.Module):
    def __init__(self, goal_dim: int, obs_dim: int = OBS_DIM, velocity: bool = False):
        super().__init__()
        self.dynamics = DynamicsNet()
        self.encoder = MotionEncoder(goal_dim)
        self.decoder = MotionDecoder(obs_dim)
        self.critic = CriticNet(goal_dim)
        self.termination = TerminationHead()
        self.velocity_encoder = VelocityEncoder(obs_dim) if velocity else None
        self.velocity_critic = VelocityCritic() if velocity else None

    def policy(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(goal), history)

    def velocity_policy(
        self, history: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        if self.velocity_encoder is None:
            raise RuntimeError("velocity encoder is not enabled")
        return self.decoder(self.velocity_encoder(command, history), history)

    def wm_params(self):
        return list(self.dynamics.parameters()) + list(self.termination.parameters())

    def actor_params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def velocity_actor_params(self):
        if self.velocity_encoder is None:
            return []
        return list(self.velocity_encoder.parameters())

    def critic_params(self):
        return list(self.critic.parameters())

    def velocity_critic_params(self):
        if self.velocity_critic is None:
            return []
        return list(self.velocity_critic.parameters())
