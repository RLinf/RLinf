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

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """Residual ``[B, 70]`` of the next state from heading-free features."""
        feat = heading_invariant_features(state)
        return self.net(torch.cat((feat, action, xi), dim=-1))


class MotionEncoder(nn.Module):
    """Reference-motion encoder: the 680/689/730-d goal to a 32-d tanh latent."""

    def __init__(self, goal_dim: int):
        super().__init__()
        self.net = _mlp([goal_dim, 2048, 1024, 512, 256, LATENT_DIM])

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        """Latent ``[B, 32]`` of a goal ``[B, goal_dim]``."""
        return torch.tanh(self.net(goal))


class MotionDecoder(nn.Module):
    """Shared decoder: a goal latent and the observation history to an action."""

    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = _mlp(
            [LATENT_DIM + HISTORY_LEN * obs_dim, 2048, 1024, 512, 256, ACTION_DIM],
            last_gain=0.01,
        )

    def forward(self, latent: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Action ``[B, 29]`` from a latent and a flat 10-frame history."""
        return self.net(torch.cat((latent, history), dim=-1))


class CriticNet(nn.Module):
    """Motion critic V(s, g, xi) with the 4-d root error as an extra input."""

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
        """Value ``[B]`` of a state under a goal and sim context."""
        feat = heading_invariant_features(state)
        return self.net(torch.cat((feat, goal, xi, root_err), dim=-1)).squeeze(-1)


class TerminationHead(nn.Module):
    """Failure logit from height and tilt errors (Eq. 1 termination head).

    Args:
        bias_init: Output bias at construction; ``None`` keeps PyTorch's init.
    """

    def __init__(self, bias_init: float | None = None):
        super().__init__()
        self.net = _mlp([4, 64, 64, 1])
        if bias_init is not None:
            nn.init.constant_(self.net[-1].bias, float(bias_init))

    def forward(self, term_feat: torch.Tensor) -> torch.Tensor:
        """Failure logit ``[B]`` from ``termination_head_input`` features."""
        return self.net(term_feat).squeeze(-1)


class VelocityEncoder(nn.Module):
    """MLP(256, 256) on the normalized command and the 10-frame observation."""

    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = _mlp([3 + HISTORY_LEN * obs_dim, 256, 256, LATENT_DIM])

    def forward(self, command: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Latent ``[B, 32]`` of a normalized command and the history."""
        return torch.tanh(self.net(torch.cat((command, history), dim=-1)))


class VelocityCritic(nn.Module):
    """One critic for the velocity interface: heading state, command, and ξ."""

    def __init__(self):
        super().__init__()
        self.net = _mlp([STATE_DIM + 3 + XI_DIM, 256, 128, 1])

    def forward(
        self, state: torch.Tensor, command: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """Value ``[B]`` of a state under a velocity command."""
        feat = heading_invariant_features(state)
        return self.net(torch.cat((feat, command, xi), dim=-1)).squeeze(-1)


class SuperMimicModels(nn.Module):
    """Every SuperMimic network: dynamics, termination, policy, critics.

    Args:
        goal_dim: Width of the motion goal (680, 689 or 730).
        obs_dim: Width of one observation frame (93 or 99).
        velocity: Also build the velocity encoder and its critic.
        termination_bias_init: Initial termination-head bias; ``None`` keeps the default.
    """

    def __init__(
        self,
        goal_dim: int,
        obs_dim: int = OBS_DIM,
        velocity: bool = False,
        termination_bias_init: float | None = None,
    ):
        super().__init__()
        self.dynamics = DynamicsNet()
        self.encoder = MotionEncoder(goal_dim)
        self.decoder = MotionDecoder(obs_dim)
        self.critic = CriticNet(goal_dim)
        self.termination = TerminationHead(termination_bias_init)
        self.velocity_encoder = VelocityEncoder(obs_dim) if velocity else None
        self.velocity_critic = VelocityCritic() if velocity else None

    def policy(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Motion-tracking action from the history and the goal."""
        return self.decoder(self.encoder(goal), history)

    def velocity_policy(
        self, history: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        """Velocity-tracking action from the history and the command."""
        if self.velocity_encoder is None:
            raise RuntimeError(
                "velocity_policy needs SuperMimicModels(velocity=True); these "
                "models were built with velocity=False"
            )
        return self.decoder(self.velocity_encoder(command, history), history)

    def wm_params(self) -> list[nn.Parameter]:
        """Parameters fit by the world-model loss (dynamics and termination head)."""
        return list(self.dynamics.parameters()) + list(self.termination.parameters())

    def actor_params(self) -> list[nn.Parameter]:
        """Parameters of the motion policy (encoder and decoder)."""
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def velocity_actor_params(self) -> list[nn.Parameter]:
        """Velocity encoder parameters; empty without velocity tracking."""
        if self.velocity_encoder is None:
            return []
        return list(self.velocity_encoder.parameters())

    def critic_params(self) -> list[nn.Parameter]:
        """Parameters of the motion critic."""
        return list(self.critic.parameters())

    def velocity_critic_params(self) -> list[nn.Parameter]:
        """Velocity critic parameters; empty without velocity tracking."""
        if self.velocity_critic is None:
            return []
        return list(self.velocity_critic.parameters())
