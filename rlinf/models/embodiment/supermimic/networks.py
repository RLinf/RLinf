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

"""World model, motion policy and critic of SuperMimic.

Every network is a stack of ``Linear -> LayerNorm -> GELU`` blocks and a
final ``Linear``.
"""

from __future__ import annotations

import torch
from torch import nn

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    ACTOR_GOAL_DIM,
    SIM_CONTEXT_DIM,
    WORLD_MODEL_GOAL_DIM,
)
from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state, g1_action_scale
from rlinf.models.embodiment.modules.utils import make_mlp
from rlinf.models.embodiment.supermimic.normalization import EmpiricalNorm

LATENT_DIM = 32
POLICY_HIDDEN = (2048, 1024, 512, 256)
CRITIC_HIDDEN = (256, 128)
DYNAMICS_HIDDEN = (2048, 2048, 2048)


def _stack(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = make_mlp(
        in_dim,
        [*hidden, out_dim],
        act_builder=nn.GELU,
        last_act=False,
        use_layer_norm=True,
    )
    return nn.Sequential(*layers)


def hidden_matrix_names(module: nn.Module) -> list[str]:
    """Weights of every ``Linear`` between the first and last of each stack.

    These are the matrices Muon updates; Adam updates every other parameter.
    """
    names = []
    for seq_name, seq in module.named_modules():
        if not isinstance(seq, nn.Sequential):
            continue
        linear = [str(i) for i, layer in enumerate(seq) if isinstance(layer, nn.Linear)]
        names += [f"{seq_name}.{i}.weight" for i in linear[1:-1]]
    return names


def _run(owner: nn.Module, net: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """``net(features)``, autocast to ``owner.autocast_dtype`` when it is set."""
    dtype = getattr(owner, "autocast_dtype", None)
    if dtype is None:
        return net(features)
    # A recorded update must not keep cast weights from the graph memory pool.
    with torch.autocast(
        device_type=features.device.type, dtype=dtype, cache_enabled=False
    ):
        out = net(features)
    return out.to(dtype=features.dtype)


class Dynamics(nn.Module):
    """Velocity increments of the next control step.

    Reads the normalized encoding, the normalized PD target
    ``action * scale + default_pose`` and the sim context.
    """

    def __init__(self, encoding_dim: int, prediction_dim: int):
        super().__init__()
        self.state_norm = EmpiricalNorm(encoding_dim)
        self.action_norm = EmpiricalNorm(ACTION_DIM)
        self.register_buffer("action_scale", g1_action_scale(), persistent=False)
        self.register_buffer(
            "action_offset", asset_default_state()[7:36].clone(), persistent=False
        )
        self.mlp = _stack(
            encoding_dim + ACTION_DIM + SIM_CONTEXT_DIM, DYNAMICS_HIDDEN, prediction_dim
        )

    def joint_target(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scale + self.action_offset

    def forward(
        self, enc: torch.Tensor, action: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat(
            (
                self.state_norm(enc),
                self.action_norm(self.joint_target(action)),
                context,
            ),
            dim=-1,
        )
        return _run(self, self.mlp, features)


class MotionPolicy(nn.Module):
    """Goal encoder to a 32-d tanh latent, decoded with the normalized encoding."""

    def __init__(self, encoding_dim: int):
        super().__init__()
        self.obs_norm = EmpiricalNorm(encoding_dim)
        self.goal_norm = EmpiricalNorm(ACTOR_GOAL_DIM)
        self.encoder = _stack(ACTOR_GOAL_DIM, POLICY_HIDDEN, LATENT_DIM)
        self.decoder = _stack(LATENT_DIM + encoding_dim, POLICY_HIDDEN, ACTION_DIM)

    def forward(self, enc: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        latent = torch.tanh(_run(self, self.encoder, self.goal_norm(goal)))
        return _run(self, self.decoder, torch.cat((latent, self.obs_norm(enc)), dim=-1))


class Critic(nn.Module):
    """Value of the encoding, the world-model goal and the sim context."""

    def __init__(self, encoding_dim: int):
        super().__init__()
        self.mlp = _stack(
            encoding_dim + WORLD_MODEL_GOAL_DIM + SIM_CONTEXT_DIM, CRITIC_HIDDEN, 1
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return _run(self, self.mlp, features).squeeze(-1)
