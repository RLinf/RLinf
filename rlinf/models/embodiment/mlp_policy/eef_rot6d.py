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

"""Prefix heads with canonical EEF rotations and normalized residual bounds."""

from typing import Any

import torch

from rlinf.models.embodiment.mlp_policy.heads import (
    MultiQMLPCritic,
    register_prefix_actor,
    register_prefix_critic,
)
from rlinf.models.embodiment.mlp_policy.residual import ResidualMLPActor
from rlinf.utils.eef_rot6d import EEFNormalizer


class EEFResidualMLPActor(ResidualMLPActor):
    """Canonicalize editor inputs while applying edits to the normalized reference."""

    def __init__(
        self, state_dim: int, reference_dim: int, action_dim: int, cfg: Any
    ) -> None:
        if state_dim < 10 or action_dim % 10:
            raise ValueError("Expected a 10-D EEF state and chunks of 10-D actions.")
        super().__init__(state_dim, reference_dim, action_dim, cfg)
        self.normalizer = EEFNormalizer(cfg.norm_stats_path)

    def editor_inputs(
        self, state: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        state_input = torch.cat(
            (
                state[..., :-10],
                self.normalizer.network_input(state[..., -10:], "state"),
            ),
            -1,
        )
        action_input = self.normalizer.network_input(
            reference.reshape(reference.shape[0], -1, 10), "actions"
        ).flatten(1)
        return torch.cat((state_input.detach(), action_input), -1)


class EEFMultiQMLPCritic(MultiQMLPCritic):
    """Score physical rotations independently of redundant rot6d coordinates."""

    def __init__(self, state_dim: int, action_dim: int, cfg: Any) -> None:
        if state_dim < 10 or action_dim % 10:
            raise ValueError("Expected a 10-D EEF state and chunks of 10-D actions.")
        super().__init__(state_dim, action_dim, cfg)
        self.normalizer = EEFNormalizer(cfg.norm_stats_path)

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        next_state=None,
        next_actions=None,
    ) -> torch.Tensor:
        state_input = torch.cat(
            (
                state[..., :-10],
                self.normalizer.network_input(state[..., -10:], "state"),
            ),
            -1,
        )
        action_input = self.normalizer.network_input(
            actions.reshape(actions.shape[0], -1, 10), "actions"
        ).flatten(1)
        return super().forward(state_input, action_input)


@register_prefix_actor(
    "eef_rot6d_residual_mlp", capabilities=EEFResidualMLPActor.capabilities
)
def build_eef_actor(
    state_dim: int, reference_dim: int, action_dim: int, cfg: Any
) -> EEFResidualMLPActor:
    """Build an EEF editor using the checkpoint's normalization statistics."""
    return EEFResidualMLPActor(state_dim, reference_dim, action_dim, cfg)


@register_prefix_critic(
    "eef_rot6d_multi_q_mlp", capabilities=EEFMultiQMLPCritic.capabilities
)
def build_eef_critic(state_dim: int, action_dim: int, cfg: Any) -> EEFMultiQMLPCritic:
    """Build EEF ensemble critics using the checkpoint's normalization statistics."""
    return EEFMultiQMLPCritic(state_dim, action_dim, cfg)
