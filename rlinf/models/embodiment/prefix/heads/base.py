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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ActorHeadCapabilities:
    """Behavior exposed by an actor-head implementation."""

    distribution: Literal["stochastic", "deterministic"]
    supports_action_noise: bool = False
    supports_reference_dropout: bool = True


@dataclass(frozen=True)
class CriticHeadCapabilities:
    """Behavior exposed by a critic-head implementation."""

    supports_cross_q: bool = False
    fixed_num_q_heads: int | None = None
    default_num_q_heads: int = 2

    def num_q_heads(self, cfg: Any) -> int:
        """Resolve the number of Q estimates returned by this head."""
        if self.fixed_num_q_heads is not None:
            return self.fixed_num_q_heads
        return int(cfg.get("num_q_heads", self.default_num_q_heads))


class PrefixActorHead(nn.Module, ABC):
    """Actor head operating on prefix state and a reference action chunk."""

    capabilities: ActorHeadCapabilities

    @abstractmethod
    def forward(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        *,
        deterministic: bool,
        apply_reference_dropout: bool,
        reference_dropout_prob: float | None,
        apply_action_noise: bool | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flat actions and per-dimension log probabilities."""

    @abstractmethod
    def mean_action(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return the noise-free action used by supervised objectives."""


class PrefixCriticHead(nn.Module, ABC):
    """Critic head operating on prefix state and a flat action chunk."""

    _fsdp_wrap_role = "prefix_critic_head"
    capabilities: CriticHeadCapabilities
    num_q_heads: int

    @abstractmethod
    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        next_state: torch.Tensor | None = None,
        next_actions: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return Q-values, optionally paired with next-state Q-values."""
