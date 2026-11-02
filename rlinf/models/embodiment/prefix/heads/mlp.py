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

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rlinf.models.embodiment.modules.q_head import MultiCrossQHead, MultiQHead
from rlinf.models.embodiment.modules.utils import get_act_func, layer_init, make_mlp
from rlinf.models.embodiment.prefix.heads.base import (
    ActorHeadCapabilities,
    CriticHeadCapabilities,
    PrefixActorHead,
    PrefixCriticHead,
)
from rlinf.models.embodiment.prefix.heads.registry import (
    register_prefix_actor,
    register_prefix_critic,
)


def _hidden_dims(cfg: Any, default: list[int]) -> list[int]:
    return [int(dim) for dim in cfg.get("hidden_dims", default)]


def _drop_reference(reference: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0.0:
        return reference
    keep_mask = (
        torch.rand((reference.shape[0], 1), device=reference.device) >= probability
    )
    return reference * keep_mask.to(dtype=reference.dtype)


class FixedStdMLPActor(PrefixActorHead):
    """Fixed-standard-deviation Gaussian MLP actor."""

    capabilities = ActorHeadCapabilities(distribution="stochastic")

    def __init__(
        self,
        state_dim: int,
        reference_dim: int,
        action_dim: int,
        cfg: Any,
    ) -> None:
        super().__init__()
        hidden_dims = _hidden_dims(cfg, [256, 256, 256])
        activation = str(cfg.get("activation", "tanh"))
        act = get_act_func(activation)
        layers: list[nn.Module] = []
        input_dim = int(state_dim) + int(reference_dim)
        for hidden_dim in hidden_dims:
            layers.extend([layer_init(nn.Linear(input_dim, hidden_dim)), act()])
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.actor_mean = layer_init(
            nn.Linear(input_dim, int(action_dim)),
            std=0.01 * np.sqrt(2),
        )
        self.fixed_std = float(cfg.get("fixed_std", 0.002))
        if self.fixed_std <= 0.0:
            raise ValueError(f"fixed_std must be positive, got {self.fixed_std}.")

    def _features(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(torch.cat([reference, state], dim=-1))

    def mean_action(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor_mean(self._features(state, reference))

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
        del apply_action_noise
        if apply_reference_dropout:
            reference = _drop_reference(
                reference,
                float(reference_dropout_prob or 0.0),
            )
        action_mean = self.mean_action(state, reference)
        action_std = torch.full_like(action_mean, self.fixed_std)
        distribution = Normal(action_mean, action_std)
        raw_action = action_mean if deterministic else distribution.rsample()
        return torch.tanh(raw_action), distribution.log_prob(raw_action)


class DeterministicMLPActor(PrefixActorHead):
    """Deterministic MLP actor with optional Gaussian exploration noise."""

    capabilities = ActorHeadCapabilities(
        distribution="deterministic",
        supports_action_noise=True,
    )

    def __init__(
        self,
        state_dim: int,
        reference_dim: int,
        action_dim: int,
        cfg: Any,
    ) -> None:
        super().__init__()
        hidden_dims = _hidden_dims(cfg, [256, 256])
        activation = get_act_func(str(cfg.get("activation", "relu")))
        layers = make_mlp(
            in_channels=int(state_dim) + int(reference_dim),
            mlp_channels=[*hidden_dims, int(action_dim)],
            act_builder=activation,
            last_act=False,
        )
        self.mlp = nn.Sequential(OrderedDict([("net", nn.Sequential(*layers))]))
        self.noise_sigma = float(cfg.get("noise_sigma", 0.1))
        self.reference_dropout = float(cfg.get("reference_dropout", 0.0))

    def mean_action(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp(torch.cat([state, reference], dim=-1)).clamp(-1.0, 1.0)

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
        if apply_reference_dropout:
            reference = _drop_reference(
                reference,
                self.reference_dropout
                if reference_dropout_prob is None
                else float(reference_dropout_prob),
            )
        action = self.mean_action(state, reference)
        add_noise = (
            not deterministic if apply_action_noise is None else apply_action_noise
        )
        if add_noise and self.noise_sigma > 0.0:
            action = action + torch.randn_like(action) * self.noise_sigma
        action = action.clamp(-1.0, 1.0)
        return action, torch.zeros_like(action)


class MultiQMLPCritic(PrefixCriticHead):
    """Independently parameterized multi-Q MLP critic."""

    capabilities = CriticHeadCapabilities()

    def __init__(self, state_dim: int, action_dim: int, cfg: Any) -> None:
        super().__init__()
        self.num_q_heads = self.capabilities.num_q_heads(cfg)
        self.q_head = MultiQHead(
            hidden_size=int(state_dim),
            action_feature_dim=int(action_dim),
            hidden_dims=_hidden_dims(cfg, [256, 256, 256]),
            num_q_heads=self.num_q_heads,
            output_dim=1,
        )

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        next_state: torch.Tensor | None = None,
        next_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del next_state, next_actions
        return self.q_head(state, actions)


class CrossQMLPCritic(PrefixCriticHead):
    """Batch-renormalized CrossQ MLP critic."""

    capabilities = CriticHeadCapabilities(supports_cross_q=True)

    def __init__(self, state_dim: int, action_dim: int, cfg: Any) -> None:
        super().__init__()
        self.num_q_heads = self.capabilities.num_q_heads(cfg)
        self.q_head = MultiCrossQHead(
            hidden_size=int(state_dim),
            action_feature_dim=int(action_dim),
            hidden_dims=_hidden_dims(cfg, [256, 256, 256]),
            num_q_heads=self.num_q_heads,
            output_dim=1,
        )

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        next_state: torch.Tensor | None = None,
        next_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_q, next_q = self.q_head(
            state,
            actions,
            next_state_features=next_state,
            next_action_features=next_actions,
        )
        if next_q is None:
            next_q = current_q.new_zeros(current_q.shape)
        return current_q, next_q


class QNetwork(nn.Module):
    """Single Q MLP used by the explicit twin-Q critic."""

    def __init__(self, state_dim: int, action_dim: int, cfg: Any) -> None:
        super().__init__()
        hidden_dims = _hidden_dims(cfg, [256, 256])
        activation = get_act_func(str(cfg.get("activation", "relu")))
        layers = make_mlp(
            in_channels=int(state_dim) + int(action_dim),
            mlp_channels=[*hidden_dims, 1],
            act_builder=activation,
            last_act=False,
        )
        self.mlp = nn.Sequential(OrderedDict([("net", nn.Sequential(*layers))]))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([state, action], dim=-1))


class TwinQMLPCritic(PrefixCriticHead):
    """Explicit pair of independent Q MLPs."""

    capabilities = CriticHeadCapabilities(fixed_num_q_heads=2)

    def __init__(self, state_dim: int, action_dim: int, cfg: Any) -> None:
        super().__init__()
        self.num_q_heads = self.capabilities.num_q_heads(cfg)
        self.q1 = QNetwork(state_dim, action_dim, cfg)
        self.q2 = QNetwork(state_dim, action_dim, cfg)

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        next_state: torch.Tensor | None = None,
        next_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del next_state, next_actions
        return torch.cat([self.q1(state, actions), self.q2(state, actions)], dim=-1)


@register_prefix_actor(
    "fixed_std_mlp",
    capabilities=FixedStdMLPActor.capabilities,
)
def build_fixed_std_mlp_actor(
    state_dim: int,
    reference_dim: int,
    action_dim: int,
    cfg: Any,
) -> PrefixActorHead:
    return FixedStdMLPActor(state_dim, reference_dim, action_dim, cfg)


@register_prefix_actor(
    "deterministic_mlp",
    capabilities=DeterministicMLPActor.capabilities,
)
def build_deterministic_mlp_actor(
    state_dim: int,
    reference_dim: int,
    action_dim: int,
    cfg: Any,
) -> PrefixActorHead:
    return DeterministicMLPActor(state_dim, reference_dim, action_dim, cfg)


@register_prefix_critic(
    "multi_q_mlp",
    capabilities=MultiQMLPCritic.capabilities,
)
def build_multi_q_mlp_critic(
    state_dim: int,
    action_dim: int,
    cfg: Any,
) -> PrefixCriticHead:
    return MultiQMLPCritic(state_dim, action_dim, cfg)


@register_prefix_critic(
    "cross_q_mlp",
    capabilities=CrossQMLPCritic.capabilities,
)
def build_cross_q_mlp_critic(
    state_dim: int,
    action_dim: int,
    cfg: Any,
) -> PrefixCriticHead:
    return CrossQMLPCritic(state_dim, action_dim, cfg)


@register_prefix_critic(
    "twin_q_mlp",
    capabilities=TwinQMLPCritic.capabilities,
)
def build_twin_q_mlp_critic(
    state_dim: int,
    action_dim: int,
    cfg: Any,
) -> PrefixCriticHead:
    return TwinQMLPCritic(state_dim, action_dim, cfg)
