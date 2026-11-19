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

"""Actor and critic heads for :class:`PrefixPolicy`, with a name registry.

A head receives the flat policy state and either a reference action chunk
(actor) or the executed action chunk (critic). ``capabilities`` declares what
the head supports, so the off-policy algorithm can reject an incompatible
combination before training starts.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rlinf.models.embodiment.modules.q_head import MultiCrossQHead, MultiQHead
from rlinf.models.embodiment.modules.utils import get_act_func, layer_init, make_mlp


@dataclass(frozen=True)
class ActorHeadCapabilities:
    """Behavior exposed by an actor-head implementation."""

    distribution: Literal["stochastic", "deterministic"]
    supports_action_noise: bool = False


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


ActorBuilder = Callable[[int, int, int, Any], PrefixActorHead]
CriticBuilder = Callable[[int, int, Any], PrefixCriticHead]


@dataclass(frozen=True)
class ActorHeadSpec:
    """Registered actor constructor and its composition contract."""

    builder: ActorBuilder
    capabilities: ActorHeadCapabilities


@dataclass(frozen=True)
class CriticHeadSpec:
    """Registered critic constructor and its composition contract."""

    builder: CriticBuilder
    capabilities: CriticHeadCapabilities


_ACTOR_SPECS: dict[str, ActorHeadSpec] = {}
_CRITIC_SPECS: dict[str, CriticHeadSpec] = {}


def _normalize_name(name: str) -> str:
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Prefix policy component name must not be empty.")
    return normalized_name


def register_prefix_actor(
    name: str,
    *,
    capabilities: ActorHeadCapabilities,
) -> Callable[[ActorBuilder], ActorBuilder]:
    """Register an actor-head builder and its capabilities."""
    normalized_name = _normalize_name(name)

    def decorator(builder: ActorBuilder) -> ActorBuilder:
        if normalized_name in _ACTOR_SPECS:
            raise ValueError(
                f"Prefix actor head {normalized_name!r} is registered twice."
            )
        _ACTOR_SPECS[normalized_name] = ActorHeadSpec(builder, capabilities)
        return builder

    return decorator


def register_prefix_critic(
    name: str,
    *,
    capabilities: CriticHeadCapabilities,
) -> Callable[[CriticBuilder], CriticBuilder]:
    """Register a critic-head builder and its capabilities."""
    normalized_name = _normalize_name(name)

    def decorator(builder: CriticBuilder) -> CriticBuilder:
        if normalized_name in _CRITIC_SPECS:
            raise ValueError(
                f"Prefix critic head {normalized_name!r} is registered twice."
            )
        _CRITIC_SPECS[normalized_name] = CriticHeadSpec(builder, capabilities)
        return builder

    return decorator


def get_prefix_actor_spec(name: str) -> ActorHeadSpec:
    """Return the registered actor specification."""
    normalized_name = _normalize_name(name)
    try:
        return _ACTOR_SPECS[normalized_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_ACTOR_SPECS))
        raise ValueError(
            f"Unsupported prefix actor {normalized_name!r}; "
            f"supported actors: {supported}."
        ) from exc


def get_prefix_critic_spec(name: str) -> CriticHeadSpec:
    """Return the registered critic specification."""
    normalized_name = _normalize_name(name)
    try:
        return _CRITIC_SPECS[normalized_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CRITIC_SPECS))
        raise ValueError(
            f"Unsupported prefix critic {normalized_name!r}; "
            f"supported critics: {supported}."
        ) from exc


def build_prefix_actor(
    name: str,
    *,
    state_dim: int,
    reference_dim: int,
    action_dim: int,
    cfg: Any,
) -> PrefixActorHead:
    """Build a registered actor head."""
    return get_prefix_actor_spec(name).builder(
        state_dim,
        reference_dim,
        action_dim,
        cfg,
    )


def build_prefix_critic(
    name: str,
    *,
    state_dim: int,
    action_dim: int,
    cfg: Any,
) -> PrefixCriticHead:
    """Build a registered critic head."""
    return get_prefix_critic_spec(name).builder(state_dim, action_dim, cfg)


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
        return self.backbone(torch.cat([state, reference], dim=-1))

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
        self.mlp = nn.Sequential(*layers)
        self.noise_sigma = float(cfg.get("noise_sigma", 0.1))

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
            reference = _drop_reference(reference, float(reference_dropout_prob or 0.0))
        action = self.mlp(torch.cat([state, reference], dim=-1))
        add_noise = (
            not deterministic if apply_action_noise is None else apply_action_noise
        )
        if add_noise and self.noise_sigma > 0.0:
            action = action + torch.randn_like(action) * self.noise_sigma
        action = action.clamp(-1.0, 1.0)
        return action, torch.zeros_like(action)


class MultiQMLPCritic(PrefixCriticHead):
    """Multi-Q_MLP critic built from the shared ``MultiQHead``."""

    capabilities = CriticHeadCapabilities()
    # ``QHead`` weight init supports only these gains.
    supported_activations = ("relu", "tanh")

    def __init__(self, state_dim: int, action_dim: int, cfg: Any) -> None:
        super().__init__()
        activation = str(cfg.get("activation", "tanh")).lower()
        if activation not in self.supported_activations:
            raise ValueError(
                f"multi_q_mlp critic_head.activation must be one of "
                f"{self.supported_activations}, got {activation!r}."
            )
        self.num_q_heads = self.capabilities.num_q_heads(cfg)
        self.q_head = MultiQHead(
            hidden_size=int(state_dim),
            action_feature_dim=int(action_dim),
            hidden_dims=_hidden_dims(cfg, [256, 256, 256]),
            num_q_heads=self.num_q_heads,
            output_dim=1,
            activation=activation,
            use_layer_norm=bool(cfg.get("use_layer_norm", True)),
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
