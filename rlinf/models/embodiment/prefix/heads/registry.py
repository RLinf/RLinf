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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rlinf.models.embodiment.prefix.heads.base import (
    ActorHeadCapabilities,
    CriticHeadCapabilities,
    PrefixActorHead,
    PrefixCriticHead,
)

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


def _load_builtin_heads() -> None:
    from rlinf.models.embodiment.prefix.heads import mlp as _mlp  # noqa: F401


def get_prefix_actor_spec(name: str) -> ActorHeadSpec:
    """Return the registered actor specification."""
    _load_builtin_heads()
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
    _load_builtin_heads()
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
