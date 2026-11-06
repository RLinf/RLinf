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

"""Pluggable routing gates for RLT rollouts."""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import torch

RLT_GATE_INFO_KEYS = (
    "rlt_gate_entered",
    "rlt_gate_entry_step",
    "rlt_gate_score_ready",
    "rlt_gate_score_min",
    "rlt_gate_actor_active",
    "rlt_route_expert_entered",
    "rlt_route_expert_entry_step",
)


@dataclass
class RoutingDecision:
    """Actor/expert routing decisions emitted for one rollout step."""

    actor_switch: torch.Tensor
    expert_requested: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    phase_features: torch.Tensor | None = None


@runtime_checkable
class RoutingGate(Protocol):
    """Stateful component that decides actor and expert routing."""

    emit_phase_features: bool

    @property
    def controls_actor_routing(self) -> bool:
        """Return whether the gate owns the base-to-actor decision."""

    def step(
        self,
        env_obs: dict[str, Any],
        *,
        mode: Literal["train", "eval"],
        stage_id: int,
        reset_mask: torch.Tensor | None = None,
        external_actor_switch: torch.Tensor | None = None,
        actor_routing_enabled: bool = True,
        expert_routing_enabled: bool = True,
    ) -> RoutingDecision:
        """Advance the gate once and return routing decisions."""

    def reset(
        self,
        *,
        mode: Literal["train", "eval"] | None = None,
        stage_id: int | None = None,
    ) -> None:
        """Reset matching gate state."""

    def empty_diagnostics(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Return stack-compatible diagnostics without advancing state."""

    def empty_phase_features(self, batch_size: int) -> torch.Tensor:
        """Return stack-compatible phase features without advancing state."""

    def to(self, device: Any) -> "RoutingGate":
        """Move gate state and parameters to a device."""

    def eval(self) -> None:
        """Put gate models in evaluation mode."""


RoutingGateBuilder = Callable[..., RoutingGate | None]
_ROUTING_GATE_BUILDERS: dict[str, RoutingGateBuilder] = {}


def register_routing_gate(
    name: str,
) -> Callable[[RoutingGateBuilder], RoutingGateBuilder]:
    """Register a routing gate builder by config type."""

    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Routing gate type must not be empty.")

    def decorator(builder: RoutingGateBuilder) -> RoutingGateBuilder:
        if normalized_name in _ROUTING_GATE_BUILDERS:
            raise ValueError(
                f"Routing gate type {normalized_name!r} is registered twice."
            )
        _ROUTING_GATE_BUILDERS[normalized_name] = builder
        return builder

    return decorator


def _load_builtin_routing_gates() -> None:
    from rlinf.algorithms.rlt import (
        rlt_steam_critical_phase_gate as _steam_gate,  # noqa: F401
    )


def build_routing_gate(
    cfg: Any,
    *,
    device: str,
    num_action_chunks: int,
    env_decoupled_mode: bool,
) -> RoutingGate | None:
    """Build the configured routing gate."""

    if cfg is None or not bool(cfg.get("enable", False)):
        return None
    gate_type = str(cfg.get("type", "")).strip().lower()
    if not gate_type:
        raise ValueError(
            "rollout.routing_gate.type is required when the gate is enabled."
        )

    _load_builtin_routing_gates()
    try:
        builder = _ROUTING_GATE_BUILDERS[gate_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_ROUTING_GATE_BUILDERS)) or "none"
        raise ValueError(
            f"Unsupported routing gate type {gate_type!r}; supported types: {supported}."
        ) from exc
    return builder(
        cfg,
        device=device,
        num_action_chunks=num_action_chunks,
        env_decoupled_mode=env_decoupled_mode,
    )


__all__ = [
    "RLT_GATE_INFO_KEYS",
    "RoutingDecision",
    "RoutingGate",
    "build_routing_gate",
    "register_routing_gate",
]
