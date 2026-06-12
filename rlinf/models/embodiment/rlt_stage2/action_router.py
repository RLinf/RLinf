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

"""Action routing for RLT Stage 2 rollout."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rollout_schema import resolve_source_from_source_chunk
from .transition import (
    COLLECTION_PHASE_ONLINE,
    COLLECTION_PHASE_WARMUP,
    TransitionSource,
)


@dataclass(frozen=True)
class RLTActionRouteInputs:
    student_actions: torch.Tensor
    base_flat: torch.Tensor
    expert_actions: torch.Tensor | None
    expert_takeover: torch.Tensor
    requested_expert_takeover: torch.Tensor
    intervention_phase: torch.Tensor
    in_critical_phase: torch.Tensor
    record_transition: torch.Tensor
    ready_for_online: bool
    online_gate_step: int
    chunk_length: int
    action_dim: int


@dataclass(frozen=True)
class RLTActionRouteResult:
    actions: torch.Tensor
    action_flat: torch.Tensor
    base_flat: torch.Tensor
    student_control: torch.Tensor
    intervention_flags: torch.Tensor
    source_chunk: torch.Tensor
    source: torch.Tensor
    collection_phase_id: torch.Tensor
    intervention_requested: torch.Tensor
    intervention_phase: torch.Tensor
    in_critical_phase: torch.Tensor
    record_transition: torch.Tensor
    ready_for_online: torch.Tensor
    online_gate_step: torch.Tensor

    def to_forward_input_updates(self) -> dict[str, torch.Tensor]:
        return {
            "base_a_tilde": self.base_flat,
            "ref_chunk": self.base_flat.detach(),
            "action": self.action_flat.detach(),
            "action_chunk": self.action_flat.detach(),
            "student_control": self.student_control[:, None],
            "intervention_flags": self.intervention_flags,
            "source_chunk": self.source_chunk,
            "source": self.source,
            "collection_phase_id": self.collection_phase_id,
            "intervention_requested": self.intervention_requested[:, None],
            "intervention_phase": self.intervention_phase[:, None],
            "in_critical_phase": self.in_critical_phase[:, None],
            "record_transition": self.record_transition[:, None],
            "ready_for_online": self.ready_for_online,
            "online_gate_step": self.online_gate_step,
        }


def route_rlt_stage2_actions(inputs: RLTActionRouteInputs) -> RLTActionRouteResult:
    """Route execution between base VLA, RLT actor, and expert correction."""
    _validate_route_inputs(inputs)

    base_actions = inputs.base_flat.reshape(
        inputs.base_flat.shape[0],
        inputs.chunk_length,
        inputs.action_dim,
    )
    actor_control = (
        torch.full(
            (inputs.student_actions.shape[0],),
            bool(inputs.ready_for_online),
            dtype=torch.bool,
            device=inputs.student_actions.device,
        )
        & inputs.in_critical_phase
    )
    actions = torch.where(
        actor_control[:, None, None],
        inputs.student_actions,
        base_actions,
    )

    intervention_flags = torch.zeros(
        (actions.shape[0], inputs.chunk_length),
        dtype=torch.bool,
        device=actions.device,
    )
    source_chunk = torch.full(
        (actions.shape[0], inputs.chunk_length),
        int(TransitionSource.BASE),
        dtype=torch.uint8,
        device=actions.device,
    )
    source_chunk[actor_control] = int(TransitionSource.RL)

    if inputs.expert_actions is not None:
        expert_mask = inputs.expert_takeover[:, None, None].to(actions.device)
        actions = torch.where(expert_mask, inputs.expert_actions, actions)
        intervention_flags[inputs.expert_takeover] = True
        source_chunk[inputs.expert_takeover] = int(TransitionSource.HUMAN)

    action_flat = actions.reshape(actions.shape[0], -1)
    return RLTActionRouteResult(
        actions=actions,
        action_flat=action_flat,
        base_flat=inputs.base_flat,
        student_control=actor_control,
        intervention_flags=intervention_flags,
        source_chunk=source_chunk,
        source=resolve_source_from_source_chunk(source_chunk),
        collection_phase_id=torch.full(
            (actions.shape[0], 1),
            COLLECTION_PHASE_ONLINE
            if inputs.ready_for_online
            else COLLECTION_PHASE_WARMUP,
            dtype=torch.uint8,
            device=actions.device,
        ),
        intervention_requested=inputs.requested_expert_takeover.to(
            actions.device,
            dtype=torch.bool,
        ),
        intervention_phase=inputs.intervention_phase.to(
            actions.device,
            dtype=torch.float32,
        ),
        in_critical_phase=inputs.in_critical_phase.to(actions.device, dtype=torch.bool),
        record_transition=inputs.record_transition.to(actions.device, dtype=torch.bool),
        ready_for_online=torch.full(
            (actions.shape[0], 1),
            bool(inputs.ready_for_online),
            dtype=torch.bool,
            device=actions.device,
        ),
        online_gate_step=torch.full(
            (actions.shape[0], 1),
            float(inputs.online_gate_step),
            dtype=torch.float32,
            device=actions.device,
        ),
    )


def _validate_route_inputs(inputs: RLTActionRouteInputs) -> None:
    if inputs.student_actions.ndim != 3:
        raise ValueError(
            "RLTActionRouter student_actions must have shape [B, T, A], got "
            f"{_shape_str(inputs.student_actions)}."
        )
    batch_size = int(inputs.student_actions.shape[0])
    expected_action_shape = (batch_size, inputs.chunk_length, inputs.action_dim)
    if tuple(inputs.student_actions.shape) != expected_action_shape:
        raise ValueError(
            "RLTActionRouter student_actions shape mismatch: expected "
            f"{expected_action_shape}, got {_shape_str(inputs.student_actions)}."
        )
    expected_flat_shape = (batch_size, inputs.chunk_length * inputs.action_dim)
    if tuple(inputs.base_flat.shape) != expected_flat_shape:
        raise ValueError(
            "RLTActionRouter base_flat shape mismatch: expected "
            f"{expected_flat_shape}, got {_shape_str(inputs.base_flat)}."
        )
    if (
        inputs.expert_actions is not None
        and tuple(inputs.expert_actions.shape) != expected_action_shape
    ):
        raise ValueError(
            "RLTActionRouter expert_actions shape mismatch: expected "
            f"{expected_action_shape}, got {_shape_str(inputs.expert_actions)}."
        )
    for name in (
        "expert_takeover",
        "requested_expert_takeover",
        "intervention_phase",
        "in_critical_phase",
        "record_transition",
    ):
        value = getattr(inputs, name)
        if tuple(value.shape) != (batch_size,):
            raise ValueError(
                f"RLTActionRouter {name} must have shape [{batch_size}], got "
                f"{_shape_str(value)}."
            )


def _shape_str(value: torch.Tensor | None) -> str:
    return "None" if value is None else str(tuple(getattr(value, "shape", ())))
