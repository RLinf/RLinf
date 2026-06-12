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

"""Rollout-side data contract for RLT Stage 2."""

from __future__ import annotations

from typing import Any

import torch

from .transition import TransitionSource


REQUIRED_RLT_STAGE2_FORWARD_INPUTS = (
    "x",
    "a_tilde",
    "base_a_tilde",
    "ref_chunk",
    "action",
    "action_chunk",
    "student_control",
    "intervention_flags",
    "source_chunk",
    "source",
    "collection_phase_id",
    "intervention_requested",
    "intervention_phase",
    "in_critical_phase",
    "record_transition",
    "ready_for_online",
    "online_gate_step",
)


def resolve_source_from_source_chunk(source_chunk: torch.Tensor) -> torch.Tensor:
    """Collapse per-step source labels to one chunk-level source label."""
    if not isinstance(source_chunk, torch.Tensor) or source_chunk.ndim != 2:
        raise ValueError(
            "RLT Stage2 source_chunk must have shape [B, T], got "
            f"{_shape_str(source_chunk)}."
        )
    return torch.where(
        source_chunk.eq(source_chunk[:, :1]).all(dim=1, keepdim=True),
        source_chunk[:, :1],
        torch.full(
            (source_chunk.shape[0], 1),
            int(TransitionSource.MIXED),
            dtype=torch.uint8,
            device=source_chunk.device,
        ),
    )


def require_rlt_stage2_forward_inputs(
    forward_inputs: dict[str, Any],
    *,
    batch_size: int,
    chunk_length: int,
    action_dim: int,
    context: str,
) -> dict[str, Any]:
    """Validate that rollout emitted the canonical RLT Stage 2 fields."""
    missing = [
        key for key in REQUIRED_RLT_STAGE2_FORWARD_INPUTS if key not in forward_inputs
    ]
    if missing:
        raise RuntimeError(
            f"RLT Stage2 {context} forward_inputs missing required keys: {missing}. "
            "Build them in the rollout RLT path instead of relying on silent "
            "fallback defaults."
        )

    action_chunk_dim = int(chunk_length) * int(action_dim)
    expected_shapes = {
        "x": (batch_size, None),
        "a_tilde": (batch_size, action_chunk_dim),
        "base_a_tilde": (batch_size, action_chunk_dim),
        "ref_chunk": (batch_size, action_chunk_dim),
        "action": (batch_size, action_chunk_dim),
        "action_chunk": (batch_size, action_chunk_dim),
        "student_control": (batch_size, 1),
        "intervention_flags": (batch_size, chunk_length),
        "source_chunk": (batch_size, chunk_length),
        "source": (batch_size, 1),
        "collection_phase_id": (batch_size, 1),
        "intervention_requested": (batch_size, 1),
        "intervention_phase": (batch_size, 1),
        "in_critical_phase": (batch_size, 1),
        "record_transition": (batch_size, 1),
        "ready_for_online": (batch_size, 1),
        "online_gate_step": (batch_size, 1),
    }
    for key, expected_shape in expected_shapes.items():
        _require_tensor_shape(
            forward_inputs[key],
            expected_shape,
            field_name=key,
            context=context,
        )
    return forward_inputs


def _require_tensor_shape(
    value: Any,
    expected_shape: tuple[int | None, ...],
    *,
    field_name: str,
    context: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"RLT Stage2 {context} forward_inputs[{field_name!r}] must be a "
            f"torch.Tensor, got {type(value).__name__}."
        )
    if value.ndim != len(expected_shape):
        raise ValueError(
            f"RLT Stage2 {context} forward_inputs[{field_name!r}] must have "
            f"{len(expected_shape)} dims, got shape {_shape_str(value)}."
        )
    for dim, expected in enumerate(expected_shape):
        if expected is not None and int(value.shape[dim]) != int(expected):
            raise ValueError(
                f"RLT Stage2 {context} forward_inputs[{field_name!r}] shape "
                f"mismatch: expected {expected_shape}, got {_shape_str(value)}."
            )


def _shape_str(value: Any) -> str:
    return "None" if value is None else str(tuple(getattr(value, "shape", ())))
