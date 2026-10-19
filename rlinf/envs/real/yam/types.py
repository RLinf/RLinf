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

"""The 14-D action layout of a dual YAM env and its command result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NUM_ARM_JOINTS = 6
PER_ARM_ACTION_DIM = NUM_ARM_JOINTS + 1
DUAL_ARM_ACTION_DIM = 2 * PER_ARM_ACTION_DIM


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    return vector.copy()


@dataclass(frozen=True)
class YamCommandResult:
    """Result of validating and dispatching one dual-arm command."""

    requested: np.ndarray
    accepted: np.ndarray
    clipped: bool = False
    rejection_reason: str | None = None


def split_dual_action(action: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split the canonical 14-D vector into left/right 7-D vectors."""
    vector = _vector(action, DUAL_ARM_ACTION_DIM, "action")
    return (
        vector[:PER_ARM_ACTION_DIM].copy(),
        vector[PER_ARM_ACTION_DIM:].copy(),
    )


def pack_dual_action(left: Any, right: Any) -> np.ndarray:
    """Pack two ``[q0..q5, gripper]`` vectors into the canonical 14-D layout."""
    left_vector = _vector(left, PER_ARM_ACTION_DIM, "left action")
    right_vector = _vector(right, PER_ARM_ACTION_DIM, "right action")
    return np.concatenate([left_vector, right_vector]).astype(np.float64, copy=False)
