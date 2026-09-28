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

"""Typed state and backend contracts for dual-arm YAM environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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
class YamArmState:
    """One follower arm state in RLinf's normalized convention."""

    joint_positions: np.ndarray
    gripper_position: float
    timestamp_s: float

    def __post_init__(self) -> None:
        joints = _vector(self.joint_positions, NUM_ARM_JOINTS, "joint_positions")
        gripper = float(self.gripper_position)
        timestamp_s = float(self.timestamp_s)
        if not np.all(np.isfinite(joints)):
            raise ValueError("joint_positions must be finite")
        if not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
            raise ValueError("gripper_position must be finite and within [0, 1]")
        if not np.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        object.__setattr__(self, "joint_positions", joints)
        object.__setattr__(self, "gripper_position", gripper)
        object.__setattr__(self, "timestamp_s", timestamp_s)

    def as_action(self) -> np.ndarray:
        """Return ``[q0..q5, gripper]`` with ``0=closed, 1=open``."""
        return np.concatenate(
            [self.joint_positions, [float(self.gripper_position)]]
        ).astype(np.float64, copy=False)


@dataclass(frozen=True)
class DualYamState:
    """Coherent left/right follower state snapshot."""

    left: YamArmState
    right: YamArmState

    def as_vector(self) -> np.ndarray:
        """Return the fixed 14-D YAM state/action layout."""
        return pack_dual_action(self.left.as_action(), self.right.as_action())


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


@runtime_checkable
class YamFollowerBackend(Protocol):
    """Low-level follower API consumed by :class:`YamControlRuntime`."""

    def connect(self) -> None:
        """Open the arm transport and wait for fresh feedback."""

    def read_state(self) -> YamArmState:
        """Read the measured arm and normalized gripper state."""

    def command(self, target: np.ndarray) -> None:
        """Command one normalized 7-D joint target."""

    def joint_limits(self) -> np.ndarray:
        """Return SDK arm limits as a ``(6, 2)`` lower/upper array."""

    def hold(self) -> None:
        """Hold the current measured pose."""

    def assert_healthy(self, max_feedback_age_s: float) -> None:
        """Raise when transport or motor feedback is unhealthy."""

    def close(self) -> None:
        """Release transport resources. Must be idempotent."""


class YamBackendFactory(Protocol):
    """Injectable factory that keeps tests independent from i2rt."""

    def create_follower(self, device_config: Any) -> YamFollowerBackend:
        """Create, but do not connect, one follower backend."""
