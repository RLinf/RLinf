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

"""Turning one device's reading into part of a robot's action."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .kinds import ActionKind

#: What a binding may ask about the robot it drives. A binding lists the ones
#: it cannot work without in :attr:`TeleopBinding.NEEDS`, and the group checks
#: them before calling it, so an env that cannot answer is refused by name
#: rather than raising a ``KeyError`` from inside somebody's arithmetic.
#:
#: tcp_pose        the arm's measured pose, xyz + quat
#: action_scale    the env's [position, rotation, gripper] divisors
#: joint_positions measured joint positions, one row per arm
#: gripper_open    whether the gripper is currently open
#: hand_reset_pose the pose a dexterous hand is put into at reset
CONTEXT_KEYS = (
    "tcp_pose",
    "action_scale",
    "joint_positions",
    "gripper_open",
    "hand_reset_pose",
)


@dataclass
class TeleopAction:
    """What one device is asking for, this step.

    Attributes:
        parts: The action parts this device fills, by name.
        driving: Whether the operator is actually driving. Devices report small
            residual motion constantly, so each binding decides its own
            threshold.
        info: Device state worth recording alongside the step it produced.
    """

    parts: dict[str, np.ndarray] = field(default_factory=dict)
    driving: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class TeleopBinding(ABC):
    """What one device means for the robot it drives.

    A device reports what the operator did. A binding says which parts of the
    action that fills, and computes them. Keeping the two apart is what lets the
    same spacemouse drive a Cartesian arm here and something else elsewhere.
    """

    #: Action parts this binding can fill, and what each command means.
    #: Matched against what the robot actually has, so a binding that offers a
    #: gripper to an arm carrying a hand simply does not fill it -- and one that
    #: offers a twist to a joint-space arm is refused rather than obeyed.
    #:
    #: A binding whose meaning depends on how it was built sets this on the
    #: instance instead, as a leader arm does for deltas versus targets.
    PRODUCES: Mapping[str, "ActionKind"] = {}

    #: Below this, motion is the device resting rather than the operator
    #: driving. Devices jitter; a person moving does not.
    MOVEMENT_EPSILON: float = 0.001

    #: How long the operator keeps control after their last active reading.
    #: ``None`` accepts the shared default; ``0`` suits a device held down to
    #: take over, where the button already says exactly when they are driving.
    HOLD_WINDOW: float | None = None

    #: Whether the parts this binding fills are clipped into the env's action
    #: space. Absolute commands can leave it; normalised deltas cannot.
    CLIPS_TO_ACTION_SPACE: bool = False

    #: Context keys this binding cannot work without, from :data:`CONTEXT_KEYS`.
    NEEDS: tuple[str, ...] = ()

    @abstractmethod
    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Say what this device asks for, given one reading.

        Everything about a reading is answered here at once. A binding whose
        answer depends on state it computed -- as one holding a pose does --
        would otherwise have to leave that state behind for a second call, and
        nothing would enforce the order the two are made in.
        """

    def publish(self, reading: Mapping[str, Any]) -> dict[str, Any]:
        """Context this device offers the bindings listed after it.

        Devices in one rig are not independent: on the dex-hand setup the
        spacemouse's left button is what puts the glove in control. Saying so
        here keeps that coupling visible and ordered, rather than hidden in a
        class that reads both devices.
        """
        return {}

    def hold(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """The parts that keep this device's share of the robot where it is.

        Asked for when a chunk of policy actions is skipped and something still
        has to be commanded. A binding that fills a part with a delta has
        nothing to say here: zero motion is already the policy's own action.
        """
        return {}

    def on_action_chunk_begin(self) -> None:
        """Let go of anything held only until the next chunk of actions."""

    def reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Forget anything held from the previous episode.

        The context says what the robot was just reset *to*. A binding holding
        a pose has to start from that pose rather than from zero, or its first
        command moves the robot away from where the env just put it.
        """
