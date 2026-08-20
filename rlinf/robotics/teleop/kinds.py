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

"""What a command for one part of a robot means.

Two arms can both take six numbers and mean entirely different things by them:
a Cartesian twist to add to where the arm is now, or six joint angles to move
to. Nothing in the width tells them apart, so the width is not what decides.

An env says what each part of its action means, a binding says what it produces,
and a device that produces one kind cannot drive a part that expects another.
Without this a spacemouse's twist lands in a joint-position command and the arm
moves somewhere nobody asked for, with every shape check passing.
"""

from __future__ import annotations

import enum


class ActionKind(enum.Enum):
    """The meaning of the numbers in one part of an action."""

    CARTESIAN_DELTA = "cartesian_delta"
    """Change in end-effector pose: ``[dx, dy, dz, drx, dry, drz]``."""

    CARTESIAN_POSE = "cartesian_pose"
    """Absolute end-effector pose as position plus rot6d rotation."""

    JOINT_POSITION = "joint_position"
    """Absolute joint angles, in radians."""

    JOINT_DELTA = "joint_delta"
    """Change in joint angles, in radians."""

    GRIPPER = "gripper"
    """One channel opening or closing a two-fingered gripper."""

    HAND = "hand"
    """Finger positions of a dexterous hand."""

    BASE_VELOCITY = "base_velocity"
    """Velocity command for a mobile base."""


class ActionPart:
    """One named span of an env's action vector, and what it means.

    Args:
        name: What the part drives, e.g. ``"arm"`` or ``"end_effector"``.
        width: How many numbers it occupies.
        kind: What those numbers mean.
    """

    __slots__ = ("name", "width", "kind")

    def __init__(self, name: str, width: int, kind: ActionKind) -> None:
        if width <= 0:
            raise ValueError(f"Action part {name!r} must occupy at least one number.")
        self.name = name
        self.width = int(width)
        self.kind = kind

    def __repr__(self) -> str:
        return f"ActionPart({self.name!r}, {self.width}, {self.kind.name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionPart):
            return NotImplemented
        return (self.name, self.width, self.kind) == (
            other.name,
            other.width,
            other.kind,
        )
