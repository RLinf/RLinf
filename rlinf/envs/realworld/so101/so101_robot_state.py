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

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class SO101RobotState:
    """State snapshot for the SO101 6-DOF robot arm.

    The SO101 arm has 6 Feetech STS3215 motors:
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.

    Joint positions are in **degrees** (LeRobot convention).
    """

    joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint positions ``[q1, ..., q6]`` in degrees.
    Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll."""

    joint_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint velocities ``[dq1, ..., dq6]`` in deg/s."""

    gripper_position: float = 0.0
    """Gripper position in degrees (0 = closed, ~90 = open for STS3215)."""

    gripper_open: bool = False
    """``True`` when the gripper is open."""

    is_connected: bool = False
    """``True`` when the motor bus is actively connected."""

    def to_dict(self):
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)
