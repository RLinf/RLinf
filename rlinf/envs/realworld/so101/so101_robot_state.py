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
    """State snapshot for the SO101 6-DOF arm.

    The arm has 5 revolute joints (``shoulder_pan``, ``shoulder_lift``,
    ``elbow_flex``, ``wrist_flex``, ``wrist_roll``) plus a 1-DOF gripper.
    Joint values are in degrees (LeRobot convention when ``use_degrees=True``).
    """

    joint_position: np.ndarray = field(default_factory=lambda: np.zeros(5))
    """Arm joint positions in degrees, shape ``(5,)``."""

    joint_velocity: np.ndarray = field(default_factory=lambda: np.zeros(5))
    """Arm joint velocities in deg/s, shape ``(5,)``."""

    gripper_position: float = 0.0
    """Gripper opening in degrees (0 = closed, ~90 = open for STS3215)."""

    gripper_open: bool = False
    """``True`` when ``gripper_position`` exceeds ``binary_gripper_threshold``."""

    is_connected: bool = False

    def to_dict(self):
        return asdict(self)
