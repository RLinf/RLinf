# Copyright 2025 The RLinf Authors.
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

"""Dual-arm robot state snapshot for DOSW1."""

import time
from dataclasses import dataclass, field

import numpy as np

NUM_JOINTS = 6


@dataclass
class DOSW1RobotState:
    """Snapshot of the dual-arm DOSW1 robot at a single timestep.

    Each arm has 6 revolute joints (radians) and 1 gripper value in metres.
    Control and state are joint-space only.
    """

    left_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_JOINTS)
    )
    left_gripper: float = 0.0
    right_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_JOINTS)
    )
    right_gripper: float = 0.0
    timestamp: float = field(default_factory=time.time)
