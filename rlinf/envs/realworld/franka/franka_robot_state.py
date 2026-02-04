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

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class FrankaRobotState:
    # https://docs.ros.org/en/kinetic/api/libfranka/html/structfranka_1_1RobotState.html
    tcp_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )  # FrankaState.O_T_EE
    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(6))
    arm_joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )  # FrankaState.q
    arm_joint_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )  # FrankaState.dq
    tcp_force: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # FrankaState.K_F_ext_hat_K[0:3]
    tcp_torque: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # FrankaState.K_F_ext_hat_K[3:6]
    arm_jacobian: np.ndarray = field(
        default_factory=lambda: np.zeros((6, 7))
    )  # ZeroJacobian.zero_jacobian

    gripper_position: int = 0  # Sum(JointState.position)
    gripper_open: bool = False

    def to_dict(self):
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)
