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

"""Shared Franka state representation and address validation."""

import ipaddress
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

# Franka Panda joint travel, in radians, and speed limits with the same
# 0.1 rad/s margin Polymetis uses.
JOINT_LIMITS_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
JOINT_VEL_LIMITS = np.array([2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51])


@dataclass
class FrankaRobotState:
    """State snapshot shared by the Franka backends.

    The built-in gripper uses ``gripper_position`` and ``gripper_open``.
    Dexterous hands use the six-dimensional ``hand_position`` field.
    """

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

    # Franka built-in gripper state.
    gripper_position: float = 0.0  # Sum(JointState.position)
    gripper_open: bool = False

    # Dexterous-hand state.
    hand_position: Optional[np.ndarray] = None  # Six values normalized to [0, 1].

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)


def validated_robot_ip(robot_ip: str, part_name: str) -> str:
    """Validate and return a nonempty IP address without accessing the network."""
    if not robot_ip:
        raise ValueError(
            f"{part_name} needs a 'robot_ip'; none was given and none could be "
            "resolved from the node's hardware infos."
        )
    try:
        ipaddress.ip_address(robot_ip)
    except ValueError:
        raise ValueError(
            f"{part_name} needs 'robot_ip' to be an IP address, but got "
            f"{robot_ip!r}. A placeholder left in a config reaches the arm as "
            "this."
        ) from None
    return robot_ip
