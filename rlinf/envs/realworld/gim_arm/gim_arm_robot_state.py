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
class GimArmRobotState:
    """State snapshot for the GimArm 6-DOF robot.

    All Cartesian quantities are expressed in the robot base frame.
    """

    tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    """End-effector pose ``[x, y, z, qx, qy, qz, qw]`` (m / quaternion).
    Computed via Pinocchio FK from joint positions."""

    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """End-effector Cartesian velocity ``[vx, vy, vz, wx, wy, wz]`` (m/s, rad/s).
    Computed as ``J @ dq``."""

    arm_joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint positions ``[q1, ..., q6]`` in radians."""

    arm_joint_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint velocities ``[dq1, ..., dq6]`` in rad/s."""

    tcp_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Estimated Cartesian force at EEF ``[fx, fy, fz]`` in N.
    Mapped from momentum-observer external torque via ``J^{-T}``.
    Zero when momentum observer is not active."""

    tcp_torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Estimated Cartesian torque at EEF ``[tx, ty, tz]`` in N-m.
    Mapped from momentum-observer external torque via ``J^{-T}``.
    Zero when momentum observer is not active."""

    arm_jacobian: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    """Body Jacobian ``(6, 6)`` in LOCAL_WORLD_ALIGNED frame.
    Computed via Pinocchio at current joint positions."""

    gripper_position: float = 0.0
    """Gripper joint position in radians (hardware units)."""

    gripper_open: bool = False
    """``True`` when the gripper position is closer to open than closed."""

    def to_dict(self):
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)
