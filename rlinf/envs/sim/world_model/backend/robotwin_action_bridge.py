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

"""Aloha joint angles to the normalised end-effector poses a RoboTwin world model reads.

Forward kinematics uses the URDF RoboTwin drives, so the poses differ from the
simulator's only by the controller's tracking error.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

__all__ = ["RobotwinActionBridge"]

_ARM_JOINTS = (
    ("fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6"),
    ("fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6"),
)
_ARM_EE = ("fl_link6", "fr_link6")


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = q[1:]
    return v + 2 * np.cross(u, np.cross(u, v) + q[0] * v)


class RobotwinActionBridge:
    """Joint14 to ``[xyz, xyz-euler, gripper]`` per arm, left arm first, in ``[-1, 1]``.

    Args:
        urdf_path: The Aloha-AgileX URDF RoboTwin loads.
        srdf_path: Its SRDF.
        stat_path: The world model's ``stat.json``; ``state_pose.p01`` / ``p99`` map
            each pose dimension to ``[-1, 1]``.
        robot_pose: The robot base pose in the world, ``[x, y, z, qw, qx, qy, qz]``.
    """

    def __init__(
        self,
        urdf_path: str,
        srdf_path: str,
        stat_path: str,
        robot_pose: Sequence[float],
    ):
        import mplib

        planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group=_ARM_EE[1])
        self._pm = planner.pinocchio_model
        joint_names = list(planner.user_joint_names)
        link_names = list(self._pm.get_link_names())
        self._n_joints = len(joint_names)
        self._joint_idx = [[joint_names.index(j) for j in arm] for arm in _ARM_JOINTS]
        self._link_idx = [link_names.index(ee) for ee in _ARM_EE]

        pose = np.asarray(robot_pose, dtype=float)
        self._base_p, self._base_q = pose[:3], pose[3:7]

        with open(stat_path) as f:
            state_pose = json.load(f)["state_pose"]
        self._p01 = np.asarray(state_pose["p01"], dtype=float)
        p99 = np.asarray(state_pose["p99"], dtype=float)
        span = p99 - self._p01
        self._span = np.where(np.abs(span) < 1e-8, 1.0, span)

    def _arm_pose(self, arm: int, joints: np.ndarray, gripper: float) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        qpos = np.zeros(self._n_joints)
        qpos[self._joint_idx[arm]] = joints
        self._pm.compute_forward_kinematics(qpos)
        pose = self._pm.get_link_pose(self._link_idx[arm])
        p = self._base_p + _quat_rot(self._base_q, np.asarray(pose.p))
        q = _quat_mul(self._base_q, np.asarray(pose.q))
        euler = Rotation.from_quat(np.roll(q, -1)).as_euler("xyz")
        return np.concatenate([p, euler, [gripper]])

    def __call__(self, joint14: np.ndarray) -> np.ndarray:
        """Convert ``[..., 14]`` joint commands to ``[..., 14]`` poses, as float32."""
        flat = np.asarray(joint14, dtype=np.float64).reshape(-1, 14)
        out = np.empty_like(flat)
        for i, a in enumerate(flat):
            out[i] = np.concatenate(
                [self._arm_pose(0, a[0:6], a[6]), self._arm_pose(1, a[7:13], a[13])]
            )
        out = 2.0 * (out - self._p01) / self._span - 1.0
        return np.clip(out, -1.0, 1.0).reshape(np.shape(joint14)).astype(np.float32)
