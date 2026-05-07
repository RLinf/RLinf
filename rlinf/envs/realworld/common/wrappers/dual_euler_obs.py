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

"""Quaternion-to-Euler observation wrapper for dual-arm pose observations."""

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from scipy.spatial.transform import Rotation as R


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """Convert dual-arm ``xyz + quat`` TCP pose to euler angles.

    This wrapper is robot-agnostic: it only requires the observation to contain
    ``state.tcp_pose`` laid out as two ``xyz + quat`` pose vectors, optionally
    with one gripper channel after each arm pose.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        raw_dim = self.observation_space["state"]["tcp_pose"].shape[0]
        if raw_dim not in (14, 16):
            raise ValueError(
                "DualQuat2EulerWrapper expects tcp_pose shape (14,) or (16,), "
                f"got {self.observation_space['state']['tcp_pose'].shape}."
            )
        output_dim = 14 if raw_dim == 16 else 12
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(output_dim,)
        )

    @staticmethod
    def _convert_arm_pose(pose: np.ndarray) -> np.ndarray:
        arm_pose = [pose[:3], R.from_quat(pose[3:7].copy()).as_euler("xyz")]
        if pose.shape[0] == 8:
            arm_pose.append(pose[7:8])
        return np.concatenate(arm_pose)

    def observation(self, observation: dict) -> dict:
        """Convert dual-arm quaternion TCP pose to euler angles in-place."""
        tcp_pose = observation["state"]["tcp_pose"]
        if tcp_pose.shape[0] not in (14, 16):
            raise ValueError(
                "DualQuat2EulerWrapper expects tcp_pose shape (14,) or (16,), "
                f"got {tcp_pose.shape}."
            )
        arm_dim = tcp_pose.shape[0] // 2
        left = tcp_pose[:arm_dim]
        right = tcp_pose[arm_dim:]
        left_euler = self._convert_arm_pose(left)
        right_euler = self._convert_arm_pose(right)
        observation["state"]["tcp_pose"] = np.concatenate([left_euler, right_euler])
        return observation
