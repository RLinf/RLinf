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
    """Convert 14D dual-arm ``xyz + quat`` TCP pose to euler angles.

    This wrapper is robot-agnostic: it only requires the observation to contain
    ``state.tcp_pose`` laid out as two ``xyz + quat`` pose vectors.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(12,)
        )

    def observation(self, observation: dict) -> dict:
        """Convert dual-arm quaternion TCP pose to euler angles in-place."""
        tcp_pose = observation["state"]["tcp_pose"]
        left = tcp_pose[:7]
        right = tcp_pose[7:]
        left_euler = np.concatenate(
            [left[:3], R.from_quat(left[3:].copy()).as_euler("xyz")]
        )
        right_euler = np.concatenate(
            [right[:3], R.from_quat(right[3:].copy()).as_euler("xyz")]
        )
        observation["state"]["tcp_pose"] = np.concatenate([left_euler, right_euler])
        return observation
