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

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from scipy.spatial.transform import Rotation as R


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles.

    The wrapped observation may be 7D ``xyz+quat`` or 8D
    ``xyz+quat+gripper``.  ``keep_gripper`` only affects 8D observations; 7D
    observations remain 6D after conversion for backward compatibility.
    """

    def __init__(self, env: Env, keep_gripper: bool = False):
        super().__init__(env)
        self.keep_gripper = bool(keep_gripper)
        raw_dim = self.observation_space["state"]["tcp_pose"].shape[0]
        if raw_dim not in (7, 8):
            raise ValueError(
                "Quat2EulerWrapper expects tcp_pose shape (7,) or (8,), "
                f"got {self.observation_space['state']['tcp_pose'].shape}."
            )
        output_dim = 7 if self.keep_gripper and raw_dim == 8 else 6
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(output_dim,)
        )

    def observation(self, observation):
        # Convert tcp pose from xyz+quat(+gripper) to xyz+rpy(+gripper).
        tcp_pose = observation["state"]["tcp_pose"]
        if tcp_pose.shape[0] not in (7, 8):
            raise ValueError(
                "Quat2EulerWrapper expects tcp_pose shape (7,) or (8,), "
                f"got {tcp_pose.shape}."
            )
        pose = [tcp_pose[:3], R.from_quat(tcp_pose[3:7].copy()).as_euler("xyz")]
        if self.keep_gripper and tcp_pose.shape[0] == 8:
            pose.append(tcp_pose[7:8])
        observation["state"]["tcp_pose"] = np.concatenate(pose)
        return observation
