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

"""Observation wrappers that express TCP orientation as Euler angles."""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from scipy.spatial.transform import Rotation as R


class Quat2EulerWrapper(gym.ObservationWrapper):
    """Convert each arm's TCP pose from ``xyz + quat`` to ``xyz + euler``.

    Args:
        env: The environment to wrap.
        arms: How many 7-element poses ``tcp_pose`` carries. Defaults to
            :attr:`ARMS`, so a subclass can fix the count for its robot.
    """

    #: Arms this wrapper reads unless told otherwise.
    ARMS: int = 1

    #: Length of one pose after the conversion.
    EULER_DIM = 6

    #: Match the environment pose dtype; SciPy returns float64 by default.
    DTYPE = np.float32

    def __init__(self, env: Env, arms: Optional[int] = None) -> None:
        super().__init__(env)
        self.arms = self.ARMS if arms is None else arms
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(self.EULER_DIM * self.arms,), dtype=self.DTYPE
        )

    def observation(self, observation: dict) -> dict[str, Any]:
        """Rewrite ``tcp_pose`` in place, one arm at a time."""
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            [
                np.concatenate((pose[:3], R.from_quat(pose[3:].copy()).as_euler("xyz")))
                for pose in np.split(tcp_pose, self.arms)
            ]
        ).astype(self.DTYPE)
        return observation


class DualQuat2EulerWrapper(Quat2EulerWrapper):
    """Convert the concatenated TCP poses of a dual-arm robot."""

    ARMS = 2
