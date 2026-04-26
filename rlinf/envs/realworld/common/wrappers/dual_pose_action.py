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

"""Dual-arm pose action-mode wrappers."""

import gymnasium as gym
import numpy as np


class DualAbsolutePoseActionWrapper(gym.Wrapper):
    """Route dual-arm absolute pose actions to an env absolute-pose step method.

    The wrapped env must expose ``step_absolute_pose(action)`` and
    ``get_absolute_pose_action_space()``. Actions are laid out per arm as
    ``[x, y, z, rx, ry, rz, gripper]``.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._step_absolute_pose = self.get_wrapper_attr("step_absolute_pose")
        self.action_space = self.get_wrapper_attr("get_absolute_pose_action_space")()

    def step(self, action: np.ndarray):
        return self._step_absolute_pose(action)


class DualRelativePoseActionWrapper(gym.Wrapper):
    """Route dual-arm relative pose actions to an env relative-pose step method."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._step_relative_pose = self.get_wrapper_attr("step_relative_pose")
        self.action_space = self.get_wrapper_attr("get_relative_pose_action_space")()

    def step(self, action: np.ndarray):
        return self._step_relative_pose(action)
