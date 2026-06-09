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

"""Observation key adapter for realworld RLT joint datasets."""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RLTJointObsWrapper(gym.ObservationWrapper):
    """Expose realworld state keys in the RLT dataset layout."""

    def __init__(self, env):
        super().__init__(env)
        state_space = copy.copy(self.observation_space["state"])
        state_space["gripper"] = state_space.get(
            "gripper_position",
            spaces.Box(-np.inf, np.inf, shape=(1,)),
        )
        state_space["joint_pos"] = state_space.get(
            "arm_joint_position",
            spaces.Box(-np.inf, np.inf, shape=(7,)),
        )
        state_space["joint_vel"] = state_space.get(
            "arm_joint_velocity",
            spaces.Box(-np.inf, np.inf, shape=(7,)),
        )
        self.observation_space["state"] = state_space

    def observation(self, observation):
        state = observation["state"]
        robot_state = self._robot_state()
        if "gripper_position" in state:
            state["gripper"] = state["gripper_position"]
        elif robot_state is not None and hasattr(robot_state, "gripper_position"):
            state["gripper"] = np.asarray([robot_state.gripper_position])
        else:
            state["gripper"] = np.zeros(1)

        if "arm_joint_position" in state:
            state["joint_pos"] = state["arm_joint_position"]
        elif robot_state is not None and hasattr(robot_state, "arm_joint_position"):
            state["joint_pos"] = np.asarray(robot_state.arm_joint_position)

        if "arm_joint_velocity" in state:
            state["joint_vel"] = state["arm_joint_velocity"]
        elif robot_state is not None and hasattr(robot_state, "arm_joint_velocity"):
            state["joint_vel"] = np.asarray(robot_state.arm_joint_velocity)
        return observation

    def _robot_state(self):
        base_env = getattr(self.env, "unwrapped", self.env)
        return getattr(base_env, "_franka_state", getattr(base_env, "_state", None))
