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

import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import SpaceMouseExpert


def sample_gripper_action(is_open: bool) -> np.ndarray:
    if is_open:
        return np.random.uniform(0.9, 1.0, size=(1,))
    return np.random.uniform(-1.0, -0.9, size=(1,))


class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, gripper_enabled: bool = True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled
        self.expert = SpaceMouseExpert()
        self.last_intervene = 0
        self.left, self.right = False, False
        self.gripper_action = None
        if self.gripper_enabled:
            # init self.gripper_action
            state = self.get_wrapper_attr("_franka_state")
            is_open = bool(getattr(state, "gripper_open", True))
            self.gripper_action = sample_gripper_action(is_open=is_open)

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)

        if np.linalg.norm(expert_a) > 0.001 or (self.left + self.right) > 0.5:
            self.last_intervene = time.time()
        if self.gripper_enabled:
            if self.left:  # close gripper
                self.gripper_action = sample_gripper_action(is_open=False)
                self.last_intervene = time.time()
            elif self.right:  # open gripper
                self.gripper_action = sample_gripper_action(is_open=True)
                self.last_intervene = time.time()
            gripper_action = self.gripper_action.copy()
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
        if time.time() - self.last_intervene < 0.5:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info
