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

"""Dual-SpaceMouse intervention wrapper for dual-arm Franka environments."""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import SpaceMouseExpert


class DualSpacemouseIntervention(gym.ActionWrapper):
    """Override the policy action with two SpaceMouse devices.

    Each SpaceMouse controls one arm (left = device 0, right = device 1).
    The output action is 14-dim: ``[left_7d, right_7d]`` when gripper is
    enabled, or ``[left_6d, right_6d]`` otherwise.

    Args:
        env: The wrapped dual-arm environment.
        gripper_enabled: Whether the gripper channel is present.
    """

    def __init__(self, env: gym.Env, gripper_enabled: bool = True):
        super().__init__(env)
        self.gripper_enabled = gripper_enabled
        self.left_expert = SpaceMouseExpert(device_index=0)
        self.right_expert = SpaceMouseExpert(device_index=1)
        self.last_intervene = 0
        self.left_btn = (False, False)
        self.right_btn = (False, False)

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        left_a, left_buttons = self.left_expert.get_action()
        right_a, right_buttons = self.right_expert.get_action()

        l_left, l_right = tuple(left_buttons)
        r_left, r_right = tuple(right_buttons)
        self.left_btn = (l_left, l_right)
        self.right_btn = (r_left, r_right)

        any_active = (
            np.linalg.norm(left_a) > 0.001
            or np.linalg.norm(right_a) > 0.001
            or (l_left + l_right) > 0.5
            or (r_left + r_right) > 0.5
        )
        if any_active:
            self.last_intervene = time.time()

        if self.gripper_enabled:
            left_grip = self._gripper_from_buttons(l_left, l_right)
            right_grip = self._gripper_from_buttons(r_left, r_right)
            left_a = np.concatenate([left_a, left_grip])
            right_a = np.concatenate([right_a, right_grip])

        expert_a = np.concatenate([left_a, right_a])

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True
        return action, False

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict, float, bool, bool, dict]:
        """Run one env step, replacing the action when the operator is active."""
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1)
        info["left_btn"] = self.left_btn
        info["right_btn"] = self.right_btn
        return obs, rew, done, truncated, info

    @staticmethod
    def _gripper_from_buttons(btn_left: bool, btn_right: bool) -> np.ndarray:
        """Map SpaceMouse button presses to a 1-D gripper action."""
        if btn_left:
            return np.random.uniform(-1, -0.9, size=(1,))
        elif btn_right:
            return np.random.uniform(0.9, 1, size=(1,))
        return np.zeros((1,))
