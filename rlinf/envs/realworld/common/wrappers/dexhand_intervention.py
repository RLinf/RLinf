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

"""Dexterous-hand intervention wrapper."""

from __future__ import annotations

import time
from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.glove.glove_expert import GloveExpert
from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import SpaceMouseExpert


class DexHandIntervention(gym.ActionWrapper):
    """Combine SpaceMouse arm control with relative glove control."""

    def __init__(
        self,
        env: gym.Env,
        left_port: Optional[str] = "/dev/ttyACM0",
        right_port: Optional[str] = None,
        glove_frequency: int = 60,
        glove_config_file: Optional[str] = None,
        timeout: float = 0.5,
    ) -> None:
        super().__init__(env)
        assert self.action_space.shape == (12,), (
            f"DexHandIntervention expects a 12-D action space, "
            f"got {self.action_space.shape}"
        )

        self._spacemouse = SpaceMouseExpert()
        self._glove = GloveExpert(
            left_port=left_port,
            right_port=right_port,
            frequency=glove_frequency,
            config_file=glove_config_file,
        )

        self._timeout = timeout
        self._last_intervene: float = 0.0
        self.left: bool = False
        self.right: bool = False

        self._prev_left: bool = False
        self._glove_baseline: np.ndarray | None = None
        self._hand_base: np.ndarray = np.zeros(6, dtype=np.float64)
        self._hand_current: np.ndarray = np.zeros(6, dtype=np.float64)

    def reset(self, **kwargs):
        """Reset the underlying env and sync internal hand state."""
        obs, info = self.env.reset(**kwargs)

        cfg = getattr(self.env, "config", None)
        hand_reset = getattr(cfg, "hand_reset_state", None)
        if hand_reset is not None:
            self._hand_current = np.array(hand_reset, dtype=np.float64)
        else:
            self._hand_current = np.zeros(6, dtype=np.float64)

        self._glove_baseline = None
        self._prev_left = False
        self._hand_base = self._hand_current.copy()
        return obs, info

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return the action after optional expert intervention."""
        arm_expert, buttons = self._spacemouse.get_action()
        self.left, self.right = bool(buttons[1]), bool(buttons[0])

        if np.linalg.norm(arm_expert) > 0.001:
            self._last_intervene = time.time()
        if self.left or self.right:
            self._last_intervene = time.time()

        glove_raw = self._glove.get_angles()

        if self.left:
            if not self._prev_left:
                self._glove_baseline = glove_raw.copy()
                self._hand_base = self._hand_current.copy()

            delta = glove_raw - self._glove_baseline
            hand_target = np.clip(self._hand_base + delta, 0.0, 1.0)
            self._hand_current = hand_target.copy()
            self._last_intervene = time.time()
        else:
            hand_target = self._hand_current.copy()

        self._prev_left = self.left

        expert_action = np.concatenate([arm_expert, hand_target])

        if time.time() - self._last_intervene < self._timeout:
            return expert_action, True

        fallback = np.array(action, dtype=np.float64)
        fallback[6:] = self._hand_current
        return fallback, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        self._glove.close()
        super().close()
