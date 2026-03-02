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

"""Dexterous-hand intervention wrapper.

This wrapper combines a :class:`SpaceMouseExpert` (for the 6-D arm) with
a :class:`GloveExpert` (for the 6-D hand fingers) to form a 12-D expert
action that can override the RL policy output.

The SpaceMouse buttons still trigger the *intervene window* (0.5 s after
the last non-zero input), but the gripper open/close logic is replaced by
continuous glove angles.
"""

from __future__ import annotations

import time
from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.glove.glove_expert import GloveExpert
from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import SpaceMouseExpert


class DexHandIntervention(gym.ActionWrapper):
    """Action wrapper for SpaceMouse + data-glove human intervention.

    Expected action space: ``(12,)`` — 6 arm DOFs + 6 hand DOFs.

    The intervention logic:

    * **Arm (first 6 dims):**  Uses SpaceMouse 6-D delta.  If the norm
      exceeds a small threshold the *intervene clock* is refreshed.
    * **Hand (last 6 dims):**  Uses data-glove absolute angles [0, 1].
      Any glove movement also refreshes the *intervene clock*.
    * **SpaceMouse buttons:**  Left/right buttons are exposed in ``info``
      for downstream usage (e.g. reward labelling) and also refresh the
      clock.

    While the *intervene clock* is active (within ``timeout`` seconds of
    the last human input), the expert action replaces the policy action.

    Args:
        env: Gymnasium environment with a 12-D action space.
        left_port: Serial port for the left data-glove (``None`` to disable).
        right_port: Serial port for the right data-glove.
        glove_frequency: Glove polling frequency in Hz.
        glove_config_file: Calibration YAML for the glove driver.
        timeout: Seconds after last expert input before yielding back
            to the policy.
    """

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

    # ------------------------------------------------------------------
    # gym.ActionWrapper interface
    # ------------------------------------------------------------------

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Build expert action and decide whether to override policy.

        Returns:
            ``(final_action, replaced)`` where *replaced* is ``True``
            when the expert overrides the policy.
        """
        # --- SpaceMouse (arm) ---
        arm_expert, buttons = self._spacemouse.get_action()
        self.left, self.right = bool(buttons[0]), bool(buttons[1])

        if np.linalg.norm(arm_expert) > 0.001:
            self._last_intervene = time.time()
        if self.left or self.right:
            self._last_intervene = time.time()

        # --- Glove (hand) ---
        hand_expert = self._glove.get_angles()  # (6,) in [0, 1]
        if np.linalg.norm(hand_expert) > 0.001:
            self._last_intervene = time.time()

        expert_action = np.concatenate([arm_expert, hand_expert])

        if time.time() - self._last_intervene < self._timeout:
            return expert_action, True

        return action, False

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
