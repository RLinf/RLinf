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

**Glove control mode (relative):**

When the SpaceMouse **left button is pressed**, the current glove reading
is captured as the *baseline*.  While the button is held, only the
**change** relative to that baseline is applied to the hand's current
position.  When the button is **released**, the hand freezes in place.

This avoids the sudden jump that absolute-mode control would cause and
gives the operator fine incremental control.
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
    * **Hand (last 6 dims):**  Relative glove control — only active
      while the SpaceMouse left button is held.  On press, the current
      glove reading is saved as the baseline; delta from baseline is
      added to the hand's position at that moment.  On release the hand
      stays where it is.
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

        # --- Relative glove control state ---
        self._prev_left: bool = False  # left-button state on previous step
        self._glove_baseline: np.ndarray | None = None  # glove reading at press
        self._hand_base: np.ndarray = np.zeros(6, dtype=np.float64)  # hand pos at press
        self._hand_current: np.ndarray = np.zeros(6, dtype=np.float64)  # latest hand target

    # ------------------------------------------------------------------
    # gym.ActionWrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        """Reset the underlying env and sync internal hand state."""
        obs, info = self.env.reset(**kwargs)

        # Sync _hand_current with the physical reset pose so the
        # operator starts the new episode from the actual hand position
        # instead of the stale position from the previous episode.
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
        """Build expert action and decide whether to override policy.

        Returns:
            ``(final_action, replaced)`` where *replaced* is ``True``
            when the expert overrides the policy.
        """
        # --- SpaceMouse (arm) ---
        arm_expert, buttons = self._spacemouse.get_action()
        # pyspacemouse: buttons[0] = physical right, buttons[1] = physical left
        self.left, self.right = bool(buttons[1]), bool(buttons[0])

        if np.linalg.norm(arm_expert) > 0.001:
            self._last_intervene = time.time()
        if self.left or self.right:
            self._last_intervene = time.time()

        # --- Glove (hand) — relative mode ---
        glove_raw = self._glove.get_angles()  # (6,) in [0, 1]

        if self.left:
            if not self._prev_left:
                # Left button just pressed — capture baseline
                self._glove_baseline = glove_raw.copy()
                self._hand_base = self._hand_current.copy()

            # Compute delta from baseline and add to hand position at press
            delta = glove_raw - self._glove_baseline
            hand_target = np.clip(self._hand_base + delta, 0.0, 1.0)
            self._hand_current = hand_target.copy()
            self._last_intervene = time.time()
        else:
            # Left button not pressed — hand stays where it is
            hand_target = self._hand_current.copy()

        self._prev_left = self.left

        expert_action = np.concatenate([arm_expert, hand_target])

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
