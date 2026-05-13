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
"""Foot-pedal-gated wrapper for autonomous policy eval.

Pedal: ``a`` starts a rollout from idle; ``c`` ends with reward=1
("success"); ``b`` ends with reward=0 ("failure"). On end, returns
``terminated=True`` and lets the outer ``auto_reset`` drive home.
"""

import math
import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class KeyboardEvalControlWrapper(gym.Wrapper):
    """Foot-pedal-gated start/stop for autonomous policy eval rollouts."""

    IDLE_POLL_S = 0.05
    PEDAL_DEBOUNCE_S = 0.2

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.listener = KeyboardListener()
        self._running = False
        self._last_obs: Any = None
        self._last_press_ts: dict[str, float] = {}

    def reset(self, *, seed=None, options=None):
        self._running = False
        self.listener.pop_pressed_keys()
        self._last_press_ts.clear()
        ret = self.env.reset(seed=seed, options=options)
        self._last_obs = ret[0] if isinstance(ret, tuple) else ret
        return ret

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not self._running:
            # Idle: poll pedal, hold robot via the controller's last target.
            time.sleep(self.IDLE_POLL_S)
            for key in self.listener.pop_pressed_keys():
                now = time.monotonic()
                if now - self._last_press_ts.get(key, -math.inf) < self.PEDAL_DEBOUNCE_S:
                    continue
                self._last_press_ts[key] = now
                if key == "a":
                    self._running = True
                    return self._idle_response(event="start")
            return self._idle_response(event=None)

        # Running: forward to the wrapped env.
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        terminated = False
        truncated = False

        result: str | None = None
        for key in self.listener.pop_pressed_keys():
            now = time.monotonic()
            if now - self._last_press_ts.get(key, -math.inf) < self.PEDAL_DEBOUNCE_S:
                continue
            self._last_press_ts[key] = now
            if key == "c":
                terminated = True
                reward = 1.0
                result = "success"
                self._running = False
                break
            if key == "b":
                terminated = True
                reward = 0.0
                result = "failure"
                self._running = False
                break

        info["eval_phase"] = "rec" if self._running else "pre"
        info["eval_result"] = result
        return obs, reward, terminated, truncated, info

    def _idle_response(self, event: str | None):
        info = {"eval_phase": "pre", "eval_event": event, "eval_result": None}
        return self._last_obs, 0.0, False, False, info
