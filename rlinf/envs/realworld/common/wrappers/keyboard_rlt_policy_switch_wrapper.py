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

import math
import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class KeyboardRLTPolicySwitchWrapper(gym.Wrapper):
    """Press ``b`` to enter the RLT critical phase."""

    PEDAL_DEBOUNCE_S = 0.2

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.listener = KeyboardListener()
        self._use_actor = False
        self._last_press_ts: dict[str, float] = {}

    @property
    def rlt_use_actor(self) -> bool:
        return self._use_actor

    def reset(self, *, seed=None, options=None):
        self._use_actor = False
        self._last_press_ts.clear()
        self.listener.pop_pressed_keys()
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        event: str | None = None
        for key in self.listener.pop_pressed_keys():
            now = time.monotonic()
            if now - self._last_press_ts.get(key, -math.inf) < self.PEDAL_DEBOUNCE_S:
                continue
            self._last_press_ts[key] = now

            if key == "b":
                if not self._use_actor:
                    event = "enter_actor"
                    self._use_actor = True
                    self._log_info(
                        "Pedal 'b' pressed; switching RLT rollout to Stage2 actor."
                    )
                else:
                    event = "actor_already_active"

        info["rlt_use_actor"] = self._use_actor
        info["rlt_policy_switch_event"] = event
        return obs, reward, terminated, truncated, info

    def _log_info(self, message: str) -> None:
        logger = getattr(self._base_env(), "_logger", None)
        if logger is not None:
            logger.info(message)

    def _base_env(self):
        return getattr(self.env, "unwrapped", self.env)
