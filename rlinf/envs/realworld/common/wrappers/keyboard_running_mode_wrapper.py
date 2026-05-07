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

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class KeyboardRunningModeWrapper(gym.Wrapper):
    """Set the X2Robot running-mode ROS param from keyboard events."""

    def __init__(
        self,
        env: gym.Env,
        config: Mapping[str, Any] | None = None,
        listener: Any | None = None,
        param_setter: Callable[[str, int], None] | None = None,
        time_fn: Callable[[], float] | None = None,
    ):
        super().__init__(env)
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.running_mode_param = str(cfg.get("running_mode_param", "/running_mode"))
        self.debounce_s = float(cfg.get("debounce_s", 0.3))
        if self.debounce_s < 0:
            raise ValueError("keyboard_running_mode.debounce_s must be non-negative.")

        self._key_to_mode = {
            str(cfg.get("normal_key", "1")): int(cfg.get("normal_mode_value", 1)),
            str(cfg.get("takeover_key", "2")): int(cfg.get("takeover_mode_value", 2)),
        }

        self.listener = listener if listener is not None else KeyboardListener()
        self._set_param = param_setter or self._set_ros_param
        self._time_fn = time_fn or time.monotonic
        self._logger = logging.getLogger(__name__)
        self._last_seen_key: str | None = None
        self._last_set_time = -float("inf")

    @staticmethod
    def _set_ros_param(param_name: str, value: int) -> None:
        import rospy

        rospy.set_param(param_name, value)

    def step(self, action):
        self.poll_keyboard_running_mode()
        return self.env.step(action)

    def poll_keyboard_running_mode(self) -> None:
        if not self.enabled:
            return
        key = self.listener.get_key()
        if key is None:
            self._last_seen_key = None
            return
        key = str(key)
        if key == self._last_seen_key:
            return

        mode_value = self._key_to_mode.get(key)
        if mode_value is None:
            return

        now = self._time_fn()
        if now - self._last_set_time < self.debounce_s:
            return

        self._set_param(self.running_mode_param, mode_value)
        self._last_seen_key = key
        self._last_set_time = now
        self._logger.info(
            "Keyboard running mode: key=%s -> %s=%s",
            key,
            self.running_mode_param,
            mode_value,
        )
