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


class KeyboardStartEndWrapper(gym.Wrapper):
    """Foot-pedal data-collection wrapper. Pedal binding (``a`` / ``b`` / ``c``):

    * ``a``        — start a rec episode (pre) or abort the current one (rec).
                     Abort drops the buffer; the arm is **not** reset (GELLO
                     keeps tracking the operator).
    * ``b`` (rec)  — bump ``segment_id``, debounced at 1 s.
    * ``c`` (rec)  — end with reward=1, terminated=True.

    Adds ``keyboard_phase`` / ``keyboard_event`` / ``pre_record`` /
    ``record_reset`` / ``segment_advance`` to ``info`` for ``CollectEpisode``.
    """

    SEGMENT_DEBOUNCE_S = 1.0
    PEDAL_DEBOUNCE_S = 0.2

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.listener = KeyboardListener()
        self._recording = False
        self._last_segment_ts = -math.inf
        self._last_press_ts: dict[str, float] = {}

    def reset(self, *, seed=None, options=None):
        self._recording = False
        self._last_segment_ts = -math.inf
        self._last_press_ts.clear()
        # Drain queued presses so they don't leak into the next episode.
        self.listener.pop_pressed_keys()
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Pedal owns episode boundaries — pre/abort must NOT auto-reset.
        terminated = False
        truncated = False

        record_reset = False
        segment_advance = False
        event: str | None = None

        for key in self.listener.pop_pressed_keys():
            now = time.monotonic()
            # Anti-bounce: drop same-key repeats within PEDAL_DEBOUNCE_S
            # (covers operator double-taps and flaky USB key-down bursts).
            if now - self._last_press_ts.get(key, -math.inf) < self.PEDAL_DEBOUNCE_S:
                continue
            self._last_press_ts[key] = now

            if key == "a":
                if self._recording:
                    # Drop the in-progress episode but stay where we are.
                    event = "abort"
                    self._recording = False
                    record_reset = True
                    self._last_segment_ts = -math.inf
                else:
                    # Begin a fresh rec episode at the current pose.
                    event = "start"
                    self._recording = True
                    record_reset = True
                    self._last_segment_ts = -math.inf
            elif key == "b" and self._recording:
                if now - self._last_segment_ts >= self.SEGMENT_DEBOUNCE_S:
                    event = "segment"
                    segment_advance = True
                    self._last_segment_ts = now
                # else: silently ignore — keeps mini-segments out of the data.
            elif key == "c" and self._recording:
                event = "end_success"
                reward = 1.0
                terminated = True
                # Keep _recording=True so this terminating frame keeps
                # pre_record=False — else CollectEpisode skips the
                # reward=1.0 step and only_success drops the episode.
                break

        info["pre_record"] = not self._recording
        info["record_reset"] = record_reset
        info["keyboard_phase"] = "rec" if self._recording else "pre"
        info["keyboard_event"] = event
        info["segment_advance"] = segment_advance
        return obs, reward, terminated, truncated, info
