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

from gymnasium.core import ActType, Env, ObsType

from .session import KeyboardSession


class KeyboardStartEndWrapper(KeyboardSession):
    """Control data-collection episodes with a three-key foot pedal.

    ``a`` starts or aborts recording, ``b`` advances the segment, and ``c``
    ends the episode successfully. Aborting preserves the current robot pose.

    Adds ``keyboard_phase`` / ``keyboard_event`` / ``pre_record`` /
    ``record_reset`` / ``segment_advance`` to ``info`` for ``CollectEpisode``.
    """

    SEGMENT_DEBOUNCE_S = 1.0

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._recording = False
        self._last_segment_ts = -math.inf

    def begin_episode(self) -> None:
        """Clear segment history before recording a new episode."""
        self._recording = False
        self._last_segment_ts = -math.inf

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # The pedal owns episode boundaries; start and abort do not reset the env.
        terminated = False
        truncated = False

        record_reset = False
        segment_advance = False
        event: str | None = None

        for key in self.presses():
            now = time.monotonic()
            if key == "a":
                if self._recording:
                    # Abort recording without moving the robot.
                    event = "abort"
                    self._recording = False
                    record_reset = True
                    self._last_segment_ts = -math.inf
                else:
                    # Start recording from the current pose.
                    event = "start"
                    self._recording = True
                    record_reset = True
                    self._last_segment_ts = -math.inf
            elif key == "b" and self._recording:
                if now - self._last_segment_ts >= self.SEGMENT_DEBOUNCE_S:
                    event = "segment"
                    segment_advance = True
                    self._last_segment_ts = now
                # Ignore rapid repeats to avoid very short segments.
            elif key == "c" and self._recording:
                event = "end_success"
                reward = 1.0
                terminated = True
                # Keep recording enabled so the successful terminal frame is saved.
                break

        info["pre_record"] = not self._recording
        info["record_reset"] = record_reset
        info["keyboard_phase"] = "rec" if self._recording else "pre"
        info["keyboard_event"] = event
        info["segment_advance"] = segment_advance
        return obs, reward, terminated, truncated, info
