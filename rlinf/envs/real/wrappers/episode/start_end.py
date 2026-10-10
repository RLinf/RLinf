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
    #: A second ``a`` this soon after start is a double tap, not an abort.
    ABORT_LOCKOUT_S = 1.5

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._recording = False
        self._recording_since = -math.inf
        self._last_segment_ts = -math.inf

    def begin_episode(self) -> None:
        """Clear segment history before recording a new episode."""
        self._recording = False
        self._recording_since = -math.inf
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
                    if now - self._recording_since < self.ABORT_LOCKOUT_S:
                        self.log(
                            "Key 'a' ignored; recording just started. "
                            "Press 'c' to save, or 'a' again after a moment to abort."
                        )
                        continue
                    # Abort recording without moving the robot.
                    event = "abort"
                    self._recording = False
                    self._recording_since = -math.inf
                    record_reset = True
                    self._last_segment_ts = -math.inf
                else:
                    # Start recording from the current pose.
                    event = "start"
                    self._recording = True
                    self._recording_since = now
                    record_reset = True
                    self._last_segment_ts = -math.inf
            elif key == "b":
                if not self._recording:
                    self.log("Key 'b' ignored; press 'a' to start recording first.")
                    continue
                if now - self._last_segment_ts >= self.SEGMENT_DEBOUNCE_S:
                    event = "segment"
                    segment_advance = True
                    self._last_segment_ts = now
                # Ignore rapid repeats to avoid very short segments.
            elif key == "c":
                if not self._recording:
                    self.log("Key 'c' ignored; press 'a' to start recording first.")
                    continue
                event = "end_success"
                reward = 1.0
                terminated = True
                # CollectEpisode(only_success=True) keys off info["success"],
                # not the scalar reward. Without this flag the buffer is dropped.
                info["success"] = True
                # Keep recording enabled so the successful terminal frame is saved.
                self.log("Key 'c': success, writing episode.")
                break

        info["pre_record"] = not self._recording
        info["record_reset"] = record_reset
        info["keyboard_phase"] = "rec" if self._recording else "pre"
        info["keyboard_event"] = event
        info["segment_advance"] = segment_advance
        if event == "start":
            self.log("Key 'a': start recording. Press 'c' to save, 'a' to abort.")
        elif event == "abort":
            self.log("Key 'a': abort, drop buffer.")
        elif event == "segment":
            self.log("Key 'b': new segment.")
        return obs, reward, terminated, truncated, info
