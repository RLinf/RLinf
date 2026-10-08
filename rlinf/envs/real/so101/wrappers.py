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

"""SO-101-specific episode controls for leader-follower collection."""

from __future__ import annotations

import math
import time
from typing import Any, SupportsFloat

from gymnasium.core import ActType, ObsType

from rlinf.envs.real.wrappers.episode.session import KeyboardAbort
from rlinf.envs.real.wrappers.episode.start_end import KeyboardStartEndWrapper


class SO101StartEndWrapper(KeyboardStartEndWrapper):
    """Collect SO-101 demonstrations with explicit leader handover."""

    LOG_PREFIX = "[SO-101]"

    def _teleop_attr(self, name: str) -> Any:
        """Return an optional lifecycle hook from the wrapped teleop stack."""
        try:
            return self.env.get_wrapper_attr(name)
        except AttributeError:
            return None

    def _countdown(self, message: str, seconds: float) -> None:
        """Give the operator time to change hand position safely."""
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.operator_log("%s: %d", message, math.ceil(remaining))
            time.sleep(min(1.0, remaining))

    def _hold_before_reset(self) -> None:
        """Reapply leader torque and wait before reset or park."""
        hold = self._teleop_attr("hold_for_reset")
        if hold is not None:
            hold()
        seconds = self._teleop_attr("manual_start_hold_seconds")
        if seconds is None:
            return
        seconds = float(seconds)
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError("manual_start_hold_seconds must be finite and nonnegative")
        if seconds:
            self._countdown("Keep hands clear; reset starts in", seconds)

    def before_recording_start(self) -> None:
        """Release the leader after the handover countdown."""
        seconds = self._teleop_attr("manual_start_hold_seconds")
        if seconds is None:
            seconds = 0.0
        seconds = float(seconds)
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError("manual_start_hold_seconds must be finite and nonnegative")
        if seconds:
            self._countdown("Manual control starts in", seconds)
        release = self._teleop_attr("release_for_manual")
        if release is not None:
            release()

    def before_recording_abort(self) -> None:
        """Hold the leader before the runner resets the discarded episode."""
        self._hold_before_reset()

    def before_recording_success(self) -> None:
        """Hold the leader before the runner resets the saved episode."""
        self._hold_before_reset()

    def _hold_action(self, action: ActType) -> ActType:
        """Use the follower pose while the absolute-position leader is idle."""
        try:
            hold = self.env.get_wrapper_attr("get_hold_action")
        except AttributeError:
            return action
        try:
            return hold(action)
        except AttributeError:
            return action

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Handle SO-101 handover before the wrapped environment steps."""
        pressed = list(self.presses())
        if any(key in {"q", "quit"} for key in pressed):
            raise KeyboardAbort("Operator requested collection shutdown.")

        started = "a" in pressed and not self._recording
        aborted = "a" in pressed and self._recording
        if started:
            self.before_recording_start()
        elif aborted:
            self.before_recording_abort()

        if not self._recording or started or aborted:
            action = self._hold_action(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        terminated = aborted
        truncated = False
        record_reset = False
        segment_advance = False
        event: str | None = None

        if started:
            event = "start"
            self._recording = True
            record_reset = True
            self._last_segment_ts = -math.inf
        elif aborted:
            event = "abort"
            self._recording = False
            record_reset = True
            self._last_segment_ts = -math.inf

        for key in pressed:
            if key == "a":
                if aborted:
                    reward = 0.0
                elif not started:
                    event = "abort"
                    self._recording = False
                    record_reset = True
                    self._last_segment_ts = -math.inf
                    terminated = True
                    reward = 0.0
            elif key == "b" and self._recording:
                now = time.monotonic()
                if now - self._last_segment_ts >= self.SEGMENT_DEBOUNCE_S:
                    event = "segment"
                    segment_advance = True
                    self._last_segment_ts = now
            elif key == "c" and self._recording:
                event = "end_success"
                reward = 1.0
                terminated = True
                self.before_recording_success()
                break

        info["pre_record"] = not self._recording
        info["record_reset"] = record_reset
        info["keyboard_phase"] = "rec" if self._recording else "pre"
        info["keyboard_event"] = event
        info["segment_advance"] = segment_advance
        return obs, reward, terminated, truncated, info
