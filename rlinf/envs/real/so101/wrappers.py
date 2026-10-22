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
from typing import Any

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
