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

from .session import KeyboardAbort, KeyboardSession


class KeyboardStartEndWrapper(KeyboardSession):
    """Control data-collection episodes with a three-key foot pedal.

    ``a`` starts or aborts recording, ``b`` advances the segment, and ``c``
    ends the episode successfully. ``q`` requests a controlled shutdown that
    lets the owning runner park and close the hardware. Aborting preserves the
    current robot pose.

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

    def _release_for_manual(self) -> None:
        """Hold the reset pose, then release the leader to the operator."""
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

    def _hold_before_reset(self) -> None:
        """Reapply leader torque and wait before a reset or park operation."""
        hold = self._teleop_attr("hold_for_reset")
        if hold is not None:
            hold()
        seconds = self._teleop_attr("manual_start_hold_seconds")
        if seconds is not None:
            seconds = float(seconds)
            if seconds < 0 or not math.isfinite(seconds):
                raise ValueError(
                    "manual_start_hold_seconds must be finite and nonnegative"
                )
            if seconds:
                self._countdown("Keep hands clear; reset starts in", seconds)

    def _hold_action(self, action: ActType) -> ActType:
        """Use the current robot pose while the recording pedal is idle."""
        try:
            hold = self.env.get_wrapper_attr("get_hold_action")
        except AttributeError:
            return action
        try:
            return hold(action)
        except AttributeError:
            # Delta-only teleop devices have no absolute pose to hold; their
            # zero action is already the stable idle command.
            return action

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        pressed = list(self.presses())
        if any(key in {"q", "quit"} for key in pressed):
            raise KeyboardAbort("Operator requested collection shutdown.")

        record_reset = False
        segment_advance = False
        event: str | None = None

        # Start handover before stepping the wrapped environment. The buffer
        # therefore produces no transition that can enter the recorded episode.
        started = "a" in pressed and not self._recording
        aborted = "a" in pressed and self._recording
        if started:
            self._release_for_manual()
            event = "start"
            self._recording = True
            record_reset = True
            self._last_segment_ts = -math.inf
        elif aborted:
            # Match the native LeRobot recorder: discard, give the operator
            # time to clear the leader, then let the runner reset both arms.
            self._hold_before_reset()
            event = "abort"
            self._recording = False
            record_reset = True
            self._last_segment_ts = -math.inf

        if not self._recording or started or aborted:
            action = self._hold_action(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # The pedal owns episode boundaries; start and abort do not reset the env.
        terminated = aborted
        truncated = False

        for key in pressed:
            if key == "a":
                if aborted:
                    terminated = True
                    reward = 0.0
                elif self._recording:
                    if event != "start":
                        # Abort recording without moving the robot.
                        event = "abort"
                        self._recording = False
                        record_reset = True
                        self._last_segment_ts = -math.inf
                        terminated = True
                        reward = 0.0
                else:
                    # The start event was handled before the environment step.
                    continue
            elif key == "b" and self._recording:
                now = time.monotonic()
                if now - self._last_segment_ts >= self.SEGMENT_DEBOUNCE_S:
                    event = "segment"
                    segment_advance = True
                    self._last_segment_ts = now
                # Ignore rapid repeats to avoid very short segments.
            elif key == "c" and self._recording:
                event = "end_success"
                reward = 1.0
                terminated = True
                self._hold_before_reset()
                # Keep recording enabled so the successful terminal frame is saved.
                break

        info["pre_record"] = not self._recording
        info["record_reset"] = record_reset
        info["keyboard_phase"] = "rec" if self._recording else "pre"
        info["keyboard_event"] = event
        info["segment_advance"] = segment_advance
        return obs, reward, terminated, truncated, info
