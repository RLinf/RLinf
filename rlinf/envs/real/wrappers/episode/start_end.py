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
    lets the owning runner clean up the environment. Device-specific handover
    hooks belong to an environment wrapper.

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

    @staticmethod
    def _operator_help() -> str:
        return (
            "Operator controls: [a] start | [a] discard recording | "
            "[b] next segment | [c] save success | [q] stop and exit | "
            "[Ctrl-C] emergency stop."
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the robot and print the controls required for this episode."""
        obs, info = super().reset(seed=seed, options=options)
        self.operator_log("Arms ready. Press [a] to start a recording.")
        self.operator_log(self._operator_help())
        return obs, info

    def before_recording_start(self) -> None:
        """Prepare an environment-specific operator handover, if required."""

    def before_recording_abort(self) -> None:
        """Prepare an environment-specific abort/reset, if required."""

    def before_recording_success(self) -> None:
        """Prepare an environment-specific successful reset, if required."""

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

        started = "a" in pressed and not self._recording
        aborted = "a" in pressed and self._recording
        if started:
            self.before_recording_start()
            self._recording = True
        elif aborted:
            self.before_recording_abort()
            self._recording = False

        if not self._recording or started or aborted:
            action = self._hold_action(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # The pedal owns episode boundaries; abort discards the recording but
        # leaves the robot in place for the next start.
        terminated = False
        truncated = False

        record_reset = False
        segment_advance = False
        event: str | None = None
        if started:
            event = "start"
            record_reset = True
            self._last_segment_ts = -math.inf
        elif aborted:
            event = "abort"
            record_reset = True
            self._last_segment_ts = -math.inf

        for key in pressed:
            if key == "a":
                if aborted:
                    reward = 0.0
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
                self.before_recording_success()
                break

        info["pre_record"] = not self._recording
        info["record_reset"] = record_reset
        info["keyboard_phase"] = "rec" if self._recording else "pre"
        info["keyboard_event"] = event
        info["segment_advance"] = segment_advance
        return obs, reward, terminated, truncated, info
