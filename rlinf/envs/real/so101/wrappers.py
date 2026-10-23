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
import threading
import time
from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ActType, ObsType

from rlinf.envs.real.wrappers.episode.start_end import KeyboardStartEndWrapper


class SO101StartEndWrapper(KeyboardStartEndWrapper):
    """Collect SO-101 demonstrations with explicit leader handover."""

    LOG_PREFIX = "[SO-101]"

    def _teleop_attr(self, name: str) -> Any:
        """Return an optional lifecycle hook from the SO-101 leader."""
        try:
            composed = self.env.get_wrapper_attr("device")
        except AttributeError:
            return None
        for device in getattr(composed.group, "devices", ()):
            hook = getattr(device, name, None)
            if hook is not None:
                return hook
        return None

    def _teleop_context(self) -> dict[str, Any]:
        """Build the context expected by the leader lifecycle hooks."""
        try:
            composed = self.env.get_wrapper_attr("device")
        except AttributeError:
            return {}
        context = composed.context_from(self.env)
        # The minimal PR leaves the common ComposedTeleop context unchanged.
        # Collect reset targets through SO-101's own environment getters so
        # the leader can synchronize its reset without widening that API.
        for key in (
            "reset_joint_positions",
            "reset_gripper_position",
            "reset_duration",
            "reset_joint_speed",
        ):
            try:
                getter = self.env.get_wrapper_attr(f"get_{key}")
            except AttributeError:
                continue
            if callable(getter):
                context[key] = getter()
        return context

    def _countdown(self, message: str, seconds: float) -> None:
        """Give the operator time to change hand position safely."""
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.log("%s: %d", message, math.ceil(remaining))
            time.sleep(min(1.0, remaining))

    def _hold_before_reset(self) -> None:
        """Reapply leader torque and wait before reset or park."""
        hold = self._teleop_attr("hold_for_reset")
        if hold is not None:
            hold(self._teleop_context())
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
            release(self._teleop_context())

    def before_recording_abort(self) -> None:
        """Hold the leader before the runner resets the discarded episode."""
        self._hold_before_reset()

    def before_recording_success(self) -> None:
        """Hold the leader before the runner resets the saved episode."""
        self._hold_before_reset()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset leader and follower together before the next episode."""
        prepare_hook = self._teleop_attr("prepare_reset")
        abort = self._teleop_attr("abort_reset")
        context = self._teleop_context()
        errors: list[BaseException] = []
        reset_thread: threading.Thread | None = None
        completed = False

        def prepare_leader() -> None:
            try:
                prepare_hook(context)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        try:
            if prepare_hook is not None:
                reset_thread = threading.Thread(
                    target=prepare_leader, name="so101-leader-reset", daemon=True
                )
                reset_thread.start()
            result = super().reset(seed=seed, options=options)
            if reset_thread is not None:
                reset_thread.join()
            if errors:
                raise errors[-1]
            completed = True
            return result
        finally:
            if reset_thread is not None:
                reset_thread.join()
            if not completed and abort is not None:
                abort(context)

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
        started = "a" in pressed and not self._recording
        aborted = "a" in pressed and self._recording
        if started:
            self.before_recording_start()
        elif aborted:
            self.before_recording_abort()

        # Abort wins when both keys arrive in one poll; never save an episode
        # after the operator explicitly discarded it.
        success = "c" in pressed and self._recording and not aborted
        if success:
            # Stop accepting leader actions before the terminal transition.
            # The terminal frame is still marked as recordable below.
            self.before_recording_success()
            self._recording = False
            self._last_segment_ts = -math.inf

        if not self._recording or started or aborted or success:
            action = self._hold_action(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        if success:
            # TeleopIntervention intentionally does not mark an inactive
            # device's hold action as an intervention. This action was chosen
            # by this wrapper, so expose it to both dataset writers.
            info["intervene_action"] = action
            info["intervene_flag"] = np.ones(1, dtype=bool)
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
            elif key == "c" and success:
                event = "end_success"
                reward = 1.0
                terminated = True
                break

        info["pre_record"] = not self._recording and event != "end_success"
        info["record_reset"] = record_reset
        info["keyboard_phase"] = (
            "rec" if self._recording or event == "end_success" else "pre"
        )
        info["keyboard_event"] = event
        info["segment_advance"] = segment_advance
        return obs, reward, terminated, truncated, info
