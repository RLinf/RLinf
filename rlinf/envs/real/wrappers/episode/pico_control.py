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

"""Gate real-world episodes with Pico controller buttons."""

from __future__ import annotations

import time
from typing import Any, Optional, SupportsFloat

import gymnasium as gym


class PicoEpisodeControlWrapper(gym.Wrapper):
    """Replace the keyboard pedal with global Pico button events.

    Each episode first resets the robot, then waits for ``start_button`` before
    allowing either teleoperation or policy actions to reach the environment.
    ``success_button`` and ``failure_button`` finish the episode. This is the
    Pico equivalent of the existing keyboard ``eval_control`` lifecycle.

    The wrapper preserves the ``keyboard_*`` metadata names because the
    existing data collector uses those names to delimit and profile manually
    collected demonstrations.
    """

    IDLE_POLL_S = 0.05
    WAIT_HEARTBEAT_S = 10.0

    def __init__(
        self,
        env: gym.Env,
        *,
        start_button: str = "B",
        success_button: str = "X",
        failure_button: str = "A",
    ) -> None:
        super().__init__(env)
        self.start_button = str(start_button)
        self.success_button = str(success_button)
        self.failure_button = str(failure_button)
        buttons = (self.start_button, self.success_button, self.failure_button)
        if any(not button for button in buttons) or len(set(buttons)) != len(buttons):
            raise ValueError("Pico episode-control buttons must be distinct and nonempty.")
        self._recording = False
        self._running = False
        self._last_obs: Any = None
        self._warned_no_source = False

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        self._recording = False
        self._running = False
        self._drain_presses()
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        if not bool((options or {}).get("skip_wait_for_start", False)):
            self._wait_for_start()
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if not self._running:
            time.sleep(self.IDLE_POLL_S)
            if self._start_if_pressed():
                return self._idle_response(event="start")
            return self._idle_response(event=None)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        # Episode boundaries are owned by the Pico buttons, never by the base env.
        terminated = False
        truncated = False
        event: str | None = None

        for button in self._drain_presses():
            if button == self.success_button:
                reward = 1.0
                terminated = True
                event = "end_success"
                self._running = False
                self._log("Pico %s: marked success", button)
                break
            if button == self.failure_button:
                reward = 0.0
                terminated = True
                event = "end_failure"
                self._running = False
                self._log("Pico %s: marked failure", button)
                break

        info["pre_record"] = not self._recording
        info["record_reset"] = False
        info["keyboard_phase"] = "rec" if self._running else "pre"
        info["keyboard_event"] = event
        info["pico_episode_event"] = event
        return obs, reward, terminated, truncated, info

    def _wait_for_start(self) -> None:
        """Block after reset until the operator authorizes the next episode."""
        self._log("Arms reset and idle. Waiting for Pico tracking calibration.")
        last_heartbeat = time.monotonic()
        calibration_reported = False
        check_stop = getattr(self.unwrapped, "check_stop_requested", None)
        while True:
            try:
                if check_stop is not None:
                    check_stop()
                calibrated = self._calibration_ready()
                if calibrated:
                    if not calibration_reported:
                        calibration_reported = True
                        self._log(
                            "Pico tracking calibrated. Press Pico %s to start the next "
                            "episode.",
                            self.start_button,
                        )
                    if self._start_if_pressed():
                        return
                else:
                    # Require a deliberate start press after tracking is ready.
                    self._drain_presses()
                time.sleep(self.IDLE_POLL_S)
                now = time.monotonic()
                if now - last_heartbeat >= self.WAIT_HEARTBEAT_S:
                    last_heartbeat = now
                    if calibrated:
                        self._log(
                            "Still waiting for Pico %s to start the episode.",
                            self.start_button,
                        )
                    else:
                        self._log("Still waiting for Pico tracking calibration.")
            except KeyboardInterrupt:
                self.close()
                raise

    def _start_if_pressed(self) -> bool:
        """Start a fresh episode when its one-shot button event arrives."""
        if self._running:
            return True
        for button in self._drain_presses():
            if button == self.start_button:
                self._recording = True
                self._running = True
                self._log("Pico %s: episode started", button)
                return True
        return False

    def _idle_response(
        self, event: str | None
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Avoid sending an action while the wrapper waits for a start event."""
        return self._last_obs, 0.0, False, False, {
            "pre_record": True,
            "record_reset": False,
            "keyboard_phase": "pre",
            "keyboard_event": event,
            "pico_episode_event": event,
        }

    def _drain_presses(self) -> list[str]:
        """Read rising-edge events from one Pico ZMQ reader, if available."""
        reader = self._pico_reader()
        if reader is None:
            if not self._warned_no_source:
                self._warned_no_source = True
                self._log("Pico episode control is waiting for a connected Pico reader")
            return []
        return list(reader.pop_button_presses())

    def _calibration_ready(self) -> bool:
        """Drive automatic calibration for every Pico reader before starting."""
        readers = self._pico_readers()
        if not readers:
            return False
        for reader in readers:
            get_reading = getattr(reader, "get_reading", None)
            if not callable(get_reading):
                return False
            reading = get_reading()
            if not bool(reading.get("ready", False)):
                return False
            if bool(getattr(reader, "require_calibration", False)) and not bool(
                reading.get("calibrated", False)
            ):
                return False
        return True

    def _pico_reader(self) -> Any | None:
        """Find the first Pico transport already owned by TeleopIntervention."""
        readers = self._pico_readers()
        return readers[0] if readers else None

    def _pico_readers(self) -> list[Any]:
        """Find every distinct Pico reader already owned by TeleopIntervention."""
        try:
            teleop = self.env.get_wrapper_attr("device")
        except AttributeError:
            return []
        group = getattr(teleop, "group", None)
        readers: list[Any] = []
        seen: set[int] = set()
        for device in getattr(group, "devices", ()):
            reader = getattr(device, "_device", None)
            if (
                callable(getattr(reader, "pop_button_presses", None))
                and id(reader) not in seen
            ):
                seen.add(id(reader))
                readers.append(reader)
        return readers

    def _log(self, message: str, *args: Any) -> None:
        base = getattr(self.env, "unwrapped", self.env)
        logger = getattr(base, "_logger", None)
        if logger is not None:
            logger.info(message, *args)
