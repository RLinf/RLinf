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

"""Episode control for a dual YAM station driven by PICO controllers.

The :class:`~rlinf.robotics.parts.teleop.yam_pico.YamPico` device decides where
the arms go. This wrapper decides what counts as an episode: it turns the
controller buttons and the R/X keys into record, discard, and successful-end
edges, keeps teleoperation ticking while it waits for the first record edge, and
publishes the phase keys ``CollectEpisode`` reads.

It sits outside :class:`~rlinf.envs.real.wrappers.teleop.TeleopIntervention`,
which merges the device's published state into ``info`` and writes the applied
action there as ``accepted_action``.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.real.wrappers.episode.keyboard import KeyboardListener
from rlinf.robotics.parts.teleop.yam_pico import YamPico
from rlinf.utils.logging import get_logger

from .config import YamPicoConfig

#: Keys the operator can press instead of reaching for a controller button.
_KEYBOARD_KEYS = {"r": "record", "x": "discard"}


class YamPicoEpisode(gym.Wrapper):
    """Record, discard, and end collection episodes from a PICO station.

    Every step is driven by the device, so this wrapper never sees a policy
    action it needs to arbitrate. It reads the device's episode buttons out of
    ``info``, debounces them together with the keyboard, and reports the result
    to the collector.

    Args:
        env: The environment to wrap, with the teleop device inside it.
        config: Episode-control settings. These are the same ``pico`` block the
            device reads its own settings from.
        keyboard: Keyboard to poll instead of opening a real one.

    Raises:
        ValueError: If ``keyboard_enabled`` is set and no keyboard is available.
    """

    def __init__(
        self,
        env: gym.Env,
        config: YamPicoConfig | Mapping[str, Any] | None = None,
        *,
        keyboard: Any | None = None,
    ) -> None:
        super().__init__(env)
        self.config = (
            config
            if isinstance(config, YamPicoConfig)
            else YamPicoConfig(**dict(config or {}))
        )
        self._keyboard = keyboard
        self._recording = False
        self._preserve_reference = False
        self._previous_buttons: Optional[tuple[bool, bool]] = None
        self._last_edges = [-float("inf"), -float("inf")]
        self._closed = False
        self._logger = get_logger()
        self._timing_started: Optional[float] = None
        self._timing_peaks: dict[str, float] = {}
        self._timing_count = 0
        self._timing_faults = 0
        self._last_tick_return_s: Optional[float] = None

    # Episode boundaries.

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset episode state, then keep teleoperating until recording starts.

        A manual episode boundary leaves the arms where they are, so the device
        is told to keep each grip anchored across it. Any other reset re-anchors
        both grips.
        """
        try:
            self._ensure_keyboard()
            controller = self._controller()
            preserve = self._preserve_reference and not (
                self.env.unwrapped.will_reset_to_configured_qpos(options)
            )
            self._preserve_reference = False
            if preserve:
                controller.preserve_reference()
            else:
                self._previous_buttons = None
            self._recording = False
            obs, info = self.env.reset(seed=seed, options=options)
            skip_wait = bool((options or {}).get("skip_wait_for_start", False))
            if self.config.wait_for_record_button and not skip_wait:
                while not self._recording:
                    obs, _, _, _, info = self._tick(preview=True)
            return obs, self._decorate(info, None, False)
        except BaseException:
            self.close()
            raise

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance the environment and apply the episode transitions it caused.

        A control path that throws leaves the arms holding their last target, so
        the device is latched and the stack released rather than left engaged
        with a stale command.
        """
        started = time.monotonic()
        try:
            result = self.env.step(action)
        except BaseException:
            self._abort_control()
            raise
        return self._finish(*result, started=started)

    def _abort_control(self) -> None:
        """Hold both arms and release the stack after a control-path failure.

        A failure anywhere under this wrapper leaves the arms holding their
        last target, so the device latches a fault that keeps them there until
        both grips are released.
        """
        try:
            self._controller().hold_until_released("control_error")
        except Exception:  # pragma: no cover - hardware failure path
            self._logger.exception("YAM PICO control-error latch failed")
        try:
            self.env.unwrapped.runtime.emergency_hold()
        except Exception:  # pragma: no cover - hardware failure path
            self._logger.exception("YAM PICO emergency hold failed")
        try:
            self.close()
        except Exception:  # pragma: no cover - cleanup path
            self._logger.exception("YAM PICO cleanup failed after control error")

    def close(self) -> None:
        """Close the keyboard and the environment, reporting the first failure."""
        if self._closed:
            return
        errors: list[Exception] = []
        keyboard, self._keyboard = self._keyboard, None
        if keyboard is not None:
            try:
                keyboard.close()
            except Exception as exc:  # pragma: no cover - cleanup path
                errors.append(exc)
        try:
            self.env.close()
        except Exception as exc:  # pragma: no cover - cleanup path
            errors.append(exc)
        self._closed = not errors
        if errors:
            raise RuntimeError("YAM PICO episode cleanup failed") from errors[0]

    # One control tick.

    def _tick(self, *, preview: bool) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Command the arms without consuming an episode step."""
        started = time.monotonic()
        if preview:
            result = self._preview_step()
        else:  # pragma: no cover - callers use step()
            result = self.env.step(self._zero_action())
        return self._finish(*result, started=started)

    def _preview_step(self) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Drive the arms through the device, bypassing the episode counter."""
        sample = self._device().read(self.env, self._zero_action())
        if sample.action is None:
            raise RuntimeError("yam_pico produced no target while waiting to record")
        obs, info = self.env.unwrapped.teleop_tick(sample.action)
        return obs, 0.0, False, False, {**info, **sample.info}

    def _finish(
        self,
        obs: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        *,
        started: float,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Turn one frame's controller state into the next episode state."""
        controller = self._controller()
        if info.get("action_rejected") is not None:
            controller.hold_until_released(f"runtime:{info['action_rejected']}")
        record, discard = self._events(info)
        fault = controller.fault
        frame_fault = fault or info.get("yam_pico_fault")
        event: Optional[str] = None
        reset_record = False
        manual_done = False
        if frame_fault is not None or discard or truncated:
            reset_record = self._recording
            self._recording = False
            event = "abort" if reset_record else None
            reward = 0.0
        elif record:
            if self._recording:
                event = "end_success"
                reward, terminated, manual_done = 1.0, True, True
                self._preserve_reference = True
                controller.preserve_reference()
            else:
                self._recording = True
                reset_record = True
                event = "start"

        info["yam_pico_fault"] = frame_fault
        # The collector closes the previous streaming shard on record_reset.
        # Mark an explicit discard before that close so it drops the shard.
        info["episode_discarded"] = bool(discard and reset_record)
        info["intervene_action"] = np.asarray(
            info["accepted_action"], dtype=np.float32
        ).copy()
        info["intervene_flag"] = np.ones(1, dtype=bool)
        info["intervened"] = True
        info["manual_done"] = manual_done
        info["success"] = manual_done
        for side in ("left", "right"):
            info[side] = (
                fault is None
                and info.get(f"{side}_ik_fault") is None
                and bool(info.get(f"{side}_pico_active", False))
            )
        info["pico_active"] = info["left"] or info["right"]
        if (terminated or truncated) and not manual_done:
            controller.hold_until_released("episode_ended")
            self.env.unwrapped.runtime.emergency_hold()
        if self.config.log_control_timing:
            self._report_timing(info, started)
        return (
            obs,
            reward,
            terminated,
            truncated,
            self._decorate(info, event, reset_record),
        )

    # Operator edges.

    def _events(self, info: Mapping[str, Any]) -> tuple[bool, bool]:
        """Return the record and discard edges this frame produced."""
        buttons = (
            bool(info.get("pico_record", False)),
            bool(info.get("pico_discard", False)),
        )
        keyboard = self._poll_keyboard()
        now = time.monotonic()
        previous = buttons if self._previous_buttons is None else self._previous_buttons
        edges = []
        for index, key in enumerate(("record", "discard")):
            edge = (buttons[index] and not previous[index]) or key in keyboard
            edge = (
                edge and now - self._last_edges[index] >= self.config.button_debounce_s
            )
            if edge:
                self._last_edges[index] = now
            edges.append(edge)
        self._previous_buttons = buttons
        return edges[0], edges[1]

    def _poll_keyboard(self) -> set[str]:
        """Return the episode keys pressed since the previous frame."""
        if self._keyboard is None:
            return set()
        return {
            _KEYBOARD_KEYS[key]
            for key in self._keyboard.pop_pressed_keys()
            if key in _KEYBOARD_KEYS
        }

    def _decorate(
        self, info: dict[str, Any], event: Optional[str], record_reset: bool
    ) -> dict[str, Any]:
        """Publish the collection state this frame ends in."""
        phase = "rec" if self._recording else "pre"
        info.update(
            {
                "pre_record": not self._recording,
                "record_reset": record_reset,
                "keyboard_phase": phase,
                "keyboard_event": event,
                "episode_control_phase": phase,
                "episode_control_event": event,
                "segment_advance": False,
            }
        )
        return info

    # Wiring.

    def _device(self) -> Any:
        """Return the composed device the intervention wrapper drives."""
        return self.env.get_wrapper_attr("device")

    def _controller(self) -> YamPico:
        """Return the PICO pair that owns the grips and the hold latch.

        The intervention wrapper holds the composed device, which flattens all
        configured devices into one action vector. Episode control is specific
        to the PICO pair inside it.
        """
        for device in self._device().group.devices:
            if isinstance(device, YamPico):
                return device
        raise RuntimeError(
            "YamPicoEpisode needs a yam_pico device under the intervention wrapper"
        )

    def _zero_action(self) -> np.ndarray:
        """Return an action the device's own parts overwrite entirely."""
        return np.zeros(self.env.action_space.shape, dtype=np.float32)

    def _ensure_keyboard(self) -> None:
        if self._keyboard is not None or not self.config.keyboard_enabled:
            return
        self._keyboard = KeyboardListener()

    # Timing.

    def _report_timing(self, info: dict[str, Any], started: float) -> None:
        """Report one-second peak timings; these do not change control targets."""
        finished = time.monotonic()
        info["yam_between_ticks_s"] = (
            started - self._last_tick_return_s
            if self._last_tick_return_s is not None
            else 0.0
        )
        info["yam_tick_s"] = finished - started
        if self._timing_started is None:
            self._timing_started = started
        self._timing_count += 1
        self._timing_faults += int(info.get("yam_pico_fault") is not None)
        metrics = {
            "gap": info.get("yam_command_interval_s", 0.0),
            "ik": sum(
                info.get(f"{side}_ik_elapsed_s", 0.0) for side in ("left", "right")
            ),
            "pace": info.get("yam_pace_s", 0.0),
            "send": info.get("yam_command_s", 0.0),
            "state": info.get("yam_state_read_s", 0.0),
            "cameras": info.get("yam_camera_read_s", 0.0),
            "outside": info["yam_between_ticks_s"],
        }
        for name, value in metrics.items():
            self._timing_peaks[name] = max(self._timing_peaks.get(name, 0.0), value)
        elapsed = finished - self._timing_started
        if elapsed >= 1.0:
            self._logger.info(
                "YAM VR timing: ticks=%d, rate=%.1f Hz, fault_ticks=%d; peak_ms %s",
                self._timing_count,
                self._timing_count / elapsed,
                self._timing_faults,
                " ".join(
                    f"{key}={value * 1000:.1f}"
                    for key, value in self._timing_peaks.items()
                ),
            )
            self._timing_started = finished
            self._timing_count = self._timing_faults = 0
            self._timing_peaks.clear()
        self._last_tick_return_s = finished
