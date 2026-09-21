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

"""Arbitrate between policy actions and operator teleoperation input."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from .trigger import RELEASE, TAKEOVER, InterventionTrigger


@dataclass
class TeleopSample:
    """Action sample produced by a teleoperation device.

    Attributes:
        action: Operator command in the environment action space, or ``None``
            when no usable reading is available.
        active: Whether the operator currently holds control.
        apply_when_inactive: Whether ``action`` contains passive state, such as
            a held dexterous-hand pose, that must still reach the environment.
        info: Device state to merge into the step information.
    """

    action: Optional[np.ndarray]
    active: bool
    apply_when_inactive: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class TeleopDevice(ABC):
    """Base interface for operator input expressed as environment actions."""

    #: Duration for which control remains with the operator after an active sample.
    #: Use zero for devices that report an explicit held state.
    timeout: float = 0.5

    @abstractmethod
    def read(self, env: gym.Env, policy_action: np.ndarray) -> TeleopSample:
        """Return what the operator is asking for, in ``env``'s action space."""

    def before_reset(self, env: gym.Env, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare the device and reset arguments before environment reset."""
        return kwargs

    def reset(self, env: gym.Env) -> None:
        """Re-sync with the robot after an episode reset."""

    def after_reset(self, env: gym.Env) -> None:
        """Run once the reset is over, whether or not it succeeded."""

    def before_step(self, env: gym.Env) -> None:
        """Hook that runs before the wrapped env steps."""

    def prepare_intervention(self, env: gym.Env) -> None:
        """Align and hold the device before the operator's entry buffer."""

    def on_intervention_start(self, env: gym.Env) -> None:
        """Hand control to the operator after the entry buffer."""

    def on_intervention_end(self, env: gym.Env) -> None:
        """Release device state when policy control resumes."""

    def on_action_chunk_begin(self) -> None:
        """Let go of anything held only until the next chunk of actions."""

    def get_hold_action(
        self, env: gym.Env, fallback_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return an action that holds the robot during a skipped chunk.

        Raises:
            AttributeError: If this device commands deltas, where a zero motion
                is already the action that holds a robot still.
        """
        raise AttributeError(
            f"{type(self).__name__} commands deltas, so it has no pose to hold."
        )

    def park(self, env: gym.Env, park_env: Any) -> None:
        """Park the wrapped environment."""
        park_env()

    def close(self) -> None:
        """Release the device."""


class TeleopIntervention(gym.Wrapper):
    """Replace the policy's action while the operator is driving.

    A regular :class:`gymnasium.Wrapper` is required because intervention
    metadata is written to ``info`` alongside the selected action.

    Args:
        env: The environment to wrap.
        device: The teleop device to read.
        mark_flag: Also write ``info["intervene_flag"]`` when overriding. Some
            dataset formats key on the flag rather than on the action.
        mode: ``"activity"`` preserves motion-triggered arbitration;
            ``"explicit"`` latches ownership through trigger events.
        trigger: Event source for explicit control ownership. It is required
            when ``mode`` is ``"explicit"``.
        buffer_seconds: In explicit mode, hold the robot for this long before
            handing control to the operator and after release is requested.
            Policy actions resume at the next action chunk after release.
    """

    def __init__(
        self,
        env: gym.Env,
        device: TeleopDevice,
        mark_flag: bool = False,
        mode: str = "activity",
        trigger: Optional[InterventionTrigger] = None,
        buffer_seconds: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.device = device
        self.mark_flag = mark_flag
        self.mode = str(mode).lower()
        if self.mode not in {"activity", "explicit"}:
            raise ValueError("mode must be 'activity' or 'explicit'")
        if self.mode == "explicit" and trigger is None:
            raise ValueError("explicit teleop intervention requires a trigger")
        if self.mode == "activity" and trigger is not None:
            raise ValueError("activity teleop intervention does not use a trigger")
        self.trigger = trigger
        self.keyboard_listener = getattr(trigger, "listener", None)
        self.buffer_seconds = float(buffer_seconds)
        if not np.isfinite(self.buffer_seconds) or self.buffer_seconds < 0:
            raise ValueError("buffer_seconds must be finite and nonnegative")
        self._last_active: float = -float("inf")
        self._takeover_active = False
        self._phase = "idle"
        self._phase_started = -float("inf")
        self._closed = False
        self._faulted = False
        configure = getattr(self.device, "configure_takeover", None)
        if callable(configure):
            configure(self.mode)

    @property
    def intervening(self) -> bool:
        """Whether the operator currently holds control."""
        if self.mode == "explicit":
            return self._takeover_active
        return time.monotonic() - self._last_active < self.device.timeout

    def _set_takeover_active(self, active: bool) -> None:
        """Update ownership in both the wrapper and composed device."""
        self._takeover_active = active
        setter = getattr(self.device, "set_takeover_active", None)
        if callable(setter):
            setter(active)

    def _poll_trigger(self) -> None:
        """Apply queued ownership events before reading the teleop device."""
        if self.trigger is None:
            return
        for event in self.trigger.poll():
            if event == TAKEOVER and not self._takeover_active:
                self._set_takeover_active(True)
                self._phase = "preparing"
            elif event == RELEASE and self._takeover_active:
                self._end_intervention()
                self._set_takeover_active(False)
                self._phase = "exiting"
                self._phase_started = time.monotonic()

    def _end_intervention(self) -> None:
        """Latch a handover failure until an explicit environment reset."""
        try:
            self.device.on_intervention_end(self)
        except Exception:
            self._faulted = True
            raise

    @property
    def manual_start_hold_seconds(self) -> float:
        """Return the handover buffer configured by the teleop device."""
        return float(getattr(self.device, "manual_start_hold_seconds", 0.0))

    def release_for_manual(self) -> None:
        """Release the teleop device after the manual-control buffer."""
        self.device.release_for_manual(self)

    def hold_for_reset(self) -> None:
        """Hold the teleop device before the wrapped robot is reset."""
        self.device.hold_for_reset(self)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and synchronize the device afterward."""
        kwargs = self.device.before_reset(self, kwargs)
        try:
            if self._takeover_active:
                self._end_intervention()
            self._set_takeover_active(False)
            if self.trigger is not None:
                self.trigger.reset()
            result = self.env.reset(**kwargs)
            self._last_active = -float("inf")
            self._phase = "idle"
            self._phase_started = -float("inf")
            self.device.reset(self)
            self._faulted = False
            return result
        finally:
            self.device.after_reset(self)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step with the operator's action when they are driving."""
        if self._closed or self._faulted:
            raise RuntimeError(
                "Teleop intervention is closed or needs reset after failure"
            )
        self.device.before_step(self)
        self._poll_trigger()
        if (
            self._phase == "entering"
            and time.monotonic() - self._phase_started >= self.buffer_seconds
        ):
            try:
                self.device.on_intervention_start(self)
            except Exception:
                self._faulted = True
                self._end_intervention()
                raise
            self._phase = "active"
        sample = self.device.read(self, action)

        if self.mode == "explicit":
            applied, overridden, buffered = self._explicit_step(action, sample)
        else:
            applied, overridden, buffered = self._activity_step(action, sample)

        obs, reward, terminated, truncated, info = self.env.step(applied)

        if self._phase == "preparing":
            # Stop the follower before a potentially blocking leader alignment.
            try:
                self.device.prepare_intervention(self)
            except Exception:
                self._faulted = True
                self._end_intervention()
                raise
            self._phase = "entering"
            self._phase_started = time.monotonic()

        if overridden:
            info["intervene_action"] = applied
            if self.mark_flag:
                info["intervene_flag"] = np.ones(1)
        info.update(sample.info)
        if buffered:
            info["intervene_buffer"] = np.ones(1, dtype=bool)

        return obs, reward, terminated, truncated, info

    def _activity_step(
        self, action: np.ndarray, sample: TeleopSample
    ) -> tuple[np.ndarray, bool, bool]:
        """Apply the established activity-triggered arbitration behavior."""
        if sample.action is None:
            return action, False, False
        if sample.active:
            self._last_active = time.monotonic()
            return sample.action, True, False
        if self.intervening:
            return sample.action, True, False
        if sample.apply_when_inactive:
            return sample.action, False, False
        return action, False, False

    def _explicit_step(
        self, action: np.ndarray, sample: TeleopSample
    ) -> tuple[np.ndarray, bool, bool]:
        """Run explicit takeover with hold buffers around operator control."""
        now = time.monotonic()
        if self._phase in {"preparing", "entering"}:
            return self._hold_action(action), True, True
        if self._phase == "exiting":
            if now - self._phase_started < self.buffer_seconds:
                return self._hold_action(action), True, True
            self._phase = "waiting_for_policy"

        if self._phase == "waiting_for_policy":
            # No intervention flag: EnvWorker must request fresh inference at
            # the next chunk instead of building another dummy hold chunk.
            return self._hold_action(action), False, True

        if self._phase == "active" and self._takeover_active:
            self._last_active = now
            if sample.action is None:
                return self._hold_action(action), True, True
            return sample.action, True, False

        if sample.apply_when_inactive and sample.action is not None:
            return sample.action, False, False
        return action, False, False

    def _hold_action(self, fallback_action: np.ndarray) -> np.ndarray:
        """Read an absolute hold pose from the stateful teleop device."""
        try:
            return self.device.get_hold_action(self, fallback_action)
        except AttributeError as error:
            raise ValueError(
                "explicit teleop intervention requires a device that can hold "
                "the robot's current pose"
            ) from error

    def on_action_chunk_begin(self) -> None:
        """Notify the device that a new policy action chunk has started."""
        if self._phase == "waiting_for_policy":
            self._phase = "idle"
        self.device.on_action_chunk_begin()

    def get_hold_action(
        self, fallback_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return an action that holds the robot during a skipped chunk."""
        return self.device.get_hold_action(self, fallback_action)

    def park(self) -> None:
        """Park the teleop rig and wrapped robot as one operation."""
        park_env = self.env.get_wrapper_attr("park")
        self.device.park(self, park_env)

    def close(self) -> None:
        """Release the device, then the wrapped env."""
        if self._closed:
            return
        self._closed = True
        try:
            try:
                if self._takeover_active:
                    self.device.on_intervention_end(self)
                self._set_takeover_active(False)
            finally:
                try:
                    if self.trigger is not None:
                        self.trigger.close()
                finally:
                    self.device.close()
        finally:
            super().close()
