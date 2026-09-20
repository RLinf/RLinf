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

"""Adapt composed teleoperation action parts to an environment action vector."""

from __future__ import annotations

import threading
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np

from rlinf.robotics.parts.teleop import TeleopGroup
from rlinf.utils.logging import get_logger

from .intervention import TeleopDevice, TeleopSample


class ComposedTeleop(TeleopDevice):
    """Flatten a teleoperation group into an environment action vector.

    Args:
        group: The composed devices.
        layout: Slice occupied by each named action part. Unset parts retain
            the policy action.
        timeout: How long the operator keeps control after their last active
            reading. Zero for rigs whose devices say exactly when they are
            driving.
        streamer: Optional direct command stream that runs faster than
            ``env.step``.
    """

    def __init__(
        self,
        group: TeleopGroup,
        layout: Mapping[str, slice],
        timeout: Optional[float] = None,
        streamer: Optional[Any] = None,
    ) -> None:
        unknown = set(group.parts) - set(layout)
        if unknown:
            raise ValueError(
                f"The teleop group drives {sorted(unknown)}, which this env's "
                f"action layout does not have. It has {sorted(layout)}."
            )
        self.group = group
        self.layout = dict(layout)
        self.streamer = streamer
        self._takeover_mode = "activity"
        self._takeover_active = False
        self._reset_thread: Optional[threading.Thread] = None
        self._reset_errors: list[BaseException] = []
        self._reset_context: dict[str, Any] = {}
        self._reset_completed = False
        if streamer is not None:
            unknown = set(getattr(streamer, "DELIVERS", ())) - set(self.layout)
            if unknown:
                raise ValueError(
                    f"{type(streamer).__name__} says it delivers "
                    f"{sorted(unknown)}, which this env's action layout does "
                    f"not have. It has {sorted(self.layout)}."
                )
        if timeout is not None:
            self.timeout = timeout

    #: Environment getters that provide optional device context.
    CONTEXT_GETTERS = (
        ("tcp_pose", "get_tcp_pose"),
        ("action_scale", "get_action_scale"),
        ("joint_positions", "get_joint_positions"),
        ("gripper_open", "get_gripper_open"),
        ("gripper_position", "get_gripper_position"),
        ("hand_reset_pose", "get_hand_reset_pose"),
        ("reset_joint_positions", "get_reset_joint_positions"),
        ("reset_gripper_position", "get_reset_gripper_position"),
        ("reset_duration", "get_reset_duration"),
        ("reset_joint_speed", "get_reset_joint_speed"),
        ("park_joint_positions", "get_park_joint_positions"),
        ("park_gripper_position", "get_park_gripper_position"),
        ("park_duration", "get_park_duration"),
    )

    @classmethod
    def context_from(cls, env: gym.Env) -> dict[str, Any]:
        """Collect the context an environment exposes to its devices."""
        context: dict[str, Any] = {}
        for key, getter in cls.CONTEXT_GETTERS:
            try:
                value = env.get_wrapper_attr(getter)
            except AttributeError:
                continue
            if callable(value):
                value = value()
            if value is not None:
                context[key] = value
        return context

    def before_reset(self, env: gym.Env, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Pause streaming and begin device reset alongside the robot reset."""
        if self._reset_thread is not None:
            raise RuntimeError("A teleop reset is already in progress")
        if self.streamer is not None:
            kwargs = self.streamer.before_reset(env, kwargs)
        self._reset_context = self.context_from(env)
        self._reset_errors = []
        self._reset_completed = False
        started = threading.Event()

        def prepare() -> None:
            started.set()
            try:
                self.group.prepare_reset(self._reset_context)
            except BaseException as error:  # noqa: BLE001 - re-raised on reset thread
                self._reset_errors.append(error)

        self._reset_thread = threading.Thread(
            target=prepare,
            name="teleop-reset",
            daemon=True,
        )
        self._reset_thread.start()
        started.wait()
        return kwargs

    def reset(self, env: gym.Env) -> None:
        """Finish device reset, then hand the devices back to the operator."""
        self._takeover_active = False
        self._join_reset()
        if self._reset_errors:
            raise self._reset_errors[-1]
        self.group.reset(self.context_from(env))
        if self.streamer is not None:
            self.streamer.reset(env)
        self._reset_completed = True

    def after_reset(self, env: gym.Env) -> None:
        """Release incomplete reset state and resume the streamer."""
        self._join_reset()
        if not self._reset_completed:
            try:
                self.group.abort_reset(self._reset_context)
            except BaseException:  # noqa: BLE001 - preserve the reset failure
                get_logger().exception("Failed to release teleop devices after reset")
            if self._reset_errors:
                get_logger().error(
                    "Teleop reset preparation failed: %s", self._reset_errors[-1]
                )
        self._reset_thread = None
        self._reset_context = {}
        self._reset_errors = []
        self._reset_completed = False
        if self.streamer is not None:
            self.streamer.after_reset(env)

    def _join_reset(self) -> None:
        """Wait for device reset preparation if one is running."""
        if self._reset_thread is not None:
            self._reset_thread.join()

    def park(self, env: gym.Env, park_env: Any) -> None:
        """Move teleop devices and the robot to park concurrently."""
        if self._reset_thread is not None:
            raise RuntimeError("Cannot park while a teleop reset is in progress")
        context = self.context_from(env)
        required = {
            "park_joint_positions",
            "park_gripper_position",
            "park_duration",
            "reset_joint_speed",
        }
        missing = required - context.keys()
        if missing:
            raise RuntimeError(
                f"Cannot park the teleop rig; missing context: {sorted(missing)}"
            )
        context.update(
            {
                "reset_joint_positions": context["park_joint_positions"],
                "reset_gripper_position": context["park_gripper_position"],
                "reset_duration": context["park_duration"],
            }
        )
        errors: list[BaseException] = []

        def prepare() -> None:
            try:
                self.group.prepare_reset(context)
            except BaseException as error:  # noqa: BLE001 - re-raised below
                errors.append(error)

        thread = threading.Thread(target=prepare, name="teleop-park", daemon=True)
        thread.start()
        completed = False
        try:
            park_env()
            thread.join()
            if errors:
                raise errors[-1]
            self.group.reset(context)
            completed = True
        finally:
            thread.join()
            if not completed:
                self.group.abort_reset(context)

    def before_step(self, env: gym.Env) -> None:
        """Start the streamer when its prerequisites are satisfied."""
        if self.streamer is not None:
            self.streamer.before_step(env)

    def _write(
        self, env: gym.Env, policy_action: np.ndarray, parts: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Write named action parts into a copy of the policy action."""
        action = np.array(policy_action, dtype=np.float64, copy=True)
        clipped = set(self.group.clipped_parts) & set(parts)
        bounds = None
        if clipped:
            bounds = (
                env.action_space.low.reshape(-1),
                env.action_space.high.reshape(-1),
            )
        for name, value in parts.items():
            where = self.layout[name]
            value = np.asarray(value, dtype=np.float64)
            if name in clipped:
                value = np.clip(value, bounds[0][where], bounds[1][where])
            action[where] = value
        return action

    def read(self, env: gym.Env, policy_action: np.ndarray) -> TeleopSample:
        """Read every device, then write each part into the action."""
        parts, driving, info = self.group.action(self.context_from(env))
        if not parts:
            return TeleopSample(action=None, active=False, info=info)
        apply_when_inactive = False
        explicitly_driving = self._takeover_mode == "explicit" and self._takeover_active
        if not driving and not explicitly_driving:
            idle = set(self.group.idle_parts)
            parts = {name: value for name, value in parts.items() if name in idle}
            apply_when_inactive = bool(parts)
            if not parts:
                return TeleopSample(action=None, active=False, info=info)
        if self.streamer is not None and self.streamer.streaming:
            # Record parts delivered outside env.step for dataset consumers.
            info = {**info, "streamed_parts": list(self.streamer.DELIVERS)}
        return TeleopSample(
            action=self._write(env, policy_action, parts),
            active=driving or explicitly_driving,
            apply_when_inactive=apply_when_inactive,
            info=info,
        )

    def configure_takeover(self, mode: str) -> None:
        """Select how this composed device reports operator control."""
        if mode not in {"activity", "explicit"}:
            raise ValueError(
                f"Unsupported teleop intervention mode {mode!r}; expected "
                "'activity' or 'explicit'."
            )
        self._takeover_mode = mode
        self._takeover_active = False

    def set_takeover_active(self, active: bool) -> None:
        """Set the latched state used by explicit takeover mode."""
        self._takeover_active = bool(active)

    def prepare_intervention(self, env: gym.Env) -> None:
        """Align and hold all devices before the entry buffer."""
        self.group.prepare_intervention(self.context_from(env))

    def on_intervention_start(self, env: gym.Env) -> None:
        """Hand the prepared devices to the operator."""
        self.group.on_intervention_start(self.context_from(env))

    def on_intervention_end(self, env: gym.Env) -> None:
        """Release all devices after an operator takeover."""
        self.group.on_intervention_end(self.context_from(env))

    def get_hold_action(
        self, env: gym.Env, fallback_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return an action that holds absolute parts during a skipped chunk."""
        parts = self.group.hold(self.context_from(env))
        if not parts:
            raise AttributeError(
                "No device in this group commands an absolute pose, so none can "
                "hold the robot anywhere. A delta of zero already does that."
            )
        if fallback_action is None:
            fallback_action = np.zeros(env.action_space.shape, dtype=np.float32)
        action = self._write(env, np.asarray(fallback_action).reshape(-1), parts)
        return action.reshape(env.action_space.shape)

    def on_action_chunk_begin(self) -> None:
        """Tell the group a fresh chunk of policy actions starts here."""
        self.group.on_action_chunk_begin()

    def close(self) -> None:
        """Stop the stream, then release every device in the group."""
        try:
            self._join_reset()
            if self._reset_thread is not None and not self._reset_completed:
                self.group.abort_reset(self._reset_context)
        finally:
            try:
                if self.streamer is not None:
                    self.streamer.close()
            finally:
                self.group.disconnect()
