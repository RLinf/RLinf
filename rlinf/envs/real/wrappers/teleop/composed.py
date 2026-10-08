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
import time
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
        self._reset_thread: Optional[threading.Thread] = None
        self._reset_errors: list[BaseException] = []
        self._reset_context: dict[str, Any] = {}
        self._reset_completed = False
        self._reset_start_at: Optional[float] = None
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

    def context_from(
        self, env: gym.Env, keys: Optional[set[str] | frozenset[str]] = None
    ) -> dict[str, Any]:
        """Collect context requested by the composed device group."""
        context: dict[str, Any] = {}
        for key in self.group.context_keys if keys is None else keys:
            getter = f"get_{key}"
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
        # A device that moves during reset declares the context it needs.  The
        # wrapper only coordinates that abstract capability; it does not know
        # which hardware owns the fields or how many axes it has.
        reset_context_keys = self.group.synchronized_reset_context_keys
        if reset_context_keys:
            missing = reset_context_keys - self._reset_context.keys()
            if missing:
                raise ValueError(
                    f"Teleop reset requires context fields {sorted(missing)}, "
                    f"but this environment provides "
                    f"{sorted(self._reset_context)}."
                )
            # The reset may cross a Ray RPC boundary.  Use a shared monotonic
            # deadline instead of threading.Barrier, which cannot be
            # serialized into the remote arm worker.
            self._reset_start_at = time.monotonic() + 0.25
            self._reset_context["reset_start_at"] = self._reset_start_at
            options = dict(kwargs.get("options") or {})
            options["_teleop_reset_start_at"] = self._reset_start_at
            kwargs["options"] = options
        else:
            self._reset_start_at = None
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
        self._reset_start_at = None
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
        if self.streamer is not None:
            before_park = getattr(self.streamer, "before_park", None)
            if callable(before_park):
                before_park(env)
        context = self.context_from(env)
        errors: list[BaseException] = []
        park_context_keys = self.group.park_context_keys
        provided_park_keys = park_context_keys & context.keys()
        if provided_park_keys and provided_park_keys != park_context_keys:
            missing = sorted(park_context_keys - provided_park_keys)
            raise ValueError(f"Teleop park context is incomplete; missing {missing}.")
        has_configured_target = bool(park_context_keys) and bool(
            park_context_keys <= context.keys()
        )
        # Keep the other declared context fields as well.  A park hook may
        # share a device setting with reset (for example, its motion speed),
        # while the capability keys above decide whether a park target exists.
        park_context = dict(context)

        # Devices declare their own park context and lifecycle semantics. The
        # wrapper only coordinates their hold/park/reset calls with the env.
        completed = False
        reset_prepared = False
        thread: Optional[threading.Thread] = None
        try:
            if has_configured_target:
                # Mark this before calling the group: one device can acquire a
                # hold and a later device can fail during the same call.
                reset_prepared = True
                self.group.hold_for_reset(park_context)

            def prepare() -> None:
                try:
                    self.group.park(park_context)
                except BaseException as error:  # noqa: BLE001 - re-raised below
                    errors.append(error)

            thread = threading.Thread(target=prepare, name="teleop-park", daemon=True)
            thread.start()
            park_env()
            thread.join()
            if errors:
                raise errors[-1]
            if has_configured_target:
                self.group.reset(park_context)
            completed = True
        finally:
            if thread is not None:
                thread.join()
            if not completed and reset_prepared:
                # A device may have acquired a temporary park hold. Reuse the
                # existing reset rollback hook to release it on failure. Keep
                # the original park error if rollback itself fails.
                try:
                    self.group.abort_reset(park_context)
                except BaseException:  # noqa: BLE001 - preserve park failure
                    get_logger().exception("Failed to release teleop park hold")

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
        if not driving:
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
            active=driving,
            apply_when_inactive=apply_when_inactive,
            info=info,
        )

    @property
    def manual_start_hold_seconds(self) -> float:
        """Return the handover buffer required by the composed devices."""
        return self.group.manual_start_hold_seconds

    def release_for_manual(self, env: gym.Env) -> None:
        """Release all devices after the manual-control handover buffer."""
        self.group.release_for_manual(self.context_from(env))

    def hold_for_reset(self, env: gym.Env) -> None:
        """Hold all devices before resetting or parking the robot."""
        self.group.hold_for_reset(self.context_from(env))

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
