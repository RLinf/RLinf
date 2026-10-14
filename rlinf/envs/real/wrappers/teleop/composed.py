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

from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np

from rlinf.robotics.parts.teleop import TeleopGroup

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
        ("hand_reset_pose", "get_hand_reset_pose"),
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
                context[key] = value()
        return context

    def before_reset(self, env: gym.Env, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Pause the streamer and allow it to adjust reset arguments."""
        if self.streamer is not None:
            return self.streamer.before_reset(env, kwargs)
        return kwargs

    def reset(self, env: gym.Env) -> None:
        """Reset devices from the current robot state and realign the streamer."""
        self.group.reset(self.context_from(env))
        if self.streamer is not None:
            self.streamer.reset(env)

    def after_reset(self, env: gym.Env) -> None:
        """Resume the streamer after reset cleanup."""
        if self.streamer is not None:
            self.streamer.after_reset(env)

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
        if self.streamer is not None:
            self.streamer.close()
        self.group.disconnect()
