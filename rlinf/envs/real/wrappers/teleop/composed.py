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

"""Presenting a composed teleop group as one device to the env.

:class:`~rlinf.robotics.teleop.group.TeleopGroup` produces action *parts*, named
after what they drive. An env takes one flat vector. This is the only place that
knows how to get from one to the other, and it is the only piece of the teleop
story that has to be on the env side at all.

The layout comes from the env, which built the action space in the first place.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np

from rlinf.robotics.teleop import TeleopGroup

from .intervention import TeleopDevice, TeleopSample


class ComposedTeleop(TeleopDevice):
    """A group of devices, flattened into this env's action.

    Args:
        group: The devices and bindings in play.
        layout: Where each named part sits in the action vector. Parts the
            group does not fill keep whatever the policy asked for.
        timeout: How long the operator keeps control after their last active
            reading. Zero for rigs whose devices say exactly when they are
            driving.
        streamer: A device that also commands the robot on its own thread,
            faster than ``env.step`` runs. Composition says what the action is;
            this says it is delivered a second way, so the two are separate.
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

    #: What a binding may ask about the robot, and the getter that answers it.
    CONTEXT_GETTERS = (
        ("tcp_pose", "get_tcp_pose"),
        ("action_scale", "get_action_scale"),
        ("joint_positions", "get_joint_positions"),
        ("gripper_open", "get_gripper_open"),
        ("hand_reset_pose", "get_hand_reset_pose"),
    )

    @classmethod
    def context_from(cls, env: gym.Env) -> dict[str, Any]:
        """Gather what bindings may ask about the robot they are driving.

        Each value is fetched only if the env offers it, so an env without
        joints costs nothing and a binding that needs them fails where it is
        used rather than here.
        """
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
        """Let a streamer quiet itself, and adjust the reset if it needs to."""
        if self.streamer is not None:
            return self.streamer.before_reset(env, kwargs)
        return kwargs

    def reset(self, env: gym.Env) -> None:
        """Let every binding drop what it held, and a streamer re-align.

        The context goes with it: a binding that commands an absolute pose has
        to resume from where the env just reset the robot to.
        """
        self.group.reset(self.context_from(env))
        if self.streamer is not None:
            self.streamer.reset(env)

    def after_reset(self, env: gym.Env) -> None:
        """Let a streamer resume, whether or not the reset succeeded."""
        if self.streamer is not None:
            self.streamer.after_reset(env)

    def before_step(self, env: gym.Env) -> None:
        """Give a streamer its chance to start."""
        if self.streamer is not None:
            self.streamer.before_step(env)

    def _write(
        self, env: gym.Env, policy_action: np.ndarray, parts: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Put each named part into a copy of the action the policy asked for."""
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
        if self.streamer is not None and self.streamer.streaming:
            # Recorded so a dataset can tell which parts of this action the
            # robot was already given, rather than dispatched by step().
            info = {**info, "streamed_parts": list(self.streamer.DELIVERS)}
        return TeleopSample(
            action=self._write(env, policy_action, parts), active=driving, info=info
        )

    def get_hold_action(
        self, env: gym.Env, fallback_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """The action that keeps the robot where it is.

        Used when a chunk of policy actions is skipped and the robot still has
        to be told something. Parts nobody holds keep ``fallback_action``.
        """
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
