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

"""Environment adapters for teleoperation devices with direct command streams."""

from __future__ import annotations

import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from rlinf.robotics.parts.teleop import GelloJoint

from .streaming import TeleopStreamer


class DualGelloJointStream(TeleopStreamer):
    """Stream two GELLO leader arms directly to follower controllers.

    Args:
        left_arm: The left leader arm, already composed into the teleop group.
        right_arm: The right leader arm, likewise.
        gripper_enabled: Whether each arm's action carries a gripper channel.
        use_delta: Whether the env takes joint deltas or absolute targets.
        action_scale: Divisor turning a joint delta into a normalized action.
        direct_stream: Whether to send targets from the streaming thread.
        stream_period: Seconds between pushes when streaming.
    """

    #: Both arms' joint targets go straight to the controllers.
    DELIVERS = ("left.arm", "right.arm")

    def __init__(
        self,
        left_arm: "GelloJoint",
        right_arm: "GelloJoint",
        gripper_enabled: bool = True,
        use_delta: bool = False,
        action_scale: float = 0.1,
        direct_stream: bool = False,
        stream_period: float = 0.001,
    ) -> None:
        super().__init__(period=stream_period, enabled=direct_stream)
        # Reuse the group's readers to keep one owner per serial port.
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.gripper_enabled = gripper_enabled
        self.use_delta = use_delta
        self.action_scale = action_scale
        # Send slow gripper RPCs only when the requested state changes.
        self._last_open: list[Optional[bool]] = [None, None]

    def before_reset(self, env: gym.Env, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Replace the environment home motion with leader-arm alignment."""
        kwargs = super().before_reset(env, kwargs)
        options = dict(kwargs.get("options") or {})
        options.setdefault("skip_reset_to_home", True)
        kwargs["options"] = options
        return kwargs

    def _arms(self, env: gym.Env) -> tuple[Optional[Any], Optional[Any]]:
        inner = env.unwrapped
        return getattr(inner, "_left_arm", None), getattr(inner, "_right_arm", None)

    def _hands(self, env: gym.Env) -> tuple[Optional[Any], Optional[Any]]:
        inner = env.unwrapped
        return getattr(inner, "_left_hand", None), getattr(inner, "_right_hand", None)

    @staticmethod
    def _run_both(env: gym.Env, left: Any, right: Any) -> tuple[Any, Any]:
        """Run paired controller calls through the env's arm queues."""
        run = getattr(env.unwrapped, "_run_arm_calls", None)
        if callable(run):
            return run(left, right)
        return left(), right()

    @staticmethod
    def _submit_one(env: gym.Env, arm: int, call: Any) -> None:
        """Queue a gripper edge without delaying the stream."""
        submit = getattr(env.unwrapped, "_submit_arm_call", None)
        if callable(submit):
            submit(arm, call)
        else:
            call()

    def ready_to_stream(self, env: gym.Env) -> bool:
        """Return whether both follower controllers are available."""
        return self._arms(env) != (None, None)

    @property
    def _ready(self) -> bool:
        """Return whether both leader arms have produced a reading."""
        return bool(self.left_arm.ready and self.right_arm.ready)

    def _joints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read joint and gripper targets from both leader arms."""
        left = self.left_arm.get_observation()
        right = self.right_arm.get_observation()
        return (
            left["joint_position"],
            left["grip"],
            right["joint_position"],
            right["grip"],
        )

    def align(self, env: gym.Env) -> bool:
        """Move both followers to their leaders' current joint poses."""
        if not self._ready:
            return False
        left_arm, right_arm = self._arms(env)
        if left_arm is None or right_arm is None:
            return False
        left_q, _, right_q, _ = self._joints()
        self._run_both(
            env,
            lambda: left_arm.reset_joint(np.asarray(left_q, dtype=np.float64).tolist()),
            lambda: right_arm.reset_joint(
                np.asarray(right_q, dtype=np.float64).tolist()
            ),
        )
        inner = env.unwrapped
        inner._left_state, inner._right_state = self._run_both(
            env,
            left_arm.get_state,
            right_arm.get_state,
        )
        return True

    def stream_once(self, env: gym.Env) -> None:
        """Push one set of joint targets, and any gripper edge, to both arms."""
        if not self._ready:
            time.sleep(self._period)
            return
        left_arm, right_arm = self._arms(env)
        if left_arm is None or right_arm is None:
            return

        left_q, left_g, right_q, right_g = self._joints()
        self._run_both(
            env,
            lambda: left_arm.send_action({"joint_position": left_q.astype(np.float32)}),
            lambda: right_arm.send_action(
                {"joint_position": right_q.astype(np.float32)}
            ),
        )

        if not self.gripper_enabled:
            return
        left_hand, right_hand = self._hands(env)
        if left_hand is None or right_hand is None:
            return
        for index, (hand, grip) in enumerate(
            zip((left_hand, right_hand), (left_g, right_g))
        ):
            is_open = grip.item() < 0.5
            if self._last_open[index] is None:
                self._last_open[index] = is_open
            elif is_open != self._last_open[index]:
                call = hand.open if is_open else hand.close
                self._submit_one(env, index, call)
                self._last_open[index] = is_open

    def close(self) -> None:
        """Stop the stream without closing the group-owned leader arms."""
        super().close()
