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

"""Devices that are more than a reading mapped onto an action.

Composition covers the rest: a device reports, a binding maps, and a group
merges, in :mod:`rlinf.robotics.teleop`. What is left here needs something
composition does not describe -- in this case a thread that pushes joint targets
to the controllers at roughly 1 kHz, outside ``env.step`` entirely.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from .streaming import TeleopStreamer


class DualGelloJointStream(TeleopStreamer):
    """The 1 kHz loop that pushes a pair of leader arms to the controllers.

    The action itself comes from composition, one leader per side. This exists
    because a leader arm tracked at the policy's step rate feels laggy to the
    operator, so its targets go straight to each controller; ``env.step`` then
    reads state and grippers but stops forwarding motion, and only one writer
    touches the motion queue.

    Args:
        left_arm: The left leader arm, already composed into the teleop group.
        right_arm: The right leader arm, likewise.
        gripper_enabled: Whether each arm's action carries a gripper channel.
        use_delta: Whether the env takes joint deltas or absolute targets.
        action_scale: Divisor turning a joint delta into a normalized action.
        direct_stream: Push targets from the device's own thread.
        stream_period: Seconds between pushes when streaming.
    """

    #: Both arms' joint targets go straight to the controllers.
    DELIVERS = ("left.arm", "right.arm")

    def __init__(
        self,
        left_arm: Any,
        right_arm: Any,
        gripper_enabled: bool = True,
        use_delta: bool = False,
        action_scale: float = 0.1,
        direct_stream: bool = False,
        stream_period: float = 0.001,
    ) -> None:
        super().__init__(period=stream_period, enabled=direct_stream)
        # The arms are the group's, not this object's. Opening a second reader
        # on the same serial port is what happens when a streamer builds its
        # own: two pollers competing for one port, and a per-entry port
        # override that only one of them has heard of.
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.gripper_enabled = gripper_enabled
        self.use_delta = use_delta
        self.action_scale = action_scale
        # Gripper commands are edge-triggered: an open/close RPC takes ~100 ms,
        # and repeating it every tick would starve the serial channel.
        self._last_open: list[Optional[bool]] = [None, None]

    def before_reset(self, env: gym.Env, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Skip the env's slew home; aligning to the leader arms replaces it.

        Homing first and then aligning would move the robot twice, and the
        second move starts from wherever the first one left it.
        """
        kwargs = super().before_reset(env, kwargs)
        options = dict(kwargs.get("options") or {})
        options.setdefault("skip_reset_to_home", True)
        kwargs["options"] = options
        return kwargs

    def _controllers(self, env: gym.Env):
        inner = env.unwrapped
        return getattr(inner, "_left_ctrl", None), getattr(inner, "_right_ctrl", None)

    def ready_to_stream(self, env: gym.Env) -> bool:
        """Both controllers must exist before a tick can command them."""
        return self._controllers(env) != (None, None)

    @property
    def _ready(self) -> bool:
        """Both leader arms have produced a reading."""
        return bool(self.left_arm.ready and self.right_arm.ready)

    def _joints(self) -> tuple:
        """Each arm's joint target and grip, through the part interface."""
        left = self.left_arm.get_observation()
        right = self.right_arm.get_observation()
        return (
            left["joint_position"],
            left["grip"],
            right["joint_position"],
            right["grip"],
        )

    def align(self, env: gym.Env) -> bool:
        """Move each follower onto its leader's current joint pose."""
        if not self._ready:
            return False
        left_ctrl, right_ctrl = self._controllers(env)
        if left_ctrl is None or right_ctrl is None:
            return False
        left_q, _, right_q, _ = self._joints()
        left_ctrl.reset_joint(np.asarray(left_q, dtype=np.float64).tolist()).wait()
        right_ctrl.reset_joint(np.asarray(right_q, dtype=np.float64).tolist()).wait()
        inner = env.unwrapped
        inner._left_state = left_ctrl.get_state().wait()[0]
        inner._right_state = right_ctrl.get_state().wait()[0]
        return True

    def stream_once(self, env: gym.Env) -> None:
        """Push one set of joint targets, and any gripper edge, to both arms."""
        if not self._ready:
            time.sleep(self._period)
            return
        left_ctrl, right_ctrl = self._controllers(env)
        if left_ctrl is None or right_ctrl is None:
            return

        left_q, left_g, right_q, right_g = self._joints()
        left_ctrl.move_joints(left_q.astype(np.float32)).wait()
        right_ctrl.move_joints(right_q.astype(np.float32)).wait()

        if not self.gripper_enabled:
            return
        for index, (ctrl, grip) in enumerate(
            zip((left_ctrl, right_ctrl), (left_g, right_g))
        ):
            is_open = grip.item() < 0.5
            if self._last_open[index] is None:
                self._last_open[index] = is_open
            elif is_open != self._last_open[index]:
                ctrl.open_gripper() if is_open else ctrl.close_gripper()
                self._last_open[index] = is_open

    def close(self) -> None:
        """Stop the loop. The arms belong to the group, which closes them."""
        super().close()
