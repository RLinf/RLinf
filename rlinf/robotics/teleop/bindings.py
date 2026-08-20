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

"""What each device means for the robot it drives."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .binding import TeleopAction, TeleopBinding
from .kinds import ActionKind


def jittered_grip(is_open: bool) -> np.ndarray:
    """An open or close command, jittered.

    The jitter is deliberate: a dataset of identical +/-1.0 gripper commands
    trains a policy that only ever emits those two values.
    """
    if is_open:
        return np.random.uniform(0.9, 1.0, size=(1,))
    return np.random.uniform(-1.0, -0.9, size=(1,))


class SpaceMouseBinding(TeleopBinding):
    """The puck drives the arm; the buttons latch the gripper.

    The gripper is a latch rather than a level: the left button closes it and
    the right opens it, and it stays there until the other button is pressed.
    """

    PRODUCES = {
        "arm": ActionKind.CARTESIAN_DELTA,
        "end_effector": ActionKind.GRIPPER,
    }

    # gripper_open is read with a fallback, so it is not required.
    NEEDS = ()

    def __init__(self) -> None:
        self._grip: Optional[np.ndarray] = None
        self.left = False
        self.right = False

    def reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Start each episode with the buttons released."""
        self.left = False
        self.right = False

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Map the twist onto the arm, and the buttons onto the gripper."""
        buttons = reading["buttons"]
        self.left, self.right = bool(buttons[0]), bool(buttons[1])

        parts: dict[str, np.ndarray] = {"arm": np.asarray(reading["twist"])}

        if self.left:
            self._grip = jittered_grip(is_open=False)
        elif self.right:
            self._grip = jittered_grip(is_open=True)
        elif self._grip is None:
            self._grip = jittered_grip(is_open=bool(context.get("gripper_open", True)))
        parts["end_effector"] = self._grip.copy()

        moved = float(np.linalg.norm(reading["twist"])) > self.MOVEMENT_EPSILON
        return TeleopAction(parts=parts, driving=moved or self.left or self.right)

    def publish(self, reading: Mapping[str, Any]) -> dict[str, Any]:
        """Hold the second button to put a glove in control of the hand.

        The dex-hand setup reads that button as its "left". The gripper latch
        above reads the buttons the other way round, which is how the two rigs
        have always behaved; the difference is stated rather than inherited.
        """
        return {"hand_driving": bool(reading["buttons"][1])}


class LeaderArmBinding(TeleopBinding):
    """A leader arm posed by hand, as a Cartesian delta.

    The device reports where the operator put it. The env takes a delta, so the
    reading is differenced against the follower's measured pose and divided by
    the env's own scale.
    """

    PRODUCES = {
        "arm": ActionKind.CARTESIAN_DELTA,
        "end_effector": ActionKind.GRIPPER,
    }

    NEEDS = ("tcp_pose", "action_scale")

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Difference the leader's pose against the follower's."""
        tcp_pose = np.asarray(context["tcp_pose"])
        scale = np.asarray(context["action_scale"])

        delta_position = (np.asarray(reading["position"]) - tcp_pose[:3]) / scale[0]
        rotation = (
            R.from_quat(np.asarray(reading["orientation"]).copy())
            * R.from_quat(tcp_pose[3:].copy()).inv()
        )
        delta_rotation = rotation.as_euler("xyz") / scale[1]

        parts = {
            "arm": np.clip(
                np.concatenate((delta_position, delta_rotation), axis=0), -1.0, 1.0
            )
        }
        grip = np.asarray(reading["grip"]) / scale[2]
        parts["end_effector"] = np.clip(-(2 * grip - 1.0), -1.0, 1.0)
        # A leader arm is posed by hand, so it is driving whenever it streams.
        return TeleopAction(parts=parts, driving=True)


class LeaderJointBinding(TeleopBinding):
    """A leader arm reported in joint space, for one side of the robot.

    Args:
        side: Index of the arm this leader drives, for reading the follower's
            joint positions.
        use_delta: Whether the env takes joint deltas or absolute targets.
        action_scale: Divisor turning a joint delta into a normalized action.
    """

    PRODUCES = {
        "arm": ActionKind.JOINT_POSITION,
        "end_effector": ActionKind.GRIPPER,
    }

    NEEDS = ()

    def __init__(
        self, side: int = 0, use_delta: bool = False, action_scale: float = 0.1
    ) -> None:
        self.side = side
        self.use_delta = use_delta
        self.action_scale = action_scale
        # Only a delta needs the follower's joints to difference against.
        self.NEEDS = ("joint_positions",) if use_delta else ()
        # Differencing against the follower makes this a delta, not a target.
        self.PRODUCES = {
            "arm": ActionKind.JOINT_DELTA if use_delta else ActionKind.JOINT_POSITION,
            "end_effector": ActionKind.GRIPPER,
        }

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Difference the leader's joints against the follower's."""
        target = np.asarray(reading["joint_position"])
        if self.use_delta:
            current = np.asarray(context["joint_positions"])[self.side]
            arm = np.clip((target - current) / self.action_scale, -1.0, 1.0)
        else:
            arm = target.copy()

        grip = np.asarray(reading["grip"])
        return TeleopAction(
            parts={
                "arm": arm,
                "end_effector": np.clip(-(2 * grip - 1.0), -1.0, 1.0),
            },
            driving=True,
        )


class GloveBinding(TeleopBinding):
    """A data glove driving a dexterous hand, relative to a baseline.

    Pressing the arm device's button re-baselines the glove against the hand's
    current pose, so the operator can reposition their hand without the robot's
    hand following.

    The hold is explicit: once the operator lets go, the hand stays where they
    posed it rather than returning to whatever the policy commands. That is the
    behavior the dex-hand setup has always had, and it is stated here rather
    than left to fall out of the wrapper.
    """

    PRODUCES = {"hand": ActionKind.HAND}

    def __init__(self, hold: bool = True) -> None:
        self.hold = hold
        self._baseline: Optional[np.ndarray] = None
        self._commanded = np.zeros(6, dtype=np.float64)
        self._base = np.zeros(6, dtype=np.float64)
        self._rebaseline = False

    def reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Start the episode from the pose the env just put the hand into.

        The env homes the hand to its configured reset pose, so starting from
        zero here would make the first command a jump away from it -- on
        hardware, an abrupt one.
        """
        start = context.get("hand_reset_pose")
        self._commanded = (
            np.zeros(6, dtype=np.float64)
            if start is None
            else np.asarray(start, dtype=np.float64).reshape(-1).copy()
        )
        self._base = self._commanded.copy()
        self._baseline = None

    def rebaseline(self) -> None:
        """Re-zero the glove against the hand's current pose."""
        self._rebaseline = True

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Track the operator's fingers, or hold where they left them."""
        angles = np.asarray(reading["angles"], dtype=np.float64)
        # Held by default: a glove with nothing gating it stays where the
        # operator last posed it rather than tracking their resting hand.
        if context.get("hand_driving", False):
            if self._rebaseline or self._baseline is None:
                # Re-zero on the edge, so taking control does not jump the hand.
                self._baseline = angles.copy()
                self._base = self._commanded.copy()
                self._rebaseline = False
            self._commanded = np.clip(self._base + (angles - self._baseline), 0.0, 1.0)
        else:
            self._baseline = None  # re-zero next time control is taken
        # The arm device's button decides who is driving, not the glove.
        return TeleopAction(parts={"hand": self._commanded.copy()}, driving=False)


def _rotvec_to_euler(action: np.ndarray, action_scale: Any) -> np.ndarray:
    """Re-express a rotvec delta as the Euler delta a Franka env expects.

    The controller reports rotation as a scaled rotvec; the env reads Euler.
    Both are divided by the same rotation scale, so the round trip through a
    rotation is what keeps the two conventions equal rather than merely close.
    """
    out = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if out.size < 6:
        return out

    scale = float(np.asarray(action_scale, dtype=np.float64)[1])
    if scale <= 1e-9:
        out[3:6] = 0.0
        return out

    delta = R.from_rotvec(np.asarray(out[3:6], dtype=np.float64) * scale)
    out[3:6] = np.clip(delta.as_euler("xyz") / scale, -1.0, 1.0)
    return out


class _PicoArmBinding(TeleopBinding):
    """What the two PICO bindings share: one controller, one arm.

    The controller says how far the operator has moved since taking hold. Where
    the arm was at that moment is what turns that into a command, and only this
    side knows it, so the anchor is kept here.

    Args:
        gripper: Whether this arm's action carries a gripper channel.
        side: Which arm's pose to read out of a dual-arm ``tcp_pose``.
    """

    PRODUCES = {
        "arm": ActionKind.CARTESIAN_DELTA,
        "end_effector": ActionKind.GRIPPER,
    }

    NEEDS = ("tcp_pose", "action_scale")

    #: The grip says exactly when the operator is driving, so no hold window.
    HOLD_WINDOW = 0.0

    def __init__(self, gripper: bool = True, side: int = 0) -> None:
        self.gripper = bool(gripper)
        self.side = int(side)
        self._held_from: Optional[tuple[np.ndarray, R]] = None

    def _measured_pose(self, context: Mapping[str, Any]) -> np.ndarray:
        pose = np.asarray(context["tcp_pose"], dtype=np.float32).reshape(-1)
        if pose.size == 7:
            return pose
        if pose.size == 14:
            return pose[self.side * 7 : self.side * 7 + 7]
        raise ValueError(
            f"{type(self).__name__} expects get_tcp_pose() to return 7 or 14 "
            f"values, got {pose.size}."
        )

    def _read(self, reading: Mapping[str, Any], context: Mapping[str, Any]):
        """Turn the operator's motion into a command for where this arm is."""
        pose = self._measured_pose(context)
        held = bool(reading.get("held", False))

        if not held:
            self._held_from = None
            return pose, np.zeros(0, dtype=np.float32), False

        if self._held_from is None:
            # They just took hold: the arm is where the motion starts from.
            self._held_from = (
                np.asarray(pose[:3], dtype=np.float64).copy(),
                R.from_quat(np.asarray(pose[3:7], dtype=np.float64)),
            )

        action = self._command(reading, pose, context["action_scale"])
        return pose, action, True

    def _command(
        self,
        reading: Mapping[str, Any],
        pose: np.ndarray,
        action_scale: Any,
    ) -> np.ndarray:
        """The normalized delta from where the arm is to where it should go."""
        anchor_pos, anchor_rot = self._held_from
        target_pos = anchor_pos + np.asarray(
            reading["position_delta"], dtype=np.float64
        )
        target_rot = (
            R.from_rotvec(np.asarray(reading["rotation_delta"], dtype=np.float64))
            * anchor_rot
        )

        scale = np.asarray(action_scale, dtype=np.float64)
        current_pos = np.asarray(pose[:3], dtype=np.float64)
        current_rot = R.from_quat(np.asarray(pose[3:7], dtype=np.float64))

        moved = (target_pos - current_pos) / float(scale[0])
        turn = (target_rot * current_rot.inv()).as_rotvec()
        max_turn = float(scale[1])
        if max_turn > 1e-9:
            angle = float(np.linalg.norm(turn))
            if angle > max_turn:
                turn = turn * (max_turn / angle)
            turned = turn / max_turn
        else:
            turned = np.zeros(3, dtype=np.float64)

        action = np.clip(np.concatenate((moved, turned)), -1.0, 1.0)
        if self.gripper:
            grip = 0.0
            if reading.get("grip_close", False):
                grip = -1.0
            elif reading.get("grip_open", False):
                grip = 1.0
            action = np.concatenate((action, np.array([grip], dtype=np.float64)))
        return action.astype(np.float32)

    @staticmethod
    def _reported(reading: Mapping[str, Any]) -> dict[str, Any]:
        """The reading, under the names a collector already records."""
        info = {
            "pico_active": bool(reading.get("held", False)),
            "pico_ready": bool(reading.get("ready", False)),
            "pico_hand": reading.get("hand"),
            "pico_calibrated": bool(reading.get("calibrated", False)),
            "pico_control_value": reading.get("control_value", 0.0),
        }
        for key in ("stale", "invalid_pose"):
            if reading.get(key):
                info[f"pico_{key}"] = True
        if reading.get("held", False):
            close, opened = reading.get("grip_close"), reading.get("grip_open")
            info["pico_gripper_close_pressed"] = bool(close)
            info["pico_gripper_open_pressed"] = bool(opened)
            info["pico_gripper_action"] = -1.0 if close else (1.0 if opened else 0.0)
            info["pico_gripper_close"] = bool(close)
        return info

    def reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Forget the previous episode's grip and where it started."""
        self._held_from = None


class PicoBinding(_PicoArmBinding):
    """A VR controller driving an arm through pose deltas.

    The env reads deltas directly, so the controller's own reading is the
    action, once its rotation is re-expressed as Euler.
    """

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Return the arm delta, and the gripper when the reader sends one."""
        _, action, held = self._read(reading, context)
        info = self._reported(reading)
        if not held:
            return TeleopAction(info=info)

        action = _rotvec_to_euler(action, context["action_scale"])
        parts = {"arm": action[:6]}
        if self.gripper and action.size >= 7:
            parts["end_effector"] = action[6:7]
        return TeleopAction(parts=parts, driving=True, info=info)


class PicoTcpBinding(_PicoArmBinding):
    """A VR controller driving an arm that takes absolute poses.

    The controller reports a delta and the env wants a pose, so each delta is
    composed onto the measured TCP pose and sent as position plus rot6d.

    Releasing the grip part-way through an action chunk holds the last commanded
    pose rather than returning to the policy's, which would jerk the arm mid
    motion. :meth:`on_action_chunk_begin` releases that hold at the next chunk.

    Args:
        gripper: Whether this arm's action carries a gripper channel.
        side: Which arm's pose to read out of a dual-arm ``tcp_pose``.
        hold_current_when_inactive: Command the measured pose while nobody is
            driving, instead of passing the policy's action through.
    """

    PRODUCES = {
        "arm": ActionKind.CARTESIAN_POSE,
        "end_effector": ActionKind.GRIPPER,
    }

    #: Absolute poses can leave the env's action space, so they are clipped
    #: back into it. Deltas from the other devices are already normalised.
    CLIPS_TO_ACTION_SPACE = True

    def __init__(
        self,
        gripper: bool = True,
        side: int = 0,
        hold_current_when_inactive: bool = True,
    ) -> None:
        super().__init__(gripper=gripper, side=side)
        self.hold_current_when_inactive = bool(hold_current_when_inactive)
        self._holding_after_release = False
        self._last_command: Optional[np.ndarray] = None

    @staticmethod
    def _pose_to_command(pose: np.ndarray, grip: float = 0.0) -> np.ndarray:
        """A pose as the env spells it: position, rot6d, gripper."""
        from rlinf.utils.rot6d import matrix_to_rot6d

        rot6d = matrix_to_rot6d(R.from_quat(pose[3:7]).as_matrix())
        return np.concatenate(
            [
                np.asarray(pose[:3], dtype=np.float32),
                rot6d.astype(np.float32),
                np.array([grip], dtype=np.float32),
            ]
        )

    def _compose(
        self, action: np.ndarray, pose: np.ndarray, action_scale: Any
    ) -> np.ndarray:
        """Put the operator's delta on top of where the arm actually is."""
        if action.size < 6:
            raise ValueError(
                "PicoTcpBinding expects at least 6 motion dims from the reader, "
                f"got {action.size}."
            )

        scale = np.asarray(action_scale, dtype=np.float64)
        position = np.asarray(pose[:3], dtype=np.float64) + action[:3] * float(scale[0])

        rotation = np.asarray(action[3:6], dtype=np.float64)
        norm = float(np.linalg.norm(rotation))
        if norm > 1.0:
            rotation = rotation / norm
        turned = R.from_rotvec(rotation * float(scale[1])) * R.from_quat(
            np.asarray(pose[3:7], dtype=np.float64)
        )

        grip = float(action[6]) if action.size >= 7 else 0.0
        return self._pose_to_command(np.concatenate([position, turned.as_quat()]), grip)

    @staticmethod
    def _split(command: np.ndarray) -> dict[str, np.ndarray]:
        return {"arm": command[:-1], "end_effector": command[-1:]}

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Return the pose to command, whoever is deciding it right now."""
        pose, action, held = self._read(reading, context)
        info = {**self._reported(reading), "pico_replaced": held}

        if held:
            command = self._compose(action, pose, context["action_scale"])
            self._last_command = command.copy()
            self._holding_after_release = True
            return TeleopAction(parts=self._split(command), driving=True, info=info)

        if (
            self._holding_after_release
            and not self.hold_current_when_inactive
            and self._last_command is not None
        ):
            # Released mid-chunk. Stay where the operator left the arm, and do
            # not mark it as intervention so training skips these frames.
            return TeleopAction(parts=self._split(self._last_command), info=info)

        if self.hold_current_when_inactive:
            # The gripper is left out so the policy's own command stands.
            return TeleopAction(
                parts={"arm": self._pose_to_command(pose)[:-1]}, info=info
            )

        return TeleopAction(info=info)

    def hold(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """The pose that keeps this arm where it is, for a skipped chunk."""
        return {"arm": self._pose_to_command(self._measured_pose(context))[:-1]}

    def on_action_chunk_begin(self) -> None:
        """Let go of a pose held since the operator released mid-chunk."""
        self._holding_after_release = False

    def reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Drop the held pose along with the previous episode's grip."""
        super().reset(context)
        self._holding_after_release = False
        self._last_command = None
