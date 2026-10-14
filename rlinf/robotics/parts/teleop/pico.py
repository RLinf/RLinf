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

"""PICO VR controller, driving an arm as Cartesian deltas, poses, or joints.

The three mappings share one config name: which one a robot gets follows from
whether its arm action is a delta, a pose, or a set of joint positions.
"""

from __future__ import annotations

import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.utils.logging import get_logger

from ...actions import ActionKind
from ..base import Features, Observation
from .base import TeleopAction, TeleopDevice


def _rotvec_to_euler(action: np.ndarray, action_scale: Sequence[float]) -> np.ndarray:
    """Convert a scaled rotation-vector delta to scaled Euler angles."""
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


@TeleopDevice.register("pico")
class Pico(TeleopDevice):
    """A PICO controller driving one arm, in whichever terms that arm takes.

    One handheld controller, three ways to say what it means: an arm that
    takes a delta gets :class:`PicoDelta`, one that takes an absolute pose gets
    :class:`PicoTcp`, and one that takes joint positions gets
    :class:`PicoJoint`. All are this class, so ``isinstance(device, Pico)``
    holds whichever the env asked for, and everything that does not depend on
    that choice -- the transport, the reading, the grip anchor -- lives here.

    :meth:`from_config` picks between them. It is abstract, so the config name
    resolves to the family rather than to any one variant.

    Args:
        gripper: Whether this arm's action carries a gripper channel.
        side: Which arm's pose to read out of a dual-arm ``tcp_pose``.
    """

    #: The delta variant's kinds. PicoTcp overrides the arm with an absolute
    #: pose; the parts filled are the same either way.
    LEGACY_FLAGS = {"use_pico": "pico"}

    PRODUCES = {
        "arm": ActionKind.CARTESIAN_DELTA,
        "end_effector": ActionKind.GRIPPER,
    }

    NEEDS = ("tcp_pose", "action_scale")

    #: The controller reports an explicit held state, so no hold window is needed.
    HOLD_WINDOW = 0.0

    def __init__(self, gripper: bool = True, side: int = 0, **pico_config: Any) -> None:
        self.gripper = bool(gripper)
        self.side = int(side)
        self._config = pico_config
        self._held_from: Optional[tuple[np.ndarray, R]] = None

    # Hardware.

    def _open(self) -> Any:
        from ..transports.pico import PicoExpert

        return PicoExpert(**self._config)

    @property
    def ready(self) -> bool:
        """Whether the reader can provide its explicit idle state.

        Unlike a leader arm, the PICO reader returns a safe ``ready=False``
        reading before the first controller packet arrives.
        """
        return self.is_connected

    @property
    def observation_features(self) -> Features:
        """Whether the operator is driving, and how far they have moved."""
        return {
            "held": {"dtype": "bool", "shape": ()},
            "position_delta": {"dtype": "float64", "shape": (3,)},
            "rotation_delta": {"dtype": "float64", "shape": (3,)},
            "grip_close": {"dtype": "bool", "shape": ()},
            "grip_open": {"dtype": "bool", "shape": ()},
        }

    def get_observation(self) -> Observation:
        """Read the controller."""
        return self._device.get_reading()

    def buttons(self) -> dict[str, bool]:
        """Return every controller button by its reported name.

        The reading carries only what drives the arm; an episode control that
        starts and stops recording from a menu button reads them here.
        """
        return self._device.get_buttons()

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], options: Mapping[str, Any], facts: Any
    ) -> Any:
        """Pick the delta, pose, or joint variant from the arm's action kind."""
        from rlinf.envs.real.wrappers.teleop.pico_config import split_dual_config

        from .group import TeleopEntry

        drives = options.get("drives")
        pico_cfg = dict(cfg.get("pico", {}))
        hold = bool(
            options.get(
                "hold_current_when_inactive",
                pico_cfg.pop("hold_current_when_inactive", True),
            )
        )
        gripper = bool(options.get("gripper", not bool(cfg.get("no_gripper", True))))

        # Match the device to the environment's Cartesian action semantics.
        arm = facts.kinds.get("arm" if drives is None else f"{drives}.arm")
        absolute = arm is ActionKind.CARTESIAN_POSE

        if drives in ("left", "right"):
            left_cfg, right_cfg = split_dual_config(pico_cfg)
            device_cfg = left_cfg if drives == "left" else right_cfg
            side = 0 if drives == "left" else 1
        else:
            device_cfg = {
                key: value
                for key, value in pico_cfg.items()
                if key not in ("left", "right")
            }
            device_cfg.setdefault("hand", "right")
            side = 0 if str(device_cfg["hand"]).lower() == "left" else 1

        if arm is ActionKind.JOINT_POSITION:
            device = PicoJoint.for_env(
                facts,
                drives,
                gripper=gripper,
                side=side,
                hold_current_when_inactive=hold,
                **device_cfg,
            )
        elif absolute:
            device = PicoTcp(
                gripper=gripper,
                side=side,
                hold_current_when_inactive=hold,
                **device_cfg,
            )
        else:
            device = PicoDelta(gripper=gripper, side=side, **device_cfg)
        return TeleopEntry(device, drives=drives)

    @classmethod
    def interlock(
        cls, cfg: Mapping[str, Any], facts: Any, entries: Sequence[Any]
    ) -> Optional["HoldTogether"]:
        """Hold every joint-driven arm while any one of them is held."""
        arms = [entry for entry in entries if isinstance(entry.device, PicoJoint)]
        return HoldTogether(arms) if len(arms) > 1 else None

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

    def _read(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        """Convert controller motion into an arm command."""
        pose = self._measured_pose(context)
        held = bool(reading.get("held", False))

        if not held:
            self._held_from = None
            return pose, np.zeros(0, dtype=np.float32), False

        action = self._command(reading, pose, context["action_scale"])
        return pose, action, True

    def _target(
        self, reading: Mapping[str, Any], pose: np.ndarray
    ) -> tuple[np.ndarray, R]:
        """Return the pose the operator has moved the arm to since the grip edge.

        The first call after a grip anchors to ``pose``, the arm's measured
        ``xyz`` and ``xyzw`` quaternion at that edge.
        """
        if self._held_from is None:
            self._held_from = (
                np.asarray(pose[:3], dtype=np.float64).copy(),
                R.from_quat(np.asarray(pose[3:7], dtype=np.float64)),
            )
        anchor_pos, anchor_rot = self._held_from
        target_pos = anchor_pos + np.asarray(
            reading["position_delta"], dtype=np.float64
        )
        target_rot = (
            R.from_rotvec(np.asarray(reading["rotation_delta"], dtype=np.float64))
            * anchor_rot
        )
        return target_pos, target_rot

    def _command(
        self,
        reading: Mapping[str, Any],
        pose: np.ndarray,
        action_scale: Sequence[float],
    ) -> np.ndarray:
        """Return the normalized delta from the measured to target pose."""
        target_pos, target_rot = self._target(reading, pose)

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
        """Return controller state using the collector's field names."""
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

    def on_reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Clear the pose anchor from the previous episode."""
        self._held_from = None


class PicoDelta(Pico):
    """Map PICO motion to Cartesian arm deltas."""

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


class PicoTcp(Pico):
    """Map PICO motion to absolute TCP pose commands.

    When configured, the device holds the latest pose after a mid-chunk
    release and clears that hold at the next action chunk.

    Args:
        gripper: Whether this arm's action carries a gripper channel.
        side: Which arm's pose to read out of a dual-arm ``tcp_pose``.
        hold_current_when_inactive: Command the measured pose while operator
            control is inactive instead of passing through the policy action.
    """

    PRODUCES = {
        "arm": ActionKind.CARTESIAN_POSE,
        "end_effector": ActionKind.GRIPPER,
    }

    #: Absolute pose commands are clipped to the environment action space.
    CLIPS_TO_ACTION_SPACE = True

    def __init__(
        self,
        gripper: bool = True,
        side: int = 0,
        hold_current_when_inactive: bool = True,
        **pico_config: Any,
    ) -> None:
        super().__init__(gripper=gripper, side=side, **pico_config)
        self.hold_current_when_inactive = bool(hold_current_when_inactive)
        self._holding_after_release = False
        self._last_command: Optional[np.ndarray] = None

    @staticmethod
    def _pose_to_command(pose: np.ndarray, grip: float = 0.0) -> np.ndarray:
        """Convert a quaternion pose to position, rot6d, and gripper fields."""
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
        self, action: np.ndarray, pose: np.ndarray, action_scale: Sequence[float]
    ) -> np.ndarray:
        """Compose an operator delta with the measured arm pose."""
        if action.size < 6:
            raise ValueError(
                "PicoTcp expects at least 6 motion dims from the reader, "
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
        """Return the active operator, hold, or pass-through command."""
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
            # Preserve the last operator pose until the current chunk ends.
            return TeleopAction(parts=self._split(self._last_command), info=info)

        if self.hold_current_when_inactive:
            # Leave the gripper unset so the policy command remains active.
            return TeleopAction(
                parts={"arm": self._pose_to_command(pose)[:-1]}, info=info
            )

        return TeleopAction(info=info)

    def hold(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return a pose command that holds the measured arm position."""
        return {"arm": self._pose_to_command(self._measured_pose(context))[:-1]}

    def on_action_chunk_begin(self) -> None:
        """Clear the pose held after a mid-chunk release."""
        self._holding_after_release = False

    def on_reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Clear the held command and pose anchor."""
        super().on_reset(context)
        self._holding_after_release = False
        self._last_command = None


#: What every PicoJoint tick reports, so a recording keeps one schema whether
#: or not the solver ran.
_IK_REPORT = {
    "joint_step_fraction": 1.0,
    "limiting_joint": 0,
    "ik_max_joint_delta": 0.0,
    "ik_attempts": 0,
    "ik_target_fraction": 0.0,
    "ik_initial_fault": None,
}


class PicoJoint(Pico):
    """Map PICO motion to absolute joint positions through the arm's IK.

    While the grip is held, the target is the arm's pose at the grip edge moved
    by the operator's whole motion since then. IK turns it into joints, and one
    tick moves every joint along the straight line toward them, at most
    ``joint_step_limits`` each. The gripper is an opening, ``0`` shut to ``1``
    open, latched by the close and open buttons.

    Released, the arm holds the joints it was let go at rather than the
    measured ones, which sag under gravity and would walk the target.

    A controller that goes stale, loses calibration, or reports an invalid pose
    latches a fault: the arm holds where it stands until its grip is released
    and taken again. The arm starts latched, so a grip squeezed at startup does
    not drive it.

    Args:
        kinematics: Builds this arm's solver when the device opens. The solver
            offers ``fk(joints, gripper)``, returning a ``4x4`` pose;
            ``solve(target, seed, gripper)``, returning a dataclass with
            ``success``, ``q_target``, ``reason``, ``elapsed_s``,
            ``position_error`` and ``rotation_error``; and ``max_solve_s``, the
            time every attempt at one target may take together.
        joints: Where this arm's joints sit in ``joint_positions``.
        grip: Where this arm's gripper sits in ``joint_positions``, or ``None``.
        joint_step_limits: Largest move one tick may command, per joint.
        ik_backtrack_attempts: Retries toward an unreachable target, each
            halving the step from the measured pose.
        max_tick_s: A solution older than this when it would be sent is
            dropped, and the arm latched.
        gripper: Whether this arm's action carries a gripper channel.
        side: Which arm this controller drives.
        hold_current_when_inactive: Command the held pose while the grip is
            released, rather than leave the arm to the policy.
    """

    PRODUCES = {
        "arm": ActionKind.JOINT_POSITION,
        "end_effector": ActionKind.GRIPPER,
    }

    #: The measured joints and grippers, laid out like the env's action.
    NEEDS = ("joint_positions",)

    def __init__(
        self,
        kinematics: Callable[[], Any],
        joints: slice,
        grip: Optional[slice],
        joint_step_limits: Sequence[float],
        *,
        ik_backtrack_attempts: int = 0,
        max_tick_s: float = 0.1,
        gripper: bool = True,
        side: int = 0,
        hold_current_when_inactive: bool = True,
        **pico_config: Any,
    ) -> None:
        super().__init__(gripper=gripper, side=side, **pico_config)
        if (
            isinstance(ik_backtrack_attempts, bool)
            or not isinstance(ik_backtrack_attempts, int)
            or ik_backtrack_attempts < 0
        ):
            raise ValueError(
                "pico.ik_backtrack_attempts must be a non-negative integer, but "
                f"got {ik_backtrack_attempts!r}."
            )
        if not np.isfinite(float(max_tick_s)) or float(max_tick_s) <= 0:
            raise ValueError(f"pico.max_tick_s must be positive, but got {max_tick_s}.")
        self._build_kinematics = kinematics
        self._kinematics: Any = None
        self._joints = joints
        self._grip_at = grip
        self._step_limits = np.asarray(joint_step_limits, dtype=np.float64)
        self.ik_backtrack_attempts = ik_backtrack_attempts
        self.max_tick_s = float(max_tick_s)
        self.hold_current_when_inactive = bool(hold_current_when_inactive)
        #: An idle arm keeps its held pose only when it holds one.
        self.APPLIES_WHILE_IDLE = self.hold_current_when_inactive
        self._hand = str(pico_config.get("hand", "right"))
        self._fault: Optional[str] = "release_grip"
        # An arm is re-latched every tick its partner is held, so a reason is
        # logged once until the arm drives again.
        self._logged_fault: Optional[str] = self._fault
        # When the solution about to be sent began, for HoldTogether to age.
        self._solved_at: Optional[float] = None
        self._preserve = False
        self._grip: Optional[float] = None
        self._idle_joints: Optional[np.ndarray] = None
        self._last_ik_warning = -float("inf")
        self._last_step_log = -float("inf")
        self._logger = get_logger()

    @classmethod
    def for_env(cls, facts: Any, drives: Optional[str], **settings: Any) -> "PicoJoint":
        """Build the device for one arm of the env that ``facts`` describe."""
        build = getattr(facts, "arm_kinematics", None)
        if build is None:
            raise ValueError(
                "This env's arm takes joint positions but the env offers no "
                "arm_kinematics, so a PICO controller cannot solve its joints."
            )
        steps = getattr(facts, "joint_step_limits", None)
        if steps is None:
            raise ValueError(
                "A PICO controller driving joints needs the env's "
                "joint_step_limits to bound each tick, but the env has none."
            )
        prefix = "" if drives is None else f"{drives}."
        return cls(
            kinematics=lambda: build(drives),
            joints=facts.layout[f"{prefix}arm"],
            grip=facts.layout.get(f"{prefix}end_effector"),
            joint_step_limits=steps,
            **settings,
        )

    def _open(self) -> Any:
        self._kinematics = self._build_kinematics()
        return super()._open()

    # Episode control hands its own transitions back through these.

    @property
    def fault(self) -> Optional[str]:
        """Why this arm is held until its grip is released, or ``None``."""
        return self._fault

    def hold_until_released(self, reason: str) -> None:
        """Hold this arm until the operator releases and retakes the grip."""
        if reason != self._logged_fault:
            self._logger.warning(
                "PICO %s arm holds until its grip is released: %s", self._hand, reason
            )
            self._logged_fault = reason
        self._fault = reason
        self._preserve = False
        self._held_from = None
        self._grip = None
        self._idle_joints = None

    def preserve_reference(self) -> None:
        """Keep the grip anchor across the next reset."""
        self._preserve = True

    def on_reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Re-anchor the grip after a reset, unless asked to keep it."""
        if self._preserve:
            self._preserve = False
            return
        self.hold_until_released("release_grip")

    # Mapping one reading onto joints.

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Return this arm's joint and gripper targets."""
        started = time.monotonic()
        measured = np.asarray(context["joint_positions"], dtype=np.float64).reshape(-1)
        joints = measured[self._joints].copy()
        grip = None if self._grip_at is None else float(measured[self._grip_at][0])
        info = {**self._reported(reading), **_IK_REPORT}

        if (
            not reading.get("ready", False)
            or not reading.get("calibrated", False)
            or reading.get("invalid_pose", False)
        ):
            self.hold_until_released(f"{self._hand}:unavailable")
        elif self._fault is not None and not reading.get("held", False):
            # A grip still squeezed after a fault must not re-engage.
            self._fault = None
            self._idle_joints = joints.copy()
        if self._fault is not None:
            self._held_from = None
            return self._holding(joints, grip, info)

        if not reading.get("held", False):
            self._held_from = None
            if self._idle_joints is None:
                self._idle_joints = joints.copy()
            self._grip = grip
            return self._holding(self._idle_joints, grip, info)
        self._idle_joints = None
        self._logged_fault = None
        if self._grip is None:
            self._grip = grip

        current = np.asarray(self._kinematics.fk(joints, grip), dtype=np.float64)
        position, rotation = self._target(
            reading,
            np.concatenate([current[:3, 3], R.from_matrix(current[:3, :3]).as_quat()]),
        )
        target = np.eye(4)
        target[:3, 3] = position
        target[:3, :3] = rotation.as_matrix()
        result, attempts, fraction, first_fault = self._solve(
            target, current, joints, grip
        )
        info.update(
            ik_attempts=attempts,
            ik_target_fraction=fraction,
            ik_initial_fault=first_fault,
            ik_elapsed_s=result.elapsed_s,
            ik_position_error=result.position_error,
            ik_rotation_error=result.rotation_error,
            ik_fault=None if result.success else result.reason,
        )
        if result.success:
            command = self._step_toward(result.q_target, joints, info)
            if reading.get("grip_close", False):
                self._grip = 0.0
            elif reading.get("grip_open", False):
                self._grip = 1.0
        else:
            # Hold the joints for this frame only; the gripper stays latched.
            command = joints
            now = time.monotonic()
            if now - self._last_ik_warning >= 1.0:
                self._logger.warning(
                    "PICO %s arm holds this frame, retrying: ik:%s",
                    self._hand,
                    result.reason,
                )
                self._last_ik_warning = now

        if not self._device.ready or (
            result.success and time.monotonic() - started > self.max_tick_s
        ):
            self.hold_until_released(f"{self._hand}:stale_after_ik")
            return self._holding(joints, grip, info)
        self._solved_at = started if result.success else None
        return TeleopAction(
            parts=self._parts(command, self._grip), driving=True, info=info
        )

    def _expire(self, now: float) -> None:
        """Latch this arm if what it is about to send is no longer current.

        Its solution may have aged past ``max_tick_s``, or its controller may
        have stopped streaming, while the other arms were mapped.
        """
        solved, self._solved_at = self._solved_at, None
        if self._fault is None and (
            not self._device.ready
            or (solved is not None and now - solved > self.max_tick_s)
        ):
            self.hold_until_released(f"{self._hand}:stale_after_ik")

    def hold(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return the measured joints, leaving the gripper to the policy."""
        measured = np.asarray(context["joint_positions"], dtype=np.float64).reshape(-1)
        return {"arm": measured[self._joints].astype(np.float32)}

    def _parts(
        self, joints: np.ndarray, grip: Optional[float]
    ) -> dict[str, np.ndarray]:
        parts = {"arm": np.asarray(joints, dtype=np.float32)}
        if self.gripper and grip is not None:
            parts["end_effector"] = np.array([grip], dtype=np.float32)
        return parts

    def _holding(
        self, joints: np.ndarray, grip: Optional[float], info: dict[str, Any]
    ) -> TeleopAction:
        if not self.hold_current_when_inactive:
            return TeleopAction(info=info)
        return TeleopAction(parts=self._parts(joints, grip), info=info)

    def _step_toward(
        self, solution: np.ndarray, joints: np.ndarray, info: dict[str, Any]
    ) -> np.ndarray:
        """Move along the straight line to the solution, one tick's worth.

        Scaling the whole step keeps the direction a clip per joint would bend.
        """
        difference = np.asarray(solution, dtype=np.float64) - joints
        demand = np.abs(difference) / self._step_limits
        limiting = int(np.argmax(demand))
        fraction = min(1.0, 1.0 / max(float(demand[limiting]), 1e-12))
        info["joint_step_fraction"] = fraction
        info["limiting_joint"] = limiting + 1 if fraction < 1.0 else 0
        info["ik_max_joint_delta"] = float(np.max(np.abs(difference)))
        now = time.monotonic()
        if fraction < 1.0 and now - self._last_step_log >= 1.0:
            self._logger.info(
                "PICO %s arm step limited: fraction=%.3f, joint=J%d, "
                "required=%.3f rad, limit=%.3f rad",
                self._hand,
                fraction,
                limiting + 1,
                abs(difference[limiting]),
                self._step_limits[limiting],
            )
            self._last_step_log = now
        return joints + fraction * difference

    def _solve(
        self,
        target: np.ndarray,
        current: np.ndarray,
        joints: np.ndarray,
        grip: Optional[float],
    ) -> tuple[Any, int, float, Optional[str]]:
        """Solve IK, retrying geometric misses at halved steps from ``current``.

        Every attempt shares the solver's ``max_solve_s``, checked between
        calls rather than preempting one.
        """
        from dataclasses import replace

        started = time.monotonic()
        result = self._kinematics.solve(target, joints.copy(), grip)
        first_fault = result.reason
        attempts, fraction = 1, 1.0
        turn = None
        while True:
            elapsed = time.monotonic() - started
            if elapsed > self._kinematics.max_solve_s:
                result = replace(result, success=False, reason="solve_timeout")
                break
            if (
                result.success
                or result.reason not in {"not_converged", "residual", "joint_limits"}
                or attempts > self.ik_backtrack_attempts
            ):
                break
            if turn is None:
                turn = R.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
            fraction *= 0.5
            step = current.copy()
            step[:3, 3] += fraction * (target[:3, 3] - current[:3, 3])
            step[:3, :3] = R.from_rotvec(fraction * turn).as_matrix() @ current[:3, :3]
            result = self._kinematics.solve(step, joints.copy(), grip)
            attempts += 1
        return replace(result, elapsed_s=elapsed), attempts, fraction, first_fault


class HoldTogether:
    """Hold every PICO-driven arm of a robot while any one of them is held.

    Each arm maps its own reading, so an arm solved first cannot see its
    partner fault later in the same tick. This runs once every arm has been
    mapped and before the action is sent: it drops a solution that aged past
    ``max_tick_s`` or whose controller stopped streaming while the other arms
    solved, and if any arm is held, holds them all at the joints measured this
    tick.

    Args:
        entries: The entries whose devices are :class:`PicoJoint` arms.
    """

    def __init__(self, entries: Sequence[Any]) -> None:
        self._entries = list(entries)

    def __call__(self, sample: Any, context: Mapping[str, Any]) -> Any:
        """Return ``sample``, or a hold of every arm if any one is held."""
        now = time.monotonic()
        arms = [entry.device for entry in self._entries]
        for arm in arms:
            arm._expire(now)
        fault = next((arm.fault for arm in arms if arm.fault is not None), None)
        if fault is None:
            return sample
        for arm in arms:
            if arm.fault is None:
                arm.hold_until_released(fault)
        sample.action = np.asarray(context["joint_positions"], dtype=np.float64).copy()
        sample.active = False
        sample.apply_when_inactive = True
        for entry in self._entries:
            prefix = "" if entry.drives is None else f"{entry.drives}_"
            sample.info[f"{prefix}pico_active"] = False
        return sample
