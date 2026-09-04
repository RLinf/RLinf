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

"""PICO VR controller, driving an arm as Cartesian deltas or absolute poses.

The two mappings share one config name: which one a robot gets follows from
whether its arm action is a pose or a delta.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

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

    One handheld controller, two ways to say what it means: an arm that takes
    a delta gets :class:`PicoDelta`, one that takes an absolute pose gets
    :class:`PicoTcp`. Both are this class, so ``isinstance(device, Pico)``
    holds whichever the env asked for, and everything that does not depend on
    that choice -- the transport, the reading, the grip anchor -- lives here.

    :meth:`from_config` picks between them. It is abstract, so the config name
    resolves to the family rather than to either variant.

    Args:
        gripper: Whether this arm's action carries a gripper channel.
        side: Which arm's pose to read out of a dual-arm ``tcp_pose``.
    """

    #: The delta variant's kinds. PicoTcp overrides the arm with an absolute
    #: pose; the parts filled are the same either way.
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

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], options: Mapping[str, Any], facts: Any
    ) -> Any:
        """Pick the pose or delta variant from the robot's arm action kind."""
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

        device = (
            PicoTcp(
                gripper=gripper,
                side=side,
                hold_current_when_inactive=hold,
                **device_cfg,
            )
            if absolute
            else PicoDelta(gripper=gripper, side=side, **device_cfg)
        )
        return TeleopEntry(device, drives=drives)

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

        if self._held_from is None:
            # Anchor controller motion to the arm pose at the grip edge.
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
        action_scale: Sequence[float],
    ) -> np.ndarray:
        """Return the normalized delta from the measured to target pose."""
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
