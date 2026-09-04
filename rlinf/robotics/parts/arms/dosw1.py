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

"""Dual-arm SDK adapter for the DOSW1 robot."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from rlinf.robotics.parts.arms.base import Arm
from rlinf.robotics.parts.base import (
    Action,
    Connection,
    Features,
    Observation,
    RobotPart,
)
from rlinf.robotics.parts.end_effectors.base import EndEffector
from rlinf.utils.logging import get_logger

#: Arm sides exposed by one DOSW1 SDK session.
_ARM_SIDES: tuple[str, ...] = ("left", "right")


NUM_JOINTS = 6


@dataclass
class DOSW1RobotState:
    """Snapshot of the dual-arm DOSW1 robot at a single timestep.

    Each arm has 6 revolute joints (radians) and 1 gripper value in metres.
    Control and state are joint-space only.
    """

    left_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_JOINTS)
    )
    left_gripper: float = 0.0
    right_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_JOINTS)
    )
    right_gripper: float = 0.0
    timestamp: float = field(default_factory=time.time)


try:
    from airbot_sdk.Airbot import AirbotRobot as _AirbotRobot
    from airbot_sdk.configs.config import DosW1Config as _AirbotSDKConfig
except ImportError:
    _AirbotRobot = None
    _AirbotSDKConfig = None

_CONTROL_LOOP_DT = 0.02
_STATE_READY_TIMEOUT_S = 5.0


class DOSW1Connection(Connection):
    """Shared SDK connection for both DOSW1 arms and grippers."""

    def __init__(
        self,
        *,
        robot_url: str = "localhost",
        left_arm_port: int = 50051,
        right_arm_port: int = 50053,
        left_lead_port: int = 50050,
        right_lead_port: int = 50052,
        enable_human_in_loop: bool = False,
        gripper_width_max: float = 0.07,
        is_dummy: bool = False,
    ) -> None:
        """Initialize a deferred DOSW1 connection.

        Args:
            robot_url: Host of the AirBot gRPC endpoint.
            left_arm_port: gRPC port of the left follower arm.
            right_arm_port: gRPC port of the right follower arm.
            left_lead_port: gRPC port of the left leader arm.
            right_lead_port: gRPC port of the right leader arm.
            enable_human_in_loop: Whether to connect the leader arms for
                teleoperation.
            gripper_width_max: Maximum gripper opening, in metres.
            is_dummy: Whether to run without the hardware SDK.
        """
        self._logger = get_logger()
        self._robot_url = robot_url
        self._left_arm_port = left_arm_port
        self._right_arm_port = right_arm_port
        self._left_lead_port = left_lead_port
        self._right_lead_port = right_lead_port
        self._enable_human_in_loop = enable_human_in_loop
        self._gripper_width_max = gripper_width_max
        self._is_dummy = is_dummy
        self._leader_arm_enabled = bool(enable_human_in_loop)
        self._robot: object | None = None

    def _open(self) -> Any:
        """Connect the follower arms and, if enabled, the leader arms."""
        if self._is_dummy:
            return None

        if _AirbotRobot is None or _AirbotSDKConfig is None:
            raise ImportError(
                "airbot_sdk is not installed. Install it or set is_dummy=True."
            )

        sdk_cfg = _AirbotSDKConfig()
        sdk_cfg.USE_CAM = False
        sdk_cfg.USE_CAR = False
        sdk_cfg.USE_LEAD_ARMS = self._enable_human_in_loop

        self._logger.info(
            "[DOSW1SDK] Connecting via AirbotRobot (url=%s, ports=%s/%s/%s/%s) ...",
            self._robot_url,
            self._left_arm_port,
            self._right_arm_port,
            self._left_lead_port,
            self._right_lead_port,
        )

        self._robot = _AirbotRobot(
            config_=sdk_cfg,
            left_lead_port=self._left_lead_port,
            left_lead_url=self._robot_url,
            right_lead_port=self._right_lead_port,
            right_lead_url=self._robot_url,
            left_port=self._left_arm_port,
            left_url=self._robot_url,
            right_port=self._right_arm_port,
            right_url=self._robot_url,
        )
        try:
            self._wait_for_initial_state()
        except Exception:
            self._shutdown_robot(self._robot)
            self._robot = None
            raise
        self._logger.info("[DOSW1SDK] Connected.")
        return self._robot

    def _release(self, device: Any) -> None:
        """Disconnect the wrapped AirbotRobot instance."""
        self._logger.info("[DOSW1SDK] Disconnecting.")
        robot = self._robot
        self._robot = None
        if robot is None:
            return

        try:
            self._shutdown_robot(robot)
        except Exception:
            self._logger.exception("[DOSW1SDK] Failed to disconnect cleanly")

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Return the arms and end effectors exported by this connection."""
        parts: dict[str, RobotPart] = {}
        for side in _ARM_SIDES:
            parts[side] = DOSW1Arm(self, side)
            parts[f"{side}_end_effector"] = DOSW1EndEffector(self, side)
        return parts

    def set_leader_arm_enabled(self, enabled: bool) -> None:
        """Toggle leader-arm linkage used by teleoperation."""
        enabled = bool(enabled)
        self._leader_arm_enabled = enabled
        if self._is_dummy:
            return
        robot = self._require_connected()
        config_ = getattr(robot, "config_", None)
        if config_ is not None and hasattr(config_, "USE_LEAD_ARMS"):
            setattr(config_, "USE_LEAD_ARMS", enabled)
        if hasattr(robot, "USE_LEAD_ARMS"):
            setattr(robot, "USE_LEAD_ARMS", enabled)
        if hasattr(robot, "use_lead_arms"):
            setattr(robot, "use_lead_arms", enabled)

    def get_left_joint(self) -> np.ndarray:
        """Return left follower arm state ``(7,)``."""
        if self._is_dummy:
            return np.zeros(7)
        robot = self._require_connected()
        return np.asarray(
            self._get_robot_joint(robot, getter_name="left_get_joint"),
            dtype=np.float64,
        )

    def get_right_joint(self) -> np.ndarray:
        """Return right follower arm state ``(7,)``."""
        if self._is_dummy:
            return np.zeros(7)
        robot = self._require_connected()
        return np.asarray(
            self._get_robot_joint(robot, getter_name="right_get_joint"),
            dtype=np.float64,
        )

    def get_state(self) -> DOSW1RobotState:
        """Return a unified follower-arm state snapshot."""
        left = self.get_left_joint()
        right = self.get_right_joint()
        return DOSW1RobotState(
            left_joint_positions=left[:6].copy(),
            left_gripper=float(left[6]),
            right_joint_positions=right[:6].copy(),
            right_gripper=float(right[6]),
            timestamp=time.time(),
        )

    def open_gripper(self) -> None:
        """Open both grippers while holding current joint positions."""
        if self._is_dummy:
            return
        open_width = float(self._gripper_width_max)
        left = self.get_left_joint()
        right = self.get_right_joint()
        self.left_go_joint(left[:6].tolist(), open_width)
        self.right_go_joint(right[:6].tolist(), open_width)

    def get_left_lead_joint(self) -> np.ndarray:
        """Return left leader arm state ``(7,)``."""
        if self._is_dummy or not self._leader_arm_enabled:
            return np.zeros(7)
        robot = self._require_connected()
        return np.asarray(
            self._get_robot_joint(robot, getter_name="lead_left_get_joint"),
            dtype=np.float64,
        )

    def get_right_lead_joint(self) -> np.ndarray:
        """Return right leader arm state ``(7,)``."""
        if self._is_dummy or not self._leader_arm_enabled:
            return np.zeros(7)
        robot = self._require_connected()
        return np.asarray(
            self._get_robot_joint(robot, getter_name="lead_right_get_joint"),
            dtype=np.float64,
        )

    def left_go_joint(
        self,
        joint: list[float],
        gripper: float,
        *,
        interp: bool = False,
    ) -> None:
        """Command the left follower arm to target joint positions."""
        if self._is_dummy:
            return
        robot = self._require_connected()
        robot.left_go_joint(list(joint), float(gripper), interp=interp)

    def right_go_joint(
        self,
        joint: list[float],
        gripper: float,
        *,
        interp: bool = False,
    ) -> None:
        """Command the right follower arm to target joint positions."""
        if self._is_dummy:
            return
        robot = self._require_connected()
        robot.right_go_joint(list(joint), float(gripper), interp=interp)

    def forward_kinematics(self, joint: list[float]) -> np.ndarray:
        """Compute ee_pose from joint angles via SDK FK (arm-agnostic)."""
        if self._is_dummy:
            return np.zeros(6)
        robot = self._require_connected()
        return np.asarray(robot.fk(joint), dtype=np.float64)

    def get_left_pose(self) -> np.ndarray:
        """Return current left arm ee_pose."""
        if self._is_dummy:
            return np.zeros(6)
        robot = self._require_connected()
        return np.asarray(robot.left_get_pose(), dtype=np.float64)

    def get_right_pose(self) -> np.ndarray:
        """Return current right arm ee_pose."""
        if self._is_dummy:
            return np.zeros(6)
        robot = self._require_connected()
        return np.asarray(robot.right_get_pose(), dtype=np.float64)

    def _require_connected(self) -> Any:
        if not self.is_connected or self._robot is None:
            raise RuntimeError(
                "DOSW1Connection is not connected. Call connect() first."
            )
        return self._robot

    def _wait_for_initial_state(self) -> None:
        deadline = time.time() + _STATE_READY_TIMEOUT_S
        while time.time() < deadline:
            robot = self._require_connected_candidate()
            left_ready = len(self._get_robot_joint(robot, "left_get_joint")) == 7
            right_ready = len(self._get_robot_joint(robot, "right_get_joint")) == 7
            lead_ready = True
            if self._enable_human_in_loop:
                lead_ready = (
                    len(self._get_robot_joint(robot, "lead_left_get_joint")) == 7
                    and len(self._get_robot_joint(robot, "lead_right_get_joint")) == 7
                )
            if left_ready and right_ready and lead_ready:
                return
            time.sleep(_CONTROL_LOOP_DT)
        raise TimeoutError("Timed out waiting for DOSW1 state from AirbotRobot.")

    def _require_connected_candidate(self) -> Any:
        if self._robot is None:
            raise RuntimeError("DOSW1Connection failed to create AirbotRobot.")
        return self._robot

    @staticmethod
    def _get_robot_joint(robot: object, getter_name: str) -> list[float]:
        getter = getattr(robot, getter_name, None)
        try:
            values = getter() if callable(getter) else []
        except Exception:
            return []
        if values is None:
            return []
        return list(values)

    @staticmethod
    def _shutdown_robot(robot: object) -> None:
        setattr(robot, "running", False)

        def _disconnect_arm(arm: object | None) -> None:
            if arm is None:
                return
            try:
                arm.disconnect()
            except Exception:
                pass
            time.sleep(0.5)

        _disconnect_arm(getattr(robot, "left_arm", None))
        _disconnect_arm(getattr(robot, "right_arm", None))

        config_ = getattr(robot, "config_", None)
        use_lead_arms = bool(getattr(config_, "USE_LEAD_ARMS", False))
        if use_lead_arms:
            _disconnect_arm(getattr(robot, "left_lead_arm", None))
            _disconnect_arm(getattr(robot, "right_lead_arm", None))


class DOSW1Arm(Arm):
    """Arm view exported by a shared DOSW1 connection."""

    def __init__(self, sdk: DOSW1Connection, side: str) -> None:
        if side not in {"left", "right"}:
            raise ValueError("DOSW1 arm side must be 'left' or 'right'.")
        self.sdk = self._owner = sdk
        self.side = side

    # The connection names its two arms with distinct methods. Resolving them
    # once, by name, keeps every call site readable to a type checker.

    def _read_joint(self) -> np.ndarray:
        """Return this side's six joints and gripper width as ``(7,)``."""
        if self.side == "left":
            return self.sdk.get_left_joint()
        return self.sdk.get_right_joint()

    def _read_pose(self) -> np.ndarray:
        """Return this side's end-effector pose."""
        if self.side == "left":
            return self.sdk.get_left_pose()
        return self.sdk.get_right_pose()

    def _read_lead_joint(self) -> np.ndarray:
        """Return this side's leader-arm joints."""
        if self.side == "left":
            return self.sdk.get_left_lead_joint()
        return self.sdk.get_right_lead_joint()

    def _go_joint(
        self, joint: list[float], gripper: float, interp: bool = False
    ) -> None:
        """Command this side's joints and gripper in one call."""
        if self.side == "left":
            self.sdk.left_go_joint(joint, gripper, interp=interp)
        else:
            self.sdk.right_go_joint(joint, gripper, interp=interp)

    @property
    def observation_features(self) -> Features:
        """Describe the DOSW1 joint, gripper, pose and leader-arm state.

        The gripper travels beside the joints here as it does in the action,
        so a caller reads both halves of one ``go_joint`` in one round trip.
        ``lead_joint_position`` is the operator's leader arm for this side,
        and reads as zeros while the leader link is disabled.
        """
        return {
            "joint_position": {"shape": (6,), "dtype": "float64"},
            "gripper_width": {"shape": (1,), "dtype": "float64"},
            "tcp_pose": {"shape": (6,), "dtype": "float64"},
            "lead_joint_position": {"shape": (7,), "dtype": "float64"},
        }

    @property
    def action_features(self) -> Features:
        """Describe the absolute joint-position and gripper command.

        The hardware takes joints and gripper width in one call, so both
        travel in one action. Omitting ``gripper_width`` holds the current
        width, which is what a caller that only moves the arm wants.
        """
        return {
            "joint_position": {"shape": (6,), "dtype": "float64"},
            "gripper_width": {"shape": (1,), "dtype": "float64"},
        }

    def reset(self) -> None:
        """Leave reset targets to the task configuration."""

    def get_observation(self) -> Observation:
        """Read this arm's joints, gripper, pose and leader arm."""
        current = self._read_joint()
        return {
            "joint_position": current[:6].copy(),
            "gripper_width": current[6:7].copy(),
            "tcp_pose": self._read_pose().copy(),
            "lead_joint_position": self._read_lead_joint().copy(),
        }

    def set_leader_arm_enabled(self, enabled: bool) -> None:
        """Link or unlink the operator's leader arms.

        One switch covers the rig: both arms share a session, so either one
        answers for it. Teleoperation needs the link, and a policy rollout
        does not.
        """
        self.sdk.set_leader_arm_enabled(enabled)

    def reset_joint(
        self,
        positions: "Sequence[float]",
        gripper_width: Optional[float] = None,
    ) -> None:
        """Travel to a configuration, interpolated, outside the action stream.

        A reset moves further than one control step, so the SDK interpolates
        it. The gripper rides along for the same reason it does in
        :meth:`send_action`: two commands would countermand each other.
        """
        current = self._read_joint()
        width = float(current[6]) if gripper_width is None else float(gripper_width)
        self._go_joint(list(positions), width, interp=True)

    def forward_kinematics(self, joint_position: "Sequence[float]") -> np.ndarray:
        """Return the end-effector pose these joints would reach."""
        return self.sdk.forward_kinematics(list(joint_position))

    def send_action(self, action: Action) -> Observation:
        """Apply an absolute joint target, and the gripper width beside it.

        Both reach the arm in a single ``go_joint`` call. Commanding them
        separately would make each command read the other half back and
        restate it, so the second would countermand the first mid-motion.
        """
        unknown = set(action) - {"joint_position", "gripper_width"}
        if unknown or "joint_position" not in action:
            raise KeyError(
                "DOSW1 arm action takes 'joint_position' and optionally "
                f"'gripper_width'; got {sorted(action)}."
            )
        target = np.asarray(action["joint_position"], dtype=np.float64).reshape(6)
        if "gripper_width" in action:
            width = float(
                np.asarray(action["gripper_width"], dtype=np.float64).reshape(1)[0]
            )
        else:
            width = float(self._read_joint()[6])
        self._go_joint(target.tolist(), width)
        return {"joint_position": target, "gripper_width": np.asarray([width])}


class DOSW1EndEffector(EndEffector):
    """Gripper view exported by a shared DOSW1 connection."""

    def __init__(self, sdk: DOSW1Connection, side: str) -> None:
        if side not in {"left", "right"}:
            raise ValueError("DOSW1 gripper side must be 'left' or 'right'.")
        self.sdk = self._owner = sdk
        self.side = side

    @property
    def state_dim(self) -> int:
        """Return the scalar gripper state dimension."""
        return 1

    @property
    def action_dim(self) -> int:
        """Return the scalar gripper action dimension."""
        return 1

    @property
    def control_mode(self) -> str:
        """Return the continuous-width control mode."""
        return "continuous"

    def reset(self) -> None:
        """Leave reset width to the task configuration."""

    def _read_joint(self) -> np.ndarray:
        """Return this side's six joints and gripper width as ``(7,)``."""
        if self.side == "left":
            return self.sdk.get_left_joint()
        return self.sdk.get_right_joint()

    def get_state(self) -> np.ndarray:
        """Read this gripper's current width."""
        return np.asarray(self._read_joint()[6:7], dtype=np.float64)

    def command(self, action: np.ndarray) -> bool:
        """Apply a gripper target while retaining joint positions."""
        target = np.asarray(action, dtype=np.float64).reshape(1)
        current = self._read_joint()
        if self.side == "left":
            self.sdk.left_go_joint(current[:6].tolist(), float(target[0]))
        else:
            self.sdk.right_go_joint(current[:6].tolist(), float(target[0]))
        return True
