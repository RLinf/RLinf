# Copyright 2025 The RLinf Authors.
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
from typing import TYPE_CHECKING

import numpy as np

from rlinf.utils.logging import get_logger

from .dosw1_robot_state import DOSW1RobotState

if TYPE_CHECKING:
    from .dosw1_env import DOSW1Config

try:
    from airbot_sdk.Airbot import AirbotRobot as _AirbotRobot
    from airbot_sdk.configs.config import DosW1Config as _AirbotSDKConfig
except ImportError:
    _AirbotRobot = None
    _AirbotSDKConfig = None

_CONTROL_LOOP_DT = 0.02
_STATE_READY_TIMEOUT_S = 5.0


class DOSW1SDKAdapter:
    """Thin wrapper around ``airbot_sdk.AirbotRobot`` for RLinf."""

    def __init__(self, config: "DOSW1Config") -> None:
        self._logger = get_logger()
        self._config = config
        self._leader_arm_enabled = bool(config.enable_human_in_loop)
        self._connected = False
        self._robot: object | None = None

    def connect(self) -> None:
        """Connect to follower and optional leader arms."""
        if self._config.is_dummy:
            self._connected = True
            return

        if _AirbotRobot is None or _AirbotSDKConfig is None:
            raise ImportError(
                "airbot_sdk is not installed. Install it or set is_dummy=True."
            )

        cfg = self._config
        sdk_cfg = _AirbotSDKConfig()
        sdk_cfg.USE_CAM = False
        sdk_cfg.USE_CAR = False
        sdk_cfg.USE_LEAD_ARMS = bool(cfg.enable_human_in_loop)

        self._logger.info(
            "[DOSW1SDK] Connecting via AirbotRobot (url=%s, ports=%s/%s/%s/%s) ...",
            cfg.robot_url,
            cfg.left_arm_port,
            cfg.right_arm_port,
            cfg.left_lead_port,
            cfg.right_lead_port,
        )

        self._robot = _AirbotRobot(
            config_=sdk_cfg,
            left_lead_port=cfg.left_lead_port,
            left_lead_url=cfg.robot_url,
            right_lead_port=cfg.right_lead_port,
            right_lead_url=cfg.robot_url,
            left_port=cfg.left_arm_port,
            left_url=cfg.robot_url,
            right_port=cfg.right_arm_port,
            right_url=cfg.robot_url,
        )
        try:
            self._wait_for_initial_state()
        except Exception:
            self._shutdown_robot(self._robot)
            self._robot = None
            raise
        self._connected = True
        self._logger.info("[DOSW1SDK] Connected.")

    def disconnect(self) -> None:
        """Disconnect the wrapped AirbotRobot instance."""
        self._logger.info("[DOSW1SDK] Disconnecting.")
        robot = self._robot
        self._robot = None
        self._connected = False
        if robot is None:
            return

        try:
            self._shutdown_robot(robot)
        except Exception:
            self._logger.exception("[DOSW1SDK] Failed to disconnect cleanly")

    def set_leader_arm_enabled(self, enabled: bool) -> None:
        """Toggle leader-arm linkage used by teleoperation."""
        enabled = bool(enabled)
        self._leader_arm_enabled = enabled
        if self._config.is_dummy:
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
        if self._config.is_dummy:
            return np.zeros(7)
        robot = self._require_connected()
        return np.asarray(
            self._get_robot_joint(robot, getter_name="left_get_joint"),
            dtype=np.float64,
        )

    def get_right_joint(self) -> np.ndarray:
        """Return right follower arm state ``(7,)``."""
        if self._config.is_dummy:
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
        if self._config.is_dummy:
            return
        open_width = float(getattr(self._config, "gripper_width_max", 0.07))
        left = self.get_left_joint()
        right = self.get_right_joint()
        self.left_go_joint(left[:6].tolist(), open_width)
        self.right_go_joint(right[:6].tolist(), open_width)

    def get_left_lead_joint(self) -> np.ndarray:
        """Return left leader arm state ``(7,)``."""
        if self._config.is_dummy or not self._leader_arm_enabled:
            return np.zeros(7)
        robot = self._require_connected()
        return np.asarray(
            self._get_robot_joint(robot, getter_name="lead_left_get_joint"),
            dtype=np.float64,
        )

    def get_right_lead_joint(self) -> np.ndarray:
        """Return right leader arm state ``(7,)``."""
        if self._config.is_dummy or not self._leader_arm_enabled:
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
        if self._config.is_dummy:
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
        if self._config.is_dummy:
            return
        robot = self._require_connected()
        robot.right_go_joint(list(joint), float(gripper), interp=interp)

    def forward_kinematics(self, joint: list[float]) -> np.ndarray:
        """Compute ee_pose from joint angles via SDK FK (arm-agnostic)."""
        if self._config.is_dummy:
            return np.zeros(6)
        robot = self._require_connected()
        return np.asarray(robot.fk(joint), dtype=np.float64)

    def get_left_pose(self) -> np.ndarray:
        """Return current left arm ee_pose."""
        if self._config.is_dummy:
            return np.zeros(6)
        robot = self._require_connected()
        return np.asarray(robot.left_get_pose(), dtype=np.float64)

    def get_right_pose(self) -> np.ndarray:
        """Return current right arm ee_pose."""
        if self._config.is_dummy:
            return np.zeros(6)
        robot = self._require_connected()
        return np.asarray(robot.right_get_pose(), dtype=np.float64)

    def _require_connected(self) -> object:
        if not self._connected or self._robot is None:
            raise RuntimeError(
                "DOSW1SDKAdapter is not connected. Call connect() first."
            )
        return self._robot

    def _wait_for_initial_state(self) -> None:
        deadline = time.time() + _STATE_READY_TIMEOUT_S
        while time.time() < deadline:
            robot = self._require_connected_candidate()
            left_ready = len(self._get_robot_joint(robot, "left_get_joint")) == 7
            right_ready = len(self._get_robot_joint(robot, "right_get_joint")) == 7
            lead_ready = True
            if self._config.enable_human_in_loop:
                lead_ready = (
                    len(self._get_robot_joint(robot, "lead_left_get_joint")) == 7
                    and len(self._get_robot_joint(robot, "lead_right_get_joint")) == 7
                )
            if left_ready and right_ready and lead_ready:
                return
            time.sleep(_CONTROL_LOOP_DT)
        raise TimeoutError("Timed out waiting for DOSW1 state from AirbotRobot.")

    def _require_connected_candidate(self) -> object:
        if self._robot is None:
            raise RuntimeError("DOSW1SDKAdapter failed to create AirbotRobot.")
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
