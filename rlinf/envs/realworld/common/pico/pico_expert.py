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

"""PICO expert input for real-world Franka teleoperation.

The PICO service and publisher still live outside RLinf.  This module only
subscribes to the existing JSON/ZeroMQ stream and converts controller motion
into the same base-frame action convention used by FrankaEnv.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

try:
    import zmq
except ImportError:  # pragma: no cover - optional runtime dependency
    zmq = None


R_PICO_TO_WORLD = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
R_PICO_TO_WORLD_ROT = R.from_matrix(R_PICO_TO_WORLD)


def _axis_vector(value: Any, default: list[float]) -> np.ndarray:
    if isinstance(value, str):
        key = value.strip().lower()
        axes = {
            "+x": [1.0, 0.0, 0.0],
            "x": [1.0, 0.0, 0.0],
            "-x": [-1.0, 0.0, 0.0],
            "+y": [0.0, 1.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "-y": [0.0, -1.0, 0.0],
            "+z": [0.0, 0.0, 1.0],
            "z": [0.0, 0.0, 1.0],
            "-z": [0.0, 0.0, -1.0],
        }
        value = axes.get(key, default)

    vec = np.array(value if value is not None else default, dtype=np.float64)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)) or np.linalg.norm(vec) < 1e-9:
        vec = np.array(default, dtype=np.float64)
    return vec / np.linalg.norm(vec)


def _is_valid_pose(pose: Any) -> bool:
    if pose is None or len(pose) < 7:
        return False
    arr = np.array(pose[:7], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return False
    quat = arr[3:7]
    return abs(np.linalg.norm(quat) - 1.0) < 1e-3


def _button_name(side: str, name: str) -> str:
    if name == "menu_button":
        return f"{side}_menu_button"
    return name


class PicoExpert:
    """Read PICO controller data and produce Franka base-frame actions."""

    def __init__(
        self,
        *,
        zmq_addr: Optional[str] = None,
        zmq_config: Optional[Mapping[str, Any]] = None,
        ipc_addr: Optional[str] = None,
        hand: str = "right",
        control_trigger: str = "grip",
        control_threshold: float = 0.9,
        gripper_trigger: str = "trigger",
        gripper_close_threshold: float = 0.5,
        gripper_close_button: str = "A",
        gripper_open_button: str = "B",
        gripper_invert: bool = False,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        operator_to_robot_yaw: float = 0.0,
        timeout_ms: int = 1000,
        reconnect_interval_ms: int = 500,
        max_stale_s: float = 0.25,
        calibration: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if zmq_config is None:
            zmq_cfg = {}
        else:
            zmq_cfg = dict(zmq_config)

        if zmq_addr is None:
            zmq_addr = ipc_addr or zmq_cfg.get("ipc_addr", "ipc:///tmp/vr_data.ipc")

        hand = hand.lower()
        if hand not in ("left", "right"):
            raise ValueError("PicoExpert hand must be 'left' or 'right'.")

        self.zmq_addr = zmq_addr
        self.hand = hand
        self.control_trigger = control_trigger
        self.control_threshold = float(control_threshold)
        self.gripper_trigger = gripper_trigger
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.gripper_close_button = gripper_close_button
        self.gripper_open_button = gripper_open_button
        self.gripper_invert = bool(gripper_invert)
        self.position_scale = float(position_scale)
        self.rotation_scale = float(rotation_scale)
        self.operator_to_robot_yaw = float(operator_to_robot_yaw)
        self._operator_to_robot_rot = R.from_euler("z", self.operator_to_robot_yaw)
        self.timeout_ms = int(zmq_cfg.get("timeout_ms", timeout_ms))
        self.reconnect_interval_s = (
            float(zmq_cfg.get("reconnect_interval_ms", reconnect_interval_ms)) / 1000.0
        )
        self.max_stale_s = float(max_stale_s)

        calibration_cfg = dict(calibration or {})
        self.calibration_enabled = bool(calibration_cfg.get("enabled", True))
        self.auto_calibrate_on_start = bool(
            calibration_cfg.get("auto_calibrate_on_start", True)
        )
        self.require_calibration = bool(
            calibration_cfg.get("required", self.calibration_enabled)
        )
        self.calibration_button = calibration_cfg.get("button", "B")
        self.calibration_threshold = float(calibration_cfg.get("threshold", 0.5))
        self.allow_calibration_while_active = bool(
            calibration_cfg.get("allow_while_active", False)
        )
        self.base_position = np.array(
            calibration_cfg.get("base_position", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        if self.base_position.shape != (3,) or not np.all(
            np.isfinite(self.base_position)
        ):
            self.base_position = np.zeros(3, dtype=np.float64)
        self.head_forward_axis = _axis_vector(
            calibration_cfg.get("head_forward_axis", "-z"),
            [0.0, 0.0, -1.0],
        )

        self._lock = threading.Lock()
        self._latest_data: Optional[dict[str, Any]] = None
        self._last_update_time = 0.0

        self._context = None
        self._socket = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._active = False
        self._ref_controller_pos: Optional[np.ndarray] = None
        self._ref_controller_rot: Optional[R] = None
        self._ref_tcp_pos: Optional[np.ndarray] = None
        self._ref_tcp_rot: Optional[R] = None

        self._calibrated = False
        self._calibration_rot = R.identity()
        self._calibration_t = np.zeros(3, dtype=np.float64)
        self._prev_calibration_button = False
        self._last_action = np.zeros(7, dtype=np.float32)
        self._last_info: dict[str, Any] = {}

        self.start()

    @property
    def ready(self) -> bool:
        with self._lock:
            return (
                self._latest_data is not None
                and time.time() - self._last_update_time <= self.max_stale_s
            )

    def start(self) -> None:
        if zmq is None:
            raise ImportError(
                "pyzmq is required for PICO teleoperation. Install with: pip install pyzmq"
            )
        if self._running:
            return

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.set_hwm(10)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(self.zmq_addr)

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info("PICO expert connected to %s", self.zmq_addr)

    def stop(self) -> None:
        self._running = False
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._context is not None:
            self._context.term()
            self._context = None

    def get_action(
        self,
        tcp_pose: np.ndarray,
        action_scale: np.ndarray,
        *,
        gripper_enabled: bool = True,
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        data = self._snapshot()
        if data is None:
            self._deactivate()
            return (
                self._last_action.copy(),
                False,
                {
                    "pico_active": False,
                    "pico_ready": False,
                    "pico_stale": True,
                    "pico_hand": self.hand,
                },
            )

        self._maybe_update_calibration(data)
        if (
            self.calibration_enabled
            and self.require_calibration
            and not self._calibrated
        ):
            self._deactivate()
            return (
                self._last_action.copy(),
                False,
                {
                    "pico_active": False,
                    "pico_ready": True,
                    "pico_calibrated": False,
                    "pico_hand": self.hand,
                },
            )

        controller_pose = self._controller_pose(data, self.hand)
        if not _is_valid_pose(controller_pose):
            self._deactivate()
            return (
                self._last_action.copy(),
                False,
                {
                    "pico_active": False,
                    "pico_ready": True,
                    "pico_invalid_pose": True,
                    "pico_hand": self.hand,
                },
            )

        control_value = self._control_value(data, self.hand, self.control_trigger)
        active = control_value >= self.control_threshold
        controller_pos, controller_rot = self._transform_raw_pose_to_world(
            controller_pose
        )

        if active and not self._active:
            self._activate(controller_pos, controller_rot, tcp_pose)
        elif not active and self._active:
            self._deactivate()
        self._active = active

        info = {
            "pico_active": active,
            "pico_ready": True,
            "pico_hand": self.hand,
            "pico_control_value": control_value,
            "pico_calibrated": self._calibrated,
        }
        if not active:
            self._last_info = info
            return self._last_action.copy(), False, info

        target_pos, target_rot = self._target_tcp_pose(controller_pos, controller_rot)
        current_pos = np.asarray(tcp_pose[:3], dtype=np.float64)
        current_rot = R.from_quat(np.asarray(tcp_pose[3:7], dtype=np.float64))
        action_scale = np.asarray(action_scale, dtype=np.float64)

        delta_pos = (target_pos - current_pos) / float(action_scale[0])
        delta_rot = target_rot * current_rot.inv()
        max_rot = float(action_scale[1])
        delta_rotvec = delta_rot.as_rotvec()
        if max_rot > 1e-9:
            angle = float(np.linalg.norm(delta_rotvec))
            if angle > max_rot:
                delta_rotvec = delta_rotvec * (max_rot / angle)
            delta_rot_action = delta_rotvec / max_rot
        else:
            delta_rot_action = np.zeros(3, dtype=np.float64)
        expert_action = np.concatenate((delta_pos, delta_rot_action), axis=0)
        expert_action = np.clip(expert_action, -1.0, 1.0)

        gripper_close = False
        if gripper_enabled:
            close_pressed = self._control_pressed(
                data,
                self.hand,
                self.gripper_close_button,
                self.gripper_close_threshold,
            )
            open_pressed = self._control_pressed(
                data,
                self.hand,
                self.gripper_open_button,
                self.gripper_close_threshold,
            )
            if self.gripper_invert:
                close_pressed, open_pressed = open_pressed, close_pressed

            gripper_action = 0.0
            if close_pressed:
                gripper_action = -1.0
            elif open_pressed:
                gripper_action = 1.0
            gripper_close = gripper_action < 0.0
            expert_action = np.concatenate(
                (expert_action, np.array([gripper_action], dtype=np.float64)),
                axis=0,
            )
            info["pico_gripper_close_pressed"] = close_pressed
            info["pico_gripper_open_pressed"] = open_pressed
            info["pico_gripper_action"] = gripper_action
            info["pico_gripper_close"] = gripper_close

        expert_action = expert_action.astype(np.float32)
        self._last_action = expert_action.copy()
        self._last_info = info
        return expert_action, True, info

    def _recv_loop(self) -> None:
        while self._running:
            try:
                msg_bytes = self._socket.recv()
                data = json.loads(msg_bytes.decode("utf-8"))
                with self._lock:
                    self._latest_data = data
                    self._last_update_time = time.time()
            except zmq.error.Again:
                continue
            except Exception as exc:
                if not self._running:
                    break
                logger.warning("Error receiving PICO data: %s", exc)
                time.sleep(self.reconnect_interval_s)

    def _snapshot(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if self._latest_data is None:
                return None
            if time.time() - self._last_update_time > self.max_stale_s:
                return None
            return dict(self._latest_data)

    def _activate(
        self,
        controller_pos: np.ndarray,
        controller_rot: R,
        tcp_pose: np.ndarray,
    ) -> None:
        self._ref_controller_pos = controller_pos.copy()
        self._ref_controller_rot = controller_rot
        self._ref_tcp_pos = np.asarray(tcp_pose[:3], dtype=np.float64).copy()
        self._ref_tcp_rot = R.from_quat(np.asarray(tcp_pose[3:7], dtype=np.float64))
        logger.info("PICO %s controller activated", self.hand)

    def _deactivate(self) -> None:
        if self._active:
            logger.info("PICO %s controller deactivated", self.hand)
        self._active = False
        self._ref_controller_pos = None
        self._ref_controller_rot = None
        self._ref_tcp_pos = None
        self._ref_tcp_rot = None

    def _target_tcp_pose(
        self,
        controller_pos: np.ndarray,
        controller_rot: R,
    ) -> tuple[np.ndarray, R]:
        if (
            self._ref_controller_pos is None
            or self._ref_controller_rot is None
            or self._ref_tcp_pos is None
            or self._ref_tcp_rot is None
        ):
            raise RuntimeError("PICO reference poses are not initialized.")

        delta_world = controller_pos - self._ref_controller_pos
        command_delta = delta_world * self.position_scale
        command_delta = self._operator_vector_to_robot_vector(command_delta)
        target_pos = self._ref_tcp_pos + command_delta

        delta_local_rot = self._ref_controller_rot.inv() * controller_rot
        command_rotvec = (
            self._operator_vector_to_robot_vector(
                self._controller_local_vector_to_command_vector(
                    delta_local_rot.as_rotvec()
                )
            )
            * self.rotation_scale
        )
        target_rot = R.from_rotvec(command_rotvec) * self._ref_tcp_rot
        return target_pos, target_rot

    def _maybe_update_calibration(self, data: Mapping[str, Any]) -> None:
        if not self.calibration_enabled:
            return

        if self.auto_calibrate_on_start and not self._calibrated:
            self._calibrate_base(data)

        if not self.calibration_button:
            return

        pressed = self._control_pressed(
            data,
            self.hand,
            self.calibration_button,
            self.calibration_threshold,
        )
        if pressed and not self._prev_calibration_button:
            if self._active and not self.allow_calibration_while_active:
                logger.warning("Ignoring PICO calibration while control is active")
            elif not self._calibrate_base(data):
                logger.warning(
                    "PICO calibration requested, but headset pose is invalid"
                )
        self._prev_calibration_button = pressed

    def _calibrate_base(self, data: Mapping[str, Any]) -> bool:
        headset_pose = data.get("headset_pose", [0.0] * 7)
        if not _is_valid_pose(headset_pose):
            return False

        head_pos, head_rot = self._transform_raw_pose_to_aligned(headset_pose)
        forward = head_rot.apply(self.head_forward_axis)
        forward[2] = 0.0

        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return False

        forward /= norm
        yaw = float(np.arctan2(forward[1], forward[0]))
        self._calibration_rot = R.from_euler("z", -yaw)
        self._calibration_t = self.base_position - self._calibration_rot.apply(head_pos)
        self._calibrated = True
        self._deactivate()
        logger.info(
            "PICO base calibrated: head_pos=%s, align_yaw=%.3f",
            np.round(head_pos, 4).tolist(),
            -yaw,
        )
        return True

    def _transform_raw_pose_to_aligned(self, pose: list[float]) -> tuple[np.ndarray, R]:
        pos = R_PICO_TO_WORLD @ np.asarray(pose[:3], dtype=np.float64)
        rot = R_PICO_TO_WORLD_ROT * R.from_quat(np.asarray(pose[3:7], dtype=np.float64))
        return pos, rot

    def _transform_raw_pose_to_world(self, pose: list[float]) -> tuple[np.ndarray, R]:
        pos, rot = self._transform_raw_pose_to_aligned(pose)
        if self.calibration_enabled:
            pos = self._calibration_rot.apply(pos) + self._calibration_t
            rot = self._calibration_rot * rot
        return pos, rot

    @staticmethod
    def _controller_local_vector_to_command_vector(vector: np.ndarray) -> np.ndarray:
        return np.array([-vector[2], -vector[0], vector[1]], dtype=np.float64)

    def _operator_vector_to_robot_vector(self, vector: np.ndarray) -> np.ndarray:
        return self._operator_to_robot_rot.apply(np.asarray(vector, dtype=np.float64))

    @staticmethod
    def _controller_pose(data: Mapping[str, Any], side: str) -> list[float]:
        controller = data.get(f"{side}_controller", {})
        return list(controller.get("position", [0.0, 0.0, 0.0])) + list(
            controller.get("orientation", [0.0, 0.0, 0.0, 1.0])
        )

    @staticmethod
    def _button(data: Mapping[str, Any], side: str, name: str) -> bool:
        buttons = data.get("buttons", {})
        return bool(buttons.get(_button_name(side, name), False))

    def _control_value(self, data: Mapping[str, Any], side: str, name: str) -> float:
        name = name.strip()
        controller = data.get(f"{side}_controller", {})
        if name in ("grip", "trigger"):
            return float(controller.get(name, 0.0))
        return 1.0 if self._button(data, side, name) else 0.0

    def _control_pressed(
        self,
        data: Mapping[str, Any],
        side: str,
        name: Optional[str],
        threshold: float,
    ) -> bool:
        if not name:
            return False
        return self._control_value(data, side, str(name)) >= float(threshold)

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.stop()
        except Exception:
            pass
