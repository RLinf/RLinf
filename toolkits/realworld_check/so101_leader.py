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

"""LeRobot SO-101 主臂读取适配。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.robotics.parts.arms.so101 import SO101Arm

SO101_JOINT_NAMES = (*SO101Arm.MOTORS, SO101Arm.GRIPPER)


def lerobot_to_rlinf_positions(joint_positions: np.ndarray) -> np.ndarray:
    """Convert LeRobot degrees/percent positions to the legacy normalized vector."""
    values = np.asarray(joint_positions, dtype=np.float32).copy()
    if values.shape != (6,):
        raise ValueError(f"SO-101 positions must have shape (6,), got {values.shape}")
    values[:-1] /= 100.0
    values[-1] = values[-1] / 50.0 - 1.0
    return np.clip(values, -1.0, 1.0).astype(np.float32)


class SO101LeaderExpert:
    """同步读取 LeRobot SO-101 主臂并输出 RLinf 归一化动作。"""

    def __init__(
        self,
        port: str,
        robot_id: str,
        calibration_dir: str | None,
        leader: Any | None = None,
    ):
        self._leader = leader
        # 主臂串口只允许一个线程访问。DAgger 控制线程持续读取位置，主线程在
        # 保存、复位和退出时还会写入目标；统一加锁可避免 Feetech 返回 Port is in use。
        self._serial_lock = threading.RLock()
        if self._leader is None:
            try:
                from lerobot.teleoperators.so_leader import (
                    SO101Leader,
                    SO101LeaderConfig,
                )
            except ImportError as exc:
                raise ImportError(
                    "SO-101 主臂采集需要安装固定版本的 LeRobot。"
                ) from exc

            config = SO101LeaderConfig(
                port=port,
                id=robot_id,
                calibration_dir=(
                    Path(calibration_dir) if calibration_dir is not None else None
                ),
                use_degrees=False,
            )
            self._leader = SO101Leader(config)

        calibration_path = getattr(self._leader, "calibration_fpath", None)
        if calibration_path is not None and not Path(calibration_path).is_file():
            raise FileNotFoundError(f"找不到 SO-101 主臂校准文件：{calibration_path}")
        try:
            self._leader.connect(calibrate=False)
            if not bool(getattr(self._leader, "is_calibrated", True)):
                raise RuntimeError(
                    "SO-101 主臂未完成中位校准，请先使用 LeRobot 校准工具完成校准。"
                )
        except Exception:
            if bool(getattr(self._leader, "is_connected", False)):
                self._leader.disconnect()
            raise

    def get_action(self) -> np.ndarray:
        """读取主臂当前位置并转换为 RLinf 六维绝对动作。"""
        with self._serial_lock:
            observation = self._read_action_with_retry()
        native_positions = np.asarray(
            [observation[f"{name}.pos"] for name in SO101_JOINT_NAMES],
            dtype=np.float32,
        )
        return lerobot_to_rlinf_positions(native_positions)

    @staticmethod
    def _pose_value(pose: dict[str, float], name: str) -> float:
        """读取两种姿态文件键名，兼容 ``joint`` 与 ``joint.pos``。"""
        if name in pose:
            return float(pose[name])
        native_name = f"{name}.pos"
        if native_name in pose:
            return float(pose[native_name])
        raise KeyError(f"姿态缺少关节 {name}（支持键名 {name} 或 {native_name}）")

    def get_native_action(self) -> dict[str, float]:
        """读取主臂的 LeRobot 原生位置，供切换对齐阶段使用。"""
        with self._serial_lock:
            observation = self._read_action_with_retry()
        return {
            f"{name}.pos": float(observation[f"{name}.pos"])
            for name in SO101_JOINT_NAMES
        }

    def move_to_native_action(
        self,
        target: dict[str, float],
        duration_s: float = 1.0,
        fps: float = 30.0,
        release_torque: bool = True,
    ) -> None:
        """将主臂平滑移动到目标姿态。

        目标使用 LeRobot 原生单位。调用方可选择在移动结束后关闭扭矩，
        适用于 DAgger 对齐和退出时收臂两种场景。
        """
        if duration_s < 0 or fps <= 0:
            raise ValueError("duration_s must be non-negative and fps must be positive")
        with self._serial_lock:
            current = self._read_action_with_retry()
            current_values = np.asarray(
                [current[f"{name}.pos"] for name in SO101_JOINT_NAMES], dtype=np.float32
            )
            target_values = np.asarray(
                [self._pose_value(target, name) for name in SO101_JOINT_NAMES],
                dtype=np.float32,
            )
            steps = max(1, int(round(duration_s * fps)))
            period = 1.0 / fps
            self._leader.bus.enable_torque()
            deadline = time.monotonic()
            for index in range(1, steps + 1):
                alpha = index / steps
                # smoothstep，降低对齐阶段的瞬时速度。
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                values = (1.0 - alpha) * current_values + alpha * target_values
                self._leader.bus.sync_write(
                    "Goal_Position",
                    {
                        name: float(value)
                        for name, value in zip(SO101_JOINT_NAMES, values, strict=True)
                    },
                )
                deadline += period
                time.sleep(max(0.0, deadline - time.monotonic()))
            if release_torque:
                self._leader.bus.disable_torque()

    def align_to_native_action(
        self,
        target: dict[str, float],
        duration_s: float = 1.0,
        fps: float = 30.0,
    ) -> None:
        """将主臂平滑移动到从臂当前姿态，再交给人工操作。"""
        self.move_to_native_action(
            target, duration_s=duration_s, fps=fps, release_torque=False
        )

    def hold_current(self) -> dict[str, float]:
        """以当前位置重新上扭矩，供保存、放弃和退出前承接主臂。"""
        with self._serial_lock:
            current = self._read_action_with_retry()
            current = {
                f"{name}.pos": float(current[f"{name}.pos"])
                for name in SO101_JOINT_NAMES
            }
            self._leader.bus.enable_torque()
            self._leader.bus.sync_write(
                "Goal_Position",
                {name: self._pose_value(current, name) for name in SO101_JOINT_NAMES},
            )
            return current

    def release_for_manual(self) -> None:
        """关闭主臂扭矩，让操作者接管已对齐的主臂。"""
        with self._serial_lock:
            self._leader.bus.disable_torque()

    def close(self) -> None:
        """断开主臂串口。"""
        with self._serial_lock:
            if bool(getattr(self._leader, "is_connected", False)):
                self._leader.disconnect()

    def _read_action_with_retry(self) -> dict[str, float]:
        """读取主臂位置；设备切换瞬间占用串口时短暂重试。"""
        for attempt in range(3):
            try:
                return self._leader.get_action()
            except ConnectionError as exc:
                if "Port is in use" not in str(exc) or attempt == 2:
                    raise
                time.sleep(0.03)
        raise RuntimeError("主臂位置读取重试异常结束")


# DAgger 上层使用的语义名称；实现仍复用 LeRobot 主臂读取器。
LeaderHandover = SO101LeaderExpert
