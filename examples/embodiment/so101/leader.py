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

"""LeRobot leader-arm adapter for the SO-101 example."""

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
    """Read a LeRobot SO-101 leader and return normalized RLinf actions."""

    def __init__(
        self,
        port: str,
        robot_id: str,
        calibration_dir: str | None,
        leader: Any | None = None,
    ):
        self._leader = leader
        # DAgger reads continuously while save, reset, and shutdown may write.
        # Serialize those calls because the Feetech bus allows one caller.
        self._serial_lock = threading.RLock()
        if self._leader is None:
            try:
                from lerobot.teleoperators.so_leader import (
                    SO101Leader,
                    SO101LeaderConfig,
                )
            except ImportError as exc:
                raise ImportError("SO-101 DAgger requires the pinned LeRobot.") from exc

            config = SO101LeaderConfig(
                port=port,
                id=robot_id,
                calibration_dir=(
                    Path(calibration_dir) if calibration_dir is not None else None
                ),
                # The conversion helpers below consume LeRobot's native
                # degree/percent representation, not its normalized mode.
                use_degrees=True,
            )
            self._leader = SO101Leader(config)

        calibration_path = getattr(self._leader, "calibration_fpath", None)
        if calibration_path is not None and not Path(calibration_path).is_file():
            raise FileNotFoundError(
                f"SO-101 leader calibration file is missing: {calibration_path}"
            )
        try:
            self._leader.connect(calibrate=False)
            if not bool(getattr(self._leader, "is_calibrated", True)):
                raise RuntimeError(
                    "Calibrate the SO-101 leader with LeRobot before starting DAgger."
                )
        except Exception:
            if bool(getattr(self._leader, "is_connected", False)):
                self._leader.disconnect()
            raise

    def get_action(self) -> np.ndarray:
        """Read the leader and convert it to a normalized six-axis action."""
        with self._serial_lock:
            observation = self._read_action_with_retry()
        native_positions = np.asarray(
            [observation[f"{name}.pos"] for name in SO101_JOINT_NAMES],
            dtype=np.float32,
        )
        return lerobot_to_rlinf_positions(native_positions)

    @staticmethod
    def _pose_value(pose: dict[str, float], name: str) -> float:
        """Read either the ``joint`` or ``joint.pos`` pose representation."""
        if name in pose:
            return float(pose[name])
        native_name = f"{name}.pos"
        if native_name in pose:
            return float(pose[native_name])
        raise KeyError(
            f"Pose is missing {name}; accepted keys are {name} and {native_name}"
        )

    def get_native_action(self) -> dict[str, float]:
        """Read leader positions in native LeRobot units for alignment."""
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
        """Move the leader smoothly to a target in native LeRobot units."""
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
                # Smoothstep limits abrupt velocity changes during alignment.
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
        """Align the leader with the follower before manual control."""
        self.move_to_native_action(
            target, duration_s=duration_s, fps=fps, release_torque=False
        )

    def hold_current(self) -> dict[str, float]:
        """Enable torque and hold the leader at its current position."""
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
        """Disable leader torque for manual control."""
        with self._serial_lock:
            self._leader.bus.disable_torque()

    def close(self) -> None:
        """Disconnect the leader serial bus."""
        with self._serial_lock:
            if bool(getattr(self._leader, "is_connected", False)):
                self._leader.disconnect()

    def _read_action_with_retry(self) -> dict[str, float]:
        """Read the leader, retrying transient serial-bus contention."""
        for attempt in range(3):
            try:
                return self._leader.get_action()
            except ConnectionError as exc:
                if "Port is in use" not in str(exc) or attempt == 2:
                    raise
                time.sleep(0.03)
        raise RuntimeError("SO-101 leader read retries ended unexpectedly")


# Semantic alias used by the DAgger controller.
LeaderHandover = SO101LeaderExpert
