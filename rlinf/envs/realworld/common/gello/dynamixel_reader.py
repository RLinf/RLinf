# Copyright 2023 Philipp Wu
# Copyright 2026 The RLinf Authors
#
# Adapted from GELLO under the MIT License. See GELLO_LICENSE.

"""Minimal passive Dynamixel reader for a seven-joint GELLO device."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class GelloDynamixelReader:
    """Read calibrated GELLO joint positions without a separate ROS node."""

    _BAUDRATE = 57600
    _PROTOCOL_VERSION = 2.0
    _PULSES_PER_REVOLUTION = 4096
    _TORQUE_ENABLE_ADDRESS = 64
    _PRESENT_POSITION_ADDRESS = 132
    _PRESENT_POSITION_LENGTH = 4

    def __init__(
        self,
        port: str,
        joint_signs: Sequence[int],
        joint_offsets: Sequence[float],
        gripper_id: int | None = None,
        gripper_range_rad: Sequence[float] | None = None,
    ) -> None:
        if len(joint_signs) != 7 or len(joint_offsets) != 7:
            raise ValueError(
                "GELLO joint_signs and joint_offsets must have length seven."
            )
        self._port_path = str(Path(port).resolve(strict=True))
        self._joint_signs = np.asarray(joint_signs, dtype=np.float64)
        self._joint_offsets = np.asarray(joint_offsets, dtype=np.float64)
        if not np.all(np.isin(self._joint_signs, (-1, 1))):
            raise ValueError("Every GELLO joint sign must be either -1 or 1.")
        if (gripper_id is None) != (gripper_range_rad is None):
            raise ValueError(
                "gripper_id and gripper_range_rad must be configured together."
            )
        if gripper_range_rad is not None:
            if len(gripper_range_rad) != 2:
                raise ValueError("gripper_range_rad must contain [closed, open].")
            self._gripper_range = np.asarray(gripper_range_rad, dtype=np.float64)
            if self._gripper_range[1] <= self._gripper_range[0]:
                raise ValueError("GELLO gripper open limit must exceed closed limit.")
        else:
            self._gripper_range = None

        try:
            from dynamixel_sdk import GroupSyncRead, PacketHandler, PortHandler
            from dynamixel_sdk.robotis_def import COMM_SUCCESS
        except ImportError as exc:
            raise RuntimeError(
                "GELLO collection requires the 'dynamixel-sdk' Python package."
            ) from exc

        self._comm_success = COMM_SUCCESS
        self._port = PortHandler(self._port_path)
        self._packet = PacketHandler(self._PROTOCOL_VERSION)
        self._reader = GroupSyncRead(
            self._port,
            self._packet,
            self._PRESENT_POSITION_ADDRESS,
            self._PRESENT_POSITION_LENGTH,
        )
        self._arm_ids = tuple(range(1, 8))
        self._gripper_id = gripper_id
        self._ids = self._arm_ids + ((gripper_id,) if gripper_id is not None else ())
        self._closed = False

        if not self._port.openPort():
            raise ConnectionError(
                f"Could not open GELLO serial port {self._port_path}."
            )
        if not self._port.setBaudRate(self._BAUDRATE):
            self._port.closePort()
            raise ConnectionError(
                f"Could not set GELLO baud rate on {self._port_path}."
            )
        try:
            for dynamixel_id in self._ids:
                model, result, error = self._packet.ping(self._port, dynamixel_id)
                if result != self._comm_success or error != 0 or model == 0:
                    raise ConnectionError(
                        f"GELLO motor {dynamixel_id} did not respond on "
                        f"{self._port_path}."
                    )
                if not self._reader.addParam(dynamixel_id):
                    raise RuntimeError(
                        f"Could not register GELLO motor {dynamixel_id} for sync read."
                    )
            self._set_torque(False)
        except Exception:
            self._port.closePort()
            raise

    def _set_torque(self, enabled: bool) -> None:
        value = int(enabled)
        for dynamixel_id in self._ids:
            result, error = self._packet.write1ByteTxRx(
                self._port,
                dynamixel_id,
                self._TORQUE_ENABLE_ADDRESS,
                value,
            )
            if result != self._comm_success or error != 0:
                raise RuntimeError(
                    f"Could not set torque={value} for GELLO motor {dynamixel_id}."
                )

    def _read_raw(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("GELLO reader is closed.")
        result = self._reader.txRxPacket()
        if result != self._comm_success:
            detail = self._packet.getTxRxResult(result)
            raise RuntimeError(f"GELLO sync read failed: {detail}.")

        pulses = []
        for dynamixel_id in self._ids:
            if not self._reader.isAvailable(
                dynamixel_id,
                self._PRESENT_POSITION_ADDRESS,
                self._PRESENT_POSITION_LENGTH,
            ):
                raise RuntimeError(
                    f"GELLO motor {dynamixel_id} did not return a position."
                )
            value = self._reader.getData(
                dynamixel_id,
                self._PRESENT_POSITION_ADDRESS,
                self._PRESENT_POSITION_LENGTH,
            )
            pulses.append(int(np.int32(np.uint32(value))))

        raw = np.asarray(pulses, dtype=np.float64)
        return raw * (2.0 * np.pi / self._PULSES_PER_REVOLUTION)

    def _calibrate_arm(self, raw: np.ndarray) -> np.ndarray:
        joints = (raw[:7] - self._joint_offsets) * self._joint_signs
        if not np.isfinite(joints).all():
            raise RuntimeError("GELLO returned a non-finite joint position.")
        return joints.astype(np.float32)

    def read(self) -> np.ndarray:
        """Return seven calibrated GELLO arm positions in radians."""
        return self._calibrate_arm(self._read_raw())

    def read_with_gripper(self) -> tuple[np.ndarray, float]:
        """Return arm positions and main-wrapper gripper value (0=open, 1=closed)."""
        if self._gripper_id is None or self._gripper_range is None:
            raise RuntimeError("GELLO gripper is not configured.")
        raw = self._read_raw()
        openness = (raw[-1] - self._gripper_range[0]) / (
            self._gripper_range[1] - self._gripper_range[0]
        )
        closedness = 1.0 - np.clip(openness, 0.0, 1.0)
        return self._calibrate_arm(raw), float(closedness)

    def close(self) -> None:
        """Disable GELLO torque and release the serial port."""
        if self._closed:
            return
        self._closed = True
        try:
            self._set_torque(False)
        finally:
            self._port.closePort()
