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

"""Aoyi dexterous five-finger hand end-effector.

Communication uses Modbus RTU over a serial port (``pymodbus``).
The hand has **6 DOFs**: thumb, index, middle, ring, pinky, and thumb
rotation.  Each DOF is normalised to ``[0, 1]`` where ``0`` means fully
open and ``1`` means fully closed.
"""

import time
from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

from .base import EndEffector


class AoyiHand(EndEffector):
    """Aoyi modular dexterous hand controlled via Modbus RTU.

    Args:
        port: Serial device path, e.g. ``"/dev/ttyUSB0"``.
        node_id: Modbus slave node ID.
        baudrate: Serial baudrate.
        default_state: Default hand state used during ``reset()``.
    """

    _NUM_DOFS = 6

    # Register address mapping (from Aoyi documentation)
    _TARGET_REG_START = 1155   # Write: target positions for 6 fingers
    _CURRENT_REG_START = 1165  # Read:  current positions for 6 fingers
    _SELFTEST_REG = 1008
    _INIT_REG = 1013

    # Calibrated position limits per finger (raw register values)
    _LOWER_LIMITS = np.array([226, 10023, 9782, 10138, 9885, 0])
    _UPPER_LIMITS = np.array([3676, 17832, 17601, 17652, 17484, 8997])

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        node_id: int = 2,
        baudrate: int = 115200,
        default_state: Optional[list[float]] = None,
    ):
        self._port = port
        self._node_id = node_id
        self._baudrate = baudrate
        self._default_state = np.array(
            default_state if default_state is not None else [0.0] * self._NUM_DOFS,
            dtype=np.float64,
        )
        self._logger = get_logger()
        self._client = None
        self._last_valid_state: np.ndarray = np.full(self._NUM_DOFS, 0.5)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._NUM_DOFS

    @property
    def state_dim(self) -> int:
        return self._NUM_DOFS

    @property
    def control_mode(self) -> str:
        return "continuous"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Open the Modbus RTU connection and initialise the hand."""
        try:
            from pymodbus.client.sync import ModbusSerialClient
        except ImportError:
            raise ImportError(
                "pymodbus is required for the Aoyi hand. "
                "Install it with: pip install pymodbus==2.5.3"
            )

        self._client = ModbusSerialClient(
            method="rtu",
            port=self._port,
            baudrate=self._baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
        )
        if not self._client.connect():
            raise ConnectionError(
                f"Failed to connect to Aoyi hand on {self._port}"
            )
        time.sleep(1.0)

        # Initialise the hand (self-test + zero calibration)
        self._write_register(self._SELFTEST_REG, 1)
        self._write_register(self._INIT_REG, 1)
        self._logger.info(
            f"AoyiHand initialised on {self._port} (node_id={self._node_id})."
        )

    def shutdown(self) -> None:
        """Close the Modbus serial connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._logger.info("AoyiHand connection closed.")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Read current finger positions and return normalised ``[0, 1]`` array."""
        raw = self._batch_read_registers(self._CURRENT_REG_START, self._NUM_DOFS)
        if raw is None:
            self._logger.warning(
                "AoyiHand: failed to read state, returning last valid state."
            )
            return self._last_valid_state.copy()

        positions = np.array(raw, dtype=np.float64)
        normalised = (positions - self._LOWER_LIMITS) / (
            self._UPPER_LIMITS - self._LOWER_LIMITS
        )
        normalised = np.clip(normalised, 0.0, 1.0)
        self._last_valid_state = normalised
        return normalised

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def command(self, action: np.ndarray) -> bool:
        """Write target finger positions.

        Args:
            action: Normalised target positions in ``[0, 1]`` of shape
                ``(6,)``.

        Returns:
            Always ``True`` (continuous control, every command is effective).
        """
        action = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        targets = (
            self._LOWER_LIMITS + (self._UPPER_LIMITS - self._LOWER_LIMITS) * action
        ).astype(int)
        self._batch_write_registers(self._TARGET_REG_START, targets.tolist())
        return True

    def reset(self, target_state: np.ndarray | None = None) -> None:
        """Reset hand to the default or specified state."""
        state = (
            np.asarray(target_state, dtype=np.float64)
            if target_state is not None
            else self._default_state
        )
        self.command(state)
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Modbus helpers
    # ------------------------------------------------------------------

    def _read_register(self, address: int, count: int = 1):
        """Read ``count`` holding registers starting at ``address``."""
        try:
            result = self._client.read_holding_registers(
                address, count, unit=self._node_id
            )
            if result.isError():
                self._logger.warning(f"Modbus read error at register {address}")
                return None
            return result.registers
        except Exception as e:
            self._logger.warning(f"Modbus communication error: {e}")
            return None

    def _write_register(self, address: int, value: int) -> bool:
        """Write a single holding register."""
        try:
            result = self._client.write_register(
                address, value, unit=self._node_id
            )
            if result.isError():
                self._logger.warning(f"Modbus write error at register {address}")
                return False
            return True
        except Exception as e:
            self._logger.warning(f"Modbus communication error: {e}")
            return False

    def _batch_read_registers(
        self, start: int, count: int, max_retries: int = 3
    ):
        """Batch-read registers with retry and fallback to single reads."""
        for attempt in range(max_retries):
            try:
                result = self._client.read_holding_registers(
                    start, count, unit=self._node_id
                )
                if not result.isError():
                    return result.registers
            except Exception:
                pass
            time.sleep(0.01 * (attempt + 1))

        # Fallback: individual reads
        registers = []
        for addr in range(start, start + count):
            val = self._read_register(addr)
            if val is None:
                return None
            registers.extend(val)
        return registers

    def _batch_write_registers(self, start: int, values: list[int]) -> bool:
        """Batch-write a contiguous block of holding registers."""
        try:
            result = self._client.write_registers(
                start, values, unit=self._node_id
            )
            if result.isError():
                self._logger.warning(
                    f"Modbus batch write error at register {start}"
                )
                return False
            return True
        except Exception as e:
            self._logger.warning(f"Modbus communication error: {e}")
            return False
