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

"""Ruiyan dexterous five-finger hand driver.

Communication uses a custom serial protocol (``pyserial``).  The hand
has **6 DOFs**: thumb rotation, thumb bend, index, middle, ring, and
pinky.  Each DOF is normalised to ``[0, 1]``.

The low-level protocol implementation is self-contained so that no
external driver package is required at runtime.
"""

import logging
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight serial protocol (self-contained, no external dependency)
# ---------------------------------------------------------------------------


class _InstructionType(IntEnum):
    READ_MOTOR_INFO = 0xA0
    CTRL_POSITION_VEL_CUR = 0xAA
    CLEAR_ERROR = 0xA5


@dataclass
class _FingerStatus:
    motor_id: int
    status: int
    position: int
    velocity: int
    current: int


class _SerialLink:
    """Minimal serial wrapper for the Ruiyan protocol."""

    def __init__(self, port: str, baudrate: int, timeout: float = 0.015):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._serial = None

    def connect(self) -> None:
        import serial

        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            timeout=self._timeout,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
        )
        logger.info("Ruiyan serial opened: %s", self._port)

    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def send_command(
        self,
        motor_id: int,
        instruction: int,
        position: int,
        velocity: int,
        current: int,
    ) -> None:
        position = max(0, min(65535, position))
        velocity = max(0, min(65535, velocity))
        current = max(0, min(65535, current))

        frame = struct.pack(
            "<B B B 2B 3H 1B",
            0xA5, motor_id, 0x00, 0x08, instruction,
            position, velocity, current, 0x00,
        )
        checksum = sum(frame) & 0xFF
        frame = struct.pack(
            "<B B B 2B 3H 1B 1B",
            0xA5, motor_id, 0x00, 0x08, instruction,
            position, velocity, current, 0x00, checksum,
        )
        self._serial.write(frame)

    def read_responses(self, num_motors: int) -> list[Optional[_FingerStatus]]:
        """Read ``num_motors`` response frames (13 bytes each)."""
        raw = self._serial.read(13 * num_motors)
        if raw is None or len(raw) != 13 * num_motors:
            return [None] * num_motors

        results: list[Optional[_FingerStatus]] = []
        for i in range(num_motors):
            chunk = raw[i * 13 : (i + 1) * 13]
            parsed = self._parse_frame(chunk)
            results.append(parsed)
        return results

    @staticmethod
    def _parse_frame(raw: bytes) -> Optional[_FingerStatus]:
        if len(raw) < 13:
            return None
        header, motor_id = struct.unpack("<BB", raw[:2])
        if header != 0xA5:
            return None
        data_u64 = struct.unpack("<Q", raw[4:12])[0]
        status = (data_u64 >> 8) & 0xFF
        position = (data_u64 >> 16) & 0xFFF
        velocity = (data_u64 >> 28) & 0xFFF
        current = (data_u64 >> 40) & 0xFFF
        if velocity & 0x800:
            velocity -= 0x1000
        if current & 0x800:
            current -= 0x1000
        return _FingerStatus(
            motor_id=motor_id,
            status=status,
            position=position,
            velocity=velocity,
            current=current,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RuiyanHandDriver:
    """Ruiyan dexterous hand controlled via custom serial protocol.

    Args:
        port: Serial device path, e.g. ``"/dev/ttyUSB0"``.
        baudrate: Serial baudrate (default 460800).
        motor_ids: Tuple of motor IDs corresponding to the 6 fingers.
        default_velocity: Default command velocity for all motors.
        default_current: Default command current for all motors.
        default_state: Default hand state used during ``reset()``.
    """

    NUM_DOFS = 6
    _POS_RAW_SCALE = 4096  # raw → normalised: pos / 4095
    FINGER_NAMES = [
        "thumb_rotation",  # thumb rotation
        "thumb_bend",      # thumb bend
        "index",           # index finger
        "middle",          # middle finger
        "ring",            # ring finger
        "pinky",           # pinky finger
    ]

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 460800,
        motor_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        default_velocity: int = 2000,
        default_current: int = 800,
        default_state: Optional[list[float]] = None,
    ):
        self._port = port
        self._baudrate = baudrate
        self._motor_ids = list(motor_ids)
        self._default_velocity = default_velocity
        self._default_current = default_current
        self._default_state = np.array(
            default_state if default_state is not None else [0.0] * self.NUM_DOFS,
            dtype=np.float64,
        )

        # Serial link (initialised in ``initialize``)
        self._link: Optional[_SerialLink] = None

        # Background control loop
        self._lock = threading.Lock()
        self._target_positions = np.zeros(self.NUM_DOFS, dtype=np.float64)
        self._current_positions = np.zeros(self.NUM_DOFS, dtype=np.float64)
        self._current_velocities = np.zeros(self.NUM_DOFS, dtype=np.float64)
        self._current_currents = np.zeros(self.NUM_DOFS, dtype=np.float64)
        self._current_statuses = np.zeros(self.NUM_DOFS, dtype=np.int32)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Open the serial port and start the background control loop."""
        self._link = _SerialLink(
            port=self._port, baudrate=self._baudrate
        )
        self._link.connect()
        self._start_loop()
        logger.info(
            f"RuiyanHandDriver initialised on {self._port} "
            f"(baudrate={self._baudrate})."
        )

    def shutdown(self) -> None:
        """Stop the background loop and close the serial port."""
        self._stop_loop()
        if self._link is not None:
            self._link.disconnect()
            self._link = None
        logger.info("RuiyanHandDriver connection closed.")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return the latest finger positions (normalised ``[0, 1]``)."""
        with self._lock:
            return self._current_positions.copy()

    def get_detailed_state(self) -> dict:
        """Return detailed per-motor diagnostic information."""
        with self._lock:
            return {
                "finger_names": list(self.FINGER_NAMES),
                "positions": self._current_positions.copy().tolist(),
                "velocities": self._current_velocities.copy().tolist(),
                "currents": self._current_currents.copy().tolist(),
                "statuses": self._current_statuses.copy().tolist(),
                "motor_ids": list(self._motor_ids),
                "port": self._port,
                "baudrate": self._baudrate,
                "target_positions": self._target_positions.copy().tolist(),
            }

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def command(self, action: np.ndarray) -> bool:
        """Set target finger positions (normalised ``[0, 1]``).

        This is non-blocking: it only updates the target buffer.  The
        background loop continuously sends the latest targets to the
        hardware at high frequency.

        Args:
            action: 6-D normalised target positions.

        Returns:
            Always ``True``.
        """
        action = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        with self._lock:
            self._target_positions = action.copy()
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
    # Background loop
    # ------------------------------------------------------------------

    def _start_loop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop_task, daemon=True)
        self._thread.start()

    def _stop_loop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop_task(self) -> None:
        """Continuously poll motor states at ~200 Hz."""
        while not self._stop_event.is_set():
            start = time.time()
            try:
                self._poll_state()
            except Exception as e:
                logger.warning(f"RuiyanHandDriver poll error: {e}")
            elapsed = time.time() - start
            time.sleep(max(0.0, 1.0 / 200.0 - elapsed))

    def _poll_state(self) -> None:
        """Send current targets and read back motor states."""
        with self._lock:
            targets = self._target_positions.copy()

        self._send_targets(targets)
        responses = self._link.read_responses(len(self._motor_ids))

        positions = []
        velocities = []
        currents = []
        statuses = []
        for resp in responses:
            if resp is not None:
                positions.append(resp.position / 4095.0)
                velocities.append(float(resp.velocity))
                currents.append(float(resp.current))
                statuses.append(int(resp.status))
            else:
                positions.append(None)
                velocities.append(None)
                currents.append(None)
                statuses.append(None)

        # Only update if we got valid readings for all motors
        if all(p is not None for p in positions):
            with self._lock:
                self._current_positions = np.array(
                    positions, dtype=np.float64
                )
                self._current_velocities = np.array(
                    velocities, dtype=np.float64
                )
                self._current_currents = np.array(
                    currents, dtype=np.float64
                )
                self._current_statuses = np.array(
                    statuses, dtype=np.int32
                )

    def _send_targets(self, targets: np.ndarray) -> None:
        """Write target positions to all motors."""
        raw_positions = (targets * self._POS_RAW_SCALE).astype(int)
        for idx, motor_id in enumerate(self._motor_ids):
            self._link.send_command(
                motor_id=motor_id,
                instruction=int(_InstructionType.CTRL_POSITION_VEL_CUR),
                position=int(raw_positions[idx]),
                velocity=self._default_velocity,
                current=self._default_current,
            )
