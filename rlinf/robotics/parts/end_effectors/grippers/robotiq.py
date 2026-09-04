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

"""Robotiq 2F-85 and 2F-140 driver for Modbus RTU over USB-RS485.

The driver uses ``pymodbus`` directly and does not require ROS.

Modbus register map (Robotiq 2F series)
---------------------------------------
**Output registers** (FC 16, base address 0x03E8):

====== ========= ===========================================
Byte   Register  Description
====== ========= ===========================================
0      reg0 hi   Action request: rACT(0) rGTO(3) rATR(4)
1      reg0 lo   Reserved
2      reg1 hi   Reserved
3      reg1 lo   rPR: position request  (0=open, 255=closed)
4      reg2 hi   rSP: speed             (0=min,  255=max)
5      reg2 lo   rFR: force             (0=min,  255=max)
====== ========= ===========================================

**Input registers** (FC 03, base address 0x07D0):

====== ========= ===========================================
Byte   Register  Description
====== ========= ===========================================
0      reg0 hi   Status: gACT(0) gGTO(3) gSTA(4-5) gOBJ(6-7)
1      reg0 lo   Reserved
2      reg1 hi   gFLT: fault status
3      reg1 lo   gPR: position request echo
4      reg2 hi   gPO: actual position  (0=open, 255=closed)
5      reg2 lo   gCU: motor current    (x10 mA)
====== ========= ===========================================
"""

import inspect
import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from rlinf.robotics.parts.claims import DeviceClaim
from rlinf.utils.logging import get_logger

from ..base import EndEffector
from .base import BaseGripper

# Modbus constants
_SLAVE_ID = 0x09
_OUTPUT_REG_ADDR = 0x03E8  # base address for output (write) registers
_INPUT_REG_ADDR = 0x07D0  # base address for input (read) registers
_NUM_REGS = 3

# Action-request bit positions (within the high byte of register 0)
_rACT = 1 << 0
_rGTO = 1 << 3


def _create_modbus_client(port: str, baudrate: int = 115200) -> Any:
    """Create a serial client compatible with pymodbus 2 and 3."""
    try:
        from pymodbus.client import ModbusSerialClient
    except ImportError:
        from pymodbus.client.sync import ModbusSerialClient

    return ModbusSerialClient(
        port=port,
        baudrate=baudrate,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=1,
    )


if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlinf.robotics.parts.transports.ros import ROSController


@EndEffector.register("robotiq_gripper")
class RobotiqGripper(BaseGripper):
    """Control a Robotiq 2F-85 or 2F-140 over Modbus RTU.

    Args:
        port: Serial device path, e.g. ``"/dev/ttyUSB0"``.
        baudrate: Modbus baud rate.
        slave_id: Modbus slave address.
        max_width: Physical opening of the fully-open gripper in metres.
            0.085 for the 2F-85, 0.140 for the 2F-140.
    """

    @classmethod
    def declare(
        cls,
        *,
        ros: Optional["ROSController"] = None,
        port: Optional[str] = None,
        robot_ip: Optional[str] = None,
        **settings: Any,
    ) -> "RobotiqGripper":
        """Declare a gripper that uses a dedicated serial connection."""
        return cls(port=port, **settings)

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        slave_id: int = _SLAVE_ID,
        max_width: float = 0.085,
    ) -> None:
        if not port:
            raise ValueError(
                "A Robotiq gripper requires a serial port. Set "
                "'gripper_connection' in the robot config, for example "
                "'/dev/ttyUSB0'."
            )
        self._logger = get_logger()
        self._port = port
        # A serial port carries one Modbus conversation at a time.
        self._claim = DeviceClaim(f"robotiq:{port}", type(self).__name__)
        self._baudrate = baudrate
        self._slave_id = slave_id
        self._max_width = max_width

        self._client = None
        self._slave_kwarg = "slave"
        self._cached_position: int = 0  # raw 0-255
        self._is_open_flag: bool = True
        self._activated: bool = False

    def _open(self) -> Any:
        """Open the serial link and run the activation sequence."""
        self._claim.acquire()
        client = _create_modbus_client(self._port, self._baudrate)
        client.connect()
        self._client = client

        # pymodbus >=3.10 renamed "slave" to "device_id"
        sig = inspect.signature(client.write_registers)
        self._slave_kwarg = "device_id" if "device_id" in sig.parameters else "slave"

        try:
            self._activate()
        except Exception:
            # Release the serial port if activation fails after opening it.
            client.close()
            self._client = None
            self._claim.release()
            raise

        self._logger.info(
            f"Robotiq gripper activated on {self._port} "
            f"(slave=0x{self._slave_id:02X}, max_width={self._max_width}m)"
        )
        return client

    # BaseGripper interface

    def open(self, speed: float = 0.3) -> None:
        self._goto(position=0, speed=speed, force=0.0)
        self._is_open_flag = True

    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        norm_force = min(force / 255.0, 1.0) if force > 1.0 else force
        self._goto(position=255, speed=speed, force=norm_force)
        self._is_open_flag = False

    def move(self, width: float, speed: float = 0.3) -> None:
        """Move to an opening width in metres.

        The driver converts metres to the reversed register scale, where zero
        is fully open and 255 is fully closed.
        """
        opening = np.clip(width, 0.0, self._max_width) / self._max_width
        self._goto(position=int(round(255 * (1.0 - opening))), speed=speed, force=0.5)

    @property
    def position(self) -> float:
        """Current opening width in metres."""
        status = self._read_status()
        if status is not None:
            self._cached_position = status["position"]
        return self._max_width * (1.0 - self._cached_position / 255.0)

    @property
    def max_width(self) -> float:
        """Opening of the fully-open gripper, as configured for this model."""
        return self._max_width

    @property
    def is_open(self) -> bool:
        return self._is_open_flag

    def is_ready(self) -> bool:
        return self._activated

    def _release(self, device: Any) -> None:
        """Close the serial link the lifecycle handed back."""
        self._activated = False
        self._client = None
        try:
            if device is not None:
                device.close()
        finally:
            self._claim.release()

    # Modbus helpers

    def _activate(self) -> None:
        """Run the clear, activate, and readiness sequence."""
        # 1. Clear (deactivate)
        self._write_output_regs(0x0000, 0x0000, 0x0000)
        time.sleep(0.5)

        # 2. Activate (rACT = 1)
        self._write_output_regs(_rACT << 8, 0x0000, 0x0000)

        # 3. Wait for gSTA == 0x03 (fully activated)
        for _ in range(50):
            time.sleep(0.1)
            status = self._read_status()
            if status is not None and status["gSTA"] == 0x03:
                self._activated = True
                return

        raise RuntimeError(
            f"Robotiq gripper on {self._port} did not activate within 5 s"
        )

    def _goto(self, position: int, speed: float, force: float) -> None:
        """Send a go-to-position command.

        Args:
            position: Raw target position (0 = open, 255 = closed).
            speed: Normalised speed [0, 1].
            force: Normalised force [0, 1].
        """
        pos = int(np.clip(position, 0, 255))
        spd = int(np.clip(speed * 255, 0, 255))
        frc = int(np.clip(force * 255, 0, 255))

        # Reg 0: (action_byte << 8) | reserved
        reg0 = (_rACT | _rGTO) << 8
        # Reg 1: (reserved << 8) | position
        reg1 = pos
        # Reg 2: (speed << 8) | force
        reg2 = (spd << 8) | frc
        self._write_output_regs(reg0, reg1, reg2)

    def _write_output_regs(self, reg0: int, reg1: int, reg2: int) -> None:
        self._client.write_registers(
            address=_OUTPUT_REG_ADDR,
            values=[reg0, reg1, reg2],
            **{self._slave_kwarg: self._slave_id},
        )

    def _read_status(self) -> Optional[dict]:
        """Read the three input registers and decode the status fields."""
        resp = self._client.read_holding_registers(
            address=_INPUT_REG_ADDR,
            count=_NUM_REGS,
            **{self._slave_kwarg: self._slave_id},
        )
        if resp.isError():
            self._logger.warning(f"Robotiq Modbus read error: {resp}")
            return None

        r0, r1, r2 = resp.registers
        status_byte = (r0 >> 8) & 0xFF
        return {
            "gACT": (status_byte >> 0) & 0x01,
            "gGTO": (status_byte >> 3) & 0x01,
            "gSTA": (status_byte >> 4) & 0x03,
            "gOBJ": (status_byte >> 6) & 0x03,
            "fault": (r1 >> 8) & 0xFF,
            "position_echo": r1 & 0xFF,
            "position": (r2 >> 8) & 0xFF,
            "current": r2 & 0xFF,
        }
