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

"""Fake gripper and hand SDKs: Modbus for Robotiq, serial for the RuiYan hand."""

from __future__ import annotations

import types
from typing import Any

from ._fakes import module


def pymodbus() -> dict[str, types.ModuleType]:
    """Return ``pymodbus`` modules backed by in-memory registers."""

    class Registers:
        def __init__(self, values):
            self.registers = values

        def isError(self):
            return False

    class ModbusSerialClient:
        """Emulate the Robotiq activation handshake and command registers.

        The real gripper reports its state in three input registers. Only two
        fields matter to the driver: ``gSTA`` says activation finished, and the
        position byte is what ``position`` reads back. Reporting activation
        immediately keeps a test from waiting out the five-second timeout.
        """

        def __init__(self, *_args, **kwargs):
            self.port = kwargs.get("port")
            self.written: list[tuple[int, list[int]]] = []
            self._position = 0
            self._activated = False

        def connect(self):
            return True

        def close(self):
            return True

        def write_registers(self, address, values, **_kwargs):
            registers = list(values)
            self.written.append((address, registers))
            action = (registers[0] >> 8) & 0xFF
            self._activated = bool(action & 0x01)  # rACT
            if len(registers) > 1:
                self._position = registers[1] & 0xFF
            return Registers(registers)

        def read_holding_registers(self, address, count=3, **_kwargs):
            status = 0x01 if self._activated else 0x00  # gACT
            if self._activated:
                status |= 0x03 << 4  # gSTA: activation complete
            return Registers([status << 8, self._position, self._position << 8])

    client = module("pymodbus.client", ModbusSerialClient=ModbusSerialClient)
    sync = module("pymodbus.client.sync", ModbusSerialClient=ModbusSerialClient)
    client.sync = sync
    root = module("pymodbus")
    root.client = client
    return {
        "pymodbus": root,
        "pymodbus.client": client,
        "pymodbus.client.sync": sync,
    }


def rlinf_dexhand() -> dict[str, types.ModuleType]:
    """Return an ``rlinf_dexhand`` module that records six-finger commands."""

    class RuiyanHandDriver:
        _DOFS = 6

        def __init__(self, port="/dev/ttyUSB0", motor_ids=(1, 2, 3, 4, 5, 6), **kwargs):
            self.port = port
            self.motor_ids = tuple(motor_ids)
            self.default_state = kwargs.get("default_state")
            self.running = False
            self.commands: list[list[float]] = []
            self._positions = [0.0] * self._DOFS

        def initialize(self) -> None:
            self.running = True

        def shutdown(self) -> None:
            self.running = False

        def get_state(self):
            import numpy as np

            return np.asarray(self._positions, dtype=np.float32)

        def command(self, action) -> bool:
            values = [float(v) for v in action]
            self.commands.append(values)
            changed = values != self._positions
            self._positions = values
            return changed

        def reset(self, target_state=None) -> None:
            target = target_state if target_state is not None else self.default_state
            self._positions = (
                [float(v) for v in target] if target is not None else [0.0] * self._DOFS
            )

        def get_detailed_state(self) -> dict:
            return {
                "positions": list(self._positions),
                "finger_names": [f"motor_{i}" for i in self.motor_ids],
            }

    ruiyan = module("rlinf_dexhand.ruiyan", RuiyanHandDriver=RuiyanHandDriver)
    root = module("rlinf_dexhand")
    root.__path__ = []  # a package, or importing a submodule of it fails
    root.ruiyan = ruiyan
    return {"rlinf_dexhand": root, "rlinf_dexhand.ruiyan": ruiyan}


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Return fake gripper and hand SDKs keyed by import name."""
    return {**pymodbus(), **rlinf_dexhand()}
