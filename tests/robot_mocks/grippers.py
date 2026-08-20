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
    """A ``pymodbus`` whose registers remember what was written."""

    class Registers:
        def __init__(self, values):
            self.registers = values

        def isError(self):
            return False

    class ModbusSerialClient:
        """Answers the Robotiq activation handshake and echoes commands.

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


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Every gripper SDK, by the name a part imports it as."""
    return pymodbus()
