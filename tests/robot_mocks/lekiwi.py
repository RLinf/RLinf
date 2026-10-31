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

"""Keyboard and motor-bus fakes for the LeKiwi bring-up commands."""

from __future__ import annotations


class FakeKeyboard:
    """Provide held-key snapshots, listener loss, or input failures."""

    def __init__(
        self, snapshots: list[set[str] | BaseException | None], available: bool = True
    ) -> None:
        self.snapshots = iter(snapshots)
        self.available = available
        self.is_connected = False

    def connect(self) -> None:
        self.is_connected = self.available

    def get_action(self) -> dict[str, None]:
        snapshot = next(self.snapshots, {"q"})
        if isinstance(snapshot, BaseException):
            raise snapshot
        if snapshot is None:
            self.is_connected = False
            return {"w": None}
        return dict.fromkeys(snapshot)

    def disconnect(self) -> None:
        self.is_connected = False


class FakeKeySource:
    """Serve scripted ``(connected, held_keys)`` snapshots to a control loop."""

    host = "127.0.0.1"
    port = 5555

    def __init__(self, snapshots: list[tuple[bool, set[str]]]) -> None:
        self._snapshots = iter(snapshots)

    def snapshot(self) -> tuple[bool, set[str]]:
        return next(self._snapshots)


class FakeWheelBus:
    """Track wheel state and simulate faults at the USB boundary."""

    def __init__(
        self,
        fail_motion: bool = False,
        fail_connect: bool = False,
        lost_wheel: str | None = None,
    ) -> None:
        self.is_connected = False
        self.connect_attempted = False
        self.fail_motion = fail_motion
        self.fail_connect = fail_connect
        self.lost_wheel = lost_wheel
        self.failed = False
        self.velocity = dict.fromkeys(("left", "back", "right"), 0)
        self.torque = dict.fromkeys(self.velocity, False)
        self.mode = dict.fromkeys(self.velocity, 0)
        self.commands: list[dict[str, int]] = []
        self.stop_attempts: list[str] = []
        self.disable_attempts: list[str] = []

    def connect(self) -> None:
        self.connect_attempted = True
        self.is_connected = True
        if self.fail_connect:
            raise OSError("Simulated handshake failure after opening USB")

    def write(self, register: str, name: str, value: int, **kwargs: object) -> None:
        if name not in self.velocity:
            raise AssertionError(f"Unexpected motor {name}; expected a wheel")
        if register == "Goal_Velocity" and value == 0:
            self.stop_attempts.append(name)
        if self.failed and name == self.lost_wheel:
            raise OSError(f"Simulated loss of wheel {name}")
        if register == "Goal_Velocity":
            self.velocity[name] = value
        elif register == "Operating_Mode":
            self.mode[name] = value

    def disable_torque(self, name: str | None = None, **kwargs: object) -> None:
        for motor in self.velocity if name is None else (name,):
            self.disable_attempts.append(motor)
            if self.failed and motor == self.lost_wheel:
                raise OSError(f"Simulated loss of wheel {motor}")
            self.torque[motor] = False

    def enable_torque(self) -> None:
        if any(self.velocity.values()) or any(mode != 1 for mode in self.mode.values()):
            raise AssertionError(
                "Expected zero speed and velocity mode before enabling torque"
            )
        self.torque = dict.fromkeys(self.torque, True)

    def sync_write(
        self, register: str, values: dict[str, int], **kwargs: object
    ) -> None:
        if register != "Goal_Velocity" or set(values) != set(self.velocity):
            raise AssertionError("Expected velocity commands for exactly three wheels")
        self.velocity.update(
            {
                name: value
                for name, value in values.items()
                if not (self.failed and name == self.lost_wheel)
            }
        )
        self.commands.append(values.copy())
        if self.fail_motion and any(values.values()):
            self.failed = True
            raise OSError("Simulated USB error during movement")

    def sync_read(
        self, register: str, names: list[str], **kwargs: object
    ) -> dict[str, int]:
        if register != "Present_Velocity":
            raise AssertionError(f"Unexpected register {register}")
        return {name: self.velocity[name] for name in names}

    def disconnect(self, disable_torque: bool = True) -> None:
        self.is_connected = False
