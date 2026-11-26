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

"""Read a joint and IMU sample from the Go2 low-state topic."""

import argparse
import threading
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from typing import Any

from rlinf.robotics.go2_dds import go2_dds_session
from rlinf.utils.logging import get_logger


@dataclass(frozen=True)
class JointState:
    """SDK-ordered joint values and body IMU at local monotonic reception time."""

    q: tuple[float, ...]
    dq: tuple[float, ...]
    quaternion_wxyz: tuple[float, ...]
    gyroscope_b: tuple[float, ...]
    timestamp: float


class Go2SDK2:
    """Read checksum-valid joint state from the requested network interface."""

    def __init__(self, interface: str) -> None:
        if not interface.strip():
            raise ValueError("Expected an explicit Go2 network interface")
        self.interface = interface
        self._resources = ExitStack()
        self._session_lock = threading.RLock()
        self._state_lock = threading.Condition()
        self._connected = False
        self._state: JointState | None = None
        self._tick: int | None = None

    def connect(self) -> None:
        """Subscribe to the SDK2 low-state topic."""
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC

        with self._session_lock:
            self._resources.enter_context(go2_dds_session(self.interface))
            try:
                self._crc = CRC()
                subscriber = ChannelSubscriber("rt/lowstate", LowState_)
                self._resources.callback(subscriber.Close)
                subscriber.Init(self._receive, 1)
                self._connected = True
            except BaseException:
                self.close()
                raise

    def _receive(self, message: Any) -> None:
        if message.crc != self._crc.Crc(message):
            return
        with self._state_lock:
            if message.tick == self._tick:
                return
            self._tick = message.tick
            self._state = JointState(
                tuple(motor.q for motor in message.motor_state[:12]),
                tuple(motor.dq for motor in message.motor_state[:12]),
                tuple(message.imu_state.quaternion),
                tuple(message.imu_state.gyroscope),
                time.monotonic(),
            )
            self._state_lock.notify_all()

    def wait_for_state(self) -> None:
        """Wait up to one second for a fresh, checksum-valid sample."""
        with self._state_lock:
            ready = self._state_lock.wait_for(
                lambda: (
                    not self._connected
                    or (
                        self._state is not None
                        and 0 <= time.monotonic() - self._state.timestamp < 0.05
                    )
                ),
                timeout=1.0,
            )
            if not self._connected:
                raise RuntimeError("Go2 connection is closed")
            if not ready:
                raise TimeoutError(
                    "No fresh checksum-valid Go2 LowState within one second"
                )

    def read(self) -> JointState:
        """Return the most recent checksum-valid state without blocking."""
        with self._state_lock:
            if self._state is None:
                raise RuntimeError("No checksum-valid Go2 LowState has arrived")
            return self._state

    def close(self) -> None:
        """Close the subscriber and discard the last sample."""
        with self._session_lock:
            try:
                self._resources.close()
            finally:
                with self._state_lock:
                    self._connected = False
                    self._state = None
                    self._tick = None
                    self._state_lock.notify_all()


def main() -> None:
    """Report the first fresh, checksum-valid low-state sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    args = parser.parse_args()
    transport = Go2SDK2(args.interface)
    transport.connect()
    try:
        transport.wait_for_state()
        get_logger().info("Go2 joint state %s", asdict(transport.read()))
    finally:
        transport.close()


if __name__ == "__main__":
    main()
