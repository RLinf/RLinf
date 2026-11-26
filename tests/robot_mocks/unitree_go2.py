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

"""In-memory SDK2 edge fake; no sockets, DDS participants, or robot access."""

import threading
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any


def lowstate(tick: int = 1) -> SimpleNamespace:
    """Return deterministic Go2 lowstate data with twelve joints."""
    return SimpleNamespace(
        tick=tick,
        crc=42,
        imu_state=SimpleNamespace(
            quaternion=[1.0, 0.0, 0.0, 0.0],
            rpy=[0.1, 0.2, 0.3],
            gyroscope=[1, 2, 3],
        ),
        motor_state=[SimpleNamespace(q=i / 10, dq=i / 20) for i in range(12)],
        power_v=31.52,
        foot_force=[40, 0, 0, 0],
    )


class FakeGo2SDK:
    """Publish state on subscription and record every motion/stop call."""

    def __init__(self) -> None:
        self.publish_initial = True
        self.stop_code = 0
        self.move_code = 0
        self.move_error = False
        self.events: list[tuple[Any, ...]] = []
        self.writes: list[tuple[object, str, float]] = []
        self.stopped = threading.Event()
        self.close_entered = threading.Event()
        self.close_release = threading.Event()
        self.close_release.set()
        self.callback: Any = None
        self.sport_callback: Any = None
        self.publish_sport = True

    def publish(self) -> None:
        """Deliver one recent state through the real adapter's SDK callback."""
        self.callback(lowstate())

    @contextmanager
    def streaming(self):
        """Deliver both SDK state topics while a manual check runs."""
        shutdown = threading.Event()

        def publish() -> None:
            while not shutdown.wait(0.01):
                if self.callback is not None:
                    self.publish()
                if self.publish_sport and self.sport_callback is not None:
                    self.sport_callback(
                        SimpleNamespace(velocity=[0.0, 0.0, 0.0], yaw_speed=0.0)
                    )

        thread = threading.Thread(target=publish, daemon=True)
        thread.start()
        try:
            yield self
        finally:
            shutdown.set()
            thread.join()

    def modules(self) -> dict[str, ModuleType]:
        """Return importable SDK modules for an isolated test process."""
        fake = self

        class Subscriber:
            def __init__(self, topic: str, dtype: type) -> None:
                self.topic = topic
                fake.events.append(("subscribe", topic))

            def Init(self, callback: Any, queue_len: int) -> None:
                if self.topic == "rt/sportmodestate":
                    fake.sport_callback = callback
                    return
                fake.callback = callback
                if fake.publish_initial:
                    fake.publish()

            def Close(self) -> None:
                if self.topic == "rt/sportmodestate":
                    fake.sport_callback = None
                    fake.events.append(("sport_close",))
                    return
                fake.close_entered.set()
                if not fake.close_release.wait(2):
                    raise TimeoutError("Test did not release subscriber close")
                fake.events.append(("close",))

        class Client:
            def __init__(self) -> None:
                self.writer = object()
                fake.events.append(("client",))

            def SetTimeout(self, timeout: float) -> None:
                self.timeout = timeout

            def Init(self) -> None:
                fake.events.append(("init",))

            def StopMove(self) -> int:
                fake.writes.append((self.writer, "stop", self.timeout))
                fake.events.append(("stop",))
                fake.stopped.set()
                return fake.stop_code

            def Move(self, vx: float, vy: float, vyaw: float) -> int:
                fake.writes.append((self.writer, "move", self.timeout))
                fake.events.append(("move", vx, vy, vyaw))
                if fake.move_error:
                    raise OSError("Simulated DDS send failure")
                return fake.move_code

        class ChannelFactory:
            __initialized = False
            __qos = None

            def Init(self, domain, interface, qos=None):
                cls = type(self)
                if not cls.__initialized:
                    fake.events.append(("dds", domain, interface))
                    cls.__qos = qos
                    cls.__initialized = True
                return True

        def initialize(domain: int, interface: str) -> None:
            ChannelFactory().Init(domain, interface)

        class CRC:
            def Crc(self, message):
                return 42

        definitions = {
            "unitree_sdk2py.core.channel": {
                "ChannelFactory": ChannelFactory,
                "ChannelFactoryInitialize": initialize,
                "Qos": object,
                "ChannelSubscriber": Subscriber,
            },
            "unitree_sdk2py.utils.crc": {"CRC": CRC},
            "unitree_sdk2py.go2.sport.sport_client": {"SportClient": Client},
            "unitree_sdk2py.idl.unitree_go.msg.dds_": {
                "LowState_": SimpleNamespace,
                "SportModeState_": SimpleNamespace,
            },
        }
        modules = {}
        for name, values in definitions.items():
            module = ModuleType(name)
            module.__dict__.update(values)
            modules[name] = module
        return modules
