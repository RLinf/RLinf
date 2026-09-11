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

"""Fake SpaceMouse, leader-arm, glove, VR, and keyboard interfaces.

Devices report an idle state by default. Tests can supply motion directly::

    with mocked_sdks() as made:
        made["pyspacemouse"].twist = (0.5, 0, 0, 0, 0, 0)
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np

from ._fakes import module

DOF = 7


def spacemouse() -> types.ModuleType:
    """Return an idle ``pyspacemouse`` module with mutable input state."""
    fake = module("pyspacemouse")
    fake.twist = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    fake.buttons = (0, 0)

    class Device:
        def __init__(self):
            self.closed = False

        def read(self):
            if self.closed:
                raise OSError("read from a closed SpaceMouse")
            x, y, z, roll, pitch, yaw = fake.twist
            return types.SimpleNamespace(
                x=x,
                y=y,
                z=z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                buttons=list(fake.buttons),
            )

        def close(self):
            self.closed = True

    # A fresh handle per open, as the real driver does, recorded so a test
    # can see what became of one after the part that opened it disconnected.
    fake.devices = []

    def _open(**_kwargs):
        device = Device()
        fake.devices.append(device)
        return device

    fake.open = _open
    return fake


def gello() -> dict[str, types.ModuleType]:
    """Return a ``gello_teleop`` module with a fixed leader-arm pose."""
    agent_module = module("gello_teleop.gello_teleop_agent")
    fk_module = module("gello_teleop.franka_fk")

    class GelloTeleopAgent:
        def __init__(self, port=None, **_kwargs):
            self.port = port
            self.closed = False
            agent_module.last_agent = self

        def get_action(self):
            return np.zeros(DOF), 0.0

        def close(self):
            self.closed = True

    class FrankaFK:
        def get_fk(self, _joints):
            return np.array([0.4, 0.0, 0.3]), np.array([0.0, 1.0, 0.0, 0.0])

    agent_module.GelloTeleopAgent = GelloTeleopAgent
    fk_module.FrankaFK = FrankaFK
    package = module("gello_teleop")
    package.gello_teleop_agent = agent_module
    package.franka_fk = fk_module
    return {
        "gello_teleop": package,
        "gello_teleop.gello_teleop_agent": agent_module,
        "gello_teleop.franka_fk": fk_module,
    }


def glove() -> dict[str, types.ModuleType]:
    """Return an ``rlinf_dexhand`` glove that reports open fingers."""

    class GloveExpert:
        def __init__(self, *_args, **_kwargs):
            self.angles = np.zeros(6)

        def get_angles(self):
            return self.angles

        def close(self):
            return None

        def stop(self):
            return None

    inner = module("rlinf_dexhand.glove", GloveExpert=GloveExpert)
    package = module("rlinf_dexhand")
    package.glove = inner
    return {"rlinf_dexhand": package, "rlinf_dexhand.glove": inner}


def vr() -> types.ModuleType:
    """Return a ``zmq`` module backed by a test-controlled packet queue."""
    fake = module("zmq")
    fake.packets: list[Any] = []

    class Socket:
        def __init__(self, *_a, **_k):
            self.subscribed = []
            self.address = None
            self.hwm = None
            self.closed = False

        def connect(self, address):
            self.address = address

        def set_hwm(self, value):
            self.hwm = value

        def setsockopt_string(self, *_a, **_k):
            return None

        def setsockopt(self, *_a, **_k):
            return None

        def close(self, linger=None):
            self.closed = True

        def recv_string(self, flags=0):
            if not fake.packets:
                raise fake.Again()
            packet = fake.packets.pop(0)
            return packet if isinstance(packet, str) else packet.decode("utf-8")

        def recv(self, flags=0):
            """Bytes, as pyzmq gives them; the publisher sends JSON."""
            packet = self.recv_string(flags)
            return packet.encode("utf-8")

    class Context:
        def __init__(self):
            self.sockets = []
            self.terminated = False

        def socket(self, _kind):
            sock = Socket()
            self.sockets.append(sock)
            fake.sockets.append(sock)
            return sock

        def term(self):
            self.terminated = True

    fake.sockets: list[Any] = []
    fake.Context = lambda: Context()
    fake.SUB = "SUB"
    fake.SUBSCRIBE = "SUBSCRIBE"
    fake.CONFLATE = "CONFLATE"
    fake.NOBLOCK = 1
    fake.RCVTIMEO = "RCVTIMEO"
    fake.Again = type("Again", (Exception,), {})
    fake.ZMQError = type("ZMQError", (Exception,), {})
    # pyzmq exposes the timeout exception as ``zmq.error.Again``, which is
    # what the transport catches to keep polling.
    fake.error = module("zmq.error", Again=fake.Again, ZMQError=fake.ZMQError)
    return fake


def keyboard() -> types.ModuleType:
    """Return an ``evdev`` module with one keyboard that types labels.

    ``RLINF_FAKE_KEYS`` names the keys it presses, cycling forever, so a run
    that waits for an operator can reach its target unattended. Left unset the
    device reports no presses, which is the shape a test wants when the
    keyboard is only incidental.
    """
    import os
    import time

    codes = types.SimpleNamespace(EV_KEY=1, KEY_A=30, KEY_B=48, KEY_C=46, KEY_Q=16)
    by_name = {"a": codes.KEY_A, "b": codes.KEY_B, "c": codes.KEY_C, "q": codes.KEY_Q}
    # The listener maps a code back to its name through bytype.
    codes.bytype = {
        codes.EV_KEY: {code: f"KEY_{n.upper()}" for n, code in by_name.items()}
    }

    class Device:
        def __init__(self, path: str) -> None:
            self.name = "fake keyboard"
            self.path = path
            self.closed = False

        def capabilities(self, verbose: bool = False) -> dict[int, list[int]]:
            # Every key the listener requires, so it accepts this as a keyboard.
            return {codes.EV_KEY: [codes.KEY_A, codes.KEY_B, codes.KEY_C, codes.KEY_Q]}

        def read_loop(self):
            keys = [
                by_name[name.strip().lower()]
                for name in os.environ.get("RLINF_FAKE_KEYS", "").split(",")
                if name.strip().lower() in by_name
            ]
            if not keys:
                # No script: block rather than spin, as a quiet keyboard does.
                while not self.closed:
                    time.sleep(0.05)
                return
            # A consumer may read the held key rather than the press edge, so
            # each key is held for a beat before it is released.
            dwell = float(os.environ.get("RLINF_FAKE_KEY_DWELL", "0.3"))
            while not self.closed:
                for code in keys:
                    yield types.SimpleNamespace(type=codes.EV_KEY, code=code, value=1)
                    time.sleep(dwell)
                    yield types.SimpleNamespace(type=codes.EV_KEY, code=code, value=0)
                    time.sleep(0.02)

        def close(self) -> None:
            self.closed = True

    return module(
        "evdev",
        InputDevice=Device,
        list_devices=lambda: ["/dev/input/event-fake"],
        ecodes=codes,
    )


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Return fake teleoperation SDKs keyed by import name."""
    zmq = vr()
    made: dict[str, types.ModuleType] = {
        "pyspacemouse": spacemouse(),
        "zmq": zmq,
        # The transport imports the submodule name to catch its timeout.
        "zmq.error": zmq.error,
        "evdev": keyboard(),
    }
    made.update(gello())
    made.update(glove())
    return made
