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

"""Fake operator hardware: spacemouse, leader arms, glove, VR and keyboard.

Each answers as a device at rest, so a binding reads something well-shaped and
reports nobody driving. A test that wants motion sets the values itself::

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
    """A ``pyspacemouse`` whose puck rests until a test moves it."""
    fake = module("pyspacemouse")
    fake.twist = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    fake.buttons = (0, 0)

    class Device:
        def read(self):
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
            return None

    fake.open = lambda **_kwargs: Device()
    return fake


def gello() -> dict[str, types.ModuleType]:
    """A ``gello_teleop`` leader arm holding a fixed pose."""
    agent_module = module("gello_teleop.gello_teleop_agent")
    fk_module = module("gello_teleop.franka_fk")

    class GelloTeleopAgent:
        def __init__(self, port=None, **_kwargs):
            self.port = port

        def get_action(self):
            return np.zeros(DOF), 0.0

        def close(self):
            return None

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
    """A ``rlinf_dexhand`` glove reporting open fingers."""

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
    """A ``zmq`` that delivers whatever a test puts in the queue."""
    fake = module("zmq")
    fake.packets: list[Any] = []

    class Socket:
        def __init__(self, *_a, **_k):
            self.subscribed = []

        def connect(self, address):
            self.address = address

        def setsockopt_string(self, *_a, **_k):
            return None

        def setsockopt(self, *_a, **_k):
            return None

        def close(self):
            return None

        def recv_string(self, flags=0):
            if not fake.packets:
                raise fake.Again()
            return fake.packets.pop(0)

        recv = recv_string

    class Context:
        def socket(self, _kind):
            return Socket()

        def term(self):
            return None

    fake.Context = lambda: Context()
    fake.SUB = "SUB"
    fake.SUBSCRIBE = "SUBSCRIBE"
    fake.CONFLATE = "CONFLATE"
    fake.NOBLOCK = 1
    fake.RCVTIMEO = "RCVTIMEO"
    fake.Again = type("Again", (Exception,), {})
    fake.ZMQError = type("ZMQError", (Exception,), {})
    return fake


def keyboard() -> types.ModuleType:
    """An ``evdev`` with no keyboards attached."""
    return module(
        "evdev",
        InputDevice=lambda _path: types.SimpleNamespace(
            name="fake", path=_path, read_loop=lambda: iter(()), close=lambda: None
        ),
        list_devices=lambda: [],
        ecodes=types.SimpleNamespace(EV_KEY=1, KEY_A=30, KEY_B=48, KEY_C=46),
    )


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Every operator-hardware SDK, by the name a reader imports it as."""
    made: dict[str, types.ModuleType] = {
        "pyspacemouse": spacemouse(),
        "zmq": vr(),
        "evdev": keyboard(),
    }
    made.update(gello())
    made.update(glove())
    return made
