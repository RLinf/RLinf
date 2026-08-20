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

"""The devices an operator drives, as parts.

Each class here wraps one vendor reader from :mod:`.readers` and presents it the
way every other part is presented: connect, report an observation, disconnect.
The observation is whatever the operator did, in the device's own terms -- a
twist, a set of joint angles, a grip. Turning that into a command for a robot is
the job of a binding, in :mod:`rlinf.robotics.teleop`.

Opening the hardware happens in ``connect`` rather than ``__init__``, so a device
can be declared on one machine and built on another with
``SpaceMouse.at(node_rank=1)``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import RobotPart


class TeleopPart(RobotPart):
    """A device the operator drives.

    Like every part, it says what its hardware is in :meth:`_open` and reads it
    in :meth:`get_observation`. Vendor readers vary in how they are released,
    so :meth:`_release` tries the two names they use.
    """

    def _release(self, device: Any) -> None:
        """Release the reader by whichever name its vendor gave the method."""
        for method in ("close", "stop"):
            release = getattr(device, method, None)
            if callable(release):
                release()
                return

    @property
    def ready(self) -> bool:
        """Whether the device has produced a usable reading yet.

        Leader arms stream, so the first reads after connecting can be empty.
        A device whose reader says nothing about this is ready once connected.
        """
        return bool(getattr(self._device, "ready", self.is_connected))


class SpaceMouse(TeleopPart):
    """A 6-DoF mouse: a twist from the puck, and two buttons.

    Args:
        device_index: Which HID device to open when several are attached.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index

    def _open(self) -> Any:
        from .readers.spacemouse import SpaceMouseExpert

        return SpaceMouseExpert(device_index=self._device_index)

    @property
    def observation_features(self) -> dict[str, Any]:
        """A twist and the button states."""
        return {
            "twist": {"shape": (6,), "dtype": "float32"},
            "buttons": {"shape": (2,), "dtype": "bool"},
        }

    def get_observation(self) -> dict[str, Any]:
        """Read the puck and the buttons."""
        twist, buttons = self._device.get_action()
        return {
            "twist": np.asarray(twist, dtype=np.float32),
            "buttons": np.asarray(buttons, dtype=bool),
        }


class TeleopLeaderArm(TeleopPart):
    """A leader arm the operator poses by hand.

    Two readers exist for the GELLO hardware and they report different things:
    the Cartesian one gives a target pose, the joint one gives joint positions.
    ``joint_space`` picks between them.

    Args:
        port: Serial port of the leader arm.
        joint_space: Report joint positions rather than a Cartesian target.
    """

    def __init__(self, port: str, joint_space: bool = False) -> None:
        self._port = port
        self._joint_space = joint_space

    def _open(self) -> Any:
        if self._joint_space:
            from .readers.gello_joint import GelloJointExpert

            return GelloJointExpert(port=self._port)
        from .readers.gello import GelloExpert

        return GelloExpert(port=self._port)

    @property
    def observation_features(self) -> dict[str, Any]:
        """Joint positions, or a Cartesian target, plus the grip."""
        if self._joint_space:
            return {
                "joint_position": {"shape": (7,), "dtype": "float32"},
                "grip": {"shape": (1,), "dtype": "float32"},
            }
        return {
            "position": {"shape": (3,), "dtype": "float32"},
            "orientation": {"shape": (4,), "dtype": "float32"},
            "grip": {"shape": (1,), "dtype": "float32"},
        }

    def get_observation(self) -> dict[str, Any]:
        """Read the arm the operator is holding."""
        if self._joint_space:
            joints, grip = self._device.get_action()
            return {
                "joint_position": np.asarray(joints, dtype=np.float32),
                "grip": np.asarray(grip, dtype=np.float32).reshape(1),
            }
        position, orientation, grip = self._device.get_action()
        return {
            "position": np.asarray(position, dtype=np.float32),
            "orientation": np.asarray(orientation, dtype=np.float32),
            "grip": np.asarray(grip, dtype=np.float32).reshape(1),
        }


class Glove(TeleopPart):
    """A data glove reporting finger angles.

    Args:
        left_port: Serial port of the left glove.
        right_port: Serial port of the right glove.
        frequency: Polling rate in Hz.
        config_file: Optional calibration file.
    """

    def __init__(
        self,
        left_port: Optional[str] = "/dev/ttyACM0",
        right_port: Optional[str] = None,
        frequency: int = 60,
        config_file: Optional[str] = None,
    ) -> None:
        self._left_port = left_port
        self._right_port = right_port
        self._frequency = frequency
        self._config_file = config_file

    def _open(self) -> Any:
        from .readers.glove import GloveExpert

        return GloveExpert(
            left_port=self._left_port,
            right_port=self._right_port,
            frequency=self._frequency,
            config_file=self._config_file,
        )

    @property
    def observation_features(self) -> dict[str, Any]:
        """One angle per finger joint."""
        return {"angles": {"shape": (6,), "dtype": "float32"}}

    def get_observation(self) -> dict[str, Any]:
        """Read the operator's finger angles."""
        return {"angles": np.asarray(self._device.get_angles(), dtype=np.float32)}


class PicoController(TeleopPart):
    """A VR controller reporting how far the operator has moved it.

    The reading is the motion since the operator took hold, in the robot's
    axes, plus the grip and gripper buttons. Where the robot was when they took
    hold is not the controller's business, so the binding remembers that.

    Args:
        pico_config: Forwarded to the vendor reader; ``hand`` selects a side.
    """

    def __init__(self, **pico_config: Any) -> None:
        self._config = pico_config

    def _open(self) -> Any:
        from .readers.pico import PicoExpert

        return PicoExpert(**self._config)

    @property
    def observation_features(self) -> dict[str, Any]:
        """Whether the operator is driving, and how far they have moved."""
        return {
            "held": {"dtype": "bool", "shape": ()},
            "position_delta": {"dtype": "float64", "shape": (3,)},
            "rotation_delta": {"dtype": "float64", "shape": (3,)},
            "grip_close": {"dtype": "bool", "shape": ()},
            "grip_open": {"dtype": "bool", "shape": ()},
        }

    def get_observation(self) -> dict[str, Any]:
        """Read the controller."""
        return self._device.get_reading()
