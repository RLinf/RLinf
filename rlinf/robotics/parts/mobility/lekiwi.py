# Copyright 2026 The RLinf Authors.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""LeKiwi mobile base, driven over its three-wheel Feetech serial bus.

The base is three STS3215 servos (left 7, back 8, right 9) mounted at 120
degree spacing. It is omnidirectional, so one body-frame velocity vector
``[x, y, theta]`` fully commands it. Units follow lerobot's LeKiwi driver:
``x`` and ``y`` in metres per second and ``theta`` in degrees per second.

The servo bus only knows per-wheel velocity, so the conversion between the
body frame and the wheel frame lives here. ``send_action`` clips a command
proportionally at ``max_raw`` so an unreachable direction is slowed rather
than skewed; ``get_observation`` reads the wheels' ``Present_Velocity`` and
inverts the same mapping.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from rlinf.utils.logging import get_logger

from ..base import Action, Features, Observation
from .base import MobileBase

if TYPE_CHECKING:
    from lerobot.motors.feetech import FeetechMotorsBus

#: Wheel motor IDs on the shared servo bus, keyed by their position name.
WHEEL_IDS: dict[str, int] = {"left": 7, "back": 8, "right": 9}

#: Mounting angle of each wheel in the base frame, in degrees.
WHEEL_ANGLES: dict[str, float] = {"left": 150.0, "back": -90.0, "right": 30.0}

#: Wheels in the order ``WHEEL_ANGLES``/``WHEEL_IDS`` are read and written.
WHEEL_ORDER: tuple[str, ...] = ("left", "back", "right")

#: Servo velocity register: raw counts per second to radians per second.
_RAD_PER_RAW: float = (2.0 * math.pi) / 4096.0
_RAW_PER_RAD: float = 4096.0 / (2.0 * math.pi)


def body_to_wheel_raw(
    x: float,
    y: float,
    theta: float,
    *,
    wheel_radius: float = 0.05,
    base_radius: float = 0.125,
) -> dict[str, float]:
    """Convert a body twist to per-wheel raw servo velocity, unclipped.

    Args:
        x: Forward velocity in metres per second.
        y: Lateral velocity in metres per second, positive to the left.
        theta: Rotation in degrees per second, positive counter-clockwise.
        wheel_radius: Wheel radius in metres.
        base_radius: Centre-to-wheel distance in metres.

    Returns:
        Raw velocity per wheel name, in servo counts per second.
    """
    if not all(math.isfinite(value) for value in (x, y, theta)):
        raise ValueError("Velocities must be finite numbers.")
    if wheel_radius <= 0 or base_radius <= 0:
        raise ValueError("Wheel radius and base radius must be positive.")
    rotation = math.radians(theta) * base_radius
    return {
        name: (
            math.cos(math.radians(WHEEL_ANGLES[name])) * x
            + math.sin(math.radians(WHEEL_ANGLES[name])) * y
            + rotation
        )
        / wheel_radius
        * _RAW_PER_RAD
        for name in WHEEL_ORDER
    }


def limit_wheel_raw(
    raw: dict[str, float], max_raw: int
) -> tuple[dict[str, int], float]:
    """Scale a wheel command down so no wheel exceeds ``max_raw``.

    Scaling every wheel by the same factor slows an unreachable direction
    instead of skewing it. Returns the integer servo command and the factor
    applied, so the caller can report the velocity actually commanded.
    """
    if not 1 <= max_raw <= 32767:
        raise ValueError(f"max_raw must be in 1..32767, got {max_raw}.")
    peak = max(abs(value) for value in raw.values())
    scale = max(1.0, peak / max_raw)
    return {name: round(value / scale) for name, value in raw.items()}, scale


@MobileBase.register("lekiwi")
class LeKiwiBase(MobileBase):
    """LeKiwi wheel base on a Feetech serial bus, via lerobot.

    Args:
        port: Serial device the servo bus is on, such as ``/dev/ttyACM0``.
        wheel_radius: Wheel radius in metres.
        base_radius: Centre-to-wheel distance in metres.
        max_raw: Absolute per-wheel servo velocity limit. A body velocity whose
            fastest wheel exceeds this is scaled down proportionally.
    """

    SDK = "lerobot.motors"

    def __init__(
        self,
        port: str,
        *,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> None:
        self._logger = get_logger()
        self._port = port
        self.wheel_radius = wheel_radius
        self.base_radius = base_radius
        self.max_raw = max_raw

    @property
    def observation_features(self) -> Features:
        """Declare the body-frame velocity, ``[x, y, theta]`` in m/s and deg/s."""
        return {"velocity": {"shape": (3,), "dtype": "float32"}}

    @property
    def action_features(self) -> Features:
        """Declare the body-frame velocity, ``[x, y, theta]`` in m/s and deg/s."""
        return {"velocity": {"shape": (3,), "dtype": "float32"}}

    def _open(self) -> FeetechMotorsBus:
        """Open the servo bus and return the connected :class:`FeetechMotorsBus`.

        The bus starts with torque disabled while each wheel is set to velocity
        mode, zero speed, and a gentle acceleration, then torque is enabled.
        """
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        bus = FeetechMotorsBus(
            port=self._port,
            motors={
                name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)
                for name, motor_id in WHEEL_IDS.items()
            },
        )
        bus.connect()
        try:
            bus.disable_torque()
            for name in WHEEL_ORDER:
                bus.write("Operating_Mode", name, 1)
                bus.write("Acceleration", name, 10)
                bus.write("Goal_Velocity", name, 0, normalize=False)
            bus.enable_torque()
        except BaseException:
            # A bus that opened but failed to configure must still be closed.
            try:
                bus.disconnect(disable_torque=False)
            finally:
                raise
        self._logger.info("LeKiwi base connected on %s", self._port)
        return bus

    def _release(self, device: FeetechMotorsBus) -> None:
        """Stop the wheels, drop torque, and close the serial bus."""
        try:
            device.sync_write(
                "Goal_Velocity", dict.fromkeys(WHEEL_ORDER, 0), normalize=False
            )
        except Exception:  # noqa: BLE001 - the bus may already be gone
            self._logger.exception("Could not send the final wheel stop")
        try:
            device.disable_torque()
        except Exception:  # noqa: BLE001 - the bus may already be gone
            self._logger.exception("Could not disable torque on release")
        finally:
            device.disconnect(disable_torque=False)

    def reset(self) -> None:
        """Stop the base by commanding zero velocity on every wheel."""
        if self.is_connected:
            self._device.sync_write(
                "Goal_Velocity", dict.fromkeys(WHEEL_ORDER, 0), normalize=False
            )

    def get_observation(self) -> Observation:
        """Read the wheels and report the body-frame velocity in m/s and deg/s."""
        raw = self._device.sync_read(
            "Present_Velocity", list(WHEEL_ORDER), normalize=False
        )
        return {
            "velocity": self._wheel_raw_to_body([raw[name] for name in WHEEL_ORDER])
        }

    def send_action(self, action: Action) -> Observation:
        """Command a body-frame velocity, returning the velocity actually sent."""
        if set(action) != {"velocity"}:
            raise KeyError("A LeKiwi base action holds only 'velocity'.")
        velocity = np.asarray(action["velocity"], dtype=np.float32)
        if velocity.shape != (3,):
            raise ValueError(f"Expected velocity shape (3,), got {velocity.shape}.")
        if not np.all(np.isfinite(velocity)):
            raise ValueError("A LeKiwi velocity must be finite.")

        x, y, theta = (float(value) for value in velocity)
        raw = body_to_wheel_raw(
            x,
            y,
            theta,
            wheel_radius=self.wheel_radius,
            base_radius=self.base_radius,
        )
        commanded, scale = limit_wheel_raw(raw, self.max_raw)
        self._device.sync_write("Goal_Velocity", commanded, normalize=False)
        applied = np.asarray([x / scale, y / scale, theta / scale], dtype=np.float32)
        return {"velocity": applied}

    def _wheel_raw_to_body(self, raw: list[int]) -> np.ndarray:
        """Invert :meth:`_body_to_wheel_raw`: raw wheel velocity to body twist."""
        linear = [value * self.wheel_radius * _RAD_PER_RAW for value in raw]
        matrix = [
            [
                math.cos(math.radians(WHEEL_ANGLES[name])),
                math.sin(math.radians(WHEEL_ANGLES[name])),
                1.0,
            ]
            for name in WHEEL_ORDER
        ]
        x, y, rotation = np.linalg.solve(
            np.asarray(matrix, dtype=float), np.asarray(linear)
        )
        theta = math.degrees(rotation / self.base_radius)
        return np.asarray([x, y, theta], dtype=np.float32)
