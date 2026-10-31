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

"""Servo-bus plumbing shared by the LeKiwi bring-up commands.

These commands talk to the wheels directly rather than through
:class:`~rlinf.robotics.parts.mobility.LeKiwiBase`, because assigning motor IDs
and reading raw velocities are exactly what they check. For the same reason
they log through the standard library: importing the training stack would cost
the robot's computer hundreds of megabytes before the first motor write.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rlinf.robotics.parts.mobility.lekiwi import WHEEL_IDS, WHEEL_ORDER

if TYPE_CHECKING:
    from lerobot.motors.feetech import FeetechMotorsBus


def cli_logger() -> logging.Logger:
    """Return the process logger, configured once for command-line output."""
    logger = logging.getLogger("rlinf.lekiwi")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def make_wheel_bus(port: str) -> FeetechMotorsBus:
    """Create a bus that talks to the three wheel servos and nothing else."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    return FeetechMotorsBus(
        port=port,
        motors={
            name: Motor(WHEEL_IDS[name], "sts3215", MotorNormMode.RANGE_M100_100)
            for name in WHEEL_ORDER
        },
    )


def start_velocity_mode(bus: FeetechMotorsBus) -> None:
    """Connect, put every wheel in velocity mode at zero, and enable torque."""
    bus.connect()
    bus.disable_torque()
    for name in WHEEL_ORDER:
        bus.write("Operating_Mode", name, 1)
        bus.write("Acceleration", name, 10)
        bus.write("Goal_Velocity", name, 0, normalize=False)
    bus.enable_torque()


def stop_and_disconnect(bus: FeetechMotorsBus, logger: logging.Logger) -> None:
    """Command a stop on every wheel, drop torque, then close the serial port.

    A wheel that stopped answering must not keep the other two moving or leave
    the port open, so each write is attempted on its own.
    """
    try:
        # Stop reachable wheels before waiting for any missing wheel's reply.
        try:
            bus.sync_write(
                "Goal_Velocity", dict.fromkeys(WHEEL_ORDER, 0), normalize=False
            )
        except Exception:  # noqa: BLE001 - keep stopping the remaining wheels
            logger.exception("Could not send the initial wheel stop")
        for name in WHEEL_ORDER:
            try:
                bus.write("Goal_Velocity", name, 0, normalize=False, num_retry=2)
            except Exception:  # noqa: BLE001 - keep stopping the remaining wheels
                logger.exception("Could not send a stop to the %s wheel", name)
        for name in WHEEL_ORDER:
            try:
                bus.disable_torque(name, num_retry=2)
            except Exception:  # noqa: BLE001 - keep releasing the remaining wheels
                logger.exception("Could not disable torque on the %s wheel", name)
    finally:
        bus.disconnect(disable_torque=False)
