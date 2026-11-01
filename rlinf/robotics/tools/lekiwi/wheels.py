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

"""Spin each LeKiwi wheel in turn and read its velocity back.

Run this after ``lerobot-setup-motors``: it proves every ID answers, that
each wheel turns the way its name says, and that the encoders report the
motion. Put the base on blocks so the wheels spin free.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import TYPE_CHECKING

from rlinf.robotics.parts.mobility.lekiwi import WHEEL_ORDER

from ._support import (
    cli_logger,
    make_wheel_bus,
    start_velocity_mode,
    stop_and_disconnect,
)

if TYPE_CHECKING:
    from lerobot.motors.feetech import FeetechMotorsBus


def spin_each_wheel(
    bus: FeetechMotorsBus,
    *,
    raw: int,
    seconds: float,
    logger: logging.Logger,
) -> dict[str, int]:
    """Spin every wheel alone at ``raw`` and return its measured velocity.

    Each wheel is measured while it is still moving, because a stopped wheel
    reports zero either way.
    """
    start_velocity_mode(bus)
    measured: dict[str, int] = {}
    try:
        for name in WHEEL_ORDER:
            command = dict.fromkeys(WHEEL_ORDER, 0)
            command[name] = raw
            logger.info("Spinning the %s wheel at raw %d", name, raw)
            bus.sync_write("Goal_Velocity", command, normalize=False)
            time.sleep(seconds / 2.0)
            reading = bus.sync_read(
                "Present_Velocity", list(WHEEL_ORDER), normalize=False
            )
            measured[name] = int(reading[name])
            time.sleep(seconds / 2.0)
            bus.sync_write(
                "Goal_Velocity", dict.fromkeys(WHEEL_ORDER, 0), normalize=False
            )
            time.sleep(0.1)
    finally:
        stop_and_disconnect(bus, logger)
    return measured


def report(raw: int, measured: dict[str, int], logger: logging.Logger) -> bool:
    """Log one line per wheel and return whether every wheel agreed."""
    ok = True
    for name in WHEEL_ORDER:
        value = measured[name]
        # The encoder sign follows the commanded sign on a healthy wheel.
        agreed = value * raw > 0
        ok = ok and agreed
        logger.info(
            "%-5s commanded %6d  measured %6d  %s",
            name,
            raw,
            value,
            "ok" if agreed else "MISMATCH: check this motor's wiring and ID",
        )
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the port, the spin speed, and how long each wheel turns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", required=True, help="USB servo controller port; see lerobot-find-port"
    )
    parser.add_argument(
        "--raw", type=int, default=100, help="Spin speed in servo units (default: 100)"
    )
    parser.add_argument(
        "--seconds", type=float, default=0.6, help="Time per wheel (default: 0.6 s)"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.raw <= 32767:
        parser.error("--raw must be in 1..32767")
    if not args.seconds > 0:
        parser.error("--seconds must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the wheel check, reporting per wheel whether it turned and read back."""
    args = parse_args(argv)
    logger = cli_logger()
    bus = make_wheel_bus(args.port)
    try:
        measured = spin_each_wheel(
            bus, raw=args.raw, seconds=args.seconds, logger=logger
        )
    except KeyboardInterrupt:
        logger.info("Wheel check interrupted; motor cleanup attempted.")
        return 1
    except Exception:  # noqa: BLE001 - report the hardware failure and exit non-zero
        logger.exception("Wheel check failed")
        return 1
    return 0 if report(args.raw, measured, logger) else 1


if __name__ == "__main__":
    raise SystemExit(main())
