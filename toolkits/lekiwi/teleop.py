# Copyright 2026 The RLinf Authors.
# Copyright 2024-2025 The HuggingFace Inc. team. All rights reserved.
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

"""Drive the LeKiwi base with a keyboard.

The key mapping follows LeRobot's ``LeKiwiClient`` and the wheel conversion is
:func:`rlinf.robotics.parts.mobility.body_to_wheel_raw`, the same one the
runtime base uses.

By default the keyboard is read on the computer wired to the motors. Keyboard
capture reads local desktop input, so over SSH that computer has no keyboard to
read; ``--listen`` and ``--remote HOST`` split the two halves instead:

* ``--listen`` runs on the robot's computer and drives the wheels from the keys
  it receives over TCP.
* ``--remote HOST`` runs where the keyboard is and streams key presses to the
  listening robot.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import time
from collections.abc import Collection
from typing import TYPE_CHECKING

from rlinf.robotics.parts.mobility.lekiwi import body_to_wheel_raw, limit_wheel_raw

from ._support import (
    cli_logger,
    make_wheel_bus,
    start_velocity_mode,
    stop_and_disconnect,
)

if TYPE_CHECKING:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop

#: Speed levels as ``(linear m/s, angular deg/s)``, slowest first.
SPEED_LEVELS = ((0.1, 30.0), (0.2, 60.0), (0.3, 90.0))
SPEED_NAMES = ("slow", "medium", "fast")

#: TCP port ``--listen`` binds and ``--remote`` connects to.
DEFAULT_PORT = 5555


class KeyboardControl:
    """Map held keys to a chassis velocity, changing speed once per press."""

    def __init__(self) -> None:
        self.speed_index = 0
        self._previous: set[str] = set()

    def action(self, keys: Collection[str]) -> tuple[float, float, float]:
        """Return forward/lateral velocity in m/s and rotation in degrees/s."""
        keys = set(keys)
        pressed = keys - self._previous
        self._previous = keys
        if "r" in pressed:
            self.speed_index = min(self.speed_index + 1, len(SPEED_LEVELS) - 1)
        if "f" in pressed:
            self.speed_index = max(self.speed_index - 1, 0)
        xy, theta = SPEED_LEVELS[self.speed_index]
        return (
            xy * (("w" in keys) - ("s" in keys)),
            xy * (("a" in keys) - ("d" in keys)),
            theta * (("z" in keys) - ("x" in keys)),
        )


def run_teleoperation(
    keyboard: KeyboardTeleop,
    bus: FeetechMotorsBus | None,
    *,
    wheel_radius: float,
    base_radius: float,
    max_raw: int,
    fps: float,
    logger: logging.Logger,
) -> None:
    """Run until Q, Esc, keyboard loss, or an exception; always attempt cleanup.

    Passing ``bus=None`` previews the wheel commands without touching hardware.
    """
    control = KeyboardControl()
    previous_command = None
    try:
        keyboard.connect()
        if not keyboard.is_connected:
            raise RuntimeError(
                "Keyboard capture is unavailable. On macOS allow your terminal in "
                "Privacy & Security > Accessibility / Input Monitoring and restart it. "
                "On Linux use a local X11 desktop session."
            )
        if bus is not None:
            start_velocity_mode(bus)

        logger.info(
            "Ready at slow speed. Hold W/A/S/D to move, Z/X to turn. "
            "R/F change speed; Q, Esc or Ctrl+C quit. Release movement keys to stop."
        )
        while keyboard.is_connected:
            started = time.monotonic()
            keys = {
                key.lower() for key in keyboard.get_action() if isinstance(key, str)
            }
            if "q" in keys or not keyboard.is_connected:
                break
            previous_speed = control.speed_index
            raw = body_to_wheel_raw(
                *control.action(keys),
                wheel_radius=wheel_radius,
                base_radius=base_radius,
            )
            command, _scale = limit_wheel_raw(raw, max_raw)
            if control.speed_index != previous_speed:
                logger.info("Speed: %s", SPEED_NAMES[control.speed_index])
            if bus is not None:
                bus.sync_write("Goal_Velocity", command, normalize=False)
            elif command != previous_command:
                logger.info("Preview wheel velocities: %s", command)
            previous_command = command
            time.sleep(max(1.0 / fps - (time.monotonic() - started), 0.0))
    finally:
        try:
            if bus is not None and bus.is_connected:
                stop_and_disconnect(bus, logger)
        finally:
            if keyboard.is_connected:
                keyboard.disconnect()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate options before opening the keyboard or motor port."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", help="USB servo controller port; see lerobot-find-port"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the wheel commands without opening a motor port",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Control frequency (default: 30 Hz)"
    )
    parser.add_argument(
        "--wheel-radius", type=float, default=0.05, help="Wheel radius in metres"
    )
    parser.add_argument(
        "--base-radius",
        type=float,
        default=0.125,
        help="Centre-to-wheel distance in metres",
    )
    parser.add_argument(
        "--max-raw",
        type=int,
        default=3000,
        help="Absolute servo velocity limit (default: 3000)",
    )
    role = parser.add_mutually_exclusive_group()
    role.add_argument(
        "--listen",
        action="store_true",
        help="Run on the machine wired to the motors, taking keys over TCP",
    )
    role.add_argument(
        "--remote",
        metavar="HOST",
        help="Run where the keyboard is, streaming keys to the listening robot",
    )
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="Interface --listen binds (default: %(default)s)",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=DEFAULT_PORT,
        help="TCP port used by --listen and --remote (default: %(default)s)",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=2.0,
        help="Stop the base when --listen hears nothing from the sender for this "
        "many seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Seconds --remote keeps retrying the connection (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.remote_port <= 65535:
        parser.error("--remote-port must be in 1..65535")
    if not math.isfinite(args.connect_timeout) or args.connect_timeout <= 0:
        parser.error("--connect-timeout must be finite and greater than zero")

    if args.remote:
        # The sender owns no hardware: it only forwards keys.
        for flag in ("port", "dry_run"):
            if getattr(args, flag) is not None and getattr(args, flag) is not False:
                parser.error(f"--{flag.replace('_', '-')} is not used with --remote")
        return args

    for name in ("fps", "wheel_radius", "base_radius"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be finite and greater than zero"
            )
    if not 1 <= args.max_raw <= 32767:
        parser.error("--max-raw must be in 1..32767")
    if args.listen:
        if not math.isfinite(args.heartbeat_timeout) or args.heartbeat_timeout <= 0:
            parser.error("--heartbeat-timeout must be finite and greater than zero")
    if not args.dry_run and not args.port:
        parser.error("--port is required unless --dry-run is used")
    return args


def _handle_sigterm(signum: int, frame: object) -> None:
    """Turn SIGTERM into the same cleanup path as Ctrl+C."""
    raise KeyboardInterrupt


def _run_local(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Read the keyboard of this machine and drive the wheels."""
    try:
        from lerobot.teleoperators.keyboard.teleop_keyboard import (
            KeyboardTeleop,
            KeyboardTeleopConfig,
        )

        keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="lekiwi_base_keyboard"))
        bus = None if args.dry_run else make_wheel_bus(args.port)
        run_teleoperation(
            keyboard,
            bus,
            wheel_radius=args.wheel_radius,
            base_radius=args.base_radius,
            max_raw=args.max_raw,
            fps=args.fps,
            logger=logger,
        )
    except KeyboardInterrupt:
        logger.info("Teleoperation interrupted; motor cleanup attempted.")
    except Exception:  # noqa: BLE001 - report the hardware failure and exit non-zero
        logger.exception("Base teleoperation failed")
        return 1
    return 0


def _run_receiver(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Drive the wheels from keys received over TCP."""
    from .remote import SocketKeySource, run_receiver

    source = SocketKeySource(args.listen_host, args.remote_port, args.heartbeat_timeout)
    source.start()
    try:
        bus = None if args.dry_run else make_wheel_bus(args.port)
        run_receiver(
            source,
            bus,
            wheel_radius=args.wheel_radius,
            base_radius=args.base_radius,
            max_raw=args.max_raw,
            fps=args.fps,
            logger=logger,
        )
    except KeyboardInterrupt:
        logger.info("Teleoperation interrupted; motor cleanup attempted.")
    except Exception:  # noqa: BLE001 - report the hardware failure and exit non-zero
        logger.exception("Receiver failed")
        return 1
    finally:
        source.stop()
    return 0


def _run_sender(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Stream this machine's key presses to a listening robot."""
    from .remote import run_sender

    try:
        run_sender(args.remote, args.remote_port, args.connect_timeout, logger)
    except KeyboardInterrupt:
        logger.info("Sender stopped.")
    except Exception:  # noqa: BLE001 - report the failure and exit non-zero
        logger.exception("Sender failed")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the keyboard teleoperation half the options select."""
    args = parse_args(argv)
    logger = cli_logger()
    previous_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        if args.remote:
            return _run_sender(args, logger)
        if args.listen:
            return _run_receiver(args, logger)
        return _run_local(args, logger)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
