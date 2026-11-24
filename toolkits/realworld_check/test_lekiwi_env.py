#!/usr/bin/env python3
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

"""Hardware bring-up and teleoperation checks for the LeKiwi mobile base.

Run from the repository root with ``python -m``::

    python -m toolkits.realworld_check.test_lekiwi_env wheels --port /dev/ttyACM0
    python -m toolkits.realworld_check.test_lekiwi_env camera --save check.png
    python -m toolkits.realworld_check.test_lekiwi_env teleop --dry-run
    python -m toolkits.realworld_check.test_lekiwi_env teleop --port /dev/ttyACM0

For remote keyboard teleoperation over TCP::

    python -m toolkits.realworld_check.test_lekiwi_env teleop --listen --port /dev/ttyACM0
    python -m toolkits.realworld_check.test_lekiwi_env teleop --remote <robot-ip>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop

    from rlinf.robotics.parts.cameras.uvc import UVCCamera

#: Frame count used to estimate a camera frame rate.
SAMPLE_FRAMES = 10

#: Speed levels as ``(linear m/s, angular deg/s)``, slowest first.
SPEED_LEVELS = ((0.1, 30.0), (0.2, 60.0), (0.3, 90.0))
SPEED_NAMES = ("slow", "medium", "fast")

#: TCP port ``--listen`` binds and ``--remote`` connects to.
DEFAULT_PORT = 5555

# The shared motor helpers accept a standard logger for cleanup failures.
_MOTOR_LOGGER = logging.getLogger(__name__)


# ───────────────────────────── wheel check ──────────────────────────────────


def spin_each_wheel(
    bus: FeetechMotorsBus,
    *,
    raw: int,
    seconds: float,
) -> dict[str, int]:
    """Spin every wheel alone at ``raw`` and return its measured velocity.

    Each wheel is measured while it is still moving, because a stopped wheel
    reports zero either way.
    """
    from rlinf.robotics.parts.mobility.lekiwi import (
        WHEEL_ORDER,
        start_velocity_mode,
        stop_and_disconnect,
    )

    measured: dict[str, int] = {}
    try:
        start_velocity_mode(bus, logger=_MOTOR_LOGGER)
        for name in WHEEL_ORDER:
            command = dict.fromkeys(WHEEL_ORDER, 0)
            command[name] = raw
            print(f"Spinning the {name} wheel at raw {raw}")
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
        if bus.is_connected:
            stop_and_disconnect(bus, _MOTOR_LOGGER)
    return measured


def report(raw: int, measured: dict[str, int]) -> bool:
    """Print one line per wheel and return whether every wheel agreed."""
    from rlinf.robotics.parts.mobility.lekiwi import WHEEL_ORDER

    ok = True
    for name in WHEEL_ORDER:
        value = measured[name]
        # The encoder sign follows the commanded sign on a healthy wheel.
        agreed = value * raw > 0
        ok = ok and agreed
        status = "ok" if agreed else "MISMATCH: check this motor's wiring and ID"
        print(f"{name:<5} commanded {raw:6d}  measured {value:6d}  {status}")
    return ok


def _run_wheels(args: argparse.Namespace) -> int:
    """Run the wheel check, reporting per wheel whether it turned and read back."""
    from rlinf.robotics.parts.mobility.lekiwi import make_wheel_bus

    bus = make_wheel_bus(args.port)
    try:
        measured = spin_each_wheel(bus, raw=args.raw, seconds=args.seconds)
    except KeyboardInterrupt:
        print("Wheel check interrupted; motor cleanup attempted.")
        return 1
    except Exception:  # noqa: BLE001 - report the hardware failure and exit non-zero
        print("Wheel check failed", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0 if report(args.raw, measured) else 1


# ───────────────────────────── camera check ─────────────────────────────────


def pick_device(requested: str | None) -> str:
    """Return the device to open, listing the alternatives when ambiguous."""
    from rlinf.robotics.parts.cameras.uvc import UVCCamera

    if requested is not None:
        return requested
    found = sorted(UVCCamera.discover())
    if not found:
        raise RuntimeError(
            "No UVC camera found. Check /dev/v4l/by-id/ and lsusb on Linux."
        )
    print(f"Webcams found: {', '.join(found)}")
    if len(found) > 1:
        raise RuntimeError(
            "Several webcams found; pass --device with one of: " + ", ".join(found)
        )
    return found[0]


def measure_frames(camera: UVCCamera, frames: int) -> tuple[np.ndarray, float]:
    """Capture ``frames`` and return the last one with the observed frame rate."""
    started = time.monotonic()
    last = None
    for _ in range(frames):
        last = camera.get_observation()["frame"]
    elapsed = time.monotonic() - started
    fps = frames / elapsed if elapsed > 0 else 0.0
    print(f"Captured {frames} frames in {elapsed:.2f} s ({fps:.1f} fps)")
    return last, fps


def _run_camera(args: argparse.Namespace) -> int:
    """Check the camera end to end: discover, open, capture, report."""
    import numpy as np

    from rlinf.robotics.parts.cameras.base import CameraInfo
    from rlinf.robotics.parts.cameras.uvc import UVCCamera

    camera = None
    try:
        device = pick_device(args.device)
        camera = UVCCamera(
            CameraInfo(
                name="front",
                serial_number=device,
                camera_type="uvc",
                resolution=(args.width, args.height),
                fps=args.fps,
            )
        )
        camera.connect()
        if not camera.is_ready(timeout=3.0):
            raise RuntimeError(
                f"{device} opened but delivered no frame within 3 s; "
                "check the cable and that no other program holds the camera."
            )
        print(f"Using {device} at {args.width}x{args.height}")
        frame, _fps = measure_frames(camera, SAMPLE_FRAMES)
        print(
            f"Frame shape {frame.shape}, dtype {frame.dtype}, "
            f"mean brightness {float(np.mean(frame)):.1f}"
        )
        if frame.shape[:2] != (args.height, args.width):
            print(
                f"Expected {args.width}x{args.height} but the device delivered {frame.shape[1]}x{frame.shape[0]}; pick a resolution the camera supports natively.",
                file=sys.stderr,
            )
        if args.save:
            import cv2

            if not cv2.imwrite(args.save, frame):
                raise OSError(f"Could not save camera frame to {args.save!r}")
            print(f"Saved the last frame to {args.save}")
    except KeyboardInterrupt:
        print("Camera check interrupted.")
        return 1
    except Exception:  # noqa: BLE001 - report the hardware failure and exit non-zero
        print("Camera check failed", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if camera is not None and camera.is_connected:
            camera.disconnect()
    return 0


# ───────────────────────────── teleoperation ────────────────────────────────


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


def _drive(
    read_keys: Callable[[], tuple[set[str], int | None] | None],
    bus: FeetechMotorsBus | None,
    *,
    wheel_radius: float,
    base_radius: float,
    max_raw: int,
    fps: float,
) -> None:
    """Apply local or remote input until quit, input loss, or an exception."""
    from rlinf.robotics.parts.mobility.lekiwi import (
        body_to_wheel_raw,
        limit_wheel_raw,
        start_velocity_mode,
        stop_and_disconnect,
    )

    control = KeyboardControl()
    previous_command = None
    try:
        if bus is not None:
            start_velocity_mode(bus, logger=_MOTOR_LOGGER)
        print("W/A/S/D move; Z/X turn; R/F change speed; Q/Esc quit.")
        while True:
            started = time.monotonic()
            sample = read_keys()
            if sample is None:
                break
            keys, speed = sample
            if keys & {"q", "esc"}:
                break
            previous_speed = control.speed_index
            if speed is not None:
                control.speed_index = speed
            raw = body_to_wheel_raw(
                *control.action(keys),
                wheel_radius=wheel_radius,
                base_radius=base_radius,
            )
            command, _scale = limit_wheel_raw(raw, max_raw)
            if control.speed_index != previous_speed:
                print(f"Speed: {SPEED_NAMES[control.speed_index]}")
            if bus is not None:
                bus.sync_write("Goal_Velocity", command, normalize=False)
            elif command != previous_command:
                print(f"Preview wheel velocities: {command}")
            previous_command = command
            time.sleep(max(1.0 / fps - (time.monotonic() - started), 0.0))
    finally:
        if bus is not None and bus.is_connected:
            stop_and_disconnect(bus, _MOTOR_LOGGER)


def run_teleoperation(
    keyboard: KeyboardTeleop,
    bus: FeetechMotorsBus | None,
    *,
    wheel_radius: float,
    base_radius: float,
    max_raw: int,
    fps: float,
) -> None:
    """Connect a local keyboard and release it after the shared control loop."""

    def read_keys() -> tuple[set[str], None] | None:
        if not keyboard.is_connected:
            return None
        keys = {key.lower() for key in keyboard.get_action() if isinstance(key, str)}
        return (keys, None) if keyboard.is_connected else None

    try:
        keyboard.connect()
        if not keyboard.is_connected:
            raise RuntimeError(
                "Keyboard capture is unavailable; check desktop permissions."
            )
        _drive(
            read_keys,
            bus,
            wheel_radius=wheel_radius,
            base_radius=base_radius,
            max_raw=max_raw,
            fps=fps,
        )
    finally:
        if keyboard.is_connected:
            keyboard.disconnect()


class SocketKeySource:
    """Read complete key snapshots from one TCP session with a finite timeout."""

    def __init__(self, host: str, port: int, input_timeout: float = 2.0) -> None:
        self.host, self.port = host, port
        self.input_timeout = input_timeout
        self._server: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._buffer = b""

    def start(self) -> None:
        """Bind synchronously so startup errors reach the caller."""
        self._server = socket.socket()
        try:
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((self.host, self.port))
            self._server.listen(1)
            self.port = self._server.getsockname()[1]
        except BaseException:
            self.stop()
            raise

    def accept(self) -> None:
        """Wait for a sender before initializing the motors."""
        self._conn, _address = self._server.accept()
        self._conn.settimeout(self.input_timeout)

    def read_keys(self) -> tuple[set[str], int] | None:
        """Return the latest complete snapshot, or None on sender disconnect."""
        deadline = time.monotonic() + self.input_timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Remote keyboard input timed out")
            self._conn.settimeout(remaining)
            chunk = self._conn.recv(4096)
            if not chunk:
                return None
            self._buffer += chunk
            if len(self._buffer) > 4096:
                raise ValueError("Remote key snapshot exceeds 4096 bytes")
        complete, self._buffer = self._buffer.rsplit(b"\n", 1)
        keys, speed = json.loads(complete.split(b"\n")[-1])
        if (
            not isinstance(keys, list)
            or not all(isinstance(key, str) and key in set("wasdzx") for key in keys)
            or type(speed) is not int
            or not 0 <= speed < len(SPEED_LEVELS)
        ):
            raise ValueError("Expected movement keys and a valid speed level")
        return set(keys), speed

    def stop(self) -> None:
        """Close the session and listening socket."""
        for sock in (self._conn, self._server):
            if sock is not None:
                sock.close()
        self._conn = self._server = None


def run_receiver(
    source: SocketKeySource,
    bus: FeetechMotorsBus | None,
    *,
    wheel_radius: float,
    base_radius: float,
    max_raw: int,
    fps: float,
) -> None:
    """Run one remote session; input loss exits through motor cleanup."""
    _drive(
        source.read_keys,
        bus,
        wheel_radius=wheel_radius,
        base_radius=base_radius,
        max_raw=max_raw,
        fps=fps,
    )


def run_sender(host: str, port: int, connect_timeout: float, fps: float = 30.0) -> None:
    """Send held movement keys and speed while keyboard capture is alive."""
    from pynput import keyboard

    keys: set[str] = set()
    control = KeyboardControl()
    lock = threading.Lock()
    quit_event = threading.Event()

    def update(key: object, pressed: bool) -> None:
        name = getattr(key, "char", None)
        if key == keyboard.Key.esc or (name and name.lower() == "q"):
            quit_event.set()
            return
        if not name or name.lower() not in "wasdzxrf":
            return
        name = name.lower()
        with lock:
            if pressed:
                keys.add(name)
            else:
                keys.discard(name)
            # Consume every speed-key edge, including taps between snapshots.
            control.action(keys)

    with socket.create_connection((host, port), timeout=connect_timeout) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        listener = keyboard.Listener(
            on_press=lambda key: update(key, True),
            on_release=lambda key: update(key, False),
        )
        try:
            listener.start()
            print(f"Forwarding keys to {host}:{port}. Q/Esc or Ctrl+C quits.")
            while not quit_event.is_set():
                if not listener.is_alive():
                    listener.join()
                    raise RuntimeError("Keyboard capture stopped")
                with lock:
                    payload = json.dumps(
                        [sorted(keys & set("wasdzx")), control.speed_index]
                    )
                sock.sendall((payload + "\n").encode())
                quit_event.wait(1.0 / fps)
        finally:
            listener.stop()


def _handle_sigterm(signum: int, frame: object) -> None:
    """Turn SIGTERM into the same cleanup path as Ctrl+C."""
    raise KeyboardInterrupt


def _run_local(args: argparse.Namespace) -> int:
    """Read the keyboard of this machine and drive the wheels."""
    from rlinf.robotics.parts.mobility.lekiwi import make_wheel_bus

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
        )
    except KeyboardInterrupt:
        print("Teleoperation interrupted; motor cleanup attempted.")
    except Exception:  # noqa: BLE001 - report the hardware failure and exit non-zero
        print("Base teleoperation failed", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


def _run_receiver(args: argparse.Namespace) -> int:
    """Drive the wheels from keys received over TCP."""
    from rlinf.robotics.parts.mobility.lekiwi import make_wheel_bus

    source = SocketKeySource(args.listen_host, args.remote_port, args.input_timeout)
    try:
        source.start()
        print(f"Receiver listening on {source.host}:{source.port}")
        source.accept()
        bus = None if args.dry_run else make_wheel_bus(args.port)
        run_receiver(
            source,
            bus,
            wheel_radius=args.wheel_radius,
            base_radius=args.base_radius,
            max_raw=args.max_raw,
            fps=args.fps,
        )
    except KeyboardInterrupt:
        print("Teleoperation interrupted; motor cleanup attempted.")
    except Exception:  # noqa: BLE001 - report the hardware failure and exit non-zero
        print("Receiver failed", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        source.stop()
    return 0


def _run_sender(args: argparse.Namespace) -> int:
    """Stream this machine's key presses to a listening robot."""
    try:
        run_sender(args.remote, args.remote_port, args.connect_timeout, args.fps)
    except KeyboardInterrupt:
        print("Sender stopped.")
    except Exception as exc:  # noqa: BLE001 - CLI errors return non-zero
        print(f"Sender failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_teleop(args: argparse.Namespace) -> int:
    """Run the keyboard teleoperation half the options select."""
    previous_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        if args.remote:
            return _run_sender(args)
        if args.listen:
            return _run_receiver(args)
        return _run_local(args)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


# ───────────────────────────── CLI parser ───────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate subcommand options before opening hardware."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    wheels = sub.add_parser(
        "wheels",
        help="Spin each wheel in turn and read its velocity back.",
    )
    wheels.add_argument(
        "--port", required=True, help="USB servo controller port; see lerobot-find-port"
    )
    wheels.add_argument(
        "--raw", type=int, default=100, help="Spin speed in servo units (default: 100)"
    )
    wheels.add_argument(
        "--seconds", type=float, default=0.6, help="Time per wheel (default: 0.6 s)"
    )

    camera = sub.add_parser(
        "camera",
        help="Open the camera and report the frames it delivers.",
    )
    camera.add_argument(
        "--device",
        help="Camera serial or index; defaults to the only webcam found",
    )
    camera.add_argument("--width", type=int, default=640, help="Frame width")
    camera.add_argument("--height", type=int, default=480, help="Frame height")
    camera.add_argument("--fps", type=int, default=30, help="Requested frame rate")
    camera.add_argument(
        "--save",
        metavar="PATH",
        help="Write the last frame to this image file, e.g. check.png",
    )

    teleop = sub.add_parser(
        "teleop",
        help="Drive the base with a local or remote keyboard.",
    )
    teleop.add_argument(
        "--port", help="USB servo controller port; see lerobot-find-port"
    )
    teleop.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the wheel commands without opening a motor port",
    )
    teleop.add_argument(
        "--fps", type=float, default=30.0, help="Control frequency (default: 30 Hz)"
    )
    teleop.add_argument(
        "--wheel-radius", type=float, default=0.05, help="Wheel radius in metres"
    )
    teleop.add_argument(
        "--base-radius",
        type=float,
        default=0.125,
        help="Centre-to-wheel distance in metres",
    )
    teleop.add_argument(
        "--max-raw",
        type=int,
        default=3000,
        help="Absolute servo velocity limit (default: 3000)",
    )
    role = teleop.add_mutually_exclusive_group()
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
    teleop.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="Interface --listen binds (default: %(default)s)",
    )
    teleop.add_argument(
        "--remote-port",
        type=int,
        default=DEFAULT_PORT,
        help="TCP port used by --listen and --remote (default: %(default)s)",
    )
    teleop.add_argument(
        "--input-timeout",
        type=float,
        default=2.0,
        help="Stop the base when --listen hears nothing from the sender for this "
        "many seconds (default: %(default)s)",
    )
    teleop.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Connection and send timeout in seconds (default: %(default)s)",
    )

    args = parser.parse_args(argv)

    if args.command == "wheels":
        if not 1 <= args.raw <= 32767:
            wheels.error("--raw must be in 1..32767")
        if not math.isfinite(args.seconds) or args.seconds <= 0:
            wheels.error("--seconds must be finite and greater than zero")
        return args

    if args.command == "camera":
        if args.width <= 0 or args.height <= 0:
            camera.error("--width and --height must be greater than zero")
        if args.fps <= 0:
            camera.error("--fps must be greater than zero")
        return args

    if not 1 <= args.remote_port <= 65535:
        teleop.error("--remote-port must be in 1..65535")
    if not math.isfinite(args.connect_timeout) or args.connect_timeout <= 0:
        teleop.error("--connect-timeout must be finite and greater than zero")

    if not math.isfinite(args.fps) or args.fps <= 0:
        teleop.error("--fps must be finite and greater than zero")

    if args.remote:
        # The sender owns no hardware: it only forwards keys.
        for flag in ("port", "dry_run"):
            if getattr(args, flag) is not None and getattr(args, flag) is not False:
                teleop.error(f"--{flag.replace('_', '-')} is not used with --remote")
        return args

    for name in ("wheel_radius", "base_radius"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            teleop.error(
                f"--{name.replace('_', '-')} must be finite and greater than zero"
            )
    if not 1 <= args.max_raw <= 32767:
        teleop.error("--max-raw must be in 1..32767")
    if args.listen:
        if not math.isfinite(args.input_timeout) or args.input_timeout <= 0:
            teleop.error("--input-timeout must be finite and greater than zero")
    if not args.dry_run and not args.port:
        teleop.error("--port is required unless --dry-run is used")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the requested LeKiwi hardware check subcommand."""
    args = parse_args(argv)
    handlers = {
        "wheels": _run_wheels,
        "camera": _run_camera,
        "teleop": _run_teleop,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
