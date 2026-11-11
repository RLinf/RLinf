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
import contextlib
import logging
import math
import signal
import socket
import threading
import time
from collections.abc import Collection
from typing import TYPE_CHECKING

import numpy as np

from rlinf.robotics.parts.cameras.base import CameraInfo
from rlinf.robotics.parts.cameras.webcam import WebcamCamera
from rlinf.robotics.parts.mobility.lekiwi import (
    WHEEL_ORDER,
    body_to_wheel_raw,
    limit_wheel_raw,
    make_wheel_bus,
    start_velocity_mode,
    stop_and_disconnect,
)
from rlinf.utils.logging import get_logger as cli_logger

if TYPE_CHECKING:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop

#: Frame count used to estimate a camera frame rate.
SAMPLE_FRAMES = 10

#: Speed levels as ``(linear m/s, angular deg/s)``, slowest first.
SPEED_LEVELS = ((0.1, 30.0), (0.2, 60.0), (0.3, 90.0))
SPEED_NAMES = ("slow", "medium", "fast")

#: TCP port ``--listen`` binds and ``--remote`` connects to.
DEFAULT_PORT = 5555
HEARTBEAT_INTERVAL = 0.5
_DOWN = "down"
_UP = "up"
_PING = "ping"


# ───────────────────────────── wheel check ──────────────────────────────────


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
    measured: dict[str, int] = {}
    try:
        start_velocity_mode(bus)
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
        if bus.is_connected:
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


def _run_wheels(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run the wheel check, reporting per wheel whether it turned and read back."""
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


# ───────────────────────────── camera check ─────────────────────────────────


def pick_device(requested: str | None, logger: logging.Logger) -> str:
    """Return the device to open, listing the alternatives when ambiguous."""
    found = sorted(WebcamCamera.discover())
    if not found:
        raise RuntimeError(
            "No webcam found. On Linux check /dev/v4l/by-id/ and lsusb; "
            "elsewhere close any other program using the camera."
        )
    logger.info("Webcams found: %s", ", ".join(found))
    if requested is not None:
        return requested
    if len(found) > 1:
        raise RuntimeError(
            "Several webcams found; pass --device with one of: " + ", ".join(found)
        )
    return found[0]


def measure_frames(
    camera: WebcamCamera, frames: int, logger: logging.Logger
) -> tuple[np.ndarray, float]:
    """Capture ``frames`` and return the last one with the observed frame rate."""
    started = time.monotonic()
    last = None
    for _ in range(frames):
        last = camera.get_observation()["frame"]
    elapsed = time.monotonic() - started
    logger.info(
        "Captured %d frames in %.2f s (%.1f fps)",
        frames,
        elapsed,
        frames / elapsed if elapsed > 0 else 0.0,
    )
    return last, frames / elapsed if elapsed > 0 else 0.0


def _run_camera(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Check the camera end to end: discover, open, capture, report."""
    camera = None
    try:
        device = pick_device(args.device, logger)
        camera = WebcamCamera(
            CameraInfo(
                name="front",
                serial_number=device,
                camera_type="webcam",
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
        logger.info("Using %s at %dx%d", device, args.width, args.height)
        frame, _fps = measure_frames(camera, SAMPLE_FRAMES, logger)
        logger.info(
            "Frame shape %s, dtype %s, mean brightness %.1f",
            frame.shape,
            frame.dtype,
            float(np.mean(frame)),
        )
        if frame.shape[:2] != (args.height, args.width):
            logger.warning(
                "Expected %dx%d but the device delivered %dx%d; "
                "pick a resolution the camera supports natively.",
                args.width,
                args.height,
                frame.shape[1],
                frame.shape[0],
            )
        if args.save:
            import cv2

            cv2.imwrite(args.save, frame)
            logger.info("Saved the last frame to %s", args.save)
    except KeyboardInterrupt:
        logger.info("Camera check interrupted.")
        return 1
    except Exception:  # noqa: BLE001 - report the hardware failure and exit non-zero
        logger.exception("Camera check failed")
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


class SocketKeySource:
    """Held keys fed by a remote sender over TCP.

    A daemon thread accepts sender connections and parses ``down`` / ``up`` /
    ``ping`` lines into a held-key set. :meth:`snapshot` reports the connection
    as live only while heartbeats arrive, so a dead or stalled sender reads as
    disconnected and the control loop stops the base.
    """

    def __init__(self, host: str, port: int, heartbeat_timeout: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.heartbeat_timeout = heartbeat_timeout
        self._pressed: set[str] = set()
        self._conn: socket.socket | None = None
        self._last_msg = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin accepting senders on a background thread."""
        self._thread = threading.Thread(
            target=self._serve, name="key-receiver", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Close the listening socket and wait for the accept thread."""
        self._stop.set()
        with self._lock:
            conn = self._conn
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> tuple[bool, set[str]]:
        """Return ``(connected, held_keys)`` atomically.

        ``connected`` is true only while a sender is attached and its heartbeats
        are recent.
        """
        with self._lock:
            connected = self._conn is not None and (
                time.monotonic() - self._last_msg <= self.heartbeat_timeout
            )
            return connected, set(self._pressed)

    def _serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(0.5)  # let the loop notice stop() promptly
            while not self._stop.is_set():
                try:
                    conn, _addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self._handle_conn(conn)

    def _handle_conn(self, conn: socket.socket) -> None:
        with self._lock:
            self._conn = conn
            self._pressed.clear()
            self._last_msg = time.monotonic()
        conn.settimeout(0.5)
        buffer = b""
        try:
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    with self._lock:
                        expired = (
                            time.monotonic() - self._last_msg > self.heartbeat_timeout
                        )
                    if expired:
                        break
                    continue
                except OSError:
                    break
                if not data:
                    break
                buffer += data
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    self._handle_line(line.decode("utf-8", errors="ignore"))
        finally:
            with contextlib.suppress(OSError):
                conn.close()
            with self._lock:
                self._conn = None
                self._pressed.clear()

    def _handle_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        event, _, key = line.partition(" ")
        with self._lock:
            self._last_msg = time.monotonic()
            if event == _PING:
                return
            if event == _DOWN:
                self._pressed.add(key)
            elif event == _UP:
                self._pressed.discard(key)


def run_receiver(
    source: SocketKeySource,
    bus: FeetechMotorsBus | None,
    *,
    wheel_radius: float,
    base_radius: float,
    max_raw: int,
    fps: float,
    logger: logging.Logger,
) -> None:
    """Drive the base from remote key events until Q/Esc, sender loss, or Ctrl+C."""
    control = KeyboardControl()
    previous_command = None
    was_connected = False
    try:
        if bus is not None:
            start_velocity_mode(bus)

        logger.info(
            "Receiver listening on %s:%d. Start the sender on the controlling "
            "machine. W/A/S/D move, Z/X turn, R/F change speed, Q or Esc quits.",
            source.host,
            source.port,
        )
        while True:
            connected, keys = source.snapshot()
            if connected and not was_connected:
                logger.info("Sender connected.")
            elif was_connected and not connected:
                logger.warning("Sender lost; stopping the base.")
                if bus is not None:
                    try:
                        bus.sync_write(
                            "Goal_Velocity",
                            dict.fromkeys(WHEEL_ORDER, 0),
                            normalize=False,
                        )
                    except Exception:  # noqa: BLE001 - keep the loop alive
                        logger.exception("Could not stop the base after sender loss")
            was_connected = connected

            if not connected:
                time.sleep(0.1)
                continue

            if "q" in keys or "esc" in keys:
                break

            started = time.monotonic()
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
        if bus is not None and bus.is_connected:
            stop_and_disconnect(bus, logger)


def run_sender(
    host: str, port: int, connect_timeout: float, logger: logging.Logger
) -> None:
    """Capture keys with pynput and stream press/release events to the receiver."""
    from pynput import keyboard

    sock = _connect(host, port, connect_timeout)
    stop_event = threading.Event()
    send_lock = threading.Lock()

    def send(payload: str) -> None:
        """Send one line, marking the session stopped on any socket error."""
        try:
            with send_lock:
                sock.sendall((payload + "\n").encode("utf-8"))
        except OSError:
            stop_event.set()

    def key_name(key: object) -> str | None:
        char = getattr(key, "char", None)
        if char and char.isprintable() and not char.isspace():
            return char.lower()
        if key == keyboard.Key.esc:
            return "esc"
        return None

    def on_press(key: object) -> None:
        name = key_name(key)
        if name is not None:
            send(f"{_DOWN} {name}")

    def on_release(key: object) -> None:
        name = key_name(key)
        if name is not None:
            send(f"{_UP} {name}")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    def heartbeat() -> None:
        while not stop_event.wait(HEARTBEAT_INTERVAL):
            send(_PING)

    threading.Thread(target=heartbeat, name="key-heartbeat", daemon=True).start()

    logger.info("Forwarding keys to %s:%d. Press Ctrl+C to stop.", host, port)
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        listener.stop()
        with contextlib.suppress(OSError):
            sock.close()


def _connect(host: str, port: int, timeout: float) -> socket.socket:
    """Open a TCP connection to the receiver, retrying until ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=min(timeout, 1.0))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise ConnectionError(
                    f"Could not connect to receiver at {host}:{port}: {exc}. "
                    "Start the receiver first."
                ) from exc
            time.sleep(0.5)


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
    try:
        run_sender(args.remote, args.remote_port, args.connect_timeout, logger)
    except KeyboardInterrupt:
        logger.info("Sender stopped.")
    except Exception:  # noqa: BLE001 - report the failure and exit non-zero
        logger.exception("Sender failed")
        return 1
    return 0


def _run_teleop(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run the keyboard teleoperation half the options select."""
    previous_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        if args.remote:
            return _run_sender(args, logger)
        if args.listen:
            return _run_receiver(args, logger)
        return _run_local(args, logger)
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
        "--heartbeat-timeout",
        type=float,
        default=2.0,
        help="Stop the base when --listen hears nothing from the sender for this "
        "many seconds (default: %(default)s)",
    )
    teleop.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Seconds --remote keeps retrying the connection (default: %(default)s)",
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

    if args.remote:
        # The sender owns no hardware: it only forwards keys.
        for flag in ("port", "dry_run"):
            if getattr(args, flag) is not None and getattr(args, flag) is not False:
                teleop.error(f"--{flag.replace('_', '-')} is not used with --remote")
        return args

    for name in ("fps", "wheel_radius", "base_radius"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            teleop.error(
                f"--{name.replace('_', '-')} must be finite and greater than zero"
            )
    if not 1 <= args.max_raw <= 32767:
        teleop.error("--max-raw must be in 1..32767")
    if args.listen:
        if not math.isfinite(args.heartbeat_timeout) or args.heartbeat_timeout <= 0:
            teleop.error("--heartbeat-timeout must be finite and greater than zero")
    if not args.dry_run and not args.port:
        teleop.error("--port is required unless --dry-run is used")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the requested LeKiwi hardware check subcommand."""
    args = parse_args(argv)
    logger = cli_logger()
    handlers = {
        "wheels": _run_wheels,
        "camera": _run_camera,
        "teleop": _run_teleop,
    }
    return handlers[args.command](args, logger)


if __name__ == "__main__":
    raise SystemExit(main())
