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

"""The TCP half of keyboard teleoperation.

Keyboard capture reads the local desktop input, so the machine wired to the
motors cannot be driven over SSH. ``rlinf-lekiwi-teleoperate --listen`` on that
machine and ``rlinf-lekiwi-teleoperate --remote HOST`` where the keyboard is
split the command into the two halves this module implements.

The wire protocol is newline-delimited text: ``down <key>``, ``up <key>``, and
a ``ping`` heartbeat. A listener whose sender stops sending heartbeats stops
the base, so a stalled controlling machine cannot leave it moving.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time
from typing import TYPE_CHECKING

from rlinf.robotics.parts.mobility.lekiwi import (
    WHEEL_ORDER,
    body_to_wheel_raw,
    limit_wheel_raw,
)

from ._support import start_velocity_mode, stop_and_disconnect
from .teleop import SPEED_NAMES, KeyboardControl

if TYPE_CHECKING:
    from lerobot.motors.feetech import FeetechMotorsBus

HEARTBEAT_INTERVAL = 0.5
_DOWN = "down"
_UP = "up"
_PING = "ping"


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
