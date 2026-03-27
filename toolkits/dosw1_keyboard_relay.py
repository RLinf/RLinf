#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
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

"""Keyboard relay for DOSW1 human-in-the-loop control.

Run this script in a terminal on the robot node **before** or **after**
launching Ray training with ``enable_human_in_loop: True``.  It captures
local keyboard events via *pynput* and forwards them to the
``SocketKeyboardListener`` running inside the Ray env worker through a
Unix domain socket.

Usage::

    python toolkits/dosw1_keyboard_relay.py [--socket-path /tmp/dosw1_keyboard.sock]

Keys:
    s  -- start episode (exit free-teleop)
    r  -- abort episode, return to free-teleop
    p  -- pause model / teleop control
    t  -- switch to teleop (from pause)
    m  -- switch back to model (from pause)
    q  -- quit this relay script
"""

from __future__ import annotations

import argparse
import socket
import sys
import time


_DEFAULT_SOCKET_PATH = "/tmp/dosw1_keyboard.sock"


def _connect(sock_path: str) -> socket.socket:
    """Connect to the SocketKeyboardListener, retrying until available."""
    while True:
        try:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.connect(sock_path)
            return conn
        except (FileNotFoundError, ConnectionRefusedError):
            print(
                f"[relay] Waiting for socket {sock_path} "
                "(start Ray training with enable_human_in_loop: True) ...",
                flush=True,
            )
            time.sleep(2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="DOSW1 keyboard relay")
    parser.add_argument(
        "--socket-path",
        default=_DEFAULT_SOCKET_PATH,
        help="Unix domain socket path (default: %(default)s)",
    )
    args = parser.parse_args()

    conn = _connect(args.socket_path)
    print(
        "[relay] Connected. Press s/r/p/t/m to control the robot, q to quit.",
        flush=True,
    )

    from pynput import keyboard

    quit_flag = False

    def on_press(key):
        nonlocal quit_flag
        char = getattr(key, "char", None)
        if char is None:
            return
        if char == "q":
            quit_flag = True
            return False
        try:
            conn.sendall(f"press:{char}\n".encode("utf-8"))
        except OSError:
            quit_flag = True
            return False

    def on_release(_key):
        try:
            conn.sendall(b"release\n")
        except OSError:
            pass

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            while not quit_flag:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    conn.close()
    print("[relay] Disconnected.", flush=True)


if __name__ == "__main__":
    main()
