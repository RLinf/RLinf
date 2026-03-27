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
from datetime import datetime


_DEFAULT_SOCKET_PATH = "/tmp/dosw1_keyboard.sock"

_KEY_DESCRIPTIONS: dict[str, str] = {
    "s": "START episode (exit free-teleop)",
    "r": "ABORT episode -> free-teleop",
    "p": "PAUSE control",
    "t": "TELEOP (from pause)",
    "m": "MODEL  (from pause)",
}

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _connect(sock_path: str) -> socket.socket:
    """Connect to the SocketKeyboardListener, retrying until available."""
    attempt = 0
    while True:
        try:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.connect(sock_path)
            return conn
        except (FileNotFoundError, ConnectionRefusedError):
            attempt += 1
            dots = "." * ((attempt % 3) + 1)
            print(
                f"\r{_DIM}[{_ts()}] Waiting for socket {sock_path}{dots}   {_RESET}",
                end="",
                flush=True,
            )
            time.sleep(2.0)


def _print_banner() -> None:
    print(f"\n{_BOLD}{'=' * 52}{_RESET}")
    print(f"{_BOLD}  DOSW1 Keyboard Relay{_RESET}")
    print(f"{'=' * 52}")
    print(f"  {_CYAN}s{_RESET}  start episode    {_CYAN}r{_RESET}  abort episode")
    print(f"  {_CYAN}p{_RESET}  pause            {_CYAN}t{_RESET}  teleop (from pause)")
    print(f"  {_CYAN}m{_RESET}  model (from pause)")
    print(f"  {_RED}q{_RESET}  quit relay")
    print(f"{'=' * 52}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="DOSW1 keyboard relay")
    parser.add_argument(
        "--socket-path",
        default=_DEFAULT_SOCKET_PATH,
        help="Unix domain socket path (default: %(default)s)",
    )
    args = parser.parse_args()

    print(f"{_DIM}[{_ts()}] Connecting to {args.socket_path} ...{_RESET}", flush=True)
    conn = _connect(args.socket_path)
    print(f"\r{_GREEN}[{_ts()}] Connected.{' ' * 40}{_RESET}")
    _print_banner()

    from pynput import keyboard

    quit_flag = False
    send_count = 0

    def on_press(key):
        nonlocal quit_flag, send_count
        char = getattr(key, "char", None)
        if char is None:
            return
        if char == "q":
            print(f"{_DIM}[{_ts()}] Quit requested.{_RESET}", flush=True)
            quit_flag = True
            return False

        desc = _KEY_DESCRIPTIONS.get(char)
        if desc:
            color = _YELLOW if char in ("r", "p") else _GREEN
            print(
                f"  {_DIM}[{_ts()}]{_RESET} {color}{_BOLD}[{char}]{_RESET} {desc}",
                flush=True,
            )
        else:
            print(
                f"  {_DIM}[{_ts()}]{_RESET} {_DIM}[{char}] (not a control key){_RESET}",
                flush=True,
            )

        try:
            conn.sendall(f"press:{char}\n".encode("utf-8"))
            send_count += 1
        except OSError as e:
            print(f"{_RED}[{_ts()}] Send failed: {e}{_RESET}", flush=True)
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
            print(f"\n{_DIM}[{_ts()}] Interrupted.{_RESET}", flush=True)

    conn.close()
    print(
        f"{_DIM}[{_ts()}] Disconnected. Total key events sent: {send_count}{_RESET}",
        flush=True,
    )


if __name__ == "__main__":
    main()
