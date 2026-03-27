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

"""Keyboard listener that receives key events via a Unix domain socket.

Designed for Ray worker processes that cannot access the keyboard directly.
A companion relay script (``toolkits/dosw1_keyboard_relay.py``) captures
local keyboard events with *pynput* and forwards them over the socket.

Protocol (line-based, UTF-8):
    press:<char>   -- a key was pressed
    release        -- the key was released
"""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path


_DEFAULT_SOCKET_PATH = "/tmp/dosw1_keyboard.sock"


class SocketKeyboardListener:
    """Drop-in replacement for ``KeyboardListener`` that reads from a socket."""

    def __init__(self, socket_path: str = _DEFAULT_SOCKET_PATH) -> None:
        self.socket_path = socket_path
        self._lock = threading.Lock()
        self._latest_key: str | None = None
        self._running = True

        self._ensure_socket_dir()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server.bind(self.socket_path)
        self._server.listen(1)
        self._server.settimeout(1.0)

        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def get_key(self) -> str | None:
        """Returns the latest key pressed (same interface as KeyboardListener)."""
        with self._lock:
            return self._latest_key

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=3.0)
        self._server.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def _ensure_socket_dir(self) -> None:
        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_client, args=(conn,), daemon=True
            ).start()

    def _handle_client(self, conn: socket.socket) -> None:
        buf = b""
        try:
            conn.settimeout(1.0)
            while self._running:
                try:
                    data = conn.recv(256)
                except socket.timeout:
                    continue
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._process_line(line.decode("utf-8", errors="ignore").strip())
        except OSError:
            pass
        finally:
            conn.close()

    def _process_line(self, line: str) -> None:
        if line.startswith("press:"):
            key = line[6:]
            with self._lock:
                self._latest_key = key if key else None
        elif line == "release":
            with self._lock:
                self._latest_key = None
