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

"""Framed messages between the RoboUniverse adapter and its simulator subprocess."""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

# Protocol 4 works across Python 3.9 through 3.14 so RLinf's parent venv and an
# external Isaac Sim 5.1.0 Python 3.11 interpreter interoperate cleanly.
_PICKLE_PROTOCOL = 4


def receive(sock: socket.socket) -> dict[str, Any]:
    """Receive one framed message, raising ``EOFError`` if the peer disconnects."""
    header = sock.recv(8, socket.MSG_WAITALL)
    if len(header) != 8:
        raise EOFError("RoboUniverse simulator peer closed the connection")
    size = struct.unpack("!Q", header)[0]
    parts = []
    while size:
        part = sock.recv(min(size, 1024 * 1024))
        if not part:
            raise EOFError("RoboUniverse simulator peer closed the connection")
        parts.append(part)
        size -= len(part)
    return pickle.loads(b"".join(parts))


def send(sock: socket.socket, value: dict[str, Any]) -> None:
    """Send one framed message prefixed with its 8-byte payload length."""
    payload = pickle.dumps(value, protocol=_PICKLE_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)
