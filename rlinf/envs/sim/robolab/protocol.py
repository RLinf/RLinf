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

"""Framed messages between the RoboLab adapter and its simulator process."""

import pickle
import socket
import struct
from typing import Any


def receive(sock: socket.socket) -> dict[str, Any]:
    """Receive one message, raising EOFError if the peer disconnects."""
    header = sock.recv(8, socket.MSG_WAITALL)
    if len(header) != 8:
        raise EOFError("RoboLab peer closed the connection")
    size = struct.unpack("!Q", header)[0]
    parts = []
    while size:
        part = sock.recv(min(size, 1024 * 1024))
        if not part:
            raise EOFError("RoboLab peer closed the connection")
        parts.append(part)
        size -= len(part)
    return pickle.loads(b"".join(parts))


def send(sock: socket.socket, value: dict[str, Any]) -> None:
    """Send one message with its payload length."""
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)
