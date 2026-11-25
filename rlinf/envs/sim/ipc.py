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

"""Shared socket IPC framing and physical GPU helpers for simulator subprocesses."""

from __future__ import annotations

import pickle
import socket
import struct
from collections.abc import Mapping
from typing import Any


def receive_message(sock: socket.socket, *, peer_name: str = "Simulator") -> Any:
    """Receive one length-prefixed pickle message, raising ``EOFError`` on disconnect."""
    header = sock.recv(8, socket.MSG_WAITALL)
    if len(header) != 8:
        raise EOFError(f"{peer_name} peer closed the connection")
    size = struct.unpack("!Q", header)[0]
    parts: list[bytes] = []
    while size:
        part = sock.recv(min(size, 1024 * 1024))
        if not part:
            raise EOFError(f"{peer_name} peer closed the connection")
        parts.append(part)
        size -= len(part)
    return pickle.loads(b"".join(parts))


def send_message(
    sock: socket.socket,
    value: Any,
    *,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> None:
    """Send one length-prefixed pickle message."""
    payload = pickle.dumps(value, protocol=protocol)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def resolve_physical_cuda_device(
    device: str,
    env: Mapping[str, str],
) -> tuple[int, dict[str, str]]:
    """Map a logical CUDA device string to a physical GPU index for Omniverse Kit.

    When ``CUDA_VISIBLE_DEVICES`` lists numeric GPU indices, unmask it in the
    returned environment because Kit's Vulkan device indices are not remapped.
    """
    resolved_env = dict(env)
    if not device.startswith("cuda"):
        return 0, resolved_env
    ordinal = int(device.split(":")[1]) if ":" in device else 0
    if ordinal < 0:
        raise ValueError(f"Expected nonnegative CUDA ordinal, found {ordinal}.")
    physical = ordinal
    visible = resolved_env.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [item.strip() for item in visible.split(",") if item.strip()]
        if entries and all(item.isdigit() for item in entries):
            if ordinal >= len(entries):
                raise ValueError(
                    f"Device {device!r} is outside CUDA_VISIBLE_DEVICES={visible!r}."
                )
            physical = int(entries[ordinal])
            resolved_env.pop("CUDA_VISIBLE_DEVICES", None)
    return physical, resolved_env
