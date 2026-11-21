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

"""Synchronous client for the versioned Tianji Stage1 service."""

from __future__ import annotations

import json
import socket
import struct
from collections.abc import Iterable

import numpy as np

from .contract import (
    PROTOCOL,
    Stage1Contract,
    Stage1Request,
    request_to_json,
    validate_response,
)
from .server import MAX_FRAME_BYTES


class Stage1Client:
    def __init__(
        self, host="127.0.0.1", port=6110, *, contract=Stage1Contract(), timeout_s=10.0
    ):
        self.host, self.port, self.contract, self.timeout_s = (
            host,
            int(port),
            contract,
            float(timeout_s),
        )

    def predict(self, request: Stage1Request) -> np.ndarray:
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        ) as connection:
            return self._predict_on_connection(connection, request)

    def predict_many(self, requests: Iterable[Stage1Request]) -> list[np.ndarray]:
        """Run consecutive requests on one TCP connection.

        The server intentionally supports multiple framed requests per
        connection.  Keeping this helper separate from ``predict`` makes the
        continuous-request smoke test explicit while preserving the simple
        one-request API used by deployment scripts.
        """
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        ) as connection:
            return [
                self._predict_on_connection(connection, request) for request in requests
            ]

    def _predict_on_connection(
        self, connection: socket.socket, request: Stage1Request
    ) -> np.ndarray:
        payload = json.dumps(
            request_to_json(request, self.contract), separators=(",", ":")
        ).encode()
        if len(payload) > MAX_FRAME_BYTES:
            raise ValueError("Stage1 request is too large")
        connection.sendall(struct.pack("!I", len(payload)) + payload)
        header = self._read_exact(connection, 4, "response header")
        size = struct.unpack("!I", header)[0]
        if size < 2 or size > MAX_FRAME_BYTES:
            raise ValueError("Stage1 response length is outside the safety bound")
        raw = self._read_exact(connection, size, "response body")
        response = json.loads(bytes(raw).decode("utf-8"))
        if "error" in response:
            raise RuntimeError(response["error"])
        if response.get("protocol") != PROTOCOL:
            raise ValueError("Stage1 response protocol mismatch")
        return validate_response(response.get("actions"), self.contract)

    @staticmethod
    def _read_exact(connection: socket.socket, size: int, description: str) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = connection.recv(size - len(result))
            if not chunk:
                raise ConnectionError(f"Stage1 server closed before {description}")
            result.extend(chunk)
        return bytes(result)
