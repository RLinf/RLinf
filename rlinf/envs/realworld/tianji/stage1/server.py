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

"""Standalone CPU-safe Stage1 server with a replaceable predictor plugin."""

from __future__ import annotations

import importlib
import json
import socket
import struct
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .contract import (
    PROTOCOL,
    Stage1Contract,
    Stage1Request,
    validate_request,
    validate_response,
)

MAX_FRAME_BYTES = 32 * 1024 * 1024


class ZeroStage1Predictor:
    """Hold-pose smoke predictor (legacy name zero); never a trained model."""

    def __init__(self, contract: Stage1Contract):
        self.contract = contract

    def __call__(self, request: Stage1Request) -> np.ndarray:
        return np.repeat(request.state[None], self.contract.action_chunk_length, axis=0)


def load_predictor(spec: str | None, contract: Stage1Contract):
    if not spec or spec == "zero":
        return ZeroStage1Predictor(contract)
    if ":" not in spec:
        raise ValueError("predictor must be 'zero' or module:callable")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    predictor = factory(contract)
    if not callable(predictor):
        raise TypeError("predictor factory must return a callable")
    return predictor


def _read_frame(connection: socket.socket) -> bytes:
    header = b""
    while len(header) < 4:
        chunk = connection.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("Stage1 client disconnected")
        header += chunk
    size = struct.unpack("!I", header)[0]
    if size < 2 or size > MAX_FRAME_BYTES:
        raise ValueError("Stage1 frame length is outside the safety bound")
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(min(1024 * 1024, size - len(result)))
        if not chunk:
            raise ConnectionError("Stage1 client disconnected mid-frame")
        result.extend(chunk)
    return bytes(result)


def _write_frame(connection: socket.socket, payload: bytes) -> None:
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("Stage1 response is too large")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


@dataclass
class Stage1Server:
    host: str = "127.0.0.1"
    port: int = 6110
    contract: Stage1Contract = Stage1Contract()
    predictor: Callable[[Stage1Request], np.ndarray] | None = None
    socket_timeout_s: float = 10.0

    def __post_init__(self):
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("port must be in 1..65535")
        if self.predictor is None:
            self.predictor = ZeroStage1Predictor(self.contract)

    def handle(self, connection: socket.socket) -> None:
        connection.settimeout(self.socket_timeout_s)
        while True:
            try:
                raw = _read_frame(connection)
            except ConnectionError:
                return
            try:
                request = validate_request(
                    json.loads(raw.decode("utf-8")), self.contract
                )
                actions = validate_response(self.predictor(request), self.contract)
                response = {"protocol": PROTOCOL, "actions": actions.tolist()}
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            _write_frame(
                connection, json.dumps(response, separators=(",", ":")).encode("utf-8")
            )

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, int(self.port)))
            listener.listen(4)
            while True:
                connection, _ = listener.accept()
                try:
                    self.handle(connection)
                finally:
                    connection.close()
