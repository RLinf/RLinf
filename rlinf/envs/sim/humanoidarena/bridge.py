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

"""Synchronous action handoff over the unmodified Arena HTTP protocol."""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from rlinf.envs.sim.humanoidarena.runtime import decode_observation, validate_actions


class ArenaBridge:
    """Pause upstream inference requests until RLinf supplies one action chunk."""

    def __init__(self, chunk_size: int, timeout: float) -> None:
        self.chunk_size, self.timeout = chunk_size, timeout
        self.observations: queue.Queue = queue.Queue(maxsize=1)
        self.actions: queue.Queue = queue.Queue(maxsize=1)
        self.error: Exception | None = None
        self.policy_seed: int | None = None
        self.closed = threading.Event()
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                try:
                    size = int(self.headers.get("Content-Length", 0))
                    if not 0 < size <= 16 << 20:
                        raise ValueError("Expected a JSON body of at most 16 MiB")
                    payload = json.loads(self.rfile.read(size))
                    if self.path == "/reset":
                        bridge.policy_seed = payload.get("seed")
                        result = {"status": "ok", "seed": bridge.policy_seed}
                    elif self.path == "/infer":
                        rgb, state = decode_observation(payload)
                        bridge.observations.put(
                            (rgb, state, bridge.policy_seed), timeout=bridge.timeout
                        )
                        action = bridge.actions.get(timeout=bridge.timeout)
                        if action is None:
                            raise RuntimeError("RLinf closed the environment")
                        result = {"action_chunk": action.tolist()}
                    else:
                        self.send_error(404)
                        return
                    body = json.dumps(result, allow_nan=False).encode()
                    self.send_response(200)
                except Exception as exc:
                    bridge.error = exc
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        """Return the loopback address used by this simulator process."""
        return f"http://127.0.0.1:{self.server.server_port}"

    def send_action(self, actions: Any) -> None:
        """Send exactly one finite semantic40 chunk to the waiting simulator."""
        array = validate_actions(actions, self.chunk_size)
        if array.shape[0] != 1:
            raise ValueError("ArenaBridge requires exactly one environment")
        self.actions.put(array[0], timeout=self.timeout)

    def close(self) -> None:
        """Unblock inference before stopping the HTTP server."""
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            self.actions.put_nowait(None)
        except queue.Full:
            pass
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
