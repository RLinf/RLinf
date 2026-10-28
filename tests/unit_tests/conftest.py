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

"""Shared unit-test setup: import paths and a fake action server."""

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

for _path in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


class ActionServer:
    """A running fake of a msgpack ``POST /v1/actions/generations`` server.

    It decodes requests with plain ``msgpack`` so tests see the wire format
    rather than RLinf's codec. ``statuses`` scripts the next responses (200
    once it is empty) and ``reply`` builds the body of a 200 response.
    """

    path = "/v1/actions/generations"

    def __init__(self) -> None:
        self.requests: list[tuple[dict[str, str], Any]] = []
        self.statuses: list[int] = []
        self.reply: Callable[[Any], Any] = lambda body: {"data": []}
        self.content_type = "application/msgpack"
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                import msgpack

                raw = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path != server.path:
                    self._send(404, b"no such route", "text/plain")
                    return
                if "msgpack" not in self.headers.get("Content-Type", ""):
                    self._send(415, b"expected msgpack", "text/plain")
                    return
                body = msgpack.unpackb(raw, raw=False)
                server.requests.append((dict(self.headers), body))
                status = server.statuses.pop(0) if server.statuses else 200
                if status != 200:
                    self._send(status, b"server busy", "text/plain")
                    return
                packed = msgpack.packb(server.reply(body), use_bin_type=True)
                self._send(200, packed, server.content_type)

            def _send(self, status: int, payload: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def action_servers() -> Iterator[Callable[[], ActionServer]]:
    """Start fake action servers on demand and stop them after the test."""
    started: list[ActionServer] = []

    def start() -> ActionServer:
        started.append(ActionServer())
        return started[-1]

    yield start
    for server in started:
        server.close()
