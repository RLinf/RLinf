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

"""A localhost HTTP server that records POSTs and answers from a script."""

from __future__ import annotations

import email.parser
import email.policy
import http.server
import threading


class FakeHTTPServer:
    """Answers the n-th POST with ``responses[n]``; the last response repeats.

    Each response is ``(status, content_type, body_bytes)``. Every request is
    recorded in :attr:`requests` as a dict with ``path``, ``content_type`` and
    ``body``.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                server.requests.append(
                    {
                        "path": self.path,
                        "content_type": self.headers.get("Content-Type", ""),
                        "body": body,
                    }
                )
                index = min(len(server.requests), len(server.responses)) - 1
                status, content_type, payload = server.responses[index]
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"

    def __enter__(self) -> "FakeHTTPServer":
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def parse_multipart(content_type: str, body: bytes):
    """Return ``(fields, files)``: name to str, and name to ``(filename, bytes, content_type)``."""
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )
    fields, files = {}, {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if part.get_filename() is None:
            fields[name] = payload.decode()
        else:
            files[name] = (part.get_filename(), payload, part.get_content_type())
    return fields, files
