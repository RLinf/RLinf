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

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .tracer import DistTracer


class TraceServer:
    """Central HTTP collector that receives trace events from all DistTracer clients.

    Endpoints: ``GET /sync`` (clock synchronization), ``GET /health`` (healthcheck),
    and ``POST /trace`` (appends a JSON list of events to the output file as JSONL).
    Stdlib-only; runs in a daemon background thread via :meth:`start`.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8888,
        output_file: str = "trace_events.jsonl",
    ):
        """Initialize the trace server. Pass port=0 to bind an ephemeral port."""
        self._host = host
        self._output_file = os.path.abspath(output_file)
        self._file_lock = threading.Lock()
        self._thread: threading.Thread = None

        # Ensure directory exists before starting
        dir_name = os.path.dirname(self._output_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self._httpd.daemon_threads = True

    @property
    def ip(self) -> str:
        """The IP address at which this server is reachable from other nodes."""
        return DistTracer.detect_node_ip()

    @property
    def port(self) -> int:
        """The actual bound port of the server."""
        return self._httpd.server_address[1]

    @property
    def output_file(self) -> str:
        """The absolute path of the JSONL output file."""
        return self._output_file

    def _make_handler(self):
        server = self

        class TraceRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Route request logs to the worker logger at debug level
                DistTracer.logger().debug(
                    "%s - %s" % (self.address_string(), format % args)
                )

            def _send_json(self, status: int, payload: dict):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/sync":
                    self._send_json(200, {"server_time_us": time.time_ns() // 1000})
                elif self.path in ("/health", "/status"):
                    self._send_json(200, {"status": "ok"})
                else:
                    self._send_json(404, {"detail": "Not found"})

            def do_POST(self):
                if self.path != "/trace":
                    self._send_json(404, {"detail": "Not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    events = json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception as e:
                    self._send_json(400, {"detail": f"Invalid JSON: {e}"})
                    return
                if not isinstance(events, list):
                    self._send_json(
                        400, {"detail": "Expected a JSON list of trace events"}
                    )
                    return
                try:
                    with server._file_lock:
                        with open(server._output_file, "a") as f:
                            for event in events:
                                f.write(json.dumps(event) + "\n")
                except Exception as e:
                    DistTracer.logger().error(
                        f"Failed to write trace events to file: {e}"
                    )
                    self._send_json(
                        500, {"detail": "Failed to write trace events to file"}
                    )
                    return
                self._send_json(200, {"status": "success", "count": len(events)})

        return TraceRequestHandler

    def start(self) -> "TraceServer":
        """Start serving in a daemon background thread and return self."""
        assert self._thread is None, "Trace server has already been started."
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="rlinf-trace-server", daemon=True
        )
        self._thread.start()
        DistTracer.logger().info(
            f"Trace server started on {self._host}:{self.port}, "
            f"writing trace events to {self._output_file}"
        )
        return self

    def stop(self):
        """Stop the server and wait for its background thread to exit."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
