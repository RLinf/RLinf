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

import asyncio
import atexit
import functools
import json
import os
import random
import socket
import threading
import time
import urllib.request
from contextlib import contextmanager
from typing import Optional


class DistTracer:
    """Distributed tracing client that sends trace events to a central HTTP server.

    A process-wide singleton is managed via the `init`/`get` classmethods, and the
    `trace_begin`/`trace_end`/`trace_span`/`trace_func` classmethods trace through
    it (no-op when tracing is not initialized).
    """

    _instance: Optional["DistTracer"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        server_ip: Optional[str] = None,
        port: int = 8888,
        process_name: Optional[str] = None,
        thread_name: Optional[str] = None,
    ):
        """Initialize the tracer client; server_ip defaults to the local node IP."""
        self.server_ip = server_ip or self.detect_node_ip()
        self.port = port
        self.server_url = f"http://{self.server_ip}:{port}"
        # Trace traffic is intra-cluster; bypass any configured HTTP(S) proxies
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        # Identity labels for Chrome Trace representation
        hostname = socket.gethostname()
        process_name = process_name or str(os.getpid())
        self.pid = f"{process_name}@{hostname}({self.detect_node_ip()})"
        self.tid = (
            thread_name if thread_name is not None else str(threading.get_ident())
        )

        # Synchronization state
        self.offset = 0
        self.last_sync_time = 0.0
        self.sync_lock = threading.Lock()

        # Connection and Recovery state
        self.is_connected = True
        self.backoff_delay = 1.0
        self.max_backoff_delay = 60.0
        self.last_health_check = 0.0
        self.connection_lock = threading.Lock()

        # Buffering state
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.default_buffer_limit = 1000
        self.max_buffer_limit = 10000
        self.buffer_limit = self.default_buffer_limit

        # Run initial time synchronization
        self.sync_clock()

        # Background thread control
        self.running = True
        self.bg_thread = threading.Thread(target=self._background_loop, daemon=True)
        self.bg_thread.start()

        # Emit initial metadata events to label process/thread in the trace viewer
        self.emit_metadata("process_name", {"name": self.pid})
        self.emit_metadata("thread_name", {"name": self.tid})

        # Register exit handler for clean final flush
        atexit.register(self.shutdown)

    @staticmethod
    def logger():
        """The logger of the current worker process (lazy to avoid a circular import)."""
        from ..worker import Worker

        return Worker.logger

    @staticmethod
    def detect_node_ip() -> str:
        """Detect this node's IP address, preferring Ray's view of it."""
        try:
            import ray

            return ray.util.get_node_ip_address()
        except Exception:
            return socket.gethostbyname(socket.gethostname())

    @classmethod
    def init(
        cls,
        server_ip: Optional[str] = None,
        port: int = 8888,
        process_name: Optional[str] = None,
        thread_name: Optional[str] = None,
    ):
        """Initialize the global tracer; server_ip defaults to the local node IP."""
        with cls._instance_lock:
            try:
                cls._instance = cls(server_ip, port, process_name, thread_name)
            except Exception as e:
                cls.logger().error(f"Failed to initialize tracer client: {e}")
                cls._instance = None

    @classmethod
    def get(cls) -> Optional["DistTracer"]:
        """Retrieve the global tracer instance, or None if tracing is disabled."""
        with cls._instance_lock:
            return cls._instance

    @classmethod
    def reset(cls):
        """Shut down and clear the global tracer instance."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None

    @classmethod
    def trace_begin(cls, name: str, cat: str = "default", args: Optional[dict] = None):
        """Log the beginning of a duration event (Chrome Trace ph: B); no-op if tracing is disabled."""
        tracer = cls.get()
        if tracer is not None:
            tracer.log_event(name=name, cat=cat, ph="B", args=args)

    @classmethod
    def trace_end(cls, name: str, cat: str = "default"):
        """Log the end of a duration event (Chrome Trace ph: E); no-op if tracing is disabled."""
        tracer = cls.get()
        if tracer is not None:
            tracer.log_event(name=name, cat=cat, ph="E")

    @classmethod
    @contextmanager
    def trace_span(cls, name: str, cat: str = "default", args: Optional[dict] = None):
        """Context manager to trace execution of a code block."""
        cls.trace_begin(name, cat=cat, args=args)
        try:
            yield
        finally:
            cls.trace_end(name, cat=cat)

    @classmethod
    def trace_func(cls, func_or_cat=None, cat: str = "default"):
        """Decorator to trace functions (supports @trace_func, @trace_func(), and @trace_func(cat="..."))."""
        category = func_or_cat if isinstance(func_or_cat, str) else cat

        def decorator(func):
            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with cls.trace_span(func.__name__, cat=category):
                        return await func(*args, **kwargs)

                return async_wrapper

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with cls.trace_span(func.__name__, cat=category):
                    return func(*args, **kwargs)

            return sync_wrapper

        if callable(func_or_cat):
            return decorator(func_or_cat)
        return decorator

    def check_health(self) -> bool:
        """Send a GET request to /health to check if the server is online."""
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with self._opener.open(req, timeout=1.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("status") == "ok"
        except Exception:
            return False

    def sync_clock(self):
        """Synchronize time with the server using Cristian's algorithm over HTTP GET."""
        self.logger().info(
            f"Synchronizing clock with trace server at {self.server_url}"
        )
        best_offset = 0
        min_rtt = float("inf")
        successful_rounds = 0

        # Perform 5 round-trip measurements
        for i in range(5):
            try:
                t0 = time.time_ns() // 1000
                req = urllib.request.Request(f"{self.server_url}/sync", method="GET")
                with self._opener.open(req, timeout=2.0) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    t_server = data["server_time_us"]
                t1 = time.time_ns() // 1000

                rtt = t1 - t0
                offset = t_server - (t0 + rtt // 2)

                if rtt < min_rtt:
                    min_rtt = rtt
                    best_offset = offset
                successful_rounds += 1
            except Exception as e:
                # Log warning but don't fail completely to keep system robust
                self.logger().warning(f"Clock sync round {i} failed: {e}")
                time.sleep(0.05)

        if successful_rounds > 0:
            with self.sync_lock:
                self.offset = best_offset
                self.last_sync_time = time.time()
            self.logger().info(
                f"Clock synchronized. Offset: {best_offset} us, Min RTT: {min_rtt} us"
            )
        else:
            self.logger().error(
                "Could not sync clock with trace server. Defaulting to 0 offset."
            )

    def _now_us(self) -> int:
        """Return the current local time adjusted to the server epoch in microseconds."""
        with self.sync_lock:
            current_offset = self.offset
        return (time.time_ns() // 1000) + current_offset

    def log_event(
        self,
        name: str,
        cat: str = "default",
        ph: str = "X",
        ts: Optional[int] = None,
        dur: Optional[int] = None,
        args: Optional[dict] = None,
    ):
        """Append a trace event to the buffer thread-safely."""
        if ts is None:
            ts = self._now_us()

        event = {
            "name": name,
            "cat": cat,
            "ph": ph,
            "ts": ts,
            "pid": self.pid,
            "tid": self.tid,
        }
        if dur is not None:
            event["dur"] = dur
        if args is not None:
            event["args"] = args

        with self.buffer_lock:
            self.buffer.append(event)
            buffer_len = len(self.buffer)
            current_limit = self.buffer_limit

        # Trigger immediate flush if buffer limit is reached
        if buffer_len >= current_limit:
            threading.Thread(target=self.flush, daemon=True).start()

    def emit_metadata(self, name: str, args: dict):
        """Log a Chrome Trace metadata event (ph: M) to label processes/threads."""
        event = {
            "name": name,
            "ph": "M",
            "ts": self._now_us(),
            "pid": self.pid,
            "tid": self.tid,
            "args": args,
        }
        with self.buffer_lock:
            self.buffer.append(event)

    def flush(self):
        """Flush buffered events to the HTTP trace server."""
        with self.buffer_lock:
            if not self.buffer:
                return

            # If currently disconnected and we haven't reached the expanded buffer limit,
            # avoid attempting to flush to prevent spamming requests.
            if not self.is_connected and len(self.buffer) < self.buffer_limit:
                return

            events_to_send = self.buffer
            self.buffer = []

        try:
            data = json.dumps(events_to_send).encode("utf-8")
            req = urllib.request.Request(
                f"{self.server_url}/trace",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self._opener.open(req, timeout=5.0) as response:
                response.read()

            # Reset connection state on successful flush
            with self.connection_lock:
                if not self.is_connected:
                    self.logger().info(
                        "Reconnected to trace server on successful flush."
                    )
                    self.is_connected = True
                    self.buffer_limit = self.default_buffer_limit
                    self.backoff_delay = 1.0
        except Exception as e:
            # Handle failure
            with self.connection_lock:
                if self.is_connected:
                    self.logger().warning(
                        f"Disconnected from trace server: {e}. "
                        f"Entering backoff recovery (max buffer size expanded to {self.max_buffer_limit})."
                    )
                    self.is_connected = False
                    self.buffer_limit = self.max_buffer_limit

            # Restore events to buffer
            with self.buffer_lock:
                self.buffer = events_to_send + self.buffer

    def _background_loop(self):
        """Loop running periodically with jitter to handle flushes, health checks, and daily sync."""
        base_interval = 1.0
        while self.running:
            try:
                # Sleep with +/- 20% jitter to prevent thundering herd
                time.sleep(base_interval * random.uniform(0.8, 1.2))

                # Check connection status and handle backoff retries
                with self.connection_lock:
                    is_connected = self.is_connected
                    backoff = self.backoff_delay

                if not is_connected:
                    now = time.time()
                    if now - self.last_health_check >= backoff:
                        self.last_health_check = now
                        if self.check_health():
                            self.logger().info(
                                "Trace server healthcheck succeeded. Restoring connection."
                            )
                            with self.connection_lock:
                                self.is_connected = True
                                self.buffer_limit = self.default_buffer_limit
                                self.backoff_delay = 1.0
                            # Flush immediately upon reconnecting
                            self.flush()
                        else:
                            # Increase backoff delay exponentially up to max limit
                            new_backoff = min(backoff * 2, self.max_backoff_delay)
                            with self.connection_lock:
                                self.backoff_delay = new_backoff
                            self.logger().debug(
                                f"Trace server healthcheck failed. Next check in {new_backoff}s."
                            )
                else:
                    # Flush normal buffer
                    self.flush()

                # Daily Re-synchronization (every 86400 seconds)
                time_since_sync = time.time() - self.last_sync_time
                if time_since_sync >= 86400.0:
                    self.sync_clock()
            except Exception as e:
                self.logger().error(f"Error in tracer background loop: {e}")

    def shutdown(self):
        """Gracefully shut down the background thread and perform a final synchronous flush."""
        if self.running:
            self.running = False
            # Final synchronous flush of any remaining events
            self.flush()
