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
import logging
import os
import random
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager

logger = logging.getLogger("rlinf.tracing")




class DistTracer:
    """Distributed tracing client that sends trace events to a central HTTP server."""

    def __init__(self, server_ip: str, port: int = 8888, process_name: str = None, thread_name: str = None):
        self.server_ip = server_ip
        self.port = port
        self.server_url = f"http://{server_ip}:{port}"
        
        import socket
        try:
            import ray
            ip_addr = ray.util.get_node_ip_address()
        except Exception:
            ip_addr = socket.gethostbyname(socket.gethostname())
        hostname = socket.gethostname()
        # Identity labels for Chrome Trace representation
        self.pid = f"{process_name}@{hostname}({ip_addr})" if process_name is not None else f"{os.getpid()}@{hostname}({ip_addr})"
        self.tid = thread_name if thread_name is not None else str(threading.get_ident())

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

    def check_health(self) -> bool:
        """Sends a GET request to /health to check if the server is online."""
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("status") == "ok"
        except Exception:
            return False

    def sync_clock(self):
        """Synchronize time with the server using Cristian's algorithm over HTTP GET."""
        logger.info(f"Synchronizing clock with trace server at {self.server_url}")
        best_offset = 0
        min_rtt = float("inf")
        successful_rounds = 0

        # Perform 5 round-trip measurements
        for i in range(5):
            try:
                t0 = time.time_ns() // 1000
                req = urllib.request.Request(f"{self.server_url}/sync", method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as response:
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
                logger.warning(f"Clock sync round {i} failed: {e}")
                time.sleep(0.05)

        if successful_rounds > 0:
            with self.sync_lock:
                self.offset = best_offset
                self.last_sync_time = time.time()
            logger.info(
                f"Clock synchronized. Offset: {best_offset} us, Min RTT: {min_rtt} us"
            )
        else:
            logger.error("Could not sync clock with trace server. Defaulting to 0 offset.")

    def log_event(self, name: str, cat: str = "default", ph: str = "X", ts: int = None, dur: int = None, args: dict = None):
        """Append a trace event to the buffer thread-safely."""
        if ts is None:
            # Adjust local time to UTC+0 synchronized server time
            with self.sync_lock:
                current_offset = self.offset
            ts = (time.time_ns() // 1000) + current_offset

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
        with self.sync_lock:
            current_offset = self.offset
        ts = (time.time_ns() // 1000) + current_offset
        event = {
            "name": name,
            "ph": "M",
            "ts": ts,
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
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                response.read()
            
            # Reset connection state on successful flush
            with self.connection_lock:
                if not self.is_connected:
                    logger.info("Reconnected to trace server on successful flush.")
                    self.is_connected = True
                    self.buffer_limit = self.default_buffer_limit
                    self.backoff_delay = 1.0
        except Exception as e:
            # Handle failure
            with self.connection_lock:
                if self.is_connected:
                    logger.warning(
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
                            logger.info("Trace server healthcheck succeeded. Restoring connection.")
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
                            logger.debug(f"Trace server healthcheck failed. Next check in {new_backoff}s.")
                else:
                    # Flush normal buffer
                    self.flush()

                # Daily Re-synchronization (every 86400 seconds)
                time_since_sync = time.time() - self.last_sync_time
                if time_since_sync >= 86400.0:
                    self.sync_clock()
            except Exception as e:
                logger.error(f"Error in tracer background loop: {e}")

    def shutdown(self):
        """Gracefully shut down the background thread and perform a final synchronous flush."""
        if self.running:
            self.running = False
            # Final synchronous flush of any remaining events
            self.flush()


# Global tracer client instance
_tracer = None
_tracer_lock = threading.Lock()


def init_tracer(server_ip: str, port: int = 8888, process_name: str = None, thread_name: str = None):
    """Initialize the global distributed tracer client."""
    global _tracer
    with _tracer_lock:
        if server_ip:
            try:
                _tracer = DistTracer(
                    server_ip=server_ip,
                    port=port,
                    process_name=process_name,
                    thread_name=thread_name
                )
            except Exception as e:
                logger.error(f"Failed to initialize tracer client: {e}")
                _tracer = None
        else:
            _tracer = None


def get_tracer():
    """Retrieve the global tracer instance, or None if disabled."""
    global _tracer
    with _tracer_lock:
        return _tracer


@contextmanager
def trace_span(name: str, cat: str = "default", args: dict = None):
    """Context manager to trace execution of a code block."""
    tracer = get_tracer()
    if tracer is None:
        yield
        return

    # Measure start time aligned to server epoch
    with tracer.sync_lock:
        offset = tracer.offset
    start_ts = (time.time_ns() // 1000) + offset

    try:
        yield
    finally:
        end_ts = (time.time_ns() // 1000) + offset
        dur = end_ts - start_ts
        tracer.log_event(name=name, cat=cat, ph="X", ts=start_ts, dur=dur, args=args)


def trace_func(func_or_cat=None, cat: str = "default"):
    """Decorator to trace functions (supports @trace_func, @trace_func(), and @trace_func(cat="..."))."""
    if callable(func_or_cat):
        func = func_or_cat
        category = cat
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with trace_span(func.__name__, cat=category):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with trace_span(func.__name__, cat=category):
                    return func(*args, **kwargs)
            return sync_wrapper

    category = func_or_cat if isinstance(func_or_cat, str) else cat

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with trace_span(func.__name__, cat=category):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with trace_span(func.__name__, cat=category):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator
