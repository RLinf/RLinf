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
import socket
import tempfile
import threading
import time
import sys
import unittest
from unittest.mock import MagicMock, patch

# Inject mock torch and omegaconf to avoid installing heavy packages for testing tracing client/server
sys.modules["torch"] = MagicMock()
sys.modules["omegaconf"] = MagicMock()

from rlinf.utils.trace_server import start_server
from rlinf.utils.tracing import (
    DistTracer,
    get_tracer,
    init_tracer,
    trace_span,
    trace_func,
)


class TraceServerThread(threading.Thread):
    def __init__(self, trace_file):
        super().__init__()
        self.trace_file = trace_file
        
        # Bind socket to avoid TOCTOU
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        
        from rlinf.utils.trace_server import app
        import uvicorn
        config = uvicorn.Config(app, fd=self.sock.fileno(), log_level="error")
        self.server = uvicorn.Server(config)
        
    def run(self):
        import rlinf.utils.trace_server
        import asyncio
        rlinf.utils.trace_server.trace_file_path = self.trace_file
        asyncio.run(self.server.serve())
        
    def stop(self):
        self.server.should_exit = True
        self.join(timeout=2.0)
        self.sock.close()


class TestDistributedTracing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_file = os.path.join(self.temp_dir.name, "test_traces.jsonl")

        # Start the trace server in a controlled thread
        self.server_thread = TraceServerThread(self.trace_file)
        self.port = self.server_thread.port
        self.server_thread.start()
        
        # Give the server a moment to boot the async loop
        time.sleep(0.3)

    def tearDown(self):
        self.server_thread.stop()
        self.temp_dir.cleanup()

    def test_clock_synchronization_and_tracing(self):
        # 1. Initialize the global tracer
        init_tracer(
            server_ip="127.0.0.1",
            port=self.port,
            process_name="test_proc",
            thread_name="test_thread",
        )
        tracer = get_tracer()
        self.assertIsNotNone(tracer)

        # 2. Check if the offset was successfully synchronized
        self.assertTrue(tracer.last_sync_time > 0.0)

        # 3. Trace a block using trace_span
        with trace_span("operation_a", cat="math", args={"data_len": 10}):
            time.sleep(0.1)

        # 4. Trace using decorator
        @trace_func(cat="dec")
        def decorated_function():
            time.sleep(0.05)

        decorated_function()

        # 5. Flush tracer events and shutdown
        tracer.shutdown()

        # 6. Verify written JSONL events
        self.assertTrue(os.path.exists(self.trace_file))
        with open(self.trace_file, "r") as f:
            lines = f.readlines()

        events = [json.loads(line) for line in lines]

        # Verify initial metadata events
        proc_meta = [e for e in events if e.get("name") == "process_name"]
        thread_meta = [e for e in events if e.get("name") == "thread_name"]
        self.assertEqual(len(proc_meta), 1)
        self.assertEqual(len(thread_meta), 1)
        self.assertTrue(proc_meta[0]["args"]["name"].startswith("test_proc"))
        self.assertEqual(thread_meta[0]["args"]["name"], "test_thread")

        # Verify trace span event (ph: "X")
        span_event = [e for e in events if e.get("name") == "operation_a"]
        self.assertEqual(len(span_event), 1)
        self.assertEqual(span_event[0]["cat"], "math")
        self.assertEqual(span_event[0]["ph"], "X")
        self.assertEqual(span_event[0]["args"]["data_len"], 10)
        self.assertTrue(span_event[0]["dur"] >= 100000)  # at least 100ms in microseconds

        # Verify decorated function event
        dec_event = [e for e in events if e.get("name") == "decorated_function"]
        self.assertEqual(len(dec_event), 1)
        self.assertEqual(dec_event[0]["cat"], "dec")
        self.assertEqual(dec_event[0]["ph"], "X")

    def test_daily_resync_drift_correction(self):
        tracer = DistTracer(
            server_ip="127.0.0.1",
            port=self.port,
            process_name="test_resync",
        )
        self.assertIsNotNone(tracer)

        # Mock the clock sync method to detect calls
        with patch.object(tracer, "sync_clock", wraps=tracer.sync_clock) as mock_sync:
            # Force tracer's last sync time to be older than 24 hours (86400 seconds)
            tracer.last_sync_time = time.time() - 90000.0

            # Allow background loop to iterate
            time.sleep(2.5)

            # Assert that sync_clock was triggered at least once due to age check
            mock_sync.assert_called()

    def test_connection_loss_and_recovery(self):
        tracer = DistTracer(
            server_ip="127.0.0.1",
            port=self.port,
            process_name="test_recovery",
        )
        self.assertIsNotNone(tracer)
        self.assertTrue(tracer.is_connected)
        self.assertEqual(tracer.buffer_limit, 1000)

        # 1. Simulate server connection loss by pointing to an invalid port
        valid_url = tracer.server_url
        tracer.server_url = "http://127.0.0.1:54321"

        # Log an event and try to flush it
        tracer.log_event("failed_event", cat="test")
        tracer.flush()

        # Check that we detected the disconnect and expanded our buffer limit
        self.assertFalse(tracer.is_connected)
        self.assertEqual(tracer.buffer_limit, 10000)

        # Verify the event is still preserved in the buffer
        with tracer.buffer_lock:
            buffer_content = list(tracer.buffer)
        self.assertTrue(any(e["name"] == "failed_event" for e in buffer_content))

        # 2. Restore connection URL to simulate server coming back online
        tracer.server_url = valid_url

        # Wait for the background loop to run a health check, recover, and flush
        time.sleep(2.5)

        # Verify we recovered and flushed successfully
        self.assertTrue(tracer.is_connected)
        self.assertEqual(tracer.buffer_limit, 1000)

        with open(self.trace_file, "r") as f:
            lines = f.readlines()
        events = [json.loads(line) for line in lines]
        self.assertTrue(any(e.get("name") == "failed_event" for e in events))

        tracer.shutdown()


if __name__ == "__main__":
    unittest.main()
