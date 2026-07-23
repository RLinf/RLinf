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
import tempfile
import time
import unittest
from unittest.mock import patch

from rlinf.scheduler.tracing import DistTracer, TraceServer


class TestDistributedTracing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_file = os.path.join(self.temp_dir.name, "test_traces.jsonl")

        # Start the trace server on an ephemeral port
        self.server = TraceServer(host="127.0.0.1", port=0, output_file=self.trace_file)
        self.server.start()
        self.port = self.server.port

    def tearDown(self):
        DistTracer.reset()
        self.server.stop()
        self.temp_dir.cleanup()

    def _read_events(self) -> list[dict]:
        with open(self.trace_file, "r") as f:
            return [json.loads(line) for line in f.readlines()]

    def test_clock_synchronization_and_tracing(self):
        # 1. Initialize the global tracer
        DistTracer.init(
            server_ip="127.0.0.1",
            port=self.port,
            process_name="test_proc",
            thread_name="test_thread",
        )
        tracer = DistTracer.get()
        self.assertIsNotNone(tracer)

        # 2. Check if the offset was successfully synchronized
        self.assertTrue(tracer.last_sync_time > 0.0)

        # 3. Trace a block using trace_span
        with DistTracer.trace_span("operation_a", cat="math", args={"data_len": 10}):
            time.sleep(0.1)

        # 4. Trace using explicit begin/end
        DistTracer.trace_begin("operation_b", cat="manual")
        DistTracer.trace_end("operation_b", cat="manual")

        # 5. Trace using decorator
        @DistTracer.trace_func(cat="dec")
        def decorated_function():
            time.sleep(0.05)

        decorated_function()

        # 6. Flush tracer events and shutdown
        tracer.shutdown()

        # 7. Verify written JSONL events
        self.assertTrue(os.path.exists(self.trace_file))
        events = self._read_events()

        # Verify initial metadata events
        proc_meta = [e for e in events if e.get("name") == "process_name"]
        thread_meta = [e for e in events if e.get("name") == "thread_name"]
        self.assertEqual(len(proc_meta), 1)
        self.assertEqual(len(thread_meta), 1)
        self.assertTrue(proc_meta[0]["args"]["name"].startswith("test_proc"))
        self.assertEqual(thread_meta[0]["args"]["name"], "test_thread")

        # Verify span events (ph: "B"/"E" pairs)
        span_begin = [
            e for e in events if e.get("name") == "operation_a" and e["ph"] == "B"
        ]
        span_end = [
            e for e in events if e.get("name") == "operation_a" and e["ph"] == "E"
        ]
        self.assertEqual(len(span_begin), 1)
        self.assertEqual(len(span_end), 1)
        self.assertEqual(span_begin[0]["cat"], "math")
        self.assertEqual(span_begin[0]["args"]["data_len"], 10)
        # The span should last at least 100ms in microseconds
        self.assertTrue(span_end[0]["ts"] - span_begin[0]["ts"] >= 100000)

        # Verify explicit begin/end events
        manual_phs = sorted(e["ph"] for e in events if e.get("name") == "operation_b")
        self.assertEqual(manual_phs, ["B", "E"])

        # Verify decorated function events
        dec_events = [e for e in events if e.get("name") == "decorated_function"]
        self.assertEqual(sorted(e["ph"] for e in dec_events), ["B", "E"])
        self.assertEqual(dec_events[0]["cat"], "dec")

    def test_disabled_tracer_is_noop(self):
        DistTracer.reset()
        self.assertIsNone(DistTracer.get())

        # All tracing entry points must be no-ops without a tracer
        DistTracer.trace_begin("noop")
        DistTracer.trace_end("noop")
        with DistTracer.trace_span("noop"):
            pass

        @DistTracer.trace_func
        def decorated_function():
            return 42

        self.assertEqual(decorated_function(), 42)

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

        tracer.shutdown()

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

        events = self._read_events()
        self.assertTrue(any(e.get("name") == "failed_event" for e in events))

        tracer.shutdown()


if __name__ == "__main__":
    unittest.main()
