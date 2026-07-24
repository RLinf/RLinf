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

from rlinf.scheduler import Tracer


class FakeProxy:
    """Stand-in for the tracer ManagerProxy that records events in memory."""

    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class TestTracerManager:
    """Server-side test: the Tracer manager writes events to a JSONL file."""

    def test_record_and_finalize(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "trace.jsonl")
            # Construct the manager as a plain object (no Ray actor needed).
            tracer = Tracer(path)
            tracer.record({"name": "a", "ph": "B", "ts": 1, "pid": "p", "tid": "main"})
            tracer.record({"name": "a", "ph": "E", "ts": 2, "pid": "p", "tid": "main"})
            assert tracer.finalize() == os.path.abspath(path)

            with open(path) as f:
                events = [json.loads(line) for line in f]
            assert [e["ph"] for e in events] == ["B", "E"]
            assert all(e["name"] == "a" for e in events)


class TestTracerEmit:
    """Client-side test: the emit API forwards well-formed events, or no-ops."""

    def teardown_method(self):
        Tracer._unavailable = False
        Tracer._labeled = False

    def test_emit_forwards_events(self, monkeypatch):
        proxy = FakeProxy()
        monkeypatch.setattr(Tracer, "_get", classmethod(lambda cls: proxy))
        monkeypatch.setattr(Tracer, "_pid", staticmethod(lambda: "driver"))
        Tracer._labeled = False

        with Tracer.trace_span("step", cat="runner", args={"i": 0}):
            Tracer.trace_begin("inner", cat="actor")
            Tracer.trace_end("inner", cat="actor")

        @Tracer.trace_func(cat="dec")
        def fn():
            return 7

        assert fn() == 7

        # A one-time process_name metadata event labels the process.
        meta = [e for e in proxy.events if e["ph"] == "M"]
        assert len(meta) == 1
        assert meta[0]["args"]["name"] == "driver"

        # Duration events are well-formed and balanced per name.
        spans = [e for e in proxy.events if e["ph"] in ("B", "E")]
        assert all({"name", "cat", "ph", "ts", "pid", "tid"} <= e.keys() for e in spans)
        step_begin = next(e for e in spans if e["name"] == "step" and e["ph"] == "B")
        assert step_begin["cat"] == "runner" and step_begin["args"] == {"i": 0}
        for name in ("step", "inner", "fn"):
            phs = sorted(e["ph"] for e in spans if e["name"] == name)
            assert phs == ["B", "E"]

    def test_disabled_is_noop(self):
        # When the tracer manager is unavailable, every emit API is a no-op.
        Tracer._unavailable = True

        Tracer.trace_begin("noop")
        Tracer.trace_end("noop")
        with Tracer.trace_span("noop"):
            pass

        @Tracer.trace_func
        def fn():
            return 42

        assert fn() == 42
        assert Tracer._get() is None


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
