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
import functools
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Optional

from .manager import Manager


class Tracer(Manager):
    """A global manager that collects Chrome Trace events and writes them to a JSONL file.

    Like the other scheduler managers, the tracer is a single Ray actor pinned to node
    rank 0 and reached from any process via :meth:`get_proxy`. The cluster launches it at
    startup when ``cluster.tracer.enable`` is set.

    Every worker and the driver emit events directly through the class methods
    :meth:`trace_begin` / :meth:`trace_end` / :meth:`trace_span` / :meth:`trace_func`,
    which forward each event to the manager over ``ManagerProxy`` — there is no
    per-process client, buffer, or background thread. All emit APIs are no-ops until the
    tracer manager has been launched, so they are safe to leave in untraced runs.
    """

    MANAGER_NAME = "Tracer"

    # Process-local caches for the emit path (never touched on the manager actor).
    _unavailable: bool = False
    _labeled: bool = False

    # =============================== Manager (server) side ===============================

    def __init__(self, output_file: str):
        """Open the JSONL output file for the collected trace events."""
        self._output_file = os.path.abspath(output_file)
        dir_name = os.path.dirname(self._output_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self._file = open(self._output_file, "a")
        self._file_lock = threading.Lock()

    def record(self, event: dict) -> None:
        """Append a single trace event to the JSONL file."""
        with self._file_lock:
            self._file.write(json.dumps(event) + "\n")
            self._file.flush()

    def finalize(self) -> str:
        """Flush and close the output file, returning its path."""
        with self._file_lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()
        return self._output_file

    # ============================= Client-facing emit side =============================

    @classmethod
    def _get(cls):
        """Return the tracer proxy, or None if the tracer manager was never launched."""
        if cls._unavailable:
            return None
        try:
            return cls.get_proxy(no_wait=True)
        except Exception:
            cls._unavailable = True
            return None

    @staticmethod
    def _pid() -> str:
        """Label of the current process: the worker name, or ``driver`` off-worker."""
        from ..worker import Worker

        worker = Worker.current_worker
        if worker is not None and worker._worker_name:
            return worker._worker_name
        return "driver"

    @classmethod
    def _label_process(cls, proxy, pid: str):
        """Emit a one-time process_name metadata event to label this process."""
        if not cls._labeled:
            cls._labeled = True
            proxy.record(
                {
                    "name": "process_name",
                    "ph": "M",
                    "ts": time.time_ns() // 1000,
                    "pid": pid,
                    "tid": "main",
                    "args": {"name": pid},
                }
            )

    @classmethod
    def trace_begin(cls, name: str, cat: str = "default", args: Optional[dict] = None):
        """Log the beginning of a duration event (ph: B); no-op if tracing is disabled."""
        proxy = cls._get()
        if proxy is None:
            return
        pid = cls._pid()
        cls._label_process(proxy, pid)
        event = {
            "name": name,
            "cat": cat,
            "ph": "B",
            "ts": time.time_ns() // 1000,
            "pid": pid,
            "tid": "main",
        }
        if args is not None:
            event["args"] = args
        proxy.record(event)

    @classmethod
    def trace_end(cls, name: str, cat: str = "default"):
        """Log the end of a duration event (ph: E); no-op if tracing is disabled."""
        proxy = cls._get()
        if proxy is None:
            return
        proxy.record(
            {
                "name": name,
                "cat": cat,
                "ph": "E",
                "ts": time.time_ns() // 1000,
                "pid": cls._pid(),
                "tid": "main",
            }
        )

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
