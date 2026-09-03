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

"""Regression tests for asynchronous metric-table logging teardown."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.runners.offline_runner import OfflineRunner
from rlinf.utils.runner_utils import AsyncMetricTableLogger


@pytest.fixture
def table_logger():
    logger = AsyncMetricTableLogger(logging.getLogger("test-metric-table"))
    yield logger
    logger.shutdown()


def _assert_queue_drained(table_logger, timeout=2.0):
    """Every dequeued entry must be acknowledged, or ``join`` would hang."""
    joined = threading.Event()
    threading.Thread(
        target=lambda: (table_logger._queue.join(), joined.set()), daemon=True
    ).start()
    assert joined.wait(timeout), "queue bookkeeping did not reach zero"


def test_a_failing_entry_is_acknowledged_and_keeps_the_thread_alive(
    table_logger, caplog
):
    following_entry_ran = threading.Event()

    def fail():
        raise RuntimeError("render failed")

    with caplog.at_level(logging.ERROR):
        table_logger.submit(fail)
        table_logger.submit(following_entry_ran.set)
        assert following_entry_ran.wait(2.0)
        table_logger.shutdown()

    assert "Metric-table logging failed: render failed" in caplog.text
    _assert_queue_drained(table_logger)


def test_shutdown_drains_queued_entries_in_fifo_order(table_logger):
    first_started = threading.Event()
    release_first = threading.Event()
    order = []

    def first():
        first_started.set()
        release_first.wait()
        order.append(0)

    table_logger.submit(first)
    table_logger.submit(order.append, 1)
    table_logger.submit(order.append, 2)
    assert first_started.wait(2.0)

    done = threading.Event()
    threading.Thread(
        target=lambda: (table_logger.shutdown(), done.set()), daemon=True
    ).start()
    try:
        assert not done.wait(0.05), "shutdown returned before the queue was drained"
    finally:
        release_first.set()

    assert done.wait(2.0)
    assert order == [0, 1, 2]
    _assert_queue_drained(table_logger)


def test_shutdown_is_bounded_when_an_entry_hangs(caplog):
    table_logger = AsyncMetricTableLogger(
        logging.getLogger("test-metric-table"), shutdown_timeout_s=0.05
    )
    entry_started = threading.Event()
    release_entry = threading.Event()

    table_logger.submit(lambda: (entry_started.set(), release_entry.wait()))
    assert entry_started.wait(2.0)

    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING):
            table_logger.shutdown()
        assert time.monotonic() - started < 1.0
        assert "did not stop within 0.1 seconds" in caplog.text
    finally:
        release_entry.set()

    _assert_queue_drained(table_logger)


@pytest.mark.parametrize("runner_cls", [EmbodiedRunner, OfflineRunner])
def test_finish_run_flushes_tables_before_closing_backends(runner_cls):
    order = []
    runner = object.__new__(runner_cls)
    runner.metric_table_logger = SimpleNamespace(
        shutdown=lambda: order.append("shutdown")
    )
    runner.metric_logger = SimpleNamespace(finish=lambda: order.append("finish"))

    runner._finish_run()

    assert order == ["shutdown", "finish"]


@pytest.mark.parametrize(
    "run_method,update_weights,expected_stops",
    [
        pytest.param(
            AsyncEmbodiedRunner.run,
            lambda no_wait: None,
            ["stop-env", "stop-rollout", "stop-actor"],
            id="async-embodied",
        ),
        pytest.param(
            AsyncPPOEmbodiedRunner.run,
            lambda: None,
            ["stop-env", "stop-rollout"],
            id="async-ppo",
        ),
    ],
)
def test_async_runners_finish_logging_before_stopping_workers(
    run_method, update_weights, expected_stops
):
    events = []

    class _Handle:
        def wait(self):
            return None

    class _WorkerGroup:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: _Handle()

        def stop(self):
            events.append(f"stop-{self.name}")
            return _Handle()

    runner = SimpleNamespace(
        global_step=0,
        max_steps=0,
        actor=_WorkerGroup("actor"),
        rollout=_WorkerGroup("rollout"),
        env=_WorkerGroup("env"),
        reward=None,
        actor_channel=object(),
        rollout_channel=object(),
        env_channel=object(),
        reward_channel=None,
        env_metric_channel=object(),
        rollout_metric_channel=object(),
        sync_weight_no_wait=False,
        update_rollout_weights=update_weights,
        _finish_run=lambda: events.append("finish"),
    )

    run_method(runner)

    assert events[0] == "finish"
    assert events[1:] == expected_stops
