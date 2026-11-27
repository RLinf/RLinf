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

"""Unit tests for the agent loop worker and the async agent runner.

Covers `run_agentloop_rollout_continuously` (stop exit, error propagation,
in-flight cancellation, metric records on the metric channel), the
multi-turn rollout's per-turn version collection, the loop's idle-time
accounting, and the runner's metric drain/aggregation. No Ray, sglang, or
Megatron is launched; the tests importing the runner or the Megatron actor
skip when those are absent.
"""

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest
from omegaconf import OmegaConf

from rlinf.data.schema import DiscardedGroup
from rlinf.workers.agent.agent_loop import (
    AgentLoopWorker,
    GroupDiscarded,
    MultiAgentLoopWorker,
)


class _QueueChannel:
    """Channel double whose get() waits when empty, like a real Channel."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()

    def put(self, item, key=None, async_op=False):
        self._queue.put_nowait(item)

        async def _wait():
            return None

        return mock.MagicMock(async_wait=_wait)

    def get(self, key=None, async_op=False):
        async def _wait():
            return await self._queue.get()

        return mock.MagicMock(async_wait=_wait)

    def get_nowait(self, key=None):
        return self._queue.get_nowait()


def _make_worker() -> AgentLoopWorker:
    """An AgentLoopWorker with the loop state but no Worker init."""
    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.session_server_client = None
    worker._agentloop_task = None
    worker._agent_metric_channel = None
    worker._producer_error = None
    worker._inflight_groups = set()
    worker.log_error = mock.MagicMock()
    return worker


def _request(n_groups=1, group_size=2, group_ids=None, task_names=None):
    """A RolloutRequest double: n_groups (input_ids, answer) pairs."""
    return SimpleNamespace(
        input_ids=[[1, 2] for _ in range(n_groups)],
        answers=[f"a{i}" for i in range(n_groups)],
        n=group_size,
        group_ids=group_ids,
        task_names=task_names,
        cohort_id=None,
        weight_version=None,
    )


def _make_group_worker(worker_cls=AgentLoopWorker):
    """A group-method worker: enough state to build one group result."""
    worker = worker_cls.__new__(worker_cls)
    worker.session_server_client = None
    worker.print_outputs = False
    worker.return_logprobs = False
    worker.cfg = OmegaConf.create(
        {
            "data": {
                "rollout_batch_size": 4,
                "shuffle": False,
                "num_workers": 0,
                "seed": 1,
                "max_prompt_length": 8,
            },
            "runner": {"seq_length": 16},
            "algorithm": {
                "group_size": 2,
                "max_num_gen_batches": 1,
            },
        }
    )
    return worker


async def _single_query_output(input_ids):
    """One finished single-turn trajectory for the base group method."""
    return SimpleNamespace(
        prompt_ids=[1, 2],
        prompt_text="p",
        response_ids=[3],
        response_text="r",
        response_mask=None,
        response_logprobs=None,
        trace_prints=[],
    )


class _NullContext:
    """A context manager standing in for the worker timer."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class TestGetAgentMetrics:
    """Draining the metric channel: only the None bucket is aggregated."""

    @staticmethod
    def _make_runner():
        pytest.importorskip("megatron.core")
        from rlinf.runners.async_agent_runner import AsyncAgentRunner

        runner = AsyncAgentRunner.__new__(AsyncAgentRunner)
        runner.agent_metric_channel = _QueueChannel()
        return runner

    def test_drains_and_aggregates(self):
        runner = self._make_runner()
        channel = runner.agent_metric_channel
        channel.put({"bucket": None, "metrics": {"__sum__/turns": 1}})
        channel.put({"bucket": None, "metrics": {"__sum__/turns": 2}})
        assert runner.get_agent_metrics() == {"turns": 3}

    def test_drain_empty_channel_returns_empty(self):
        runner = self._make_runner()
        assert runner.get_agent_metrics() == {}
        # Draining must clear the channel: the second call sees no records.
        runner.agent_metric_channel.put(
            {"bucket": None, "metrics": {"__sum__/turns": 1}}
        )
        assert runner.get_agent_metrics() == {"turns": 1}
        assert runner.get_agent_metrics() == {}

    def test_records_with_other_buckets_are_ignored(self):
        runner = self._make_runner()
        runner.agent_metric_channel.put({"bucket": 4, "metrics": {"__sum__/a": 1}})
        runner.agent_metric_channel.put({"bucket": None, "metrics": {"__sum__/a": 2}})
        assert runner.get_agent_metrics() == {"a": 2}


class TestRunAgentloopRolloutContinuously:
    """The continuous agent loop: stop, errors, in-flight cancellation."""

    def test_groups_run_and_metrics_reach_the_channel(self):
        """Groups run; each puts one {"bucket": None, "metrics": ...} record."""

        async def run():
            worker = _make_worker()
            in_channel = _QueueChannel()
            out_channel = _QueueChannel()
            metric_channel = _QueueChannel()
            calls = []

            async def _group(
                input_ids,
                answer,
                group_size,
                output_channel,
                group_id,
                task_name=None,
            ):
                calls.append((answer, group_size, task_name))
                return {"__sum__/turns": 1}

            worker.run_agentloop_rollout_group = _group
            continuously = asyncio.create_task(
                worker.run_agentloop_rollout_continuously(
                    in_channel, out_channel, metric_channel
                )
            )
            in_channel.put(
                _request(n_groups=2, group_size=3, task_names=["searchr1", "searchr1"])
            )
            while metric_channel._queue.qsize() < 2:
                await asyncio.sleep(0.01)
            # Drain the records; empty-metric groups put nothing.
            records = [metric_channel.get_nowait() for _ in range(2)]
            assert all(record["bucket"] is None for record in records)
            assert [record["metrics"] for record in records] == [
                {"__sum__/turns": 1},
                {"__sum__/turns": 1},
            ]
            worker.stop()
            await asyncio.wait_for(continuously, timeout=5)

            # Each group got its task name by index from the request.
            assert calls == [("a0", 3, "searchr1"), ("a1", 3, "searchr1")]

        asyncio.run(run())

    def test_stop_returns_the_loop_even_when_blocked_on_get(self):
        """stop() cancels the loop also while it waits on the channel get."""

        async def run():
            worker = _make_worker()
            in_channel = _QueueChannel()
            continuously = asyncio.create_task(
                worker.run_agentloop_rollout_continuously(
                    in_channel, _QueueChannel(), _QueueChannel()
                )
            )
            await asyncio.sleep(0.05)
            assert not continuously.done()
            worker.stop()
            await asyncio.wait_for(continuously, timeout=5)

        asyncio.run(run())

    def test_group_failure_is_reraised_by_the_loop(self):
        """A group failure logs and makes the loop raise the group's error."""

        async def run():
            worker = _make_worker()
            in_channel = _QueueChannel()
            boom = RuntimeError("group exploded")

            async def _group(
                input_ids,
                answer,
                group_size,
                output_channel,
                group_id,
                task_name=None,
            ):
                raise boom

            worker.run_agentloop_rollout_group = _group
            continuously = asyncio.create_task(
                worker.run_agentloop_rollout_continuously(
                    in_channel, _QueueChannel(), _QueueChannel()
                )
            )
            in_channel.put(_request())
            try:
                await asyncio.wait_for(continuously, timeout=5)
                raise AssertionError("expected the continuous loop to raise")
            except RuntimeError as e:
                assert e is boom
            worker.log_error.assert_called_once()

        asyncio.run(run())

    def test_inflight_groups_cancelled_on_stop(self):
        """stop cancels in-flight groups instead of leaving them running."""

        async def run():
            worker = _make_worker()
            in_channel = _QueueChannel()
            cancelled = asyncio.Event()

            async def _group(
                input_ids,
                answer,
                group_size,
                output_channel,
                group_id,
                task_name=None,
            ):
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                return {}

            worker.run_agentloop_rollout_group = _group
            continuously = asyncio.create_task(
                worker.run_agentloop_rollout_continuously(
                    in_channel, _QueueChannel(), _QueueChannel()
                )
            )
            in_channel.put(_request())
            await asyncio.sleep(0.05)
            assert worker.num_inflight_groups() == 1
            worker.stop()
            await asyncio.wait_for(continuously, timeout=5)
            await asyncio.wait_for(cancelled.wait(), timeout=5)

        asyncio.run(run())

    def test_empty_metrics_not_put(self):
        """A group returning None or {} puts no record on the channel."""

        async def run():
            worker = _make_worker()
            in_channel = _QueueChannel()
            metric_channel = _QueueChannel()

            async def _group(
                input_ids,
                answer,
                group_size,
                output_channel,
                group_id,
                task_name=None,
            ):
                return None if answer == "a0" else {}

            worker.run_agentloop_rollout_group = _group
            continuously = asyncio.create_task(
                worker.run_agentloop_rollout_continuously(
                    in_channel, _QueueChannel(), metric_channel
                )
            )
            in_channel.put(_request(n_groups=2))
            await asyncio.sleep(0.05)
            assert metric_channel._queue.empty()
            worker.stop()
            await asyncio.wait_for(continuously, timeout=5)

        asyncio.run(run())

    def test_group_ids_reach_the_group_tasks(self):
        """Each (input_ids, answer) pair gets its request's group id."""

        async def run():
            worker = _make_worker()
            in_channel = _QueueChannel()
            out_channel = _QueueChannel()
            seen = []

            async def _group(
                input_ids,
                answer,
                group_size,
                output_channel,
                group_id,
                task_name=None,
            ):
                seen.append((answer, group_id))
                return {}

            worker.run_agentloop_rollout_group = _group
            continuously = asyncio.create_task(
                worker.run_agentloop_rollout_continuously(
                    in_channel, out_channel, _QueueChannel()
                )
            )
            in_channel.put(_request(n_groups=2, group_ids=[7, 3]))
            while len(seen) < 2:
                await asyncio.sleep(0.01)
            worker.stop()
            await asyncio.wait_for(continuously, timeout=5)

            assert seen == [("a0", 7), ("a1", 3)]

        asyncio.run(run())

    def test_requests_without_group_ids_pass_none(self):
        """Synchronous-style requests carry no group ids; groups get None."""

        async def run():
            worker = _make_worker()
            in_channel = _QueueChannel()
            seen = []

            async def _group(
                input_ids,
                answer,
                group_size,
                output_channel,
                group_id,
                task_name=None,
            ):
                seen.append(group_id)
                return {}

            worker.run_agentloop_rollout_group = _group
            continuously = asyncio.create_task(
                worker.run_agentloop_rollout_continuously(
                    in_channel, _QueueChannel(), _QueueChannel()
                )
            )
            in_channel.put(_request(n_groups=2))
            while len(seen) < 2:
                await asyncio.sleep(0.01)
            worker.stop()
            await asyncio.wait_for(continuously, timeout=5)

            assert seen == [None, None]

        asyncio.run(run())


class TestGroupDiscarded:
    """GroupDiscarded turns into a discard record; other errors stay fatal."""

    @staticmethod
    def _make_group_worker():
        from rlinf.workers.agent.agent_loop import AgentLoopWorker

        worker = AgentLoopWorker.__new__(AgentLoopWorker)
        worker.session_server_client = None
        worker.print_outputs = False
        worker.return_logprobs = False
        worker.cfg = OmegaConf.create(
            {
                "data": {
                    "rollout_batch_size": 4,
                    "shuffle": False,
                    "num_workers": 0,
                    "seed": 1,
                    "max_prompt_length": 8,
                },
                "runner": {"seq_length": 16},
                "algorithm": {
                    "group_size": 2,
                    "max_num_gen_batches": 1,
                },
            }
        )
        return worker

    def test_discard_puts_a_record_and_the_group_ends_normally(self):
        """A GroupDiscarded with a group id yields a DiscardedGroup."""

        async def run():
            worker = self._make_group_worker()
            out_channel = _QueueChannel()

            async def _query(input_ids):
                raise GroupDiscarded("reward_rule_failed")

            worker.run_one_query = _query
            await worker.run_agentloop_rollout_group(
                input_ids=[1, 2],
                answer="a",
                group_size=2,
                output_channel=out_channel,
                group_id=9,
                task_name="searchr1",
            )
            record = out_channel.get_nowait()
            assert isinstance(record, DiscardedGroup)
            assert record.group_id == 9
            assert record.reason == "reward_rule_failed"

        asyncio.run(run())

    def test_discard_cancels_the_siblings(self):
        """A discarded trajectory cancels the group's remaining ones."""

        async def run():
            worker = self._make_group_worker()
            out_channel = _QueueChannel()
            cancelled = asyncio.Event()
            calls = 0

            async def _query(input_ids):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise GroupDiscarded("reward_rule_failed")
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            worker.run_one_query = _query
            await worker.run_agentloop_rollout_group(
                input_ids=[1, 2],
                answer="a",
                group_size=3,
                output_channel=out_channel,
                group_id=4,
            )
            record = out_channel.get_nowait()
            assert record.group_id == 4
            assert cancelled.is_set()

        asyncio.run(run())

    def test_discard_without_group_id_is_fatal(self):
        """Synchronous callers get the exception, not a record."""

        async def run():
            worker = self._make_group_worker()
            out_channel = _QueueChannel()

            async def _query(input_ids):
                raise GroupDiscarded("reward_rule_failed")

            worker.run_one_query = _query
            try:
                await worker.run_agentloop_rollout_group(
                    input_ids=[1, 2],
                    answer="a",
                    group_size=2,
                    output_channel=out_channel,
                )
                raise AssertionError("expected the exception to propagate")
            except GroupDiscarded as e:
                assert e.reason == "reward_rule_failed"
            assert out_channel._queue.empty()

        asyncio.run(run())

    def test_other_errors_are_fatal(self):
        """A plain failure of one query ends the group with that error."""

        async def run():
            worker = self._make_group_worker()
            out_channel = _QueueChannel()
            boom = RuntimeError("tool call failed")

            async def _query(input_ids):
                raise boom

            worker.run_one_query = _query
            try:
                await worker.run_agentloop_rollout_group(
                    input_ids=[1, 2],
                    answer="a",
                    group_size=2,
                    output_channel=out_channel,
                    group_id=4,
                )
                raise AssertionError("expected the exception to propagate")
            except RuntimeError as e:
                assert e is boom
            assert out_channel._queue.empty()

        asyncio.run(run())


class TestRunnerConstruction:
    """The runner's placement requirement at construction."""

    def test_collocated_placement_rejected(self):
        """A non-disaggregated placement fails construction."""
        pytest.importorskip("megatron.core")
        from omegaconf import DictConfig

        from rlinf.runners.async_agent_runner import AsyncAgentRunner

        placement = SimpleNamespace(is_disaggregated=False)
        cfg = DictConfig({"algorithm": {"recompute_logprobs": False}})
        try:
            AsyncAgentRunner(
                cfg,
                placement,
                train_dataset=None,
                val_dataset=None,
                rollout=None,
                inference=None,
                actor=None,
                reward=None,
                agent_loop=None,
                rollout_buffer=object(),
            )
            raise AssertionError("expected construction to fail")
        except AssertionError as e:
            assert "disaggregated" in str(e)


class TestSyncWeightsInstanceSubset:
    """Weight syncs currently cover every rollout instance."""

    def test_instance_subset_rejected(self):
        """Passing instance ids fails: the actor pushes to all instances."""
        pytest.importorskip("megatron.core")
        from rlinf.runners.async_agent_runner import AsyncAgentRunner

        runner = AsyncAgentRunner.__new__(AsyncAgentRunner)
        try:
            runner._sync_weights_for_instances(version=0, instance_ids={0})
            raise AssertionError("expected the sync to fail")
        except AssertionError as e:
            assert "subset" in str(e)


def _single_turn_output(
    version_start: int | None,
    version_end: int | None,
    not_training: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_ids=[1, 2],
        response_ids=[3],
        response_logprobs=[0.5],
        response_weight_versions=None,
        response_images=None,
        response_mask=None,
        is_end=True,
        reward_score=0.0,
        extra_fields={"not_training": not_training},
        version_start=version_start,
        version_end=version_end,
    )


def _make_multi_worker():
    worker = MultiAgentLoopWorker.__new__(MultiAgentLoopWorker)
    worker.session_server_client = None
    worker._next_rollout_id = 0
    worker._world_size = 1
    worker.print_outputs = False
    worker.return_logprobs = True
    return worker


class TestGetRolloutResultVersions:
    """MultiAgentLoopWorker.get_rollout_result collects per-turn versions."""

    @staticmethod
    def _get_rollout_result(turns_per_task):
        worker = _make_multi_worker()
        task_results = [
            SimpleNamespace(single_turn_outputs=turns, trace_prints=[])
            for turns in turns_per_task
        ]
        return worker.get_rollout_result(
            task_results, None, None, None, extra_fields_train={}
        )

    def test_versions_written_and_aligned(self):
        result = self._get_rollout_result(
            [
                [
                    _single_turn_output(5, 5),
                    _single_turn_output(5, 6),
                ],
                [_single_turn_output(6, 6)],
            ]
        )
        assert result.version_start == [5, 5, 6]
        assert result.version_end == [5, 6, 6]
        assert len(result.version_start) == result.num_sequence

    def test_not_training_turns_skipped_stay_aligned(self):
        result = self._get_rollout_result(
            [
                [
                    _single_turn_output(5, 5),
                    _single_turn_output(None, None, not_training=True),
                    _single_turn_output(5, 6),
                ],
            ]
        )
        assert result.version_start == [5, 5]
        assert result.version_end == [5, 6]

    def test_no_versions_yields_none(self):
        result = self._get_rollout_result(
            [[_single_turn_output(None, None), _single_turn_output(None, None)]]
        )
        assert result.version_start is None
        assert result.version_end is None

    def test_mixed_versions_assert(self):
        with pytest.raises(AssertionError, match="mix"):
            self._get_rollout_result(
                [[_single_turn_output(5, 5), _single_turn_output(None, None)]]
            )


class TestIdleStats:
    """The continuous agent loop's idle-time accounting."""

    @staticmethod
    def _make_worker():
        worker = AgentLoopWorker.__new__(AgentLoopWorker)
        worker.session_server_client = None
        worker._agentloop_task = None
        worker._agent_metric_channel = None
        worker._producer_error = None
        worker._inflight_groups = set()
        worker.log_error = mock.MagicMock()
        return worker

    def test_idle_window_and_reset(self):
        worker = self._make_worker()
        worker._idle_seconds = 0.0
        worker._idle_window_start = 100.0
        worker._idle_since = 100.0

        with mock.patch(
            "rlinf.workers.agent.agent_loop.time.monotonic",
            side_effect=[104.0, 106.0],
        ):
            # pop_idle_stats reads the clock twice: once for the window end,
            # once as the new window start.
            idle, window = worker.pop_idle_stats()
            assert idle == 4.0
            assert window == 4.0
            # A second pop on an ongoing idle segment counts up to the call.
            idle, window = worker.pop_idle_stats()
            assert idle == 2.0
            assert window == 2.0

    def test_busy_time_not_counted_as_idle(self):
        worker = self._make_worker()
        worker._idle_seconds = 0.0
        worker._idle_window_start = 100.0
        worker._idle_since = None

        with mock.patch(
            "rlinf.workers.agent.agent_loop.time.monotonic", return_value=105.0
        ):
            idle, window = worker.pop_idle_stats()
            assert idle == 0.0
            assert window == 5.0

    def test_transitions_accumulate_idle_between_groups(self):
        async def run():
            worker = self._make_worker()
            in_channel = _QueueChannel()
            started = asyncio.Event()
            release = asyncio.Event()

            async def _group(
                input_ids,
                answer,
                group_size,
                output_channel,
                group_id,
                task_name=None,
            ):
                started.set()
                await release.wait()
                return {}

            worker.run_agentloop_rollout_group = _group
            continuously = asyncio.create_task(
                worker.run_agentloop_rollout_continuously(
                    in_channel, _QueueChannel(), _QueueChannel()
                )
            )
            # The loop is idle from the method start until the request
            # arrives and the group task is registered.
            in_channel.put(_request())
            await started.wait()
            with mock.patch(
                "rlinf.workers.agent.agent_loop.time.monotonic", return_value=100.0
            ):
                idle, _ = worker.pop_idle_stats()
            assert idle > 0.0
            release.set()
            await asyncio.wait_for(continuously, timeout=5)

        asyncio.run(run())


class TestGroupIdPassthrough:
    """The group id stamped by the agent loop group methods."""

    def test_group_id_written_onto_the_result(self):
        """run_agentloop_rollout_group stamps group_id on the result."""

        async def run():
            worker = _make_group_worker()
            out_channel = _QueueChannel()
            worker.run_one_query = _single_query_output
            await worker.run_agentloop_rollout_group(
                input_ids=[1, 2],
                answer="a",
                group_size=2,
                output_channel=out_channel,
                group_id=9,
            )
            result = out_channel.get_nowait()
            assert result.group_id == 9

        asyncio.run(run())

    def test_results_without_group_id_stay_none(self):
        """The synchronous path leaves group_id at its None default."""

        async def run():
            worker = _make_group_worker()
            out_channel = _QueueChannel()
            worker.run_one_query = _single_query_output
            await worker.run_agentloop_rollout_group(
                input_ids=[1, 2],
                answer="a",
                group_size=2,
                output_channel=out_channel,
            )
            result = out_channel.get_nowait()
            assert result.group_id is None

        asyncio.run(run())


class TestTaskNamePassthrough:
    """The task name stamped by the agent loop group methods."""

    def test_task_name_written_onto_the_result(self):
        """run_agentloop_rollout_group stamps task_name on the result."""

        async def run():
            worker = _make_group_worker()
            out_channel = _QueueChannel()
            worker.run_one_query = _single_query_output
            await worker.run_agentloop_rollout_group(
                input_ids=[1, 2],
                answer="a",
                group_size=2,
                output_channel=out_channel,
                group_id=9,
                task_name="searchr1",
            )
            result = out_channel.get_nowait()
            assert result.task_name == "searchr1"

        asyncio.run(run())

    def test_synchronous_path_leaves_task_name_none(self):
        """Without a group id the task name stays at its None default."""

        async def run():
            worker = _make_group_worker()
            out_channel = _QueueChannel()
            worker.run_one_query = _single_query_output
            await worker.run_agentloop_rollout_group(
                input_ids=[1, 2],
                answer="a",
                group_size=2,
                output_channel=out_channel,
            )
            result = out_channel.get_nowait()
            assert result.task_name is None

        asyncio.run(run())

    def test_multi_group_method_writes_task_name(self):
        """MultiAgentLoopWorker.run_agentloop_rollout_group stamps it too."""

        async def run():
            worker = _make_group_worker(MultiAgentLoopWorker)
            worker._next_rollout_id = 0
            worker._world_size = 1
            worker.return_logprobs = True
            worker.extra_keys_turn = None
            worker.extra_keys_traj = None
            worker.is_eval = False
            out_channel = _QueueChannel()

            async def _query(input_ids, answer=None):
                return SimpleNamespace(
                    single_turn_outputs=[_single_turn_output(None, None)],
                    trace_prints=[],
                    extra_fields={},
                )

            worker.run_one_query = _query
            await worker.run_agentloop_rollout_group(
                input_ids=[1, 2],
                answer="a",
                group_size=2,
                output_channel=out_channel,
                group_id=5,
                task_name="searchr1",
            )
            result = out_channel.get_nowait()
            assert result.task_name == "searchr1"

        asyncio.run(run())

    def test_synchronous_rollout_passes_no_task_name(self):
        """run_agentloop_rollout calls groups without a task name."""

        async def run():
            worker = _make_group_worker()
            out_channel = _QueueChannel()
            request = _request(n_groups=2)
            # The synchronous path reads the request from a plain get().
            in_channel = SimpleNamespace(get=lambda key=None, async_op=False: request)
            seen = []

            async def _group(
                input_ids,
                answer,
                group_size,
                output_channel,
                group_id=None,
                task_name=None,
            ):
                seen.append((group_id, task_name))
                return {}

            worker.run_agentloop_rollout_group = _group
            worker.worker_timer = lambda: _NullContext()
            await worker.run_agentloop_rollout(in_channel, out_channel)
            assert seen == [(None, None), (None, None)]

        asyncio.run(run())


def _candidate_worker(query):
    worker = _make_group_worker(MultiAgentLoopWorker)
    worker.cfg.algorithm.group_size = 1
    worker._rank, worker._world_size, worker._next_rollout_id = 0, 1, 0
    worker._candidate_cohort = None
    worker._candidate_started = {}
    worker._candidate_admissions = {}
    worker._generation_cohort_id = None
    worker.extra_keys_turn = worker.extra_keys_traj = None
    worker.run_one_query = query
    worker.worker_timer = _NullContext
    return worker


def _candidate_output(reward=0.0, version=2):
    from rlinf.workers.agent.agent_loop import AgentLoopOutput, MultiAgentLoopOutput

    return MultiAgentLoopOutput(
        single_turn_outputs=[
            AgentLoopOutput(
                prompt_ids=[1],
                response_ids=[2],
                response_mask=[1],
                is_end=True,
                reward_score=reward,
                version_start=version,
                version_end=version,
            )
        ]
    )


@pytest.mark.parametrize("is_end", [True, False])
def test_sync_candidates_emit_completed_failures_and_cancel_tail_after_cleanup(is_end):
    async def run():
        gates = [asyncio.Event() for _ in range(5)]
        cleaned = set()
        metadata = {}

        async def query(input_ids, answer):
            metadata[answer] = MultiAgentLoopWorker.rollout_candidate()
            try:
                await gates[answer].wait()
                output = _candidate_output(reward=1.0 if answer == 0 else 0.0)
                output.single_turn_outputs[-1].is_end = is_end
                return output
            finally:
                await asyncio.sleep(0)
                cleaned.add(answer)

        worker = _candidate_worker(query)
        channel = _QueueChannel()
        request = _request(n_groups=5, group_size=1, group_ids=list(range(5)))
        request.answers = list(range(5))
        request.cohort_id, request.weight_version = "cohort-a", 2
        rollout = asyncio.create_task(
            worker.run_agentloop_rollout(SimpleNamespace(get=lambda: request), channel)
        )
        events = []
        for candidate_id in (2, 0, 4):
            gates[candidate_id].set()
            event = await channel.get(async_op=True).async_wait()
            events.append(event)
        assert [event["candidate_id"] for event in events] == [2, 0, 4]
        assert events[0]["terminal_reward"] == 0.0
        assert all("result" not in event for event in events)
        assert all(event["status"] == "complete" for event in events)
        assert not rollout.done()
        receipt = await worker.cancel_rollout_candidates("cohort-a", [2, 0, 4])
        await asyncio.wait_for(rollout, timeout=3)
        assert cleaned == set(range(5))
        records = {row["candidate_id"]: row for row in receipt["candidates"]}
        assert {key for key, row in records.items() if row["kept"]} == {0, 2, 4}
        assert records[1]["status"] == records[3]["status"] == "cancelled"
        assert all(row["cleanup_complete"] for row in records.values())
        assert all(
            metadata[key]
            == {
                "cohort_id": "cohort-a",
                "candidate_id": key,
                "weight_version": 2,
            }
            for key in range(5)
        )
        assert worker._candidate_cohort is None
        assert worker._generation_cohort_id is None
        assert channel._queue.empty()
        learner = _QueueChannel()
        admitted = await worker.admit_rollout_candidates("cohort-a", [learner])
        assert admitted["admitted"] == 3
        admitted_results = [await learner.get().async_wait() for _ in range(3)]
        assert sorted(result.group_id for result in admitted_results) == [0, 2, 4]
        assert not worker._candidate_admissions

    asyncio.run(run())


def test_sync_sampling_counts_remain_worker_local_after_metric_gather():
    async def run():
        gates = [asyncio.Event() for _ in range(3)]

        async def query(input_ids, answer):
            await gates[answer].wait()
            return _candidate_output()

        worker = _candidate_worker(query)
        worker._world_size = 2
        worker._group_name = "AgentGroup"
        remote_messages = iter(
            [
                [{}],
                [
                    {
                        "__sum__/sampling/candidates": 3,
                        "__sum__/sampling/selected": 1,
                        "__sum__/sampling/dropped": 2,
                        "__sum__/sampling/cancelled": 2,
                    }
                ],
            ]
        )
        worker.recv = lambda *args, **kwargs: next(remote_messages)
        channel = _QueueChannel()
        request = _request(n_groups=3, group_size=1, group_ids=[0, 1, 2])
        request.answers = [0, 1, 2]
        request.cohort_id, request.weight_version = "counted-cohort", 2
        rollout = asyncio.create_task(
            worker.run_agentloop_rollout(SimpleNamespace(get=lambda: request), channel)
        )
        for candidate in (0, 1):
            gates[candidate].set()
            await channel.get(async_op=True).async_wait()
        receipt = await worker.cancel_rollout_candidates("counted-cohort", [0, 1, 3])
        metrics = await asyncio.wait_for(rollout, timeout=3)
        assert metrics["sampling/candidates"] == 6
        assert metrics["sampling/selected"] == 3
        assert metrics["sampling/dropped"] == 3
        assert metrics["sampling/cancelled"] == 3
        assert sum(row["kept"] for row in receipt["candidates"]) == 2
        learner = _QueueChannel()
        admitted = await worker.admit_rollout_candidates("counted-cohort", [learner])
        assert admitted["admitted"] == 2

    asyncio.run(run())


def test_sync_candidate_failure_remains_an_error_not_a_replacement_or_zero_reward():
    async def run():
        async def query(input_ids, answer):
            if answer == 0:
                raise RuntimeError("environment transport failed after retry")
            await asyncio.Event().wait()

        worker = _candidate_worker(query)
        request = _request(n_groups=2, group_size=1, group_ids=[0, 1])
        request.answers = [0, 1]
        request.cohort_id, request.weight_version = "failed-cohort", 2
        channel = _QueueChannel()
        rollout = asyncio.create_task(
            worker.run_agentloop_rollout(SimpleNamespace(get=lambda: request), channel)
        )
        event = await channel.get(async_op=True).async_wait()
        assert event["status"] == "error" and "result" not in event
        assert "after retry" in event["error"]
        receipt = await worker.cancel_rollout_candidates("failed-cohort", [])
        await asyncio.wait_for(rollout, timeout=3)
        assert receipt["candidates"][0]["status"] == "error"
        assert not any(row["kept"] for row in receipt["candidates"])
        await worker.admit_rollout_candidates("failed-cohort", None)

    asyncio.run(run())


def test_sync_cohort_cancel_waits_for_workers_not_started_yet():
    async def run():
        started = []

        async def query(input_ids, answer):
            started.append(answer)
            await asyncio.Event().wait()

        worker = _candidate_worker(query)
        cancellation = asyncio.create_task(
            worker.cancel_rollout_candidates("late-cohort", [])
        )
        await asyncio.sleep(0)
        assert not cancellation.done()
        request = _request(n_groups=2, group_size=1, group_ids=[0, 1])
        request.cohort_id, request.weight_version = "late-cohort", 2
        rollout = asyncio.create_task(
            worker.run_agentloop_rollout(
                SimpleNamespace(get=lambda: request), _QueueChannel()
            )
        )
        receipt = await asyncio.wait_for(cancellation, timeout=3)
        await asyncio.wait_for(rollout, timeout=3)
        assert len(receipt["candidates"]) == 2
        assert all(row["status"] == "cancelled" for row in receipt["candidates"])
        await worker.admit_rollout_candidates("late-cohort", None)

    asyncio.run(run())


def test_sync_candidate_mixed_weight_version_is_fatal():
    async def run():
        async def query(input_ids, answer):
            return _candidate_output(version=3)

        worker = _candidate_worker(query)
        request = _request(n_groups=1, group_size=1, group_ids=[0])
        request.cohort_id, request.weight_version = "wrong-policy", 2
        channel = _QueueChannel()
        rollout = asyncio.create_task(
            worker.run_agentloop_rollout(SimpleNamespace(get=lambda: request), channel)
        )
        event = await channel.get(async_op=True).async_wait()
        assert event["status"] == "error" and "weight versions" in event["error"]
        await worker.cancel_rollout_candidates("wrong-policy", [])
        await asyncio.wait_for(rollout, timeout=3)
        await worker.admit_rollout_candidates("wrong-policy", None)

    asyncio.run(run())


def test_sync_selection_waits_for_selected_publication_without_cancelling_it():
    async def run():
        acknowledged = asyncio.Event()

        class PublicationChannel(_QueueChannel):
            def put(self, item, key=None, async_op=False):
                self._queue.put_nowait(item)
                return SimpleNamespace(async_wait=acknowledged.wait)

        async def query(input_ids, answer):
            return _candidate_output()

        worker = _candidate_worker(query)
        channel = PublicationChannel()
        request = _request(n_groups=1, group_size=1, group_ids=[0])
        request.cohort_id, request.weight_version = "publishing", 2
        rollout = asyncio.create_task(
            worker.run_agentloop_rollout(SimpleNamespace(get=lambda: request), channel)
        )
        await channel.get().async_wait()
        cancellation = asyncio.create_task(
            worker.cancel_rollout_candidates("publishing", [0])
        )
        await asyncio.sleep(0)
        assert not cancellation.done()
        acknowledged.set()
        receipt = await asyncio.wait_for(cancellation, timeout=3)
        await asyncio.wait_for(rollout, timeout=3)
        assert receipt["candidates"][0]["kept"]
        assert receipt["candidates"][0]["status"] == "complete"
        await worker.admit_rollout_candidates("publishing", None)

    asyncio.run(run())


def test_cancelled_sync_generation_drains_its_reply_before_releasing_candidate():
    async def run():
        worker = _make_worker()
        worker.cfg = OmegaConf.create(
            {"data": {"rollout_batch_size": 1, "sampling_batch_size": 2}}
        )
        worker._generation_cohort_id = "cancel-request"
        worker.generate_input_channel = _QueueChannel()
        worker.generate_output_channel = _QueueChannel()
        generate = asyncio.create_task(worker.generate([1]))
        request = await worker.generate_input_channel.get().async_wait()
        assert request["cohort_id"] == "cancel-request"
        generate.cancel()
        await asyncio.sleep(0)
        assert not generate.done()
        worker.generate_output_channel.put({"output_ids": [2]})
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(generate, timeout=3)
        assert worker.generate_output_channel._queue.empty()

    asyncio.run(run())
