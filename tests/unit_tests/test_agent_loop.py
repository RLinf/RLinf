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
    )


def _make_group_worker(worker_cls=AgentLoopWorker):
    """A group-method worker: enough state to build one group result."""
    worker = worker_cls.__new__(worker_cls)
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


class TestAggregateAgentMetrics:
    """The runner's one-pass aggregation of per-group metric dicts."""

    @staticmethod
    def _make_runner():
        # The runner module pulls in the megatron actor, which is not
        # installed in every unit-test env.
        pytest.importorskip("megatron.core")
        from rlinf.runners.async_agent_runner import AsyncAgentRunner

        return AsyncAgentRunner.__new__(AsyncAgentRunner)

    def test_aggregates_each_stat_kind(self):
        runner = self._make_runner()
        aggregated = runner._aggregate_agent_metrics(
            [
                {"__sum__/a": 1, "__max__/b": 3, "__min__/c": 2, "__mean__/d": (1, 4)},
                {"__sum__/a": 2, "__max__/b": 5, "__min__/c": 1, "__mean__/d": (2, 4)},
            ]
        )
        assert aggregated == {"a": 3, "b": 5, "c": 1, "d": 3 / 8}

    def test_drops_keys_without_a_stat_prefix(self):
        runner = self._make_runner()
        assert runner._aggregate_agent_metrics([{"stray": 1}]) == {}

    def test_empty_list_returns_empty(self):
        runner = self._make_runner()
        assert runner._aggregate_agent_metrics([]) == {}


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
                    "staleness_threshold": 0,
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
        is_end=True,
        reward_score=0.0,
        extra_fields={"not_training": not_training},
        version_start=version_start,
        version_end=version_end,
    )


def _make_multi_worker():
    worker = MultiAgentLoopWorker.__new__(MultiAgentLoopWorker)
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
            worker.return_logprobs = True
            worker.extra_keys_turn = None
            worker.extra_keys_traj = None
            worker.is_eval = False
            out_channel = _QueueChannel()

            async def _query(input_dict, answer=None):
                return SimpleNamespace(
                    single_turn_outputs=[_single_turn_output(None, None)],
                    trace_prints=[],
                    extra_fields={},
                )

            worker.run_one_query = _query
            await worker.run_agentloop_rollout_group(
                input_dict={"input_ids": [1, 2]},
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
