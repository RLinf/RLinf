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

"""Unit tests for the continuous agent loop and the async agent runner.

Covers `run_agentloop_rollout_continuously` (stop exit, error propagation,
in-flight cancellation, metric records on the metric channel) and the
runner's metric drain/aggregation. No Ray or sglang is launched.
"""

import asyncio
from types import SimpleNamespace
from unittest import mock

from omegaconf import OmegaConf

from rlinf.data.schema import DiscardedGroup
from rlinf.workers.agent.agent_loop import AgentLoopWorker, GroupDiscarded


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


def _request(n_groups=1, group_size=2, group_ids=None):
    """A RolloutRequest double: n_groups (input_ids, answer) pairs."""
    return SimpleNamespace(
        input_ids=[[1, 2] for _ in range(n_groups)],
        answers=[f"a{i}" for i in range(n_groups)],
        n=group_size,
        group_ids=group_ids,
    )


class TestAggregateAgentMetrics:
    """The runner's one-pass aggregation of per-group metric dicts."""

    @staticmethod
    def _make_runner():
        # Imported here, not at module top: the runner module pulls in the
        # megatron actor, which is not installed in every unit-test env.
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

            async def _group(input_ids, answer, group_size, output_channel, group_id):
                calls.append((answer, group_size))
                return {"__sum__/turns": 1}

            worker.run_agentloop_rollout_group = _group
            continuously = asyncio.create_task(
                worker.run_agentloop_rollout_continuously(
                    in_channel, out_channel, metric_channel
                )
            )
            in_channel.put(_request(n_groups=2, group_size=3))
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

            assert calls == [("a0", 3), ("a1", 3)]

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

            async def _group(input_ids, answer, group_size, output_channel, group_id):
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

            async def _group(input_ids, answer, group_size, output_channel, group_id):
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

            async def _group(input_ids, answer, group_size, output_channel, group_id):
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

            async def _group(input_ids, answer, group_size, output_channel, group_id):
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

            async def _group(input_ids, answer, group_size, output_channel, group_id):
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
        from rlinf.runners.async_agent_runner import AsyncAgentRunner

        runner = AsyncAgentRunner.__new__(AsyncAgentRunner)
        try:
            runner._sync_weights_for_instances(version=0, instance_ids={0})
            raise AssertionError("expected the sync to fail")
        except AssertionError as e:
            assert "subset" in str(e)
