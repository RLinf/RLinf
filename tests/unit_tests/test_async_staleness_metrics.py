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

"""Unit tests for the weight-staleness metrics of the async agent runner.

Covers the pure metric function (values, self-consistency, boundaries),
version-field preservation in `merge_result_list` / `split_results`, version
collection in `MultiAgentLoopWorker.get_rollout_result`, the agent loop idle
statistics, and the actor-side record extraction. No Ray, sglang, or
Megatron is launched.
"""

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.utils.metric_utils import (
    VersionedTrajectoryRecord,
    compute_staleness_metrics,
)


def _record(
    group_index: int,
    earliest_version: int,
    completion_version: int,
    generated_tokens: int,
    call_count: int = 1,
    cross_version_calls: int = 0,
    reward: float = 0.0,
    token_weighted: int | None = None,
) -> VersionedTrajectoryRecord:
    if token_weighted is None:
        token_weighted = earliest_version * generated_tokens
    return VersionedTrajectoryRecord(
        group_index=group_index,
        earliest_version=earliest_version,
        completion_version=completion_version,
        generated_tokens=generated_tokens,
        call_count=call_count,
        cross_version_calls=cross_version_calls,
        reward=reward,
        token_weighted_version_start_sum=token_weighted,
    )


class TestComputeStalenessMetrics:
    """The pure metric function over per-trajectory records."""

    def test_metric_values_on_a_known_step(self):
        # Two groups of two trajectories each. Group 0 started on version 8
        # and finished on version 9; group 1 started and finished on 9.
        records = [
            _record(0, 8, 9, 100, 2, 1, reward=1.0, token_weighted=800),
            _record(0, 8, 8, 100, 1, 0, reward=1.0, token_weighted=800),
            _record(1, 9, 9, 200, 1, 0, reward=0.0, token_weighted=1800),
            _record(1, 9, 9, 300, 1, 0, reward=0.0, token_weighted=2700),
        ]
        metrics = compute_staleness_metrics(records, version=10)

        assert metrics["data_staleness_2/ratio"] == 0.5
        assert metrics["data_staleness_1/ratio"] == 0.5
        # Tokens: staleness 2 carries group 0's 200 tokens, staleness 1 the
        # remaining 500 of the 700 total.
        assert metrics["data_staleness_2/token_ratio"] == 200 / 700
        assert metrics["data_staleness_1/token_ratio"] == 500 / 700
        assert metrics["data_staleness/mean"] == 1.5
        assert metrics["data_staleness/max"] == 2
        assert metrics["data_staleness/token_mean"] == pytest.approx(10 - 6100 / 700)
        # Group 0 spans versions 8..9, group 1 spans nothing.
        assert metrics["version_span/mean"] == 0.5
        assert metrics["version_span/max"] == 1
        assert metrics["queue_steps/mean"] == 1.0
        assert metrics["queue_steps/max"] == 1
        # 1 cross-version call out of 5 total.
        assert metrics["cross_version_call_ratio"] == 1 / 5
        # Trajectory staleness by length quartile (all lengths distinct).
        assert metrics["data_staleness/length_q1"] == 2
        assert metrics["data_staleness/length_q2"] == 2
        assert metrics["data_staleness/length_q3"] == 1
        assert metrics["data_staleness/length_q4"] == 1
        # Group rewards: [1.0, 1.0] and [0.0, 0.0].
        assert metrics["equal_reward_group_ratio"] == 1.0
        assert metrics["zero_reward_group_ratio"] == 0.5

    def test_self_consistency(self):
        # staleness = version_span + queue_steps per group; ratio sets sum
        # to 1; token_ratio sets sum to 1.
        records = [
            _record(0, 7, 8, 40),
            _record(1, 8, 10, 60),
            _record(1, 9, 9, 10),
            _record(2, 6, 7, 25),
            _record(3, 7, 7, 75),
        ]
        metrics = compute_staleness_metrics(records, version=11)

        assert (
            metrics["data_staleness/mean"]
            == metrics["version_span/mean"] + metrics["queue_steps/mean"]
        )
        assert sum(
            v
            for k, v in metrics.items()
            if k.startswith("data_staleness_") and k.endswith("/ratio")
        ) == pytest.approx(1.0)
        assert sum(
            v
            for k, v in metrics.items()
            if k.startswith("data_staleness_") and k.endswith("/token_ratio")
        ) == pytest.approx(1.0)
        # queue_steps is version - completion, never negative.
        assert metrics["queue_steps/mean"] >= 0
        assert metrics["queue_steps/max"] >= 0

    def test_length_quartiles(self):
        # Four trajectories with increasing token counts and decreasing
        # staleness: each quartile holds exactly one trajectory.
        records = [
            _record(0, 10, 10, 10),
            _record(1, 11, 11, 20),
            _record(2, 12, 12, 30),
            _record(3, 13, 13, 40),
        ]
        metrics = compute_staleness_metrics(records, version=15)
        assert metrics["data_staleness/length_q1"] == 5
        assert metrics["data_staleness/length_q2"] == 4
        assert metrics["data_staleness/length_q3"] == 3
        assert metrics["data_staleness/length_q4"] == 2

    def test_length_quartiles_skipped_below_four_trajectories(self):
        records = [_record(0, 10, 10, 10) for _ in range(3)]
        metrics = compute_staleness_metrics(records, version=12)
        assert not any(k.startswith("data_staleness/length_q") for k in metrics)

    def test_empty_records_return_empty_dict(self):
        assert compute_staleness_metrics([], version=5) == {}

    def test_zero_generated_tokens_do_not_divide_by_zero(self):
        records = [_record(0, 10, 10, 0), _record(0, 10, 10, 0)]
        metrics = compute_staleness_metrics(records, version=12)
        assert metrics["data_staleness_2/ratio"] == 1.0
        # No tokens anywhere: no token-weighted metric is defined.
        assert "data_staleness/token_mean" not in metrics
        assert not any(k.endswith("token_ratio") for k in metrics)


def _dynamic_result(
    num_sequence: int = 2,
    group_size: int = 1,
    version_start: list[int] | None = None,
    version_end: list[int] | None = None,
    rewards: list[float] | None = None,
) -> DynamicRolloutResult:
    return DynamicRolloutResult(
        num_sequence=num_sequence,
        group_size=group_size,
        idx_to_traj=list(range(num_sequence)),
        input_ids=[[1, 2, 3] for _ in range(num_sequence)],
        prompt_lengths=[2] * num_sequence,
        response_lengths=[1] * num_sequence,
        is_end=[True] * num_sequence,
        rewards=[0.0] * num_sequence if rewards is None else rewards,
        version_start=version_start,
        version_end=version_end,
    )


class TestVersionFieldPreservation:
    """merge_result_list and split_results keep version fields aligned."""

    def test_merge_concatenates_versions(self):
        merged = DynamicRolloutResult.merge_result_list(
            [
                _dynamic_result(version_start=[1, 2], version_end=[1, 3]),
                _dynamic_result(version_start=[4], version_end=[5]),
            ]
        )
        assert merged.version_start == [1, 2, 4]
        assert merged.version_end == [1, 3, 5]

    def test_merge_without_versions_stays_none(self):
        merged = DynamicRolloutResult.merge_result_list(
            [_dynamic_result(), _dynamic_result()]
        )
        assert merged.version_start is None
        assert merged.version_end is None

    def test_merge_mixed_versions_asserts(self):
        with pytest.raises(AssertionError, match="mix"):
            DynamicRolloutResult.merge_result_list(
                [
                    _dynamic_result(version_start=[1], version_end=[1]),
                    _dynamic_result(),
                ]
            )

    def test_split_slices_versions(self):
        result = _dynamic_result(
            num_sequence=3, version_start=[1, 2, 3], version_end=[1, 2, 3]
        )
        splits = DynamicRolloutResult.split_results(result, [1, 2])
        assert [s.version_start for s in splits] == [[1], [2, 3]]
        assert [s.version_end for s in splits] == [[1], [2, 3]]

    def test_split_without_versions_stays_none(self):
        result = _dynamic_result(num_sequence=2)
        splits = DynamicRolloutResult.split_results(result, [1, 1])
        assert all(s.version_start is None for s in splits)
        assert all(s.version_end is None for s in splits)


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
    from rlinf.workers.agent.agent_loop import MultiAgentLoopWorker

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
        from rlinf.workers.agent.agent_loop import AgentLoopWorker

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

            async def _group(input_ids, answer, group_size, output_channel):
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


def _request(n_groups=1, group_size=2):
    """A RolloutRequest double: n_groups (input_ids, answer) pairs."""
    return SimpleNamespace(
        input_ids=[[1, 2] for _ in range(n_groups)],
        answers=[f"a{i}" for i in range(n_groups)],
        n=group_size,
    )


class TestExtractVersionedTrajectoryRecords:
    """The actor's record extraction from DynamicRolloutResult objects."""

    @staticmethod
    def _make_actor():
        # __new__ bypasses Megatron init; only the private extraction
        # method is exercised.
        from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

        return MAMegatronActor.__new__(MAMegatronActor)

    def test_extracts_per_trajectory_records(self):
        actor = self._make_actor()
        # One result = one group (result_index 0); its trajectories are
        # trajectory 0 (two turns) and trajectory 1 (one turn).
        result = DynamicRolloutResult(
            num_sequence=3,
            group_size=2,
            idx_to_traj=[0, 0, 1],
            input_ids=[[1] * 4, [1] * 4, [1] * 4],
            prompt_lengths=[1, 1, 1],
            response_lengths=[10, 30, 20],
            is_end=[True, True, True],
            rewards=[0.0, 1.0, 2.0],
            version_start=[5, 6, 7],
            version_end=[6, 6, 7],
        )
        records = actor._extract_versioned_trajectory_records([result])
        assert len(records) == 2
        first, second = records
        # Every trajectory of one result shares the result's group index.
        assert first.group_index == 0 and second.group_index == 0
        assert first.earliest_version == 5
        assert first.completion_version == 6
        assert first.generated_tokens == 40
        assert first.call_count == 2
        assert first.cross_version_calls == 1
        assert first.reward == 0.5
        assert first.token_weighted_version_start_sum == 5 * 10 + 6 * 30
        assert second.earliest_version == 7
        assert second.generated_tokens == 20
        assert second.reward == 2.0

    def test_returns_none_without_versions(self):
        actor = self._make_actor()
        result = _dynamic_result()
        assert actor._extract_versioned_trajectory_records([result]) is None

    def test_returns_none_if_any_result_lacks_versions(self):
        actor = self._make_actor()
        versioned = _dynamic_result(version_start=[1, 2], version_end=[1, 2])
        assert (
            actor._extract_versioned_trajectory_records([versioned, _dynamic_result()])
            is None
        )
