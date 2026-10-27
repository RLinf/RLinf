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

"""Unit tests for the rollout buffer worker of the async agent runner.

Covers the buffer's admission budget and release order, the discard
paths (equal-reward, staleness, task-declared, drop limit), the
checkpoint round-trip of its dataloader, the group-id passthrough on
DynamicRolloutResult, and the worker's resident coroutines over fake
channels. No Ray or sglang is launched.
"""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset

from rlinf.data.datasets.common.item import DatasetItem
from rlinf.data.schema import DiscardedGroup, GroupMetadata
from rlinf.scheduler import Worker
from rlinf.workers.rollout_buffer.rollout_buffer_worker import (
    RolloutBuffer,
    RolloutBufferWorker,
)


def _cfg(
    rollout_batch_size: int = 4,
    staleness_threshold: int = 0,
    num_samples: int = 64,
    shuffle: bool = False,
    reject_equal_reward: bool = False,
    max_actor_accept_staleness: int | None = None,
    max_dropped_groups_per_step: int = 1000,
):
    algorithm = {
        "staleness_threshold": staleness_threshold,
        "group_size": 2,
        "max_num_gen_batches": 1,
        "max_dropped_groups_per_step": max_dropped_groups_per_step,
    }
    if reject_equal_reward:
        algorithm["down_sampling"] = {
            "do_down_sampling": True,
            "down_sampling_config": {"reject_equal_reward": True},
        }
    if max_actor_accept_staleness is not None:
        algorithm["max_actor_accept_staleness"] = max_actor_accept_staleness
    return OmegaConf.create(
        {
            "data": {
                "rollout_batch_size": rollout_batch_size,
                "shuffle": shuffle,
                "num_workers": 0,
                "seed": 1,
                "max_prompt_length": 8,
            },
            "runner": {"seq_length": 16},
            "algorithm": algorithm,
        }
    )


class _ToyDataset(Dataset):
    """Minimal dataset yielding collate-compatible DatasetItems."""

    def __init__(self, num_samples: int = 64):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return DatasetItem(
            prompt=torch.tensor([idx, idx + 1, idx + 2], dtype=torch.long),
            length=3,
            answer=f"a{idx}",
            idx=idx,
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


def _result(group_id=None, earliest_version=None, rewards=None):
    """A single-group DynamicRolloutResult double.

    `group_metadata()` mirrors the real method's contract; other fields
    are absent because no buffer path reads them.
    """

    def group_metadata():
        return GroupMetadata(
            group_id=group_id,
            rewards=rewards,
            earliest_version=earliest_version,
            completion_version=earliest_version,
            num_gen_tokens=1,
            num_generate_calls=1,
        )

    return SimpleNamespace(group_id=group_id, group_metadata=group_metadata)


class _BufferFixture:
    """A RolloutBuffer without the dataloader (tests patch it out)."""

    @staticmethod
    def make(
        rollout_batch_size=4,
        staleness_threshold=0,
        loaded_versions=None,
        start_step=0,
    ):
        with mock.patch.object(RolloutBuffer, "_build_dataloader"):
            buffer = RolloutBuffer(_cfg(rollout_batch_size, staleness_threshold), None)
        buffer.dataloader = None
        if loaded_versions is not None:
            buffer.loaded_versions = dict(loaded_versions)
        buffer.start_step = start_step
        return buffer


def _make_worker(
    rollout_batch_size=2,
    staleness_threshold=0,
    num_samples=8,
    reject_equal_reward=False,
    max_actor_accept_staleness=None,
    max_dropped_groups_per_step=1000,
):
    """A RolloutBufferWorker with the loop state but no Worker init."""
    cfg = _cfg(
        rollout_batch_size,
        staleness_threshold,
        reject_equal_reward=reject_equal_reward,
        max_actor_accept_staleness=max_actor_accept_staleness,
        max_dropped_groups_per_step=max_dropped_groups_per_step,
    )
    worker = RolloutBufferWorker.__new__(RolloutBufferWorker)
    worker.cfg = cfg
    worker.buffer = RolloutBuffer(cfg, _ToyDataset(num_samples))
    # This double holds the parsed flag directly; parsing from cfg is
    # covered by test_reject_equal_flag_is_parsed_from_repo_shape.
    worker.reject_equal_reward = reject_equal_reward
    worker.max_actor_accept_staleness = max_actor_accept_staleness
    worker.max_dropped_groups_per_step = max_dropped_groups_per_step
    worker.log_warning = mock.MagicMock()
    worker.log_info = mock.MagicMock()
    worker._resident_task = None
    worker._data_iter = None
    worker._rollout_channel = None
    worker._pending_fetch = None
    return worker


class TestAdmissionBudget:
    """The admission budget of the rollout buffer."""

    def test_initial_budget_is_one_batch(self):
        """Before any sync, one batch of prompts may be issued."""
        buffer = _BufferFixture.make(rollout_batch_size=4)
        assert buffer._prompts_to_issue() == 4

    def test_issuing_consumes_the_budget(self):
        """Each issued group takes one slot of the budget."""
        buffer = _BufferFixture.make(rollout_batch_size=4)
        for _ in range(4):
            buffer.issue_group(0)
        assert buffer._prompts_to_issue() == 0

    def test_loaded_version_growth_adds_budget(self):
        """A new loaded version admits another batch."""
        buffer = _BufferFixture.make(rollout_batch_size=4, loaded_versions={0: 0, 1: 0})
        for _ in range(4):
            buffer.issue_group(0)
        assert buffer._prompts_to_issue() == 0
        buffer.set_loaded_versions({0: 1, 1: 1})
        assert buffer._prompts_to_issue() == 4

    def test_minimum_loaded_version_gates(self):
        """Only the minimum loaded version across instances counts."""
        buffer = _BufferFixture.make(rollout_batch_size=4, loaded_versions={0: 3, 1: 0})
        assert buffer._prompts_to_issue() == 4

    def test_budget_counts_from_the_resume_step(self):
        """After resuming at step 7 with version 7 loaded, one batch fits."""
        buffer = _BufferFixture.make(
            rollout_batch_size=4,
            loaded_versions={0: 7, 1: 7},
            start_step=7,
        )
        assert buffer._prompts_to_issue() == 4

    def test_staleness_threshold_widens_the_budget(self):
        """A threshold of 1 admits one extra batch."""
        buffer = _BufferFixture.make(
            rollout_batch_size=4, staleness_threshold=1, loaded_versions={0: 0}
        )
        assert buffer._prompts_to_issue() == 8

    def test_lowering_the_threshold_blocks_new_prompts_only(self):
        """Lowering the threshold stops new admissions; issued stays."""
        buffer = _BufferFixture.make(
            rollout_batch_size=4, staleness_threshold=1, loaded_versions={0: 0}
        )
        for _ in range(8):
            buffer.issue_group(0)
        buffer.set_staleness_threshold(0)
        # Over-issued relative to the new budget: nothing more may be
        # issued, but the issued groups are not revoked.
        assert buffer._prompts_to_issue() <= 0
        assert buffer.num_issued_groups == 8
        # The update wakes admission; it would see the smaller budget and
        # wait again.
        assert buffer._admission_event.is_set()

    def test_wake_without_a_free_slot_keeps_waiting(self):
        """A wake-up that frees no slot does not hand one out."""

        async def run():
            buffer = _BufferFixture.make(rollout_batch_size=2)
            for _ in range(2):
                buffer.issue_group(0)
            slot_taken = False

            async def issue_if_admitted():
                nonlocal slot_taken
                await buffer.wait_for_admission_slot()
                slot_taken = True

            waiter = asyncio.create_task(issue_if_admitted())
            await asyncio.sleep(0.05)
            # Wake admission without freeing a slot.
            buffer.set_staleness_threshold(0)
            await asyncio.sleep(0.05)
            assert not slot_taken
            assert not waiter.done()
            # Freeing a slot (new loaded version) lets it through.
            buffer.set_loaded_versions({0: 1, 1: 1})
            await asyncio.wait_for(waiter, timeout=5)
            assert slot_taken

        asyncio.run(run())


class TestRelease:
    """Release of buffered groups."""

    def test_release_takes_the_oldest_groups_in_order(self):
        """Released results come out in group-id order."""
        buffer = _BufferFixture.make(rollout_batch_size=3)
        for group_id in [5, 1, 9]:
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
        released = buffer.release_groups()
        assert [result.group_id for result in released] == [1, 5, 9]

    def test_release_removes_the_groups_from_the_buffer(self):
        """Released groups leave the buffer and the issue-version dict."""
        buffer = _BufferFixture.make(rollout_batch_size=2)
        for group_id in [1, 2, 3]:
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
            buffer.issue_versions[group_id] = 0
        buffer.release_groups()
        assert set(buffer.buffer) == {3}
        assert set(buffer.issue_versions) == {3}

    def test_can_release_requires_a_full_batch(self):
        """can_release flips only at rollout_batch_size groups."""
        buffer = _BufferFixture.make(rollout_batch_size=3)
        assert not buffer.can_release()
        buffer.record_result(_result(1), _result(1).group_metadata())
        buffer.record_result(_result(2), _result(2).group_metadata())
        assert not buffer.can_release()
        buffer.record_result(_result(3), _result(3).group_metadata())
        assert buffer.can_release()


class TestRecordResult:
    """Buffering of finished groups."""

    def test_results_without_group_id_raise(self):
        """Merged or split results (group_id None) fail the buffer."""
        buffer = _BufferFixture.make()
        with pytest.raises(ValueError, match="without a group id"):
            buffer.record_result(_result(None), _result(None).group_metadata())
        assert buffer.buffer == {}
        assert buffer.record_result(_result(3), _result(3).group_metadata()) == 3
        assert buffer.buffer[3][1].group_id == 3

    def test_inflight_counts_issued_minus_finished(self):
        """In-flight groups drop back to zero as results arrive."""
        buffer = _BufferFixture.make(rollout_batch_size=4)
        issued = [buffer.issue_group(0)[0] for _ in range(4)]
        assert buffer.inflight_groups() == 4
        for group_id in issued[:2]:
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
        assert buffer.inflight_groups() == 2
        for group_id in issued[2:]:
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
        assert buffer.inflight_groups() == 0


class TestStepMetrics:
    """Step-scoped metrics of the buffer."""

    def test_metrics_drain_and_reset(self):
        """Per-step counters reset; gauges read live values."""
        buffer = _BufferFixture.make(rollout_batch_size=4)
        for _ in range(2):
            buffer.issue_group(0)
        buffer.record_result(_result(0), _result(0).group_metadata())
        metrics = buffer.get_step_metrics()
        assert metrics["issued_groups"] == 2
        assert metrics["inflight_groups"] == 1
        assert metrics["size"] == 1
        metrics = buffer.get_step_metrics()
        assert metrics["issued_groups"] == 0
        assert metrics["inflight_groups"] == 1

    def test_batch_issue_version_metrics(self):
        """A released batch reports its distinct and oldest-issue share."""
        buffer = _BufferFixture.make(rollout_batch_size=3)
        for group_id, version in [(1, 0), (2, 0), (3, 1)]:
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
            buffer.issue_versions[group_id] = version
        buffer.release_groups()
        metrics = buffer.get_step_metrics()
        assert metrics["batch_issue_versions"] == 2
        assert metrics["batch_oldest_issue_version_ratio"] == 2 / 3


class TestAgentLoopPassthrough:
    """The group id stamped by the agent loop group methods."""

    @staticmethod
    def _multi_worker():
        from rlinf.workers.agent.agent_loop import MultiAgentLoopWorker

        worker = MultiAgentLoopWorker.__new__(MultiAgentLoopWorker)
        return worker

    def test_group_id_written_onto_the_result(self):
        """run_agentloop_rollout_group stamps group_id on the result."""
        from rlinf.workers.agent.agent_loop import AgentLoopWorker

        worker = AgentLoopWorker.__new__(AgentLoopWorker)
        out_channel = _QueueChannel()

        async def run():
            async def _query(input_ids):
                return SimpleNamespace(
                    prompt_ids=[1, 2],
                    prompt_text="p",
                    response_ids=[3],
                    response_text="r",
                    response_mask=None,
                    response_logprobs=None,
                    trace_prints=[],
                )

            worker.run_one_query = _query
            worker.print_outputs = False
            worker.return_logprobs = False
            worker.cfg = _cfg()
            await worker.run_agentloop_rollout_group(
                input_ids=[1, 2],
                answer="a",
                group_size=2,
                output_channel=out_channel,
                group_id=9,
            )

        asyncio.run(run())
        result = out_channel.get_nowait()
        assert result.group_id == 9

    def test_results_without_group_id_stay_none(self):
        """The synchronous path leaves group_id at its None default."""
        from rlinf.workers.agent.agent_loop import AgentLoopWorker

        worker = AgentLoopWorker.__new__(AgentLoopWorker)
        out_channel = _QueueChannel()

        async def run():
            async def _query(input_ids):
                return SimpleNamespace(
                    prompt_ids=[1, 2],
                    prompt_text="p",
                    response_ids=[3],
                    response_text="r",
                    response_mask=None,
                    response_logprobs=None,
                    trace_prints=[],
                )

            worker.run_one_query = _query
            worker.print_outputs = False
            worker.return_logprobs = False
            worker.cfg = _cfg()
            await worker.run_agentloop_rollout_group(
                input_ids=[1, 2],
                answer="a",
                group_size=2,
                output_channel=out_channel,
            )

        asyncio.run(run())
        result = out_channel.get_nowait()
        assert result.group_id is None


class TestCheckpoint:
    """Dataloader checkpoint round-trip of the buffer."""

    @staticmethod
    def _make_worker(cfg):
        worker = RolloutBufferWorker.__new__(RolloutBufferWorker)
        worker.buffer = RolloutBuffer(cfg, _ToyDataset(8))
        worker.log_warning = mock.MagicMock()
        worker.log_info = mock.MagicMock()
        worker._pending_fetch = None
        return worker

    def test_save_then_load_resumes_the_iteration_position(self):
        """After a reload, the dataloader continues after the saved batch."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = self._make_worker(_cfg(rollout_batch_size=4))

            data_iter = iter(worker.buffer.dataloader)
            first = next(data_iter)["idx"].tolist()
            asyncio.run(worker.save_checkpoint(tmp_dir))

            reloaded = self._make_worker(_cfg(rollout_batch_size=4))
            asyncio.run(reloaded.load_checkpoint(tmp_dir, start_step=1))
            assert reloaded.buffer.start_step == 1
            second = next(iter(reloaded.buffer.dataloader))["idx"].tolist()
            assert second[0] == first[0] + 4

    def test_data_pt_is_written_atomically(self):
        """save_checkpoint leaves no .tmp file behind."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = self._make_worker(_cfg(rollout_batch_size=4))
            asyncio.run(worker.save_checkpoint(tmp_dir))
            assert os.path.isfile(os.path.join(tmp_dir, "data.pt"))
            assert not os.path.exists(os.path.join(tmp_dir, "data.pt.tmp"))

    def test_missing_state_still_sets_the_start_step(self):
        """Without data.pt the dataset restarts; start_step is still set."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = self._make_worker(_cfg(rollout_batch_size=4))
            asyncio.run(worker.load_checkpoint(tmp_dir, start_step=5))
            worker.log_warning.assert_called_once()
            assert worker.buffer.start_step == 5

    def test_save_checkpoint_logs_the_resume_skip_count(self):
        """The info log counts prompts a resume will skip."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = self._make_worker(_cfg(rollout_batch_size=4))
            buffer = worker.buffer
            # Two fetched batches of 4: 8 samples taken from the
            # dataloader. 4 groups were released to training, 2 dropped,
            # 2 still buffered: a resume skips the remaining 2.
            buffer.num_samples_fetched = 8
            for group_id in range(8):
                buffer.issue_group(group_id)
            for group_id in range(4):
                buffer.record_result(
                    _result(group_id), _result(group_id).group_metadata()
                )
            buffer.release_groups()
            for group_id in (4, 5):
                buffer.drop_group(
                    group_id, "equal_reward", 1000, already_buffered=False
                )
            for group_id in (6, 7):
                buffer.record_result(
                    _result(group_id), _result(group_id).group_metadata()
                )

            asyncio.run(worker.save_checkpoint(tmp_dir))

            worker.log_info.assert_called_once()
            message = worker.log_info.call_args[0][0]
            assert "fetched 8 samples" in message
            assert "4 groups released" in message
            assert "2 dropped" in message
            assert "skip 2 prompts" in message

    def test_skip_count_after_a_resume_counts_only_this_process(self):
        """After a resume the skip count restarts from the reload.

        A fresh process loads data.pt, fetches on, and saves again: the
        log must count only what the new process fetched, dropped, and
        buffered — not the samples the previous process consumed.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = self._make_worker(_cfg(rollout_batch_size=4))
            buffer = first.buffer
            buffer.num_samples_fetched = 8
            for group_id in range(8):
                buffer.issue_group(group_id)
            for group_id in range(4):
                buffer.record_result(
                    _result(group_id), _result(group_id).group_metadata()
                )
            buffer.release_groups()
            for group_id in (4, 5):
                buffer.drop_group(
                    group_id, "equal_reward", 1000, already_buffered=False
                )
            for group_id in (6, 7):
                buffer.record_result(
                    _result(group_id), _result(group_id).group_metadata()
                )
            asyncio.run(first.save_checkpoint(tmp_dir))
            first.log_info.reset_mock()

            # The reloaded worker: the dataloader fast-forwards past the
            # first 8 samples, then this process fetches one more batch
            # of 4. 2 groups are dropped, 2 are buffered.
            second = self._make_worker(_cfg(rollout_batch_size=4))
            asyncio.run(second.load_checkpoint(tmp_dir, start_step=1))
            data_iter = iter(second.buffer.dataloader)
            next(data_iter)
            second.buffer.num_samples_fetched = 4
            for group_id in range(4):
                second.buffer.issue_group(group_id)
            for group_id in (0, 1):
                second.buffer.drop_group(
                    group_id, "stale", 1000, already_buffered=False
                )
            for group_id in (2, 3):
                second.buffer.record_result(
                    _result(group_id), _result(group_id).group_metadata()
                )

            asyncio.run(second.save_checkpoint(tmp_dir))

            message = second.log_info.call_args[0][0]
            assert "fetched 4 samples" in message
            assert "0 groups released" in message
            assert "2 dropped" in message
            assert "skip 2 prompts" in message


class TestWorkerResident:
    """The worker's resident coroutines over fake channels."""

    @staticmethod
    def _make_worker(**kwargs):
        return _make_worker(**kwargs)

    def test_issue_buffer_release_round_trip(self):
        """Prompts are issued, results buffered, a release reaches the actor."""

        async def run():
            worker = _make_worker(rollout_batch_size=2)
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                # The initial budget of one batch issues both prompts.
                request = await asyncio.wait_for(
                    dataloader_channel._queue.get(), timeout=5
                )
                second = await asyncio.wait_for(
                    dataloader_channel._queue.get(), timeout=5
                )
                assert request.group_ids == [0]
                assert second.group_ids == [1]
                assert request.n == 2 and len(request.input_ids) == 1

                # Results come back; the batch is not released until asked.
                input_channel.put(_result(group_id=0))
                input_channel.put(_result(group_id=1))
                await asyncio.sleep(0.05)
                assert rollout_channel._queue.empty()

                await worker.release_batch(version=0)
                released = [
                    await asyncio.wait_for(rollout_channel._queue.get(), timeout=5)
                    for _ in range(2)
                ]
                assert [r.group_id for r in released] == [0, 1]
                assert worker.buffer.pending_release_version is None
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_release_waits_until_the_batch_is_complete(self):
        """An early release request is held until enough results arrive."""

        async def run():
            worker = _make_worker(rollout_batch_size=2)
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                await worker.release_batch(version=0)
                input_channel.put(_result(group_id=0))
                await asyncio.sleep(0.05)
                # Only one group back: the release is still held.
                assert rollout_channel._queue.empty()
                input_channel.put(_result(group_id=1))
                released = [
                    await asyncio.wait_for(rollout_channel._queue.get(), timeout=5)
                    for _ in range(2)
                ]
                assert [r.group_id for r in released] == [0, 1]
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_second_release_while_pending_is_rejected(self):
        """A release request while one is pending fails the assert."""

        async def run():
            worker = _make_worker(rollout_batch_size=2)
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                # Wait for the first prompt: the resident coroutines are up
                # and the release channel is wired by then.
                await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                await worker.release_batch(version=0)
                try:
                    await worker.release_batch(version=1)
                    raise AssertionError("expected the second release to fail")
                except AssertionError as e:
                    assert "already pending" in str(e)
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_results_without_group_id_fail_the_resident_method(self):
        """An unroutable result ends the resident method with an error."""

        async def run():
            worker = _make_worker(rollout_batch_size=2)
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                input_channel.put(_result(group_id=None))
                with pytest.raises(ValueError, match="without a group id"):
                    await asyncio.wait_for(resident, timeout=5)
                assert worker.buffer.buffer == {}
            finally:
                await worker.stop()

        asyncio.run(run())

    def test_new_versions_admit_more_prompts(self):
        """set_loaded_versions wakes admission for the next batch."""

        async def run():
            worker = self._make_worker(rollout_batch_size=2, num_samples=8)
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                # Budget spent; admission now waits on the event. The
                # second batch is already fetched (4 samples) but none
                # of it is issued yet.
                await asyncio.sleep(0.05)
                assert dataloader_channel._queue.empty()
                assert worker.buffer.num_samples_fetched == 4
                await worker.set_loaded_versions({0: 1, 1: 1})
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                # Third batch fetched once the budget is spent again.
                await asyncio.sleep(0.05)
                assert dataloader_channel._queue.empty()
                assert worker.buffer.num_samples_fetched == 6
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_wake_without_budget_growth_issues_nothing(self):
        """A threshold update that frees no slot admits no prompt."""

        async def run():
            worker = self._make_worker(rollout_batch_size=2, staleness_threshold=0)
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                # The initial budget of 2 is spent on the first batch.
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                # Rewriting the same threshold wakes admission without
                # freeing a slot; nothing may be issued.
                await worker.set_staleness_threshold(0)
                await asyncio.sleep(0.1)
                assert dataloader_channel._queue.empty()
                # A version bump frees slots and the next batch follows.
                await worker.set_loaded_versions({0: 1, 1: 1})
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_stop_ends_the_resident_method(self):
        """stop() cancels the resident coroutines and returns normally."""

        async def run():
            worker = self._make_worker()
            resident = asyncio.create_task(
                worker.run_continuously(
                    _QueueChannel(), _QueueChannel(), _QueueChannel()
                )
            )
            await asyncio.sleep(0.05)
            assert not resident.done()
            await worker.stop()
            await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())


class TestDiscard:
    """Dropping a finished group returns its admission slot."""

    def test_drop_returns_the_admission_slot(self):
        """After a drop the same number of prompts may be issued again."""
        buffer = _BufferFixture.make(rollout_batch_size=2)
        for _ in range(2):
            buffer.issue_group(0)
        assert buffer._prompts_to_issue() == 0
        buffer.drop_group(0, "equal_reward", 1000, already_buffered=False)
        assert buffer._prompts_to_issue() == 1
        assert buffer.inflight_groups() == 1

    def test_drop_never_reissues_the_group(self):
        """A dropped group is not buffered and not released."""
        buffer = _BufferFixture.make(rollout_batch_size=2)
        for _ in range(2):
            buffer.issue_group(0)
        buffer.drop_group(0, "stale", 1000, already_buffered=False)
        assert 0 not in buffer.buffer
        buffer.record_result(_result(1), _result(1).group_metadata())
        released = buffer.release_groups()
        assert [result.group_id for result in released] == [1]

    def test_discarded_group_record_takes_the_receive_path(self):
        """A DiscardedGroup from the agent loop wakes admission."""

        async def run():
            worker = _make_worker(rollout_batch_size=2)
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                input_channel.put(DiscardedGroup(group_id=0, reason="task_rule"))
                # The freed slot admits the next prompt (dataset sample 2).
                request = await asyncio.wait_for(
                    dataloader_channel._queue.get(), timeout=5
                )
                assert request.group_ids == [2]
                assert 0 not in worker.buffer.buffer
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_repeatedly_dropped_samples_are_counted(self):
        """The same dataset sample dropped twice enters the metric."""
        buffer = _BufferFixture.make(rollout_batch_size=4)
        for sample_idx in (3, 7, 3):
            group_id = buffer.issue_group(sample_idx)[0]
            buffer.drop_group(group_id, "stale", 1000, already_buffered=False)
        metrics = buffer.get_step_metrics()
        assert metrics["dropped_groups"] == 3
        assert metrics["dropped_stale"] == 3
        assert metrics["repeatedly_dropped_prompts"] == 1

    def test_drop_limit_raises_with_reason_counts(self):
        """Dropping over the limit between releases fails the resident."""
        buffer = _BufferFixture.make(rollout_batch_size=2)
        for group_id in (0, 1):
            buffer.issue_group(0)
            buffer.drop_group(group_id, "stale", 2, already_buffered=False)
        # Two drops sit at the limit; the third crosses it.
        buffer.issue_group(0)
        with pytest.raises(RuntimeError, match="stale: 3"):
            buffer.drop_group(2, "stale", 2, already_buffered=False)

    def test_drop_limit_resets_after_a_release(self):
        """A release clears the between-release drop window."""
        buffer = _BufferFixture.make(rollout_batch_size=2)
        buffer.drop_group(0, "stale", 1, already_buffered=False)
        buffer.record_result(_result(1), _result(1).group_metadata())
        buffer.record_result(_result(2), _result(2).group_metadata())
        buffer.release_groups()
        # The window restarts at zero: one more drop stays under it.
        buffer.drop_group(3, "stale", 1, already_buffered=False)


class TestRejectEqualReward:
    """The equal-reward group filter in the receive path."""

    def test_reject_equal_flag_is_parsed_from_repo_shape(self):
        """The flag is read from down_sampling_config, RLinf's own shape."""
        cfg = _cfg(rollout_batch_size=2, reject_equal_reward=True)
        with mock.patch.object(Worker, "__init__", return_value=None):
            worker = RolloutBufferWorker(cfg, _ToyDataset(num_samples=8))
        assert worker.reject_equal_reward is True

    def test_reject_equal_flag_defaults_to_off(self):
        """Without down_sampling the worker keeps every group."""
        cfg = _cfg(rollout_batch_size=2)
        with mock.patch.object(Worker, "__init__", return_value=None):
            worker = RolloutBufferWorker(cfg, _ToyDataset(num_samples=8))
        assert worker.reject_equal_reward is False

    def test_reject_equal_flag_respects_master_switch(self):
        """A set sub-option with do_down_sampling off stays inert."""
        cfg = _cfg(rollout_batch_size=2)
        with open_dict(cfg.algorithm):
            cfg.algorithm.down_sampling = {
                "do_down_sampling": False,
                "down_sampling_config": {"reject_equal_reward": True},
            }
        with mock.patch.object(Worker, "__init__", return_value=None):
            worker = RolloutBufferWorker(cfg, _ToyDataset(num_samples=8))
        assert worker.reject_equal_reward is False

    def test_all_equal_rewards_are_dropped(self):
        """A group whose trajectories share one reward never buffers."""

        async def run():
            worker = _make_worker(rollout_batch_size=2, reject_equal_reward=True)
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                input_channel.put(_result(group_id=0, rewards=[1.0, 1.0]))
                await asyncio.sleep(0.05)
                assert 0 not in worker.buffer.buffer
                assert worker.buffer.num_dropped_groups == {"equal_reward": 1}
                # The freed slot admits the next prompt.
                await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_mixed_rewards_are_buffered(self):
        """A reward spread keeps the group."""

        async def run():
            worker = _make_worker(rollout_batch_size=2, reject_equal_reward=True)
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(input_channel, _QueueChannel(), rollout_channel)
            )
            try:
                input_channel.put(_result(group_id=0, rewards=[0.0, 1.0]))
                await asyncio.sleep(0.05)
                assert 0 in worker.buffer.buffer
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())


class TestStalenessDrops:
    """Receive-time first pass and release-time authoritative staleness."""

    def test_receive_time_pass_drops_over_stale_groups(self):
        """A group staler than what the actor has loaded is dropped."""

        async def run():
            worker = _make_worker(rollout_batch_size=2, max_actor_accept_staleness=1)
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                await worker.set_loaded_versions({0: 4, 1: 4})
                # version 4 loaded, earliest generate version 0: gap 4 > 1.
                input_channel.put(_result(group_id=0, earliest_version=0))
                await asyncio.sleep(0.05)
                assert 0 not in worker.buffer.buffer
                assert worker.buffer.num_dropped_groups == {"stale": 1}
                await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_release_time_check_is_authoritative(self):
        """A group fresh at receive time can still be dropped at release."""

        async def run():
            worker = _make_worker(
                rollout_batch_size=2, max_actor_accept_staleness=1, num_samples=16
            )
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                # No versions loaded yet: the receive-time pass cannot
                # judge, so all three groups buffer.
                input_channel.put(_result(group_id=0, earliest_version=0))
                input_channel.put(_result(group_id=1, earliest_version=1))
                input_channel.put(_result(group_id=2, earliest_version=1))
                await asyncio.sleep(0.05)
                assert set(worker.buffer.buffer) == {0, 1, 2}
                # The release batch runs under version 2: group 0
                # (earliest 0) is 2 versions stale, the others only 1.
                await worker.release_batch(version=2)
                released = [
                    await asyncio.wait_for(rollout_channel._queue.get(), timeout=5)
                    for _ in range(2)
                ]
                assert [r.group_id for r in released] == [1, 2]
                assert worker.buffer.num_dropped_groups == {"stale": 1}
                # The dropped group was already counted finished by
                # record_result; a release-time drop must not count it
                # twice.
                assert worker.buffer.inflight_groups() == 0
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_release_waits_when_stale_drops_empty_the_batch(self):
        """A release emptied by stale drops stays pending for replacements."""

        async def run():
            worker = _make_worker(rollout_batch_size=2, max_actor_accept_staleness=0)
            dataloader_channel = _QueueChannel()
            input_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    input_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                input_channel.put(_result(group_id=0, earliest_version=0))
                input_channel.put(_result(group_id=1, earliest_version=0))
                await asyncio.sleep(0.05)
                await worker.release_batch(version=3)
                # Both groups are 3 versions stale under a threshold of
                # 0: dropped, nothing released yet.
                await asyncio.sleep(0.05)
                assert rollout_channel._queue.empty()
                assert worker.buffer.pending_release_version == 3
                # Fresh replacements arrive and the pending release goes
                # out under the original version.
                input_channel.put(_result(group_id=2, earliest_version=3))
                input_channel.put(_result(group_id=3, earliest_version=3))
                released = [
                    await asyncio.wait_for(rollout_channel._queue.get(), timeout=5)
                    for _ in range(2)
                ]
                assert [r.group_id for r in released] == [2, 3]
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())


class TestGroupMetadataView:
    """DynamicRolloutResult.group_metadata over real results."""

    @staticmethod
    def _dynamic_result(
        rewards=None,
        version_start=None,
        version_end=None,
        idx_to_traj=None,
        response_lengths=None,
        is_end=None,
    ):
        from rlinf.data.schema.reasoning_results import DynamicRolloutResult

        num_sequence = len(idx_to_traj) if idx_to_traj else 2
        return DynamicRolloutResult(
            num_sequence=num_sequence,
            group_size=2,
            idx_to_traj=idx_to_traj or [0, 1],
            input_ids=[[1, 2, 3] for _ in range(num_sequence)],
            prompt_lengths=[2] * num_sequence,
            response_lengths=response_lengths or [1] * num_sequence,
            is_end=is_end or [True] * num_sequence,
            rewards=rewards,
            version_start=version_start,
            version_end=version_end,
        )

    def test_metadata_summarizes_rewards_versions_tokens_calls(self):
        result = self._dynamic_result(
            rewards=[1.0, 0.0, 1.0, 0.0],
            version_start=[1, 1, 2, 3],
            version_end=[1, 2, 3, 3],
            idx_to_traj=[0, 0, 1, 1],
            response_lengths=[2, 3, 4, 5],
        )
        metadata = result.group_metadata()
        assert metadata.group_id is None
        assert metadata.rewards == [1.0, 1.0]
        assert metadata.earliest_version == 1
        assert metadata.completion_version == 3
        assert metadata.num_gen_tokens == 14
        # One generate call per sequence, is_end is irrelevant.
        assert metadata.num_generate_calls == 4

    def test_metadata_without_versions_or_rewards(self):
        result = self._dynamic_result(rewards=None, is_end=[False, False])
        metadata = result.group_metadata()
        assert metadata.rewards == [None, None]
        assert metadata.earliest_version is None
        assert metadata.completion_version is None
        assert metadata.num_generate_calls == 2


class TestRewardsAllEqual:
    """rewards_all_equal over per-trajectory reward views."""

    def test_equal_rewards_are_detected(self):
        from rlinf.utils.down_sampling import rewards_all_equal

        assert rewards_all_equal([1.0, 1.0, 1.0])
        assert rewards_all_equal([0.5, 0.5])

    def test_unequal_rewards_are_kept(self):
        from rlinf.utils.down_sampling import rewards_all_equal

        assert not rewards_all_equal([1.0, 0.0])
        assert not rewards_all_equal([0.0, 0.0, 0.1])

    def test_none_or_empty_rewards_are_kept(self):
        """A None reward cannot be judged; the group is kept, not crashed."""
        from rlinf.utils.down_sampling import rewards_all_equal

        assert not rewards_all_equal(None)
        assert not rewards_all_equal([])
        assert not rewards_all_equal([1.0, None])
        assert not rewards_all_equal([None, None])
