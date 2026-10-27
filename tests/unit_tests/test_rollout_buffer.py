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

Covers the buffer's admission budget and release order, the checkpoint
round-trip of its dataloader, the group-id passthrough on
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
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from rlinf.data.datasets.common.item import DatasetItem
from rlinf.workers.rollout_buffer.rollout_buffer_worker import (
    RolloutBuffer,
    RolloutBufferWorker,
)


def _cfg(
    rollout_batch_size: int = 4,
    staleness_threshold: int = 0,
    num_samples: int = 64,
    shuffle: bool = False,
):
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
            "algorithm": {
                "staleness_threshold": staleness_threshold,
                "group_size": 2,
                "max_num_gen_batches": 1,
            },
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


def _result(group_id=None):
    """A single-group DynamicRolloutResult double."""
    return SimpleNamespace(group_id=group_id)


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
            buffer.issue_group()
        assert buffer._prompts_to_issue() == 0

    def test_loaded_version_growth_adds_budget(self):
        """A new loaded version admits another batch."""
        buffer = _BufferFixture.make(rollout_batch_size=4, loaded_versions={0: 0, 1: 0})
        for _ in range(4):
            buffer.issue_group()
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
            buffer.issue_group()
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
                buffer.issue_group()
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
            buffer.buffer[group_id] = _result(group_id)
        released = buffer.release_groups()
        assert [result.group_id for result in released] == [1, 5, 9]

    def test_release_removes_the_groups_from_the_buffer(self):
        """Released groups leave the buffer and the issue-version dict."""
        buffer = _BufferFixture.make(rollout_batch_size=2)
        for group_id in [1, 2, 3]:
            buffer.buffer[group_id] = _result(group_id)
            buffer.issue_versions[group_id] = 0
        buffer.release_groups()
        assert set(buffer.buffer) == {3}
        assert set(buffer.issue_versions) == {3}

    def test_can_release_requires_a_full_batch(self):
        """can_release flips only at rollout_batch_size groups."""
        buffer = _BufferFixture.make(rollout_batch_size=3)
        assert not buffer.can_release()
        buffer.buffer[1] = _result(1)
        buffer.buffer[2] = _result(2)
        assert not buffer.can_release()
        buffer.buffer[3] = _result(3)
        assert buffer.can_release()


class TestRecordResult:
    """Buffering of finished groups."""

    def test_results_without_group_id_raise(self):
        """Merged or split results (group_id None) fail the buffer."""
        buffer = _BufferFixture.make()
        with pytest.raises(ValueError, match="without a group id"):
            buffer.record_result(_result(None))
        assert buffer.buffer == {}
        assert buffer.record_result(_result(3)) == 3
        assert buffer.buffer[3].group_id == 3

    def test_inflight_counts_issued_minus_finished(self):
        """In-flight groups drop back to zero as results arrive."""
        buffer = _BufferFixture.make(rollout_batch_size=4)
        issued = [buffer.issue_group()[0] for _ in range(4)]
        assert buffer.inflight_groups() == 4
        for group_id in issued[:2]:
            buffer.record_result(_result(group_id))
        assert buffer.inflight_groups() == 2
        for group_id in issued[2:]:
            buffer.record_result(_result(group_id))
        assert buffer.inflight_groups() == 0


class TestStepMetrics:
    """Step-scoped metrics of the buffer."""

    def test_metrics_drain_and_reset(self):
        """Per-step counters reset; gauges read live values."""
        buffer = _BufferFixture.make(rollout_batch_size=4)
        for _ in range(2):
            buffer.issue_group()
        buffer.record_result(_result(0))
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
            buffer.buffer[group_id] = _result(group_id)
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


class TestWorkerResident:
    """The worker's resident coroutines over fake channels."""

    @staticmethod
    def _make_worker(rollout_batch_size=2, staleness_threshold=0, num_samples=8):
        cfg = _cfg(rollout_batch_size, staleness_threshold)
        worker = RolloutBufferWorker.__new__(RolloutBufferWorker)
        worker.cfg = cfg
        worker.buffer = RolloutBuffer(cfg, _ToyDataset(num_samples))
        worker.log_warning = mock.MagicMock()
        worker.log_info = mock.MagicMock()
        worker._resident_task = None
        worker._data_iter = None
        worker._rollout_channel = None
        worker._pending_fetch = None
        return worker

    def test_issue_buffer_release_round_trip(self):
        """Prompts are issued, results buffered, a release reaches the actor."""

        async def run():
            worker = self._make_worker(rollout_batch_size=2)
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
            worker = self._make_worker(rollout_batch_size=2)
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
            worker = self._make_worker(rollout_batch_size=2)
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
            worker = self._make_worker(rollout_batch_size=2)
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
                # Budget spent; admission now waits on the event.
                await asyncio.sleep(0.05)
                assert dataloader_channel._queue.empty()
                await worker.set_loaded_versions({0: 1, 1: 1})
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
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
