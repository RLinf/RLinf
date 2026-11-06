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
checkpoint round-trip of its dataloader, and the worker's resident
coroutines over fake channels. No Ray or sglang is launched.
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
    max_inflight_groups: int = 8,
    max_buffered_groups: int = 8,
    num_samples: int = 64,
    shuffle: bool = False,
    reject_equal_reward: bool = False,
    max_actor_accept_staleness: int | None = None,
    max_dropped_groups_per_step: int = 1000,
    task_name: str | None = None,
):
    algorithm = {
        "group_size": 2,
        "max_num_gen_batches": 1,
    }
    if reject_equal_reward:
        algorithm["down_sampling"] = {
            "do_down_sampling": True,
            "down_sampling_config": {"reject_equal_reward": True},
        }
    agentloop = {}
    if max_actor_accept_staleness is not None:
        agentloop["max_actor_accept_staleness"] = max_actor_accept_staleness
    if task_name is not None:
        agentloop["task_name"] = task_name
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
            "agentloop": agentloop,
            "rollout_buffer": {
                "max_inflight_groups": max_inflight_groups,
                "max_buffered_groups": max_buffered_groups,
                "max_dropped_groups_per_step": max_dropped_groups_per_step,
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


def _result(group_id=None, earliest_version=None, rewards=None, task_name=None):
    """A single-group DynamicRolloutResult double.

    `group_metadata()` mirrors the real method's contract; other fields
    are absent because no buffer path reads them.
    """

    def group_metadata():
        return GroupMetadata(
            group_id=group_id,
            task_name=task_name,
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
        max_inflight_groups=8,
        max_buffered_groups=8,
        loaded_versions=None,
    ):
        with mock.patch.object(RolloutBuffer, "_build_dataloader"):
            buffer = RolloutBuffer(
                _cfg(
                    rollout_batch_size,
                    max_inflight_groups=max_inflight_groups,
                    max_buffered_groups=max_buffered_groups,
                ),
                None,
            )
        buffer.dataloader = None
        if loaded_versions is not None:
            buffer.loaded_versions = dict(loaded_versions)
        return buffer


def _make_worker(
    rollout_batch_size=2,
    max_inflight_groups=8,
    max_buffered_groups=8,
    num_samples=8,
    reject_equal_reward=False,
    max_actor_accept_staleness=None,
    max_dropped_groups_per_step=1000,
    task_name=None,
):
    """A RolloutBufferWorker with the loop state but no Worker init."""
    cfg = _cfg(
        rollout_batch_size,
        max_inflight_groups=max_inflight_groups,
        max_buffered_groups=max_buffered_groups,
        reject_equal_reward=reject_equal_reward,
        max_actor_accept_staleness=max_actor_accept_staleness,
        max_dropped_groups_per_step=max_dropped_groups_per_step,
        task_name=task_name,
    )
    worker = RolloutBufferWorker.__new__(RolloutBufferWorker)
    worker.cfg = cfg
    worker.buffer = RolloutBuffer(cfg, _ToyDataset(num_samples))
    # This double holds the parsed flag directly; parsing from cfg is
    # covered by test_reject_equal_flag_is_parsed_from_repo_shape.
    worker.reject_equal_reward = reject_equal_reward
    worker.max_actor_accept_staleness = max_actor_accept_staleness
    worker.max_dropped_groups_per_step = max_dropped_groups_per_step
    worker.task_name = task_name
    worker.log_warning = mock.MagicMock()
    worker.log_info = mock.MagicMock()
    worker._resident_task = None
    worker._data_iter = None
    worker._rollout_channel = None
    worker._pending_fetch = None
    return worker


def _task_state(buffer: RolloutBuffer, task_name=None) -> dict:
    """The buffer's per-task state for `task_name` (None = no task)."""
    return buffer._task(task_name)


class TestAdmissionByCapacity:
    """The two admission conditions of the rollout buffer."""

    def test_both_conditions_free_admits(self):
        """With capacity in flight and in the buffer, admission passes."""
        buffer = _BufferFixture.make(max_inflight_groups=2, max_buffered_groups=2)
        assert buffer._admission_block_reason() is None

    def test_inflight_limit_alone_blocks(self):
        """In-flight groups at the limit block admission by themselves."""
        buffer = _BufferFixture.make(max_inflight_groups=2, max_buffered_groups=8)
        buffer.issue_group(0, None)
        buffer.issue_group(0, None)
        assert buffer._admission_block_reason() == "inflight_limit"

    def test_buffer_full_alone_blocks(self):
        """Buffered groups at the limit block admission by themselves."""
        buffer = _BufferFixture.make(max_inflight_groups=8, max_buffered_groups=2)
        for group_id in range(2):
            buffer.issue_group(group_id, None)
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
        assert buffer._admission_block_reason() == "buffer_full"

    def test_inflight_groups_count_toward_the_buffer_cap(self):
        """Groups still in flight block admission once the total reaches
        the buffer cap, even with the buffer itself below it."""
        buffer = _BufferFixture.make(max_inflight_groups=8, max_buffered_groups=4)
        for group_id in range(2):
            buffer.issue_group(group_id, None)
        buffer.record_result(_result(0), _result(0).group_metadata())
        # 1 in flight + 1 buffered: two more fit, the third is blocked.
        assert buffer._admission_block_reason() is None
        for group_id in (2, 3):
            buffer.issue_group(group_id, None)
        assert buffer._buffered_groups() == 1
        assert buffer._admission_block_reason() == "buffer_full"

    def test_one_result_at_the_limit_frees_one_slot(self):
        """A single result at the in-flight limit admits exactly one group."""

        async def run():
            buffer = _BufferFixture.make(max_inflight_groups=2, max_buffered_groups=8)
            for _ in range(2):
                buffer.issue_group(0, None)
            slot_taken = False

            async def issue_if_admitted():
                nonlocal slot_taken
                await buffer.wait_for_admission_slot()
                slot_taken = True

            waiter = asyncio.create_task(issue_if_admitted())
            await asyncio.sleep(0.05)
            assert not slot_taken
            # One group returns: one replacement may be issued, not two.
            buffer.record_result(_result(0), _result(0).group_metadata())
            await asyncio.wait_for(waiter, timeout=5)
            buffer.issue_group(0, None)
            assert buffer._admission_block_reason() == "inflight_limit"

        asyncio.run(run())

    def test_one_discard_at_the_limit_frees_one_slot(self):
        """A discard record at the in-flight limit admits exactly one group."""

        async def run():
            buffer = _BufferFixture.make(max_inflight_groups=2, max_buffered_groups=8)
            for _ in range(2):
                buffer.issue_group(0, None)
            slot_taken = False

            async def issue_if_admitted():
                nonlocal slot_taken
                await buffer.wait_for_admission_slot()
                slot_taken = True

            waiter = asyncio.create_task(issue_if_admitted())
            await asyncio.sleep(0.05)
            assert not slot_taken
            buffer.drop_group(0, "task_rule", 1000, already_buffered=False)
            await asyncio.wait_for(waiter, timeout=5)
            buffer.issue_group(0, None)
            assert buffer._admission_block_reason() == "inflight_limit"

        asyncio.run(run())

    def test_buffer_full_recovers_after_actor_handoff(self):
        """A full buffer admits again once a batch is handed to the actor."""

        async def run():
            buffer = _BufferFixture.make(
                rollout_batch_size=2, max_inflight_groups=8, max_buffered_groups=2
            )
            for group_id in range(2):
                buffer.issue_group(group_id, None)
                buffer.record_result(
                    _result(group_id), _result(group_id).group_metadata()
                )
            slot_taken = False

            async def issue_if_admitted():
                nonlocal slot_taken
                await buffer.wait_for_admission_slot()
                slot_taken = True

            waiter = asyncio.create_task(issue_if_admitted())
            await asyncio.sleep(0.05)
            assert not slot_taken
            # The release empties the buffer: admission recovers.
            buffer.release_groups()
            await asyncio.wait_for(waiter, timeout=5)
            assert slot_taken

        asyncio.run(run())

    def test_wake_without_capacity_keeps_waiting(self):
        """A wake-up that frees no capacity does not hand a slot out."""

        async def run():
            buffer = _BufferFixture.make(max_inflight_groups=2, max_buffered_groups=8)
            for _ in range(2):
                buffer.issue_group(0, None)
            slot_taken = False

            async def issue_if_admitted():
                nonlocal slot_taken
                await buffer.wait_for_admission_slot()
                slot_taken = True

            waiter = asyncio.create_task(issue_if_admitted())
            await asyncio.sleep(0.05)
            # Loaded versions no longer affect admission; the wake frees
            # no in-flight capacity.
            buffer.set_loaded_versions({0: 5, 1: 5})
            await asyncio.sleep(0.05)
            assert not slot_taken
            assert not waiter.done()
            buffer.record_result(_result(0), _result(0).group_metadata())
            await asyncio.wait_for(waiter, timeout=5)
            assert slot_taken

        asyncio.run(run())

    def test_set_loaded_versions_no_longer_wakes_admission(self):
        """A version update does not set the admission event at all."""
        buffer = _BufferFixture.make(max_inflight_groups=2, max_buffered_groups=8)
        for _ in range(2):
            buffer.issue_group(0, None)
        buffer._admission_event.clear()
        buffer.set_loaded_versions({0: 5, 1: 5})
        assert not buffer._admission_event.is_set()


class TestPerTaskState:
    """The buffer's per-task keyed state."""

    def test_state_is_keyed_by_task_name(self):
        """Issue, buffer and drop file their state under the task key."""
        buffer = _BufferFixture.make()
        buffer.issue_group(0, "searchr1")
        buffer.record_result(
            _result(group_id=0, task_name="searchr1"),
            _result(group_id=0, task_name="searchr1").group_metadata(),
        )
        task = _task_state(buffer, "searchr1")
        assert task["num_issued_groups"] == 1
        assert task["num_finished_groups"] == 1
        assert set(task["buffer"]) == {0}
        assert buffer.tasks.keys() == {"searchr1"}

    def test_inflight_and_buffer_count_across_tasks(self):
        """Gauges sum over the per-task state."""
        buffer = _BufferFixture.make()
        for group_id, task_name in [(0, "searchr1"), (1, "code")]:
            buffer.issue_group(group_id, task_name)
        buffer.record_result(
            _result(group_id=0, task_name="searchr1"),
            _result(group_id=0, task_name="searchr1").group_metadata(),
        )
        assert buffer.inflight_groups() == 1
        assert buffer._buffered_groups() == 1

    def test_release_takes_the_oldest_groups_across_tasks(self):
        """A release still hands the actor the oldest buffered groups."""
        buffer = _BufferFixture.make(rollout_batch_size=2)
        group_ids = {}
        for sample, task_name in [(0, "searchr1"), (1, "code"), (2, "searchr1")]:
            group_ids[sample] = buffer.issue_group(sample, task_name)[0]
            buffer.record_result(
                _result(group_id=group_ids[sample], task_name=task_name),
                _result(
                    group_id=group_ids[sample], task_name=task_name
                ).group_metadata(),
            )
        released = buffer.release_groups()
        assert [result.group_id for result in released] == [
            group_ids[0],
            group_ids[1],
        ]
        assert set(_task_state(buffer, "searchr1")["buffer"]) == {group_ids[2]}
        assert _task_state(buffer, "code")["buffer"] == {}

    def test_result_routes_by_the_issue_record_not_the_producer(self):
        """A result whose task stamp differs from issue time files under
        the issuing task, and the issue record is consumed."""
        buffer = _BufferFixture.make()
        buffer.issue_group(0, "searchr1")
        buffer.record_result(
            _result(group_id=0, task_name=None),
            _result(group_id=0, task_name=None).group_metadata(),
        )
        assert _task_state(buffer, "searchr1")["num_finished_groups"] == 1
        assert buffer.tasks.keys() == {"searchr1"}
        assert buffer.group_tasks == {}

    def test_discard_routes_by_the_issue_record_not_the_producer(self):
        """A receive-time drop files under the task the group was issued
        under, regardless of any name the record carries."""
        buffer = _BufferFixture.make()
        buffer.issue_group(0, "searchr1")
        buffer.drop_group(0, "task_rule", 1000, already_buffered=False)
        task = _task_state(buffer, "searchr1")
        assert task["num_dropped_groups"] == {"task_rule": 1}
        assert task["num_finished_groups"] == 1
        assert buffer.group_tasks == {}

    def test_result_for_an_unknown_group_raises(self):
        """A result for a group never issued here is a producer bug."""
        buffer = _BufferFixture.make()
        with pytest.raises(ValueError, match="unknown group id 7"):
            buffer.record_result(_result(7), _result(7).group_metadata())

    def test_discard_for_an_unknown_group_raises(self):
        """A receive-time drop for a group never issued here fails."""
        buffer = _BufferFixture.make()
        with pytest.raises(ValueError, match="unknown group id 7"):
            buffer.drop_group(7, "task_rule", 1000, already_buffered=False)


class TestRelease:
    """Release of buffered groups."""

    def test_release_takes_the_oldest_groups_in_order(self):
        """Released results come out in group-id order."""
        buffer = _BufferFixture.make(rollout_batch_size=3)
        # Issue in shuffled order: group ids are assigned in issue order.
        group_ids = [buffer.issue_group(sample, None)[0] for sample in [5, 1, 9]]
        for group_id in group_ids:
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
        released = buffer.release_groups()
        assert [result.group_id for result in released] == sorted(group_ids)

    def test_release_removes_the_groups_from_the_buffer(self):
        """Released groups leave the buffer and the issue-version dict."""
        buffer = _BufferFixture.make(rollout_batch_size=2, loaded_versions={0: 1})
        group_ids = [buffer.issue_group(sample, None)[0] for sample in range(3)]
        for group_id in group_ids:
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
        buffer.release_groups()
        task = _task_state(buffer, None)
        assert set(task["buffer"]) == {group_ids[2]}
        assert set(task["issue_versions"]) == {group_ids[2]}

    def test_can_release_requires_a_full_batch(self):
        """can_release flips only at rollout_batch_size groups."""
        buffer = _BufferFixture.make(rollout_batch_size=3)
        group_ids = [buffer.issue_group(sample, None)[0] for sample in range(3)]
        assert not buffer.can_release()
        buffer.record_result(
            _result(group_ids[0]), _result(group_ids[0]).group_metadata()
        )
        buffer.record_result(
            _result(group_ids[1]), _result(group_ids[1]).group_metadata()
        )
        assert not buffer.can_release()
        buffer.record_result(
            _result(group_ids[2]), _result(group_ids[2]).group_metadata()
        )
        assert buffer.can_release()


class TestRecordResult:
    """Buffering of finished groups."""

    def test_results_without_group_id_raise(self):
        """Merged or split results (group_id None) fail the buffer."""
        buffer = _BufferFixture.make()
        with pytest.raises(ValueError, match="without a group id"):
            buffer.record_result(_result(None), _result(None).group_metadata())
        assert _task_state(buffer, None)["buffer"] == {}
        group_id = buffer.issue_group(0, None)[0]
        assert (
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
            == group_id
        )
        assert _task_state(buffer, None)["buffer"][group_id][1].group_id == group_id

    def test_inflight_counts_issued_minus_finished(self):
        """In-flight groups drop back to zero as results arrive."""
        buffer = _BufferFixture.make(rollout_batch_size=4)
        issued = [buffer.issue_group(0, None)[0] for _ in range(4)]
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
            buffer.issue_group(0, None)
        buffer.record_result(_result(0), _result(0).group_metadata())
        metrics = buffer.get_step_metrics()
        assert metrics["issued_groups"] == 2
        assert metrics["inflight_groups"] == 1
        assert metrics["size"] == 1
        metrics = buffer.get_step_metrics()
        assert metrics["issued_groups"] == 0
        assert metrics["inflight_groups"] == 1

    def test_inflight_group_limit_equals_the_configured_value(self):
        """The limit metric mirrors the configured max_inflight_groups."""
        buffer = _BufferFixture.make(max_inflight_groups=5)
        metrics = buffer.get_step_metrics()
        assert metrics["inflight_group_limit"] == 5

    def test_admission_wait_split_by_reason(self):
        """Blocked time is reported per reason, as a window fraction."""

        async def run():
            # In-flight limit of 1 with a roomier buffer, so each phase is
            # blocked by exactly one condition.
            buffer = _BufferFixture.make(
                rollout_batch_size=1, max_inflight_groups=1, max_buffered_groups=8
            )
            buffer.issue_group(0, None)
            first = buffer.get_step_metrics()
            assert first["admission_wait_buffer_full"] == 0.0
            assert first["admission_wait_inflight_limit"] == 0.0

            async def wait_once():
                await buffer.wait_for_admission_slot()

            # Blocked by the in-flight limit, then unblocked by a result.
            waiter = asyncio.create_task(wait_once())
            await asyncio.sleep(0.05)
            assert not waiter.done()
            buffer.record_result(_result(0), _result(0).group_metadata())
            await asyncio.wait_for(waiter, timeout=5)
            assert buffer._step_admission_wait["buffer_full"] == 0.0

            # Fill the buffer to its limit, then blocked by the buffer and
            # unblocked by a release.
            for sample in range(1, 8):
                buffer.issue_group(sample, None)
                buffer.record_result(_result(sample), _result(sample).group_metadata())
            assert buffer._admission_block_reason() == "buffer_full"
            waiter = asyncio.create_task(wait_once())
            await asyncio.sleep(0.05)
            assert not waiter.done()
            buffer.release_groups()
            await asyncio.wait_for(waiter, timeout=5)
            metrics = buffer.get_step_metrics()
            assert 0.0 < metrics["admission_wait_inflight_limit"] <= 1.0
            assert 0.0 < metrics["admission_wait_buffer_full"] <= 1.0

        asyncio.run(run())

    def test_admission_wait_spans_metrics_windows(self):
        """A wait crossing two metrics reads lands in each window."""

        async def run():
            buffer = _BufferFixture.make(max_inflight_groups=1)
            buffer.issue_group(0, None)
            buffer.get_step_metrics()  # open the first window

            async def wait_once():
                await buffer.wait_for_admission_slot()

            waiter = asyncio.create_task(wait_once())
            await asyncio.sleep(0.05)
            assert not waiter.done()
            # First read mid-wait: the elapsed part of the wait is already
            # charged, and the next window starts from this read.
            first = buffer.get_step_metrics()
            assert 0.0 < first["admission_wait_inflight_limit"] <= 1.0
            assert not waiter.done()
            await asyncio.sleep(0.05)
            # Second read mid-wait: only the part since the first read.
            second = buffer.get_step_metrics()
            assert 0.0 < second["admission_wait_inflight_limit"] <= 1.0
            # The wait then ends; whatever accrued after the second read
            # belongs to the next window and stays within it.
            buffer.record_result(_result(0), _result(0).group_metadata())
            await asyncio.wait_for(waiter, timeout=5)
            third = buffer.get_step_metrics()
            assert 0.0 <= third["admission_wait_inflight_limit"] <= 1.0

        asyncio.run(run())

    def test_batch_issue_version_metrics(self):
        """A released batch reports its distinct and oldest-issue share."""
        buffer = _BufferFixture.make(rollout_batch_size=3)
        for sample, version in enumerate([0, 0, 1]):
            group_id = buffer.issue_group(sample, None)[0]
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
            buffer._task(None)["issue_versions"][group_id] = version
        buffer.release_groups()
        metrics = buffer.get_step_metrics()
        assert metrics["batch_issue_versions"] == 2
        assert metrics["batch_oldest_issue_version_ratio"] == 2 / 3


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
            second = next(iter(reloaded.buffer.dataloader))["idx"].tolist()
            assert second[0] == first[0] + 4

    def test_data_pt_is_written_atomically(self):
        """save_checkpoint leaves no .tmp file behind."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = self._make_worker(_cfg(rollout_batch_size=4))
            asyncio.run(worker.save_checkpoint(tmp_dir))
            assert os.path.isfile(os.path.join(tmp_dir, "data.pt"))
            assert not os.path.exists(os.path.join(tmp_dir, "data.pt.tmp"))

    def test_missing_state_warns(self):
        """Without data.pt the dataset restarts, with a warning."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = self._make_worker(_cfg(rollout_batch_size=4))
            asyncio.run(worker.load_checkpoint(tmp_dir, start_step=5))
            worker.log_warning.assert_called_once()

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
                buffer.issue_group(group_id, None)
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
                buffer.issue_group(group_id, None)
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
                second.buffer.issue_group(group_id, None)
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
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                # The initial capacity issues both prompts.
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
                agent_output_channel.put(_result(group_id=0))
                agent_output_channel.put(_result(group_id=1))
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
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                await worker.release_batch(version=0)
                agent_output_channel.put(_result(group_id=0))
                await asyncio.sleep(0.05)
                # Only one group back: the release is still held.
                assert rollout_channel._queue.empty()
                agent_output_channel.put(_result(group_id=1))
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
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
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
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                agent_output_channel.put(_result(group_id=None))
                with pytest.raises(ValueError, match="without a group id"):
                    await asyncio.wait_for(resident, timeout=5)
                assert worker.buffer._buffered_groups() == 0
            finally:
                await worker.stop()

        asyncio.run(run())

    def test_inflight_limit_pauses_and_results_resume_issuing(self):
        """Issuing stops at the in-flight limit and resumes on results."""

        async def run():
            worker = self._make_worker(
                rollout_batch_size=2, max_inflight_groups=2, num_samples=8
            )
            dataloader_channel = _QueueChannel()
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                # In-flight at the limit; admission waits. The second
                # batch is already fetched (4 samples) but none of it is
                # issued yet.
                await asyncio.sleep(0.05)
                assert dataloader_channel._queue.empty()
                assert worker.buffer.num_samples_fetched == 4
                # A version update does not unblock admission.
                await worker.set_loaded_versions({0: 1, 1: 1})
                await asyncio.sleep(0.05)
                assert dataloader_channel._queue.empty()
                # A result frees in-flight capacity; issuing resumes.
                agent_output_channel.put(_result(group_id=0))
                await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                await asyncio.sleep(0.05)
                assert dataloader_channel._queue.empty()
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_training_bottleneck_keeps_total_within_the_buffer_cap(self):
        """Results arriving faster than releases keep inflight + buffered
        under the buffer cap, and a release re-issues exactly the missing
        amount.

        In-flight groups enter the buffer on completion, so admission
        counts them toward max_buffered_groups; the buffer size alone can
        never exceed it.
        """

        async def run():
            worker = _make_worker(
                rollout_batch_size=2,
                max_inflight_groups=8,
                max_buffered_groups=4,
                num_samples=64,
            )
            buffer = worker.buffer
            dataloader_channel = _QueueChannel()
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )

            def check_invariant():
                # Strict: admission runs before each issue, so the running
                # total only reaches the cap once an issue is in flight.
                assert buffer.inflight_groups() + buffer._buffered_groups() <= 4
                assert buffer._buffered_groups() <= 4

            try:
                # Training is the bottleneck: nothing is released while
                # the engine finishes every group it is given. Each
                # result frees in-flight capacity, so issuing runs ahead
                # of this loop until the in-flight + buffered total
                # reaches the cap: exactly four groups are issued.
                for _ in range(4):
                    request = await asyncio.wait_for(
                        dataloader_channel._queue.get(), timeout=5
                    )
                    check_invariant()
                    agent_output_channel.put(_result(group_id=request.group_ids[0]))
                    await asyncio.sleep(0.05)
                    check_invariant()
                # Total at the cap: admission is blocked by the buffer
                # condition and the buffer alone holds every group.
                assert buffer._admission_block_reason() == "buffer_full"
                assert buffer._buffered_groups() == 4
                assert buffer.inflight_groups() == 0
                await asyncio.sleep(0.05)
                assert dataloader_channel._queue.empty()

                # A release hands one training batch over: two groups
                # leave, and exactly two are issued to top the total back
                # up to the cap.
                await worker.release_batch(version=0)
                released = [
                    await asyncio.wait_for(rollout_channel._queue.get(), timeout=5)
                    for _ in range(2)
                ]
                assert [r.group_id for r in released] == [0, 1]
                refilled = [
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                    for _ in range(2)
                ]
                await asyncio.sleep(0.05)
                assert [r.group_ids for r in refilled] == [[4], [5]]
                assert dataloader_channel._queue.empty()
                check_invariant()
                assert buffer.inflight_groups() == 2
                assert buffer._buffered_groups() == 2
                assert buffer._admission_block_reason() == "buffer_full"
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

    def test_drop_returns_the_admission_capacity(self):
        """After a drop the same number of prompts may be issued again."""
        buffer = _BufferFixture.make(max_inflight_groups=2)
        for _ in range(2):
            buffer.issue_group(0, None)
        assert buffer._admission_block_reason() == "inflight_limit"
        buffer.drop_group(0, "equal_reward", 1000, already_buffered=False)
        assert buffer._admission_block_reason() is None
        assert buffer.inflight_groups() == 1

    def test_drop_never_reissues_the_group(self):
        """A dropped group is not buffered and not released."""
        buffer = _BufferFixture.make(rollout_batch_size=2)
        first, second = (buffer.issue_group(sample, None)[0] for sample in range(2))
        buffer.drop_group(first, "stale", 1000, already_buffered=False)
        assert first not in _task_state(buffer, None)["buffer"]
        buffer.record_result(_result(second), _result(second).group_metadata())
        released = buffer.release_groups()
        assert [result.group_id for result in released] == [second]

    def test_discarded_group_record_takes_the_receive_path(self):
        """A DiscardedGroup from the agent loop wakes admission."""

        async def run():
            worker = _make_worker(rollout_batch_size=2)
            dataloader_channel = _QueueChannel()
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                agent_output_channel.put(DiscardedGroup(group_id=0, reason="task_rule"))
                # The freed slot admits the next prompt (dataset sample 2).
                request = await asyncio.wait_for(
                    dataloader_channel._queue.get(), timeout=5
                )
                assert request.group_ids == [2]
                assert worker.buffer._buffered_groups() == 0
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_repeatedly_dropped_samples_are_counted(self):
        """The same dataset sample dropped twice enters the metric."""
        buffer = _BufferFixture.make(rollout_batch_size=4)
        for sample_idx in (3, 7, 3):
            group_id = buffer.issue_group(sample_idx, None)[0]
            buffer.drop_group(group_id, "stale", 1000, already_buffered=False)
        metrics = buffer.get_step_metrics()
        assert metrics["dropped_groups"] == 3
        assert metrics["dropped_stale"] == 3
        assert metrics["repeatedly_dropped_prompts"] == 1

    def test_drop_limit_raises_with_reason_counts(self):
        """Dropping over the limit between releases fails the resident."""
        buffer = _BufferFixture.make(rollout_batch_size=2)
        for group_id in (0, 1):
            buffer.issue_group(0, None)
            buffer.drop_group(group_id, "stale", 2, already_buffered=False)
        # Two drops sit at the limit; the third crosses it.
        buffer.issue_group(0, None)
        with pytest.raises(RuntimeError, match="stale: 3"):
            buffer.drop_group(2, "stale", 2, already_buffered=False)

    def test_drop_limit_resets_after_a_release(self):
        """A release clears the between-release drop window."""
        buffer = _BufferFixture.make(rollout_batch_size=2)
        first = buffer.issue_group(0, None)[0]
        buffer.drop_group(first, "stale", 1, already_buffered=False)
        for sample in (1, 2):
            group_id = buffer.issue_group(sample, None)[0]
            buffer.record_result(_result(group_id), _result(group_id).group_metadata())
        buffer.release_groups()
        # The window restarts at zero: one more drop stays under it.
        fourth = buffer.issue_group(3, None)[0]
        buffer.drop_group(fourth, "stale", 1, already_buffered=False)


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
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                agent_output_channel.put(_result(group_id=0, rewards=[1.0, 1.0]))
                await asyncio.sleep(0.05)
                assert worker.buffer._buffered_groups() == 0
                assert _task_state(worker.buffer, None)["num_dropped_groups"] == {
                    "equal_reward": 1
                }
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
            dataloader_channel = _QueueChannel()
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                # The admission issues one group; its result carries a
                # reward spread and buffers.
                await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                agent_output_channel.put(_result(group_id=0, rewards=[0.0, 1.0]))
                await asyncio.sleep(0.05)
                assert _task_state(worker.buffer, None)["buffer"]
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
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                await worker.set_loaded_versions({0: 4, 1: 4})
                # version 4 loaded, earliest generate version 0: gap 4 > 1.
                agent_output_channel.put(_result(group_id=0, earliest_version=0))
                await asyncio.sleep(0.05)
                assert worker.buffer._buffered_groups() == 0
                assert _task_state(worker.buffer, None)["num_dropped_groups"] == {
                    "stale": 1
                }
                await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_release_time_check_is_authoritative(self):
        """A group fresh at receive time can still be dropped at release."""

        async def run():
            worker = _make_worker(
                rollout_batch_size=2,
                max_inflight_groups=3,
                max_buffered_groups=8,
                max_actor_accept_staleness=1,
                num_samples=16,
            )
            dataloader_channel = _QueueChannel()
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(3):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                # No versions loaded yet: the receive-time pass cannot
                # judge, so all three groups buffer.
                agent_output_channel.put(_result(group_id=0, earliest_version=0))
                agent_output_channel.put(_result(group_id=1, earliest_version=1))
                agent_output_channel.put(_result(group_id=2, earliest_version=1))
                await asyncio.sleep(0.05)
                assert set(_task_state(worker.buffer, None)["buffer"]) == {0, 1, 2}
                # The release batch runs under version 2: group 0
                # (earliest 0) is 2 versions stale, the others only 1.
                await worker.release_batch(version=2)
                released = [
                    await asyncio.wait_for(rollout_channel._queue.get(), timeout=5)
                    for _ in range(2)
                ]
                assert [r.group_id for r in released] == [1, 2]
                assert _task_state(worker.buffer, None)["num_dropped_groups"] == {
                    "stale": 1
                }
                # The dropped group was already counted finished by
                # record_result; a release-time drop must not count it
                # twice: the three results account for all three groups
                # (two released, one dropped, none still buffered).
                task = _task_state(worker.buffer, None)
                assert task["num_finished_groups"] == 3
                assert len(task["buffer"]) == 0
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_release_waits_when_stale_drops_empty_the_batch(self):
        """A release emptied by stale drops stays pending for replacements."""

        async def run():
            worker = _make_worker(rollout_batch_size=2, max_actor_accept_staleness=0)
            dataloader_channel = _QueueChannel()
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                for _ in range(2):
                    await asyncio.wait_for(dataloader_channel._queue.get(), timeout=5)
                agent_output_channel.put(_result(group_id=0, earliest_version=0))
                agent_output_channel.put(_result(group_id=1, earliest_version=0))
                await asyncio.sleep(0.05)
                await worker.release_batch(version=3)
                # Both groups are 3 versions stale under a threshold of
                # 0: dropped, nothing released yet.
                await asyncio.sleep(0.05)
                assert rollout_channel._queue.empty()
                assert worker.buffer.pending_release_version == 3
                # Fresh replacements arrive and the pending release goes
                # out under the original version.
                agent_output_channel.put(_result(group_id=2, earliest_version=3))
                agent_output_channel.put(_result(group_id=3, earliest_version=3))
                released = [
                    await asyncio.wait_for(rollout_channel._queue.get(), timeout=5)
                    for _ in range(2)
                ]
                assert [r.group_id for r in released] == [2, 3]
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())


class TestIssuedRequestTaskName:
    """The task name the admission stamps on issued requests."""

    def test_issued_request_carries_task_name(self):
        """The admission stamps the configured task name on the request."""

        async def run():
            worker = _make_worker(rollout_batch_size=2, task_name="searchr1")
            dataloader_channel = _QueueChannel()
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                request = await asyncio.wait_for(
                    dataloader_channel._queue.get(), timeout=5
                )
                assert request.task_names == ["searchr1"]
                assert request.group_ids == [0]
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_task_name_and_limits_are_read_from_cfg(self):
        """The real __init__ reads agentloop.task_name and the buffer caps."""

        async def run():
            cfg = _cfg(
                rollout_batch_size=2,
                max_inflight_groups=3,
                max_buffered_groups=4,
                task_name="searchr1",
            )
            with mock.patch.object(Worker, "__init__", return_value=None):
                worker = RolloutBufferWorker(cfg, _ToyDataset(num_samples=8))
            assert worker.task_name == "searchr1"
            assert worker.max_actor_accept_staleness is None
            assert worker.buffer.max_inflight_groups == 3
            assert worker.buffer.max_buffered_groups == 4
            dataloader_channel = _QueueChannel()
            agent_output_channel = _QueueChannel()
            rollout_channel = _QueueChannel()
            resident = asyncio.create_task(
                worker.run_continuously(
                    agent_output_channel, dataloader_channel, rollout_channel
                )
            )
            try:
                request = await asyncio.wait_for(
                    dataloader_channel._queue.get(), timeout=5
                )
                assert request.task_names == ["searchr1"]
            finally:
                await worker.stop()
                await asyncio.wait_for(resident, timeout=5)

        asyncio.run(run())

    def test_max_actor_accept_staleness_is_read_from_agentloop(self):
        """The staleness cap moved from algorithm to agentloop."""
        cfg = _cfg(rollout_batch_size=2, max_actor_accept_staleness=3)
        with mock.patch.object(Worker, "__init__", return_value=None):
            worker = RolloutBufferWorker(cfg, _ToyDataset(num_samples=8))
        assert worker.max_actor_accept_staleness == 3
