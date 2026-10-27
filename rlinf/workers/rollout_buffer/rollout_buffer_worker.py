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
import os
import time
from typing import Any, Optional

import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.schema.reasoning_requests import build_rollout_requests_from_batch
from rlinf.scheduler import Channel, Worker
from rlinf.utils.runner_utils import local_mkdir_safe


class RolloutBuffer:
    """State and decisions of the buffered rollout pipeline, no channels.

    One admission coroutine issues prompts and one receive coroutine
    buffers finished groups; both run on the worker's event loop, so no
    method here needs locking. The admission unit is one query group (one
    prompt); `rollout_batch_size` groups form one training batch.

    The admission budget allows issuing while
    `issued < (staleness_threshold + (min(loaded versions) -
    start_step) + 1) * rollout_batch_size`. The issued count is
    incremented before the prompt is awaited onto the dataloader channel,
    so a prompt holds its slot from the moment it is issued.
    """

    def __init__(self, cfg: DictConfig, train_dataset: Dataset):
        self.rollout_batch_size = cfg.data.rollout_batch_size
        self.staleness_threshold = cfg.algorithm.staleness_threshold
        self.start_step = 0
        # Query groups issued since start_step.
        self.num_issued_groups = 0
        # Finished groups, whether later released or discarded.
        self.num_finished_groups = 0
        # Weight version each rollout instance has loaded; None before
        # the first sync. Prompts are admitted against the minimum.
        self.loaded_versions: Optional[dict[int, int]] = None
        # Finished groups awaiting a release, keyed by group id.
        self.buffer: dict[int, Any] = {}
        # Loaded version at issue time of every not-yet-released group.
        self.issue_versions: dict[int, int] = {}
        # Version of the release request waiting for a full batch.
        self.pending_release_version: Optional[int] = None
        self.next_group_id = 0
        self._admission_event = asyncio.Event()
        self._admission_event.set()
        # Counters drained by get_step_metrics(); the gauges (buffer
        # size, in-flight count) are read live when it is called.
        self._step_issued_groups = 0
        self._step_admission_wait_seconds = 0.0
        self._step_batch_issue_versions = 0
        self._step_batch_oldest_issue_version_ratio = 0.0
        self._metrics_window_start = time.monotonic()
        self._build_dataloader(cfg, train_dataset)

    def _build_dataloader(self, cfg: DictConfig, train_dataset: Dataset):
        """Create the dataloader behind the buffer.

        Mirrors `ReasoningRunner._build_dataloader` (same sampler, seed,
        batch size and collate) so the checkpoint states stay
        interchangeable with the synchronous runner's.
        """
        from rlinf.data.datasets.reasoning import collate_fn

        if cfg.data.shuffle:
            generator = torch.Generator()
            generator.manual_seed(cfg.data.get("seed", 1))
            sampler = RandomSampler(data_source=train_dataset, generator=generator)
        else:
            sampler = SequentialSampler(data_source=train_dataset)

        self.dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=cfg.data.rollout_batch_size
            * cfg.algorithm.get("max_num_gen_batches", 1),
            num_workers=cfg.data.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )
        assert len(self.dataloader) >= 1, "Train dataloader is empty!"

    def set_loaded_versions(self, versions: dict[int, int]):
        """Record the version each rollout instance has loaded."""
        self.loaded_versions = dict(versions)
        self._admission_event.set()

    def set_staleness_threshold(self, value: int):
        """Update the threshold; lowering it affects later admissions only."""
        self.staleness_threshold = value
        self._admission_event.set()

    def _min_loaded_version(self) -> int:
        return min(self.loaded_versions.values())

    def _admission_budget(self) -> int:
        """Query groups that may be issued-but-unreleased right now."""
        quota = self.staleness_threshold + 1
        if self.loaded_versions is not None:
            quota += self._min_loaded_version() - self.start_step
        return quota * self.rollout_batch_size

    def _prompts_to_issue(self) -> int:
        return self._admission_budget() - self.num_issued_groups

    async def wait_for_admission_slot(self):
        """Suspend until the budget has a free slot; accumulate wait time.

        Wakers that do not free a slot (a lower threshold, a version bump
        that only some instances loaded) are handled by re-checking the
        budget in a loop after each wake-up.
        """
        while self._prompts_to_issue() < 1:
            self._admission_event.clear()
            wait_start = time.monotonic()
            await self._admission_event.wait()
            self._step_admission_wait_seconds += time.monotonic() - wait_start

    def issue_group(self) -> tuple[int, Optional[int]]:
        """Reserve the next group id and its issue-time loaded version.

        The issued count is incremented here, before the caller awaits
        the prompt onto the dataloader channel: the slot is taken even
        while the prompt is still queued.
        """
        group_id = self.next_group_id
        self.next_group_id += 1
        self.num_issued_groups += 1
        self._step_issued_groups += 1
        issue_version = self._min_loaded_version() if self.loaded_versions else None
        if issue_version is not None:
            self.issue_versions[group_id] = issue_version
        return group_id, issue_version

    def record_result(self, result: Any) -> int:
        """Buffer one finished group and return its group id.

        A result without a group id is not routable to a slot of the
        admission budget: merged and split results span several groups.
        The caller must treat that as a producer bug, so the buffer
        raises instead of returning a sentinel.
        """
        group_id = result.group_id
        if group_id is None:
            raise ValueError(
                "RolloutBuffer got a result without a group id. Merged or "
                "split results span several groups and cannot be buffered; "
                "only single-group results from the agent loop are routable."
            )
        self.buffer[group_id] = result
        self.num_finished_groups += 1
        return group_id

    def inflight_groups(self) -> int:
        """Issued groups whose results have not come back yet."""
        return self.num_issued_groups - self.num_finished_groups

    def can_release(self) -> bool:
        return len(self.buffer) >= self.rollout_batch_size

    def release_groups(self) -> list[Any]:
        """Take the oldest `rollout_batch_size` groups out of the buffer."""
        group_ids = sorted(self.buffer)[: self.rollout_batch_size]
        results = [self.buffer.pop(group_id) for group_id in group_ids]
        issue_versions = [
            self.issue_versions.pop(group_id, None) for group_id in group_ids
        ]
        known = [version for version in issue_versions if version is not None]
        if known:
            self._step_batch_issue_versions = len(set(known))
            self._step_batch_oldest_issue_version_ratio = known.count(min(known)) / len(
                known
            )
        return results

    def get_step_metrics(self) -> dict:
        """Drain the per-step counters and read the gauges.

        Called once per training step; the window between calls is the
        step duration the admission-wait ratio is taken against.
        """
        now = time.monotonic()
        window = now - self._metrics_window_start
        wait_seconds = self._step_admission_wait_seconds
        metrics = {
            "size": len(self.buffer),
            "issued_groups": self._step_issued_groups,
            "inflight_groups": self.inflight_groups(),
            "batch_issue_versions": self._step_batch_issue_versions,
            "batch_oldest_issue_version_ratio": self._step_batch_oldest_issue_version_ratio,
            "admission_wait_ratio": wait_seconds / window if window > 0 else 0.0,
        }
        self._step_issued_groups = 0
        self._step_admission_wait_seconds = 0.0
        self._step_batch_issue_versions = 0
        self._step_batch_oldest_issue_version_ratio = 0.0
        self._metrics_window_start = now
        return metrics


class RolloutBufferWorker(Worker):
    """Runs on the driver's node.
    Finished groups currently pass through this worker on their way to the
    actor; keeping the tensors on the rollout side and holding only their
    metadata here is not implemented. The dataloader, the admission budget
    and the buffer state live in this worker, so a checkpoint counts as
    complete only after its `data/data.pt` (written last) exists.
    """

    def __init__(self, cfg: DictConfig, train_dataset: Dataset):
        super().__init__()
        self.cfg = cfg
        self.buffer = RolloutBuffer(cfg, train_dataset)
        self._resident_task: Optional[asyncio.Task] = None
        self._data_iter = None
        self._rollout_channel: Optional[Channel] = None
        self._pending_fetch: Optional[asyncio.Future] = None

    async def run_continuously(
        self,
        input_channel: Channel,
        dataloader_channel: Channel,
        rollout_channel: Channel,
    ):
        """Admit prompts and buffer finished groups until `stop()`.

        Two coroutines: admission issues prompts from the dataloader onto
        `dataloader_channel` as the budget allows; receive buffers results
        from `input_channel` and dispatches a pending release onto
        `rollout_channel`. A failure of either ends this method; a stop
        cancels both and returns normally.
        """
        assert self._resident_task is None, "the resident method is already running."
        self._resident_task = asyncio.current_task()
        self._rollout_channel = rollout_channel
        admission = asyncio.create_task(self._run_admission(dataloader_channel))
        receive = asyncio.create_task(self._run_receive(input_channel, rollout_channel))
        try:
            await asyncio.gather(admission, receive)
        except asyncio.CancelledError:
            # stop(): in-flight prompts are dropped, the run is ending.
            return
        finally:
            self._resident_task = None
            for task in (admission, receive):
                task.cancel()
            await asyncio.gather(admission, receive, return_exceptions=True)

    async def _next_batch(self) -> Any:
        """Pull the next dataloader batch without blocking the event loop.

        A cold `next()` can block on disk even with prefetching workers;
        blocking the loop would stall receive and the RPCs sharing it.
        Restarts the dataloader at the end of an epoch.
        """
        if self._data_iter is None:
            self._data_iter = iter(self.buffer.dataloader)

        def _next():
            return next(self._data_iter, None)

        batch = await self._run_fetch(_next)
        if batch is None:
            self._data_iter = iter(self.buffer.dataloader)
            batch = await self._run_fetch(_next)
        return batch

    async def _run_fetch(self, fetch) -> Any:
        """Run one dataloader access in the executor and record the future.

        `save_checkpoint` awaits the recorded future first: a
        `state_dict()` taken while a `next()` is still running in the
        thread pool would race with it (StatefulDataLoader is not
        thread-safe).
        """
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, fetch)
        self._pending_fetch = future
        try:
            return await future
        finally:
            if self._pending_fetch is future:
                self._pending_fetch = None

    async def _run_admission(self, dataloader_channel: Channel):
        while True:
            batch = await self._next_batch()
            # One request carries one group (one prompt): the agent loop
            # starts one task per request and groups complete
            # independently, so a finished group can be buffered as soon
            # as it alone is done.
            for request in build_rollout_requests_from_batch(
                batch,
                group_size=self.cfg.algorithm.group_size,
                split_size=len(batch["prompt"]),
            ):
                await self.buffer.wait_for_admission_slot()
                group_id, _ = self.buffer.issue_group()
                request.group_ids = [group_id]
                await dataloader_channel.put(request, async_op=True).async_wait()

    async def _run_receive(self, input_channel: Channel, rollout_channel: Channel):
        while True:
            result = await input_channel.get(async_op=True).async_wait()
            self.buffer.record_result(result)
            if (
                self.buffer.pending_release_version is not None
                and self.buffer.can_release()
            ):
                await self._dispatch_release(rollout_channel)

    async def _dispatch_release(self, rollout_channel: Channel):
        self.buffer.pending_release_version = None
        for result in self.buffer.release_groups():
            await rollout_channel.put(result, async_op=True).async_wait()

    async def release_batch(self, version: int):
        """Request a training batch of the oldest buffered groups.

        If the buffer already holds a full batch it is dispatched from
        here; otherwise the request is held and dispatched by the receive
        coroutine once enough groups have arrived.
        """
        assert self.buffer.pending_release_version is None, (
            "a release request is already pending; the runner must wait "
            "for the actor to consume the previous batch before releasing "
            "the next one."
        )
        assert self._rollout_channel is not None, (
            "release_batch was called before run_continuously wired the "
            "rollout channel."
        )
        self.buffer.pending_release_version = version
        if self.buffer.can_release():
            await self._dispatch_release(self._rollout_channel)

    async def set_loaded_versions(self, versions: dict[int, int]):
        self.buffer.set_loaded_versions(versions)

    async def set_staleness_threshold(self, value: int):
        self.buffer.set_staleness_threshold(value)

    async def get_metrics(self) -> dict:
        return self.buffer.get_step_metrics()

    async def save_checkpoint(self, path: str):
        """Write the dataloader state to `path/data.pt` atomically."""
        if self._pending_fetch is not None:
            await self._pending_fetch
        local_mkdir_safe(path)
        local_path = os.path.join(path, "data.pt")
        tmp_path = f"{local_path}.tmp"
        try:
            torch.save(self.buffer.dataloader.state_dict(), tmp_path)
            os.replace(tmp_path, local_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    async def load_checkpoint(self, path: str, start_step: int):
        """Load the dataloader state and set the admission start step.

        `start_step` is set unconditionally: the loaded weight version is
        the resume point, so without it the budget would admit `start_step`
        extra batches. Only the dataloader state depends on `data.pt`
        existing; without it the buffer starts from the beginning of the
        dataset.
        """
        self.buffer.start_step = start_step
        local_path = os.path.join(path, "data.pt")
        if not os.path.exists(local_path):
            self.log_warning(
                f"No dataloader state found at {local_path}; the buffer "
                f"starts from the beginning of the dataset."
            )
            return
        state_dict = torch.load(local_path, weights_only=False)
        self.buffer.dataloader.load_state_dict(state_dict)

    async def stop(self):
        """Cancel the resident coroutines; in-flight prompts are dropped."""
        if self._resident_task is not None and not self._resident_task.done():
            self._resident_task.cancel()
