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
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.schema.reasoning_requests import build_rollout_requests_from_batch
from rlinf.data.schema.reasoning_results import DiscardedGroup, GroupMetadata
from rlinf.scheduler import Channel, Worker
from rlinf.utils.down_sampling import rewards_all_equal
from rlinf.utils.runner_utils import local_mkdir_safe


class RolloutBuffer:
    """State and decisions of the buffered rollout pipeline, no channels.

    One admission coroutine issues prompts and one receive coroutine
    buffers finished groups; both run on the worker's event loop, so no
    method here needs locking. The admission unit is one query group (one
    prompt); `rollout_batch_size` groups form one training batch.

    Admission is by capacity, in two conditions: the number of in-flight
    groups (issued minus finished, tracked per task) stays below
    `max_inflight_groups`, and the number of in-flight plus buffered
    groups below `max_buffered_groups`. In-flight groups are counted
    because they enter the buffer on completion, so the buffer size never
    exceeds that limit. The issued count is incremented before the
    prompt is awaited onto the dataloader channel, so a prompt holds its
    slot from the moment it is issued; a result, a discard record, or a
    release to the actor returns capacity and wakes admission.

    Every group belongs to exactly one task (see `task_name`), and the
    per-group state is keyed by task name first. Only one task exists
    today; a release still takes the oldest buffered groups across all
    tasks, and per-task quotas or per-task batches are left for when a
    second task arrives.
    """

    def __init__(self, cfg: DictConfig, train_dataset: Dataset):
        self.rollout_batch_size = cfg.data.rollout_batch_size
        self.max_inflight_groups = cfg.rollout_buffer.max_inflight_groups
        self.max_buffered_groups = cfg.rollout_buffer.max_buffered_groups
        # Per-task state, keyed by task name; a task's dict is created at
        # its first issued group.
        self.tasks: dict[str, dict[str, Any]] = {}
        # Task name of every issued but not-yet-released group, so a
        # result, discard, or release finds the owning task without
        # trusting the producer's task stamp.
        self.group_tasks: dict[int, str] = {}
        # Counters drained by get_step_metrics(); the gauges (buffer
        # size, in-flight count) are read live when it is called.
        self._step_issued_groups = 0
        self._step_dropped_groups = 0
        self._step_dropped_by_reason: dict[str, int] = {}
        self._step_admission_wait: dict[str, float] = {
            "buffer_full": 0.0,
            "inflight_limit": 0.0,
        }
        # The wait currently in progress (start time and block reason),
        # None while admission is not suspended. get_step_metrics charges
        # it up to the read so a wait spanning several windows lands in
        # each of them instead of entirely in the one where it ended.
        self._ongoing_wait_start: Optional[float] = None
        self._ongoing_wait_reason: Optional[str] = None
        self._step_batch_issue_versions = 0
        self._step_batch_oldest_issue_version_ratio = 0.0
        self._metrics_window_start = time.monotonic()
        self._admission_event = asyncio.Event()
        self._admission_event.set()
        # Weight version each rollout instance has loaded; None before
        # the first sync. Release-time staleness checks run against the
        # minimum; admission no longer depends on it.
        self.loaded_versions: Optional[dict[int, int]] = None
        # Dataset samples this process has taken from the dataloader,
        # whether issued yet or not. A resume-relevant quantity, so it is
        # a process-local counter rather than dataloader state.
        self.num_samples_fetched = 0
        # Version of the release request waiting for a full batch.
        self.pending_release_version: Optional[int] = None
        self.next_group_id = 0
        self._build_dataloader(cfg, train_dataset)

    def _task(self, task_name: Optional[str]) -> dict[str, Any]:
        """The per-task state dict, created at the task's first issued
        group. A None key appears only when the buffer is constructed
        directly without the config validation (tests); async training
        requires a task name.
        """
        return self.tasks.setdefault(
            task_name,
            {
                # Query groups issued so far.
                "num_issued_groups": 0,
                # Groups dropped before training, by reason; both the
                # running totals and the counts since the last release.
                "num_dropped_groups": {},
                # Finished groups, whether later released or discarded.
                "num_finished_groups": 0,
                # Finished groups awaiting a release, keyed by group id
                # and held as (metadata, result): decisions read the
                # metadata, the release forwards the result untouched.
                "buffer": {},
                # Loaded version at issue time of every not-yet-released
                # group.
                "issue_versions": {},
                # Dataset sample index of each issued group, recorded at
                # issue time so a discard can count how often its prompt
                # was dropped.
                "group_sample_indices": {},
                # Dataset sample indices dropped twice or more so far.
                "repeatedly_dropped_samples": set(),
                # Drop counts per dataset sample index (kept while > 0).
                "sample_drop_counts": {},
                # Drops since the last release, by reason.
                "drops_since_release": {},
            },
        )

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
        """Record the version each rollout instance has loaded.

        Used by release-time staleness checks and the loaded-version
        metrics; admission no longer depends on the loaded versions.
        """
        self.loaded_versions = dict(versions)

    def _min_loaded_version(self) -> int:
        return min(self.loaded_versions.values())

    def min_loaded_version(self) -> Optional[int]:
        """The smallest version any rollout instance has loaded."""
        return self._min_loaded_version() if self.loaded_versions else None

    def _inflight_group_limit(self) -> int:
        """In-flight groups the rollout capacity allows right now.

        Returns the configured `max_inflight_groups` unchanged; a future
        per-task generate-occupancy estimate would adjust this per task,
        which is why admission and metrics both read it through here.
        """
        return self.max_inflight_groups

    def inflight_groups(self) -> int:
        """Issued groups whose results have not come back yet."""
        return sum(
            task["num_issued_groups"] - task["num_finished_groups"]
            for task in self.tasks.values()
        )

    def _buffered_groups(self) -> int:
        return sum(len(task["buffer"]) for task in self.tasks.values())

    def _admission_block_reason(self) -> Optional[str]:
        """Why admission is blocked right now, or None when it is not.

        Checked again after every wake-up: a wake does not imply a slot
        freed. Reports the in-flight limit first so a caller blocked by
        both learns the capacity reason. The second condition counts
        in-flight groups too: they enter the buffer on completion, so
        `buffer_full` means the buffer would be full once every in-flight
        group finishes.
        """
        if self.inflight_groups() >= self._inflight_group_limit():
            return "inflight_limit"
        if self.inflight_groups() + self._buffered_groups() >= (
            self.max_buffered_groups
        ):
            return "buffer_full"
        return None

    async def wait_for_admission_slot(self):
        """Suspend until both admission conditions hold.

        The blocked time is charged per reason to the current metrics
        window only: a wake-up accrues now − max(wait_start,
        window_start), and a wait still running when get_step_metrics is
        called is charged up to that read by get_step_metrics itself, so
        no window ever reports more wait than it lasted.
        """
        while (reason := self._admission_block_reason()) is not None:
            self._admission_event.clear()
            wait_start = time.monotonic()
            self._ongoing_wait_start = wait_start
            self._ongoing_wait_reason = reason
            await self._admission_event.wait()
            now = time.monotonic()
            self._ongoing_wait_start = None
            self._ongoing_wait_reason = None
            self._step_admission_wait[reason] += now - max(
                wait_start, self._metrics_window_start
            )

    def issue_group(
        self, sample_idx: int, task_name: Optional[str]
    ) -> tuple[int, Optional[int]]:
        """Reserve the next group id and its issue-time loaded version.

        The issued count is incremented here, before the caller awaits
        the prompt onto the dataloader channel: the slot is taken even
        while the prompt is still queued.
        """
        group_id = self.next_group_id
        self.next_group_id += 1
        task = self._task(task_name)
        task["num_issued_groups"] += 1
        self.group_tasks[group_id] = task_name
        self._step_issued_groups += 1
        task["group_sample_indices"][group_id] = sample_idx
        issue_version = self._min_loaded_version() if self.loaded_versions else None
        if issue_version is not None:
            task["issue_versions"][group_id] = issue_version
        return group_id, issue_version

    def record_result(self, result: Any, metadata: GroupMetadata) -> int:
        """Buffer one finished group and return its group id.

        `metadata` is the decision view of `result`, computed once by the
        caller at receive time; buffer decisions never touch the result
        body again. The owning task is the one the group was issued
        under, recorded in `group_tasks`; a task name the producer
        carries that differs from it is ignored. A result whose group id
        is unknown (or None, as merged and split results carry) cannot
        return capacity to its task: the caller must treat that as a
        producer bug, so the buffer raises instead of returning a
        sentinel.
        """
        group_id = result.group_id
        if group_id is None:
            raise ValueError(
                "RolloutBuffer got a result without a group id. Merged or "
                "split results span several groups and cannot be buffered; "
                "only single-group results from the agent loop are routable."
            )
        try:
            task_name = self.group_tasks.pop(group_id)
        except KeyError:
            raise ValueError(
                f"RolloutBuffer got a result for unknown group id "
                f"{group_id}: the group was never issued here or has "
                "already returned. This is a producer bug."
            ) from None
        task = self._task(task_name)
        task["buffer"][group_id] = (metadata, result)
        task["num_finished_groups"] += 1
        self._admission_event.set()
        return group_id

    def drop_group(
        self,
        group_id: int,
        reason: str,
        max_dropped: int,
        already_buffered: bool,
        task_name: Optional[str] = None,
    ):
        """Drop one finished group: return its admission capacity and count it.

        The dropped group's data is released, never re-issued; its prompt
        comes around again in a later epoch. `already_buffered` marks a
        drop decided at release time, after `record_result` already
        counted the group as finished; that caller knows the owning task
        and passes it, while a receive-time drop looks it up from the
        issue record. Raises once the drops since the last release
        exceed `max_dropped`, which ends the resident method and the
        job.
        """
        if not already_buffered:
            try:
                task_name = self.group_tasks.pop(group_id)
            except KeyError:
                raise ValueError(
                    f"RolloutBuffer got a discard for unknown group id "
                    f"{group_id}: the group was never issued here or has "
                    "already returned. This is a producer bug."
                ) from None
        task = self._task(task_name)
        if already_buffered:
            task["buffer"].pop(group_id, None)
        else:
            task["num_finished_groups"] += 1
        num_dropped = task["num_dropped_groups"]
        num_dropped[reason] = num_dropped.get(reason, 0) + 1
        drops = task["drops_since_release"]
        drops[reason] = drops.get(reason, 0) + 1
        self._step_dropped_groups += 1
        self._step_dropped_by_reason[reason] = (
            self._step_dropped_by_reason.get(reason, 0) + 1
        )
        task["issue_versions"].pop(group_id, None)
        sample_idx = task["group_sample_indices"].pop(group_id, None)
        if sample_idx is not None:
            counts = task["sample_drop_counts"]
            count = counts.get(sample_idx, 0) + 1
            counts[sample_idx] = count
            if count >= 2:
                task["repeatedly_dropped_samples"].add(sample_idx)
        self._admission_event.set()
        total_dropped = sum(drops.values())
        if total_dropped > max_dropped:
            counts = ", ".join(
                f"{reason}: {count}" for reason, count in sorted(drops.items())
            )
            raise RuntimeError(
                f"RolloutBuffer dropped {total_dropped} groups since the "
                f"last release, over the limit of {max_dropped} "
                f"({counts})."
            )

    def can_release(self) -> bool:
        return self._buffered_groups() >= self.rollout_batch_size

    def stale_group_ids(
        self, task: dict[str, Any], version: int, max_staleness: int
    ) -> list[int]:
        """Buffered groups too stale for the training version `version`.

        A group is too stale when `version - earliest generate version`
        exceeds `max_staleness`. The check runs here, at release time in
        the RolloutBufferWorker, rather than in the actor: dropping here
        returns the admission capacity so a replacement group is issued
        while the actor still trains.
        """
        stale = []
        for group_id, (metadata, _) in task["buffer"].items():
            earliest = metadata.earliest_version
            if earliest is not None and version - earliest > max_staleness:
                stale.append(group_id)
        return stale

    def release_groups(self) -> list[Any]:
        """Take the oldest `rollout_batch_size` groups out of the buffer.

        Also clears the drops-since-release window of every task: the
        caller counts drops between consecutive releases.
        """
        buffered = sorted(
            (group_id, task_name)
            for task_name, task in self.tasks.items()
            for group_id in task["buffer"]
        )[: self.rollout_batch_size]
        results = []
        issue_versions = []
        for group_id, task_name in buffered:
            task = self.tasks[task_name]
            results.append(task["buffer"].pop(group_id)[1])
            task["group_sample_indices"].pop(group_id, None)
            issue_versions.append(task["issue_versions"].pop(group_id, None))
        known = [version for version in issue_versions if version is not None]
        if known:
            self._step_batch_issue_versions = len(set(known))
            self._step_batch_oldest_issue_version_ratio = known.count(min(known)) / len(
                known
            )
        for task in self.tasks.values():
            task["drops_since_release"] = {}
        # Handing groups to the actor freed buffer capacity; admission
        # may issue replacements while the actor trains.
        self._admission_event.set()
        return results

    def get_step_metrics(self) -> dict:
        """Drain the per-step counters and read the gauges.

        Called once per training step; the window between calls is the
        step duration the admission-wait ratios are taken against. A
        wait still in progress is charged up to this read (the next
        window then accrues from here), so a long wait spanning several
        steps lands in each of them and no ratio exceeds 1.
        """
        now = time.monotonic()
        if self._ongoing_wait_start is not None:
            self._step_admission_wait[self._ongoing_wait_reason] += now - max(
                self._ongoing_wait_start, self._metrics_window_start
            )
            self._ongoing_wait_start = now
        window = now - self._metrics_window_start
        metrics = {
            "size": self._buffered_groups(),
            "issued_groups": self._step_issued_groups,
            "dropped_groups": self._step_dropped_groups,
            "inflight_groups": self.inflight_groups(),
            "inflight_group_limit": self._inflight_group_limit(),
            "batch_issue_versions": self._step_batch_issue_versions,
            "batch_oldest_issue_version_ratio": self._step_batch_oldest_issue_version_ratio,
            "admission_wait_buffer_full": (
                self._step_admission_wait["buffer_full"] / window if window > 0 else 0.0
            ),
            "admission_wait_inflight_limit": (
                self._step_admission_wait["inflight_limit"] / window
                if window > 0
                else 0.0
            ),
            "repeatedly_dropped_prompts": sum(
                len(task["repeatedly_dropped_samples"]) for task in self.tasks.values()
            ),
        }
        for reason, count in self._step_dropped_by_reason.items():
            metrics[f"dropped_{reason}"] = count
        self._step_issued_groups = 0
        self._step_dropped_groups = 0
        self._step_dropped_by_reason = {}
        self._step_admission_wait = {
            "buffer_full": 0.0,
            "inflight_limit": 0.0,
        }
        self._step_batch_issue_versions = 0
        self._step_batch_oldest_issue_version_ratio = 0.0
        self._metrics_window_start = now
        return metrics


class RolloutBufferWorker(Worker):
    """Runs on the driver's node.
    Finished groups currently pass through this worker on their way to the
    actor; keeping the tensors on the rollout side and holding only their
    metadata here is not implemented. The dataloader, the admission
    capacity and the buffer state live in this worker, so a checkpoint
    counts as complete only after its `data/data.pt` (written last)
    exists.
    """

    def __init__(self, cfg: DictConfig, train_dataset: Dataset):
        super().__init__()
        self.cfg = cfg
        self.buffer = RolloutBuffer(cfg, train_dataset)
        down_sampling_cfg = cfg.algorithm.get("down_sampling", None) or {}
        down_sampling_config = down_sampling_cfg.get("down_sampling_config", None) or {}
        self.reject_equal_reward = bool(
            down_sampling_cfg.get("do_down_sampling", False)
            and down_sampling_config.get("reject_equal_reward", False)
        )
        self.max_actor_accept_staleness = OmegaConf.select(
            cfg, "agentloop.max_actor_accept_staleness", default=None
        )
        self.max_dropped_groups_per_step = (
            cfg.rollout_buffer.max_dropped_groups_per_step
        )
        # Agent task name stamped on every issued request; the buffer keys
        # its per-task state by it. Required in async training.
        self.task_name = OmegaConf.select(cfg, "agentloop.task_name", default=None)
        self._resident_task: Optional[asyncio.Task] = None
        self._data_iter = None
        self._rollout_channel: Optional[Channel] = None
        self._pending_fetch: Optional[asyncio.Future] = None

    async def run_continuously(
        self,
        agent_output_channel: Channel,
        dataloader_channel: Channel,
        rollout_channel: Channel,
    ):
        """Admit prompts and buffer finished groups until `stop()`.

        Two coroutines: admission issues prompts from the dataloader onto
        `dataloader_channel` as the capacity allows; receive buffers
        results from `agent_output_channel` and dispatches a pending
        release onto `rollout_channel`. A failure of either ends this
        method; a stop cancels both and returns normally.
        """
        assert self._resident_task is None, "the resident method is already running."
        self._resident_task = asyncio.current_task()
        self._rollout_channel = rollout_channel
        admission = asyncio.create_task(self._run_admission(dataloader_channel))
        receive = asyncio.create_task(
            self._run_receive(agent_output_channel, rollout_channel)
        )
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
            sample_indices = batch["idx"].tolist()
            self.buffer.num_samples_fetched += len(sample_indices)
            # One request carries one group (one prompt): the agent loop
            # starts one task per request and groups complete
            # independently, so a finished group can be buffered as soon
            # as it alone is done.
            for request, sample_idx in zip(
                build_rollout_requests_from_batch(
                    batch,
                    group_size=self.cfg.algorithm.group_size,
                    split_size=len(batch["prompt"]),
                ),
                sample_indices,
            ):
                await self.buffer.wait_for_admission_slot()
                group_id, _ = self.buffer.issue_group(sample_idx, self.task_name)
                request.group_ids = [group_id]
                request.task_names = [self.task_name]
                await dataloader_channel.put(request, async_op=True).async_wait()

    async def _run_receive(
        self, agent_output_channel: Channel, rollout_channel: Channel
    ):
        # Group filtering runs here rather than in RewardWorker so that a
        # dropped group returns its admission capacity and gets replaced.
        while True:
            result = await agent_output_channel.get(async_op=True).async_wait()
            if isinstance(result, DiscardedGroup):
                self.buffer.drop_group(
                    result.group_id,
                    result.reason,
                    self.max_dropped_groups_per_step,
                    already_buffered=False,
                )
                continue
            metadata = result.group_metadata()
            if self.reject_equal_reward and rewards_all_equal(metadata.rewards):
                self.buffer.drop_group(
                    metadata.group_id,
                    "equal_reward",
                    self.max_dropped_groups_per_step,
                    already_buffered=False,
                )
                continue
            if (
                self.max_actor_accept_staleness is not None
                and metadata.earliest_version is not None
                and self.buffer.min_loaded_version() is not None
                and self.buffer.min_loaded_version() - metadata.earliest_version
                > self.max_actor_accept_staleness
            ):
                # First pass at receive time against the current loaded
                # version; release_batch re-checks against the version the
                # training batch actually runs under, which is the
                # authoritative one.
                self.buffer.drop_group(
                    metadata.group_id,
                    "stale",
                    self.max_dropped_groups_per_step,
                    already_buffered=False,
                )
                continue
            self.buffer.record_result(result, metadata)
            if (
                self.buffer.pending_release_version is not None
                and self.buffer.can_release()
            ):
                await self._dispatch_release(rollout_channel)

    async def _dispatch_release(self, rollout_channel: Channel):
        version = self.buffer.pending_release_version
        self.buffer.pending_release_version = None
        if self.max_actor_accept_staleness is not None:
            # Release-time check against the version the training batch
            # actually runs under; the receive-time check only prunes
            # early. The check runs here, not in the actor: dropping here
            # returns the admission capacity so a replacement is issued
            # while the actor still trains.
            for task_name, task in list(self.buffer.tasks.items()):
                for group_id in self.buffer.stale_group_ids(
                    task, version, self.max_actor_accept_staleness
                ):
                    self.buffer.drop_group(
                        group_id,
                        "stale",
                        self.max_dropped_groups_per_step,
                        already_buffered=True,
                        task_name=task_name,
                    )
        if not self.buffer.can_release():
            # Stale drops emptied the batch; keep the request pending and
            # let the receive coroutine dispatch once replacements arrive.
            self.buffer.pending_release_version = version
            return
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

    async def get_metrics(self) -> dict:
        return self.buffer.get_step_metrics()

    async def save_checkpoint(self, path: str):
        """Write the dataloader state to `path/data.pt` atomically."""
        if self._pending_fetch is not None:
            await self._pending_fetch
        state_dict = self.buffer.dataloader.state_dict()
        dropped = sum(
            sum(task["num_dropped_groups"].values())
            for task in self.buffer.tasks.values()
        )
        finished = sum(
            task["num_finished_groups"] for task in self.buffer.tasks.values()
        )
        released = finished - dropped - self.buffer._buffered_groups()
        # The dataloader runs ahead of training: a resume restores the
        # weight checkpoint, so everything this process took beyond the
        # released groups is skipped (in flight, buffered, or fetched
        # but not yet issued). Dropped groups are subtracted: their
        # prompts are never re-issued, resume or not.
        skipped = self.buffer.num_samples_fetched - released - dropped
        self.log_info(
            f"RolloutBuffer checkpoint: fetched "
            f"{self.buffer.num_samples_fetched} samples, {released} "
            f"groups released to training, {dropped} dropped; a resume "
            f"will skip {skipped} prompts that were issued ahead of the "
            f"resume point."
        )
        local_mkdir_safe(path)
        local_path = os.path.join(path, "data.pt")
        tmp_path = f"{local_path}.tmp"
        try:
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, local_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    async def load_checkpoint(self, path: str, start_step: int):
        """Load the dataloader state; `start_step` is accepted for the
        synchronous runner's checkpoint interface and unused here (admission
        no longer depends on the training step).

        Only the dataloader state depends on `data.pt` existing; without
        it the buffer starts from the beginning of the dataset.
        """
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
