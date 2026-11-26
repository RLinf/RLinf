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

"""Checkpoint unconsumed RPent episodes without changing buffer admission rules."""

import asyncio
import os
from collections import deque
from dataclasses import replace
from pathlib import Path

import torch

from rlinf.data.schema.reasoning_requests import build_rollout_requests_from_batch
from rlinf.data.schema.reasoning_results import (
    DiscardedGroup,
    DynamicRolloutResult,
    GroupMetadata,
)
from rlinf.scheduler import Channel
from rlinf.workers.rollout.session_server.assembly import assemble_group_payload
from rlinf.workers.rollout.session_server.payload import payload_channel_for
from rlinf.workers.rollout_buffer.rollout_buffer_worker import RolloutBufferWorker


def fold_planner_turns(result: DynamicRolloutResult, max_scored_tokens: int):
    """Fold same-version actions while captured image history and token prefix agree."""
    has_versions = result.version_start is not None and result.version_end is not None
    groups = []
    scored = 0
    for i, length in enumerate(result.response_lengths):
        image = result.multi_modal_inputs[i] if result.multi_modal_inputs else None
        prev = result.input_ids[i - 1] if i else []
        same_version = not has_versions or (
            result.version_start[i]
            == result.version_end[i]
            == result.version_start[i - 1]
            == result.version_end[i - 1]
        )
        if (
            i
            and result.idx_to_traj[i] == result.idx_to_traj[i - 1]
            and same_version
            and image
            and image.get("extends_previous_turn")
            and len(prev) <= result.prompt_lengths[i]
            and result.input_ids[i][: len(prev)] == prev
            and scored + length <= max_scored_tokens
        ):
            groups[-1].append(i)
        else:
            groups.append([i])
            scored = 0
        scored += length
    if len(groups) == result.num_sequence:
        return result

    token_versions = (result.extra_fields_turn or {}).get("rollout_weight_versions")
    masks, logprobs, versions = [], [], []
    for group in groups:
        start = result.prompt_lengths[group[0]]
        mask = [False] * (len(result.input_ids[group[-1]]) - start)
        scores = [0.0] * len(mask)
        sampled = []
        for i in group:
            offset = result.prompt_lengths[i] - start
            end = offset + result.response_lengths[i]
            mask[offset:end] = [True] * result.response_lengths[i]
            if result.rollout_logprobs is not None:
                scores[offset:end] = result.rollout_logprobs[i]
            if token_versions is not None:
                sampled.extend(token_versions[i])
        masks.append(mask)
        logprobs.append(scores)
        versions.append(sampled)

    def last(rows):
        return [rows[group[-1]] for group in groups]

    return replace(
        result,
        num_sequence=len(groups),
        idx_to_traj=last(result.idx_to_traj),
        input_ids=last(result.input_ids),
        prompt_lengths=[result.prompt_lengths[g[0]] for g in groups],
        response_lengths=[len(mask) for mask in masks],
        response_mask=masks,
        rollout_logprobs=logprobs if result.rollout_logprobs is not None else None,
        multi_modal_inputs=(
            last(result.multi_modal_inputs)
            if result.multi_modal_inputs is not None
            else None
        ),
        rewards=last(result.rewards) if result.rewards is not None else None,
        is_end=last(result.is_end),
        version_start=(
            [result.version_start[g[0]] for g in groups]
            if result.version_start is not None
            else None
        ),
        version_end=(
            last(result.version_end) if result.version_end is not None else None
        ),
        extra_fields_turn={
            **(
                {"rollout_weight_versions": versions}
                if token_versions is not None
                else {}
            ),
            "generate_calls": [len(g) for g in groups],
        },
        extra_fields_train={
            key: last(rows) for key, rows in result.extra_fields_train.items()
        },
    )


class RPentRolloutBufferWorker(RolloutBufferWorker):
    """Keep completed groups and replay unfinished episodes in a fresh environment.

    Model checkpoints are saved before this worker's state. Already released groups
    belong to that update; fetched and issued groups remain available after resume.
    Simulator processes and partially generated episodes are never serialized.
    """

    def __init__(self, cfg, train_dataset):
        width = cfg.data.sampling_batch_size if cfg.runner.synchronous else None
        super().__init__(cfg, train_dataset, data_batch_size=width)
        self._fetch_lock = asyncio.Lock()
        self._queued_requests = deque()
        self._issued_requests = {}
        self._replay_requests = deque()

    async def _prepare_result(self, result, metadata):
        if metadata.payload_ref is None:
            return result, metadata
        ref = metadata.payload_ref
        channel = payload_channel_for(ref.locator, self._payload_channels)
        payload = await channel.get(key=ref.key, async_op=True).async_wait()
        result = assemble_group_payload(payload, self.cfg.rollout.return_logprobs)
        result.group_id, result.task_name = metadata.group_id, metadata.task_name
        # In the RPent protocol is_end describes a normal model stop (length
        # truncation stays false), independently of the native terminal reward.
        trained_parts = {
            (part.traj_index, part.turn_index): part
            for part in payload.parts
            if not part.last
        }
        verdicts = payload.parts[-1].verdicts
        result.is_end = [
            part.finish_reason.get("type") == "stop"
            for (traj, turn), part in sorted(trained_parts.items())
            if turn not in verdicts[traj]["not_training"]
        ]

        result = fold_planner_turns(
            result, self.cfg.algorithm.sampling_params.max_new_tokens
        )

        # One planner acts throughout each trajectory, including after history rewrites.
        if "agent_level" in self.cfg.algorithm.loss_scales:
            result.extra_fields_train["idx_to_sub_traj"] = [0] * result.num_sequence
        await channel.remove_key(key=ref.key, async_op=True).async_wait()
        return result, replace(metadata, payload_ref=None)

    async def _next_request(self):
        if not self._queued_requests:
            async with self._fetch_lock:
                batch = await self._next_batch()
                indices = batch["idx"].tolist()
                self.buffer.num_samples_fetched += len(indices)
                requests = build_rollout_requests_from_batch(
                    batch,
                    group_size=self.cfg.algorithm.group_size,
                    split_size=len(indices),
                )
                self._queued_requests.extend(zip(requests, indices, strict=True))
        request, sample_idx = self._queued_requests.popleft()
        group_id, _ = self.buffer.issue_group(sample_idx, self.task_name)
        request.group_ids = [group_id]
        request.task_names = [self.task_name]
        self._issued_requests[group_id] = request
        return request

    async def _run_admission(self, dataloader_channel: Channel):
        while True:
            if self._replay_requests:
                request = self._replay_requests.popleft()
            else:
                await self.buffer.wait_for_admission_slot()
                request = await self._next_request()
            self._issued_requests = {
                gid: item
                for gid, item in self._issued_requests.items()
                if gid in self.buffer.group_tasks
            }
            await dataloader_channel.put(request, async_op=True).async_wait()

    async def save_checkpoint(self, path: str):
        """Save the dataloader, completed groups and not-yet-consumed requests."""
        async with self._fetch_lock:
            if self.buffer.pending_release_version is not None:
                raise RuntimeError(
                    "Checkpoint requires the previous training batch to be consumed"
                )
            completed = {
                gid for task in self.buffer.tasks.values() for gid in task["buffer"]
            }
            pending = {
                gid: self._issued_requests[gid]
                for gid in self.buffer.group_tasks
                if gid not in completed
            }
            state = {
                "format": "rpent-episode-buffer-v1",
                "group_size": self.cfg.algorithm.group_size,
                "task_name": self.task_name,
                "dataloader": self.buffer.dataloader.state_dict(),
                "dataloader_batch_size": self.buffer.dataloader.batch_size,
                "tasks": self.buffer.tasks,
                "group_tasks": self.buffer.group_tasks,
                "next_group_id": self.buffer.next_group_id,
                "num_samples_fetched": self.buffer.num_samples_fetched,
                "queued_requests": list(self._queued_requests),
                "pending_requests": pending,
            }
            directory = Path(path)
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "data.pt"
            temporary = target.with_suffix(".pt.tmp")
            try:
                torch.save(state, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

    async def load_checkpoint(self, path: str, start_step: int):
        """Restore completed groups once; restart unfinished episodes with the same IDs."""
        checkpoint_file = Path(path) / "data.pt"
        if not checkpoint_file.is_file():
            self.log_warning(
                f"No RPent rollout buffer state at {checkpoint_file}; starting with an empty buffer."
            )
            return
        state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if state.get("format") != "rpent-episode-buffer-v1":
            raise ValueError(
                "Expected an RPent episode-buffer checkpoint from this branch"
            )
        if (state["group_size"], state["task_name"]) != (
            self.cfg.algorithm.group_size,
            self.task_name,
        ):
            raise ValueError(
                "Checkpoint group size and task must match the resumed run"
            )
        completed = sum(len(task["buffer"]) for task in state["tasks"].values())
        if completed + len(state["pending_requests"]) > self.buffer.max_buffered_groups:
            raise ValueError(
                "Restored groups exceed rollout_buffer.max_buffered_groups"
            )
        if len(state["pending_requests"]) > self.buffer.max_inflight_groups:
            raise ValueError(
                "Restored pending episodes exceed rollout_buffer.max_inflight_groups"
            )
        loader = self.buffer.dataloader
        legacy_width = self.cfg.data.rollout_batch_size * self.cfg.algorithm.get(
            "max_num_gen_batches", 1
        )
        saved_width = state.get("dataloader_batch_size", legacy_width)
        if saved_width != loader.batch_size:
            # A saved sampler cursor counts batches of its original width.
            self.buffer._build_dataloader(
                self.cfg, loader.dataset, data_batch_size=saved_width
            )
            loader = self.buffer.dataloader
        if self.cfg.data.shuffle:
            # Torch's sampler stores no generator state in StatefulDataLoader.
            # Recreate completed shuffles before restoring the within-epoch cursor.
            samples_per_epoch = len(loader) * loader.batch_size
            epochs = max(0, (state["num_samples_fetched"] - 1) // samples_per_epoch)
            for _ in range(epochs):
                for _ in loader.sampler:
                    pass
        loader.load_state_dict(state["dataloader"])
        self._data_iter = None
        for field in ("tasks", "group_tasks", "next_group_id", "num_samples_fetched"):
            setattr(self.buffer, field, state[field])
        self._queued_requests = deque(state["queued_requests"])
        self._issued_requests = state["pending_requests"]
        self._replay_requests = deque(self._issued_requests.values())


class RPentCohortBuffer(RPentRolloutBufferWorker):
    """Select the first K complete groups out of N under one installed policy."""

    async def run_continuously(
        self,
        agent_output_channel,
        dataloader_channel,
        rollout_channel,
        payload_channels=None,
    ):
        self._incoming = agent_output_channel
        self._prompts = dataloader_channel
        self._rollout_channel = rollout_channel
        self._payload_channels = payload_channels
        self._resident_task = asyncio.current_task()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    async def release_batch(self, version: int):
        issued = {}
        for _ in range(self.cfg.data.sampling_batch_size):
            request = await self._next_request()
            gid = request.group_ids[0]
            issued[gid] = request.answers[0]
            await self._prompts.put(request, async_op=True).async_wait()
        selected = []
        self._selected = []
        rewards = []
        while len(selected) < self.cfg.data.rollout_batch_size:
            metadata = await self._incoming.get(async_op=True).async_wait()
            if (
                not isinstance(metadata, GroupMetadata)
                or metadata.group_id not in issued
                or metadata.group_id in selected
            ):
                raise RuntimeError(
                    f"Unexpected synchronous group completion: {metadata}"
                )
            if (metadata.earliest_version, metadata.completion_version) != (
                version,
                version,
            ):
                raise RuntimeError(
                    f"GRPO group {metadata.group_id} spans policy versions"
                )
            if len(metadata.rewards) != self.cfg.algorithm.group_size:
                raise RuntimeError(f"Incomplete GRPO group {metadata.group_id}")
            selected.append(metadata.group_id)
            self._selected.append(metadata)
            rewards.append(metadata.rewards)
        return {
            "version": version,
            "tasks": issued,
            "selected": selected,
            "remaining": sorted(set(issued).difference(selected)),
            "rewards": rewards,
        }

    async def admit_cohort(self, cohort):
        remaining = set(cohort["remaining"])
        late_completions = []
        while remaining:
            record = await self._incoming.get(async_op=True).async_wait()
            remaining.remove(record.group_id)
            if isinstance(record, GroupMetadata):
                await self._release_payload(record)
                late_completions.append(record.group_id)
            elif (
                not isinstance(record, DiscardedGroup) or record.reason != "cohort_full"
            ):
                raise RuntimeError(f"Unselected episode group failed: {record}")
            self.buffer.drop_group(
                record.group_id,
                "cohort_full",
                len(cohort["remaining"]),
                already_buffered=False,
            )
        for metadata in self._selected:
            result, metadata = await self._prepare_result(None, metadata)
            self.buffer.record_result(result, metadata)
        for _, result in self.buffer.release_groups():
            await self._rollout_channel.put(result, async_op=True).async_wait()
        self._selected.clear()
        self._issued_requests.clear()
        return late_completions
