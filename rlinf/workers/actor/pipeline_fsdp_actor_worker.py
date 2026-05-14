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
import random
from dataclasses import dataclass

import torch

from rlinf.scheduler import Channel, Worker
from rlinf.utils.metric_utils import (
    compute_pipeline_expected_actor_recv_num,
    compute_rollout_metrics,
    unpack_pipeline_train_batch_from_transport,
)
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    _merge_pipeline_train_batches,
)


@dataclass
class _PipelineGlobalBatchState:
    micro_batches: list[dict[str, torch.Tensor]]
    train_count: int = 0


@dataclass
class _PipelineMicroBatchState:
    batch: dict[str, torch.Tensor]
    arrival_id: int
    train_count: int = 0


class PipelineEmbodiedFSDPActor(EmbodiedFSDPActor):
    def _get_pipeline_schedule(self) -> str:
        pipeline_schedule = self.cfg.runner.pipeline_schedule
        if pipeline_schedule not in {"global_batch", "micro_batch"}:
            raise ValueError(
                "runner.pipeline_schedule must be 'global_batch' or 'micro_batch'."
            )
        return pipeline_schedule

    def _recv_pipeline_train_batch(
        self,
        input_channel: Channel,
        *,
        blocking: bool,
    ) -> dict[str, torch.Tensor]:
        if blocking:
            with self.worker_timer("wait_pipeline_rollout_batch"):
                packed_batch = input_channel.get()
        else:
            packed_batch = input_channel.get_nowait()
        return unpack_pipeline_train_batch_from_transport(packed_batch)

    def _train_pipeline_step(
        self,
        micro_batches: list[dict[str, torch.Tensor]],
        metrics: dict[str, list[float]],
    ) -> None:
        self.optimizer.zero_grad()
        for idx, batch in enumerate(micro_batches):
            with self.worker_timer("pipeline_train_micro_batch"):
                self._train_embodied_micro_batch(
                    batch,
                    metrics,
                    is_last_micro_batch=(idx + 1) == len(micro_batches),
                )
        with self.worker_timer("pipeline_train_step"):
            self._finish_embodied_global_batch(metrics)

    def _select_pipeline_global_batch(
        self,
        global_batches: list[_PipelineGlobalBatchState],
        update_epoch: int,
        *,
        shuffle_rollout: bool,
        rng: random.Random,
    ) -> _PipelineGlobalBatchState | None:
        candidates = [
            batch_state
            for batch_state in global_batches
            if batch_state.train_count < update_epoch
        ]
        if not candidates:
            return None
        if shuffle_rollout:
            rng.shuffle(candidates)
        candidates.sort(key=lambda batch_state: batch_state.train_count)
        return candidates[0]

    def _select_pipeline_micro_batches(
        self,
        micro_batches: list[_PipelineMicroBatchState],
        update_epoch: int,
        *,
        shuffle_rollout: bool,
        rng: random.Random,
    ) -> list[_PipelineMicroBatchState]:
        candidates = [
            batch_state
            for batch_state in micro_batches
            if batch_state.train_count < update_epoch
        ]
        if len(candidates) < self.gradient_accumulation:
            return []
        if shuffle_rollout:
            rng.shuffle(candidates)
            candidates.sort(key=lambda batch_state: batch_state.train_count)
        else:
            candidates.sort(
                key=lambda batch_state: (
                    batch_state.train_count,
                    batch_state.arrival_id,
                )
            )
        return candidates[: self.gradient_accumulation]

    @Worker.timer("run_training_pipeline")
    def run_training_pipeline(self, input_channel: Channel) -> dict[str, dict]:
        self._prepare_embodied_training()
        pipeline_schedule = self._get_pipeline_schedule()
        shuffle_rollout = bool(self.cfg.algorithm.get("shuffle_rollout", True))
        rng = random.Random(
            self.cfg.actor.seed + self.version * self._world_size + self._rank
        )
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)

        expected_split_num = compute_pipeline_expected_actor_recv_num(
            total_num_envs=self.cfg.env.train.total_num_envs,
            actor_world_size=self._world_size,
            micro_batch_size=self.cfg.actor.micro_batch_size,
            rollout_epoch=self.cfg.algorithm.rollout_epoch,
            n_train_chunk_steps=self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks,
        )
        metrics: dict[str, list[float]] = {}
        pipeline_train_batches = []
        received_split_num = 0
        pending_global_batch: list[dict[str, torch.Tensor]] = []
        ready_global_batches: list[_PipelineGlobalBatchState] = []
        ready_micro_batches: list[_PipelineMicroBatchState] = []
        next_arrival_id = 0

        while True:
            while received_split_num < expected_split_num:
                try:
                    train_micro_batch = self._recv_pipeline_train_batch(
                        input_channel,
                        blocking=False,
                    )
                except asyncio.QueueEmpty:
                    break

                pipeline_train_batches.append(train_micro_batch)
                received_split_num += 1
                if pipeline_schedule == "global_batch":
                    pending_global_batch.append(train_micro_batch)
                    if len(pending_global_batch) == self.gradient_accumulation:
                        ready_global_batches.append(
                            _PipelineGlobalBatchState(
                                micro_batches=pending_global_batch,
                            )
                        )
                        pending_global_batch = []
                else:
                    ready_micro_batches.append(
                        _PipelineMicroBatchState(
                            batch=train_micro_batch,
                            arrival_id=next_arrival_id,
                        )
                    )
                    next_arrival_id += 1

            if pipeline_schedule == "global_batch":
                global_batch_state = self._select_pipeline_global_batch(
                    ready_global_batches,
                    update_epoch,
                    shuffle_rollout=shuffle_rollout,
                    rng=rng,
                )
                if global_batch_state is not None:
                    self._train_pipeline_step(global_batch_state.micro_batches, metrics)
                    global_batch_state.train_count += 1
                    continue
            else:
                micro_batch_states = self._select_pipeline_micro_batches(
                    ready_micro_batches,
                    update_epoch,
                    shuffle_rollout=shuffle_rollout,
                    rng=rng,
                )
                if micro_batch_states:
                    self._train_pipeline_step(
                        [batch_state.batch for batch_state in micro_batch_states],
                        metrics,
                    )
                    for batch_state in micro_batch_states:
                        batch_state.train_count += 1
                    continue

            if received_split_num == expected_split_num:
                break

            train_micro_batch = self._recv_pipeline_train_batch(
                input_channel,
                blocking=True,
            )
            pipeline_train_batches.append(train_micro_batch)
            received_split_num += 1
            if pipeline_schedule == "global_batch":
                pending_global_batch.append(train_micro_batch)
                if len(pending_global_batch) == self.gradient_accumulation:
                    ready_global_batches.append(
                        _PipelineGlobalBatchState(
                            micro_batches=pending_global_batch,
                        )
                    )
                    pending_global_batch = []
            else:
                ready_micro_batches.append(
                    _PipelineMicroBatchState(
                        batch=train_micro_batch,
                        arrival_id=next_arrival_id,
                    )
                )
                next_arrival_id += 1

        if pending_global_batch:
            raise RuntimeError(
                "The training pipeline ended with an incomplete global batch."
            )

        if pipeline_schedule == "global_batch":
            if any(
                batch_state.train_count != update_epoch
                for batch_state in ready_global_batches
            ):
                raise RuntimeError(
                    "The global-batch pipeline schedule did not finish all update epochs."
                )
        else:
            if any(
                batch_state.train_count != update_epoch
                for batch_state in ready_micro_batches
            ):
                raise RuntimeError(
                    "The micro-batch pipeline schedule did not finish all update epochs."
                )

        self.rollout_batch = _merge_pipeline_train_batches(pipeline_train_batches)
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics.update(
            {
                "pipeline_received_splits": float(len(pipeline_train_batches)),
                "pipeline_expected_splits": float(expected_split_num),
            }
        )
        training_metrics = self._finalize_embodied_training_metrics(metrics)
        return {"rollout": rollout_metrics, "train": training_metrics}
