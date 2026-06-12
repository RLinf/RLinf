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
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.scheduler import Channel, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import compute_rollout_metrics
from rlinf.utils.utils import unpack_batch
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


@dataclass
class GlobalBatchState:
    micro_batches: list[dict[str, torch.Tensor]]
    train_count: int = 0


class PipelineEmbodiedFSDPActor(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.micro_batches_per_step = self.compute_micro_batches(
            total_num_envs=self.cfg.env.train.total_num_envs,
            actor_world_size=self._world_size,
            micro_batch_size=self.cfg.actor.micro_batch_size,
            rollout_epoch=self.cfg.algorithm.rollout_epoch,
            n_train_chunk_steps=self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks,
        )
        assert self.micro_batches_per_step % self.gradient_accumulation == 0, (
            f"micro_batches_per_step ({self.micro_batches_per_step}) must be divisible by "
            f"gradient_accumulation ({self.gradient_accumulation})."
        )
        self.global_batches_per_step = (
            self.micro_batches_per_step // self.gradient_accumulation
        )

    def try_recv_micro_batch(
        self,
        input_channel: Channel,
    ) -> dict[str, torch.Tensor] | None:
        try:
            packed_batch = input_channel.get_nowait(
                key=CommMapper.build_channel_key(
                    self._rank, self._rank, "pipeline_actor"
                )
            )
            return unpack_batch(packed_batch)
        except asyncio.QueueEmpty:
            return None

    def recv_micro_batch(self, input_channel: Channel) -> dict[str, torch.Tensor]:
        packed_batch = input_channel.get(
            key=CommMapper.build_channel_key(self._rank, self._rank, "pipeline_actor")
        )
        return unpack_batch(packed_batch)

    def select_global_batch(
        self, global_batches: dict[int, deque[GlobalBatchState]]
    ) -> GlobalBatchState | None:
        for epoch in range(1, self.update_epoch):
            if global_batches[epoch]:
                return global_batches[epoch].popleft()
        return None

    @Worker.timer("run_training")
    def run_training(
        self, input_channel: Channel
    ) -> dict[str, dict[str, float] | dict]:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()

        metrics: dict[str, list[float]] = {}
        global_batches: dict[int, deque[GlobalBatchState]] = defaultdict(deque)
        current_global_batch: list[dict[str, torch.Tensor]] = []
        pending_global_batch: deque[dict[str, torch.Tensor]] = deque()
        received_rollout_micro_batches: list[dict[str, torch.Tensor]] = []
        received_micro_batch_count = 0

        while True:
            while received_micro_batch_count < self.micro_batches_per_step:
                if not (micro_batch := self.try_recv_micro_batch(input_channel)):
                    break
                pending_global_batch.append(micro_batch)
                received_rollout_micro_batches.append(micro_batch)
                received_micro_batch_count += 1

            while pending_global_batch:
                micro_batch = pending_global_batch.popleft()
                is_last_micro_batch = (
                    len(current_global_batch) == self.gradient_accumulation - 1
                )
                self.train_micro_batch(
                    micro_batch=micro_batch,
                    metrics=metrics,
                    is_last=is_last_micro_batch,
                )
                current_global_batch.append(micro_batch)
                if is_last_micro_batch:
                    self.finish_global_batch(metrics)
                    global_batches[1].append(
                        GlobalBatchState(
                            micro_batches=current_global_batch, train_count=1
                        )
                    )
                    current_global_batch = []

            if len(global_batches[self.update_epoch]) == self.global_batches_per_step:
                break

            if not (global_batch := self.select_global_batch(global_batches)):
                if received_micro_batch_count < self.micro_batches_per_step:
                    micro_batch = self.recv_micro_batch(input_channel)
                    pending_global_batch.append(micro_batch)
                    received_rollout_micro_batches.append(micro_batch)
                    received_micro_batch_count += 1
                continue

            for idx, micro_batch in enumerate(global_batch.micro_batches):
                is_last_micro_batch = idx == len(global_batch.micro_batches) - 1
                self.train_micro_batch(
                    micro_batch=micro_batch,
                    metrics=metrics,
                    is_last=is_last_micro_batch,
                )

            self.finish_global_batch(metrics)
            global_batch.train_count += 1
            global_batches[global_batch.train_count].append(global_batch)
        self.lr_scheduler.step()

        rollout_metric_batch = {
            key: torch.cat(
                [batch[key] for batch in received_rollout_micro_batches], dim=0
            )
            for key in ("rewards", "advantages", "returns")
            if received_rollout_micro_batches
            and key in received_rollout_micro_batches[0]
        }
        rollout_metrics = compute_rollout_metrics(rollout_metric_batch)

        del (
            current_global_batch,
            global_batches,
            pending_global_batch,
            received_rollout_micro_batches,
            rollout_metric_batch,
        )
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return {
            "rollout_metrics": rollout_metrics,
            "training_metrics": mean_metric_dict,
        }

    def compute_micro_batches(
        self,
        total_num_envs: int,
        actor_world_size: int,
        micro_batch_size: int,
        rollout_epoch: int,
        n_train_chunk_steps: int,
    ) -> int:
        """Compute how many pipeline local micro batches each actor rank should receive."""
        total_rollout_samples = total_num_envs * rollout_epoch * n_train_chunk_steps
        per_rank_micro_batch_samples = actor_world_size * micro_batch_size
        assert total_rollout_samples % per_rank_micro_batch_samples == 0, (
            f"Total flattened rollout samples ({total_rollout_samples}) must be "
            f"divisible by actor_world_size * micro_batch_size "
            f"({per_rank_micro_batch_samples})."
        )
        return total_rollout_samples // per_rank_micro_batch_samples
