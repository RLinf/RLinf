# Copyright 2025 The RLinf Authors.
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

import torch

from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class AsyncEmbodiedSACFSDPPolicy(EmbodiedSACFSDPPolicy):
    should_stop = False

    async def recv_rollout_trajectories(self, input_channel):
        while not self.should_stop:
            await super().recv_rollout_trajectories(input_channel)

    async def _is_replay_buffer_ready_all_ranks(self, min_buffer_size: int) -> bool:
        local_ready = await self.replay_buffer.is_ready_async(min_buffer_size)
        if not torch.distributed.is_initialized():
            return local_ready
        ready_tensor = torch.tensor(
            1 if local_ready else 0, device=self.device, dtype=torch.int32
        )
        torch.distributed.all_reduce(ready_tensor, op=torch.distributed.ReduceOp.SUM)
        dist_ready = ready_tensor.item() == torch.distributed.get_world_size()
        return dist_ready

    @Worker.timer("run_training")
    async def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not (await self._is_replay_buffer_ready_all_ranks(min_buffer_size)):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            await asyncio.sleep(0)
            metrics_data = self.update_one_epoch()
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    async def stop(self):
        self.should_stop = True
