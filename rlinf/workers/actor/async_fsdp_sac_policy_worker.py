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

from rlinf.scheduler import Channel
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class AsyncEmbodiedSACFSDPPolicy(EmbodiedSACFSDPPolicy):
    async def start_replay_buffer(self, replay_channel: Channel):
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        replay_buffer_task = asyncio.create_task(
            self.replay_buffer.run(
                self.cfg, data_channel=replay_channel, split_num=split_num
            )
        )
        await replay_buffer_task

    async def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = (
            self.cfg.algorithm.get("min_buffer_size", 100) // self._world_size
        )
        train_actor_steps = (
            self.cfg.algorithm.get("train_actor_steps", 0) // self._world_size
        )
        train_actor_steps = max(min_buffer_size, train_actor_steps)

        if not (await self.replay_buffer.is_ready_async(min_buffer_size)):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return False

        train_actor = await self.replay_buffer.is_ready_async(train_actor_steps)

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
            metrics_data = self.update_one_epoch(train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict
