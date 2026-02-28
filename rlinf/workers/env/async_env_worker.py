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
from collections import defaultdict

import torch
from omegaconf.omegaconf import DictConfig

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class AsyncEnvWorker(EnvWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._interact_task: asyncio.Task = None
        assert not self.enable_offload, "Offload not supported in AsyncEnvWorker"

    async def interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        assert self._interact_task is None or self._interact_task.done(), (
            "Previous interact task is still running while a new interact call is made."
        )
        self._interact_task = asyncio.create_task(
            self._interact(input_channel, output_channel, metric_channel)
        )
        try:
            await self._interact_task
        except asyncio.CancelledError:
            pass

    async def _interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        epoch = 0
        while True:
            env_metrics = defaultdict(list)
            env_output_list = self.init_env_outputs()
            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_output_list[stage_id]
                self.send_env_batch(output_channel, env_output.to_dict())

            for _ in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    await asyncio.sleep(0)
                    raw_chunk_actions = self.recv_chunk_actions(input_channel)
                    env_output, env_info = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    self.send_env_batch(output_channel, env_output.to_dict())
                    env_output_list[stage_id] = env_output
                    self.record_env_metrics(env_metrics, env_info, epoch)

            for key, value in env_metrics.items():
                env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            env_interact_time_metrics = self.pop_execution_times()
            env_interact_time_metrics = {
                f"time/env/{k}": v for k, v in env_interact_time_metrics.items()
            }
            metrics = {
                "env": env_metrics,
                "time": env_interact_time_metrics,
            }
            metric_channel.put(metrics, async_op=True)

            self.store_last_obs_and_intervened_info(env_output_list)
            self.finish_rollout()
            epoch += 1

    async def stop(self):
        if self._interact_task is not None and not self._interact_task.done():
            self._interact_task.cancel()
