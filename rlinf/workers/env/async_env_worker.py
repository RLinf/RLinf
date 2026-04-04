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

from omegaconf.omegaconf import DictConfig

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
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        metric_channel: Channel,
    ):
        assert self._interact_task is None or self._interact_task.done(), (
            "Previous interact task is still running while a new interact call is made."
        )
        self._interact_task = asyncio.create_task(
            self._interact(
                input_channel,
                rollout_channel,
                reward_channel,
                actor_channel,
                metric_channel,
            )
        )
        try:
            await self._interact_task
        except asyncio.CancelledError:
            pass

    async def _interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        metric_channel: Channel,
    ):
        while True:
            env_metrics = await self._run_interact_once(
                input_channel,
                rollout_channel,
                reward_channel,
                actor_channel,
                cooperative_yield=True,
            )

            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            env_interact_time_metrics = self.pop_execution_times()
            env_interact_time_metrics = {
                f"time/env/{k}": v for k, v in env_interact_time_metrics.items()
            }
            metrics = {
                "rank": self._rank,
                "env": env_metrics,
                "time": env_interact_time_metrics,
            }
            metric_channel.put(metrics, async_op=True)

    async def stop(self):
        if self._interact_task is not None and not self._interact_task.done():
            self._interact_task.cancel()
