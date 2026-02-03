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


from rlinf.data.embodied_io_struct import EmbodiedRolloutResult
from rlinf.scheduler import Channel
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        replay_channel: Channel,
        metric_channel: Channel,
    ):
        while not self.should_stop:
            # rollout_results[stage_id]
            self.rollout_results: list[EmbodiedRolloutResult] = [
                EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.train.max_episode_steps,
                    model_weights_id=self.model_weights_id,
                )
                for _ in range(self.num_pipeline_stages)
            ]

            await self.generate_one_epoch(input_channel, output_channel)
            for stage_id in range(self.num_pipeline_stages):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], replay_channel
                )

            rollout_metrics = self.pop_execution_times()
            rollout_metrics = {
                f"time/rollout/{k}": v for k, v in rollout_metrics.items()
            }
            metric_channel.put(rollout_metrics, async_op=True)

    async def stop(self):
        self.should_stop = True
