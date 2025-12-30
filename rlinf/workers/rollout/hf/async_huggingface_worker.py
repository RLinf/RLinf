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

from tqdm import tqdm

from rlinf.data.io_struct import AsyncEmbodiedRolloutBuffer
from rlinf.scheduler import Channel
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    async def generate(
        self, input_channel: Channel, output_channel: Channel, replay_channel: Channel
    ):
        self.buffer_list: list[AsyncEmbodiedRolloutBuffer] = [
            AsyncEmbodiedRolloutBuffer() for _ in range(self.num_pipeline_stages)
        ]

        self.buffer_tasks: list[asyncio.Task] = []
        for buffer in self.buffer_list:
            self.buffer_tasks.append(
                asyncio.create_task(
                    buffer.run(replay_channel, self.get_actor_split_num())
                )
            )

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        progress_bar = tqdm(
            total=None,
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        )

        while not self.should_stop:
            last_extracted_obs = [None for i in range(self.num_pipeline_stages)]
            last_results = [None for i in range(self.num_pipeline_stages)]

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_results[stage_id] is not None:
                        last_results[stage_id]["forward_inputs"] = (
                            self.update_intervene_actions(
                                env_output, last_results[stage_id]["forward_inputs"]
                            )
                        )

                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs
                    )

                    actions, result = self.predict(extracted_obs)

                    await self.buffer_list[stage_id].add(
                        "truncations",
                        env_output["truncations"].bool().cpu().contiguous(),
                    )
                    await self.buffer_list[stage_id].add(
                        "terminations",
                        env_output["terminations"].bool().cpu().contiguous(),
                    )
                    await self.buffer_list[stage_id].add("dones", dones)
                    if rewards is not None:
                        await self.buffer_list[stage_id].add("rewards", rewards)
                    if last_results[stage_id] is not None:
                        await self.buffer_list[stage_id].add_result(
                            last_results[stage_id]
                        )

                    if last_extracted_obs[stage_id] is not None and hasattr(
                        self.hf_model, "q_head"
                    ):
                        await self.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )

                    last_extracted_obs[stage_id] = extracted_obs
                    last_results[stage_id] = result

                    self.send_chunk_actions(output_channel, actions)

            for i in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )
                await self.buffer_list[i].add(
                    "truncations", env_output["truncations"].bool().cpu().contiguous()
                )
                await self.buffer_list[i].add(
                    "terminations", env_output["terminations"].bool().cpu().contiguous()
                )
                await self.buffer_list[i].add("dones", dones)
                if rewards is not None:
                    await self.buffer_list[i].add("rewards", rewards)
                if last_results is not None:
                    await self.buffer_list[i].add_result(
                        put_tensor_device(last_results[i], "cpu")
                    )

                with self.worker_timer():
                    actions, result = self.predict(extracted_obs)
                if "prev_values" in result:
                    await self.buffer_list[i].add(
                        "prev_values", result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    await self.buffer_list[i].add_transition(
                        last_extracted_obs[i], real_extracted_obs
                    )

            progress_bar.update(1)

    async def stop(self):
        self.should_stop = True
        for buffer in self.buffer_list:
            await buffer.stop()
        await asyncio.gather(*self.buffer_tasks)
