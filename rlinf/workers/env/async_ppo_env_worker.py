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


from typing import Any

import torch

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class AsyncPPOEnvWorker(EnvWorker):
    def init_env_outputs(self) -> list[EnvOutput]:
        env_output_list = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = (
                    torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                    .unsqueeze(1)
                    .repeat(1, self.cfg.actor.model.num_action_chunks)
                )
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_output_list.append(env_output)
        else:
            dones = (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_output_list.append(env_output)

        return env_output_list

    def store_env_states(self, env_outputs: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_outputs]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_outputs
        ]

    def record_env_metrics(self, env_metrics: dict[Any, list], env_info: dict):
        for key, value in env_info.items():
            if value is not None:
                if key not in env_metrics:
                    env_metrics[key] = []
                env_metrics[key].append(value)

    async def start_interacting(
        self, input_channel: Channel, output_channel: Channel, metrics_channel: Channel
    ):
        while not self.should_stop:
            env_outputs = self.init_env_outputs()

            env_metrics: dict[list] = {}

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                self.send_env_batch(output_channel, env_output.to_dict())

            for _ in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(input_channel)
                    env_output, env_info = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    self.send_env_batch(output_channel, env_output.to_dict())
                    env_outputs[stage_id] = env_output

                    self.record_env_metrics(env_metrics, env_info)

            aggregated_metrics = {}
            for key, values in env_metrics.items():
                if values:
                    aggregated_metrics[key] = (
                        torch.cat(values, dim=0).contiguous().cpu()
                    )

            metrics_channel.put(aggregated_metrics, async_op=True)

            self.store_env_states(env_outputs)
            self.finish_rollout()

    async def stop(self):
        self.should_stop = True
        for env in self.env_list:
            env.stop_env()
