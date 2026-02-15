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

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class AsyncEnvWorker(EnvWorker):
    async def interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        epoch = 0
        while not self.should_stop:
            env_metrics = defaultdict(list)
            env_output_list = []
            if not self.cfg.env.train.auto_reset:
                for i in range(self.stage_num):
                    extracted_obs, infos = self.env_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
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
                self.num_done_envs = 0
                self.num_succ_envs = 0
                dones = (
                    torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                    .unsqueeze(1)
                    .repeat(1, self.cfg.actor.model.num_action_chunks)
                )
                terminations = dones.clone()
                truncations = dones.clone()
                for i in range(self.stage_num):
                    env_output = EnvOutput(
                        obs=self.last_obs_list[i],
                        rewards=None,
                        dones=dones,
                        terminations=terminations,
                        truncations=truncations,
                        intervene_actions=self.last_intervened_info_list[i][0],
                        intervene_flags=self.last_intervened_info_list[i][1],
                    )
                    env_output_list.append(env_output)

            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_output_list[stage_id]
                self.send_env_batch(output_channel, env_output.to_dict())

            for _ in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    await asyncio.sleep(0)
                    raw_chunk_actions = self.recv_chunk_actions(input_channel)
                    env_output, env_info = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    self.send_env_batch(output_channel, env_output.to_dict())
                    env_output_list[stage_id] = env_output
                    for key, value in env_info.items():
                        if (
                            not self.cfg.env.train.auto_reset
                            and not self.cfg.env.train.ignore_terminations
                        ):
                            if key in env_metrics and len(env_metrics[key]) > epoch:
                                env_metrics[key][epoch] = value
                            else:
                                env_metrics[key].append(value)
                        else:
                            env_metrics[key].append(value)

            for key, value in env_metrics.items():
                env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            env_interact_time_metrics = self.pop_execution_times()
            env_metrics.update(
                {f"time/env/{k}": v for k, v in env_interact_time_metrics.items()}
            )
            metric_channel.put(env_metrics)

            self.last_obs_list = [env_output.obs for env_output in env_output_list]
            self.last_intervened_info_list = [
                (env_output.intervene_actions, env_output.intervene_flags)
                for env_output in env_output_list
            ]
            self.finish_rollout()
            epoch += 1

    async def stop(self):
        self.should_stop = True
