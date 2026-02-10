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

from typing import Any, Literal

import torch

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class AsyncPPOEnvWorker(EnvWorker):
    """Async PPO env worker with persistent interaction loop."""

    def _get_n_train_chunk_steps(self) -> int:
        return (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

    def _get_zero_dones(self) -> torch.Tensor:
        return (
            torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
            .unsqueeze(1)
            .repeat(1, self.cfg.actor.model.num_action_chunks)
        )

    def init_env_outputs(self) -> list[EnvOutput]:
        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = self._get_zero_dones()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=dones.clone(),
                    truncations=dones.clone(),
                    final_obs=infos.get("final_observation", None),
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            zero_dones = self._get_zero_dones()
            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=zero_dones,
                    terminations=zero_dones,
                    truncations=zero_dones,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)
        return env_outputs

    def store_env_states(self, env_outputs: list[EnvOutput]) -> None:
        self.last_obs_list = [env_output.obs for env_output in env_outputs]
        self.last_dones_list = [env_output.dones for env_output in env_outputs]
        self.last_truncations_list = [
            env_output.truncations for env_output in env_outputs
        ]
        self.last_terminations_list = [
            env_output.terminations for env_output in env_outputs
        ]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_outputs
        ]

    def record_env_metrics(self, env_metrics: dict[Any, list], env_info: dict) -> None:
        for key, value in env_info.items():
            if value is None:
                continue
            if key not in env_metrics:
                env_metrics[key] = []
            env_metrics[key].append(value)

    def send_env_batch_async(
        self,
        output_channel: Channel,
        env_batch: dict,
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Async version of send_env_batch for use in async contexts."""
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        for gather_id in range(self.gather_num):
            env_batch_i = self.split_env_batch(env_batch, gather_id, mode)
            output_channel.put(
                item=env_batch_i,
                key=f"{gather_id + self._rank * self.gather_num}_{mode}",
                async_op=True,
            )

    async def start_interacting(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metrics_channel: Channel,
    ) -> None:
        for env in self.env_list:
            env.start_env()

        while not self.should_stop:
            env_outputs = self.init_env_outputs()
            env_metrics: dict[Any, list] = {}

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                self.send_env_batch(output_channel, env_output.to_dict())

            for _ in range(self._get_n_train_chunk_steps()):
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

    async def stop(self) -> None:
        self.should_stop = True
        for env in self.env_list:
            env.stop_env()
