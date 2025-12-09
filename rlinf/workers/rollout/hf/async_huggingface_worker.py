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

import gc
import asyncio

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import List

from rlinf.data.io_struct.embodied.rollout_buffer import AsyncEmbodiedRolloutBuffer
from rlinf.models import get_model, get_vla_model_config_and_processor
from rlinf.scheduler import Cluster, Worker, Channel
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.utils import (
    init_real_next_obs
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    async def update_env_output(self, i, env_output, next_extracted_obs):
        real_next_extracted_obs  = None
        
        # first step for env_batch
        if env_output["rewards"] is None:
            if hasattr(self.hf_model, "q_head"):
                real_next_extracted_obs = init_real_next_obs(next_extracted_obs)
            return real_next_extracted_obs

        
        await self.buffer_list[i].add("dones", env_output["dones"].bool().cpu().contiguous())
        rewards = env_output["rewards"].cpu().contiguous()

        # Note: currently this is not correct for chunk-size>1 with partial reset
        if env_output["dones"].any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                dones = env_output["dones"]
                final_obs = env_output["final_obs"]
                last_step_dones = dones[:, -1]  # [bsz, ]

                with torch.no_grad():
                    final_extracted_obs = self.hf_model.preprocess_env_obs(
                        final_obs
                    )
                    if hasattr(self.hf_model, "q_head"):
                        real_next_extracted_obs = init_real_next_obs(final_extracted_obs)
                    
                    actions, result = self.predict(final_extracted_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        await self.buffer_list[i].add("rewards", rewards)

        if real_next_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_next_extracted_obs = init_real_next_obs(next_extracted_obs)
        return real_next_extracted_obs

    def get_split_num(self):
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num


    async def generate(
            self, 
            data_channel: Channel
    ):  
        self.buffer_list: List[AsyncEmbodiedRolloutBuffer] = [
            AsyncEmbodiedRolloutBuffer() for _ in range(self.stage_num)
        ]

        tasks = []
        for buffer in self.buffer_list:
            tasks.append(
                asyncio.create_task(buffer.run(data_channel, self.get_split_num()))
            )
        
        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            extracted_obs = [None for i in range(self.stage_num)]
            for chunk_step in range(self.cfg.algorithm.n_chunk_steps):
                for i in range(self.stage_num):
                    env_output = await self.recv_env_output()

                    next_extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    real_next_extracted_obs = await self.update_env_output(i, env_output, next_extracted_obs)

                    predict = False \
                         if self.random_predict_steps > 0 \
                            and self.all_cumulated_steps[i] < self.random_predict_steps \
                         else True
                    actions, result = self.predict(next_extracted_obs, predict=predict)
                    self.all_cumulated_steps[i] += actions.shape[0]


                    await self.buffer_list[i].add_result(result)

                    if extracted_obs[i] is not None and hasattr(self.hf_model, "q_head"):
                        await self.buffer_list[i].add_transition(extracted_obs[i], real_next_extracted_obs)
                    
                    extracted_obs[i] = next_extracted_obs

                    await self.send_chunk_actions(actions)

            for i in range(self.stage_num):
                env_output = await self.recv_env_output()
                next_extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                real_next_extracted_obs = await self.update_env_output(i, env_output, next_extracted_obs)
                actions, result = self.predict(next_extracted_obs)
                if "prev_values" in result:
                    await self.buffer_list[i].add(
                        "prev_values", 
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    await self.buffer_list[i].add_transition(extracted_obs[i], real_next_extracted_obs)

    async def sync_model_from_actor(self):
        param_state_dict = await self.recv(self._actor_group_name, src_rank=self._rank, async_op=True).async_wait()
        self.hf_model.load_state_dict(param_state_dict)
        
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()
