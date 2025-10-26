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

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rlinf.data.io_struct import EmbodiedRolloutResult
from rlinf.models import get_model, get_vla_model_config_and_processor
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.algorithms.utils import expand_to_target_dim

def init_real_next_obs(next_extracted_obs):
    # Copy the next-extracted-obs
    if isinstance(next_extracted_obs, torch.Tensor):
        real_next_extracted_obs = next_extracted_obs.clone()
    elif isinstance(next_extracted_obs, dict):
        real_next_extracted_obs = dict()
        for key, value in next_extracted_obs.items():
            if value is None:
                continue
            assert isinstance(value, torch.Tensor), f"{key}, {type(value)}"
            real_next_extracted_obs[key] = value.clone()
    else:
        raise NotImplementedError
    return real_next_extracted_obs

def update_real_next_obs(real_next_extracted_obs, final_extracted_obs, last_step_dones):
    # Update the next-extracted-obs according to the final doness
    if isinstance(real_next_extracted_obs, torch.Tensor):
        dones_mask = expand_to_target_dim(last_step_dones, final_extracted_obs[key].shape)
        dones_mask = dones_mask.expand_as(final_extracted_obs[key])
        real_next_extracted_obs[dones_mask] = final_extracted_obs[dones_mask]
    elif isinstance(real_next_extracted_obs, dict):
        for key in real_next_extracted_obs.keys():
            dones_mask = expand_to_target_dim(last_step_dones, final_extracted_obs[key].shape)
            dones_mask = dones_mask.expand_as(final_extracted_obs[key])
            real_next_extracted_obs[key][dones_mask] = final_extracted_obs[key][dones_mask]
    return real_next_extracted_obs

class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        self.stage_num = cfg.rollout.pipeline_stage_num

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.channel = self.connect_channel(cfg.rollout.channel.name)

    def init_worker(self):
        self.hf_model = get_model(self.cfg.rollout.model_dir, self.cfg.actor.model)

        if self.cfg.actor.model.model_name in ["openvla", "openvla_oft"]:
            model_config, input_processor = get_vla_model_config_and_processor(
                self.cfg.actor
            )
            self.hf_model.setup_config_and_processor(
                model_config, self.cfg, input_processor
            )

        self.hf_model.eval()

        self.setup_sample_params()
        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, do_sample=True, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )
        kwargs["do_sample"] = do_sample
        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        if self.cfg.actor.model.model_name in ["openpi", "mlp"]:
            kwargs = {"mode": mode}

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def update_env_output(self, i, env_output, next_extracted_obs):
        real_next_extracted_obs = init_real_next_obs(next_extracted_obs)

        # first step for env_batch
        if env_output["rewards"] is None:
            self.buffer_list[i].dones.append(env_output["dones"].contiguous().cpu())
            return real_next_extracted_obs

        
        self.buffer_list[i].rewards.append(env_output["rewards"].cpu().contiguous())
        self.buffer_list[i].dones.append(env_output["dones"].bool().cpu().contiguous())

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
                    real_next_extracted_obs = update_real_next_obs(
                        real_next_extracted_obs, final_extracted_obs, last_step_dones
                    )
                    
                    actions, result = self.predict(final_extracted_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                self.buffer_list[i].rewards[-1][:, -1] += (
                    self.cfg.algorithm.gamma * final_values.cpu()
                )
        return real_next_extracted_obs

    async def generate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()
        self.buffer_list = [EmbodiedRolloutResult() for _ in range(self.stage_num)]

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            extracted_obs = [None for i in range(self.stage_num)]
            for chunk_step in range(self.cfg.algorithm.n_chunk_steps):
                for i in range(self.stage_num):
                    env_output = await self.recv_env_output()

                    next_extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"]) # 但这里没有 final obs
                    real_next_extracted_obs = self.update_env_output(i, env_output, next_extracted_obs) # 这里处理了 final obs

                    actions, result = self.predict(next_extracted_obs) # results 里面才包含这一步 obs 的格式

                    self.buffer_list[i].append_result(result)

                    if extracted_obs[i] is not None and hasattr(self.hf_model, "q_head"):
                        self.buffer_list[i].add_transition(extracted_obs[i], real_next_extracted_obs)
                    
                    extracted_obs[i] = real_next_extracted_obs

                    await self.send_chunk_actions(actions)

            for i in range(self.stage_num):
                env_output = await self.recv_env_output()
                next_extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                real_next_extracted_obs = self.update_env_output(i, env_output, next_extracted_obs)
                actions, result = self.predict(next_extracted_obs)
                if "prev_values" in result:
                    self.buffer_list[i].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    self.buffer_list[i].add_transition(extracted_obs[i], real_next_extracted_obs)

        for i in range(self.stage_num):
            await self.send_rollout_batch(i)

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    async def evaluate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()

        for _ in range(self.cfg.algorithm.n_eval_chunk_steps):
            for _ in range(self.stage_num):
                env_output = await self.recv_env_output()
                next_extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                actions, _ = self.predict(next_extracted_obs, mode="eval")
                await self.send_chunk_actions(actions)

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    def sync_model_from_actor(self):
        param_state_dict = self.recv(self._actor_group_name, src_rank=self._rank)
        self.hf_model.load_state_dict(param_state_dict)
        
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    async def recv_env_output(self):
        env_output = await self.channel.get(
            key=f"{self._obs_queue_name}_{self._rank}", async_op=True
        ).async_wait()
        return env_output

    async def send_chunk_actions(self, chunk_actions):
        await self.channel.put(
            item=chunk_actions,
            key=f"{self._action_queue_name}_{self._rank}",
            async_op=True,
        ).async_wait()

    async def send_rollout_batch(self, stage_id):
        # send rollout_batch to actor
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        splited_rollout_result = self.buffer_list[stage_id].to_splited_dict(split_num)
        for i in range(split_num):
            await self.channel.put(
                item=splited_rollout_result[i],
                key=self._replay_buffer_name,
                async_op=True,
            ).async_wait()

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
