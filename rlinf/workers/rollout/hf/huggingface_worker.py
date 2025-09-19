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
from collections import defaultdict

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rlinf.config import torch_dtype_from_precision
from rlinf.models import get_model, get_model_config_and_processor
from rlinf.models.embodiment.model_namspace import (
    append_to_buffer,
    get_model_buffer_namespace,
)
from rlinf.models.embodiment.model_utils import (
    default_logits_processor,
    prepare_observations,
)
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement


def create_rollout_batch(data):
    ret_data = {}
    for key, value in data.items():
        if "env_info/" not in key:
            ret_data[key] = torch.stack(value, dim=0).contiguous().cpu()
        else:
            ret_data[key] = torch.cat(value, dim=0).contiguous().cpu()
    return ret_data


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.model_config, self.input_processor = get_model_config_and_processor(
            cfg.actor
        )
        self.precision = torch_dtype_from_precision(cfg.actor.model.precision)

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        self._num_images_in_input = cfg.actor.model.num_images_in_input
        self.stage_num = cfg.rollout.pipeline_stage_num

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.channel = self.connect_channel(cfg.rollout.channel.name)
        for i in range(self._component_placement.get_world_size("rollout")):
            self.channel.create_queue(
                f"{self._action_queue_name}_{i}", maxsize=cfg.rollout.channel.queue_size
            )

        self.use_proprio = self.cfg.actor.model.get("use_proprio", False)

        # Cache model namespace for buffer operations
        self.model_name = self.cfg.actor.model.model_name
        self.buffer_namespace = get_model_buffer_namespace(self.model_name)

    def init_worker(self):
        self.hf_model = get_model(self.cfg.rollout.model_dir, self.cfg.actor.model)
        self.hf_model.setup_params(self.model_config, self.cfg)
        if self.cfg.actor.model.model_name != "openpi":
            self.hf_model.to(self.precision)
        self.hf_model.eval()
        if self.cfg.actor.model.model_name == "openpi":
            self.input_processor = self.hf_model.input_transform

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

    def predict(self, processed_obs, do_sample=True, mode="train"):
        if self.cfg.actor.model.model_name in ["openpi"]:
            with torch.no_grad():
                result = self.hf_model(
                    processed_obs,
                    mode=mode,
                )
                # openpi patch, drop the input for the model
                processed_obs.pop("image")
                processed_obs.pop("image_mask")
                processed_obs.pop("state")
            return result

        action_token_len = self.hf_model.action_dim * self.hf_model.num_action_chunks

        sample_kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        with torch.no_grad():
            actions, action_tokens, action_logits, last_hidden_state = (
                self.hf_model.predict_action_batch(
                    input_ids=processed_obs["input_ids"],
                    attention_mask=processed_obs["attention_mask"],
                    pixel_values=processed_obs["pixel_values"],
                    do_sample=do_sample,
                    **sample_kwargs,
                )
            )

        chunk_logprobs = default_logits_processor(
            action_logits,
            action_tokens,
            self.hf_model.vocab_size,
            self.hf_model.config.n_action_bins,
        )["logprobs"]

        chunk_values = None
        if self.cfg.algorithm.require_values:
            if self.cfg.actor.model.vh_mode == "a0":
                hidden_features = last_hidden_state[
                    :, -action_token_len
                ]  # [batch_size, hidden_dim]
                with torch.no_grad():
                    chunk_values = self.hf_model.value_head(
                        hidden_features
                    )  # [batch_size, 1]

        if chunk_values is None:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions = actions.reshape(
            -1, self.hf_model.num_action_chunks, self.hf_model.action_dim
        )
        chunk_action_tokens = action_tokens.reshape(
            -1, self.hf_model.num_action_chunks, self.hf_model.action_dim
        )
        result = {
            "actions": chunk_actions,
            "action_tokens": chunk_action_tokens,
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
        }
        return result

    def update_env_batch(self, i, env_batch):
        # first step for env_batch
        if env_batch["rews"] is None:
            self.buffer_list[i]["dones"].append(env_batch["dones"].contiguous().cpu())
            return

        self.buffer_list[i]["rewards"].append(env_batch["rews"].cpu().contiguous())
        self.buffer_list[i]["dones"].append(
            env_batch["dones"].bool().cpu().contiguous()
        )

        if self.cfg.env.train.auto_reset or self.cfg.env.train.ignore_terminations:
            env_info_list = env_batch["meta"]
            for key, value in env_info_list.items():
                self.buffer_list[i][f"env_info/{key}"].append(value)

        # Note: currently this is not correct for chunk-size>1 with partial reset
        if env_batch["dones"].any() and self.cfg.env.train.auto_reset:
            if self.cfg.algorithm.require_values:
                dones = env_batch["dones"]
                # if self.require_values:
                final_obs = env_batch["infos"]["final_observation"]
                with torch.no_grad():
                    processed_obs = prepare_observations(
                        simulator_type=self.cfg.env.train.simulator_type,
                        model_name=self.cfg.actor.model.model_name,
                        raw_obs=final_obs,
                        use_proprio=self.use_proprio,
                        max_length=self.hf_model.max_prompt_length,
                        processor=self.input_processor,
                        precision=self.precision,
                        num_images_in_input=self._num_images_in_input,
                    )
                    result = self.predict(processed_obs)
                    _final_values = result["prev_values"]
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                self.buffer_list[i]["rewards"][-1][:, -1] += (
                    self.cfg.algorithm.gamma * final_values.cpu()
                )

    async def generate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()
        self.buffer_list = []
        for i in range(self.stage_num):
            self.buffer_list.append(defaultdict(list))

        for rollout_epoch in range(self.cfg.algorithm.rollout_epoch):
            self._logger.info(f"Now epoch is={rollout_epoch}")
            for step in tqdm(
                range(self.cfg.algorithm.n_chunk_steps),
                desc=f"Rollout ID {self._rank} Epoch {rollout_epoch} in Generate Step",
            ):
                for i in range(self.stage_num):
                    env_batch = await self.recv_env_batch()
                    self.update_env_batch(i, env_batch)
                    processed_obs = prepare_observations(
                        simulator_type=self.cfg.env.train.simulator_type,
                        model_name=self.cfg.actor.model.model_name,
                        raw_obs=env_batch["obs"],
                        use_proprio=self.use_proprio,
                        max_length=self.hf_model.max_prompt_length,
                        processor=self.input_processor,
                        precision=self.precision,
                        num_images_in_input=self._num_images_in_input,
                    )
                    result = self.predict(processed_obs)
                    # Extract actions for sending
                    chunk_actions = result["actions"]

                    # Automatically append data to buffer based on cached namespace
                    append_to_buffer(
                        buffer_list=self.buffer_list,
                        stage_idx=i,
                        namespace=self.buffer_namespace,
                        processed_obs=processed_obs,
                        result=result,
                    )

                    await self.send_chunk_actions(chunk_actions)

            for i in range(self.stage_num):
                env_batch = await self.recv_env_batch()
                self.update_env_batch(i, env_batch)
                processed_obs = prepare_observations(
                    simulator_type=self.cfg.env.train.simulator_type,
                    model_name=self.cfg.actor.model.model_name,
                    raw_obs=env_batch["obs"],
                    use_proprio=self.use_proprio,
                    max_length=self.hf_model.max_prompt_length,
                    processor=self.input_processor,
                    precision=self.precision,
                    num_images_in_input=self._num_images_in_input,
                )
                result = self.predict(processed_obs)
                final_chunk_values = result["prev_values"]
                self.buffer_list[i]["prev_values"].append(
                    final_chunk_values.cpu().contiguous()
                )

                if (
                    not self.cfg.env.train.auto_reset
                    and not self.cfg.env.train.ignore_terminations
                ):
                    infos = env_batch["infos"]
                    if "episode" in infos:
                        for key, value in infos["episode"].items():
                            self.buffer_list[i][f"env_info/{key}"].append(value.cpu())

        for i in range(self.stage_num):
            await self.send_rollout_batch(i)

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    async def evaluate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()
        eval_info = defaultdict(list)

        for step in tqdm(
            range(self.cfg.algorithm.n_eval_chunk_steps), desc="Rollout in Eval Step"
        ):
            for i in range(self.stage_num):
                env_batch = await self.recv_env_batch()
                processed_obs = prepare_observations(
                    simulator_type=self.cfg.env.eval.simulator_type,
                    model_name=self.cfg.actor.model.model_name,
                    raw_obs=env_batch["obs"],
                    use_proprio=self.use_proprio,
                    max_length=self.hf_model.max_prompt_length,
                    processor=self.input_processor,
                    precision=self.precision,
                    num_images_in_input=self._num_images_in_input,
                )
                result = self.predict(processed_obs, mode="eval")
                chunk_actions = result["actions"]
                await self.send_chunk_actions(chunk_actions)

                if "meta" in env_batch:
                    env_info_list = env_batch["meta"]
                    for key, value in env_info_list.items():
                        eval_info[f"env_info/{key}"].append(value)

        for i in range(self.stage_num):
            env_batch = await self.recv_env_batch()
            if "meta" in env_batch:
                env_info_list = env_batch["meta"]
                for key, value in env_info_list.items():
                    eval_info[f"env_info/{key}"].append(value)

        eval_metrics = create_rollout_batch(eval_info)
        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()
        return eval_metrics

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

    async def recv_env_batch(self):
        env_batch = await self.channel.get(
            queue_name=f"{self._obs_queue_name}_{self._rank}", async_op=True
        ).async_wait()
        return env_batch

    async def send_chunk_actions(self, chunk_actions):
        await self.channel.put(
            item=chunk_actions,
            queue_name=f"{self._action_queue_name}_{self._rank}",
            async_op=True,
        ).async_wait()

    async def send_rollout_batch(self, stage_id):
        # send rollout_batch to actor
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        rollout_batch = create_rollout_batch(self.buffer_list[stage_id])
        for i in range(split_num):
            rollout_batch_i = {}
            for key in rollout_batch.keys():
                if "env_info/" not in key:
                    rollout_batch_i[key] = torch.chunk(
                        rollout_batch[key], split_num, dim=1
                    )[i].contiguous()
                else:
                    rollout_batch_i[key] = torch.chunk(
                        rollout_batch[key], split_num, dim=0
                    )[i].contiguous()
            await self.channel.put(
                item=rollout_batch_i, queue_name=self._replay_buffer_name, async_op=True
            ).async_wait()
