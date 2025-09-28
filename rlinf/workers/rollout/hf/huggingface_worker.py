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
        for i in range(self._component_placement.get_world_size("rollout")):
            self.channel.create_queue(
                f"{self._action_queue_name}_{i}", maxsize=cfg.rollout.channel.queue_size
            )

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

        if self.cfg.actor.model.model_name in ["openpi"]:
            kwargs = {"mode": mode}

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def update_env_output(self, i, env_output):
        # first step for env_batch
        if env_output["rewards"] is None:
            self.buffer_list[i].dones.append(env_output["dones"].contiguous().cpu())
            return

        self.buffer_list[i].rewards.append(env_output["rewards"].cpu().contiguous())
        self.buffer_list[i].dones.append(env_output["dones"].bool().cpu().contiguous())

        # Note: currently this is not correct for chunk-size>1 with partial reset
        if env_output["dones"].any() and self.cfg.env.train.auto_reset:
            if self.cfg.algorithm.require_values:
                dones = env_output["dones"]

                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    actions, result = self.predict(final_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                self.buffer_list[i].rewards[-1][:, -1] += (
                    self.cfg.algorithm.gamma * final_values.cpu()
                )

    async def generate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()
        self.buffer_list = [EmbodiedRolloutResult() for _ in range(self.stage_num)]

        for rollout_epoch in range(self.cfg.algorithm.rollout_epoch):
            self._logger.info(f"Now epoch is={rollout_epoch}")
            for _ in tqdm(
                range(self.cfg.algorithm.n_chunk_steps),
                desc=f"Rollout ID {self._rank} Epoch {rollout_epoch} in Generate Step",
            ):
                for i in range(self.stage_num):
                    env_output = await self.recv_env_output()
                    self.update_env_output(i, env_output)
                    actions, result = self.predict(env_output["obs"])

                    self.buffer_list[i].append_result(result)

                    await self.send_chunk_actions(actions)

            for i in range(self.stage_num):
                env_output = await self.recv_env_output()
                self.update_env_output(i, env_output)
                actions, result = self.predict(env_output["obs"])
                if "prev_values" in result:
                    self.buffer_list[i].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )

        for i in range(self.stage_num):
            await self.send_rollout_batch(i)

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    async def evaluate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()

        for _ in tqdm(
            range(self.cfg.algorithm.n_eval_chunk_steps), desc="Rollout in Eval Step"
        ):
            for _ in range(self.stage_num):
                env_output = await self.recv_env_output()
                actions, _ = self.predict(env_output["obs"], mode="eval")
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
            queue_name=f"{self._obs_queue_name}_{self._rank}", async_op=True
        ).async_wait()
        return env_output

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
        splited_rollout_result = self.buffer_list[stage_id].to_splited_dict(split_num)
        for i in range(split_num):
            await self.channel.put(
                item=splited_rollout_result[i],
                queue_name=self._replay_buffer_name,
                async_op=True,
            ).async_wait()

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
