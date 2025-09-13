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

from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.env_manager import EnvManager
from rlinf.scheduler import Worker
from rlinf.utils.placement import HybridComponentPlacement


def put_tensor_cpu(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_cpu(value)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.cpu().contiguous()
    return data_dict


def create_env_batch(obs, rews, dones, infos, meta=None):
    ret_dict = {"obs": obs, "rews": rews, "dones": dones, "infos": infos}
    if meta is not None:
        ret_dict.update(meta=meta)

    ret_dict = put_tensor_cpu(ret_dict)
    return ret_dict


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self._actor_group_name = cfg.actor.group_name
        self._rollout_group_name = cfg.rollout.group_name

        self.simulator_list = []
        self.last_obs_list = []
        self.last_dones_list = []
        self.eval_simulator_list = []

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name

        self._component_placement = HybridComponentPlacement(cfg)
        assert (
            self._component_placement.get_world_size("rollout")
            % self._component_placement.get_world_size("env")
            == 0
        )
        # gather_num: number of rollout for each env process
        self.gather_num = self._component_placement.get_world_size(
            "rollout"
        ) // self._component_placement.get_world_size("env")
        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = self.cfg.rollout.pipeline_stage_num
        self.batch_size = self.cfg.env.train.num_group * self.cfg.env.train.group_size
        self.eval_batch_size = (
            self.cfg.env.eval.num_group * self.cfg.env.eval.group_size
        )

        # only need rank0 to create channel
        if self._rank == 0:
            self.channel = self.create_channel(cfg.env.channel.name)
        else:
            self.channel = self.connect_channel(cfg.env.channel.name)
        for i in range(self.gather_num):
            self.channel.create_queue(
                f"{self._obs_queue_name}_{i + self._rank * self.gather_num}",
                maxsize=cfg.env.channel.queue_size,
            )

    def init_worker(self):
        enable_offload = self.cfg.env.enable_offload
        only_eval = getattr(self.cfg.runner, "only_eval", False)
        if self.cfg.env.train.simulator_type == "maniskill":
            from rlinf.envs.maniskill.maniskill_env import ManiskillEnv

            if not only_eval:
                for _ in range(self.stage_num):
                    self.simulator_list.append(
                        EnvManager(
                            self.cfg.env.train,
                            rank=self._rank,
                            world_size=self._world_size,
                            env_cls=ManiskillEnv,
                            enable_offload=enable_offload,
                        )
                    )
            if self.cfg.runner.val_check_interval > 0 or only_eval:
                for _ in range(self.stage_num):
                    self.eval_simulator_list.append(
                        EnvManager(
                            self.cfg.env.eval,
                            rank=self._rank,
                            world_size=self._world_size,
                            env_cls=ManiskillEnv,
                            enable_offload=enable_offload,
                        )
                    )
        elif self.cfg.env.train.simulator_type == "libero":
            from rlinf.envs.libero.libero_env import LiberoEnv

            if not only_eval:
                for _ in range(self.stage_num):
                    self.simulator_list.append(
                        EnvManager(
                            self.cfg.env.train,
                            rank=self._rank,
                            world_size=self._world_size,
                            env_cls=LiberoEnv,
                            enable_offload=enable_offload,
                        )
                    )
            if self.cfg.runner.val_check_interval > 0 or only_eval:
                for _ in range(self.stage_num):
                    self.eval_simulator_list.append(
                        EnvManager(
                            self.cfg.env.eval,
                            rank=self._rank,
                            world_size=self._world_size,
                            env_cls=LiberoEnv,
                            enable_offload=enable_offload,
                        )
                    )
        elif self.cfg.env.train.simulator_type == "robotwin":
            from rlinf.envs.robotwin.RoboTwin_env import RoboTwin

            if not only_eval:
                for _ in range(self.stage_num):
                    self.simulator_list.append(
                        EnvManager(
                            self.cfg.env.train,
                            rank=self._rank,
                            world_size=self._world_size,
                            env_cls=RoboTwin,
                            enable_offload=enable_offload,
                        )
                        # RoboTwin(self.cfg.env.train, rank=self._rank, world_size=self._world_size)
                    )
            if self.cfg.runner.val_check_interval > 0 or only_eval:
                for _ in range(self.stage_num):
                    self.eval_simulator_list.append(
                        EnvManager(
                            self.cfg.env.eval,
                            rank=self._rank,
                            world_size=self._world_size,
                            env_cls=RoboTwin,
                            enable_offload=enable_offload,
                        )
                    )
        else:
            raise NotImplementedError(
                f"Simulator type {self.cfg.env.train.simulator_type} not implemented"
            )

        if not only_eval:
            self._init_simulator()

    def _init_simulator(self):
        for i in range(self.stage_num):
            self.simulator_list[i].start_simulator()
            extracted_obs, rewards, terminations, truncations, infos = (
                self.simulator_list[i].step()
            )
            self.last_obs_list.append(extracted_obs)
            dones = torch.logical_or(terminations, truncations)
            self.last_dones_list.append(
                dones.unsqueeze(1).repeat(1, self.cfg.actor.model.num_action_chunks)
            )
            self.simulator_list[i].stop_simulator()

    def env_interact_step(self, chunk_actions: torch.Tensor, stage_id: int) -> Dict:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            simulator_type=self.cfg.env.train.simulator_type,
            raw_chunk_actions=chunk_actions,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
        )
        env_info_list = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.simulator_list[stage_id].chunk_step(chunk_actions)
        )
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info_list[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info_list[key] = final_info["episode"][key][
                        chunk_dones[:, -1]
                    ].cpu()

        env_batch = create_env_batch(
            obs=extracted_obs,
            rews=chunk_rewards,
            dones=chunk_dones,
            infos=infos,
            meta=env_info_list,
        )
        return env_batch

    def env_evaluate_step(self, raw_actions: torch.Tensor, stage_id: int) -> Dict:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            simulator_type=self.cfg.env.train.simulator_type,
            raw_chunk_actions=raw_actions,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
        )
        env_info_list = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.eval_simulator_list[stage_id].chunk_step(chunk_actions)
        )
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            final_info = infos["final_info"]
            for key in final_info["episode"]:
                env_info_list[key] = final_info["episode"][key][
                    chunk_dones[:, -1]
                ].cpu()

        env_batch = create_env_batch(
            obs=extracted_obs, rews=None, dones=None, infos=infos, meta=env_info_list
        )
        return env_batch

    async def recv_chunk_actions(self):
        chunk_action = []
        for gather_id in range(self.gather_num):
            chunk_action.append(
                await self.channel.get(
                    queue_name=f"{self._action_queue_name}_{gather_id + self._rank * self.gather_num}",
                    async_op=True,
                ).async_wait()
            )
        chunk_action = np.concatenate(chunk_action, axis=0)
        return chunk_action

    def concat_tensor(self, tensor_dict):
        for key, value in tensor_dict.items():
            if "env_info/" not in key:
                tensor_dict[key] = torch.stack(value, dim=0).contiguous()
            else:
                tensor_dict[key] = torch.cat(value, dim=0).contiguous()
        return tensor_dict

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            if self.cfg.env.train.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.simulator_list[i].flush_video(video_sub_dir=f"stage_{i}")
            for i in range(self.stage_num):
                self.simulator_list[i].update_reset_state_ids()
        elif mode == "eval":
            if self.cfg.env.eval.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.eval_simulator_list[i].flush_video(video_sub_dir=f"stage_{i}")

    def split_env_batch(self, env_batch, gather_id, mode):
        env_batch_i = {}
        for key, value in env_batch.items():
            if isinstance(value, torch.Tensor):
                env_batch_i[key] = value.chunk(self.gather_num, dim=0)[
                    gather_id
                ].contiguous()
            elif isinstance(value, list):
                length = len(value)
                if mode == "train":
                    assert length == self.batch_size, (
                        f"key {key}, length: {length}, batch_size: {self.batch_size}"
                    )
                elif mode == "eval":
                    assert length == self.eval_batch_size, (
                        f"key {key}, length: {length}, batch_size: {self.eval_batch_size}"
                    )
                env_batch_i[key] = value[
                    gather_id * length // self.gather_num : (gather_id + 1)
                    * length
                    // self.gather_num
                ]
            elif isinstance(value, dict):
                env_batch_i[key] = self.split_env_batch(value, gather_id, mode)
            else:
                env_batch_i[key] = value
        return env_batch_i

    async def send_env_batch(self, env_batch, mode="train"):
        # split env_batch into num_processes chunks, each chunk contains gather_num env_batch
        for gather_id in range(self.gather_num):
            env_batch_i = self.split_env_batch(env_batch, gather_id, mode)
            await self.channel.put(
                item=env_batch_i,
                queue_name=f"{self._obs_queue_name}_{gather_id + self._rank * self.gather_num}",
                async_op=True,
            ).async_wait()

    async def interact(self):
        for simulator in self.simulator_list:
            simulator.start_simulator()
        for rollout_epoch in range(self.cfg.algorithm.rollout_epoch):
            env_batch_list = []
            if not self.cfg.env.train.auto_reset:
                for i in range(self.stage_num):
                    self.simulator_list[i].is_start = True
                    extracted_obs, rewards, terminations, truncations, infos = (
                        self.simulator_list[i].step()
                    )
                    dones = (
                        torch.logical_or(terminations, truncations)
                        .unsqueeze(1)
                        .repeat(1, self.cfg.actor.model.num_action_chunks)
                    )
                    env_batch = create_env_batch(extracted_obs, rewards, dones, infos)
                    env_batch_list.append(env_batch)
            else:
                self.num_done_envs = 0
                self.num_succ_envs = 0
                infos = {}
                for i in range(self.stage_num):
                    env_batch = create_env_batch(
                        self.last_obs_list[i], None, self.last_dones_list[i], infos
                    )
                    env_batch_list.append(env_batch)

            for stage_id in range(self.stage_num):
                env_batch = env_batch_list[stage_id]
                await self.send_env_batch(env_batch)

            for _ in range(self.cfg.algorithm.n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = await self.recv_chunk_actions()
                    env_batch = self.env_interact_step(raw_chunk_actions, stage_id)
                    await self.send_env_batch(env_batch)
                    env_batch_list[stage_id] = env_batch

            self.last_obs_list = [env_batch["obs"] for env_batch in env_batch_list]
            self.last_dones_list = [env_batch["dones"] for env_batch in env_batch_list]
            self.finish_rollout()

        for simulator in self.simulator_list:
            simulator.stop_simulator()

    async def evaluate(self):
        for i in range(self.stage_num):
            self.eval_simulator_list[i].start_simulator()
            self.eval_simulator_list[i].is_start = True
            extracted_obs, rewards, terminations, truncations, infos = (
                self.eval_simulator_list[i].step()
            )
            env_batch = create_env_batch(extracted_obs, None, None, infos)
            await self.send_env_batch(env_batch, mode="eval")

        for eval_step in range(self.cfg.algorithm.n_eval_chunk_steps):
            for i in range(self.stage_num):
                raw_chunk_actions = await self.recv_chunk_actions()
                env_batch = self.env_evaluate_step(raw_chunk_actions, i)
                await self.send_env_batch(env_batch, mode="eval")

        self.finish_rollout(mode="eval")
        for i in range(self.stage_num):
            self.eval_simulator_list[i].stop_simulator()
