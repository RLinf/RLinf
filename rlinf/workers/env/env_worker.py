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

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from rlinf.data.io_struct import EnvOutput
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.env_manager import EnvManager
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


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

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
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
        assert self.cfg.env.train.num_envs == self.batch_size

        # only need rank0 to create channel
        if self._rank == 0:
            self.channel = self.create_channel(cfg.env.channel.name)
        else:
            self.channel = self.connect_channel(cfg.env.channel.name)

    def init_worker(self):
        enable_offload = self.cfg.env.enable_offload
        only_eval = getattr(self.cfg.runner, "only_eval", False)
        if self.cfg.env.train.simulator_type == "maniskill":
            from rlinf.envs.maniskill.maniskill_env import ManiskillEnv

            if not only_eval:
                for stage_id in range(self.stage_num):
                    self.simulator_list.append(
                        EnvManager(
                            self.cfg.env.train,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=ManiskillEnv,
                            enable_offload=enable_offload,
                        )
                    )
            if self.cfg.runner.val_check_interval > 0 or only_eval:
                for stage_id in range(self.stage_num):
                    self.eval_simulator_list.append(
                        EnvManager(
                            self.cfg.env.eval,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=ManiskillEnv,
                            enable_offload=enable_offload,
                        )
                    )
        elif self.cfg.env.train.simulator_type == "libero":
            from rlinf.envs.libero.libero_env import LiberoEnv

            if not only_eval:
                for stage_id in range(self.stage_num):
                    self.simulator_list.append(
                        EnvManager(
                            self.cfg.env.train,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=LiberoEnv,
                            enable_offload=enable_offload,
                        )
                    )
            if self.cfg.runner.val_check_interval > 0 or only_eval:
                for stage_id in range(self.stage_num):
                    self.eval_simulator_list.append(
                        EnvManager(
                            self.cfg.env.eval,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=LiberoEnv,
                            enable_offload=enable_offload,
                        )
                    )
        elif self.cfg.env.train.simulator_type == "robotwin":
            from rlinf.envs.robotwin.RoboTwin_env import RoboTwin

            if not only_eval:
                for stage_id in range(self.stage_num):
                    self.simulator_list.append(
                        EnvManager(
                            self.cfg.env.train,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=RoboTwin,
                            enable_offload=enable_offload,
                        )
                        # RoboTwin(self.cfg.env.train, rank=self._rank, total_num_processes=self._world_size)
                    )
            if self.cfg.runner.val_check_interval > 0 or only_eval:
                for stage_id in range(self.stage_num):
                    self.eval_simulator_list.append(
                        EnvManager(
                            self.cfg.env.eval,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=RoboTwin,
                            enable_offload=enable_offload,
                        )
                    )
        elif self.cfg.env.train.simulator_type == "isaaclab":
            from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

            assert self.cfg.env.train.init_params.id in REGISTER_ISAACLAB_ENVS, (
                f"Task type {self.cfg.env.train.init_params.id} have not been registered!"
            )
            isaaclab_env_cls = REGISTER_ISAACLAB_ENVS[self.cfg.env.train.init_params.id]
            if not only_eval:
                for stage_id in range(self.stage_num):
                    self.simulator_list.append(
                        EnvManager(
                            self.cfg.env.train,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=isaaclab_env_cls,
                            enable_offload=enable_offload,
                        )
                    )
            if self.cfg.runner.val_check_interval > 0 or only_eval:
                for stage_id in range(self.stage_num):
                    self.eval_simulator_list.append(
                        EnvManager(
                            self.cfg.env.eval,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=isaaclab_env_cls,
                            enable_offload=enable_offload,
                        )
                    )
        elif self.cfg.env.train.simulator_type == "metaworld":
            from rlinf.envs.metaworld.metaworld_env import MetaWorldEnv

            if not only_eval:
                for stage_id in range(self.stage_num):
                    self.simulator_list.append(
                        EnvManager(
                            self.cfg.env.train,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=MetaWorldEnv,
                            enable_offload=enable_offload,
                        )
                    )
            if self.cfg.runner.val_check_interval > 0 or only_eval:
                for stage_id in range(self.stage_num):
                    self.eval_simulator_list.append(
                        EnvManager(
                            self.cfg.env.eval,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=MetaWorldEnv,
                            enable_offload=enable_offload,
                        )
                    )
        elif self.cfg.env.train.simulator_type == "behavior":
            with open_dict(self.cfg):
                # self.cfg.env.train.tasks.task_idx = self.cfg.env.train.tasks.activity_task_indices[self._rank]
                self.cfg.env.train.tasks.task_idx = 0
                self.cfg.env.eval.tasks.task_idx = 0
            from rlinf.envs.behavior.behavior_env import BehaviorEnv

            if not only_eval:
                for stage_id in range(self.stage_num):
                    self.simulator_list.append(
                        EnvManager(
                            self.cfg.env.train,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=BehaviorEnv,
                            enable_offload=enable_offload,
                        )
                    )
            if self.cfg.runner.val_check_interval > 0 or only_eval:
                for stage_id in range(self.stage_num):
                    self.eval_simulator_list.append(
                        EnvManager(
                            self.cfg.env.eval,
                            rank=self._rank,
                            seed_offset=self._rank * self.stage_num + stage_id,
                            total_num_processes=self._world_size * self.stage_num,
                            env_cls=BehaviorEnv,
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

    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            simulator_type=self.cfg.env.train.simulator_type,
            model_name=self.cfg.actor.model.model_name,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

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
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            simulator_type=self.cfg.env.train.simulator_type,
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
        )
        return env_output, env_info

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            simulator_type=self.cfg.env.train.simulator_type,
            model_name=self.cfg.actor.model.model_name,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.eval_simulator_list[stage_id].chunk_step(chunk_actions)
        )
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            if "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key].cpu()
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            simulator_type=self.cfg.env.train.simulator_type,
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
        )
        return env_output, env_info

    def recv_chunk_actions(self):
        chunk_action = []
        for gather_id in range(self.gather_num):
            chunk_action.append(
                self.channel.get(
                    key=f"{self._action_queue_name}_{gather_id + self._rank * self.gather_num}",
                )
            )
        chunk_action = np.concatenate(chunk_action, axis=0)
        return chunk_action

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            if self.cfg.env.train.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.simulator_list[i].flush_video()
            for i in range(self.stage_num):
                self.simulator_list[i].update_reset_state_ids()
        elif mode == "eval":
            if self.cfg.env.eval.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.eval_simulator_list[i].flush_video()

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

    def send_env_batch(self, env_batch, mode="train"):
        # split env_batch into num_processes chunks, each chunk contains gather_num env_batch
        for gather_id in range(self.gather_num):
            env_batch_i = self.split_env_batch(env_batch, gather_id, mode)
            self.channel.put(
                item=env_batch_i,
                key=f"{self._obs_queue_name}_{gather_id + self._rank * self.gather_num}",
            )

    def interact(self):
        for simulator in self.simulator_list:
            simulator.start_simulator()

        env_metrics = defaultdict(list)
        for epoch in range(self.cfg.algorithm.rollout_epoch):
            env_output_list = []
            if not self.cfg.env.train.auto_reset:
                for i in range(self.stage_num):
                    extracted_obs, infos = self.simulator_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
                    dones = (
                        torch.zeros((self.cfg.env.train.num_envs,), dtype=bool)
                        .unsqueeze(1)
                        .repeat(1, self.cfg.actor.model.num_action_chunks)
                    )
                    self.last_dones_list.append(dones)
                    env_output = EnvOutput(
                        simulator_type=self.cfg.env.train.simulator_type,
                        obs=extracted_obs,
                        dones=dones,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                    )
                    env_output_list.append(env_output)
            else:
                self.num_done_envs = 0
                self.num_succ_envs = 0
                for i in range(self.stage_num):
                    env_output = EnvOutput(
                        simulator_type=self.cfg.env.train.simulator_type,
                        obs=self.last_obs_list[i],
                        rewards=None,
                        dones=self.last_dones_list[i],
                    )
                    env_output_list.append(env_output)

            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_output_list[stage_id]
                self.send_env_batch(env_output.to_dict())

            for _ in range(self.cfg.algorithm.n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions()
                    env_output, env_info = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    self.send_env_batch(env_output.to_dict())
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

            self.last_obs_list = [env_output.obs for env_output in env_output_list]
            self.last_dones_list = [env_output.dones for env_output in env_output_list]
            self.finish_rollout()

        for simulator in self.simulator_list:
            simulator.stop_simulator()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    def evaluate(self):
        for i in range(self.stage_num):
            self.eval_simulator_list[i].start_simulator()
            self.eval_simulator_list[i].is_start = True
            extracted_obs, _, _, _, infos = self.eval_simulator_list[i].step()
            env_output = EnvOutput(
                simulator_type=self.cfg.env.eval.simulator_type,
                obs=extracted_obs,
                final_obs=infos["final_observation"]
                if "final_observation" in infos
                else None,
            )
            self.send_env_batch(env_output.to_dict(), mode="eval")

        eval_metrics = defaultdict(list)

        for eval_step in range(self.cfg.algorithm.n_eval_chunk_steps):
            for i in range(self.stage_num):
                raw_chunk_actions = self.recv_chunk_actions()
                env_output, env_info = self.env_evaluate_step(raw_chunk_actions, i)

                for key, value in env_info.items():
                    eval_metrics[key].append(value)
                if eval_step == self.cfg.algorithm.n_eval_chunk_steps - 1:
                    continue
                self.send_env_batch(env_output.to_dict(), mode="eval")

        self.finish_rollout(mode="eval")
        for i in range(self.stage_num):
            self.eval_simulator_list[i].stop_simulator()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics
