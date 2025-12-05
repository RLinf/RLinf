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
from omegaconf import DictConfig

from rlinf.data.io_struct import EnvOutput
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.env_manager import EnvManager
from rlinf.scheduler import Channel, Worker


class EnvWorker(Worker):
    """The EnvWorker is responsible for controlling the embodied environments like simulators or physical robots.

    It calls the corresponding gym env's step function to generate observations, rewards, and done signals based on the actions received from the RolloutWorker, and sends them back to the RolloutWorker.

    The EnvWorker supports running multiple environment instances in parallel to improve data collection efficiency.
    The main entry point is the `interact` method, which performs environment interactions for a specified number of steps (called chunk_step) and put the collected environment metrics into an output channel to the RolloutWorker.

    Specially, the EnvWorker supports pipeline rollout process, where the parallel environment instances are further divided into multiple stages. Each stage interacts with the environment sequentially, while different stages can run in parallel. This design helps to further improve the efficiency of data collection.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the EnvWorker.

        Args:
            cfg (DictConfig): The configuration for the EnvWorker.
        """
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0

        # Some EnvWorker ranks may run faster than others, leading to them putting future step data into the channel before other ranks have finished the current step.
        # This causes RolloutWorker to read data out-of-order from the channel, resulting in incorrect behavior.
        # To solve this problem, we add a step counter to the EnvWorker to ensure that data is put and get from the channel in the correct order.
        self.put_batch_cnt = 0

        self.train_env_list: list[EnvManager] = []
        self.eval_env_list: list[EnvManager] = []
        self.last_obs_list = []
        self.last_dones_list = []

        # Used for pipelined rollout interactions
        self.num_pipeline_stages = self.cfg.rollout.pipeline_stage_num

        # Env configurations
        self.eval_only = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.eval_only
        if not self.eval_only:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs
                // self._world_size
                // self.num_pipeline_stages
            )
            self.train_num_group_envs_per_stage = (
                self.train_num_envs_per_stage // self.cfg.env.train.group_size
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs
                // self._world_size
                // self.num_pipeline_stages
            )
            self.eval_num_group_envs_per_stage = (
                self.eval_num_envs_per_stage // self.cfg.env.eval.group_size
            )

        self.env_type = cfg.env.train.simulator_type

    def init_worker(self):
        """Create the environment instances for the EnvWorker and start the environments."""
        enable_offload = self.cfg.env.enable_offload
        total_num_processes = self._world_size * self.num_pipeline_stages

        for stage_id in range(self.num_pipeline_stages):
            seed_offset = self._rank * self.num_pipeline_stages + stage_id
            if not self.eval_only:
                self.train_env_list.append(
                    EnvManager(
                        self.cfg,
                        rank=self._rank,
                        num_envs=self.train_num_envs_per_stage,
                        seed_offset=seed_offset,
                        total_num_processes=total_num_processes,
                        env_type=self.env_type,
                        is_eval=False,
                        enable_offload=enable_offload,
                    )
                )
            if self.enable_eval:
                self.eval_env_list.append(
                    EnvManager(
                        self.cfg,
                        rank=self._rank,
                        num_envs=self.eval_num_envs_per_stage,
                        seed_offset=seed_offset,
                        total_num_processes=total_num_processes,
                        env_type=self.env_type,
                        is_eval=True,
                        enable_offload=enable_offload,
                    )
                )

        if not self.eval_only:
            self._init_envs()

    def _init_envs(self):
        """Initialize the training environments and store the initial observations and done signals."""
        for i in range(self.num_pipeline_stages):
            self.train_env_list[i].start_env()
            extracted_obs, _ = self.train_env_list[i].reset()
            dones = (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )
            self.last_obs_list.append(extracted_obs)
            self.last_dones_list.append(dones)
            self.train_env_list[i].stop_env()

    def _env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """A single interact step with the environment."""
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.train_env_list[stage_id].chunk_step(chunk_actions)
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
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
            worker_rank=self._rank,
            stage_id=stage_id,
            num_group_envs=self.train_num_group_envs_per_stage,
            group_size=self.cfg.env.train.group_size,
        )
        return env_output, env_info

    def _env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """A single evaluate step with the environment."""
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
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
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            worker_rank=self._rank,
            stage_id=stage_id,
            num_group_envs=self.eval_num_group_envs_per_stage,
            group_size=self.cfg.env.eval.group_size,
        )
        return env_output, env_info

    def _env_reset_step(self, stage_id: int):
        if not self.cfg.env.train.auto_reset:
            obs, infos = self.train_env_list[stage_id].reset()
            self.last_obs_list.append(obs)
            dones = (
                torch.zeros((self.train_num_envs_per_stage,), dtype=torch.bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )
            self.last_dones_list.append(dones)
            final_obs = infos.get("final_observation", None)
        else:
            obs = self.last_obs_list[stage_id]
            dones = self.last_dones_list[stage_id]
            final_obs = None

        return EnvOutput(
            obs=obs,
            dones=dones,
            final_obs=final_obs,
            worker_rank=self._rank,
            stage_id=stage_id,
            num_group_envs=self.train_num_group_envs_per_stage,
            group_size=self.cfg.env.train.group_size,
        )

    def _finish_rollout(self, mode="train"):
        """Finish the rollout process by flushing videos and updating reset states."""
        if mode == "train":
            if self.cfg.env.train.video_cfg.save_video:
                for i in range(self.num_pipeline_stages):
                    self.train_env_list[i].flush_video()
            for i in range(self.num_pipeline_stages):
                self.train_env_list[i].update_reset_state_ids()
        elif mode == "eval":
            if self.cfg.env.eval.video_cfg.save_video:
                for i in range(self.num_pipeline_stages):
                    self.eval_env_list[i].flush_video()
            for i in range(self.num_pipeline_stages):
                self.eval_env_list[i].update_reset_state_ids()

    def get_actions(
        self, input_channel: Channel, stage_id: int, num_group_envs: int
    ) -> np.ndarray:
        """This function is used to get the actions from the input channel.

        It retrieves a list of chunk actions from the input channel with the key (worker_rank, stage_id).
        The retrieved chunk actions are also accompanied with the group_env id, which indicates the group_env index of the chunk action in the current stage.
        After retrieving all the chunk actions for the current stage, they are first sorted according to the group_env id to ensure the correct order, and then concatenated into a single numpy array before being returned.
        """
        chunk_action = []
        for _ in range(num_group_envs):
            group_env_id, action = input_channel.get(key=(self._rank, stage_id))
            chunk_action.append((group_env_id, action))
        chunk_action.sort(key=lambda x: x[0])
        chunk_action = np.concatenate([action for _, action in chunk_action], axis=0)
        return chunk_action

    def put_batch(
        self,
        output_channel: Channel,
        env_output: EnvOutput,
    ):
        """This function is used to put the environment output into the output channel.

        It accepts an env_output and split it into a list of env_outputs containing num_envs env_outputs and put them into the output channel along with three identifiers: the env id, the pipeline stage id, and the worker rank, which are used by the RolloutWorker to put the corresponding actions back to the correct environment instances.

        Args:
            output_channel (Channel): The output channel to put the environment output.
            env_output (EnvOutput): The environment output to be put into the output channel.
        """
        env_outputs = env_output.split_by_group()
        for env_output in env_outputs:
            output_channel.put(item=env_output, key=self.put_batch_cnt)
        self.put_batch_cnt += 1

    def interact(self, input_channel: Channel, output_channel: Channel):
        """The main entry point for environment interaction.

        Args:
            input_channel (Channel): The input channel to receive actions from the RolloutWorker.
            output_channel (Channel): The output channel to send environment outputs to the RolloutWorker.
        """
        for env in self.train_env_list:
            env.start_env()

        n_chunk_steps = (
            self.cfg.env.train.max_episode_steps
            // self.cfg.actor.model.num_action_chunks
        )

        env_metrics = defaultdict(list)
        self.device_lock.acquire()
        for epoch in range(self.cfg.algorithm.rollout_epoch):
            env_output_list: list[EnvOutput] = []

            # Reset environments at the beginning of each epoch
            for stage_id in range(self.num_pipeline_stages):
                with self.worker_timer():
                    env_output = self._env_reset_step(stage_id)
                self.put_batch(output_channel, env_output)
                env_output_list.append(env_output)

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    # Retrieve actions from the RolloutWorker
                    self.device_lock.release()  # Release lock to allow RolloutWorker to run
                    raw_chunk_actions = self.get_actions(
                        input_channel, stage_id, self.train_num_group_envs_per_stage
                    )
                    self.device_lock.acquire()  # Re-acquire lock for environment interaction

                    # Environment interaction using the actions
                    with self.worker_timer():
                        env_output, env_info = self._env_interact_step(
                            raw_chunk_actions, stage_id
                        )

                    # Put the results into the output channel
                    self.put_batch(output_channel, env_output)
                    env_output_list[stage_id] = env_output

                    # Collect environment info metrics
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
            self._finish_rollout()

        for env in self.train_env_list:
            env.stop_env()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        self.device_lock.release()

        return env_metrics

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        """The main entry point for environment evaluation.

        Args:
            input_channel (Channel): The input channel to receive actions from the RolloutWorker.
            output_channel (Channel): The output channel to send environment outputs to the RolloutWorker.
        """
        eval_metrics = defaultdict(list)

        n_chunk_steps = (
            self.cfg.env.eval.max_episode_steps
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            for stage_id in range(self.num_pipeline_stages):
                self.eval_env_list[stage_id].start_env()
                self.eval_env_list[stage_id].is_start = True
                extracted_obs, infos = self.eval_env_list[stage_id].reset()
                env_output = EnvOutput(
                    obs=extracted_obs,
                    final_obs=infos.get("final_observation", None),
                    worker_rank=self._rank,
                    stage_id=stage_id,
                    num_group_envs=self.eval_num_group_envs_per_stage,
                    group_size=self.cfg.env.eval.group_size,
                )
                self.put_batch(output_channel, env_output)

            for eval_step in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    raw_chunk_actions = self.get_actions(
                        input_channel, stage_id, self.eval_num_group_envs_per_stage
                    )
                    env_output, env_info = self._env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)
                    if eval_step == n_chunk_steps - 1:
                        continue
                    self.put_batch(output_channel, env_output)

            self._finish_rollout(mode="eval")
        for stage_id in range(self.num_pipeline_stages):
            self.eval_env_list[stage_id].stop_env()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics
