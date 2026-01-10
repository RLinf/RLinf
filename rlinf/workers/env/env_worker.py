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

import json
import os
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from rlinf.data.io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import normalize_openpi_state, prepare_actions
from rlinf.envs.env_manager import EnvManager
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list: list[EnvManager] = []
        self.eval_env_list: list[EnvManager] = []

        self.last_obs_list = []
        self.last_dones_list = []
        self.last_terminations_list = []
        self.last_truncations_list = []
        self.last_intervened_info_list = []

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

        # Env configurations
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )

    def _check_unnorm_key(
        self, norm_stats: dict[str, dict[str, Any]], unnorm_key: Optional[str]
    ) -> str:
        if unnorm_key not in norm_stats and f"{unnorm_key}_no_noops" in norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in norm_stats, (
            f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
        )

        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            else:
                raise ValueError(
                    f"Your model was trained on more than one dataset, "
                    f"please pass a `unnorm_key` from the following options to choose the statistics "
                    f"used for un-normalizing actions: {list(norm_stats.keys())}"
                )

        # Try to find the key, with fallback to _no_noops variant
        if unnorm_key not in norm_stats:
            if f"{unnorm_key}_no_noops" in norm_stats:
                unnorm_key = f"{unnorm_key}_no_noops"
            else:
                # Provide helpful error message with available keys
                available_keys = list(norm_stats.keys())
                raise ValueError(
                    f"Action un-norm key '{unnorm_key}' not found in norm_stats! "
                    f"Available keys: {available_keys}. "
                    f"Please check your configuration and use one of the available keys."
                )

        return unnorm_key

    def _get_norm_stats(
        self, stats_file_path: str, unnorm_key: Optional[str]
    ) -> dict[str, Any]:
        """Get all the logged statistics for the given dataset.

        Supports two formats:
        1. config.json: reads from config["norm_stats"]
        2. dataset_statistics.json or norm_stats.json: reads the entire file as norm_stats
        """

        if not os.path.isfile(stats_file_path):
            raise FileNotFoundError(
                f"Stats file not found: {stats_file_path}. "
                f"Please check the path in your configuration file."
            )

        with open(stats_file_path, "r") as f:
            data = json.load(f)

        # Check if this is a config.json file (has norm_stats field)
        if "norm_stats" in data:
            norm_stats = data["norm_stats"]
        else:
            # Assume this is a dataset_statistics.json or norm_stats.json file
            norm_stats = data

        # Check if this is OpenPI format (directly contains 'state' and 'actions' keys)
        # In this case, we don't need unnorm_key
        if "state" in norm_stats or "actions" in norm_stats or "action" in norm_stats:
            # OpenPI format: norm_stats directly contains state and actions
            if "state" in norm_stats:
                state_norm_stats = norm_stats["state"]
            else:
                state_norm_stats = None
            if "action" in norm_stats:
                action_norm_stats = norm_stats["action"]
            elif "actions" in norm_stats:
                action_norm_stats = norm_stats["actions"]
            else:
                raise ValueError(
                    f"Neither 'action' nor 'actions' key found in norm_stats file: {stats_file_path}"
                )
            return action_norm_stats, state_norm_stats

        # Otherwise, treat as nested format (e.g., {"libero_10": {"state": ..., "actions": ...}})
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        if "state" in norm_stats[unnorm_key]:
            state_norm_stats = norm_stats[unnorm_key]["state"]
        else:
            state_norm_stats = None
        if "action" in norm_stats[unnorm_key]:
            action_norm_stats = norm_stats[unnorm_key]["action"]
        else:
            action_norm_stats = norm_stats[unnorm_key]["actions"]
        return action_norm_stats, state_norm_stats

    def init_worker(self):
        enable_offload = self.cfg.env.enable_offload

        self.broadcast(True, list(range(self._world_size)))
        if self.cfg.env.train.env_type == "mixed":
            train_simulator_id = self._rank % len(self.cfg.env.train.simulator_list)
            eval_simulator_id = self._rank % len(self.cfg.env.eval.simulator_list)
            self.train_env_type = self.cfg.env.train.simulator_list[
                train_simulator_id
            ].env_type
            self.eval_env_type = self.cfg.env.eval.simulator_list[
                eval_simulator_id
            ].env_type

            # merge simulator_list entry into cfg.env.train
            train_env_cfg = self.cfg.env.train.copy()
            env_train_resolved = OmegaConf.to_container(
                self.cfg.env.train, resolve=True
            )
            # Get the entire simulator_list entry and convert to dict
            simulator_config = OmegaConf.to_container(
                self.cfg.env.train.simulator_list[train_simulator_id], resolve=True
            )
            # Extract init_params (keep as nested field, don't expand)
            init_params = simulator_config.pop("init_params", None)
            # Remove fields that should not be merged into env config
            simulator_config.pop("env_type", None)
            simulator_config.pop("unnorm_key", None)
            simulator_config.pop("unnorm_key_file", None)
            # Merge: first env_train, then simulator_config fields (like task_suite_name)
            merged_dict = {**env_train_resolved, **simulator_config}
            # Add init_params as nested field if it exists
            if init_params is not None:
                merged_dict["init_params"] = init_params
            train_env_cfg = OmegaConf.create(merged_dict)
            train_env_cfg.env_type = self.train_env_type
            train_env_cls = get_env_cls(self.train_env_type, train_env_cfg)

            # get unnorm_keys for each env_type
            unnorm_key = self.cfg.env.train.simulator_list[
                train_simulator_id
            ].unnorm_key
            unnorm_key_file = self.cfg.env.train.simulator_list[
                train_simulator_id
            ].unnorm_key_file

            self.train_action_norm_stats, self.train_state_norm_stats = (
                self._get_norm_stats(unnorm_key_file, unnorm_key)
            )

            eval_env_cfg = self.cfg.env.eval.copy()
            env_eval_resolved = OmegaConf.to_container(self.cfg.env.eval, resolve=True)
            # Get the entire simulator_list entry and convert to dict
            simulator_config = OmegaConf.to_container(
                self.cfg.env.eval.simulator_list[eval_simulator_id], resolve=True
            )
            # Extract init_params (keep as nested field, don't expand)
            init_params = simulator_config.pop("init_params", None)
            # Remove fields that should not be merged into env config
            simulator_config.pop("env_type", None)
            simulator_config.pop("unnorm_key", None)
            simulator_config.pop("unnorm_key_file", None)
            # Merge: first env_eval, then simulator_config fields (like task_suite_name)
            merged_dict = {**env_eval_resolved, **simulator_config}
            # Add init_params as nested field if it exists
            if init_params is not None:
                merged_dict["init_params"] = init_params
            eval_env_cfg = OmegaConf.create(merged_dict)
            eval_env_cfg.env_type = self.eval_env_type
            eval_env_cls = get_env_cls(self.eval_env_type, eval_env_cfg)

            unnorm_key = self.cfg.env.eval.simulator_list[eval_simulator_id].unnorm_key
            unnorm_key_file = self.cfg.env.eval.simulator_list[
                eval_simulator_id
            ].unnorm_key_file

            self.eval_action_norm_stats, self.eval_state_norm_stats = (
                self._get_norm_stats(unnorm_key_file, unnorm_key)
            )

        else:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            train_env_cfg = self.cfg.env.train
            eval_env_cfg = self.cfg.env.eval
            self.train_env_type = self.cfg.env.train.env_type
            self.eval_env_type = self.cfg.env.eval.env_type

            unnorm_key = self.cfg.actor.model.unnorm_key
            # Use lora_path's config.json if lora_path exists, otherwise use model_path's config.json
            if (
                hasattr(self.cfg.actor.model, "lora_path")
                and self.cfg.actor.model.lora_path is not None
            ):
                config_path = os.path.join(
                    self.cfg.actor.model.lora_path, "config.json"
                )
            else:
                config_path = os.path.join(
                    self.cfg.actor.model.model_path, "config.json"
                )
            self.train_action_norm_stats, self.train_state_norm_stats = (
                self._get_norm_stats(config_path, unnorm_key)
            )
            self.eval_action_norm_stats, self.eval_state_norm_stats = (
                self._get_norm_stats(config_path, unnorm_key)
            )

        if not self.only_eval:
            for stage_id in range(self.stage_num):
                self.env_list.append(
                    EnvManager(
                        train_env_cfg,
                        rank=self._rank,
                        num_envs=self.train_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=train_env_cls,
                        worker_info=self.worker_info,
                        enable_offload=enable_offload,
                    )
                )
        if self.enable_eval:
            for stage_id in range(self.stage_num):
                self.eval_env_list.append(
                    EnvManager(
                        eval_env_cfg,
                        rank=self._rank,
                        num_envs=self.eval_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=eval_env_cls,
                        worker_info=self.worker_info,
                        enable_offload=enable_offload,
                    )
                )

        if not self.only_eval:
            self._init_env()

    def _init_env(self):
        if self.cfg.env.train.auto_reset:
            for i in range(self.stage_num):
                self.env_list[i].start_env()
                extracted_obs, _ = self.env_list[i].reset()
                if "states" in extracted_obs:
                    # print(f"Normalizing state with stats: {self.train_state_norm_stats}")
                    # print(extracted_obs.keys())
                    extracted_obs["states"] = normalize_openpi_state(
                        extracted_obs["states"], self.train_state_norm_stats
                    )
                    # print(extracted_obs["states"])
                    # for i in range(extracted_obs["states"].shape[-1]):
                    #     print(f"State {i}: {extracted_obs['states'][..., i]}")
                    # exit(0)
                dones = (
                    torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                    .unsqueeze(1)
                    .repeat(1, self.cfg.actor.model.num_action_chunks)
                )
                self.last_obs_list.append(extracted_obs)
                self.last_dones_list.append(dones)
                self.last_terminations_list.append(dones.clone())
                self.last_truncations_list.append(dones.clone())
                self.last_intervened_info_list.append((None, None))
                self.env_list[i].stop_env()

    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.train_env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            action_norm_stats=self.train_action_norm_stats,
        )
        env_info = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if "states" in extracted_obs:
            extracted_obs["states"] = normalize_openpi_state(
                extracted_obs["states"], self.train_state_norm_stats
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

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
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
            env_type=self.eval_env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            action_norm_stats=self.eval_action_norm_stats,
        )
        env_info = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        if "states" in extracted_obs:
            extracted_obs["states"] = normalize_openpi_state(
                extracted_obs["states"], self.eval_state_norm_stats
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
        )
        return env_output, env_info

    def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        chunk_action = []
        for gather_id in range(self.gather_num):
            chunk_action.append(
                input_channel.get(
                    key=f"{gather_id + self._rank * self.gather_num}_{mode}",
                )
            )
        chunk_action = np.concatenate(chunk_action, axis=0)
        return chunk_action

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            if self.cfg.env.train.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.env_list[i].flush_video()
            for i in range(self.stage_num):
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            if self.cfg.env.eval.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.eval_env_list[i].flush_video()
            if not self.cfg.env.eval.auto_reset:
                for i in range(self.stage_num):
                    self.eval_env_list[i].update_reset_state_ids()

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
                    assert length == self.train_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.train_num_envs_per_stage} "
                        f"(train_num_envs_per_stage), got {length}"
                    )
                elif mode == "eval":
                    assert length == self.eval_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.eval_num_envs_per_stage} "
                        f"(eval_num_envs_per_stage), got {length}"
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

    def send_env_batch(self, output_channel: Channel, env_batch, mode="train"):
        # split env_batch into num_processes chunks, each chunk contains gather_num env_batch
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        for gather_id in range(self.gather_num):
            env_batch_i = self.split_env_batch(env_batch, gather_id, mode)
            output_channel.put(
                item=env_batch_i,
                key=f"{gather_id + self._rank * self.gather_num}_{mode}",
            )

    def interact(self, input_channel: Channel, output_channel: Channel):
        for env in self.env_list:
            env.start_env()

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        env_metrics = defaultdict(list)
        for epoch in range(self.cfg.algorithm.rollout_epoch):
            env_output_list = []
            if not self.cfg.env.train.auto_reset:
                for stage_id in range(self.stage_num):
                    self.env_list[stage_id].is_start = True
                    extracted_obs, infos = self.env_list[stage_id].reset()
                    if "states" in extracted_obs:
                        extracted_obs["states"] = normalize_openpi_state(
                            extracted_obs["states"], self.train_state_norm_stats
                        )
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
                for stage_id in range(self.stage_num):
                    env_output = EnvOutput(
                        obs=self.last_obs_list[stage_id],
                        rewards=None,
                        dones=self.last_dones_list[stage_id],
                        terminations=self.last_terminations_list[stage_id],
                        truncations=self.last_truncations_list[stage_id],
                        intervene_actions=self.last_intervened_info_list[stage_id][0],
                        intervene_flags=self.last_intervened_info_list[stage_id][1],
                    )
                    env_output_list.append(env_output)

            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_output_list[stage_id]
                self.send_env_batch(output_channel, env_output.to_dict())

            for _ in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(input_channel)
                    env_output, env_info = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    self.send_env_batch(output_channel, env_output.to_dict())
                    env_output_list[stage_id] = env_output
                    for key, value in env_info.items():
                        # Add simulator type prefix in mixed mode
                        if self.cfg.env.train.env_type == "mixed":
                            prefixed_key = f"{self.train_env_type}/{key}"
                        else:
                            prefixed_key = key
                        if (
                            not self.cfg.env.train.auto_reset
                            and not self.cfg.env.train.ignore_terminations
                        ):
                            if (
                                prefixed_key in env_metrics
                                and len(env_metrics[prefixed_key]) > epoch
                            ):
                                env_metrics[prefixed_key][epoch] = value
                            else:
                                env_metrics[prefixed_key].append(value)
                        else:
                            env_metrics[prefixed_key].append(value)

            self.last_obs_list = [env_output.obs for env_output in env_output_list]
            self.last_dones_list = [env_output.dones for env_output in env_output_list]
            self.last_truncations_list = [
                env_output.truncations for env_output in env_output_list
            ]
            self.last_terminations_list = [
                env_output.terminations for env_output in env_output_list
            ]
            self.last_intervened_info_list = [
                (env_output.intervene_actions, env_output.intervene_flags)
                for env_output in env_output_list
            ]
            self.finish_rollout()

        for env in self.env_list:
            env.stop_env()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        eval_metrics = defaultdict(list)

        for stage_id in range(self.stage_num):
            self.eval_env_list[stage_id].start_env()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            for stage_id in range(self.stage_num):
                self.eval_env_list[stage_id].is_start = True
                extracted_obs, infos = self.eval_env_list[stage_id].reset()
                if "states" in extracted_obs:
                    extracted_obs["states"] = normalize_openpi_state(
                        extracted_obs["states"], self.eval_state_norm_stats
                    )
                env_output = EnvOutput(
                    obs=extracted_obs,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                )
                self.send_env_batch(output_channel, env_output.to_dict(), mode="eval")

            for eval_step in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        # Add simulator type prefix in mixed mode
                        if self.cfg.env.train.env_type == "mixed":
                            prefixed_key = f"{self.eval_env_type}/{key}"
                        else:
                            prefixed_key = key
                        eval_metrics[prefixed_key].append(value)
                    if eval_step == n_chunk_steps - 1:
                        continue
                    self.send_env_batch(
                        output_channel, env_output.to_dict(), mode="eval"
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            self.eval_env_list[stage_id].stop_env()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics
