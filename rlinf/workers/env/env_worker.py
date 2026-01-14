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

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.env_manager import EnvManager
from rlinf.envs.wrappers import DataCollectorWrapper
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement

logger = logging.getLogger(__name__)


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

        # Data collector for reward model training
        self.data_collector: Optional[DataCollectorWrapper] = None
        # Episode buffers for each parallel env (to collect full trajectories)
        self._episode_buffers: list[list[dict]] = []
        self._episode_grasp_flags: list[bool] = []
        self._episode_success_flags: list[bool] = []

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

    def init_worker(self):
        enable_offload = self.cfg.env.enable_offload

        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(True, list(range(self._world_size)))

        if not self.only_eval:
            for stage_id in range(self.stage_num):
                self.env_list.append(
                    EnvManager(
                        self.cfg.env.train,
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
                        self.cfg.env.eval,
                        rank=self._rank,
                        num_envs=self.eval_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=eval_env_cls,
                        worker_info=self.worker_info,
                        enable_offload=enable_offload,
                    )
                )

        # Initialize data collector if enabled
        data_collection_cfg = self.cfg.get("reward_data_collection", {})
        if data_collection_cfg.get("enabled", False):
            self.data_collector = DataCollectorWrapper(
                env=None,  # Not wrapping env directly, using as standalone collector
                save_dir=data_collection_cfg.get("save_dir", "./reward_data"),
                max_frames=self.cfg.env.train.get("max_episode_steps", 50),
                target_episodes=data_collection_cfg.get("target_success", 5000)
                + data_collection_cfg.get("target_failure", 5000),
            )
            # Initialize episode buffers for each parallel environment
            total_envs = self.train_num_envs_per_stage * self.stage_num
            self._episode_buffers = [[] for _ in range(total_envs)]
            self._episode_grasp_flags = [False] * total_envs
            self._episode_success_flags = [False] * total_envs
            logger.info(
                f"DataCollector initialized. Save dir: {self.data_collector.save_dir}, "
                f"Tracking {total_envs} parallel envs"
            )

        if not self.only_eval:
            self._init_env()

    def _init_env(self):
        if self.cfg.env.train.auto_reset:
            for i in range(self.stage_num):
                self.env_list[i].start_env()
                extracted_obs, _ = self.env_list[i].reset()
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
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.env_list[stage_id].chunk_step(chunk_actions)
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

        # Normalize infos: ensure "success" is at top level for data collection
        # When auto_reset happens, success moves to infos["final_info"]["success"]
        # IMPORTANT: Check final_info FIRST because after auto_reset,
        # infos["success"] is from the reset state (all False)
        normalized_infos = {}
        if "final_info" in infos and "success" in infos["final_info"]:
            # Success from before auto_reset - this has the real success values
            normalized_infos["success"] = infos["final_info"]["success"]
        elif "success" in infos:
            # Direct success (no auto_reset happened this step)
            normalized_infos["success"] = infos["success"]

        # Data collection: collect every frame for each parallel env
        if self.data_collector is not None:
            batch_size = extracted_obs[list(extracted_obs.keys())[0]].shape[0]
            env_offset = stage_id * self.train_num_envs_per_stage
            
            for env_idx in range(batch_size):
                global_idx = env_offset + env_idx
                
                # Get obs for this env
                obs_to_save = {
                    k: v[env_idx].cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in extracted_obs.items()
                }
                
                # Get success/grasp status for this step
                step_success = False
                if "success" in normalized_infos:
                    success_val = normalized_infos["success"]
                    if isinstance(success_val, torch.Tensor):
                        step_success = bool(success_val[env_idx].item())
                    else:
                        step_success = bool(success_val)
                
                step_grasp = False
                if "is_grasped" in infos:
                    grasp_val = infos["is_grasped"]
                    if isinstance(grasp_val, torch.Tensor):
                        step_grasp = bool(grasp_val[env_idx].item())
                
                # Update episode-level flags
                if step_success:
                    self._episode_success_flags[global_idx] = True
                if step_grasp:
                    self._episode_grasp_flags[global_idx] = True
                
                # Build frame data
                frame = {
                    "obs": obs_to_save,
                    "action": chunk_actions[env_idx].cpu().numpy() if isinstance(chunk_actions, torch.Tensor) else chunk_actions[env_idx],
                    "reward": float(chunk_rewards[env_idx, -1]) if chunk_rewards is not None else 0.0,
                    "terminated": bool(chunk_terminations[env_idx, -1]),
                    "truncated": bool(chunk_truncations[env_idx, -1]),
                    "done": bool(chunk_dones[env_idx, -1]),
                    "grasp": step_grasp,
                    "success": step_success,
                }
                self._episode_buffers[global_idx].append(frame)
                
                # If episode done, save full trajectory and reset buffer
                if chunk_dones[env_idx, -1]:
                    self._save_episode_trajectory(global_idx)
                    # Reset buffer for this env
                    self._episode_buffers[global_idx] = []
                    self._episode_grasp_flags[global_idx] = False
                    self._episode_success_flags[global_idx] = False

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
            infos=normalized_infos,  # Pass normalized infos with success at top level
        )
        return env_output, env_info

    def _save_episode_trajectory(self, env_idx: int) -> None:
        """Save full episode trajectory to pkl file.
        
        Args:
            env_idx: Index of the parallel environment.
        """
        if self.data_collector is None:
            return
        
        frames = self._episode_buffers[env_idx]
        if not frames:
            return
        
        episode_success = self._episode_success_flags[env_idx]
        episode_grasp = self._episode_grasp_flags[env_idx]
        
        # Use DataCollectorWrapper's internal counter and save logic
        self.data_collector._current_episode = frames
        self.data_collector._episode_success = episode_success
        self.data_collector._episode_grasp = episode_grasp
        self.data_collector._save_episode()
        
        # Log periodically
        if self.data_collector.episode_count % 50 == 0:
            logger.info(
                f"Saved episode {self.data_collector.episode_count}: "
                f"{'success' if episode_success else 'failure'}, "
                f"frames={len(frames)}, grasp={episode_grasp}. "
                f"Total: {self.data_collector.success_episode_count} success, "
                f"{self.data_collector.failure_episode_count} failure"
            )

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.train.env_type,
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
                    dones = (
                        torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                        .unsqueeze(1)
                        .repeat(1, self.cfg.actor.model.num_action_chunks)
                    )
                    terminations = dones.clone()
                    truncations = dones.clone()

                    # For reset stage, success is always False (no action taken yet)
                    reset_infos = {
                        "success": torch.zeros(
                            self.train_num_envs_per_stage, dtype=torch.bool
                        )
                    }
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
                        infos=reset_infos,  # Pass infos with success=False for reset stage
                    )
                    env_output_list.append(env_output)
            else:
                self.num_done_envs = 0
                self.num_succ_envs = 0
                for stage_id in range(self.stage_num):
                    # For reset stage, success is always False (no action taken yet)
                    reset_infos = {
                        "success": torch.zeros(
                            self.train_num_envs_per_stage, dtype=torch.bool
                        )
                    }
                    env_output = EnvOutput(
                        obs=self.last_obs_list[stage_id],
                        rewards=None,
                        dones=self.last_dones_list[stage_id],
                        terminations=self.last_terminations_list[stage_id],
                        truncations=self.last_truncations_list[stage_id],
                        intervene_actions=self.last_intervened_info_list[stage_id][0],
                        intervene_flags=self.last_intervened_info_list[stage_id][1],
                        infos=reset_infos,  # Pass infos with success=False for reset stage
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
                        eval_metrics[key].append(value)
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
