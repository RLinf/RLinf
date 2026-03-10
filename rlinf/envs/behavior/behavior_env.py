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

import copy
import json
import os
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from omnigibson.envs import Environment
from omnigibson.learning.utils.eval_utils import (
    TASK_INDICES_TO_NAMES,
)
from omnigibson.macros import gm

from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.envs.venv.venv import SubprocVectorEnv
from rlinf.utils.logging import get_logger

# Make sure object states are enabled
gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

__all__ = ["BehaviorEnv"]


class _BehaviorSingleEnv(gym.Env):
    def __init__(self, configs: dict):
        self.env = Environment(configs=configs)

    def reset(self) -> tuple[dict, dict]:
        obs, info = self.env.reset()
        return obs, info

    def step(
        self, action: dict | torch.Tensor | np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()

    @staticmethod
    def create_env(configs: dict) -> "_BehaviorSingleEnv":
        return _BehaviorSingleEnv(copy.deepcopy(configs))


class BehaviorEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.cfg = cfg

        self.num_envs = num_envs
        self.ignore_terminations = cfg.ignore_terminations
        self.seed_offset = seed_offset
        self.seed = self.cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self.is_start = True
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.logger = get_logger()

        self.auto_reset = cfg.auto_reset
        if not self.record_metrics:
            self.logger.warning(
                "Metrics will still be recorded even if record_metrics is set to False."
            )
        self._init_metrics()
        self._init_env()

    def _load_tasks_cfg(self):
        with open_dict(self.cfg):
            self.cfg.omnigibson_cfg["task"]["activity_name"] = TASK_INDICES_TO_NAMES[
                self.cfg.task_idx
            ]
            # here let rlinf handle auto_reset
            self.cfg.omnigibson_cfg["env"]["automatic_reset"] = False
        # Read task description
        task_description_path = os.path.join(
            os.path.dirname(__file__), "behavior_task.jsonl"
        )
        with open(task_description_path, "r") as f:
            text = f.read()
            task_description = [json.loads(x) for x in text.strip().split("\n") if x]
        task_description_map = {
            task_description[i]["task_name"]: task_description[i]["task"]
            for i in range(len(task_description))
        }
        self.task_description = task_description_map[
            self.cfg.omnigibson_cfg["task"]["activity_name"]
        ]

    def _init_env(self):
        self._load_tasks_cfg()
        env_cfg = OmegaConf.to_container(self.cfg.omnigibson_cfg, resolve=True)
        env_fns = [
            partial(_BehaviorSingleEnv.create_env, configs=env_cfg)
            for _ in range(self.num_envs)
        ]
        self.env = SubprocVectorEnv(env_fns=env_fns)

    def _extract_obs_image(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        state, left_image, right_image, zed_image = None, None, None, None
        for sensor_data in raw_obs.values():
            assert isinstance(sensor_data, dict)
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_image = v["rgb"].to(torch.uint8)[..., :3]
                elif "right_realsense_link:Camera:0" in k:
                    right_image = v["rgb"].to(torch.uint8)[..., :3]
                elif "zed_link:Camera:0" in k:
                    zed_image = v["rgb"].to(torch.uint8)[..., :3]
                elif "proprio" in k:
                    state = v

        assert state is not None, (
            "state is not found in the observation which is required for the behavior training."
        )
        assert left_image is not None, "left_image is not found in the observation."
        assert right_image is not None, "right_image is not found in the observation."
        assert zed_image is not None, "zed_image is not found in the observation."

        return {
            "main_images": zed_image,  # [H, W, C]
            "wrist_images": torch.stack(
                [left_image, right_image], axis=0
            ),  # [N_IMG, H, W, C]
            "state": state,
        }

    def _wrap_obs(self, obs_list: list[dict[str, Any]]) -> dict[str, list]:
        extracted_obs_list = []
        for obs in obs_list:
            extracted_obs = self._extract_obs_image(obs)
            extracted_obs_list.append(extracted_obs)

        obs = {
            "main_images": torch.stack(
                [obs["main_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, H, W, C]
            "wrist_images": torch.stack(
                [obs["wrist_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, N_IMG, H, W, C]
            "task_descriptions": [
                self.task_description for _ in range(len(extracted_obs_list))
            ],
            "states": torch.stack(
                [obs["state"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, 32]
        }
        return obs

    def reset(self, env_idx: list[int] | None = None) -> tuple[dict, dict]:
        raw_obs, infos = self.env.reset(id=env_idx)
        obs = self._wrap_obs(raw_obs)
        self._reset_metrics(env_idx)
        return obs, {}

    def step(
        self, actions: np.ndarray | torch.Tensor | None = None
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        raw_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        obs = self._wrap_obs(raw_obs)
        infos = self._record_metrics(rewards, infos)
        if self.ignore_terminations:
            terminations[:] = False

        return (
            obs,
            to_tensor(rewards),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(
        self, chunk_actions: np.ndarray | torch.Tensor
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        infos_list = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_rewards, terminations, truncations, infos = self.step(
                actions
            )
            obs_list.append(extracted_obs)
            chunk_rewards.append(step_rewards)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)
            infos_list.append(infos)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)

        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    @property
    def device(self):
        return "cuda"

    @property
    def elapsed_steps(self):
        return torch.tensor(self.cfg.max_episode_steps)

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.prev_step_reward = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx: list[int] | None = None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
        else:
            mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        self.prev_step_reward[mask] = 0.0
        self.success_once[mask] = False
        self.fail_once[mask] = False
        self.returns[mask] = 0

    def _record_metrics(self, rewards: torch.Tensor, infos: list[dict]) -> dict:
        info_lists = []
        for env_idx, (reward, info) in enumerate(zip(rewards, infos)):
            episode_info = {
                "success": info.get("done", {}).get("success", False),
                "episode_length": info["episode_length"],
            }
            self.returns[env_idx] += reward
            if "success" in info:
                self.success_once[env_idx] = (
                    self.success_once[env_idx] | info["success"]
                )
                episode_info["success_once"] = self.success_once[env_idx].clone()
            if "fail" in info:
                self.fail_once[env_idx] = self.fail_once[env_idx] | info["fail"]
                episode_info["fail_once"] = self.fail_once[env_idx].clone()
            episode_info["return"] = self.returns[env_idx].clone()
            episode_info["average_reward"] = (
                episode_info["return"] / episode_info["episode_length"]
                if episode_info["episode_length"] > 0
                else 0.0
            )
            if self.ignore_terminations:
                episode_info["success_at_end"] = info["success"]

            info_lists.append(episode_info)

        infos = {"episode": to_tensor(list_of_dict_to_dict_of_list(info_lists))}
        return infos

    def _handle_auto_reset(
        self, dones: torch.Tensor, extracted_obs: dict[str, Any], infos: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        def merge_reset_obs(
            obs: dict[str, Any], reset_obs: dict[str, Any], env_idx: list[int]
        ) -> dict[str, Any]:
            merged_obs = {}
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    merged_obs[k] = v.clone()
                    merged_obs[k][env_idx] = reset_obs[k]
                elif isinstance(v, list):
                    merged_obs[k] = v.copy()
                    for i, idx in enumerate(env_idx):
                        merged_obs[k][idx] = reset_obs[k][i]
                else:
                    raise NotImplementedError(
                        f"Unsupported obs type {type(v)} for key {k} in auto reset,should only contain list and torch.Tensor."
                    )
            return merged_obs

        env_idx = torch.nonzero(dones, as_tuple=False).squeeze(-1).cpu().tolist()
        assert len(env_idx) > 0, (
            "There should be at least one done env to trigger auto reset."
        )

        final_obs = extracted_obs.copy()
        final_info = infos.copy()
        reset_extracted_obs, _ = self.reset(env_idx=env_idx)

        extracted_obs = merge_reset_obs(extracted_obs, reset_extracted_obs, env_idx)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def update_reset_state_ids(self):
        # use for multi task training
        pass
