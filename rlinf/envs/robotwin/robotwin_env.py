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

import os
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
from robotwin.envs.vector_env import VectorEnv

__all__ = ["RoboTwinEnv"]


class RoboTwinEnv(gym.Env):
    def __init__(self, cfg, seed_offset, total_num_processes, record_metrics=True):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.total_num_processes = total_num_processes
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = cfg.num_group
        self.group_size = cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.use_custom_reward = False

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []

        self.cfg = cfg
        self.record_metrics = record_metrics
        self._is_start = True

        self.task_name = cfg.task_config.task_name
        self.num_envs = cfg.num_envs

        self._init_env()

        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32)
        self.info_logging_keys = ["is_success"]
        if self.record_metrics:
            self._init_metrics()
            self._elapsed_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

    def _init_env(self):
        os.environ["ASSETS_PATH"] = self.cfg.assets_path

        group_seeds = torch.randint(0, 30, (self.num_group,))
        env_seeds = group_seeds.repeat_interleave(self.group_size).tolist()

        self.venv = VectorEnv(
            task_config=OmegaConf.to_container(self.cfg.task_config, resolve=True),
            n_envs=self.num_envs,
            horizon=1,  # Set horizon to 1 since we handle chunk steps externally
            max_step=self.cfg.max_step,
            env_seeds=env_seeds,
        )

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

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

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
                self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _extract_obs_image(self, raw_obs, infos):
        images = []
        wrist_images = []
        states = []
        for obs in raw_obs:
            # obs images: [N_IMG, H, W, C], N_IMG = 6, 0: head_prev, 1: right_prev, 2: left_prev, 3: head_curr, 4: right_curr, 5: left_curr
            images.append(obs["images"][3, ...])
            wrist_images.append(obs["images"][-2:, ...])
            states.append(obs["state"])

        images = torch.stack([torch.from_numpy(img) for img in images])
        wrist_images = torch.stack([torch.from_numpy(img) for img in wrist_images])
        states = torch.stack([torch.from_numpy(state) for state in states])

        images = images.permute(0, 3, 1, 2).unsqueeze(
            1
        )  # [B, H, W, C] -> [B, 1, C, H, W]
        wrist_images = wrist_images.permute(
            0, 1, 4, 2, 3
        )  # [B, N_IMG, H, W, C] -> [B, N_IMG, C, H, W]

        extracted_obs = {
            "images": images,
            "wrist_images": wrist_images,
            "states": states,
            "task_descriptions": infos["instructions"],
        }
        return extracted_obs

    def _calc_step_reward(self, info):
        reward = torch.zeros(self.num_envs, dtype=torch.float32)

        if "is_success" in info:
            success_reward = torch.tensor([float(x) for x in info["is_success"]])
            reward += success_reward * 1.0

        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def reset(
        self,
        env_idx: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = {},
    ):
        if self._is_start:
            raw_obs, _, _, _, infos = self.venv.init_process()
            extracted_obs = self._extract_obs_image(raw_obs, infos)
            self._is_start = False
        else:
            raw_obs, _, _, _, infos = self.venv.reset(env_idx=env_idx)
            self._reset_metrics(env_idx)

            extracted_obs = self._extract_obs_image(raw_obs, infos)

        return extracted_obs, infos

    def step(
        self, actions: Union[torch.Tensor, np.ndarray, Dict] = None, auto_reset=True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."

        if self.is_start:
            extracted_obs, infos = self.reset(seed=self.seed)
            self._is_start = False
            terminations = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            truncations = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            return extracted_obs, None, terminations, truncations, infos

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, dict):
            actions = actions.get("actions", actions)

        # [n_envs, horizon, action_dim]
        if len(actions.shape) == 2:
            # [n_envs, action_dim] -> [n_envs, 1, action_dim]
            actions = actions[:, None, :]

        self._elapsed_steps += 1
        raw_obs, step_reward, terminations, truncations, infos = self.venv.step(actions)
        extracted_obs = self._extract_obs_image(raw_obs, infos)

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(infos)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        infos = self._record_metrics(step_reward, infos)
        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)

        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []

        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

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
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, infos
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = infos.copy()

        extracted_obs, infos = self.reset(options=options)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def clear(self):
        if hasattr(self, "venv"):
            self.venv.clear()

    def sample_action_space(self):
        return np.random.randn(self.num_envs, self.horizon, 14)
