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


from typing import Optional, Union

import gym
import numpy as np
import torch
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.worldmodel.backend import WorldModelBackend
from rlinf.envs.worldmodel.dataset import LeRobotDatasetWrapper

# 默认都是gpu模式，需要加载模型

class WorldModelEnv(gym.Env):
    def __init__(self, cfg, seed_offset: int, total_num_processes, record_metrics=True):
        self.cfg = cfg
        
        # 基础配置信息加载
        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.num_envs = cfg.num_envs
        self.group_size = cfg.group_size
        self.num_group = cfg.num_group
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        

        dataset_cfg = OmegaConf.to_container(cfg.dataset_cfg, resolve=True)
        self.task_dataset = LeRobotDatasetWrapper(**dataset_cfg)
        self.total_num_group_envs = len(self.task_dataset)

        self.device = "cuda"
        env_cfg = OmegaConf.to_container(cfg.backend_cfg, resolve=True)
        self.env = WorldModelBackend(env_cfg, self.task_dataset, self.device)

        self._is_start = True
        self._init_reset_state_ids()
        self._init_metrics()
        self.record_metrics = record_metrics

        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(self.device)

        # 录制结果
        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value
    
    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

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
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
    
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

    # TODO：实现计算reward的逻辑，底层需要调用模型（分类、回归相似性）
    def _calc_step_reward(self, info):
        pass
    
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
    ):
        obs, infos = self.env.reset(seed=seed, options=options)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return obs, infos

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            extracted_obs, infos = self.reset(
                seed=self.seed,
                options={"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {},
            )
            self._is_start = False
            terminations = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            truncations = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            # TODO：实现渲染相关逻辑
            # if self.video_cfg.save_video:
            #     self.add_new_frames(infos=infos)
            return extracted_obs, None, terminations, truncations, infos

        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        
        # step_reward = self._calc_step_reward(infos)

        # TODO：实现渲染相关逻辑
        # if self.video_cfg.save_video:
        #     self.add_new_frames(infos=infos, rewards=step_reward)

        infos = self._record_metrics(step_reward, infos)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
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
        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(infos)
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset(options=options)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def run(self):
        obs, info = self.reset()
        self.reset(options={"env_idx": torch.arange(0, self.num_envs-4, device=self.device)})
        for step in range(100):
            action = self.env.action_space.sample()
            obs, rew, terminations, truncations, infos = self.step(action)
            print(
                f"Step {step}: obs={obs.keys()}, rew={rew.mean()}, terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )
    
    # def add_new_frames(self, infos, rewards=None):
    #     image = self.render(infos, rewards)
    #     self.render_images.append(image)