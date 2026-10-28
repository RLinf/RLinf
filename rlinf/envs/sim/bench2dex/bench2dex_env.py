# Copyright 2026 The RLinf Authors.
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

from functools import partial
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv


class Bench2DexEnv(IsaaclabBaseEnv):
    """Run one Bench2Dex scene in an isolated Isaac Sim process."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ):
        if num_envs != 1:
            raise ValueError(
                f"Bench2Dex requires one scene per env process, got {num_envs}."
            )
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.video_cfg = cfg.video_cfg
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.use_rel_reward = cfg.use_rel_reward
        self._is_start = True
        self._init_isaaclab_env()
        self.device = self.env.device()
        self.prev_step_reward = torch.zeros(num_envs, device=self.device)
        self._elapsed_steps = torch.zeros(
            num_envs, dtype=torch.int32, device=self.device
        )
        self._init_metrics()

    def _make_env_function(self):
        from rlinf.envs.sim.bench2dex.runtime import make_env

        return partial(
            make_env,
            OmegaConf.to_container(self.cfg.init_params, resolve=True),
            self.ignore_terminations,
        )

    def _wrap_obs(self, obs: dict) -> dict:
        return obs

    def step(
        self, actions: np.ndarray, auto_reset: bool = True
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Advance one control step and record current and historical success."""
        obs, reward, terminated, truncated, infos = self.env.step(actions)
        reward = self._calc_step_reward(terminated)
        self._elapsed_steps += 1
        truncated = truncated | (self.elapsed_steps >= self.cfg.max_episode_steps)
        infos = self._record_metrics(reward, terminated, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = infos["success_at_end"].clone()
            terminated = torch.zeros_like(terminated)
        dones = terminated | truncated
        if auto_reset and self.auto_reset and dones.any():
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return obs, reward, terminated, truncated, infos

    def chunk_step(
        self, chunk_actions: np.ndarray
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Execute an action chunk with the Isaac Lab reset contract."""
        if (
            chunk_actions.ndim != 3
            or chunk_actions.shape[0] != self.num_envs
            or chunk_actions.shape[2] != self.cfg.init_params.action_dim
        ):
            raise ValueError(
                "Bench2Dex requires action chunks with shape "
                f"({self.num_envs}, T, {self.cfg.init_params.action_dim}), "
                f"got {tuple(chunk_actions.shape)}."
            )
        return super().chunk_step(chunk_actions)
