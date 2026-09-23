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

"""Adapt a RoboDojo runtime bridge to RLinf's vector environment contract.

The optional rlinf-robodojo-runtime>=0.3.0 distribution exports
robodojo_runtime.bridge.VectorEnv:
* VectorEnv(task_config: dict, n_envs: int, env_seeds: list[int]) owns simulator
  resources. ROBODOJO_ASSETS_ROOT is set from cfg.assets_path when provided;
  it names the directory containing Assets/, not Assets/ itself.
* reset(env_idx=None, env_seeds=...) resets all or selected slots. Seeds always
  contain n_envs entries, indexed by global slot, even for a partial reset.
* get_obs() returns n_envs dictionaries with full_image (H, W, 3 uint8),
  optional left_wrist_image/right_wrist_image of the same shape, state (D
  float32), and instruction (str). Camera availability must match across slots.
* step(actions) accepts a NumPy array (N, H, A) and returns
  (obs, rewards, terminations, truncations, infos). Obs has the get_obs format;
  rewards and boolean flags have shape (N,), and infos is N dictionaries with
  matching keys (optional success: bool). Rewards are totals over the chunk.
  The bridge must not auto-reset. Terminal slots stay latched until reset.
* check_seeds(seeds) returns one validity boolean per input seed.
* close(clear_cache: bool) releases simulator resources.

The runtime converts vision.cam_head/cam_left_wrist/cam_right_wrist and state.*
to the observation schema above, and maps policy vectors to RoboDojo's
left_arm_joint_state, right_arm_joint_state, left_ee_joint_state and
right_ee_joint_state action keys. Joint ordering and widths belong to task_config;
the wrapper does not assume a particular robot or import simulator code.

step returns batched observations and (N,) tensors. chunk_step returns one final
observation/info entry and (N, H) tensors: aggregate reward and terminal flags
occupy the last column. elapsed_steps counts submitted control steps, including
the padded remainder after early termination; simulator step limits are conveyed
through truncations. RLinf also enforces cfg.max_episode_steps.
Keep enable_offload false: this adapter exposes terminal close, not suspend/resume.
"""

import json
import os
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from rlinf.envs.sim.robotwin.seed_utils import partition_success_seeds
from rlinf.envs.utils import center_crop_image, list_of_dict_to_dict_of_list

__all__ = ["RoboDojoEnv"]


class RoboDojoEnv(gym.Env):
    """Own a runtime bridge and expose RoboTwin-style rollout operations.

    Args:
        cfg: RoboTwin-style settings, with RoboDojo-specific task_config.
        num_envs: Number of local vector slots, divisible by group_size.
        seed_offset: Worker seed offset.
        total_num_processes: Number of environment processes for seed partitioning.
        worker_info: Scheduler metadata retained for the worker.
        record_metrics: Whether to include episode summaries in infos.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.base_seed = cfg.seed
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.group_size = cfg.group_size
        if self.group_size <= 0 or num_envs <= 0 or num_envs % self.group_size:
            raise ValueError("num_envs must be positive and divisible by group_size.")
        self.num_group = num_envs // self.group_size
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.use_rel_reward = cfg.use_rel_reward
        self.use_custom_reward = cfg.use_custom_reward
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.task_name = cfg.task_config.task_name
        self.center_crop = cfg.get("center_crop", False)
        self.record_metrics = record_metrics
        self._is_start = True
        self._closed = False
        self._generator = torch.Generator().manual_seed(self.seed)
        self.success_seeds = None
        self._current_seed_index = 0
        if cfg.get("seeds_path") is not None:
            with open(cfg.seeds_path) as stream:
                seeds = json.load(stream)[self.task_name].get("success_seeds")
            if seeds is not None:
                self.success_seeds = partition_success_seeds(
                    torch.as_tensor(seeds, dtype=torch.long),
                    base_seed=self.base_seed,
                    seed_offset=seed_offset,
                    total_num_processes=total_num_processes,
                    num_group=self.num_group,
                )
                if not self.success_seeds.numel():
                    raise ValueError(
                        "Not enough success seeds for this worker's groups."
                    )
        self.update_reset_state_ids()
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.long)
        self.prev_step_reward = torch.zeros(num_envs)
        if record_metrics:
            self._init_metrics()
        self._init_env()

    def _init_env(self) -> None:
        if self.cfg.get("assets_path") is not None:
            os.environ["ROBODOJO_ASSETS_ROOT"] = str(self.cfg.assets_path)
        from robodojo_runtime.bridge import VectorEnv

        self.venv = VectorEnv(
            task_config=OmegaConf.to_container(self.cfg.task_config, resolve=True),
            n_envs=self.num_envs,
            env_seeds=self.reset_state_ids.tolist(),
        )

    @property
    def device(self) -> torch.device:
        """Return the CPU device used for wrapper counters and rewards."""
        return torch.device("cpu")

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Return submitted control-step counts for each slot."""
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        """Whether this wrapper has not yet been reset."""
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs)

    def _reset_metrics(self, env_idx: Any = None) -> None:
        index = slice(None) if env_idx is None else env_idx
        self._elapsed_steps[index] = 0
        self.prev_step_reward[index] = 0
        if self.record_metrics:
            self.success_once[index] = False
            self.returns[index] = 0

    def _record_metrics(self, reward: torch.Tensor, infos: dict) -> None:
        self.returns += reward
        episode = {
            "return": self.returns.clone(),
            "episode_len": self.elapsed_steps.clone(),
            "reward": self.returns / self.elapsed_steps.clamp_min(1),
        }
        if "success" in infos:
            infos["success"] = torch.as_tensor(
                infos["success"], dtype=torch.bool
            ).reshape(-1)
            self.success_once |= infos["success"]
            episode["success_once"] = self.success_once.clone()
            if self.ignore_terminations:
                episode["success_at_end"] = infos["success"].clone()
        infos["episode"] = episode

    def _extract_obs_image(self, raw_obs: list[dict]) -> dict:
        def image_tensor(image: np.ndarray) -> torch.Tensor:
            image = Image.fromarray(np.asarray(image)).convert("RGB")
            if self.center_crop:
                image = center_crop_image(image)
            return torch.from_numpy(np.array(image))

        wrists = [
            [
                image_tensor(obs[key])
                for key in ("left_wrist_image", "right_wrist_image")
                if obs.get(key) is not None
            ]
            for obs in raw_obs
        ]
        return {
            "main_images": torch.stack(
                [image_tensor(obs["full_image"]) for obs in raw_obs]
            ),
            "wrist_images": torch.stack([torch.stack(images) for images in wrists])
            if wrists[0]
            else None,
            "states": torch.stack(
                [torch.as_tensor(obs["state"], dtype=torch.float32) for obs in raw_obs]
            ),
            "task_descriptions": [obs["instruction"] for obs in raw_obs],
        }

    def reset(
        self, env_idx: int | list[int] | None = None, env_seeds: list[int] | None = None
    ) -> tuple[dict, dict]:
        """Reset selected slots and return all slots' observations and empty infos."""
        self.venv.reset(
            env_idx=env_idx,
            env_seeds=self.reset_state_ids.tolist() if env_seeds is None else env_seeds,
        )
        observations = self._extract_obs_image(self.venv.get_obs())
        self._reset_metrics(env_idx)
        self._is_start = False
        return observations, {}

    def step(
        self, actions: torch.Tensor | np.ndarray | dict, auto_reset: bool = True
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Submit (N, A) or (N, H, A) actions and return aggregate chunk results."""
        if isinstance(actions, dict):
            actions = actions["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[1] == 0
        ):
            raise ValueError(
                "Actions must have shape (num_envs, positive horizon, action_dim)."
            )
        raw_obs, reward, terminated, truncated, info_list = self.venv.step(actions)
        obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)
        terminated = (
            torch.as_tensor(terminated, dtype=torch.bool).reshape(self.num_envs).clone()
        )
        truncated = (
            torch.as_tensor(truncated, dtype=torch.bool).reshape(self.num_envs).clone()
        )
        reward = torch.as_tensor(reward, dtype=torch.float32).reshape(self.num_envs)
        if self.use_custom_reward:
            current = self.cfg.reward_coef * terminated.float()
            reward = current - self.prev_step_reward if self.use_rel_reward else current
            self.prev_step_reward = current
        self._elapsed_steps += actions.shape[1]
        truncated |= self.elapsed_steps >= self.cfg.max_episode_steps
        if self.record_metrics:
            self._record_metrics(reward, infos)
        if self.ignore_terminations:
            terminated.zero_()
        dones = terminated | truncated
        if auto_reset and self.auto_reset and dones.any():
            final_obs, final_info = obs, infos
            env_idx = dones.nonzero(as_tuple=True)[0].tolist()
            if self.cfg.get("is_eval", False):
                self.update_reset_state_ids(env_idx)
            obs, infos = self.reset(env_idx)
            infos.update(
                final_observation=final_obs,
                final_info=final_info,
                _final_observation=dones,
                _final_info=dones,
                _elapsed_steps=dones,
            )
        return obs, reward, terminated, truncated, infos

    def chunk_step(
        self, chunk_actions: torch.Tensor | np.ndarray
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Return one final observation and place aggregate results at chunk end."""
        if chunk_actions.ndim != 3 or chunk_actions.shape[1] == 0:
            raise ValueError("Chunk actions must have shape (N, positive horizon, A).")
        obs, reward, terminated, truncated, infos = self.step(chunk_actions)
        shape = (self.num_envs, chunk_actions.shape[1])
        rewards = torch.zeros(shape, dtype=torch.float32)
        terminations = torch.zeros(shape, dtype=torch.bool)
        truncations = torch.zeros(shape, dtype=torch.bool)
        rewards[:, -1] = reward
        terminations[:, -1] = terminated
        truncations[:, -1] = truncated
        return [obs], rewards, terminations, truncations, [infos]

    def update_reset_state_ids(self, env_idx: list[int] | None = None) -> None:
        """Select grouped seeds, preserving unselected slots and fixed reset IDs."""
        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return
        if self.success_seeds is None:
            seeds = torch.randint(
                10000, 200000, (self.num_group,), generator=self._generator
            )
        else:
            indices = (
                torch.arange(self.num_group) + self._current_seed_index
            ) % self.success_seeds.numel()
            seeds = self.success_seeds[indices]
            self._current_seed_index = (
                self._current_seed_index + self.num_group
            ) % self.success_seeds.numel()
        seeds = seeds.repeat_interleave(self.group_size)
        if env_idx is None:
            self.reset_state_ids = seeds
        else:
            self.reset_state_ids[env_idx] = seeds[env_idx]

    def check_seeds(self, seeds: list[int]) -> list[bool]:
        """Delegate seed validity checks to the runtime."""
        return self.venv.check_seeds(seeds)

    def close(self, clear_cache: bool = True) -> None:
        """Release the owned bridge once; repeated calls are harmless."""
        if not self._closed:
            self.venv.close(clear_cache)
            self._closed = True
