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

"""Vectorized ALOHA bimanual sim tasks from gym-aloha."""

import copy
from typing import Optional, Union

import gym_aloha  # noqa: F401  (registers the gym_aloha/* env ids)
import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.envs.venv import SubprocVectorEnv

# gym-aloha rewards each stage of a task; the top stage is task success.
ALOHA_SUCCESS_REWARD = 4


def _make_aloha_env(task_name: str, init_params: dict) -> gym.Env:
    return gym.make(
        f"gym_aloha/{task_name}",
        obs_type="pixels_agent_pos",
        disable_env_checker=True,
        **init_params,
    )


class AlohaEnv(gym.Env):
    # gym-aloha's control rate, used by RecordVideo.
    metadata = {"render_fps": 50}

    """``gym_aloha/<task_name>`` envs with the ``pixels_agent_pos`` observation.

    Observations are the ``top`` camera image (``main_images``, uint8
    ``[N, H, W, 3]``) and the 14-dim joint state (``states``). The step reward
    is ``reward_coef`` on success (``reward_type: success``) or ``reward_coef``
    times gym-aloha's staged reward divided by 4 (``reward_type: staged``).
    Success terminates the episode, as in gym-aloha.

    Every episode is reset with its own seed. Eval and
    ``use_ordered_reset_state_ids`` take consecutive seeds from ``seed``, spread
    across env processes, so evaluation runs are reproducible. Envs of one
    group always share a seed.
    """

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = cfg.seed + seed_offset
        self.group_size = cfg.group_size
        self.num_group = num_envs // self.group_size
        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.use_rel_reward = cfg.use_rel_reward
        self.reward_type = cfg.get("reward_type", "success")
        if self.reward_type not in ("success", "staged"):
            raise ValueError(
                f"reward_type must be 'success' or 'staged', got {self.reward_type!r}."
            )
        self.task_descriptions = [cfg.task_description] * num_envs
        self._is_start = True

        self._generator = np.random.default_rng(seed=self.seed)
        self._num_ordered_seeds = 0
        self.update_reset_state_ids()

        task_name = cfg.task_name
        init_params = dict(cfg.get("init_params", None) or {})
        self.env = SubprocVectorEnv(
            [lambda: _make_aloha_env(task_name, init_params) for _ in range(num_envs)]
        )
        self._last_obs: list[Optional[dict]] = [None] * num_envs
        self.prev_step_reward = np.zeros(num_envs)
        self.success_once = np.zeros(num_envs, dtype=bool)
        self.returns = np.zeros(num_envs)
        self._elapsed_steps = np.zeros(num_envs, dtype=np.int32)

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _sample_seeds(self, num_seeds: int) -> np.ndarray:
        if self.cfg.is_eval or self.cfg.get("use_ordered_reset_state_ids", False):
            # Process p takes seeds seed + p, seed + p + P, ... for P processes.
            index = self._num_ordered_seeds + np.arange(num_seeds)
            self._num_ordered_seeds += num_seeds
            return self.cfg.seed + self.seed_offset + index * self.total_num_processes
        return self._generator.integers(0, 2**31 - 1, size=num_seeds)

    def _sample_group_seeds(self, env_idx: np.ndarray) -> np.ndarray:
        """Draw one new seed per group among ``env_idx``, for each env in it."""
        groups, group_of_env = np.unique(
            env_idx // self.group_size, return_inverse=True
        )
        return self._sample_seeds(len(groups))[group_of_env]

    def update_reset_state_ids(self):
        self.reset_state_ids = self._sample_group_seeds(np.arange(self.num_envs))
        # group -> (seed, members that already started an episode with it)
        self._group_seeds = {
            i // self.group_size: (int(self.reset_state_ids[i]), set())
            for i in range(0, self.num_envs, self.group_size)
        }

    def _next_group_seeds(self, env_idx: np.ndarray) -> np.ndarray:
        """Seeds for resetting ``env_idx`` mid-rollout.

        The first member of a group to reset draws a new seed; the others reuse
        it when they reset, even on later steps, so a group keeps sharing its
        start state.
        """
        seeds = []
        for i in env_idx:
            group = int(i) // self.group_size
            seed, used = self._group_seeds[group]
            if int(i) in used:
                seed, used = int(self._sample_seeds(1)[0]), set()
            used.add(int(i))
            self._group_seeds[group] = (seed, used)
            seeds.append(seed)
        return np.array(seeds)

    def _wrap_obs(self):
        images = np.stack([o["pixels"]["top"] for o in self._last_obs])
        states = np.stack([o["agent_pos"] for o in self._last_obs])
        return {
            "main_images": torch.from_numpy(images),
            "states": torch.from_numpy(states.astype(np.float32)),
            "task_descriptions": list(self.task_descriptions),
        }

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        env_idx = np.atleast_1d(env_idx)
        if reset_state_ids is None:
            if self.use_fixed_reset_state_ids:
                reset_state_ids = self.reset_state_ids[env_idx]
            else:
                reset_state_ids = self._next_group_seeds(env_idx)
        # The vector env sends the same reset kwargs to every env it resets, so
        # envs are reset one at a time to give each its own seed.
        for i, seed in zip(env_idx, reset_state_ids):
            obs, _ = self.env.reset(id=[int(i)], seed=int(seed))
            self._last_obs[i] = obs[0]
        self.prev_step_reward[env_idx] = 0.0
        self.success_once[env_idx] = False
        self.returns[env_idx] = 0.0
        self._elapsed_steps[env_idx] = 0
        return self._wrap_obs(), {}

    def _calc_step_reward(self, raw_reward: np.ndarray, success: np.ndarray):
        if self.reward_type == "success":
            reward = self.cfg.reward_coef * success
        else:
            reward = self.cfg.reward_coef * raw_reward / ALOHA_SUCCESS_REWARD
        step_reward = reward - self.prev_step_reward if self.use_rel_reward else reward
        self.prev_step_reward = reward
        return step_reward

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        self._elapsed_steps += 1
        raw_obs, raw_reward, _, _, info_list = self.env.step(actions)
        self._last_obs = list(raw_obs)
        raw_reward = np.asarray(raw_reward, dtype=np.float64)
        terminations = raw_reward >= ALOHA_SUCCESS_REWARD
        truncations = self._elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs()

        step_reward = self._calc_step_reward(raw_reward, terminations)
        self.returns += step_reward
        self.success_once |= terminations
        infos = list_of_dict_to_dict_of_list(list(info_list))
        infos["episode"] = to_tensor(
            {
                "success_once": self.success_once.copy(),
                "return": self.returns.copy(),
                "episode_len": self._elapsed_steps.copy(),
                "reward": self.returns / np.maximum(self._elapsed_steps, 1),
            }
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations = np.zeros_like(terminations)

        dones = terminations | truncations
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, num_action_chunks, action_dim]
        obs_list, infos_list = [], []
        rewards, terminations, truncations = [], [], []
        for i in range(chunk_actions.shape[1]):
            obs, reward, term, trunc, infos = self.step(
                chunk_actions[:, i], auto_reset=False
            )
            obs_list.append(obs)
            infos_list.append(infos)
            rewards.append(reward)
            terminations.append(term)
            truncations.append(trunc)

        chunk_rewards = torch.stack(rewards, dim=1)
        raw_chunk_terminations = torch.stack(terminations, dim=1)
        raw_chunk_truncations = torch.stack(truncations, dim=1)
        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = past_terminations | past_truncations

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, final_obs, infos):
        final_obs = copy.deepcopy(final_obs)
        final_info = copy.deepcopy(infos)
        env_idx = np.arange(self.num_envs)[dones]
        if self.use_fixed_reset_state_ids:
            reset_state_ids = self.reset_state_ids[env_idx]
        else:
            # The worker does not refresh seeds between auto-reset episodes.
            reset_state_ids = self._next_group_seeds(env_idx)
        obs, infos = self.reset(env_idx=env_idx, reset_state_ids=reset_state_ids)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def close(self):
        self.env.close()
