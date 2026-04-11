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
from typing import Optional, Union

import gym
import habitat
import numpy as np
import torch
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat_baselines.config.default import get_config
from hydra.core.global_hydra import GlobalHydra

from rlinf.envs.habitat.extensions.utils import observations_to_image
from rlinf.envs.habitat.venv import HabitatRLEnv, ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor


@registry.register_task_action
class NoOpAction(SimulatorTaskAction):
    """Register manually defined No-operation action for habitat env."""

    def step(self, *args, **kwargs):
        return self._sim.get_sensor_observations()


class HabitatEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes):
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.start_idx = 0
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.auto_reset = cfg.auto_reset
        self.max_episode_steps = cfg.max_steps_per_rollout_epoch

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)

        self._init_env()

        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None

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

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        # Truncate chunk if it contains "stop" and pad with "no_op"
        for env_idx, chunk_action in enumerate(chunk_actions):
            stop_idx = np.where(chunk_action == "stop")[0]
            if len(stop_idx) > 0:
                stop_idx = stop_idx[0] + 1
                truncated_chunk = chunk_action[:stop_idx].copy()
                chunk_actions[env_idx] = np.concatenate(
                    [truncated_chunk, ["no_op"] * (chunk_size - len(truncated_chunk))]
                )

        # Truncate chunk if it would exceed max_episode_steps and pad with "no_op"
        for env_idx, elapsed_step in enumerate(self.elapsed_steps):
            if elapsed_step + chunk_size >= self.max_episode_steps:
                reserved_idx = self.max_episode_steps - elapsed_step
                truncated_chunk = chunk_actions[env_idx][:reserved_idx].copy()
                truncated_chunk[reserved_idx - 1] = "stop"
                chunk_actions[env_idx] = np.concatenate(
                    [
                        truncated_chunk,
                        ["no_op"] * (chunk_size - len(truncated_chunk)),
                    ]
                )

        chunk_terminations = []
        chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            chunk_terminations.append(terminations)
            chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        chunk_terminations = torch.stack(
            chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        chunk_truncations = torch.stack(
            chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def step(self, actions=None):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        for i, action in enumerate(actions):
            if action != "no_op":
                self._elapsed_steps[i] += 1

        # After excuting "stop" action, habitat env needs reset to process the next action
        # Replace "stop" with "no_op" before stepping the underlying env
        # to avoid unable to process the next action.
        is_stop = actions == "stop"
        actions[is_stop] = "no_op"

        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        terminations[is_stop] = True
        self.current_raw_obs = raw_obs
        obs = self._wrap_obs(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.max_episode_steps

        # TODO: what if termination means failure? (e.g. robot falling down)
        step_reward = self._calc_step_reward(terminations)

        dones = terminations | truncations
        if dones.any() and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        raw_obs = self.env.reset(env_idx)
        self._elapsed_steps[env_idx] = 0
        infos = {}

        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs

        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]
        obs = self._wrap_obs(self.current_raw_obs)

        return obs, infos

    def update_reset_state_ids(self):
        self.reset()

    def _wrap_obs(self, obs_list):
        image_list = []
        for obs in obs_list:
            image_list.append(observations_to_image(obs))

        image_tensor = to_tensor(list_of_dict_to_dict_of_list(image_list))

        obs = {}
        rgb_image_tensor = torch.stack(
            [value.clone().permute(2, 0, 1) for value in image_tensor["rgb"]]
        )
        obs["rgb"] = rgb_image_tensor

        if "depth" in image_tensor:
            depth_image_tensor = torch.stack(
                [value.clone().permute(2, 0, 1) for value in image_tensor["depth"]]
            )
            obs["depth"] = depth_image_tensor

        return obs

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(env_idx=env_idx)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _init_env(self):
        env_fns = self._get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def _get_env_fns(self):
        env_fn_params = self._get_env_fn_params()
        env_fns = []

        for param in env_fn_params:

            def env_fn(p=param):
                config_path = p["config_path"]
                episode_ids = p["episode_ids"]
                seed = p["seed"]

                config = get_config(config_path)

                dataset = habitat.datasets.make_dataset(
                    config.habitat.dataset.type,
                    config=config.habitat.dataset,
                )

                dataset.episodes = [
                    ep for ep in dataset.episodes if ep.episode_id in episode_ids
                ]

                env = HabitatRLEnv(config=config, dataset=dataset)
                env.seed(seed)
                return env

            env_fns.append(env_fn)

        return env_fns

    def _get_env_fn_params(self):
        env_fn_params = []

        # Habitat uses hydra to load the config,
        # but the hydra maybe initialized somewhere else,
        # so we need to clear it to avoid conflicts
        hydra_initialized = GlobalHydra.instance().is_initialized()
        if hydra_initialized:
            GlobalHydra.instance().clear()

        config_path = self.cfg.init_params.config_path
        habitat_config = get_config(config_path)

        habitat_dataset = habitat.datasets.make_dataset(
            habitat_config.habitat.dataset.type,
            config=habitat_config.habitat.dataset,
        )

        episode_ids = self._build_ordered_episodes(habitat_dataset)

        num_episodes = len(episode_ids)
        episodes_per_env = num_episodes // self.num_envs

        episode_ranges = []
        start = 0
        for i in range(self.num_envs - 1):
            episode_ranges.append((start, start + episodes_per_env))
            start += episodes_per_env
        episode_ranges.append((start, num_episodes))

        for env_id in range(self.num_envs):
            start, end = episode_ranges[env_id]
            assigned_ids = episode_ids[start:end]

            env_fn_params.append(
                {
                    "config_path": config_path,
                    "episode_ids": assigned_ids,
                    "seed": self.seed + env_id,
                }
            )

        return env_fn_params

    def _build_ordered_episodes(self, dataset):
        """
        rearrange the episode ids to be consecutive for each scene
        """
        scene_ids = []
        episode_ids = []
        scene_id_to_idx = {}  # scene_id(str) -> scene_idx(int)
        scene_to_episodes = {}  # scene_idx(int) -> episode_ids(list[int])

        for episode in dataset.episodes:
            sid = episode.scene_id
            eid = episode.episode_id
            if sid not in scene_id_to_idx:
                scene_idx = len(scene_ids)
                scene_id_to_idx[sid] = scene_idx
                scene_ids.append(sid)
                scene_to_episodes[scene_idx] = []
            else:
                scene_idx = scene_id_to_idx[sid]
            scene_to_episodes[scene_idx].append(eid)

        for scene_idx in range(len(scene_ids)):
            episode_ids.extend(scene_to_episodes[scene_idx])

        return episode_ids
