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
import os
import cv2
from typing import Optional, Union

import gym
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf

import habitat
from habitat_baselines.config.default import get_config as get_habitat_config

from rlinf.envs.habitat.utils import get_habitat_rgb_image, get_habitat_state, HabitatRLEnv
from rlinf.envs.habitat.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    put_info_on_image,
    save_rollout_video,
    tile_images,
    to_tensor,
)


class HabitatEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0
        
        self._init_task_suite()
        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []
        self.current_raw_obs = None

    def _init_task_suite(self):
        """Initialize Habitat task suite: load all scenes and episodes from config."""    
        config_path = self.cfg.init_params.config_path
        config = get_habitat_config(config_path)
        
        # Load dataset to get all scenes and episodes
        # Habitat stores scene and episode information in the dataset
        if hasattr(config.habitat, 'dataset'):
            dataset = habitat.datasets.make_dataset(
                config.habitat.dataset.type, config=config.habitat.dataset
            )

        self.scenes = []         
        self.scene_groups = {}   # scene_idx -> list of episode indices
        scene_id_to_idx = {}     # scene_id(str) -> scene_idx(int)

        for ep_idx, episode in enumerate(dataset.episodes):
            if hasattr(episode, "scene_id") and episode.scene_id is not None:
                sid = episode.scene_id
                if sid not in scene_id_to_idx:
                    scene_idx = len(self.scenes)
                    scene_id_to_idx[sid] = scene_idx
                    self.scenes.append(sid)
                    self.scene_groups[scene_idx] = []
                else:
                    scene_idx = scene_id_to_idx[sid]
                self.scene_groups[scene_idx].append(ep_idx)
            else:
                raise ValueError(f"Episode {ep_idx} does not have scene_id!")

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []
        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param):
                config = param.pop("config")
                seed = param.pop("seed")
                env = HabitatRLEnv(config=config)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        """
        Get environment function parameters.
        
        For each environment, set the scene based on current task_id.
        This ensures same-scene episodes run in the same environment when possible.
        """
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)
        config_path = base_env_args.get("config_path", self.cfg.init_params.config_path)
        config = get_habitat_config(config_path)
        
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        
        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                continue
            env_params = {
                "config": config,
                "seed": self.seed,
            }
            env_fn_params.append(env_params)

        return env_fn_params

    def _compute_total_num_group_envs(self):
        """
        Compute total number of group environments.
        
        Total = sum of all episodes across all scenes.
        Each scene has multiple episodes (trials).
        """
        self.scene_episode_counts = []
        self.cumsum_trial_id_bins = []
        for scene_idx in self.scene_groups:
            self.scene_episode_counts.append(len(self.scene_groups[scene_idx]))
            # cumsum_trial_id_bins: cumulative sum for decoding reset_state_id
            # Example: [5, 3, 4] -> [5, 8, 12]
            # reset_state_id < 5 -> scene 0, < 8 -> scene 1, < 12 -> scene 2
            self.cumsum_trial_id_bins.append(np.sum(self.scene_episode_counts[:scene_idx+1]))
        # Total number of episodes = sum of episodes in all scenes
        self.total_num_group_envs = np.sum(self.scene_episode_counts)

    def get_reset_state_ids_all(self):
        """
        Get all reset state IDs, organized to keep same-scene episodes together.
        
        Strategy: Group episodes by scene, then distribute scenes across processes.
        Within each scene, episodes are kept together to minimize scene switching.
        """
        
        # Shuffle scene order, but keep episodes within scenes together
        scene_ids = list(self.scene_groups.keys())
        self._generator_ordered.shuffle(scene_ids)
        
        # Flatten: all episodes from scene 0, then all from scene 1, etc.
        # This ensures same-scene episodes are consecutive
        reset_state_ids = []
        for scene_id in scene_ids:
            reset_state_ids.extend(self.scene_groups[scene_id])
        
        reset_state_ids = np.array(reset_state_ids)
        
        # Adjust size to be divisible by total_num_processes
        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        
        return reset_state_ids

    def update_reset_state_ids(self):
        if self.cfg.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (self.num_group,), dtype=int
            )
        else:
            if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
                self.reset_state_ids_all = self.get_reset_state_ids_all()
                self.start_idx = 0
            reset_state_ids = self.reset_state_ids_all[self.seed_offset][
                self.start_idx : self.start_idx + num_reset_states
            ]
            self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        """
        Decode reset_state_id to (scene_id, episode_id).
        
        reset_state_id is a global episode index.
        We decode it to (scene_id, episode_in_scene_id).
        """
        task_ids = []  # scene_id
        trial_ids = []  # episode_in_scene_id
        
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

    def _update_task_and_trial_ids(self, reset_state_ids, env_idx):
        """
        Update task and trial IDs based on reset_state_ids.
        
        Since Habitat environments load all scenes, we don't need to reconfigure
        when scene changes. Habitat's reset() will automatically handle scene switching.
        """
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        
        # Update task and trial IDs for tracking
        for j, env_id in enumerate(env_idx):
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]

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

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        """Extract image and state from Habitat observations."""
        return {
            "rgb_image": get_habitat_rgb_image(obs),    
            "state": get_habitat_state(obs),
        }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        image_tensor = torch.stack(
            [
                value.clone().permute(2, 0, 1)
                for value in images_and_states["rgb_image"]
            ]
        )

        states = images_and_states["state"]

        obs = {
            "rgb_images": image_tensor,
            "states": states,
        }
        return obs

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            if self.task_ids[env_id] != task_ids[j]:
                reconfig_env_idx.append(env_id)
                self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        raw_obs = self.env.reset(id=env_idx)
        return raw_obs

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)
        
        raw_obs = self._reconfigure(reset_state_ids, env_idx)
        
        # TODO: what if the reconfigure env_idx is not the same as the env_idx?
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs

        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        # Vectorized environment step - following LiberoEnv pattern
        # Returns (obs, reward, done, info) for vectorized env
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        # TODO: what if termination means failure? (e.g. robot falling down)
        step_reward = self._calc_step_reward(terminations)

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": step_reward,
                "terminations": terminations,
            }
            self.add_new_frames(raw_obs, plot_infos)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        if self.cfg.is_eval:
            self.update_reset_state_ids()
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )
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

    def add_new_frames(self, raw_obs, plot_infos):
        images = []
        for env_id, raw_single_obs in enumerate(raw_obs):
            info_item = {
                k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()
            }
            img = get_habitat_rgb_image(raw_single_obs)
            img = put_info_on_image(img, info_item)
            images.append(img)
        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)

    def save_frames(self, raw_obs, plot_infos, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for env_id, raw_single_obs in enumerate(raw_obs):
            info_item = {
                k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()
            }
            img = get_habitat_rgb_image(raw_single_obs)
            img = put_info_on_image(img, info_item)
            img_name = f"env_{env_id}_step_{self._elapsed_steps[env_id]}.png"
            cv2.imwrite(os.path.join(output_dir, img_name), img)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []


