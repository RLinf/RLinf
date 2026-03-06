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

# Ensure MW envs only register once
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.calvin import CalvinBenchmark, make_env
from rlinf.envs.calvin.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor


class CalvinEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: CalvinBenchmark = CalvinBenchmark(
            self.cfg.task_suite_name, self._generator
        )
        self.num_tasks = self.task_suite.get_num_tasks()
        self.task_num_trials = self.task_suite.get_task_num_trials()
        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_task_info()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []
        for env_fn_param in env_fn_params:

            def env_fn(params=env_fn_param):
                os.environ["EGL_VISIBLE_DEVICES"] = str(self.seed_offset)
                env = make_env(**params)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        if self.cfg.task_suite_name == "calvin_d":
            candidated_scenes = ["calvin_scene_D"]
        elif self.cfg.task_suite_name == "calvin_abc":
            candidated_scenes = ["calvin_scene_A", "calvin_scene_B", "calvin_scene_C"]
        elif self.cfg.task_suite_name == "calvin_abcd":
            candidated_scenes = [
                "calvin_scene_A",
                "calvin_scene_B",
                "calvin_scene_C",
                "calvin_scene_D",
            ]
        else:
            raise NotImplementedError(
                f"task suite {self.cfg.task_suite_name} is not yet supported."
            )
        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                continue
            scene_idx = env_id % len(candidated_scenes)
            env_fn_params.append({"scene": candidated_scenes[scene_idx]})
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.num_tasks):
            self.trial_id_bins.append(self.task_num_trials)
            self.total_num_group_envs += self.task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

    def update_reset_state_ids(self):
        if self.cfg.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _init_task_info(self):
        self.task_sequence = [None] * self.num_envs
        self.current_task = [None] * self.num_envs
        self.current_task_idx = [0] * self.num_envs
        self.previous_info = [None] * self.num_envs
        self.task_descriptions = [None] * self.num_envs

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        reset_state_ids = np.arange(self.total_num_group_envs)
        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        self._generator_ordered.shuffle(reset_state_ids)
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        """
        By constructing a 2D seed matrix (self.reset_state_ids_all) and applying a seed_offset,
        users can evenly distribute the initialization of environments (e.g., 1,000 states) across all available GPUs, ensuring each GPU runs a unique set of tasks.

        Args:
            num_reset_states (int): The number of environment reset states to allocate to this env worker.

        Returns:
            np.ndarray: An array of reset state IDs of length `num_reset_states`, assigned in an ordered fashion
                according to the current process's seed_offset.

        Notes:
            Each row in `self.reset_state_ids_all` corresponds to the reset state IDs allocated to a specific
            process (distinguished by seed_offset), ensuring non-overlapping task subsets for parallel evaluation or training.
        """
        if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
            self.reset_state_ids_all = self.get_reset_state_ids_all()
            self.start_idx = 0
        reset_state_ids = self.reset_state_ids_all[self.seed_offset][
            self.start_idx : self.start_idx + num_reset_states
        ]
        self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
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

    def _get_reset_states(self, env_idx):
        init_state = [
            self.task_suite.get_task_init_states(self.trial_ids[env_id])
            for env_id in env_idx
        ]
        return init_state

    def _get_task_info(self, env_idx):
        for i in env_idx:
            self.task_sequence[i] = self.task_suite.get_task_sequence(self.trial_ids[i])
            task = self.task_sequence[i][0]
            self.current_task[i] = task
            self.current_task_idx[i] = 0
            self.previous_info[i] = self.env.get_info(id=i)[0]
            self.task_descriptions[i] = self.task_suite.get_task_descriptions(task)

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
        episode_info["avg_len"] = self.current_task_idx.copy()
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        img = obs["rgb_obs"]["rgb_static"]
        wrist_img = obs["rgb_obs"]["rgb_gripper"]
        state = obs["robot_obs"][:7]

        return {
            "full_image": img,
            "wrist_image": wrist_img,
            "state": state,
        }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["wrist_image"]]
        )
        states = images_and_states["state"]

        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }

        return obs

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
        init_state = self._get_reset_states(env_idx=env_idx)
        robot_obs, scene_obs = self.task_suite.get_obs_for_initial_condition(init_state)
        self.env.reset(id=env_idx, robot_obs=robot_obs, scene_obs=scene_obs)
        self._get_task_info(env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if self.is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        raw_obs = self.env.get_obs()
        obs = self._wrap_obs(raw_obs)
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        infos = {}
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _, _, info_lists = self.env.step(actions)
        subtask_success = self._check_subtask_success(info_lists)
        self._reset_current_task(subtask_success, info_lists)
        infos = list_of_dict_to_dict_of_list(info_lists)
        terminations = np.array(self.current_task_idx) == 5
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(subtask_success)

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

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

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
        return reward

    def _check_subtask_success(self, info_lists):
        subtask_success_list = []
        for env_id, info in enumerate(info_lists):
            info = info_lists[env_id]
            prev_info = self.previous_info[env_id]
            sub_task = self.current_task[env_id]
            subtask_success = self.task_suite.check_subtask_success(
                prev_info, info, sub_task
            )
            if subtask_success:
                subtask_success_list.append(True)
            else:
                subtask_success_list.append(False)
        return np.array(subtask_success_list)

    def _reset_current_task(self, subtask_success, info_lists):
        for env_id, success in enumerate(subtask_success):
            if success:
                self.current_task_idx[env_id] += 1
                if self.current_task_idx[env_id] <= 4:
                    self.previous_info[env_id] = info_lists[env_id]
                    self.current_task[env_id] = self.task_sequence[env_id][
                        self.current_task_idx[env_id]
                    ]
                    self.task_descriptions[env_id] = (
                        self.task_suite.get_task_descriptions(self.current_task[env_id])
                    )
                else:
                    self.current_task_idx[env_id] = 5
