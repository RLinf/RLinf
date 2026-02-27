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
import pathlib
import time
from functools import partial
from typing import OrderedDict

import gymnasium as gym
import numpy as np
import psutil
import torch
from filelock import FileLock
from omegaconf import OmegaConf

from rlinf.envs.realworld.common.wrappers import (
    GripperCloseEnv,
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
    Quat2EulerWrapper,
    RelativeFrame,
    SpacemouseIntervention,
)
from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv
from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo


class RealWorldEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        assert num_envs == 1, (
            f"Currently, only 1 realworld env can be started per worker, but {num_envs=} is received."
        )

        self.cfg = cfg
        self.override_cfg = OmegaConf.to_container(
            cfg.get("override_cfg", {}), resolve=True
        )

        self.video_cfg = cfg.video_cfg

        self.seed = cfg.seed + seed_offset
        self.num_envs = num_envs
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.main_image_key = cfg.main_image_key

        self._init_env()

        self._is_start = True
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._init_reset_state_ids()

    def _create_env(self, env_idx: int):
        worker_info: WorkerInfo = self.worker_info
        hardware_info = None
        if worker_info is not None and env_idx < len(worker_info.hardware_infos):
            hardware_info = worker_info.hardware_infos[env_idx]
        override_cfg = copy.deepcopy(self.override_cfg)
        env = gym.make(
            id=self.cfg.init_params.id,
            override_cfg=override_cfg,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
        )
        if self.cfg.get("no_gripper", True):
            env = GripperCloseEnv(env)
        if not env.config.is_dummy and self.cfg.get("use_spacemouse", True):
            env = SpacemouseIntervention(env)
        if not env.config.is_dummy and self.cfg.get("keyboard_reward_wrapper", None):
            if self.cfg.keyboard_reward_wrapper == "multi_stage":
                env = KeyboardRewardDoneMultiStageWrapper(env)
            elif self.cfg.keyboard_reward_wrapper == "single_stage":
                env = KeyboardRewardDoneWrapper(env)

        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        return env

    @staticmethod
    def realworld_setup():
        """Setup RealWorld environment upon env class import.

        This is for any node-level setup required by RealWorld environments. For example, ROS
        requires a single roscore instance per node, so we ensure that any existing roscore
        processes are terminated before starting a new one.

        This function is called once when the RealWorldEnv class is first imported.
        """
        # Concurrency control is needed for multiple processes on the same node
        node_lock_file = "/tmp/.realworld.lock"
        # Check if the path is valid
        if not os.path.exists(os.path.dirname(node_lock_file)):
            node_lock_file = os.path.join(pathlib.Path.home(), ".realworld.lock")
        node_lock = FileLock(node_lock_file)

        with node_lock:
            ros_proc_names = ["roscore", "rosmaster", "rosout"]
            for proc in psutil.process_iter():
                if proc.name() in ros_proc_names:
                    proc.kill()
                    time.sleep(0.5)

    def _init_env(self):
        env_fns = [
            partial(self._create_env, env_idx=env_idx)
            for env_idx in range(self.num_envs)
        ]
        self.env = NoAutoResetSyncVectorEnv(env_fns)
        self.task_descriptions = list(self.env.call("task_description"))

    @property
    def total_num_group_envs(self):
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _init_metrics(self):
        self.prev_step_reward = np.zeros(self.num_envs)

        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[mask] = 0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, intervene_current_step, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        self.intervened_once = self.intervened_once | intervene_current_step
        self.intervened_steps += intervene_current_step.astype(int)

        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["intervened_once"] = self.intervened_once
        episode_info["intervened_steps"] = self.intervened_steps
        episode_info["success_no_intervened"] = self.success_once.copy() & (
            ~self.intervened_once
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def reset(self, *, reset_state_ids=None, seed=None, options=None, env_idx=None):
        # TODO: handle partial reset
        raw_obs, infos = self.env.reset(seed=seed, options=options)

        extracted_obs = self._wrap_obs(raw_obs)
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, infos

    def _wrap_obs(self, raw_obs):
        """
        raw_obs: Dict of list
        """
        obs = {}

        # Process states
        full_states = []
        raw_states = OrderedDict(sorted(raw_obs["state"].items()))
        for value in raw_states.values():
            full_states.append(value)
        full_states = np.concatenate(full_states, axis=-1)
        obs["states"] = full_states

        # Process images
        if self.main_image_key not in raw_obs["frames"]:
            available_keys = list(raw_obs["frames"].keys())
            raise KeyError(
                f"main_image_key '{self.main_image_key}' not found in raw_obs['frames']. "
                f"Available keys: {available_keys}. "
                f"Please set 'main_image_key' in your env config to one of the available keys."
            )
        obs["main_images"] = raw_obs["frames"][self.main_image_key]
        raw_images = OrderedDict(sorted(raw_obs["frames"].items()))
        raw_images.pop(self.main_image_key)

        if raw_images:
            obs["extra_view_images"] = np.stack(list(raw_images.values()), axis=1)

        obs = to_tensor(obs)
        obs["task_descriptions"] = self.task_descriptions
        return obs

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps

        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(_reward)

        intervene_flag = np.zeros(self.num_envs, dtype=bool)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                if infos["intervene_action"][env_id] is not None:
                    intervene_flag[env_id] = True

        infos = self._record_metrics(step_reward, terminations, intervene_flag, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        intervene_action = np.zeros_like(actions)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                env_intervene_action = infos["intervene_action"][env_id]
                if env_intervene_action is not None:
                    intervene_action[env_id] = env_intervene_action.copy()
        infos["intervene_action"] = to_tensor(intervene_action)
        infos["intervene_flag"] = to_tensor(intervene_flag)

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

        raw_chunk_intervene_actions = []
        raw_chunk_intervene_flag = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            if "intervene_action" in infos:
                raw_chunk_intervene_actions.append(infos["intervene_action"])
                raw_chunk_intervene_flag.append(infos["intervene_flag"])

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

        infos_last = infos_list[-1] if infos_list else {}
        if raw_chunk_intervene_actions:
            infos_last["intervene_action"] = torch.stack(
                raw_chunk_intervene_actions, dim=1
            ).reshape(self.num_envs, -1)
            infos_last["intervene_flag"] = torch.stack(raw_chunk_intervene_flag, dim=1)
            infos_list[-1] = infos_last

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

    def _calc_step_reward(self, reward: np.ndarray):
        return reward.astype(np.float32)

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids

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
        )
