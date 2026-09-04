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

import copy
import os
import pathlib
import time
from functools import partial
from typing import Any, Mapping, Optional, OrderedDict, Union

import gymnasium as gym
import numpy as np
import psutil
import torch
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from rlinf.envs.real.venv import NoAutoResetSyncVectorEnv
from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo

#: One batched observation, by the keys the runner reads: ``states``, the image
#: keys a config names, and whatever a task adds.
Observation = dict[str, Any]

#: The per-env info dicts a step or reset returns, plus the metrics recorded
#: into them.
EnvInfos = dict[str, Any]

#: What :meth:`RealWorldEnv.step` answers with: observation, reward,
#: termination, truncation, infos -- the Gymnasium five, batched.
StepResult = tuple[Observation, Tensor, Tensor, Tensor, EnvInfos]

#: The same, one entry per step of an action chunk.
ChunkStepResult = tuple[
    list[Observation], list[Tensor], list[Tensor], list[Tensor], list[EnvInfos]
]


class RealWorldEnv(gym.Env):
    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: WorkerInfo,
    ) -> None:
        assert num_envs == 1, (
            f"Currently, only 1 realworld env can be started per worker, but {num_envs=} is received."
        )

        self.cfg = cfg
        self.override_cfg = OmegaConf.to_container(
            cfg.get("override_cfg", OmegaConf.create({})), resolve=True
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
        self.manual_episode_control_only = bool(
            self.override_cfg.get("manual_episode_control_only", False)
        )

        self._init_env()

        self._is_start = True
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._init_reset_state_ids()

    def _create_env(self, env_idx: int) -> gym.Env:
        worker_info: WorkerInfo = self.worker_info
        robot_info = None
        if worker_info is not None and env_idx < len(worker_info.hardware_infos):
            robot_info = worker_info.hardware_infos[env_idx]
        override_cfg = copy.deepcopy(self.override_cfg)
        # A worker that was handed this class by value never ran the package
        # import that registers the task ids.
        from rlinf.envs.real import load_tasks

        load_tasks()
        env = gym.make(
            id=self.cfg.init_params.id,
            override_cfg=override_cfg,
            worker_info=worker_info,
            robot_info=robot_info,
            env_idx=env_idx,
            env_cfg=self.cfg,
        )
        return env

    @staticmethod
    def realworld_setup() -> None:
        """Run node-level setup before creating real-world environments.

        The setup is serialized because ROS permits only one core per node.
        Existing ROS core processes are stopped before environments start.
        """
        # Serialize setup across environment processes on the same node.
        node_lock_file = "/tmp/.realworld.lock"
        # Fall back to the user directory if the temporary path is unavailable.
        if not os.path.exists(os.path.dirname(node_lock_file)):
            node_lock_file = os.path.join(pathlib.Path.home(), ".realworld.lock")
        node_lock = FileLock(node_lock_file)

        with node_lock:
            ros_proc_names = ["roscore", "rosmaster", "rosout"]
            for proc in psutil.process_iter():
                if proc.name() in ros_proc_names:
                    proc.kill()
                    time.sleep(0.5)

    def _init_env(self) -> None:
        env_fns = [
            partial(self._create_env, env_idx=env_idx)
            for env_idx in range(self.num_envs)
        ]
        self.env = NoAutoResetSyncVectorEnv(env_fns)
        self.task_descriptions = list(
            self.env.call("get_wrapper_attr", "task_description")
        )

    def get_hold_actions(
        self, fallback_actions: np.ndarray | None = None
    ) -> np.ndarray:
        """Return per-environment actions that hold the current robot state.

        Absolute-action teleoperation wrappers provide their own hold action.
        Other environments use a zero action.
        """
        action_dim = int(self.action_space.shape[-1])
        holds: list[np.ndarray] = []
        for env_id, env in enumerate(self.env.envs):
            fallback = None
            if fallback_actions is not None:
                fallback = np.asarray(fallback_actions[env_id], dtype=np.float32)

            # A wrapper stack without a hold action falls back to zero.
            try:
                hold = np.asarray(
                    env.get_wrapper_attr("get_hold_action")(fallback),
                    dtype=np.float32,
                ).reshape(-1)
            except AttributeError:
                hold = np.zeros(action_dim, dtype=np.float32)

            if hold.size != action_dim:
                raise ValueError(
                    "get_hold_actions expected action dim "
                    f"{action_dim}, got {hold.size} for env_id={env_id}."
                )
            holds.append(hold)
        return np.stack(holds, axis=0)

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def total_num_group_envs(self) -> int:
        # TODO(agent): Replace this placeholder with task-specific reset-state data.
        return np.iinfo(np.uint8).max // 2

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    def _init_metrics(self) -> None:
        self.prev_step_reward = np.zeros(self.num_envs)

        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)

    def _reset_metrics(self, env_idx: Optional[np.ndarray] = None) -> None:
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

    def _record_metrics(
        self,
        step_reward: np.ndarray,
        terminations: np.ndarray,
        success_current_step: np.ndarray,
        intervene_current_step: np.ndarray,
        infos: EnvInfos,
    ) -> EnvInfos:
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | success_current_step
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

    def reset(
        self,
        *,
        reset_state_ids: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
        env_idx: Optional[np.ndarray] = None,
    ) -> tuple[Observation, EnvInfos]:
        # TODO(agent): Honor reset_state_ids for partial real-environment resets.
        raw_obs, infos = self.env.reset(seed=seed, options=options)

        extracted_obs = self._wrap_obs(raw_obs)
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, infos

    def _wrap_obs(self, raw_obs: Mapping[str, Any]) -> Observation:
        """Convert batched raw observations to the runner representation."""
        obs = {}

        state = raw_obs["state"]
        full_states = np.concatenate([state[k] for k in sorted(state)], axis=-1)
        obs["states"] = full_states

        frames = raw_obs["frames"]
        if self.main_image_key not in frames:
            raise KeyError(
                f"main_image_key {self.main_image_key!r} not in {list(frames)}"
            )
        obs["main_images"] = frames[self.main_image_key]
        raw_images = OrderedDict(sorted(frames.items()))
        raw_images.pop(self.main_image_key)

        if raw_images:
            obs["extra_view_images"] = np.stack(list(raw_images.values()), axis=1)

        obs = to_tensor(obs)
        obs["task_descriptions"] = self.task_descriptions
        return obs

    def step(
        self,
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        auto_reset: bool = True,
    ) -> StepResult:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        # A null limit delegates episode boundaries to an external wrapper.
        if self.cfg.max_episode_steps is None:
            timeout_truncations = np.zeros_like(truncations, dtype=bool)
        else:
            timeout_truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        if not self.manual_episode_control_only:
            truncations = timeout_truncations

        obs = self._wrap_obs(raw_obs)
        step_reward = self._calc_step_reward(_reward)
        success_current_step = np.isclose(step_reward, 1.0)
        intervene_flag = np.zeros(self.num_envs, dtype=bool)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                if infos["intervene_action"][env_id] is not None:
                    intervene_flag[env_id] = True

        infos = self._record_metrics(
            step_reward,
            terminations,
            success_current_step,
            intervene_flag,
            infos,
        )
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
        if "rlt_switch_flags" in infos:
            infos["rlt_switch_flags"] = to_tensor(
                np.asarray(infos["rlt_switch_flags"], dtype=bool)
            )

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

    def _notify_action_chunk_begin(self) -> None:
        """Tell intervention wrappers a new action chunk is starting."""
        for env in self.env.envs:
            try:
                on_begin = env.get_wrapper_attr("on_action_chunk_begin")
            except AttributeError:
                continue
            if callable(on_begin):
                on_begin()

    def chunk_step(self, chunk_actions: np.ndarray) -> ChunkStepResult:
        # Shape: [num_envs, chunk_steps, action_dim].
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []

        raw_chunk_intervene_actions = []
        raw_chunk_intervene_flag = []
        raw_chunk_rlt_switch_flags = []
        self._notify_action_chunk_begin()
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
            if "rlt_switch_flags" in infos:
                raw_chunk_rlt_switch_flags.append(infos["rlt_switch_flags"])

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
        if raw_chunk_rlt_switch_flags:
            infos_last["rlt_switch_flags"] = torch.stack(
                raw_chunk_rlt_switch_flags, dim=1
            )
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

    def _handle_auto_reset(
        self,
        dones: np.ndarray,
        _final_obs: Observation,
        infos: EnvInfos,
    ) -> tuple[Observation, EnvInfos]:
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=(
                self.reset_state_ids[env_idx]
                if self.use_fixed_reset_state_ids
                else None
            ),
        )
        # Follow Gymnasium's final-observation naming for the pre-reset result.
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, reward: np.ndarray) -> np.ndarray:
        return reward.astype(np.float32)

    def _get_random_reset_state_ids(self, num_reset_states: int) -> np.ndarray:
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids

    def _init_reset_state_ids(self) -> None:
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self) -> None:
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        )
