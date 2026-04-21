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
import numpy as np
import torch
from metasim.task.registry import get_task_class
from metasim.utils.ik_solver import setup_ik_solver

from rlinf.envs.roboverse.utils import (
    apply_action_guard,
    build_policy_states,
    build_roboverse_camera_cfgs,
    cfg_get,
    convert_roboverse_action,
    extract_roboverse_obs,
    get_valid_joint_names,
    resolve_camera_rgb,
    select_initial_states,
    serialize_actions_dict,
)
from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    to_tensor,
)
from rlinf.utils.logging import get_logger


class RoboVerseEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        # default codes
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + self.seed_offset
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

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        init_params = getattr(self.cfg, "init_params", None)
        self.enable_wrist_camera = bool(
            cfg_get(self.cfg, init_params, "enable_wrist_camera", True)
        )

        simulator = self.cfg.simulator_backend
        task_cls = get_task_class(self.cfg.task_name)
        robot = self.cfg.robot_name
        self.robot_name = robot
        self.main_camera_name, self.wrist_camera_name, cameras = (
            build_roboverse_camera_cfgs(self.cfg, task_cls, self.robot_name)
        )

        scenario = task_cls.scenario.update(
            robots=[robot],
            simulator=simulator,
            num_envs=self.num_envs,
            headless=self.cfg.headless,
            env_spacing=2.5,
            cameras=cameras,
        )
        self.scenario = scenario

        self.env = task_cls(scenario=scenario, device="cuda")
        self.reorder_idx = self.env.handler.get_joint_reindex(self.robot_name)
        self.inverse_reorder_idx = [0] * len(self.reorder_idx)
        for new_idx, old_idx in enumerate(self.reorder_idx):
            self.inverse_reorder_idx[old_idx] = new_idx

        initial_states = getattr(self.env, "_initial_states", None)
        self.total_num_group_envs = (
            len(initial_states) if initial_states is not None else self.num_group
        )

        self.reset_state_ids_all = self.get_reset_state_ids_all()
        if self.use_fixed_reset_state_ids:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
            self.reset_state_ids = reset_state_ids.repeat(self.group_size)
        else:
            self.reset_state_ids = None

        self.solver = "pyroki"
        self._setup_ik()
        self.ee_body_idx = None
        self.ee_body_name = self.scenario.robots[0].ee_body_name
        task_description = str(
            getattr(self.env, "task_desc", None)
            or getattr(task_cls, "task_desc", None)
            or self.cfg.task_name
        )
        self.task_descriptions = [task_description for _ in range(self.num_envs)]

        self.action_coordinate_frame = getattr(cfg, "action_coordinate_frame", "world")
        self.action_dim = int(getattr(cfg, "action_dim", 7))
        self.state_dim = int(
            getattr(cfg, "state_dim", getattr(cfg, "policy_state_dim", 8))
        )
        self.action_clip = float(getattr(cfg, "action_clip", 1.0))
        self.ee_pos_delta_scale = float(getattr(cfg, "ee_pos_delta_scale", 0.02))
        self.ee_rot_delta_scale = float(getattr(cfg, "ee_rot_delta_scale", 0.05))
        self._prev_guarded_action = None

        self.last_obs = None

    def get_reset_state_ids_all(self):
        reset_state_ids = np.arange(self.total_num_group_envs)
        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        self._generator_ordered.shuffle(reset_state_ids)
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

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

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
        options: Optional[dict] = {},
    ):
        env_idx = self._normalize_env_idx(env_idx)

        if self.is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            reset_state_ids = self._get_random_reset_state_ids(len(env_idx))
        else:
            reset_state_ids = np.asarray(reset_state_ids, dtype=np.int64).reshape(-1)
            if reset_state_ids.size == 1 and len(env_idx) > 1:
                reset_state_ids = np.repeat(reset_state_ids, len(env_idx))
            elif reset_state_ids.size != len(env_idx):
                reset_state_ids = np.resize(reset_state_ids, len(env_idx))

        selected_states = select_initial_states(
            self.env._initial_states, reset_state_ids
        )
        raw_obs, _ = self.env.reset(states=selected_states, env_ids=env_idx.tolist())
        self.last_obs = raw_obs

        self._reset_metrics(env_idx)

        self._step_counter = 0
        infos = {}

        obs_dict = self._wrap_obs(raw_obs)

        return obs_dict, infos

    def _normalize_env_idx(self, env_idx):
        if env_idx is None:
            return np.arange(self.num_envs, dtype=np.int64)
        if np.isscalar(env_idx):
            return np.asarray([int(env_idx)], dtype=np.int64)
        return np.asarray(env_idx, dtype=np.int64).reshape(-1)

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

    def _get_main_camera_rgb(self, raw_obs):
        camera_name, camera_rgb = resolve_camera_rgb(
            raw_obs, self.main_camera_name, prefer_wrist=False
        )
        self._resolved_main_camera_name = camera_name
        return camera_rgb

    def _get_wrist_camera_rgb(self, raw_obs):
        if not self.enable_wrist_camera:
            return None
        camera_name, camera_rgb = resolve_camera_rgb(
            raw_obs, self.wrist_camera_name, prefer_wrist=True
        )
        self._resolved_wrist_camera_name = camera_name
        return camera_rgb

    def _resolve_camera_rgb(self, raw_obs, preferred_name: str, prefer_wrist: bool):
        return resolve_camera_rgb(raw_obs, preferred_name, prefer_wrist)

    def _extract_image_and_state(self, raw_obs):
        extracted, self.ee_body_idx = extract_roboverse_obs(
            raw_obs=raw_obs,
            main_camera_name=self.main_camera_name,
            wrist_camera_name=self.wrist_camera_name,
            robot_name=self.robot_name,
            ee_body_name=self.ee_body_name,
            inverse_reorder_idx=self.inverse_reorder_idx,
            device=self.device,
            ee_body_idx=self.ee_body_idx,
            state_dim=self.state_dim,
            has_wrist_camera=self.enable_wrist_camera,
        )
        return extracted

    def _wrap_obs(self, raw_obs):
        extracted = self._extract_image_and_state(raw_obs)
        wrist_images = (
            extracted["wrist_image"].permute(0, 3, 1, 2)
            if extracted["wrist_image"] is not None
            else None
        )
        return {
            "main_images": extracted["full_image"].permute(0, 3, 1, 2),
            "wrist_images": wrist_images,
            "task_descriptions": self.task_descriptions,
            "states": extracted["state"],
        }

    def step(self, actions=None, auto_reset=True):
        """
        Step through the environment for one timestep.

        Args:
            actions (np.ndarray or torch.Tensor): Actions for each env.
        Returns:
            obs (torch.Tensor): Wrapped observations.
            rewards (torch.Tensor): Step rewards.
            terminations (torch.Tensor): Done due to success/failure.
            truncations (torch.Tensor): Done due to time limit.
            infos (dict): Additional information.
        """
        # 1. deal with reset
        if actions is None:
            if not self._is_start:
                logger = get_logger()
                msg = "Actions must be provided after the first reset."
                logger.error(msg)
                raise ValueError(msg)

        # 2. convert end-effector commands to robot actions
        policy_actions = torch.as_tensor(actions, dtype=torch.float32)
        actions = self._convert_action(policy_actions)

        if isinstance(actions, dict):
            actions = self._serialize_actions_dict(actions)
        elif isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        # 3. interact with metasim
        raw_obs, reward, terminated, time_out, info_dict = self.env.step(actions)
        # MetaSim returns a single info dict, we need to create per-env list
        if isinstance(info_dict, dict):
            info_list = [info_dict for _ in range(self.num_envs)]
            infos = list_of_dict_to_dict_of_list(info_list)
        else:
            infos = list_of_dict_to_dict_of_list(info_dict)
        self.last_obs = raw_obs

        # 4. time steps
        self._elapsed_steps += 1
        terminations = terminated.cpu().numpy()
        truncations = self._elapsed_steps >= self.cfg.max_episode_steps
        if isinstance(time_out, torch.Tensor):
            env_timeouts = time_out.cpu().numpy()
        else:
            env_timeouts = np.array(time_out, dtype=bool)
        truncations = truncations | env_timeouts

        # 5. generate rewards
        step_reward = self._calc_step_reward(terminations)

        # 6. get image/state observation via unified extraction path (same style as Libero)
        extracted = self._extract_image_and_state(raw_obs)

        # 7. reward metrics
        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        # Compute dones after optional termination masking so auto-reset
        dones = terminations | truncations

        obs_dict = {
            "main_images": extracted["full_image"].permute(0, 3, 1, 2),
            "wrist_images": (
                extracted["wrist_image"].permute(0, 3, 1, 2)
                if extracted["wrist_image"] is not None
                else None
            ),
            "task_descriptions": self.task_descriptions,
            "states": extracted["state"],
        }

        # 8. auto-reset
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs_dict, infos = self._handle_auto_reset(dones, obs_dict, infos)

        self.last_obs = raw_obs

        # 9. return information in Gym style
        return (
            obs_dict,
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

        if self.cfg.only_eval:
            self.update_reset_state_ids()

        obs_dict, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs_dict, infos

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _build_policy_states(self, raw_obs):
        states_tensor, self.ee_body_idx = build_policy_states(
            raw_obs=raw_obs,
            robot_name=self.robot_name,
            ee_body_name=self.ee_body_name,
            inverse_reorder_idx=self.inverse_reorder_idx,
            device=self.device,
            ee_body_idx=self.ee_body_idx,
            state_dim=self.state_dim,
        )
        return states_tensor

    def update_reset_state_ids(self):
        if self.num_envs == 1:
            return
        if self.cfg.only_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
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
                low=0,
                high=self.total_num_group_envs,
                size=(num_reset_states,),
            )
        return reset_state_ids

    # ---------------- IK ----------------
    def _setup_ik(self):
        self.robot_cfg = self.scenario.robots[0]
        self.ik_solver = setup_ik_solver(self.robot_cfg, self.solver)

    def _serialize_actions_dict(self, actions_dict):
        return serialize_actions_dict(
            actions_dict,
            valid_joint_names=get_valid_joint_names(
                self.env,
                self.robot_name,
                last_obs=self.last_obs,
                robot_cfg=self.robot_cfg,
            ),
        )

    def _apply_action_guard(self, action: torch.Tensor):
        """Apply clip/scale"""
        return apply_action_guard(
            action,
            action_clip=self.action_clip,
            ee_pos_delta_scale=self.ee_pos_delta_scale,
            ee_rot_delta_scale=self.ee_rot_delta_scale,
        )

    def _convert_action(self, action):
        return convert_roboverse_action(
            action=action,
            last_obs=self.last_obs,
            robot_name=self.robot_name,
            inverse_reorder_idx=self.inverse_reorder_idx,
            ee_body_name=self.ee_body_name,
            ee_body_idx=self.ee_body_idx,
            ik_solver=self.ik_solver,
            robot_cfg=self.robot_cfg,
            action_coordinate_frame=self.action_coordinate_frame,
            action_clip=self.action_clip,
            ee_pos_delta_scale=self.ee_pos_delta_scale,
            ee_rot_delta_scale=self.ee_rot_delta_scale,
            device=self.device,
        )

    @property
    def device(self):
        return "cuda:0"
