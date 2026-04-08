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
#
# GenieSimEnv — RLinf gymnasium.Env wrapper for the GeneSim lightweight RL
# pipeline (MuJoCo physics + IsaacSim rendering, no Cosine / mc-in-loop).
#
# The sim-side GenieSimVectorEnv organises obs, reward, terminated, truncated,
# info.  This wrapper converts numpy results to torch, tracks local metrics
# for RLinf's logging infrastructure, and allows subclasses to override the
# reward computation (e.g. PlaceWorkpieceEnv computes dense reward from
# info["body_poses"]).

from __future__ import annotations

import copy
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.geniesim import REGISTER_GENIESIM_ENVS  # noqa: F401
from rlinf.envs.geniesim.container_manager import SimContainerManager
from rlinf.envs.geniesim.process_manager import SimProcessManager
from rlinf.envs.geniesim.shm_client import GenieSimShmClient, GenieSimVectorEnvConfig
from rlinf.envs.geniesim.sim_manager_base import BaseSimManager


class GenieSimBaseEnv(gym.Env):
    """
    Base class for all GeneSim RL environments in RLinf.

    Subclasses must implement:
        _make_vec_env_config()  -> GenieSimVectorEnvConfig
        _wrap_obs(obs_dict)     -> obs_dict in RLinf canonical format

    The actual simulation runs in GenieSimVectorEnv (sim side), which manages
    MuJoCo + IsaacSim lifecycle and organises obs/reward/terminated/truncated/info.
    This class delegates to GenieSimShmClient for SHM-based communication.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: dict,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.video_cfg = getattr(cfg, "video_cfg", {})

        self.task_description = cfg.init_params.task_description
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self._is_start = True

        self._elapsed_steps = np.zeros(num_envs, dtype=np.int32)
        self._prev_step_reward = np.zeros(num_envs, dtype=np.float32)
        self.returns = np.zeros(num_envs, dtype=np.float32)
        self.success_once = np.zeros(num_envs, dtype=bool)
        self.fail_once = np.zeros(num_envs, dtype=bool)

        container_cfg = cfg.container_cfg
        _mode = str(getattr(container_cfg, "mode", "auto"))
        if _mode == "auto":
            _mode = "local" if os.environ.get("GENIESIM_CONTAINER") else "docker"
        if _mode == "local":
            self._container_mgr: BaseSimManager = SimProcessManager(container_cfg)
        else:
            self._container_mgr = SimContainerManager(container_cfg)

        vec_cfg = self._make_vec_env_config()

        pm_kwargs = BaseSimManager.pm_kwargs_from_vec_cfg(vec_cfg)
        self._container_mgr.ensure_running(pm_kwargs)
        vec_cfg.attach_to_running = True
        self.env = GenieSimShmClient(vec_cfg)

    # ---------------------------------------------------------------------- #
    # To override in subclasses
    # ---------------------------------------------------------------------- #

    def _make_vec_env_config(self):
        import json as _json

        _rand_cfg = getattr(self.cfg, "randomization", None)
        _object_map = getattr(self.cfg, "object_map", None)
        _rand_json = ""
        if _rand_cfg is not None or _object_map is not None:
            try:
                from omegaconf import OmegaConf

                _rand_dict = (
                    OmegaConf.to_container(_rand_cfg, resolve=True)
                    if _rand_cfg is not None
                    else {}
                )
                if _object_map is not None:
                    _rand_dict["_object_map"] = OmegaConf.to_container(
                        _object_map, resolve=True
                    )
                _rand_json = _json.dumps(_rand_dict)
            except Exception:
                pass
        _init_qpos = getattr(self.cfg.init_params, "init_qpos", None)
        _init_qpos_json = (
            _json.dumps([float(v) for v in _init_qpos])
            if _init_qpos is not None
            else ""
        )
        _reset_ee_r = getattr(self.cfg.init_params, "reset_ee_r", None)
        _reset_ee_r_json = (
            _json.dumps(list(_reset_ee_r)) if _reset_ee_r is not None else ""
        )
        p = self.cfg.init_params
        resolved_cameras = self._resolve_cameras(p)
        return GenieSimVectorEnvConfig(
            mjcf_path=getattr(p, "mjcf_path", ""),
            scene_usd=getattr(p, "scene_usd", ""),
            robot_usd=getattr(p, "robot_usd", ""),
            robot_prim=getattr(p, "robot_prim", "/robot"),
            task_file=getattr(p, "task_file", ""),
            task_name=p.id,
            task_description=getattr(p, "task_description", ""),
            robot_cfg=getattr(p, "robot_cfg", "G2_omnipicker"),
            robot_type=getattr(p, "robot_type", "G2"),
            task_instance_id=getattr(p, "task_instance_id", 0),
            num_envs=self.num_envs,
            max_episode_steps=self.cfg.max_episode_steps,
            cameras=resolved_cameras,
            cam_width=resolved_cameras[0].get("width", 640)
            if resolved_cameras
            else getattr(p, "cam_width", 640),
            cam_height=resolved_cameras[0].get("height", 480)
            if resolved_cameras
            else getattr(p, "cam_height", 480),
            main_cam_prim=getattr(p, "main_cam_prim", "/camera_main"),
            wrist_cam_prim=getattr(p, "wrist_cam_prim", ""),
            enable_reward=getattr(self.cfg, "enable_reward", True),
            reward_coef=getattr(self.cfg, "reward_coef", 1.0),
            use_rel_reward=self.use_rel_reward,
            ignore_terminations=self.ignore_terminations,
            auto_reset=self.auto_reset,
            physics_hz=getattr(p, "physics_hz", 1000),
            render_hz=getattr(p, "render_hz", 30.0),
            shm_name=getattr(p, "shm_name", "geniesim_frames"),
            headless=getattr(p, "headless", True),
            ros_domain_id=getattr(p, "ros_domain_id", 0),
            isaac_python=getattr(p, "isaac_python", "/isaac-sim/python.sh"),
            state_dim=getattr(p, "state_dim", 28),
            action_dim=getattr(p, "action_dim", 14),
            state_joint_offset=getattr(p, "state_joint_offset", 0),
            ctrl_offset=getattr(p, "ctrl_offset", 0),
            control_mode=getattr(p, "control_mode", "joint"),
            gripper_ctrl_l=getattr(p, "gripper_ctrl_l", -1),
            gripper_ctrl_r=getattr(p, "gripper_ctrl_r", -1),
            ee_body_l=getattr(p, "ee_body_l", "arm_l_link7"),
            ee_body_r=getattr(p, "ee_body_r", "arm_r_link7"),
            ik_max_iter=getattr(p, "ik_max_iter", 10),
            ik_damp=getattr(p, "ik_damp", 0.05),
            randomization_cfg_json=_rand_json,
            init_qpos_json=_init_qpos_json,
            reset_ee_r_json=_reset_ee_r_json,
            seed=getattr(self.cfg, "seed", 42),
            mujoco_python=getattr(p, "mujoco_python", ""),
            info_body_names=list(getattr(p, "info_body_names", []) or []),
        )

    @staticmethod
    def _resolve_cameras(params) -> list:
        raw = getattr(params, "cameras", None)
        if raw is not None:
            try:
                from omegaconf import OmegaConf

                return [dict(c) for c in OmegaConf.to_container(raw, resolve=True)]
            except Exception:
                return list(raw)
        cams = []
        main_prim = getattr(params, "main_cam_prim", "")
        w = getattr(params, "cam_width", 640)
        h = getattr(params, "cam_height", 480)
        if main_prim:
            cams.append({"name": "main", "prim": main_prim, "width": w, "height": h})
        wrist_prim = getattr(params, "wrist_cam_prim", "")
        if wrist_prim:
            cams.append({"name": "wrist", "prim": wrist_prim, "width": w, "height": h})
        return cams or [
            {"name": "main", "prim": "/camera_main", "width": 640, "height": 480}
        ]

    def _wrap_obs(self, obs_dict: dict) -> dict:
        states = obs_dict["states"]
        task_descriptions = obs_dict["task_descriptions"]
        states = self._extract_states(states)

        result: dict = {
            "states": torch.from_numpy(states),
            "task_descriptions": task_descriptions,
        }

        cameras = getattr(self.env.cfg, "cameras", None) or []
        if cameras:
            first_key = f"{cameras[0]['name']}_images"
            img = obs_dict.get(first_key)
            if img is not None:
                result["main_images"] = torch.from_numpy(img)
            else:
                result["main_images"] = None
            extras = []
            for cam_cfg in cameras[1:]:
                key = f"{cam_cfg['name']}_images"
                img = obs_dict.get(key)
                if img is not None:
                    extras.append(torch.from_numpy(img))
            if extras:
                result["extra_view_images"] = torch.stack(extras, dim=1)
        else:
            main_images = obs_dict.get("main_images")
            if main_images is not None:
                result["main_images"] = torch.from_numpy(main_images)
            else:
                result["main_images"] = None

        result = self._extract_images(result)
        return result

    def _extract_images(self, obs_dict: dict) -> dict:
        return obs_dict

    def _extract_states(self, states: np.ndarray) -> np.ndarray:
        return states

    def _expand_actions(self, actions: np.ndarray) -> np.ndarray:
        return actions

    # ---------------------------------------------------------------------- #
    # gym.Env interface
    # ---------------------------------------------------------------------- #

    def reset(
        self,
        seed: Optional[int] = None,
        env_ids: Optional[np.ndarray] = None,
        **kwargs,
    ) -> tuple[dict, dict]:
        env_idx = env_ids.tolist() if env_ids is not None else None
        raw_obs, raw_infos = self.env.reset(env_idx=env_idx)
        obs = self._wrap_obs(raw_obs)
        self._reset_metrics(env_idx)
        return obs, {}

    def step(
        self,
        actions,
        auto_reset: bool = True,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute one environment step.

        Subclass reward-computation pitfall
        ------------------------------------
        When ``auto_reset=True`` (the default), :meth:`_handle_auto_reset`
        replaces ``obs`` and ``infos`` with the *post-reset* values **before**
        this method returns.  Any reward computed from those values will reflect
        the initial state of the *next* episode, not the terminal state of the
        current one — producing systematically inflated rewards for termination
        transitions and causing the critic to over-estimate their Q-value.

        Subclasses that compute task-specific rewards have two safe options:

        1. **Override** :meth:`_compute_task_reward` (preferred for simple tasks).
           It is called here with the terminal ``obs``/``infos``, before any reset
           is performed.

        2. **Call** ``super().step(..., auto_reset=False)`` and perform the reset
           manually after computing rewards (used by ``PlaceWorkpieceEnv``
           which also needs custom termination logic in its ``step()``).
        """
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = np.asarray(actions, dtype=np.float32)
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(1, -1)

        actions_np = self._expand_actions(actions_np)

        raw_obs, rewards, terminated, truncated, infos = self.env.step(
            actions_np,
        )

        obs = self._wrap_obs(raw_obs)

        rewards_t = torch.from_numpy(rewards.astype(np.float32))
        terminated_t = torch.from_numpy(terminated)
        truncated_t = torch.from_numpy(truncated)
        dones = terminated | truncated

        # Hook: subclasses may override reward computation here, while obs/infos
        # still reflect the terminal state (before any auto-reset takes place).
        task_rewards = self._compute_task_reward(
            obs, infos, rewards_t, terminated_t, truncated_t
        )
        if task_rewards is not None:
            rewards_t = task_rewards

        self._elapsed_steps += 1
        infos = self._record_metrics(rewards_t, terminated_t, infos)

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminated_t.clone()
            terminated_t = torch.zeros_like(terminated_t)

        _do_auto_reset = auto_reset and self.auto_reset
        if dones.any() and _do_auto_reset:
            obs, infos = self._handle_auto_reset(torch.from_numpy(dones), obs, infos)

        return obs, rewards_t, terminated_t, truncated_t, infos

    def _compute_task_reward(
        self,
        obs: dict,
        infos: dict,
        sim_rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ):
        """Hook for task-specific reward computation from the terminal state.

        Called **before** auto-reset, so ``obs`` and ``infos`` still contain
        the state at the end of the current step (terminal or not).

        Return a ``torch.Tensor`` of shape ``(num_envs,)`` to replace the
        simulator's raw reward, or ``None`` to keep it unchanged.

        Subclasses that need custom termination logic in addition to custom
        rewards should instead call ``super().step(..., auto_reset=False)`` and
        handle the reset manually (see :meth:`step` docstring).
        """
        return None

    def chunk_step(self, chunk_actions) -> tuple:
        if not isinstance(chunk_actions, torch.Tensor):
            chunk_actions = torch.as_tensor(np.asarray(chunk_actions, dtype=np.float32))

        if chunk_actions.ndim == 2:
            chunk_actions = chunk_actions.unsqueeze(1)

        num_action_chunks = chunk_actions.shape[1]

        obs_list: list = []
        rewards_list: list = []
        terminated_list: list = []
        truncated_list: list = []
        infos_list: list = []

        for i in range(num_action_chunks):
            act = chunk_actions[:, i, :]
            obs, rewards, terminated, truncated, infos = self.step(
                act, auto_reset=self.auto_reset
            )
            obs_list.append(obs)
            rewards_list.append(rewards)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            infos_list.append(infos)

        chunk_rewards_t = torch.stack(rewards_list, dim=1)
        chunk_terminations_t = torch.stack(terminated_list, dim=1)
        chunk_truncations_t = torch.stack(truncated_list, dim=1)

        return (
            obs_list,
            chunk_rewards_t,
            chunk_terminations_t,
            chunk_truncations_t,
            infos_list,
        )

    def close(self):
        self.env.close()
        self._container_mgr.shutdown()

    # ---------------------------------------------------------------------- #
    # Metrics helpers
    # ---------------------------------------------------------------------- #

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            for i in env_idx if hasattr(env_idx, "__iter__") else [env_idx]:
                self._prev_step_reward[i] = 0.0
                self.success_once[i] = False
                self.fail_once[i] = False
                self.returns[i] = 0.0
                self._elapsed_steps[i] = 0
        else:
            self._prev_step_reward[:] = 0.0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(
        self, step_reward: torch.Tensor, terminations: torch.Tensor, infos: dict
    ) -> dict:
        self.returns += step_reward.numpy()
        self.success_once |= step_reward.numpy() > 0
        episode_info = {
            "success_once": torch.from_numpy(self.success_once.copy()),
            "return": torch.from_numpy(self.returns.copy()),
            "episode_len": torch.from_numpy(self._elapsed_steps.copy()),
            "reward": torch.from_numpy(
                np.where(
                    self._elapsed_steps > 0,
                    self.returns / np.maximum(self._elapsed_steps, 1),
                    0.0,
                ).astype(np.float32)
            ),
        }
        infos["episode"] = episode_info
        return infos

    def _handle_auto_reset(
        self, dones: torch.Tensor, final_obs: dict, infos: dict
    ) -> tuple[dict, dict]:
        _final_obs = copy.deepcopy(final_obs)
        _final_info = copy.deepcopy(infos)
        env_idx = torch.where(dones)[0].numpy().tolist()
        obs, new_infos = self.reset(env_ids=np.array(env_idx, dtype=np.int32))
        new_infos["final_observation"] = _final_obs
        new_infos["final_info"] = _final_info
        new_infos["_final_observation"] = dones
        new_infos["_final_info"] = dones
        new_infos["_elapsed_steps"] = dones
        return obs, new_infos

    # ---------------------------------------------------------------------- #
    # Properties
    # ---------------------------------------------------------------------- #

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool):
        self._is_start = value

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return torch.from_numpy(self._elapsed_steps.copy())

    def update_reset_state_ids(self):
        pass
