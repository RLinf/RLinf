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

import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.robotwin.seed_utils import partition_success_seeds
from rlinf.envs.utils import center_crop_image, list_of_dict_to_dict_of_list

__all__ = ["RoboTwinEnv"]


class RoboTwinEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.base_seed = env_seed
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations

        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.use_custom_reward = cfg.use_custom_reward

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg
        self.record_metrics = record_metrics
        self._is_start = True

        self.task_name = cfg.task_config.task_name
        self.action_type = cfg.task_config.get("action_type", "qpos")

        self.center_crop = cfg.get("center_crop", False)
        self._init_reset_state_ids()

        self._init_env()

        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        if self.record_metrics:
            self._init_metrics()
            self._elapsed_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

    def _init_env(self):
        mp.set_start_method("spawn", force=True)
        os.environ["ASSETS_PATH"] = self.cfg.assets_path

        try:
            from robotwin.envs.vector_env import VectorEnv
        except ModuleNotFoundError:
            robotwin_path = self._resolve_robotwin_repo_path(self.cfg.assets_path)
            if robotwin_path not in sys.path:
                sys.path.insert(0, robotwin_path)
            from robotwin.envs.vector_env import VectorEnv

        env_seeds = self.reset_state_ids.tolist()
        task_config = OmegaConf.to_container(self.cfg.task_config, resolve=True)
        if self.action_type == "ee":
            data_type = task_config.setdefault("data_type", {})
            data_type["endpose"] = True

        self.venv = VectorEnv(
            task_config=task_config,
            n_envs=self.num_envs,
            env_seeds=env_seeds,
        )

    @staticmethod
    def _resolve_robotwin_repo_path(assets_path: str) -> str:
        explicit_path = os.environ.get("ROBOTWIN_PATH")
        candidate = Path(explicit_path or assets_path).expanduser()
        repo_marker = candidate / "robotwin" / "envs" / "vector_env.py"
        if repo_marker.exists():
            return str(candidate)
        if explicit_path:
            raise FileNotFoundError(
                "ROBOTWIN_PATH must point to a RoboTwin support repo root containing "
                f"'robotwin/envs/vector_env.py', got: {explicit_path}"
            )
        raise FileNotFoundError(
            "ROBOTWIN_PATH is required when env.assets_path is not itself a RoboTwin "
            f"support repo root. Received assets_path={assets_path!r}."
        )

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
                self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            if isinstance(infos["success"], list):
                infos["success"] = torch.as_tensor(
                    np.array(infos["success"]).reshape(-1), device=self.device
                )
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def center_and_crop(self, image, center_crop=False):
        image = np.array(image)

        image = Image.fromarray(image).convert("RGB")
        if center_crop:
            image = center_crop_image(image)
        return np.array(image)

    @staticmethod
    def _resolve_obs_image(obs: dict, key: str, camera_key: str):
        if key in obs and obs[key] is not None:
            return obs[key]
        return obs.get("observation", {}).get(camera_key, {}).get("rgb", None)

    @staticmethod
    def _resolve_obs_state(obs: dict):
        if "state" in obs and obs["state"] is not None:
            return obs["state"]
        return obs.get("joint_action", {}).get("vector", None)

    def _extract_obs_image(self, raw_obs):
        batch_images = []
        batch_wrist_images = []
        batch_states = []
        batch_instructions = []
        for obs in raw_obs:
            batch_images.append(
                self.center_and_crop(
                    self._resolve_obs_image(obs, "full_image", "head_camera"),
                    center_crop=self.center_crop,
                )
            )
            wrist_images = []
            left_wrist_image = self._resolve_obs_image(
                obs, "left_wrist_image", "left_camera"
            )
            if left_wrist_image is not None:
                wrist_images.append(
                    self.center_and_crop(left_wrist_image, center_crop=self.center_crop)
                )
            right_wrist_image = self._resolve_obs_image(
                obs, "right_wrist_image", "right_camera"
            )
            if right_wrist_image is not None:
                wrist_images.append(
                    self.center_and_crop(
                        right_wrist_image, center_crop=self.center_crop
                    )
                )
            if len(wrist_images) > 0:
                batch_wrist_images.append(
                    torch.stack([torch.from_numpy(img) for img in wrist_images])
                )
            state = self._resolve_obs_state(obs)
            if state is None:
                raise ValueError("RoboTwin observation is missing proprio state.")
            batch_states.append(state)
            batch_instructions.append(obs["instruction"])

        batch_images = torch.stack([torch.from_numpy(img) for img in batch_images])
        if len(batch_wrist_images) > 0:
            batch_wrist_images = torch.stack(batch_wrist_images)
        else:
            batch_wrist_images = None
        batch_states = torch.stack([torch.from_numpy(state) for state in batch_states])

        extracted_obs = {
            "main_images": batch_images,
            "wrist_images": batch_wrist_images,
            "states": batch_states,
            "task_descriptions": batch_instructions,
        }

        return extracted_obs

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations

        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _cal_chunk_rewards(self, step_reward, chunk_step, terminations, infos):
        n_steps_to_run = np.array(
            [[0] for i in range(self.num_envs)]
        )  # infos.get("n_steps_to_run", np.array([[0] for i in range(self.num_envs)]))

        n_steps_to_run = torch.as_tensor(
            np.array(n_steps_to_run).reshape(-1), device=self.device
        )
        chunk_rewards = torch.zeros(self.num_envs, chunk_step, device=self.device)
        for env_id in range(self.num_envs):
            steps_left = n_steps_to_run[env_id]
            reward = step_reward[env_id]
            start_idx = chunk_step - steps_left - 1

            if terminations[env_id] and start_idx > 0:
                if self.use_rel_reward:
                    chunk_rewards[env_id, start_idx] = reward
                else:
                    chunk_rewards[env_id, start_idx:] = reward

        return chunk_rewards

    def _extract_eef_poses_from_raw_obs(
        self, raw_obs: list[dict[str, dict[str, np.ndarray | float]]]
    ) -> torch.Tensor:
        eef_poses = []
        for obs in raw_obs:
            endpose = obs["endpose"]
            pose = np.concatenate(
                [
                    np.asarray(endpose["left_endpose"], dtype=np.float32),
                    np.asarray([endpose["left_gripper"]], dtype=np.float32),
                    np.asarray(endpose["right_endpose"], dtype=np.float32),
                    np.asarray([endpose["right_gripper"]], dtype=np.float32),
                ]
            )
            eef_poses.append(torch.from_numpy(pose))
        return torch.stack(eef_poses)

    def _attach_extra_obs(
        self,
        extracted_obs: dict[str, torch.Tensor | list[str] | None],
        *,
        raw_task_obs: list[dict],
        chunk_observations: list[list[dict]] | None = None,
        episode_dones: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[str] | dict[str, object] | None]:
        obs = dict(extracted_obs)
        extra_obs = dict(obs.get("extra_obs") or {})
        extra_obs["eef_poses"] = self._extract_eef_poses_from_raw_obs(raw_task_obs)
        if chunk_observations is not None:
            extra_obs["chunk_observations"] = chunk_observations
        if episode_dones is not None:
            extra_obs["episode_dones"] = episode_dones
        obs["extra_obs"] = extra_obs
        return obs

    def _collect_current_obs(self) -> tuple[dict, list[dict]]:
        raw_obs = self.venv.get_obs()
        extracted_obs = self._extract_obs_image(raw_obs)
        extracted_obs = self._attach_extra_obs(extracted_obs, raw_task_obs=raw_obs)
        return extracted_obs, raw_obs

    def reset(
        self,
        env_idx: Optional[Union[int, list[int]]] = None,
        env_seeds=None,
    ):
        if self._is_start:
            self._is_start = False

        env_seeds = self.reset_state_ids.tolist() if env_seeds is None else env_seeds

        self.venv.reset(env_idx=env_idx, env_seeds=env_seeds)
        raw_obs = self.venv.get_obs()
        infos = {}

        self._reset_metrics(env_idx)

        extracted_obs = self._extract_obs_image(raw_obs)
        if self.action_type == "ee":
            extracted_obs = self._attach_extra_obs(extracted_obs, raw_task_obs=raw_obs)

        return extracted_obs, infos

    def step(
        self, actions: Union[torch.Tensor, np.ndarray, dict] = None, auto_reset=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, dict):
            actions = actions.get("actions", actions)

        # [n_envs, horizon, action_dim]
        if len(actions.shape) == 2:
            # [n_envs, action_dim] -> [n_envs, 1, action_dim]
            actions = actions[:, None, :]

        if self.action_type == "ee":
            (
                obs_list,
                chunk_rewards,
                chunk_terminations,
                chunk_truncations,
                infos_list,
            ) = self.chunk_step(actions)
            return (
                obs_list[-1],
                chunk_rewards[:, -1],
                chunk_terminations[:, -1],
                chunk_truncations[:, -1],
                infos_list[-1],
            )

        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)

        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        self._elapsed_steps += actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)

        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.cpu().numpy()

        if self.action_type == "ee":
            return self._ee_chunk_step(chunk_actions)

        # chunk_actions: [num_envs, chunk_step, action_dim]
        num_envs = chunk_actions.shape[0]
        chunk_step = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            chunk_actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)
        obs_list.append(extracted_obs)
        infos_list.append(infos)
        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        chunk_rewards = self._cal_chunk_rewards(
            step_reward, chunk_step, terminations, infos
        )

        self._elapsed_steps += chunk_actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        past_dones = torch.logical_or(terminations, truncations)
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_terminations[:, -1] = terminations

        chunk_truncations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_truncations[:, -1] = truncations

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _ee_chunk_step(self, chunk_actions):
        num_envs, chunk_step, _ = chunk_actions.shape
        obs_list = []
        infos_list = []
        chunk_start_elapsed = self._elapsed_steps.clone()
        (
            raw_chunk_obs,
            raw_chunk_rewards,
            raw_chunk_terminations,
            raw_chunk_truncations,
            _raw_chunk_infos,
        ) = self.venv.chunk_step(chunk_actions)
        chunk_rewards = torch.as_tensor(
            np.asarray(raw_chunk_rewards, dtype=np.float32),
            device=self.device,
            dtype=torch.float32,
        )
        chunk_terminations = torch.as_tensor(
            np.asarray(raw_chunk_terminations, dtype=bool),
            device=self.device,
            dtype=torch.bool,
        )
        chunk_truncations = torch.as_tensor(
            np.asarray(raw_chunk_truncations, dtype=bool),
            device=self.device,
            dtype=torch.bool,
        )
        metrics_chunk_rewards = chunk_rewards.clone()
        repeated_success_mask = (
            chunk_terminations.to(dtype=torch.long).cumsum(dim=1) > 1
        )
        metrics_chunk_rewards[repeated_success_mask] = 0.0
        chunk_observations = [list(env_obs) for env_obs in raw_chunk_obs]

        for step_idx in range(chunk_step):
            raw_step_obs = [env_obs[step_idx] for env_obs in raw_chunk_obs]
            extracted_obs = self._extract_obs_image(raw_step_obs)
            step_success = chunk_terminations[:, step_idx].clone()
            infos = {"success": step_success.clone()}
            if self.record_metrics:
                self._elapsed_steps = chunk_start_elapsed + step_idx + 1
                infos = self._record_metrics(metrics_chunk_rewards[:, step_idx], infos)

            obs_list.append(extracted_obs)
            infos_list.append(infos)

        self._elapsed_steps = chunk_start_elapsed + chunk_step
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        terminated_within_chunk = chunk_terminations.any(dim=1)
        if truncated.any():
            chunk_truncations[:, -1] = torch.logical_or(
                truncated & ~terminated_within_chunk, chunk_truncations[:, -1]
            )

        if self.ignore_terminations:
            chunk_terminations[:] = False
            if self.record_metrics and infos_list:
                infos_list[-1]["episode"]["success_at_end"] = infos_list[-1][
                    "success"
                ].clone()

        effective_chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        episode_dones = effective_chunk_dones.any(dim=1)

        # Downstream env/rollout workers consume chunk-level episode boundaries
        # from the last chunk slot. For ee chunk stepping, success/truncation can
        # first appear before the final substep, so propagate any effective
        # within-chunk terminal signal to the final slot.
        chunk_terminations[:, -1] = chunk_terminations.any(dim=1)
        chunk_truncations[:, -1] = chunk_truncations.any(dim=1)

        if self.record_metrics and infos_list and "episode" in infos_list[-1]:
            done_mask = episode_dones
            if done_mask.any():
                done_offsets = torch.zeros(
                    num_envs, dtype=torch.long, device=self.device
                )
                done_offsets[done_mask] = (
                    effective_chunk_dones[done_mask].to(dtype=torch.long).argmax(dim=1)
                    + 1
                )
                episode_info = infos_list[-1]["episode"]
                episode_info["episode_len"][done_mask] = (
                    chunk_start_elapsed[done_mask] + done_offsets[done_mask]
                )
                episode_info["reward"] = episode_info["return"] / torch.clamp(
                    episode_info["episode_len"].float(), min=1.0
                )
        obs_list[-1] = self._attach_extra_obs(
            obs_list[-1],
            raw_task_obs=self.venv.get_obs(),
            chunk_observations=chunk_observations,
            episode_dones=episode_dones,
        )

        if episode_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                episode_dones, obs_list[-1], infos_list[-1]
            )
            obs_list[-1] = self._attach_extra_obs(
                obs_list[-1],
                raw_task_obs=self.venv.get_obs(),
                chunk_observations=chunk_observations,
                episode_dones=episode_dones,
            )

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        final_info = infos.copy()
        if self.cfg.is_eval:
            self.update_reset_state_ids(env_idx=env_idx)

        extracted_obs, infos = self.reset(env_idx=env_idx.tolist())
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def close(self, clear_cache=True):
        if hasattr(self, "venv"):
            self.venv.close(clear_cache)

    def sample_action_space(self):
        return np.random.randn(self.num_envs, self.horizon, 14)

    def _init_reset_state_ids(self):
        if self.cfg.get("seeds_path", None) is not None and os.path.exists(
            self.cfg.seeds_path
        ):
            with open(self.cfg.seeds_path, "r") as f:
                data = json.load(f)
            success_seeds = data[self.task_name].get("success_seeds", None)
            if success_seeds is not None:
                success_seeds = torch.as_tensor(success_seeds, dtype=torch.long)
                self.success_seeds = partition_success_seeds(
                    success_seeds,
                    base_seed=self.base_seed,
                    seed_offset=self.seed_offset,
                    total_num_processes=self.total_num_processes,
                    num_group=self.num_group,
                )
                self._current_seed_index = 0
            else:
                self.success_seeds = None
                self._current_seed_index = 0
        else:
            self.success_seeds = None
            self._current_seed_index = 0

        if not hasattr(self, "_generator"):
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self, env_idx=None):
        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return

        if env_idx is not None and hasattr(self, "reset_state_ids"):
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            for idx in env_idx:
                self.reset_state_ids[idx] = reset_state_ids[idx]
        else:
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            self.reset_state_ids = reset_state_ids

    def check_seeds(self, seeds):
        resutls = self.venv.check_seeds(seeds)

        return resutls
