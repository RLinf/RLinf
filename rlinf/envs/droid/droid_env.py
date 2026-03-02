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
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.droid.venv import SubProcDroidEnv
from rlinf.envs.utils import to_tensor


def _create_droid_env(
    scene: int = 1,
    device: str = "cuda:0",
    num_envs: int = 1,
    task_description: str = "put the cube in the bowl",
    headless: bool = True,
    warmup_reset: bool = True,
):
    """Create DROID env and sim app in the subprocess. Called only inside the worker."""
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="DROID env for RLinf")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args([])
    args.enable_cameras = True
    args.headless = headless
    if hasattr(args, "device"):
        args.device = device
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        "DROID",
        device=device,
        num_envs=num_envs,
        use_fabric=True,
    )
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)
    if warmup_reset:
        env.reset()
        env.reset()
    return env, simulation_app


class DroidEnv(gym.Env):
    """DROID environment for RLinf (Isaac Lab / sim_evals). Runs sim in a subprocess."""

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.group_size = getattr(self.cfg, "group_size", 1)
        self.num_group = self.num_envs // self.group_size
        self.ignore_terminations = getattr(self.cfg, "ignore_terminations", False)
        self.auto_reset = getattr(self.cfg, "auto_reset", False)
        self.use_rel_reward = getattr(self.cfg, "use_rel_reward", True)
        self.video_cfg = getattr(self.cfg, "video_cfg", {})

        init_params = getattr(self.cfg, "init_params", None) or {}
        scene = init_params.get("scene", 1)
        device = init_params.get("device", "cuda:0")
        self.task_description = init_params.get(
            "task_description", "put the cube in the bowl"
        )
        headless = init_params.get("headless", True)
        warmup_reset = init_params.get("warmup_reset", True)

        def env_fn():
            return _create_droid_env(
                scene=scene,
                device=device,
                num_envs=num_envs,
                task_description=self.task_description,
                headless=headless,
                warmup_reset=warmup_reset,
            )

        self.env = SubProcDroidEnv(env_fn)
        try:
            self.device = torch.device(self.env.device())
        except Exception:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_step_reward = np.zeros(self.num_envs, dtype=np.float32)

    @property
    def total_num_group_envs(self) -> int:
        return max(1, self.num_group)

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
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)

    def _reset_metrics(self, env_idx: Optional[Union[int, np.ndarray]] = None) -> None:
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

    def _wrap_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Convert DROID raw obs to RLinf format (main_images, wrist_images, states, task_descriptions)."""
        policy = obs.get("policy", obs)
        # Images: external_cam [B,H,W,C], wrist_cam [B,H,W,C]; optional external_cam_2
        main_img = policy.get("external_cam", policy.get("table_cam"))
        wrist_img = policy.get("wrist_cam")
        if main_img is None or wrist_img is None:
            raise KeyError(
                "DROID obs['policy'] must contain 'external_cam' (or 'table_cam') and 'wrist_cam'"
            )
        if isinstance(main_img, torch.Tensor):
            if main_img.dim() == 4 and main_img.shape[1] == 1:
                main_img = main_img.squeeze(1)
            if wrist_img.dim() == 4 and wrist_img.shape[1] == 1:
                wrist_img = wrist_img.squeeze(1)
            main_images = main_img
            wrist_images = wrist_img
        else:
            main_images = torch.as_tensor(np.asarray(main_img), dtype=torch.uint8)
            wrist_images = torch.as_tensor(np.asarray(wrist_img), dtype=torch.uint8)
        # State: arm 7 + gripper 1 = 8
        arm = policy.get("arm_joint_pos", policy.get("eef_pos"))
        gripper = policy.get("gripper_pos")
        if arm is not None and gripper is not None:
            if isinstance(arm, torch.Tensor):
                states = torch.cat([arm, gripper], dim=-1)
            else:
                states = torch.from_numpy(
                    np.concatenate([np.asarray(arm), np.asarray(gripper)], axis=-1)
                ).float()
        else:
            states = torch.zeros(self.num_envs, 8, dtype=torch.float32)
        task_descriptions = [self.task_description] * self.num_envs
        extra = None
        if "external_cam_2" in policy:
            ex2 = policy["external_cam_2"]
            if isinstance(ex2, torch.Tensor):
                if ex2.dim() == 4 and ex2.shape[1] == 1:
                    ex2 = ex2.squeeze(1)
                extra = ex2.unsqueeze(1)
            else:
                ex2 = torch.as_tensor(np.asarray(ex2))
                if ex2.dim() == 3:
                    ex2 = ex2.unsqueeze(0)
                extra = ex2.unsqueeze(1)
        out = {
            "main_images": main_images,
            "wrist_images": wrist_images,
            "states": states,
            "task_descriptions": task_descriptions,
        }
        if extra is not None:
            out["extra_view_images"] = extra
        return out

    def _record_metrics(
        self, step_reward: np.ndarray, terminations: np.ndarray, infos: dict
    ) -> dict:
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_info["episode_len"], 1
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _calc_step_reward(self, terminations: np.ndarray) -> np.ndarray:
        reward = self.cfg.reward_coef * terminations.astype(np.float32)
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward
        if self.use_rel_reward:
            return reward_diff
        return reward

    def reset(
        self,
        seed: Optional[int] = None,
        env_ids: Optional[Union[int, np.ndarray]] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        seed = seed if seed is not None else self.seed
        obs, info = self.env.reset(seed=seed, env_ids=env_ids)
        extracted_obs = self._wrap_obs(obs)
        self._reset_metrics(env_idx=env_ids)
        return extracted_obs, info or {}

    def step(
        self, actions: Union[np.ndarray, torch.Tensor], auto_reset: bool = True
    ) -> tuple[dict, Any, Any, Any, dict]:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        obs, step_reward, terminations, truncations, infos = self.env.step(actions)
        self._elapsed_steps += 1
        max_steps = getattr(self.cfg, "max_episode_steps", 256)
        truncations = (self._elapsed_steps >= max_steps) | np.asarray(truncations)
        if isinstance(terminations, torch.Tensor):
            terminations = terminations.cpu().numpy()
        if isinstance(truncations, torch.Tensor):
            truncations = truncations.cpu().numpy()
        if isinstance(step_reward, torch.Tensor):
            step_reward = step_reward.cpu().numpy()
        step_reward = np.asarray(step_reward, dtype=np.float32)
        if step_reward.size == 1:
            step_reward = np.broadcast_to(step_reward, (self.num_envs,))
        extracted_obs = self._wrap_obs(obs)
        step_reward = self._calc_step_reward(terminations)
        infos = infos if isinstance(infos, dict) else {}
        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations = np.zeros_like(terminations, dtype=bool)
        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                dones, extracted_obs, infos
            )
        return (
            extracted_obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def _handle_auto_reset(
        self, dones: np.ndarray, _final_obs: dict, infos: dict
    ) -> tuple[dict, dict]:
        final_obs = copy.deepcopy(_final_obs)
        final_info = copy.deepcopy(infos)
        # DROID subprocess env typically supports full reset only
        obs, infos = self.reset(env_ids=None)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def chunk_step(
        self, chunk_actions: Union[np.ndarray, torch.Tensor]
    ) -> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)
        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)
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

    def update_reset_state_ids(self) -> None:
        pass

    def close(self) -> None:
        self.env.close()
