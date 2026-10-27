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
import random
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    to_tensor,
    valid_action_mask_from_counts,
)
from rlinf.envs.venv import SubprocVectorEnv

# DexJoCo keeps its prompts in the repository's eval configs, outside the package.
TASK_DESCRIPTIONS = {
    "water_plant": "Grasp the watering can and apply water to the plant.",
    "pick_bucket": "Place the boxed food into the bucket and then lift the bucket.",
    "fold_glasses": "Fold the glasses and place them into the case.",
    "click_mouse": (
        "Move the mouse to the purple mouse pad and click the left mouse button."
    ),
    "hammer_nail": "Use the hammer to drive the nail into the wooden board.",
    "pinch_tongs": (
        "Grasp the tongs and perform three consecutive open-close motions."
    ),
    "bimanual_assembly": (
        "Grasp the tray with the left hand and the peg with the right hand, "
        "then insert the peg into the hole."
    ),
    "bimanual_hanoi": (
        "Execute the final two moves of the three-level Tower of Hanoi: move "
        "the medium disk from the middle peg to the right peg with the right "
        "hand, then move the small disk from the left peg to the right peg with "
        "the left hand."
    ),
    "bimanual_microwave_cook": (
        "Open the microwave door, place the food inside the microwave, close "
        "the door, and press the start button."
    ),
    "bimanual_photograph": (
        "Grasp the camera with the left hand, align it with the logo, and press "
        "the shutter button with the right hand."
    ),
    "bimanual_unlock_ipad": (
        "Grasp the iPad and enter the password 123 to unlock the device."
    ),
}

# DexJoCo ends each task after this many steps, reporting it as a termination.
TASK_STEP_LIMITS = {
    "water_plant": 1000,
    "pick_bucket": 1200,
    "fold_glasses": 1200,
    "click_mouse": 1200,
    "hammer_nail": 1000,
    "pinch_tongs": 1000,
    "bimanual_assembly": 1500,
    "bimanual_hanoi": 1500,
    "bimanual_microwave_cook": 1100,
    "bimanual_photograph": 1000,
    "bimanual_unlock_ipad": 1200,
}


class DexJoCoTaskEnv(gym.Wrapper):
    """Run one installed DexJoCo task with seeded resets and named RGB cameras.

    Observations hold ``state`` (the 23- or 46-dim robot proprioception) and one
    image per name in ``camera_names``. Actions use DexJoCo's policy layout.
    """

    def __init__(
        self,
        task_name: str,
        camera_names: list[str],
        camera_heights: int = 640,
        camera_widths: int = 640,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        **task_options,
    ) -> None:
        from dexjoco.tasks import CONFIG_MAPPING

        if task_name == "bimanual_hanoi":
            # Hanoi otherwise resets itself a few wall-clock seconds after an
            # illegal move; RLinf resets its envs itself.
            task_options.setdefault("illegal_reset_delay", float("inf"))
        env = CONFIG_MAPPING[task_name]().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=randomize,
            randomize_dynamics=randomize_dynamics,
            **task_options,
        )
        super().__init__(env)
        # DexJoCo sleeps in every step to pace teleoperation at 30 Hz.
        env.unwrapped.hz = float("inf")
        self._step_limit = TASK_STEP_LIMITS[task_name]
        self._state_dim = 46 if task_name.startswith("bimanual_") else 23
        # With randomize, DexJoCo renames the moved base camera to random_camera.
        self._image_keys = {
            name: "random_camera" if randomize and name in ("front", "ego") else name
            for name in camera_names
        }
        self._image_size = (camera_widths, camera_heights)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            # DexJoCo samples initial states from the global RNGs.
            random.seed(seed)
            np.random.seed(seed)
        obs, _ = self.env.reset()
        return self._observation(obs), {"success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, _, info = self.env.step(action)
        success = bool(info["succeed"])
        truncated = self.env.unwrapped.env_step >= self._step_limit
        return (
            self._observation(obs),
            reward,
            success or (terminated and not truncated),
            truncated,
            {"success": success},
        )

    def _observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        images = {}
        for name, key in self._image_keys.items():
            image = obs[key]
            if image.shape[1::-1] != self._image_size:
                image = cv2.resize(
                    image, self._image_size, interpolation=cv2.INTER_AREA
                )
            images[name] = image
        return {"state": obs["state"][: self._state_dim], **images}


class DexJoCoEnv(gym.Env):
    """Run DexJoCo tasks with batched observations and quaternion-pose actions."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        if cfg.task_name not in TASK_DESCRIPTIONS:
            raise ValueError(
                f"Unknown DexJoCo task {cfg.task_name!r}; expected one of "
                f"{list(TASK_DESCRIPTIONS)}."
            )
        self.dual_arm = cfg.task_name.startswith("bimanual_")
        self.task_description = TASK_DESCRIPTIONS[cfg.task_name]
        self._generator = np.random.default_rng(seed=self.seed)
        self.update_reset_state_ids()
        self._init_env()
        self.current_raw_obs = [None] * num_envs
        self.current_info_list = [None] * num_envs
        self._elapsed_steps = np.zeros(num_envs, dtype=np.int32)
        self._terminated = np.zeros(num_envs, dtype=bool)
        self._truncated = np.zeros(num_envs, dtype=bool)
        self._success = np.zeros(num_envs, dtype=bool)
        self._init_metrics()
        self.prev_step_reward = np.zeros(num_envs, dtype=np.float32)
        self.use_rel_reward = cfg.use_rel_reward

        self.video_cfg = cfg.video_cfg

    def _init_env(self) -> None:
        env_fns = self.get_env_fns()
        self.env = SubprocVectorEnv(env_fns)

    def get_env_fns(self):
        """Create environment factory functions for each parallel environment."""
        init_params = OmegaConf.to_container(self.cfg.init_params, resolve=True)
        init_params["camera_names"] = list(self.cfg.camera_names)

        def env_fn(task_name=self.cfg.task_name, params=init_params):
            return DexJoCoTaskEnv(task_name, **params)

        return [env_fn for _ in range(self.num_envs)]

    @property
    def elapsed_steps(self) -> np.ndarray:
        """Number of executed actions in each current episode."""
        return self._elapsed_steps

    @property
    def info_logging_keys(self) -> list[str]:
        return []

    @property
    def is_start(self) -> bool:
        """Whether the environment worker must reset before its first rollout."""
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    def _init_metrics(self) -> None:
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)

    def _reset_metrics(self, env_idx: np.ndarray | None = None) -> None:
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

    def _record_metrics(
        self, step_reward: np.ndarray, success: np.ndarray, infos: dict[str, Any]
    ) -> dict[str, Any]:
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | success
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_info["episode_len"], 1
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def update_reset_state_ids(self) -> None:
        """Choose one seed per rollout group, shared by its environments."""
        reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _get_random_reset_state_ids(self, num_reset_states: int) -> np.ndarray:
        return self._generator.integers(low=0, high=2**31, size=(num_reset_states,))

    def _wrap_obs(self, obs_list: list[dict[str, Any]]) -> dict[str, Any]:
        def images(camera: str) -> torch.Tensor:
            return torch.from_numpy(np.stack([obs[camera] for obs in obs_list]))

        obs = {
            "main_images": images(self.cfg.camera_names[0]),
            "states": torch.from_numpy(
                np.stack([obs["state"] for obs in obs_list])
            ).float(),
            "task_descriptions": [self.task_description] * self.num_envs,
        }
        if len(self.cfg.camera_names) > 1:
            wrists = [images(camera) for camera in self.cfg.camera_names[1:]]
            obs["wrist_images"] = (
                torch.stack(wrists, dim=1) if self.dual_arm else wrists[0]
            )
        return obs

    def reset(
        self,
        env_idx: int | list[int] | np.ndarray | None = None,
        reset_state_ids: np.ndarray | None = None,
    ) -> tuple[dict, dict]:
        """Reset selected environments and retain the other observations."""
        env_idx = (
            np.arange(self.num_envs) if env_idx is None else np.atleast_1d(env_idx)
        )
        if self.is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False
        if reset_state_ids is None:
            reset_state_ids = self._get_random_reset_state_ids(len(env_idx))
        for idx, seed in zip(env_idx, reset_state_ids):
            raw_obs, _ = self.env.reset(id=[idx], seed=int(seed))
            self.current_raw_obs[idx] = raw_obs[0]
        self._terminated[env_idx] = False
        self._truncated[env_idx] = False
        self._success[env_idx] = False
        self._reset_metrics(env_idx)
        return self._wrap_obs(self.current_raw_obs), {}

    def step(
        self, actions: np.ndarray | torch.Tensor, auto_reset: bool = True
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step active environments, retaining terminal states until reset."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        active = np.flatnonzero(~(self._terminated | self._truncated))
        step_reward = np.zeros(self.num_envs, dtype=np.float32)
        if len(active):
            raw_obs, reward, terminated, truncated, info_lists = self.env.step(
                actions[active], id=active
            )
            for idx, obs, info in zip(active, raw_obs, info_lists):
                self.current_raw_obs[idx] = obs
                self.current_info_list[idx] = info
            self._elapsed_steps[active] += 1
            self._terminated[active] = terminated
            self._truncated[active] = truncated | (
                self._elapsed_steps[active] >= self.cfg.max_episode_steps
            )
            self._success[active] = [info["success"] for info in info_lists]
            step_reward[active] = self._calc_step_reward(
                reward.astype(np.float32), active
            )
        infos = list_of_dict_to_dict_of_list(self.current_info_list)
        infos = self._record_metrics(step_reward, self._success, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(self._success)
            self._terminated[:] = False
        obs = self._wrap_obs(self.current_raw_obs)
        terminations = self._terminated.copy()
        truncations = self._truncated.copy()
        dones = terminations | truncations
        if auto_reset and self.auto_reset and dones.any():
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(
        self, chunk_actions: np.ndarray | torch.Tensor
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Execute a chunk, padding each episode after its last valid action."""
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        chunk_actions = np.asarray(chunk_actions)
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        counts = np.zeros(self.num_envs, dtype=np.int64)
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            counts += ~(self._terminated | self._truncated)
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
        if not self.cfg.is_eval:
            infos_list[-1]["valid_action_mask"] = valid_action_mask_from_counts(
                counts, chunk_size
            )
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(
        self, dones: np.ndarray, _final_obs: dict[str, Any], infos: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
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
        return obs, infos

    def _calc_step_reward(self, reward: np.ndarray, env_idx: np.ndarray) -> np.ndarray:
        reward = self.cfg.reward_coef * reward
        reward_diff = reward - self.prev_step_reward[env_idx]
        self.prev_step_reward[env_idx] = reward
        if self.use_rel_reward:
            return reward_diff
        return reward

    def close(self) -> None:
        """Release the simulator subprocesses and their rendering resources."""
        if not self.env.is_closed:
            self.env.close()
