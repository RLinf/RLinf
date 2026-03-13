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

import os
from typing import Optional, Union
import gymnasium as gym
import numpy as np
import torch
import pdb
from rlinf.envs.world_model.utils import (
    torch_clone_dict,
    Array,
    images_to_video,
    put_info_on_image,
    tile_images,
    to_numpy,
)
from omegaconf.omegaconf import OmegaConf
from rlinf.envs.world_model.dataset import LeRobotDatasetWrapper
from rlinf.models.worldmodel.Wan import WanInference


class WanEnv(gym.Env):
    """A Gym environment that wraps a world model for reinforcement learning.

    This environment provides a Gym-compatible interface for interacting with a learned
    world model that simulates the dynamics of a real environment. It handles environment
    resets, step execution, reward calculation, and metrics tracking while supporting
    both single-step and chunked action execution.

    Key features:
    - Supports fixed or random reset state selection
    - Handles relative or absolute reward computation
    - Provides video recording capabilities
    - Tracks episode success/failure metrics
    - Supports chunked action execution for efficiency
    - Auto-reset functionality for continuous episodes

    Args:
        cfg: The configuration object containing environment, dataset, and video settings.
        seed_offset (int): An offset added to the base seed for creating different
            environment instances with reproducible randomness.
        total_num_processes (int): The total number of parallel processes for distributed training.
        record_metrics (bool, optional): Whether to track and log episode metrics like
            success rates and returns. Defaults to True.
    """

    def __init__(
        self, cfg, num_envs, seed_offset: int, total_num_processes, record_metrics=True
    ):
        """Initializes the WorldModelEnv with configuration and setup parameters.

        This method sets up the world model environment by:
        - Initializing configuration parameters from the config object
        - Setting up the dataset wrapper for state management
        - Creating the world model backend for simulation
        - Initializing metrics tracking and video recording capabilities
        - Setting up reset state ID generation for episode initialization

        Args:
            cfg: The configuration object containing all environment settings including
                dataset configuration, backend settings, and video recording options.
            seed_offset (int): An offset added to the base seed to ensure different
                environment instances have different random seeds for reproducibility.
            total_num_processes (int): The total number of parallel processes for
                distributed training scenarios.
            record_metrics (bool, optional): Whether to track episode-level metrics
                such as success rates, failure rates, and returns. Defaults to True.
        """

        self.cfg = cfg

        # Load basic configuration information
        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.num_envs = num_envs
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        dataset_cfg = OmegaConf.to_container(cfg.dataset_cfg, resolve=True)
        self.task_dataset = LeRobotDatasetWrapper(**dataset_cfg)
        self.total_num_group_envs = len(self.task_dataset)
        self.camera_names = self.task_dataset.camera_names
        if hasattr(torch, "musa") and torch.musa.is_available():
            device = "musa"
        elif torch.cuda.is_available() and hasattr(torch._C, "_cuda_setDevice"):
            device = "cuda"
        else:
            device = "cpu"
        self.device = device
        self.env = WanInference(
            cfg.backend_cfg, self.num_envs,self.task_dataset, self.device)

        self._is_start = True
        self._init_reset_state_ids()
        self._init_metrics()
        self.record_metrics = record_metrics
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = {camera_name: []
                              for camera_name in self.camera_names}

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    def _init_reset_state_ids(self):
        """Initializes the random generator for reset state ID selection."""
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        """Updates the reset state IDs for environment initialization."""
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _init_metrics(self):
        """Initializes episode tracking metrics for all environments."""
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _select_latest_obs(self, obs):
        """Selects the latest observation from a list or returns the observation directly."""
        if isinstance(obs, list):
            assert len(obs) >= 1, "obs length must bigger than 0."
            return obs[-1]
        return obs

    def _reset_metrics(self, env_idx=None):
        """Resets episode metrics for specified environments or all environments.

        Args:
            env_idx: Optional tensor or list of environment indices to reset.
                If None, resets metrics for all environments.
        """

        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        """Records episode metrics including success, failure, and return information."""
        if not self.record_metrics:
            return infos

        episode_info = {}
        self.returns += step_reward

        if isinstance(terminations, torch.Tensor):
            self.success_once = self.success_once | terminations
        else:
            terminations_tensor = torch.tensor(
                terminations, device=self.device, dtype=torch.bool
            )
            self.success_once = self.success_once | terminations_tensor

        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = torch.full(
            (self.num_envs,),
            self.elapsed_steps,
            dtype=torch.float32,
            device=self.device,
        )
        episode_info["reward"] = episode_info["return"] / \
            episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _build_step_info(self, infos, terminations, step_rewards):
        if isinstance(infos, dict):
            info_dict = infos
        else:
            info_dict = {}

            def _extract_last(value):
                if isinstance(value, torch.Tensor):
                    if value.ndim >= 1 and value.shape[0] > 0:
                        return value[-1]
                    return value
                if isinstance(value, (list, tuple, np.ndarray)):
                    return value[-1] if len(value) > 0 else value
                return value

            if isinstance(infos, list) and len(infos) > 0:
                keys = set().union(*(info.keys() for info in infos))
                for key in keys:
                    values = []
                    for info in infos:
                        values.append(_extract_last(info.get(key)))
                    try:
                        info_dict[key] = torch.tensor(
                            values, device=self.device)
                    except Exception:
                        info_dict[key] = values

        step_reward_sum = (
            torch.stack(step_rewards, dim=0).sum(dim=0)
            if isinstance(step_rewards, list)
            else step_rewards
        )
        last_terminations = (
            terminations[-1] if isinstance(terminations,
                                           list) else terminations
        )
        return self._record_metrics(step_reward_sum, last_terminations, info_dict)

    def _calc_step_reward(self, rewards, terminations):
        """(Initial implementation) Calculates step rewards based on termination states and configuration.

        This method computes rewards for each environment based on termination signals
        and the configured reward calculation mode. It supports both relative rewards
        (difference from previous step) and absolute rewards.

        Args:
            rewards: Raw reward values from the world model backend
            terminations: Boolean tensor indicating which environments have terminated

        Returns:
            list: A list of calculated rewards for each environment, where each reward
                is either the absolute reward or the difference from the previous step
                based on the use_rel_reward configuration.
        """

        return_rewards = []
        for i, termination in enumerate(terminations):
            reward = rewards[i] * self.cfg.reward_coef * termination
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward

            if self.use_rel_reward:
                return_rewards.append(reward_diff)
            else:
                return_rewards.append(reward)
        return return_rewards

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
    ):
        """Resets the environment to initial states and returns observations.

        Args:
            seed: Optional seed or list of seeds for deterministic resets.
                If provided, ensures reproducible environment initialization.
            options: Optional dictionary containing reset options.
                Can include 'env_idx' to reset specific environments,
                or 'episode_id' for fixed reset state selection.

        Returns:
            tuple: A tuple containing:
                - obs: The initial observations from the reset environment
                - info: Additional information about the reset state
        """

        obs, info = self.env.reset(seed=seed, options=options)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return obs[-1], info

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
        """Executes one environment step with the given actions.

        Args:
            actions: The actions to execute in the environment. Can be None only
                for the first step after reset, where a reset operation is performed.
            auto_reset: Whether to automatically reset environments that reach
                termination or truncation states. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - obs: The observations after executing the actions
                - rewards: The calculated rewards for this step
                - terminations: Boolean tensor indicating episode terminations
                - truncations: Boolean tensor indicating episode truncations
                - info: Dictionary containing additional information including
                    episode metrics and final observations for auto-reset episodes

        Note:
            For the initial step (is_start=True), this method performs a reset
            operation and returns zero rewards with no terminations/truncations.
        """

        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            extracted_obs, infos = self.reset(
                seed=self.seed,
                options={"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {},
            )
            self._is_start = False
            terminations = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            truncations = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            if self.video_cfg.save_video:
                self.add_new_frames(extracted_obs, infos)

        new_obs, rewards, terminations, truncations, infos = self.env.step(
            actions)
        step_rewards = self._calc_step_reward(rewards, terminations)
        self._elapsed_steps = getattr(
            self, "_elapsed_steps", 0) + len(step_rewards)
        info = self._build_step_info(infos, terminations, step_rewards)
        # print("info",len(step_rewards),len(infos))
        # infos = self._record_metrics(step_rewards, infos)

        # if self.ignore_terminations:
        #     for termination, info in zip(terminations, infos):
        #         termination[:] = False
        #         if self.record_metrics:
        #             if "success" in info:
        #                 info["episode"]["success_at_end"] = info["success"].clone()
        #             if "fail" in info:
        #                 info["episode"]["fail_at_end"] = info["fail"].clone()

        # only consider the last step output
        dones = torch.logical_or(terminations[-1], truncations[-1])
        _auto_reset = auto_reset and self.auto_reset

        if dones.any() and _auto_reset:
            new_obs[-1], info = self._handle_auto_reset(
                dones, new_obs[-1], info)

        if self.video_cfg.save_video:
            self.add_new_frames(new_obs, infos)

        return (
            self._select_latest_obs(new_obs),
            step_rewards,
            terminations,
            truncations,
            info,
        )

    def chunk_step(self, chunk_actions):
        """Executes multiple environment steps with chunked actions for efficiency.

        Args:
            chunk_actions: A tensor of shape [num_envs, chunk_step, action_dim] containing
                the sequence of actions to execute for each environment.

        Returns:
            tuple: A tuple containing:
                - extracted_obs: The observations after executing all chunked actions
                - chunk_rewards: Tensor of rewards for each step in the chunk [num_envs, chunk_steps]
                - chunk_terminations: Boolean tensor indicating terminations [num_envs, chunk_steps]
                - chunk_truncations: Boolean tensor indicating truncations [num_envs, chunk_steps]
                - info: Dictionary containing information about the final state

        Note:
            The chunk_step must be divisible by gen_num_image_each_step for proper processing.
            Terminations and truncations are only marked at the end of the chunk sequence.
        """
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []

        extracted_obs, chunk_rewards, raw_chunk_terminations, raw_chunk_truncations, info = self.step(
            chunk_actions, auto_reset=False
        )

        # [num_envs, chunk_steps]
        chunk_rewards = torch.stack(chunk_rewards, dim=1)
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
            extracted_obs, info = self._handle_auto_reset(
                past_dones, extracted_obs, info
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            info,
        )

    def _handle_auto_reset(self, dones, extracted_obs, info):
        """Handles automatic reset of environments that have reached terminal states.

        Args:
            dones: Boolean tensor indicating which environments need reset
            extracted_obs: Current observations before reset
            info: Current information dictionary before reset

        Returns:
            tuple: Updated observations and info after reset, with final state information preserved
        """

        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(info)
        # Update episode IDs to get different episodes on each reset
        self.update_reset_state_ids()
        print('episode_id',self.reset_state_ids)
        options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, info = self.reset(options=options)
        info["final_observation"] = final_obs
        info["final_info"] = final_info
        info["_final_info"] = dones
        info["_final_observation"] = dones
        info["_elapsed_steps"] = dones
        return extracted_obs, info

    def run(self, options: None):
        """Runs a test execution of the environment with random actions.

        Note:
            This is primarily used for testing and debugging the environment setup.
        """
        if options is None:
            obs, _ = self.reset()
        else:
            self.env.reset(options=options)
        self.reset(
            options={"env_idx": torch.arange(
                0, self.num_envs - 4, device=self.device)}
        )
        self.step()
        for step in range(10):
            base = (
                torch.tensor(self.env.action_space.sample(),
                             dtype=torch.float32)
                .to(self.device)
                .unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
            actions = base.unsqueeze(1).repeat(1, 15, 1)
            obs, reward, terminations, truncations, info = self.chunk_step(
                actions.cpu().numpy()
            )
            print(
                f"Step {step}: obs={obs.keys()}, reward={reward.mean()}, \
terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )
        self.flush_video()

    def add_new_frames(self, extracted_obs, infos):
        """Adds new frames to the video recording buffer.

        Args:
            extracted_obs: Dictionary containing observations with image data for each camera
            infos: Information dictionary that may contain metadata to overlay on frames

        Note:
            Frames are processed for each camera view and tiled into a grid layout
            for efficient video recording across multiple parallel environments.
        """
        is_info_on_video = self.video_cfg.info_on_video
        if isinstance(infos, dict) and infos == {}:
            is_info_on_video = False
        else:
            infos = to_numpy(infos)

        if isinstance(extracted_obs, dict):
            extracted_obs = [extracted_obs]

        for frame_idx, frame_obs in enumerate(extracted_obs):
            images = {camera_name: [] for camera_name in self.camera_names}
            for env_id in range(self.num_envs):
                for camera_name in self.camera_names:
                    image = frame_obs["images_and_states"][camera_name][env_id, :]
                    image = to_numpy(image).astype(np.uint8)
                    if is_info_on_video:
                        info = infos[frame_idx]
                        info_item = {
                            k: v if np.size(v) == 1 else "" for k, v in info.items()
                        }
                        image = put_info_on_image(image, info_item)
                    images[camera_name].append(image)
            for camera_name in self.camera_names:
                full_image = tile_images(
                    images[camera_name], nrows=int(np.sqrt(self.num_envs))
                )
                self.render_images[camera_name].append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Generates and saves video files from accumulated frames.

        Args:
            video_sub_dir: Optional subdirectory name within the base video directory.
                If provided, videos will be saved in a subdirectory of the seed-specific folder.

        Note:
            Videos are saved with unique names based on a counter and camera name.
            The frame buffer is cleared after successful video generation.
        """

        for camera_name in self.camera_names:
            print(
                f"total {len(self.render_images[camera_name])} frames for {camera_name}"
            )

        output_dir = os.path.join(
            self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        for camera_name in self.camera_names:
            images_to_video(
                self.render_images[camera_name],
                output_dir=output_dir,
                video_name=f"{self.video_cnt}_{camera_name}",
            )
        self.video_cnt += 1
        self.render_images = {camera_name: []
                              for camera_name in self.camera_names}


if __name__ == '__main__':
    test_config = "tests/e2e_tests/embodied/wan.yaml"
    cfg = OmegaConf.load(test_config)
    cfg_env = cfg.env.train
    env = WanEnv(cfg_env, cfg_env.num_envs,
                 seed_offset=0, total_num_processes=1)
    print(env.video_cfg.video_base_dir, f"seed_{env.seed}")
    action = torch.tensor([-5.1146e-11, -1.2105e-10, -9.3151e-11,  2.5257e-07, -9.5845e-08,
                           -5.6846e-08,  1.0000e+00])
    actions = action.unsqueeze(0).unsqueeze(0).repeat(cfg_env.num_envs, 16, 1)
    extracted_obs, infos = env.reset()
    env_chunk = env.chunk_step(actions)
    env_chunk = env.chunk_step(actions)
    env.flush_video()
