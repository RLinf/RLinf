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

import gym
import numpy as np
import torch
from mani_skill.utils import common
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.worldmodel.backend import WorldModelBackend
from rlinf.envs.worldmodel.dataset import LeRobotDatasetWrapper


class WorldModelEnv(gym.Env):
    """A Gym environment that wraps a world model.

    This environment is designed for interacting with a learned world model that
    simulates the dynamics of a real environment. It handles the interaction
    with the world model, manages environment resets, and calculates rewards.

    Args:
        cfg: The configuration object for the environment.
        seed_offset (int): An offset to be added to the base seed.
        total_num_processes (int): The total number of parallel processes.
        record_metrics (bool, optional): Whether to record metrics. Defaults to True.
    """

    def __init__(self, cfg, seed_offset: int, total_num_processes, record_metrics=True):
        """Initializes the WorldModelEnv.

        Args:
            cfg: The configuration object for the environment.
            seed_offset (int): An offset added to the base seed to create different
                environment instances with different seeds.
            total_num_processes (int): The total number of parallel processes,
                which can be used for distributed training.
            record_metrics (bool, optional): Whether to record and log metrics
                like success rates and returns. Defaults to True.
        """
        self.cfg = cfg

        # Load basic configuration information
        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.num_envs = cfg.num_envs
        self.group_size = cfg.group_size
        self.num_group = cfg.num_group
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.gen_num_image_each_step = cfg.gen_num_image_each_step

        dataset_cfg = OmegaConf.to_container(cfg.dataset_cfg, resolve=True)
        self.task_dataset = LeRobotDatasetWrapper(**dataset_cfg)
        self.total_num_group_envs = len(self.task_dataset)
        self.camera_names = self.task_dataset.camera_names

        self.device = "cuda"
        env_cfg = OmegaConf.to_container(cfg.backend_cfg, resolve=True)
        self.env = WorldModelBackend(env_cfg, self.task_dataset, self.device)

        self._is_start = True
        self._init_reset_state_ids()
        self._init_metrics()
        self.record_metrics = record_metrics
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = {camera_name: [] for camera_name in self.camera_names}

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
        ).to(self.device)

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
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, terminations):
        """
        (Initial implementation) Calculates the reward for the current step.

        This function computes the reward based on the `terminations` tensor and
        optionally returns a relative reward (the difference from the previous step)
        or an absolute reward.

        Args:
            terminations: A tensor indicating whether the episode has terminated.

        Returns:
            The calculated reward.
        """
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
    ):
        obs, infos = self.env.reset(seed=seed, options=options)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return obs, infos

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
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
            return (
                extracted_obs,
                torch.zeros(self.num_envs, dtype=torch.float32).to(self.device),
                terminations,
                truncations,
                infos,
            )

        extracted_obs, _reward, terminations, truncations, infos = self.env.step(
            actions
        )

        step_reward = self._calc_step_reward(terminations)

        if self.video_cfg.save_video:
            self.add_new_frames(extracted_obs, infos)

        infos = self._record_metrics(step_reward, infos)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)
        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]

        chunk_step = chunk_actions.shape[1]
        assert chunk_step % self.gen_num_image_each_step == 0, (
            "chunk_step must be divisible by gen_num_image_each_step"
        )
        chunk_size = chunk_step // self.gen_num_image_each_step
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[
                :,
                i * self.gen_num_image_each_step : (i + 1)
                * self.gen_num_image_each_step,
                :,
            ]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )

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
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, infos
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
            infos,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(infos)
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset(options=options)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def run(self):
        obs, _ = self.reset()
        self.reset(
            options={"env_idx": torch.arange(0, self.num_envs - 4, device=self.device)}
        )
        for step in range(10):
            action = self.env.action_space.sample()
            obs, rew, terminations, truncations, infos = self.step(action)
            print(
                f"Step {step}: obs={obs.keys()}, rew={rew.mean()}, \
                terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )
        self.flush_video()

    def add_new_frames(self, extracted_obs, infos):
        infos = common.to_numpy(infos)
        images = {camera_name: [] for camera_name in self.camera_names}
        for env_id in range(self.num_envs):
            for camera_name in self.camera_names:
                image = extracted_obs["images_and_states"][camera_name][
                    env_id, :
                ].permute(1, 2, 0)
                image = common.to_numpy(image)
                image = (image * 255).astype(np.uint8)
                if self.video_cfg.info_on_video:
                    info_item = {
                        k: v if np.size(v) == 1 else v[env_id] for k, v in infos.items()
                    }
                    image = put_info_on_image(image, info_item)
                images[camera_name].append(image)
        for camera_name in self.camera_names:
            full_image = tile_images(
                images[camera_name], nrows=int(np.sqrt(self.num_envs))
            )
            self.render_images[camera_name].append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        for camera_name in self.camera_names:
            images_to_video(
                self.render_images[camera_name],
                output_dir=output_dir,
                video_name=f"{self.video_cnt}_{camera_name}",
            )
        self.video_cnt += 1
        self.render_images = {camera_name: [] for camera_name in self.camera_names}
