# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from omegaconf.omegaconf import OmegaConf

__all__ = ["ManiskillEnv"]


def extract_termination_from_info(info, num_envs, device):
    if "success" in info:
        if "fail" in info:
            terminated = torch.logical_or(info["success"], info["fail"])
        else:
            terminated = info["success"].clone()
    else:
        if "fail" in info:
            terminated = info["fail"].clone()
        else:
            terminated = torch.zeros(num_envs, dtype=bool, device=device)
    return terminated


class ManiskillEnv(gym.Env):
    def __init__(self, cfg, rank, record_metrics=True):
        env_seed = cfg.seed
        self.seed = env_seed + rank
        self.rank = rank
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = cfg.num_group
        self.group_size = cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []

        self.cfg = cfg

        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        self.env: BaseEnv = gym.make(**env_args)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )  # [B, ]
        self.record_metrics = record_metrics
        self._is_start = True
        self._init_reset_state_ids()
        self.info_logging_keys = ["is_src_obj_grasped", "consecutive_grasp", "success"]
        if self.record_metrics:
            self._init_metrics()

    @property
    def total_num_group_envs(self):
        if hasattr(self.env.unwrapped, "total_num_trials"):
            return self.env.unwrapped.total_num_trials

        # Tonghe revised on 10/03/2025.
        # assert hasattr(self.env, "xyz_configs") and hasattr(self.env, "quat_configs")
        assert self.env.get_wrapper_attr("xyz_configs") is not None
        assert self.env.get_wrapper_attr("quat_configs") is not None

        # return len(self.env.xyz_configs) * len(self.env.quat_configs)
        return len(self.env.unwrapped.xyz_configs) * len(
            self.env.unwrapped.quat_configs
        )

    @property
    def num_envs(self):
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def elapsed_steps(self):
        return self.env.unwrapped.elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def instruction(self):
        return self.env.unwrapped.get_language_instruction()

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.all_reset_state_ids = torch.randperm(
            self.total_num_group_envs, generator=self._generator
        ).to(self.device)
        self.update_reset_state_ids()

    def update_reset_state_ids(self, eval_epoch=0):
        reset_state_ids = torch.randint(
            low=0,
            high=len(self.all_reset_state_ids),
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _extract_obs_image(self, raw_obs):
        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        obs_image = obs_image.permute(0, 3, 1, 2)  # [B, C, H, W]
        proprioception: torch.Tensor = self.env.unwrapped.agent.robot.get_qpos().to(
            obs_image.device
        )
        extracted_obs = {
            "images": obs_image,
            "state": proprioception,
            "task_descriptions": self.instruction,
        }
        return extracted_obs

    def _calc_step_reward(self, info):
        """
        The SimplerEnv defines info["success"] as src_on_target, but this may not be enough to test a successful pick-and-place, because the robot could throw object to place but also get info['success']==True.
        We consider src_on_target & is_src_obj_grasped as a trully successful pick&place, which implies robot gripper places, but not throws, the object on the plate.
        But we also encourage the robot to grasp continuously.
        to hack the reward.
        """
        reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.env.unwrapped.device
        )  # [B, ]   # unwrapped added by Tonghe on 10/03/2025.
        reward += info["consecutive_grasp"] * 0.1  # encourage continuous grasping.

        reward += (
            info["is_src_obj_grasped"] * 0.1
        )  # in the env, success means src_on_target. But we make it stricter, that we ask obj on plate & gripped grasps it, so that robot does not throw things on the plate.
        reward += (info["success"] & info["is_src_obj_grasped"]) * 1.0

        # diff
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if reward is None:
            from termcolor import colored

            print(colored(f"reward={reward}!!", "red"))
        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

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
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["task_id"] = torch.zeros_like(
            self.success_once, dtype=torch.int8
        ).to(self.success_once.device)
        infos["episode"] = episode_info
        return infos

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = {},
    ):
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        extracted_obs = self._extract_obs_image(raw_obs)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, infos

    def step(
        self, actions: Union[Array, Dict] = None, auto_reset=True
    ) -> Tuple[Array, Array, Array, Array, Dict]:
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
                # self.add_new_frames(infos=infos)
                self.add_new_frames_from_obs(extracted_obs, infos=infos, rewards=None)
            # start_reward=None
            start_reward = truncations = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )  # Tonghe revised on 10/03/2025 to prevent reward=None concatenate problem during chunk resets.
            return extracted_obs, start_reward, terminations, truncations, infos

        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        extracted_obs = self._extract_obs_image(raw_obs)
        step_reward = self._calc_step_reward(infos)

        if self.video_cfg.save_video:
            self.add_new_frames_from_obs(
                extracted_obs, infos=infos, rewards=step_reward
            )
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
        chunk_size = chunk_actions.shape[1]
        # print(f"Debug:: calling chunk_step: chunk_size={chunk_size}")

        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        # Open-loop execution in this action chunk.
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )  # Don't reset within chunk, but later do auto reset after chunk execution.
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

        # Reset after chunk execution.
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
            chunk_rewards,  # If there is a success within chunk, chunk_rewards >= 1.1
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = torch_clone_dict(extracted_obs)
        final_info = torch_clone_dict(infos)

        # Reset those environments that received a 'done' flag.
        reset_env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": reset_env_idx}
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[reset_env_idx])
            # In the customized simpler envs, episode_id refers to a specific task. To fix this id enhances reproduciability but hurts diversity.

        extracted_obs, infos = self.reset(options=options)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def run(self):
        obs, info = self.reset()
        for step in range(100):
            action = self.env.action_space.sample()
            obs, rew, terminations, truncations, infos = self.step(action)
            print(
                f"Step {step}: obs={obs.keys()}, rew={rew.mean()}, terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )

    # render utils
    def capture_image(self, infos=None):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]

        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=int(np.sqrt(self.num_envs)))
        return img

    def render(self, info, rew=None):
        if self.video_cfg.info_on_video:
            scalar_info = gym_utils.extract_scalars_from_info(
                common.to_numpy(info), batch_size=self.num_envs
            )
            if rew is not None:
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
            image = self.capture_image(scalar_info)
        else:
            image = self.capture_image()
        return image

    def sample_action_space(self):
        return self.env.action_space.sample()

    def add_new_frames(self, infos, rewards=None):
        image = self.render(infos, rewards)
        self.render_images.append(image)

    def add_new_frames_from_obs(self, raw_obs, infos=None, rewards=None):
        raw_imgs = common.to_numpy(raw_obs["images"].permute(0, 2, 3, 1))

        # Add statistics overlay if info_on_video is enabled and infos are provided
        if self.video_cfg.info_on_video and infos is not None:
            scalar_info = gym_utils.extract_scalars_from_info(
                common.to_numpy(infos), batch_size=self.num_envs
            )
            if rewards is not None:
                scalar_info["reward"] = common.to_numpy(rewards)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])

            # Overlay info on each image
            for i in range(len(raw_imgs)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in scalar_info.items()
                }
                raw_imgs[i] = put_info_on_image(raw_imgs[i], info_item)

        raw_full_img = tile_images(raw_imgs, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(raw_full_img)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"rank_{self.rank}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        images_to_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
            fps=self.cfg.init_params.sim_config.control_freq,
            verbose=False,
        )
        self.video_cnt += 1
        self.render_images = []
