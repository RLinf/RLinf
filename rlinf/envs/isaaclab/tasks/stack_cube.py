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

import gymnasium as gym
import torch

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv


class IsaaclabStackCubeEnv(IsaaclabBaseEnv):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )

    def _make_env_function(self):
        """
        function for make isaaclab
        """

        def make_env_isaaclab():
            import os

            # Remove DISPLAY variable to force headless mode and avoid GLX errors
            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app
            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"
            )
            isaac_env_cfg.scene.num_envs = (
                self.cfg.init_params.num_envs
            )  # default 4096 ant_env_spaces.pkl

            isaac_env_cfg.scene.wrist_cam.height = self.cfg.init_params.wrist_cam.height
            isaac_env_cfg.scene.wrist_cam.width = self.cfg.init_params.wrist_cam.width
            isaac_env_cfg.scene.table_cam.height = self.cfg.init_params.table_cam.height
            isaac_env_cfg.scene.table_cam.width = self.cfg.init_params.table_cam.width

            env = gym.make(
                self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def _wrap_obs(self, obs):
        instruction = [self.task_description] * self.num_envs
        wrist_image = obs["policy"]["wrist_cam"]
        table_image = obs["policy"]["table_cam"]
        quat = obs["policy"]["eef_quat"][
            :, [1, 2, 3, 0]
        ]  # In isaaclab, quat is wxyz not like libero
        states = torch.concatenate(
            [
                obs["policy"]["eef_pos"],
                quat2axisangle_torch(quat),
                obs["policy"]["gripper_pos"],
            ],
            dim=1,
        )

        env_obs = {
            "main_images": table_image,
            "task_descriptions": instruction,
            "states": states,
            "wrist_images": wrist_image,
        }
        return env_obs
