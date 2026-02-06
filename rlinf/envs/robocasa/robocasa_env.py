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

import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401 Robocasa must be imported to register envs
import torch
from omegaconf import OmegaConf

from rlinf.envs.robocasa.venv import RobocasaSubprocEnv
from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    to_tensor,
)


class RobocasaEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.get("use_fixed_reset_state_ids", False)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)

        # Get task list from config
        # Convert OmegaConf ListConfig to standard Python list
        task_names_raw = OmegaConf.to_container(cfg.task_names, resolve=True)
        self.task_names = (
            task_names_raw if isinstance(task_names_raw, list) else [task_names_raw]
        )
        self.num_tasks = len(self.task_names)

        # Task descriptions
        self.task_descriptions_all = self._load_task_descriptions()

        # Initialize reset state IDs for group_size repetition
        # Each unique scenario (num_group) will be repeated group_size times
        self._init_reset_state_ids()

        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg

    def _load_task_descriptions(self):
        """Load task descriptions for robocasa tasks."""
        # Map task names to natural language descriptions for all 24 atomic tasks
        # (excluding NavigateKitchen)
        task_desc_map = {
            # Door tasks
            "OpenSingleDoor": "open cabinet or microwave door",
            "CloseSingleDoor": "close cabinet or microwave door",
            "OpenDoubleDoor": "open double cabinet doors",
            "CloseDoubleDoor": "close double cabinet doors",
            "OpenDrawer": "open drawer",
            "CloseDrawer": "close drawer",
            # Pick and place tasks
            "PnPCounterToCab": "pick and place from counter to cabinet",
            "PnPCabToCounter": "pick and place from cabinet to counter",
            "PnPCounterToSink": "pick and place from counter to sink",
            "PnPSinkToCounter": "pick and place from sink to counter",
            "PnPCounterToStove": "pick and place from counter to stove",
            "PnPStoveToCounter": "pick and place from stove to counter",
            "PnPCounterToMicrowave": "pick and place from counter to microwave",
            "PnPMicrowaveToCounter": "pick and place from microwave to counter",
            # Appliance control tasks
            "TurnOnMicrowave": "turn on microwave",
            "TurnOffMicrowave": "turn off microwave",
            "TurnOnSinkFaucet": "turn on sink faucet",
            "TurnOffSinkFaucet": "turn off sink faucet",
            "TurnSinkSpout": "turn sink spout",
            "TurnOnStove": "turn on stove",
            "TurnOffStove": "turn off stove",
            # Coffee tasks
            "CoffeeSetupMug": "setup mug for coffee",
            "CoffeeServeMug": "serve coffee into mug",
            "CoffeePressButton": "press coffee machine button",
        }
        return [task_desc_map.get(task, task) for task in self.task_names]

    def _init_reset_state_ids(self):
        """Initialize reset state IDs - simplified version.

        For robocasa, we don't use dynamic reset_state_ids because:
        1. Robocasa doesn't support changing scenes via reset options
        2. Each environment is created with a fixed seed

        We simply assign each parallel environment a unique, fixed seed.
        """
        # Assign sequential seeds to each environment: seed_offset*num_envs + [0, 1, 2, ...]
        base_seed = self.seed
        self.env_seeds = [base_seed + i for i in range(self.num_envs)]

    def update_reset_state_ids(self):
        """Update reset state IDs for the next rollout.

        For robocasa, we use fixed seeds, so this is a no-op.
        """
        pass

    def _init_env(self):
        """Initialize robocasa environments using subprocess isolation."""
        self.task_ids = []

        # Determine task IDs for each environment
        for env_id in range(self.num_envs):
            task_idx = env_id % self.num_tasks
            self.task_ids.append(task_idx)
        self.task_ids = np.array(self.task_ids)

        # Create environment factory functions for subprocess isolation
        env_fns = self.get_env_fns()

        # Use subprocess vector environment to avoid OpenGL context sharing
        self.env = RobocasaSubprocEnv(env_fns)

    def get_env_fns(self):
        """Create environment factory functions for each parallel environment."""
        env_fns = []

        for env_id in range(self.num_envs):
            task_idx = self.task_ids[env_id]
            task_name = self.task_names[task_idx]
            env_seed = self.env_seeds[env_id]

            # Convert OmegaConf configs to standard Python types
            camera_names = OmegaConf.to_container(self.cfg.camera_names, resolve=True)
            camera_widths = self.cfg.init_params.camera_widths
            camera_heights = self.cfg.init_params.camera_heights
            robot_name = self.cfg.robot_name

            def env_fn(
                task=task_name,
                seed=env_seed,
                cameras=camera_names,
                width=camera_widths,
                height=camera_heights,
                robot=robot_name,
            ):
                """Factory function to create a robosuite environment in subprocess."""
                import robosuite
                from robosuite.controllers import load_composite_controller_config

                controller_config = load_composite_controller_config(
                    controller=None,
                    robot=robot,
                )

                env = robosuite.make(
                    env_name=task,
                    robots=robot,
                    controller_configs=controller_config,
                    camera_names=cameras,
                    camera_widths=width,
                    camera_heights=height,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    ignore_done=True,
                    use_object_obs=True,
                    use_camera_obs=True,
                    camera_depths=False,
                    seed=seed,
                    translucent_robot=False,
                    render_camera="robot0_agentview_center",  # Use same camera as observation
                )
                return env

            env_fns.append(env_fn)

        return env_fns

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

    def _record_metrics(self, step_reward, terminations, infos):
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

    def _extract_image_and_state(self, obs):
        """Extract images and states from robocasa observations.

        Pi0 expects:
        - Two 128x128 images: robot0_agentview_left_image, robot0_eye_in_hand_image
        - 16D state matching training data (padded to 32D internally by Pi0)

        Based on dataset analysis and norm_stats.json, Pi0 expects 16D state:
        [0:2]   robot0_base_pos (x, y) - 2D
        [2:5]   zeros (padding, base z is constant) - 3D
        [5:9]   robot0_base_to_eef_quat - 4D
        [9:12]  robot0_base_to_eef_pos - 3D
        [12:14] robot0_gripper_qvel - 2D (gripper velocities)
        [14:16] robot0_gripper_qpos - 2D (gripper positions)
        """
        base_images = []
        wrist_images = []
        states = []

        for env_id in range(len(obs)):
            # Get camera images
            base_img = obs[env_id].get("robot0_agentview_left_image")
            wrist_img = obs[env_id].get("robot0_eye_in_hand_image")

            # Flip images vertically (OpenGL coordinates are upside down)
            if base_img is not None:
                base_img = base_img[::-1]
            if wrist_img is not None:
                wrist_img = wrist_img[::-1]

            base_images.append(base_img)
            wrist_images.append(wrist_img)

            # Construct 16D state matching Pi0's training format
            state_16d = np.zeros(16, dtype=np.float32)

            if "robot0_base_pos" in obs[env_id]:
                base_pos = obs[env_id]["robot0_base_pos"]  # 3D
                base_to_eef_pos = obs[env_id]["robot0_base_to_eef_pos"]  # 3D
                base_to_eef_quat = obs[env_id]["robot0_base_to_eef_quat"]  # 4D
                gripper_qpos = obs[env_id]["robot0_gripper_qpos"]  # 2D
                gripper_qvel = obs[env_id]["robot0_gripper_qvel"]  # 2D

                # Map to Pi0's expected format (inferred from dataset analysis):
                state_16d[0:2] = base_pos[0:2]  # base x, y (z is constant)
                # [2:5] remain zeros (padding)
                state_16d[5:9] = (
                    base_to_eef_quat  # end-effector quaternion relative to base
                )
                state_16d[9:12] = (
                    base_to_eef_pos  # end-effector position relative to base
                )
                state_16d[12:14] = gripper_qvel  # gripper joint velocities ✅ NEW!
                state_16d[14:16] = gripper_qpos  # gripper joint positions

            states.append(state_16d)

        return {
            "base_image": np.array(base_images),
            "wrist_image": np.array(wrist_images),
            "state": np.array(states),
        }

    def _wrap_obs(self, obs_list):
        extracted = self._extract_image_and_state(obs_list)

        images_and_states_list = []
        for idx in range(self.num_envs):
            images_and_states = {
                "base_image": extracted["base_image"][idx],
                "wrist_image": extracted["wrist_image"][idx],
                "state": extracted["state"][idx],
            }
            images_and_states_list.append(images_and_states)

        images_and_states_tensor = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        # Convert images from [H, W, C] -> [B, H, W, C]
        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states_tensor["base_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states_tensor["wrist_image"]]
        )

        states = images_and_states_tensor["state"]

        # Flatten structure to match libero format
        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": [
                self.task_descriptions_all[task_id] for task_id in self.task_ids
            ],
        }
        return obs

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        options: Optional[dict] = {},
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if isinstance(env_idx, int):
            env_idx = [env_idx]

        # Reset using vectorized environment (subprocess isolation avoids OpenGL issues)
        # Use libero's SubprocVectorEnv reset interface
        raw_obs = self.env.reset(id=env_idx)

        obs = self._wrap_obs(raw_obs)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            # Initial reset at the start of evaluation
            obs, infos = self.reset()
            self._is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            rewards = np.zeros(self.num_envs, dtype=np.float32)

            return (
                obs,
                to_tensor(rewards),
                to_tensor(terminations),
                to_tensor(truncations),
                infos,
            )

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1

        # Use vectorized environment step (subprocess isolation avoids OpenGL issues)
        # Robosuite returns 4 values: (obs, reward, done, info)
        raw_obs, rewards, dones, info_lists = self.env.step(actions)
        infos = list_of_dict_to_dict_of_list(info_lists)

        # Extract success from infos
        terminations = np.array(
            [info.get("success", False) for info in info_lists]
        ).astype(bool)
        truncations = self._elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(terminations)

        infos = list_of_dict_to_dict_of_list(info_lists)
        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

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
        obs, infos = self.reset(env_idx=env_idx)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def close(self):
        """Close all environments."""
        if hasattr(self, "env"):
            self.env.close()
