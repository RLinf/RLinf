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

import math
import os
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange
from scipy.spatial.transform import Rotation as R

from rlinf.envs.utils import (
    to_tensor,
)

from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from rlinf.envs.world_model.dataset import NpyTrajectoryDatasetWrapper
from rlinf.models.world_model.evac.lvdm.data.domain_table import DomainTable
from rlinf.models.world_model.evac.lvdm.data.statistics import StatisticInfo
from rlinf.models.world_model.evac.evac_utils.general_utils import (
    instantiate_from_config,
    load_checkpoints,
)

from rlinf.models.world_model.evac.lvdm.data.get_actions import get_action_from_abs_act

# load reward model and action_predictor

__all__ = ["EvacEnv"]


class EvacEnv(BaseWorldEnv):
    def __init__(self, cfg, seed_offset, total_num_processes, record_metrics=True):
        super().__init__(cfg, seed_offset, total_num_processes, record_metrics)

        # Load model
        self.model = self._load_model().eval().to(self.device)
        self.reward_model = self._load_reward_model().eval().to(self.device)
        self.action_predictor = self._load_action_predictor().eval().to(self.device)
        self._log_model_parameters()

        # Model hyperparameters
        self.chunk = self.cfg.chunk
        self.n_previous = self.cfg.n_previous
        self.sample_size = tuple(self.cfg.sample_size)

        # Initialize camera parameters
        self._init_camera_params()

        # Initialize state
        # For multi-env, we maintain separate current_obs for each environment
        # Each current_obs has shape [b, c, v, t, h, w] where b=1, v=1
        self.current_obs = (
            None  # Will be a list for multi-env, or single tensor for single env
        )
        self.task_descriptions = [""] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs

        # Initialize inference-related state
        self.all_x_samples = None
        self.all_samples = None
        self.all_c2ws_list = None
        self.all_trajs = None
        self.x_samples = None

        # Initialize data preprocessing
        self.trans_resize = transforms.Compose(
            [
                transforms.Resize(self.sample_size),
            ]
        )
        self.trans_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        # Inference parameters
        self.inference_dtype = torch.float32
        self.ddim_steps = 50
        self.ddim_eta = 1.0
        self.unconditional_guidance_scale = 7.5
        self.guidance_rescale = 0.7
        self.use_cat_mask = True
        self.sparse_memory = True

        # Set model mode
        self.model.rand_cond_frame = False
        self.model.ddim_num_chunk = 1

        # Initialize render buffers for debugging
        self.render_rgb = None
        self.render_actions = None
        self.render_rewards = None

    def _build_dataset(self, cfg):
        return NpyTrajectoryDatasetWrapper(cfg.initial_image_path)

    def _load_model(self):
        model = instantiate_from_config(self.cfg.world_model_cfg)
        model = load_checkpoints(
            model, self.cfg.world_model_cfg, ignore_mismatched_sizes=False
        )
        return model

    def _load_reward_model(self):
        from rlinf.models.world_model.evac.evac_utils.models import (
            RewModel
        )

        checkpoint = torch.load(
            self.cfg.reward_model_cfg.pretrained_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        rew_model = RewModel()
        rew_model.load_state_dict(checkpoint)
        # reward_model = instantiate_from_config(self.cfg.reward_model_cfg)
        # reward_model = load_checkpoints(
        #     reward_model, self.cfg.reward_model_cfg, ignore_mismatched_sizes=False
        # )
        return rew_model

    def _load_action_predictor(self):
        from rlinf.models.world_model.evac.evac_utils.models import (
            ActionPredictorMLP,
        )

        checkpoint = torch.load(
            self.cfg.action_predictor_cfg.pretrained_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        action_predictor = ActionPredictorMLP()
        action_predictor.load_state_dict(checkpoint["model_state_dict"])
        abs_action_mean = checkpoint.get("abs_action_mean", None)
        abs_action_std = checkpoint.get("abs_action_std", None)
        action_predictor.set_normalization_params(abs_action_mean, abs_action_std)
        return action_predictor

    def _log_model_parameters(self):
        """Log parameter counts for loaded models."""

        def count_params(model):
            if model is None:
                return 0, 0
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total, trainable

        world_total, world_trainable = count_params(self.model)
        reward_total, reward_trainable = count_params(self.reward_model)
        action_total, action_trainable = count_params(self.action_predictor)

        print(
            "[EvacEnv] World Model params: "
            f"total={world_total:,}, trainable={world_trainable:,}",
            flush=True,
        )
        print(
            "[EvacEnv] Reward Model params: "
            f"total={reward_total:,}, trainable={reward_trainable:,}",
            flush=True,
        )
        print(
            "[EvacEnv] Action Predictor params: "
            f"total={action_total:,}, trainable={action_trainable:,}",
            flush=True,
        )

    def _init_camera_params(self):
        """Initialize camera intrinsic and extrinsic parameters"""
        # Fixed intrinsic parameters
        self.intrinsic = torch.tensor(
            [[64, 0, 64], [0, 64, 64], [0, 0, 1]], dtype=torch.float32
        )

        # Fixed extrinsic parameters
        pos = np.array([0.30000001192092896, 0.0, 0.6000000238418579])
        quat_wxyz = np.array([0.0, -0.43318870663642883, 0.0, 0.901303231716156])
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot = R.from_quat(quat_xyzw).as_matrix()

        self.c2w = torch.eye(4, dtype=torch.float32)
        self.c2w[:3, :3] = torch.from_numpy(rot).float()
        self.c2w[:3, 3] = torch.from_numpy(pos).float()
        self.w2c = torch.linalg.inv(self.c2w)

        # Repeat for temporal dimension
        self.c2w_list = self.c2w.unsqueeze(0).repeat(self.n_previous, 1, 1)
        self.w2c_list = self.w2c.unsqueeze(0).repeat(self.n_previous, 1, 1)

    def _get_action_bias_std(self, domain_name="agibotworld"):
        return torch.tensor(StatisticInfo[domain_name]["mean"]).unsqueeze(
            0
        ), torch.tensor(StatisticInfo[domain_name]["std"]).unsqueeze(0)

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        # self.success_once = torch.zeros(
        #     self.num_envs, dtype=torch.bool
        # )
        # self.returns = torch.zeros(
        #     self.num_envs, dtype=torch.float32
        # )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            # mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask = torch.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
            self._elapsed_steps = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps = 0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        episode_info["return"] = self.returns.clone()
        # Ensure episode_len has shape [num_envs] to match returns
        episode_info["episode_len"] = torch.full(
            (self.num_envs,),
            self.elapsed_steps,
            dtype=torch.float32,
            device=self.device,
        )
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards):
        """Calculate step reward"""
        reward_diffs = torch.zeros((self.num_envs, self.chunk), dtype=torch.float32).to(
            self.device
        )
        # reward_diffs = torch.zeros((self.num_envs, self.chunk), dtype=torch.float32)
        for i in range(self.chunk):
            reward_diffs[:, i] = self.cfg.reward_coef * chunk_rewards[:, i] - self.prev_step_reward
            self.prev_step_reward = self.cfg.reward_coef * chunk_rewards[:, i]

        if self.use_rel_reward:
            return reward_diffs
        else:
            return chunk_rewards

    def update_reset_state_ids(self):
        """Updates the reset state IDs for environment initialization."""
        pass

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
    ):
        print("excute the reset function")
        self.elapsed_steps = 0

        num_envs = self.num_envs
        if len(self.dataset) < num_envs:
            raise ValueError(
                f"Not enough episodes in dataset. Found {len(self.dataset)}, need {num_envs}"
            )

        # Set random seed if provided
        if seed is not None:
            if isinstance(seed, list):
                np.random.seed(seed[0])
            else:
                np.random.seed(seed)

        # Randomly select episode indices
        episode_indices = np.random.choice(
            len(self.dataset), size=num_envs, replace=False
        )

        # Load first frame from each selected episode
        img_tensors = []
        task_descriptions = []
        init_ee_poses = []
        for episode_idx in episode_indices:
            # Get episode data from dataset wrapper
            episode_data = self.dataset[episode_idx]

            # Get first frame from start_items (should contain only one frame due to first_frame policy)
            if len(episode_data["start_items"]) == 0:
                raise ValueError(f"Empty start_items for episode {episode_idx}")

            first_frame = episode_data["start_items"][0]

            # Get task description
            task_desc = episode_data.get("task", "")
            task_descriptions.append(str(task_desc))

            # Get image from frame (already converted to torch tensor by wrapper)
            # The wrapper converts image to CHW format and normalizes to [0, 1]
            if "image" not in first_frame:
                raise ValueError(f"No 'image' key in frame for episode {episode_idx}")

            img_tensor = first_frame[
                "image"
            ]  # Shape: [3, H, W], dtype: float, range: [0, 1]

            # Get init_ee_pose if available (from observation.state)
            if "observation.state" in first_frame:
                init_ee_poses.append(first_frame["observation.state"].numpy())
            else:
                init_ee_poses.append(None)

            # Resize if needed using functional resize for tensors
            if img_tensor.shape[1:] != self.sample_size:
                img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W] for interpolation
                img_tensor = F.interpolate(
                    img_tensor,
                    size=self.sample_size,
                    mode="bilinear",
                    align_corners=False,
                )
                img_tensor = img_tensor.squeeze(0)  # [3, H, W]

            # Normalize to [-1, 1]
            img_tensor = self.trans_norm(img_tensor)

            # Repeat to fill memory frames: [3, H, W] -> [3, n_previous, H, W]
            img_tensor = img_tensor.unsqueeze(1).repeat(
                1, self.n_previous, 1, 1
            )  # [3, n_previous, H, W]

            img_tensors.append(img_tensor)

        self.current_obs = []
        for img_tensor in img_tensors:
            # [3, n_previous, H, W] -> [1, 3, 1, n_previous, H, W]
            env_obs = img_tensor.unsqueeze(1)
            self.current_obs.append(env_obs)
        # self.current_obs = torch.stack(self.current_obs, dim=0) # [num_envs, 3, 1, n_previous, H, W]
        self.current_obs = torch.stack(self.current_obs, dim=0).to(
            self.device
        )  # [num_envs, 3, 1, n_previous, H, W]

        self._is_start = False
        self._reset_metrics()

        # Initialize action buffer using init_ee_pose from dataset
        # Action format: [xyz_l, quat_xyzw_l, gripper_l, xyz_r, quat_xyzw_r, gripper_r] = [16]
        # init_ee_pose is 8-dim: [xyz, quat_xyzw, gripper] for one arm
        # We duplicate it to 16-dim for both arms (left and right are the same initially)
        init_actions = []
        for init_ee_pose in init_ee_poses:
            # Convert to numpy if it's a tensor
            if isinstance(init_ee_pose, torch.Tensor):
                init_ee_pose = init_ee_pose.numpy()

            # Ensure it's 1D array and duplicate to 16-dim for both arms
            init_ee_pose = init_ee_pose.flatten()
            init_action = np.concatenate([init_ee_pose, init_ee_pose], axis=0)
            init_actions.append(init_action)

        # Stack and repeat for n_previous frames
        init_actions_array = np.stack(init_actions, axis=0)  # [num_envs, 16]
        self.action_buffer = (
            torch.from_numpy(init_actions_array)
            .unsqueeze(1)
            .repeat(
                1, self.n_previous + self.chunk, 1
            )  # [num_envs, n_previous + chunk, 16]
            .to(self.device)
        )

        # Reset inference state
        self.all_x_samples = None
        self.all_samples = None
        self.all_c2ws_list = None
        self.all_trajs = None
        self.x_samples = None

        # Store task descriptions and init_ee_poses for later use
        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses

        # Wrap observation to match libero_env format
        extracted_obs = self._wrap_obs()
        infos = {}

        return extracted_obs, infos

    def step(self, actions=None, auto_reset=True):
        """Take a step in the environment"""
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            obs, infos = self.reset()
            self._is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            return obs, None, to_tensor(terminations), to_tensor(truncations), infos

        else:
            raise NotImplementedError("step in Evac Env is not impl")

    def _infer_next_chunk_rewards(
        self,
    ):
        """Predict next reward using the reward model"""
        # [num_envs, 3, 1, chunk, h, w] -> [num_envs, chunk,3, h, w]
        num_envs, c, v, chunk, h, w = self.current_obs.shape
        extract_chunk_obs = self.current_obs[:, :, :, -self.chunk :, :, :].permute(
            0, 3, 1, 2, 4, 5
        )  # [num_envs, chunk, 3, h, w]
        # 先整理为 [num_envs * chunk, 3, h, w], 推理后获得 [num_envs * chunk, 1]
        extract_chunk_obs = extract_chunk_obs.reshape(
            self.num_envs * self.chunk, 3, h, w
        )
        # Move to device before passing to reward model
        extract_chunk_obs = extract_chunk_obs.to(self.device)
        
        rewards = self.reward_model.predict_rew(extract_chunk_obs)
        # [num_envs, chunk, 1]
        rewards = rewards.reshape(self.num_envs, self.chunk)
        return rewards

    def _infer_next_chunk_frames(self):
        """Predict next frame using the world model"""
        from copy import deepcopy

        # Prepare current state
        # For single env: self.current_obs shape: [b, c, v, t, h, w] where b=1, v=1
        # For multi env: self.current_obs is a list of [b, c, v, t, h, w] tensors
        num_envs = self.num_envs
        b, c, v, t, h, w = self.current_obs.shape

        assert num_envs == b, (
            "Number of environments in current_obs does not match num_envs"
        )

        # Prepare video input
        # Multi-env: stack all envs' current_obs along batch dimension
        stacked_obs = deepcopy(self.current_obs)  # [num_envs, 3, 1, t, h, w]
        # when elapsed_steps=0, T=n_previous ; when elapsed_steps!=0, T=chunk+n_previous
        # Reshape to [num_envs, 3, 1, t, h, w] -> treat num_envs as batch
        if self.elapsed_steps == 0:
            video = torch.cat(
                (
                    stacked_obs,
                    stacked_obs[:, :, :, -1:, :, :].repeat(1, 1, 1, self.chunk, 1, 1),
                ),
                dim=3,
            )  # [num_envs, 3, 1, t+chunk, h, w]
        else:
            # Use previous predictions for each env
            # For now, use first env's history (will need to handle all envs)
            n_history = self.all_x_samples.shape[3]
            idx_history = [
                n_history * i // (self.n_previous - 1)
                for i in range(self.n_previous - 1)
            ]
            # This needs to be fixed for multi-env
            video = torch.cat(
                (
                    self.all_x_samples[:, :, :, idx_history, :, :]
                    if hasattr(self, "all_x_samples") and self.all_x_samples is not None
                    else stacked_obs,
                    stacked_obs[:, :, :, -1:, :, :].repeat(
                        1, 1, 1, self.chunk + 1, 1, 1
                    ),
                ),
                dim=3,
            )

        # video shape: for single env [b, c, v, t, h, w] where b=1

        # Prepare trajectory input
        if self.elapsed_steps == 0:
            traj_end_idx = self.n_previous + self.chunk

            # Ensure camera pose history is long enough for current trajectory
            if self.c2w_list.shape[0] < traj_end_idx:
                pad_count = traj_end_idx - self.c2w_list.shape[0]
                pad_c2w = self.c2w_list[-1:].repeat(pad_count, 1, 1)
                pad_w2c = self.w2c_list[-1:].repeat(pad_count, 1, 1)
                self.c2w_list = torch.cat([self.c2w_list, pad_c2w], dim=0)
                self.w2c_list = torch.cat([self.w2c_list, pad_w2c], dim=0)

            i_c2w_list = self.c2w_list[:traj_end_idx].unsqueeze(0)

            # Select camera params: w2c_list and c2w_list are (n_previous, 4, 4)
            # We need (V, 4, 4) where V=1 for single view
            # Use the first camera param (or average) for the trajectory
            w2c_for_traj = self.w2c_list[
                0:1
            ]  # (1, 4, 4) -> select first frame's camera
            c2w_for_traj = self.c2w_list[0:1]  # (1, 4, 4)

            # Multiple environments: process each environment separately
            trajs = []
            for env_idx in range(num_envs):
                action_for_traj = self.action_buffer[env_idx]  # (time_steps, 16)
                traj_env = self.model.get_traj(
                    self.sample_size,
                    action_for_traj.detach().cpu().numpy(),
                    w2c_for_traj,  # (1, 4, 4)
                    c2w_for_traj,  # (1, 4, 4)
                    self.intrinsic.unsqueeze(0),  # (1, 3, 3)
                )
                traj_env = rearrange(traj_env, "c v t h w -> (v t) c h w")
                traj_env = self.trans_norm(traj_env)
                traj_env = rearrange(traj_env, "(v t) c h w -> c v t h w", v=1)
                trajs.append(traj_env)
            # Stack: [num_envs, c, v, t, h, w] -> [c, num_envs, t, h, w]
            traj = torch.stack(trajs, dim=0)  # [num_envs, c, v, t, h, w]
            traj = traj.squeeze(2)  # [num_envs, c, t, h, w]
            traj = traj.permute(1, 0, 2, 3, 4)  # [c, num_envs, t, h, w]
            traj = traj.unsqueeze(0)  # [1, c, num_envs, t, h, w] for compatibility

            i_delta_action = self._compute_delta_action(self.action_buffer)
        else:
            # Use previous predictions for trajectory
            # Reference: ddpm3d.py line 1772-1775
            n_history = self.all_trajs.shape[3]
            idx_history = [
                n_history * i // (self.n_previous - 1)
                for i in range(self.n_previous - 1)
            ]

            # Calculate trajectory for current chunk
            # Use the last camera param for trajectory computation
            w2c_for_traj = (
                self.w2c_list[-1:] if len(self.w2c_list) > 0 else self.w2c_list[0:1]
            )  # (1, 4, 4)
            c2w_for_traj = (
                self.c2w_list[-1:] if len(self.c2w_list) > 0 else self.c2w_list[0:1]
            )  # (1, 4, 4)

            # Multiple environments: process each environment separately
            trajs_new = []
            for env_idx in range(num_envs):
                action_for_traj = self.action_buffer[
                    env_idx, -self.chunk - self.n_previous :
                ]
                traj_env_new = self.model.get_traj(
                    self.sample_size,
                    action_for_traj.detach().cpu().numpy(),
                    w2c_for_traj,  # (1, 4, 4)
                    c2w_for_traj,  # (1, 4, 4)
                    self.intrinsic.unsqueeze(0),  # (1, 3, 3)
                )
                traj_env_new = rearrange(traj_env_new, "c v t h w -> (v t) c h w")
                traj_env_new = self.trans_norm(traj_env_new)
                traj_env_new = rearrange(traj_env_new, "(v t) c h w -> c v t h w", v=1)
                trajs_new.append(traj_env_new)

            # Stack and process all environments (moved outside loop)
            # trajs_new: list of [c, v, t, h, w] where v=1
            traj_new = torch.stack(trajs_new, dim=0)  # [num_envs, c, v, t, h, w]
            traj_new = traj_new.squeeze(
                2
            )  # [num_envs, c, t, h, w] - remove v dimension
            traj_new = traj_new.permute(1, 0, 2, 3, 4)  # [c, num_envs, t, h, w]
            traj_new = traj_new.unsqueeze(0)  # [1, c, num_envs, t, h, w]

            # Concatenate history with new trajectory
            # self.all_trajs shape: [1, c, num_envs, t_history, h, w]
            # traj_new shape: [1, c, num_envs, t_new, h, w]
            # Reference: ddpm3d.py line 1772-1775
            traj = torch.cat(
                (
                    self.all_trajs[
                        :, :, :, idx_history, :, :
                    ],  # [1, c, num_envs, n_previous-1, h, w]
                    traj_new[
                        :, :, :, self.n_previous - 1 :, :, :
                    ],  # [1, c, num_envs, chunk+1, h, w]
                ),
                dim=3,
            )  # Result: [1, c, num_envs, chunk+n_previous, h, w]

            # Concatenate camera poses
            # Reference: ddpm3d.py line 1779-1782
            # self.all_c2ws_list shape: [1, t_history, 4, 4]
            # self.c2w_list shape: [t, 4, 4]
            i_c2w_list = torch.cat(
                (
                    self.all_c2ws_list[:, idx_history, :, :],  # [1, n_previous-1, 4, 4]
                    self.c2w_list[-self.chunk - self.n_previous :].unsqueeze(0)[
                        :, self.n_previous - 1 :, :, :
                    ],  # [1, chunk+1, 4, 4]
                ),
                dim=1,
            )  # Result: [1, chunk+n_previous, 4, 4]

            i_delta_action = self._compute_delta_action(
                self.action_buffer[:, -self.chunk - self.n_previous :]
            )

        if traj.shape[3] < self.chunk + self.n_previous:
            # Pad trajectory
            traj = torch.cat(
                (
                    traj,
                    traj[:, :, :, -1:].repeat(
                        1, 1, 1, self.chunk + self.n_previous - traj.shape[3], 1, 1
                    ),
                ),
                dim=3,
            )

        # Clamp values
        video = torch.clamp(video, min=-1, max=1)
        traj = torch.clamp(traj, min=-1, max=1)

        # Prepare batch
        intrinsic_batch = (
            self.intrinsic.unsqueeze(0).unsqueeze(0).repeat(num_envs, 1, 1, 1)
        )
        extrinsic_batch = i_c2w_list.repeat(num_envs, 1, 1, 1)

        fps = torch.tensor([2.0]).to(dtype=torch.float32, device=self.device)
        # fps = torch.tensor([2.0]).to(dtype=torch.float32)
        # pad to num_envs
        fps = fps.repeat(num_envs)
        domain_id = torch.LongTensor([DomainTable["agibotworld"]]).to(
            device=self.device
        )
        # domain_id = torch.LongTensor([DomainTable["agibotworld"]])
        # pad to num_envs
        domain_id = domain_id.repeat(num_envs)

        batch = {
            "video": video.to(dtype=self.inference_dtype, device=self.device),
            "traj": traj.to(dtype=self.inference_dtype, device=self.device),
            "delta_action": i_delta_action.to(
                dtype=self.inference_dtype, device=self.device
            ),
            "domain_id": domain_id,
            "intrinsic": intrinsic_batch.to(dtype=torch.float32, device=self.device),
            "extrinsic": extrinsic_batch.to(dtype=torch.float32, device=self.device),
            "caption": [""],
            "cond_id": torch.tensor(
                [-self.n_previous - self.chunk], dtype=torch.int64
            ).to(device=self.device),
            "fps": fps,
        }

        # Get batch input
        pre_z = None
        pre_img_emb = None
        if self.elapsed_steps != 0 and self.all_samples is not None:
            # Reference: ddpm3d.py line 1826-1829
            # self.all_samples shape: [num_envs, c, t, h, w] (latent space)
            n_history = self.all_samples.shape[2]
            idx_history = [
                n_history * i // (self.n_previous - 1)
                for i in range(self.n_previous - 1)
            ]
            # Use the last sample from previous chunk (in latent space, not decoded)
            # Need to get the last sample from the most recent chunk
            # Since we store samples after decoding, we need to use all_samples
            pre_z = torch.cat(
                (
                    self.all_samples[
                        :, :, idx_history
                    ],  # [num_envs, c, n_previous-1, h, w]
                    self.all_samples[:, :, -1:].repeat(
                        1, 1, self.chunk + 1, 1, 1
                    ),  # [num_envs, c, chunk+1, h, w]
                ),
                dim=2,
            ).to(
                dtype=self.inference_dtype, device=self.device
            )  # [num_envs, c, n_previous+chunk, h, w]

            # pre_z = torch.cat(
            #     (
            #         self.all_samples[:, :, idx_history],  # [num_envs, c, n_previous-1, h, w]
            #         self.all_samples[:, :, -1:]
            #         .repeat(1, 1, self.chunk + 1, 1, 1),  # [num_envs, c, chunk+1, h, w]
            #     ),
            #     dim=2,
            # ).to(
            #     dtype=self.inference_dtype
            # )  # [num_envs, c, n_previous+chunk, h, w]

        z, cond, xc, fs, did, img_emb = self.model.get_batch_input(
            batch,
            random_uncond=False,
            return_first_stage_outputs=False,
            return_original_cond=True,
            return_fs=True,
            return_did=True,
            return_traj=False,
            return_img_emb=True,
            pre_z=pre_z,
            pre_img_emb=pre_img_emb,
        )

        # Prepare conditions
        kwargs = {
            "fs": fs.long(),
            "domain_id": did.long(),
            "dtype": self.inference_dtype,
            "timestep_spacing": "uniform_trailing",
            "guidance_rescale": self.guidance_rescale,
            "return_intermediates": False,
        }

        # No unconditional guidance for now
        uc = None

        # Ensure correct dtype
        for _c_cat in range(len(cond["c_concat"])):
            cond["c_concat"][_c_cat] = cond["c_concat"][_c_cat].to(
                dtype=self.inference_dtype
            )
        for _c_cro in range(len(cond["c_crossattn"])):
            cond["c_crossattn"][_c_cro] = cond["c_crossattn"][_c_cro].to(
                dtype=self.inference_dtype
            )

        # Sample
        N = z.shape[0]
        samples, _ = self.model.sample_log(
            cond=cond,
            batch_size=N,
            ddim=True,
            ddim_steps=self.ddim_steps,
            causal=True,
            eta=self.ddim_eta,
            unconditional_guidance_scale=self.unconditional_guidance_scale,
            unconditional_conditioning=uc,
            x0=z.to(self.inference_dtype),
            chunk=self.chunk,
            cat_mask=self.use_cat_mask,
            sparse=self.sparse_memory,
            traj=False,
            ddim_dtype=torch.float16,
            **kwargs,
        )

        # Decode samples
        x_samples = self.model.decode_first_stage(samples.to(z.device)).data.cpu()
        x_samples = rearrange(x_samples, "(b v) c t h w -> b c v t h w", v=v)

        # Store for next iteration
        # x_samples shape: [b, c, v, t, h, w]
        if self.all_x_samples is None:
            self.all_x_samples = x_samples.data.cpu()
            self.all_samples = samples.data.cpu()
            self.all_c2ws_list = i_c2w_list.data.cpu()
            self.all_trajs = traj.data.cpu()
        else:
            # Concatenate along time dimension (dim=3)
            self.all_x_samples = torch.cat(
                (
                    self.all_x_samples,
                    x_samples[:, :, :, self.n_previous :, :, :].data.cpu(),
                ),
                dim=3,
            )
            self.all_samples = torch.cat(
                (self.all_samples, samples[:, :, self.n_previous :, :, :].data.cpu()),
                dim=2,
            )
            self.all_c2ws_list = torch.cat(
                (self.all_c2ws_list, i_c2w_list[:, self.n_previous :].data.cpu()), dim=1
            )
            self.all_trajs = torch.cat(
                (self.all_trajs, traj.data.cpu()[:, :, :, self.n_previous :]), dim=3
            )

        self.x_samples = x_samples

        # Update current observation
        # x_samples shape: [b, c, v, T, h, w], extract from n_previous onwards  T=chunk+n_previous
        assert x_samples.shape[0] == num_envs, (
            f"Unexpected x_samples shape: {x_samples.shape}, expected {num_envs}"
        )
        # self.current_obs shape: [num_envs, c, v, T, h, w] T=chunk
        self.current_obs = x_samples[:, :, :, self.n_previous :, :, :]
        # Note: elapsed_steps is incremented in chunk_step method after inference, not here

        # Update camera list for next iteration
        self.c2w_list = torch.cat((self.c2w_list, self.c2w_list[-1:]), dim=0)
        self.w2c_list = torch.cat((self.w2c_list, self.w2c_list[-1:]), dim=0)

    def _wrap_obs(self):
        """Wrap observation to match libero_env format"""
        num_envs = self.num_envs

        # Extract the last frame (most recent observation) for each environment
        # self.current_obs is [b, c, v, t, h, w]  v=1 for single view
        b, c, v, t, h, w = self.current_obs.shape
        assert b == num_envs, (
            f"Unexpected current_obs shape: {self.current_obs.shape}, expected {num_envs}"
        )

        last_frame = self.current_obs[
            :, :, 0, -1, :, :
        ]  # [b,3, v, t,h,w] -> [b, 3, 1, h, w] -> [b, 3, h, w]

        full_image = last_frame.permute(0, 2, 3, 1)  # [b, H, W, 3]

        # Denormalize from [-1, 1] to [0, 255]
        full_image = (full_image + 1.0) / 2.0 * 255.0

        full_image = torch.clamp(full_image, 0, 255)

        target_size = self.sample_size

        # Resize to 256x256 to match libero_env format
        if full_image.shape[1:3] != target_size:
            # Reshape for interpolation: [num_envs, H, W, 3] -> [num_envs, 3, H, W]
            full_image = full_image.permute(0, 3, 1, 2)  # [num_envs, 3, H, W]
            # Resize using F.interpolate
            full_image = F.interpolate(
                full_image, size=self.sample_size, mode="bilinear", align_corners=False
            )
            # Convert back: [num_envs, 3, 256, 256] -> [num_envs, 256, 256, 3]
            full_image = full_image.permute(0, 2, 3, 1)  # [num_envs, 256, 256, 3]

        # Convert to uint8 tensor (keep as tensor, not numpy)
        full_image = full_image.to(torch.uint8)

        states = self.action_buffer[:, -1]  # [num_envs, 16]

        # Get task descriptions
        if hasattr(self, "task_descriptions"):
            task_descriptions = self.task_descriptions
        else:
            raise ValueError("task_descriptions not found")

        # Wrap observation
        obs = {
            "images_and_states": {
                "full_image": full_image,  # [num_envs, H, W, 3]
                "state": states,  # [num_envs, 16] - padded to match model compatibility
            },
            "task_descriptions": task_descriptions,  # list of strings
        }

        return obs

    def policy_output_to_abs_action(self, policy_output):
        """Convert policy output to absolute action"""
        # policy output: [num_envs, chunk, 7]
        # abs_action_mean: [1, 8]
        # abs_action_std: [1, 8]
        policy_output = torch.tensor(policy_output, device=self.device).float()
        pre_ee_pose = self.action_buffer[:, -self.chunk - 1, :8]
        ee_pose_list = torch.zeros((self.num_envs, self.chunk, 8), device=self.device)

        num_envs, chunk, _ = policy_output.shape
        for i in range(chunk):
            ee_pose_list[:, i] = self.action_predictor(policy_output[:, i], pre_ee_pose)
            pre_ee_pose = ee_pose_list[:, i]

        # padding ee_pose_list
        ee_pose_list = torch.cat([ee_pose_list, ee_pose_list], dim=-1)

        # replace self.action_buffer
        self.action_buffer[:, -self.chunk - 1 : -1] = ee_pose_list
        abs_action = self.action_buffer[:, -self.chunk - 1 : -1]

        return abs_action

    def _compute_delta_action(self, action_buffer):
        """Compute delta action from action buffer"""

        num_envs = self.num_envs

        # Process each environment separately
        delta_actions = []
        for env_idx in range(num_envs):
            action_np = action_buffer[env_idx].detach().cpu().numpy()
            _, delta_action = get_action_from_abs_act(action_np)
            if isinstance(delta_action, np.ndarray) and len(delta_action.shape) > 1:
                delta_action = delta_action[-1]  # Take only the last one
            delta_actions.append(delta_action)

        return torch.from_numpy(np.stack(delta_actions, axis=0))  # [num_envs, 16]

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        """Handle automatic reset on episode termination"""
        final_obs = extracted_obs
        final_info = infos

        extracted_obs, infos = self.reset()

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones

        return extracted_obs, infos

    def chunk_step(self, policy_output_action):
        """Execute a chunk of actions - optimized version that processes chunk actions together"""
        # chunk_actions: [num_envs, chunk_steps, action_dim=8]

        self.policy_output_to_abs_action(policy_output_action)

        _, _, action_dim = self.action_buffer.shape

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            # extracted_chunk_obs shape: [num_envs, c, v, chunk, h, w]
            self._infer_next_chunk_frames()

        # Update elapsed steps (incremented after inference)
        self.elapsed_steps += self.chunk

        extracted_obs = self._wrap_obs()

        chunk_rewards = self._infer_next_chunk_rewards()

        # Calculate reward for this step (dummy for now)
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        # No terminations/truncations for now (could implement based on max steps)
        raw_chunk_terminations = deepcopy(chunk_rewards)

        raw_chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        # raw_chunk_truncations = torch.zeros(
        #     num_envs, chunk_size, dtype=torch.bool
        # )
        truncations = torch.tensor(self.elapsed_steps >= self.cfg.max_episode_steps).to(
            self.device
        )
        # truncations = torch.tensor(self.elapsed_steps >= self.cfg.max_episode_steps)

        if truncations.any():
            raw_chunk_truncations[:, -1] = truncations

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, {}
            )
        else:
            infos = {}

        infos = self._record_metrics(chunk_rewards_tensors.sum(dim=1), infos)

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        # Get actions for the current chunk (extract before updating action_buffer)
        # action_buffer shape: [num_envs, n_previous + chunk, action_dim=16]
        # Extract last chunk steps: [num_envs, chunk, action_dim]
        chunk_actions_for_render = self.action_buffer[:, -self.chunk:, :].detach().cpu().numpy()
        # Get rewards for the current chunk (chunk_rewards_tensors shape: [num_envs, chunk])
        chunk_rewards_for_render = chunk_rewards_tensors.detach().cpu().numpy()

        # Reshape actions: [num_envs, chunk, action_dim] -> [chunk, num_envs, action_dim]
        chunk_actions_for_render = chunk_actions_for_render.transpose(1, 0, 2)
        # Reshape rewards: [num_envs, chunk] -> [chunk, num_envs]
        chunk_rewards_for_render = chunk_rewards_for_render.T

        self.add_new_frames(actions=chunk_actions_for_render, rewards=chunk_rewards_for_render)

        # Update action_buffer after extracting actions for rendering
        self.action_buffer = self.action_buffer[:, -self.chunk - self.n_previous :, :]

        return (
            extracted_obs,
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def add_new_frames(self, actions=None, rewards=None):
        """Append all frames from the latest chunk into the render buffer."""
        if self.current_obs is None:
            return

        num_envs, channels, num_views, num_steps, height, width = self.current_obs.shape
        view_idx = 0  # visualize the first camera view
        chunk_len = min(self.chunk, num_steps)
        start_step = num_steps - chunk_len

        # Collect RGB info (CHW format) for all frames in chunk
        rgb_info_list = []
        for step_idx in range(chunk_len):
            images = []
            step_rgb_list = []
            for env_idx in range(num_envs):
                frame_tensor = self.current_obs[
                    env_idx, :, view_idx, start_step + step_idx, :, :
                ]
                # Convert to HWC for display
                frame_np = frame_tensor.detach().cpu().permute(1, 2, 0).numpy()
                frame_np = (frame_np + 1.0) / 2.0 * 255.0
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                images.append(frame_np)

                # Store CHW format for RGB info (same as libero_env_copy.py)
                frame_chw = frame_tensor.detach().cpu().numpy()  # Already CHW format [C, H, W]
                # Normalize from [-1, 1] to [0, 255] for CHW
                frame_chw = (frame_chw + 1.0) / 2.0 * 255.0
                frame_chw = np.clip(frame_chw, 0, 255).astype(np.uint8)
                step_rgb_list.append(frame_chw)

            # Stack RGB info for this step: [num_envs, C, H, W]
            step_rgb_info = np.stack(step_rgb_list, axis=0)
            rgb_info_list.append(step_rgb_info)

            tiled = self._tile_images(images)
            if tiled is not None:
                self.render_images.append(tiled)

        # Stack RGB info for all steps: [chunk_len, num_envs, C, H, W]
        if rgb_info_list:
            chunk_rgb_info = np.stack(rgb_info_list, axis=0)
            if self.render_rgb is None:
                self.render_rgb = chunk_rgb_info  # [chunk_len, num_envs, C, H, W]
            else:
                self.render_rgb = np.concatenate([self.render_rgb, chunk_rgb_info], axis=0)

        # Add actions into buffer
        if actions is not None:
            actions_np = actions
            if isinstance(actions, torch.Tensor):
                actions_np = actions.detach().cpu().numpy()
            if actions_np.ndim == 2:
                # [num_envs, action_dim] -> [1, num_envs, action_dim]
                actions_np = np.expand_dims(actions_np, axis=0)
            if actions_np.shape[1] != num_envs:
                try:
                    actions_np = actions_np.reshape(-1, num_envs, actions_np.shape[-1])
                except Exception:
                    raise ValueError(
                        f"actions shape {tuple(actions_np.shape)} incompatible with num_envs={num_envs}"
                    )
            # actions_np shape: [chunk_len, num_envs, action_dim]
            if self.render_actions is None:
                self.render_actions = actions_np
            else:
                self.render_actions = np.concatenate(
                    [self.render_actions, actions_np], axis=0
                )

        # Add rewards into buffer
        if rewards is not None:
            rewards_np = rewards
            if isinstance(rewards, torch.Tensor):
                rewards_np = rewards.detach().cpu().numpy()
            if rewards_np.ndim == 1:
                # [num_envs] -> [1, num_envs]
                rewards_np = np.expand_dims(rewards_np, axis=0)
            elif rewards_np.ndim == 2 and rewards_np.shape[0] != chunk_len:
                # If shape is [num_envs, chunk_len], transpose to [chunk_len, num_envs]
                if rewards_np.shape[1] == chunk_len:
                    rewards_np = rewards_np.T
            if rewards_np.shape[1] != num_envs:
                try:
                    rewards_np = rewards_np.reshape(-1, num_envs)
                except Exception:
                    raise ValueError(
                        f"rewards shape {tuple(rewards_np.shape)} incompatible with num_envs={num_envs}"
                    )
            # rewards_np shape: [chunk_len, num_envs]
            if self.render_rewards is None:
                self.render_rewards = rewards_np
            else:
                self.render_rewards = np.concatenate(
                    [self.render_rewards, rewards_np], axis=0
                )

    def _tile_images(self, images, nrows: Optional[int] = None):
        """Tile multiple images into a single grid."""
        if not images:
            return None

        num_images = len(images)
        height, width, channels = images[0].shape
        rows = nrows or max(1, int(math.sqrt(num_images)))
        cols = int(math.ceil(num_images / rows))

        canvas = np.zeros(
            (rows * height, cols * width, channels), dtype=images[0].dtype
        )
        for idx, image in enumerate(images):
            row = idx // cols
            col = idx % cols
            y0, y1 = row * height, (row + 1) * height
            x0, x1 = col * width, (col + 1) * width
            canvas[y0:y1, x0:x1] = image

        return canvas

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Save accumulated video frames"""
        if len(self.render_images) == 0:
            return

        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")

        os.makedirs(output_dir, exist_ok=True)

        from mani_skill.utils.visualization.misc import images_to_video

        images_to_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
            fps=self.cfg.get("fps", 10),
            verbose=False,
        )

        self.video_cnt += 1
        self.render_images = []
        self.render_rgb = None
        self.render_actions = None
        self.render_rewards = None


if __name__ == "__main__":
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize_config_dir

    # Set required environment variable
    os.environ.setdefault(
        "EMBODIED_PATH", "/mnt/mnt/public/jzn/workspace/RLinf/examples/embodiment"
    )

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    config_dir = "/mnt/mnt/public/jzn/workspace/RLinf/examples/embodiment/config"
    config_name = "libero_spatial_evac_grpo_openvlaoft_impl"

    print(f"Loading config: {config_name} from {config_dir}")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]
    cfg.num_envs = 2
    cfg.video_cfg.video_base_dir = "/mnt/mnt/public/jzn/workspace/latest/RLinf"
    env = EvacEnv(cfg, seed_offset=0, total_num_processes=1)

    # Reset environment
    obs, info = env.reset()
    print("\nAfter reset:")
    print(f"  obs keys: {list(obs.keys())}")

    # Test 1: chunk_steps = self.chunk
    print("\n" + "-" * 80)
    chunk_steps = 8
    num_envs = cfg.num_envs

    def read_delta_actions(num_frames=8):
        path = "/mnt/mnt/public/jzn/workspace/RLinf/reward_model/reward_data/embodiment_converted/train_data/step_0_seed_0_traj_0.npy"
        traj = np.load(path, allow_pickle=True)
        delta_actions = np.stack(
            [frame["delta_action"] for frame in traj[:num_frames]], axis=0
        )
        return delta_actions

    num_frames = 8
    chunk_traj = num_frames // chunk_steps
    delta_actions = read_delta_actions()
    action_dim = delta_actions.shape[-1]
    
    delta_actions = delta_actions.reshape(chunk_traj, chunk_steps, action_dim)

    for chunk_idx in range(delta_actions.shape[0]):
        chunk_actions = np.tile(delta_actions[chunk_idx], (num_envs, 1, 1))
        obs, reward, term, trunc, info = env.chunk_step(chunk_actions)
        print(f"chunk {chunk_idx} done")
        print(env.action_buffer.shape)
        print(env.current_obs.shape)

    env.flush_video()
