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

import io
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffsynth.models.reward_model import ResnetRewModel, TaskEmbedResnetRewModel
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
from PIL import Image

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["WanEnv"]


class WanEnv(BaseWorldEnv):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        record_metrics=True,
        worker_info=None,
    ):
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )
        # Reset state management
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size

        # Initialize reset state generator
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)

        # Update reset state ids
        self.update_reset_state_ids()

        # Model hyperparameters
        self.chunk = cfg.chunk  # Ta = 8
        self.condition_frame_length = cfg.condition_frame_length  # To = 5
        self.num_frames = cfg.num_frames  # Total number of frames to encode = 13
        assert self.num_frames == self.condition_frame_length + self.chunk, (
            "num_frames must be equal to condition_frame_length + action_chunk_length"
        )

        self.image_size = tuple(cfg.image_size)

        #
        self.retain_action = cfg.get("retain_action", True)  # Default True
        self.enable_kir = cfg.get("enable_kir", True)

        # load pipeline
        self.pipe = self._build_pipeline()

        # Load reward model if specified
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # Initialize state
        self.current_obs = None  # Will be a tensor [num_envs, 3, 1, t, h, w]
        self.task_descriptions = [""] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs

        # Image queue for condition frames
        self.image_queue = [
            [None] * self.condition_frame_length for _ in range(self.num_envs)
        ]

        self.condition_action = torch.zeros(
            self.num_envs,
            self.condition_frame_length,
            7,
        )
        self.condition_action[..., -1] = -1
        self.trans_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        self._is_offloaded = False

    def _build_dataset(self, cfg):
        return NpyTrajectoryDatasetWrapper(
            cfg.initial_image_path, enable_kir=self.enable_kir
        )

    def _build_pipeline(self):
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda:0",
            model_configs=[
                # Paths are loaded from yaml
                ModelConfig(path=self.cfg.model_path, offload_device="cpu"),
                ModelConfig(path=self.cfg.VAE_path, offload_device="cpu"),
            ],
        )
        # pipe.enable_vram_management()
        pipe.dit.to(self.device)
        pipe.vae.to(self.device)
        return pipe

    def _load_reward_model(self):
        if self.cfg.reward_model.type == "ResnetRewModel":
            rew_model = ResnetRewModel(self.cfg.reward_model.from_pretrained)
        elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
            rew_model = TaskEmbedResnetRewModel(
                checkpoint_path=self.cfg.reward_model.from_pretrained,
                task_suite_name=self.cfg.task_suite_name,
            )
        else:
            raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")
        return rew_model

    def _init_metrics(self):
        self.success_once = torch.zeros(
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
                self.returns[mask] = 0
            # self._elapsed_steps = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        # Update success_once based on terminations
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
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards):
        """Calculate step reward"""
        reward_diffs = torch.zeros(
            (self.num_envs, self.chunk), dtype=torch.float32, device=self.device
        )
        for i in range(self.chunk):
            reward_diffs[:, i] = (
                self.cfg.reward_coef * chunk_rewards[:, i] - self.prev_step_reward
            )
            self.prev_step_reward = self.cfg.reward_coef * chunk_rewards[:, i]

        if self.use_rel_reward:
            return reward_diffs
        else:
            return chunk_rewards

    def _estimate_success_from_rewards(self, chunk_rewards):
        """
        Estimate success (terminations) based on reward values.
        Success is estimated when reward exceeds a threshold (default: 0.9).
        """
        # Get success threshold from config, default to 0.9
        success_threshold = getattr(self.cfg, "success_reward_threshold", 0.9)

        # Check if any reward in the chunk exceeds the threshold
        # chunk_rewards shape: [num_envs, chunk]
        max_reward_in_chunk = chunk_rewards.max(dim=1)[0]  # [num_envs]
        success_estimated = max_reward_in_chunk >= success_threshold

        return success_estimated.to(self.device)

    def update_reset_state_ids(self):
        """Updates the reset state IDs for environment initialization."""
        # Get total number of episodes available
        total_num_episodes = len(self.dataset)

        # Generate random reset state ids
        reset_state_ids = torch.randint(
            low=0,
            high=total_num_episodes,
            size=(self.num_group,),
            generator=self._generator,
        )

        # Repeat for each environment in the group
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        self.onload()
        self.elapsed_steps = 0

        # Handle first reset with fixed reset state ids
        if self.is_start:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        num_envs = self.num_envs
        if len(self.dataset) < num_envs:
            raise ValueError(
                f"Not enough episodes in dataset. Found {len(self.dataset)}, need {num_envs}"
            )

        # If episode_indices not provided, randomly select
        if episode_indices is None:
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
        else:
            # Convert to numpy if tensor
            if isinstance(episode_indices, torch.Tensor):
                episode_indices = episode_indices.cpu().numpy()

        # Load first frame from each selected episode
        img_tensors = []
        task_descriptions = []
        init_ee_poses = []

        for env_idx, episode_idx in enumerate(episode_indices):
            # Get episode data from dataset wrapper
            episode_data = self.dataset[episode_idx]

            # Get first frame from start_items
            if len(episode_data["start_items"]) == 0:
                raise ValueError(f"Empty start_items for episode {episode_idx}")

            first_frame = episode_data["start_items"][0]

            # Get task description
            task_desc = episode_data.get("task", "")
            task_descriptions.append(str(task_desc))

            # Get image from frame
            if "image" not in first_frame:
                raise ValueError(f"No 'image' key in frame for episode {episode_idx}")

            img_tensor = first_frame[
                "image"
            ]  # Shape: [3, H, W], dtype: float, range: [0, 1]
            # [3, 256, 256], float32, [0,1]
            # Wan requires images in PIL format

            # Get init_ee_pose if available
            if "observation.state" in first_frame:
                init_ee_poses.append(first_frame["observation.state"].numpy())
            else:
                init_ee_poses.append(None)

            # Resize if needed
            if img_tensor.shape[1:] != self.image_size:
                img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
                img_tensor = F.interpolate(
                    img_tensor,
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                img_tensor = img_tensor.squeeze(0)  # [3, H, W]

            # Normalize to [-1, 1]
            img_tensor = self.trans_norm(img_tensor)

            # KIR trick: use the last four frames as condition frames, while
            # keeping the reference frame unchanged as the first frame.
            target_items = episode_data.get("target_items", [])
            if len(target_items) == self.condition_frame_length - 1:
                final_frames = []
                for target_frame in target_items:
                    if "image" not in target_frame:
                        raise ValueError(
                            f"No 'image' key in target frame for episode {episode_idx}"
                        )
                    target_img = target_frame["image"]
                    if target_img.shape[1:] != self.image_size:
                        target_img = target_img.unsqueeze(0)  # [1, 3, H, W]
                        target_img = F.interpolate(
                            target_img,
                            size=self.image_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        target_img = target_img.squeeze(0)  # [3, H, W]
                    target_img = self.trans_norm(target_img)
                    final_frames.append(target_img)
                final_img_tensor = torch.stack(final_frames, dim=1)
                img_tensor = img_tensor.unsqueeze(1)
                # [3, condition_frame_length, H, W]
                img_tensor = torch.cat([img_tensor, final_img_tensor], dim=1)
            else:
                # Repeat to fill condition frames: [3, H, W] -> [3, condition_frame_length, H, W]
                img_tensor = img_tensor.unsqueeze(1).repeat(
                    1, self.condition_frame_length, 1, 1
                )  # [3, condition_frame_length, H, W]
            # print(f'img_tensor:{img_tensor.shape}')
            img_tensors.append(img_tensor)

        # Stack all environments: [num_envs, 3, condition_frame_length, H, W]
        # [8, 3, 5, 256, 256]
        stacked_imgs = torch.stack(img_tensors, dim=0).to(self.device)

        # Reshape to [num_envs, 3, 1, condition_frame_length, H, W] for compatibility
        # [8, 3, 1, 5, 256, 256]
        self.current_obs = stacked_imgs.unsqueeze(2).to(self.device)
        # Shape: [num_envs, 3, 1, condition_frame_length, H, W]

        num_envs, c, v, t, h, w = self.current_obs.shape

        # Fill image queues for each environment
        for env_idx in range(num_envs):
            # self.image_queue[env_idx].clear()
            self.image_queue[env_idx] = [None] * self.condition_frame_length
            for t_idx in range(t):
                self.image_queue[env_idx][t_idx : t_idx + 1] = self.current_obs[
                    env_idx : env_idx + 1, :, 0, t_idx : t_idx + 1, :, :
                ]

        self._reset_metrics()

        # Initialize action buffer (if needed)
        # But we'll keep it for compatibility
        if hasattr(self.cfg, "action_dim"):
            action_dim = self.cfg.action_dim
        else:
            action_dim = 7  # Default for LIBERO

        # Initialize with zeros or from init_ee_pose
        init_actions = []
        for init_ee_pose in init_ee_poses:
            if init_ee_pose is not None:
                init_action = init_ee_pose.flatten()
                # Pad or truncate to action_dim
                if len(init_action) < action_dim:
                    init_action = np.pad(
                        init_action, (0, action_dim - len(init_action))
                    )
                elif len(init_action) > action_dim:
                    init_action = init_action[:action_dim]
            else:
                init_action = np.zeros(action_dim, dtype=np.float32)
            init_actions.append(init_action)

        # Store init_ee_poses
        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses

        # Wrap observation to match libero_env format
        extracted_obs = self._wrap_obs()
        infos = {}

        return extracted_obs, infos

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError("step in Wan Env is not impl, use chunk_step instead")

    def _infer_next_chunk_rewards(self):
        """Predict next reward using the reward model"""
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        # Extract chunk observations
        num_envs, c, v, t, h, w = self.current_obs.shape
        extract_chunk_obs = self.current_obs.permute(
            0, 3, 1, 2, 4, 5
        )  # [num_envs, chunk + condition_frame_length, 3, v, h, w]

        if self.cfg.reward_model.type == "ResnetRewModel":
            extract_chunk_obs = extract_chunk_obs[
                :, -self.chunk :, :, :, :, :
            ]  # [num_envs, chunk, 3, v, h, w]
            extract_chunk_obs = extract_chunk_obs.reshape(
                self.num_envs * self.chunk, 3, v, h, w
            )
            extract_chunk_obs = extract_chunk_obs.squeeze(
                2
            )  # [num_envs * chunk, 3, h, w]
            extract_chunk_obs = extract_chunk_obs.to(self.device)

            rewards = self.reward_model.predict_rew(extract_chunk_obs)
            rewards = rewards.reshape(self.num_envs, self.chunk)
        elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
            extract_chunk_obs = extract_chunk_obs[
                :, -self.chunk :, :, :, :, :
            ]  # [num_envs, chunk, 3, v, h, w]
            extract_chunk_obs = extract_chunk_obs.reshape(
                self.num_envs * self.chunk, 3, v, h, w
            )
            extract_chunk_obs = extract_chunk_obs.squeeze(
                2
            )  # [num_envs * chunk, 3, h, w]
            extract_chunk_obs = extract_chunk_obs.to(self.device)

            # Prepare instructions for each frame in the chunk
            # Each environment has one task description, repeat it for each frame in the chunk
            instructions = []
            for env_idx in range(self.num_envs):
                task_desc = self.task_descriptions[env_idx]
                # Repeat the instruction for each frame in the chunk
                instructions.extend([task_desc] * self.chunk)

            # Predict rewards with instruction conditioning
            rewards = self.reward_model.predict_rew(extract_chunk_obs, instructions)
            rewards = rewards.reshape(self.num_envs, self.chunk)
        else:
            raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")

        return rewards

    def _infer_next_chunk_frames(self, actions):
        """Predict next frame chunk using the wan model"""
        num_envs = self.num_envs
        assert actions.shape[0] == self.num_envs, (
            f"Actions shape {actions.shape} does not match num_envs {self.num_envs}"
        )

        # Normalize actions
        actions_tensor = torch.from_numpy(actions).to(self.device)
        self.condition_action = self.condition_action.to(
            device=actions_tensor.device, dtype=actions_tensor.dtype
        )

        if self.retain_action:
            actions_tensor = torch.cat([self.condition_action, actions_tensor], dim=1)

        self.condition_action[:, 1 : self.condition_frame_length, :] = actions_tensor[
            :, -(self.condition_frame_length - 1) :, :
        ]
        # print(f'actions_tensor:{actions_tensor.shape}')
        # Process each environment separately
        all_samples = []

        B = num_envs

        batch_input_image = []
        batch_input_image4 = []

        for env_idx in range(num_envs):
            # image_queue: [8, 3, 1, H, W]
            imgs = []
            for frame in self.image_queue[env_idx]:
                frame = frame[:, 0].cpu().numpy()  # [3, H, W]
                img = np.transpose(frame, (1, 2, 0))
                if img.max() <= 1.2:
                    img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255)
                imgs.append(Image.fromarray(img.astype(np.uint8)))

            batch_input_image.append(imgs[0])  # First frame
            batch_input_image4.append(imgs[-4:])  # Last 4 frames

        kwargs = {
            "seed": 0,
            "tiled": False,
            "input_image": batch_input_image,  # List[PIL], len = B
            "input_image4": batch_input_image4,  # List[List[PIL]], B×4
            "action": actions_tensor,  # [B, T, A], T=13
            "height": 256,
            "width": 256,
            "num_frames": self.num_frames,
            "num_inference_steps": 5,
            "cfg_scale": 1.0,
            "progress_bar_cmd": lambda x: x,
            "batch_size": B,
        }

        output = self.pipe(**kwargs)
        for env_idx in range(num_envs):
            frames = []
            for img in output[env_idx]:
                arr = np.array(img) / 255.0
                arr = arr * 2.0 - 1.0
                frames.append(arr)

            video = np.stack(frames, axis=0)  # [T, H, W, 3]
            video = video.transpose(0, 3, 1, 2)  # [T, 3, H, W]
            video = torch.from_numpy(video)
            video = video.transpose(0, 1)  # [3, T, H, W]

            # Update image_queue
            for t in range(video.shape[1] - 4, video.shape[1]):
                self.image_queue[env_idx][t - 8] = video[:, t : t + 1]

            all_samples.append(video[:, 5:])

        # Stack all environments: [num_envs, C, T, H, W]
        x_samples = torch.stack(all_samples, dim=0).to(self.device)

        # Reshape to match current_obs format: [num_envs, C, 1, T, H, W]
        x_samples = x_samples.unsqueeze(2)

        # Update current observation
        # For first chunk, we need to replace condition frames
        # For subsequent chunks, we concatenate new frames
        if self.current_obs.shape[3] == self.condition_frame_length:
            # First chunk: keep condition frames and add new frames
            # But we need to decode the condition frames back to image space first
            # Actually, current_obs is already in image space, so we just concatenate
            self.current_obs = torch.cat([self.current_obs, x_samples], dim=3)
        else:
            # Subsequent chunks: concatenate new frames
            self.current_obs = torch.cat([self.current_obs, x_samples], dim=3)

        # Keep only the last condition_frame_length + chunk frames
        # Note: chunk might be different from T in x_samples due to VAE decoding
        # We'll keep a sliding window of recent frames
        max_frames = self.condition_frame_length + self.chunk * 2  # Keep some buffer
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]

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
        # [8, 3, 256, 256]

        full_image = last_frame.permute(0, 2, 3, 1)  # [b, H, W, 3]
        # Denormalize from [-1, 1] to [0, 255]
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255)
        # print(f'full_image:{full_image.shape}')
        # print(f'image_size:{self.image_size}')
        # Resize to 256x256 to match libero_env format
        if full_image.shape[1:3] != self.image_size:
            # Reshape for interpolation: [num_envs, H, W, 3] -> [num_envs, 3, H, W]
            full_image = full_image.permute(0, 3, 1, 2)  # [num_envs, 3, H, W]
            # Resize using F.interpolate
            full_image = F.interpolate(
                full_image, size=self.image_size, mode="bilinear", align_corners=False
            )
            # Convert back: [num_envs, 3, 256, 256] -> [num_envs, 256, 256, 3]
            full_image = full_image.permute(0, 2, 3, 1)  # [num_envs, 256, 256, 3]

        # Convert to uint8 tensor (keep as tensor, not numpy)
        full_image = full_image.to(torch.uint8)

        # Get states (dummy for now, can be extended)
        states = torch.zeros((num_envs, 16), device=self.device, dtype=torch.float32)

        # Get task descriptions
        if hasattr(self, "task_descriptions"):
            task_descriptions = self.task_descriptions
        else:
            task_descriptions = [""] * num_envs

        # Wrap observation - format aligned with libero_env
        obs = {
            "main_images": full_image,  # [num_envs, H, W, 3]
            "wrist_images": None,  # Not available in world model
            "states": states,  # [num_envs, 16]
            "task_descriptions": task_descriptions,  # list of strings
        }

        return obs

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

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        """Execute a chunk of actions - optimized version that processes chunk actions together"""
        # chunk_actions: [num_envs, chunk_steps, action_dim=8]
        self.onload()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            self._infer_next_chunk_frames(policy_output_action)

        # Update elapsed steps (incremented after inference)
        # print(f'elapsed_steps:{self.elapsed_steps}')
        self.elapsed_steps += self.chunk

        # Read the last frame from self.current_obs
        extracted_obs = self._wrap_obs()

        # Get rewards
        chunk_rewards = self._infer_next_chunk_rewards()
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        # Estimate success (terminations) based on rewards
        estimated_success = self._estimate_success_from_rewards(chunk_rewards)

        # Create terminations tensor: success is marked at the last step of chunk
        raw_chunk_terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        raw_chunk_terminations[:, -1] = estimated_success

        raw_chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        truncations = torch.tensor(self.elapsed_steps >= self.cfg.max_episode_steps).to(
            self.device
        )

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

        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_terminations, infos
        )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        # Get actions and rewards for rendering
        chunk_actions_for_render = policy_output_action
        if isinstance(chunk_actions_for_render, torch.Tensor):
            chunk_actions_for_render = chunk_actions_for_render.detach().cpu().numpy()
        chunk_rewards_for_render = chunk_rewards_tensors.detach().cpu().numpy()

        # Reshape for rendering: [num_envs, chunk, action_dim] -> [chunk, num_envs, action_dim]
        chunk_actions_for_render = chunk_actions_for_render.transpose(1, 0, 2)
        chunk_rewards_for_render = chunk_rewards_for_render.T  # [chunk, num_envs]

        return (
            [extracted_obs],
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )

    def offload(self):
        """Move heavy models and runtime tensors to CPU."""
        if self._is_offloaded:
            return
        self.pipe.vae = self.pipe.vae.to("cpu")
        self.pipe.dit = self.pipe.dit.to("cpu")
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = recursive_to_device(self.current_obs, "cpu")
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        torch.cuda.empty_cache()
        self._is_offloaded = True

    def onload(self):
        """Move models and runtime tensors back to execution device."""
        if not self._is_offloaded:
            return
        self.pipe.dit = self.pipe.dit.to(self.device)
        self.pipe.vae = self.pipe.vae.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False

    def get_state(self) -> bytes:
        """Serialize runtime state to CPU bytes buffer for offload."""
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "task_descriptions": self.task_descriptions,
            "init_ee_poses": self.init_ee_poses,
            "elapsed_steps": self.elapsed_steps,
            "prev_step_reward": self.prev_step_reward.cpu(),
            "_is_start": self._is_start,
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
        }
        if self.record_metrics:
            env_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )

        # image_queue_state = []
        # for env_idx in range(self.num_envs):
        #     queue_frames = []
        #     for frame in self.image_queue[env_idx]:
        #         queue_frames.append(recursive_to_device(frame, "cpu"))
        #     image_queue_state.append(queue_frames)
        # env_state["image_queue"] = image_queue_state

        buffer = io.BytesIO()
        torch.save(env_state, buffer)
        return buffer.getvalue()


# PYTHONPATH="/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio:$PYTHONPATH" python -m rlinf.envs.world_model.world_model_wan_env
if __name__ == "__main__":
    from pathlib import Path

    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize_config_dir

    # # Set required environment variable
    os.environ.setdefault("EMBODIED_PATH", "examples/embodiment")

    repo_root = Path(__file__).resolve().parents[3]

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    config_dir = Path(
        os.environ.get("EMBODIED_CONFIG_DIR", repo_root / "examples/embodiment/config")
    ).resolve()
    config_name = "wan_libero_spatial_grpo_openvlaoft_quick"

    print(f"Loading config: {config_name} from {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]

    env = WanEnv(cfg, cfg.total_num_envs, seed_offset=0, total_num_processes=1)

    # Reset environment
    for i in range(20):
        obs, info = env.reset()

    print("\nAfter reset:")
    print(f"  obs keys: {list(obs.keys())}")

    # Test 1: chunk_steps = self.chunk
    print("\n" + "-" * 80)

    # chunk_steps = cfg.chunk
    chunk_steps = cfg.chunk
    num_envs = cfg.total_num_envs

    chunk_traj = 1
    zeros_actions = np.zeros((num_envs, chunk_steps, 7))

    for i in range(chunk_traj):
        print(f"Chunk {i} of {chunk_traj}")
        print("-" * 100)
        o, r, te, tr, infos = env.chunk_step(
            zeros_actions[:, i * chunk_steps : (i + 1) * chunk_steps, :]
        )
