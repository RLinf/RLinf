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

"""
This environment is used to evaluate the OpenSora  world model with the Video reward model.
"""

import io
import json
import os
from collections import deque
from contextlib import nullcontext
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import OmegaConf

# OpenSora imports
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.inference_utils import prepare_multi_resolution_info
from opensora.utils.misc import to_torch_dtype

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["OpenSoraEnv"]


class OpenSoraEnv(BaseWorldEnv):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )
        self.world_model_cfg = self.cfg.world_model_cfg
        self.inference_dtype = to_torch_dtype(self.world_model_cfg.get("dtype", "bf16"))
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

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
        self.chunk = self.world_model_cfg.chunk  # Ta
        self.condition_frame_length = self.world_model_cfg.condition_frame_length  # To
        self.num_frames = self.chunk + self.condition_frame_length

        self.image_size = tuple(self.world_model_cfg.image_size)

        # Load models
        self.vae = self._load_vae().eval().to(self.device, self.inference_dtype)
        self.model = self._load_model().eval().to(self.device, self.inference_dtype)
        self.scheduler = self._load_scheduler()

        # Load reward model if specified
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # Determine VAE type for frame calculations
        vae_type = self.world_model_cfg.vae.get("type", "OpenSoraVAE_V1_2")
        self.is_vae_v1_2 = vae_type == "OpenSoraVAE_V1_2"
        self.z_mask_frame_num = int(self.chunk / 4 if self.is_vae_v1_2 else self.chunk)
        self.z_condition_frame_length = int(
            self.condition_frame_length / 4
            if self.is_vae_v1_2
            else self.condition_frame_length
        )

        # Initialize state
        self.current_obs = None  # Will be a tensor [num_envs, 3, 1, t, h, w]
        self.task_descriptions = [""] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs

        # Image queue for condition frames (latent space)
        self.image_queue = [
            deque(maxlen=self.z_condition_frame_length) for _ in range(self.num_envs)
        ]

        # Action normalization stats
        self.action_stats = self._load_action_stats()

        # Initialize data preprocessing
        self.trans_resize = transforms.Compose(
            [
                transforms.Resize(self.image_size),
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
        self.fps = self.world_model_cfg.get("fps", 3.0)
        self.multi_resolution = self.world_model_cfg.get("multi_resolution", "STDiT2")

        # Prepare multi-resolution info
        self.model_args = prepare_multi_resolution_info(
            self.multi_resolution,
            1,
            self.image_size,
            self.num_frames,
            self.fps,
            self.device,
            self.inference_dtype,
        )
        self._is_offloaded = False

    def _build_dataset(self, cfg):
        return NpyTrajectoryDatasetWrapper(cfg.initial_image_path)

    def _load_vae(self):
        # Convert OmegaConf DictConfig to regular dict
        vae_cfg = OmegaConf.to_container(self.world_model_cfg.vae, resolve=True)
        vae = build_module(vae_cfg, MODELS)
        return vae

    def _load_model(self):
        # Get latent size from VAE
        input_size = (self.num_frames, *self.image_size)
        latent_size = self.vae.get_latent_size(input_size)

        # Convert OmegaConf DictConfig to regular dict
        model_cfg = OmegaConf.to_container(self.world_model_cfg.model, resolve=True)
        model = build_module(
            model_cfg,
            MODELS,
            input_size=latent_size,
            in_channels=self.vae.out_channels,
            enable_sequence_parallelism=False,
        )
        return model

    def _load_scheduler(self):
        # Convert OmegaConf DictConfig to regular dict
        scheduler_cfg = OmegaConf.to_container(
            self.world_model_cfg.scheduler, resolve=True
        )
        scheduler = build_module(scheduler_cfg, SCHEDULERS)
        return scheduler

    def _load_reward_model(self):
        rm_cfg = OmegaConf.to_container(self.world_model_cfg.reward_model, resolve=True)
        rew_model = build_module(rm_cfg, MODELS)

        return rew_model

    def _load_action_stats(self):
        """Load action normalization statistics"""
        stats_path = self.world_model_cfg.get("stats_path", None)
        if stats_path is not None and os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
                q01 = np.asarray(stats["action"]["q01"], np.float32)
                q99 = np.asarray(stats["action"]["q99"], np.float32)
            return {"q01": q01, "q99": q99}
        else:
            raise ValueError(f"Action stats path {stats_path} does not exist")

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

        image_queue_state = []
        for env_idx in range(self.num_envs):
            queue_frames = []
            for frame in self.image_queue[env_idx]:
                queue_frames.append(recursive_to_device(frame, "cpu"))
            image_queue_state.append(queue_frames)
        env_state["image_queue"] = image_queue_state

        buffer = io.BytesIO()
        torch.save(env_state, buffer)
        return buffer.getvalue()

    def load_state(self, state_buffer: bytes):
        """Restore runtime state from CPU bytes buffer."""
        buffer = io.BytesIO(state_buffer)
        state = torch.load(buffer, map_location="cpu", weights_only=False)

        self.current_obs = (
            recursive_to_device(state["current_obs"], self.device)
            if state["current_obs"] is not None
            else None
        )
        self.task_descriptions = state["task_descriptions"]
        self.init_ee_poses = state["init_ee_poses"]
        self.elapsed_steps = state["elapsed_steps"]
        self.prev_step_reward = state["prev_step_reward"].to(self.device)
        self._is_start = state["_is_start"]
        self.reset_state_ids = state["reset_state_ids"].to(self.device)
        self._generator.set_state(state["generator_state"])

        image_queue_state = state["image_queue"]
        for env_idx in range(self.num_envs):
            self.image_queue[env_idx].clear()
            for frame in image_queue_state[env_idx]:
                self.image_queue[env_idx].append(
                    recursive_to_device(frame, self.device)
                )

        if self.record_metrics and "success_once" in state:
            self.success_once = state["success_once"].to(self.device)
            self.returns = state["returns"].to(self.device)

    def offload(self):
        """Move heavy models and runtime tensors to CPU."""
        if self._is_offloaded:
            return
        self.vae = self.vae.to("cpu")
        self.model = self.model.to("cpu")
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = recursive_to_device(self.current_obs, "cpu")
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        for env_idx in range(self.num_envs):
            self.image_queue[env_idx] = deque(
                [
                    recursive_to_device(frame, "cpu")
                    for frame in self.image_queue[env_idx]
                ],
                maxlen=self.z_condition_frame_length,
            )
        self._clear_accelerator_cache()
        self._is_offloaded = True

    def onload(self):
        """Move models and runtime tensors back to execution device."""
        if not self._is_offloaded:
            return
        self.vae = self.vae.to(self.device, self.inference_dtype)
        self.model = self.model.to(self.device, self.inference_dtype)
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        for env_idx in range(self.num_envs):
            self.image_queue[env_idx] = deque(
                [
                    recursive_to_device(frame, self.device)
                    for frame in self.image_queue[env_idx]
                ],
                maxlen=self.z_condition_frame_length,
            )
        self._is_offloaded = False

    def _normalize_action(self, actions):
        """Normalize actions to [-1, 1] range"""
        if self.action_stats is not None:
            q01 = self.action_stats["q01"]
            q99 = self.action_stats["q99"]
            actions = 2 * ((actions - q01) / (q99 - q01)) - 1
        return actions

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

            # Repeat to fill condition frames: [3, H, W] -> [3, condition_frame_length, H, W]
            img_tensor = img_tensor.unsqueeze(1).repeat(
                1, self.condition_frame_length, 1, 1
            )  # [3, condition_frame_length, H, W]

            img_tensors.append(img_tensor)

        # Stack all environments: [num_envs, 3, condition_frame_length, H, W]
        stacked_imgs = torch.stack(img_tensors, dim=0).to(self.device)

        # Reshape to [num_envs, 3, 1, condition_frame_length, H, W] for compatibility
        self.current_obs = stacked_imgs.unsqueeze(2).to(self.device)
        # Shape: [num_envs, 3, 1, condition_frame_length, H, W]

        # Encode condition frames to latent space and fill image_queue
        images_for_encode = self.current_obs  # [num_envs, 3, 1, T, H, W]
        num_envs, c, v, t, h, w = images_for_encode.shape

        # Reshape for VAE encoding: [num_envs, 3, 1, T, H, W] -> [num_envs * T, 3, H, W]
        images_flat = images_for_encode.permute(0, 3, 1, 2, 4, 5).reshape(
            num_envs * t, c, h, w
        )
        # Convert to the same dtype as VAE model (matching inference_rlinf_libero_yaml.py)
        images_flat = images_flat.to(self.device).to(self.inference_dtype)
        # Encode to latent
        with torch.no_grad():
            z_encoded = self.vae.encode(
                images_flat.unsqueeze(2)
            )  # [num_envs * T, C, 1, H', W']

        # Reshape back and fill queues
        z_encoded = z_encoded.squeeze(2)  # [num_envs * T, C, H', W']
        z_encoded = z_encoded.reshape(
            num_envs, t, *z_encoded.shape[1:]
        )  # [num_envs, T, C, H', W']
        z_encoded = z_encoded.permute(0, 2, 1, 3, 4)  # [num_envs, C, T, H', W']

        # Fill image queues for each environment
        for env_idx in range(num_envs):
            self.image_queue[env_idx].clear()
            for t_idx in range(t):
                frame_latent = z_encoded[
                    env_idx : env_idx + 1, :, t_idx : t_idx + 1, :, :
                ]  # [1, C, 1, H', W']
                self.image_queue[env_idx].append(frame_latent)

        self._is_start = False
        self._reset_metrics()

        # Initialize action buffer (if needed)
        # For OpenSora, we might not need action_buffer in the same way as EvacEnv
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

        # Store task descriptions and init_ee_poses
        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses

        # Wrap observation to match libero_env format
        extracted_obs = self._wrap_obs()
        infos = {}

        return extracted_obs, infos

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError(
            "step in OpenSora Env is not impl, use chunk_step instead"
        )

    def _infer_next_chunk_rewards(self):
        """Predict next reward using the reward model"""
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        # Extract chunk observations
        num_envs, c, v, t, h, w = self.current_obs.shape
        extract_chunk_obs = self.current_obs.permute(
            0, 3, 1, 2, 4, 5
        )  # [num_envs, chunk + condition_frame_length, 3, v, h, w]

        if self.cfg.world_model_cfg.reward_model.type == "ResnetRM":
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
        else:
            raise ValueError(
                f"Unknown reward model type: {self.cfg.world_model_cfg.reward_model.type}"
            )

        return rewards

    def _infer_next_chunk_frames(self, actions):
        """Predict next frame chunk using the OpenSora model with batch processing"""
        num_envs = self.num_envs

        assert actions.shape[0] == self.num_envs, (
            f"Actions shape {actions.shape} does not match num_envs {self.num_envs}"
        )

        # Normalize actions
        actions_np = (
            actions if isinstance(actions, np.ndarray) else actions.cpu().numpy()
        )
        actions_normalized = self._normalize_action(actions_np)
        actions_tensor = (
            torch.from_numpy(actions_normalized)
            .to(self.device)
            .to(self.inference_dtype)
        )

        # Get latent size
        latent_size = self.vae.get_latent_size((self.num_frames, *self.image_size))

        # Collect condition frames from all environments and stack them
        # Each queue contains frames of shape [1, C, 1, H', W']
        mask_images_list = []
        for env_idx in range(num_envs):
            # Concatenate frames from queue: [1, C, T_cond, H', W']
            mask_images = torch.concat(list(self.image_queue[env_idx]), dim=2)
            mask_images_list.append(
                mask_images.squeeze(0)
            )  # Remove batch dim: [C, T_cond, H', W']

        # Stack all environments: [num_envs, C, T_cond, H', W']
        mask_images_batch = torch.stack(mask_images_list, dim=0)

        # Prepare actions for all environments: [num_envs, chunk, action_dim]
        actions_batch = actions_tensor.reshape(num_envs, -1, actions_tensor.shape[-1])

        # Create noise for masked frames for all environments: [num_envs, C, T_mask, H', W']
        z = torch.randn(
            num_envs,
            self.vae.out_channels,
            self.z_mask_frame_num,
            *latent_size[1:],
            device=self.device,
            dtype=self.inference_dtype,
        )

        # Concatenate condition and mask frames: [num_envs, C, T_cond + T_mask, H', W']
        z_full = torch.concat([mask_images_batch, z], dim=2)

        # Create mask for all environments: [num_envs, T_cond + T_mask]
        masks = torch.tensor(
            [[0] * self.z_condition_frame_length + [1] * self.z_mask_frame_num]
            * num_envs,
            device=self.device,
            dtype=self.inference_dtype,
        )

        # Prepare actions for model: [num_envs, chunk, action_dim]
        y = actions_batch.to(self.device).to(self.inference_dtype)

        # Sample using scheduler with batch processing
        samples = self.scheduler.sample(
            self.model,
            z=z_full,
            y=y,
            device=self.device,
            additional_args=self.model_args,
            progress=False,
            mask=masks,
        )

        # Extract only the generated frames (masked part): [num_envs, C, T_mask, H', W']
        pred_latents = samples[:, :, -self.z_mask_frame_num :, :, :].to(
            self.inference_dtype
        )

        # Update image queues before decoding (still need to process each environment separately)
        if self.is_vae_v1_2:
            # For VAE_V1_2, chunk into z_mask_frame_num parts
            for env_idx in range(num_envs):
                env_pred_latents = pred_latents[
                    env_idx : env_idx + 1
                ]  # [1, C, T_mask, H', W']
                for frame in env_pred_latents.clone().chunk(
                    self.z_mask_frame_num, dim=2
                ):
                    self.image_queue[env_idx].append(frame)

            # Decode with num_frames parameter: [num_envs, C, T_mask, H', W'] -> [num_envs, C, T, H, W]
            pred_images = self.vae.decode(pred_latents, num_frames=12)
        else:
            # For regular VAE, chunk into action_chunk_length parts
            for env_idx in range(num_envs):
                env_pred_latents = pred_latents[
                    env_idx : env_idx + 1
                ]  # [1, C, T_mask, H', W']
                for frame in env_pred_latents.clone().chunk(self.chunk, dim=2):
                    self.image_queue[env_idx].append(frame)

            # Decode: [num_envs, C, T_mask, H', W'] -> [num_envs, C, T, H, W]
            pred_images = self.vae.decode(pred_latents)

        # pred_images shape: [num_envs, C, T, H, W] where T depends on VAE type

        # Reshape to match current_obs format: [num_envs, C, 1, T, H, W]
        x_samples = pred_images.unsqueeze(2)

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
            # 4 + 8 / 5 + 8
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]

    def _wrap_obs(self):
        """Wrap observation to match libero_env format"""
        num_envs = self.num_envs

        # Extract the last frame (most recent observation) for each environment
        b, c, v, t, h, w = self.current_obs.shape
        assert b == num_envs, (
            f"Unexpected current_obs shape: {self.current_obs.shape}, expected {num_envs}"
        )

        last_frame = self.current_obs[:, :, 0, -1, :, :]  # [num_envs, 3, H, W]

        full_image = last_frame.permute(0, 2, 3, 1)  # [num_envs, H, W, 3]
        # Denormalize from [-1, 1] to [0, 255]
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255)

        # Resize to target size if needed
        if full_image.shape[1:3] != self.image_size:
            full_image = full_image.permute(0, 3, 1, 2)  # [num_envs, 3, H, W]
            full_image = F.interpolate(
                full_image, size=self.image_size, mode="bilinear", align_corners=False
            )
            full_image = full_image.permute(0, 2, 3, 1)  # [num_envs, H, W, 3]

        # Convert to uint8 tensor
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
        """Execute a chunk of actions"""
        self.onload()
        # policy_output_action: [num_envs, chunk, action_dim]

        autocast_context = (
            torch.amp.autocast(device_type=self.device.type, dtype=self.inference_dtype)
            if self.device.type in {"cuda", "musa"}
            else nullcontext()
        )
        with autocast_context:
            # Infer next chunk frames
            self._infer_next_chunk_frames(policy_output_action)

        # Update elapsed steps
        self.elapsed_steps += self.chunk

        extracted_obs = self._wrap_obs()
        obs_list = [extracted_obs]

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
            obs_list[-1], infos = self._handle_auto_reset(past_dones, obs_list[-1], {})
        else:
            infos = {}

        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_terminations, infos
        )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return (
            obs_list,
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )


# PYTHONPATH="/mnt/project_rlinf/jzn/workspace/opensora:$PYTHONPATH" python -m rlinf.envs.world_model.world_model_opensora_env
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
    config_name = "opensora_libero_spatial_grpo_openvlaoft_impl"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]

    num_envs = cfg.total_num_envs

    env = OpenSoraEnv(cfg, num_envs, seed_offset=0, total_num_processes=1)

    obs, info = env.reset()
    print("Reset OK. Keys:", list(obs.keys()))

    chunk_steps = cfg.world_model_cfg.chunk

    num_frames = chunk_steps
    chunk_traj = 1
    zeros_actions = np.zeros((num_envs, chunk_steps, 7))

    for i in range(chunk_traj):
        print(f"Chunk {i} of {chunk_traj}")
        print("-" * 100)
        o, r, te, tr, infos = env.chunk_step(
            zeros_actions[:, i * chunk_steps : (i + 1) * chunk_steps, :]
        )
