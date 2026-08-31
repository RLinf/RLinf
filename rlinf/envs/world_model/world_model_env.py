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

"""Episode semantics for a world-model environment, shared by every backend.

A subclass says which :class:`~rlinf.envs.world_model.backend.WorldModelBackend` generates the frames
and which reward model scores them; everything else — sampling reset states, the condition window the
session starts from, rewards, terminations, auto reset, metrics, offload — lives here.
"""

from __future__ import annotations

import io
from abc import abstractmethod
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.backend import WorldModelBackend
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["WorldModelEnv"]

DEFAULT_ACTION_DIM = 7  # LIBERO


class WorldModelEnv(BaseWorldEnv):
    """A gym-style env whose dynamics come from a generative world model."""

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

        # Inference backend; it owns generation and the condition window, the env owns episode
        # semantics. Generation geometry comes from the model, so it is read back from the backend.
        self.backend: WorldModelBackend = self._build_backend()
        self.chunk = self.backend.chunk  # Ta
        self.condition_frame_length = self.backend.condition_frame_length  # To
        self.image_size = self.backend.image_size

        self.reward_model = self._load_reward_model().eval().to(self.device)

        # Initialize state
        # Will be a tensor [num_envs, 3, 1, T, h, w]
        self.current_obs = None
        self.task_descriptions = [""] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs

        self.reset_gripper_open = cfg.get("reset_gripper_open", True)
        self.is_libero_env = cfg.get("wm_env_type", "libero") == "libero"

        self.trans_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        self._is_offloaded = False

    @abstractmethod
    def _build_backend(self) -> WorldModelBackend:
        """Return the backend that generates frames for this world model."""

    @abstractmethod
    def _load_reward_model(self):
        """Return the reward model that scores generated frames."""

    def _reward_instructions(self) -> Optional[list[str]]:
        """Per-frame task instructions, for reward models that condition on the task."""
        return None

    def _build_dataset(self, cfg):
        return NpyTrajectoryDatasetWrapper(
            cfg.initial_image_path, enable_kir=self.enable_kir
        )

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
        episode_info["episode_len"] = self.elapsed_steps.to(torch.float32)
        # A slot restarted by auto reset sits at zero steps, so the divisor is clamped.
        episode_info["reward"] = episode_info["return"] / episode_info[
            "episode_len"
        ].clamp(min=1)
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
        """Estimate success (terminations) from the reward the world model predicts.

        Success is estimated when a reward in the chunk exceeds a threshold (default: 0.9).
        """
        success_threshold = getattr(self.cfg, "success_reward_threshold", 0.9)

        # chunk_rewards shape: [num_envs, chunk]
        max_reward_in_chunk = chunk_rewards.max(dim=1)[0]  # [num_envs]
        success_estimated = max_reward_in_chunk >= success_threshold

        return success_estimated.to(self.device)

    def update_reset_state_ids(self):
        """Updates the reset state IDs for environment initialization."""
        total_num_episodes = len(self.dataset)

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

    def _build_condition_window(self, episode_idx):
        """The initial condition window of one episode.

        Returns the condition frames as ``[3, condition_frame_length, H, W]`` in ``[-1, 1]``, the
        actions that led to them, the task description and the initial end-effector pose.
        """
        episode_data = self.dataset[episode_idx]

        # Get first frame from start_items
        if len(episode_data["start_items"]) == 0:
            raise ValueError(f"Empty start_items for episode {episode_idx}")

        first_frame = episode_data["start_items"][0]
        task_desc = str(episode_data.get("task", ""))

        if "image" not in first_frame:
            raise ValueError(f"No 'image' key in frame for episode {episode_idx}")

        # Get init_ee_pose if available
        if "observation.state" in first_frame:
            init_ee_pose = first_frame["observation.state"].numpy()
        else:
            init_ee_pose = None

        action_dim = getattr(self.cfg, "action_dim", DEFAULT_ACTION_DIM)

        # Repeat to fill condition frames: [3, H, W] -> [3, condition_frame_length, H, W]
        env_img_tensor = self._to_condition_frame(first_frame["image"]).unsqueeze(1)
        env_img_tensor = env_img_tensor.repeat(1, self.condition_frame_length, 1, 1)

        env_condition_action = np.zeros(
            (self.condition_frame_length, action_dim), dtype=np.float32
        )

        if self.reset_gripper_open and self.is_libero_env:
            env_condition_action[:, -1] = -1

        # KIR trick: use the last four frames as condition frames, while
        # keeping the reference frame unchanged as the first frame.
        target_items = episode_data.get("target_items", [])

        # first condition frame is the reference frame,
        # so the length of target_items should be condition_frame_length - 1
        if len(target_items) == self.condition_frame_length - 1:
            for target_idx, target_frame in enumerate(target_items):
                if "image" not in target_frame or "action" not in target_frame:
                    raise ValueError(
                        f"No 'image' or 'action' key in target frame for episode {episode_idx}"
                    )
                # keep first frame as reference frame, update the rest
                env_img_tensor[:, target_idx + 1] = self._to_condition_frame(
                    target_frame["image"]
                )
                env_condition_action[target_idx + 1] = target_frame["action"]

        return (
            env_img_tensor,
            torch.from_numpy(env_condition_action),
            task_desc,
            init_ee_pose,
        )

    def _restart_slots(self, env_idx, condition_windows):
        """Restart a subset of episodes in place, leaving the other slots running.

        The time axis of ``current_obs`` is a dimension of the whole tensor, so a restarted slot
        cannot be shorter than the others. It keeps the current length: the condition window goes
        to the tail, where ``_wrap_obs`` and the reward model read, and the reference frame fills
        the rest.
        """
        cfl = self.condition_frame_length
        num_frames = self.current_obs.shape[3]
        if num_frames < cfl:
            raise ValueError(
                f"current_obs has {num_frames} frames, fewer than the condition "
                f"window length {cfl}; cannot restart a subset of slots"
            )

        for slot, (env_img_tensor, _, task_desc, init_ee_pose) in zip(
            env_idx, condition_windows
        ):
            env_img_tensor = env_img_tensor.to(
                device=self.current_obs.device, dtype=self.current_obs.dtype
            )
            self.current_obs[slot, :, 0, num_frames - cfl :] = env_img_tensor
            self.current_obs[slot, :, 0, : num_frames - cfl] = env_img_tensor[:, :1]
            self.task_descriptions[slot] = task_desc
            self.init_ee_poses[slot] = init_ee_pose

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
        env_idx: Optional[Union[list[int], np.ndarray, torch.Tensor]] = None,
    ):
        self.onload()

        # Handle first reset with fixed reset state ids
        if self.is_start:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        # ``env_idx=None`` restarts every slot and rebuilds the observation tensor; a subset
        # restarts those slots in place and leaves the other episodes running.
        target_slots = (
            list(range(self.num_envs))
            if env_idx is None
            else [int(slot) for slot in env_idx]
        )
        num_slots = len(target_slots)

        if len(self.dataset) < self.num_envs:
            raise ValueError(
                f"Not enough episodes in dataset. Found {len(self.dataset)}, need {self.num_envs}"
            )

        # If episode_indices not provided, randomly select
        if episode_indices is None:
            # Set random seed if provided
            if seed is not None:
                if isinstance(seed, list):
                    np.random.seed(seed[0])
                else:
                    np.random.seed(seed)

            episode_indices = np.random.choice(
                len(self.dataset), size=num_slots, replace=False
            )
        else:
            # Convert to numpy if tensor
            if isinstance(episode_indices, torch.Tensor):
                episode_indices = episode_indices.cpu().numpy()
            if len(episode_indices) != num_slots:
                raise ValueError(
                    f"Got {len(episode_indices)} episode indices for {num_slots} env slots"
                )

        condition_windows = [
            self._build_condition_window(episode_idx) for episode_idx in episode_indices
        ]

        if env_idx is None:
            # Stack all environments: [num_envs, 3, condition_frame_length, H, W], then reshape to
            # [num_envs, 3, 1, condition_frame_length, H, W] for compatibility
            stacked_imgs = torch.stack(
                [window[0] for window in condition_windows], dim=0
            ).to(self.device)
            self.current_obs = stacked_imgs.unsqueeze(2)

            num_envs, c, v, t, h, w = self.current_obs.shape
            assert t == self.condition_frame_length, (
                f"Unexpected current_obs shape: {self.current_obs.shape}, expected {num_envs, c, v, self.condition_frame_length, h, w}"
            )

            self.task_descriptions = [window[2] for window in condition_windows]
            self.init_ee_poses = [window[3] for window in condition_windows]
        else:
            self._restart_slots(target_slots, condition_windows)

        # The condition window of each restarted slot, as [C, 1, H, W] frames; the backend keeps it
        # from here on. It sits at the tail of the time axis, which a subset restart does not shorten.
        num_frames = self.current_obs.shape[3]
        init_frames = [
            [
                self.current_obs[slot, :, 0, t_idx : t_idx + 1, :, :]
                for t_idx in range(num_frames - self.condition_frame_length, num_frames)
            ]
            for slot in target_slots
        ]

        # Noise is drawn from a single seed shared by the batch; per-trajectory seeds are future work.
        self.backend.close_session(target_slots)
        self.backend.open_session(
            env_ids=target_slots,
            init_frames=init_frames,
            init_actions=torch.stack(
                [window[1] for window in condition_windows], dim=0
            ).to(self.device),
            task_ids=list(episode_indices),
            seeds=[0] * num_slots,
        )

        self._reset_metrics(env_idx=None if env_idx is None else target_slots)

        # Wrap observation to match libero_env format
        extracted_obs = self._wrap_obs()
        infos = {}

        return extracted_obs, infos

    def _to_condition_frame(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """A dataset frame as a ``[3, H, W]`` tensor in ``[-1, 1]`` at the model's resolution."""
        if img_tensor.shape[1:] != self.image_size:
            img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
            img_tensor = F.interpolate(
                img_tensor,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
            img_tensor = img_tensor.squeeze(0)  # [3, H, W]
        return self.trans_norm(img_tensor)

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError(
            "step is not implemented for world-model envs, use chunk_step instead"
        )

    def _infer_next_chunk_rewards(self):
        """Predict the reward of the chunk just generated."""
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        num_envs, c, v, t, h, w = self.current_obs.shape
        # [num_envs, T, 3, v, h, w], then the chunk's own frames only
        chunk_obs = self.current_obs.permute(0, 3, 1, 2, 4, 5)[:, -self.chunk :]
        chunk_obs = (
            chunk_obs.reshape(self.num_envs * self.chunk, 3, v, h, w)
            .squeeze(2)  # [num_envs * chunk, 3, h, w]
            .to(self.device)
        )

        instructions = self._reward_instructions()
        if instructions is None:
            rewards = self.reward_model.predict_rew(chunk_obs)
        else:
            rewards = self.reward_model.predict_rew(chunk_obs, instructions)

        return rewards.reshape(self.num_envs, self.chunk)

    def _infer_next_chunk_frames(self, actions):
        """Advance the world model by one action chunk."""
        num_envs = self.num_envs
        assert actions.shape[0] == self.num_envs, (
            f"Actions shape {actions.shape} does not match num_envs {self.num_envs}"
        )

        # The new frames only, [num_envs, C, T, H, W] in [-1, 1]. T follows the model, not the chunk
        # length: a latent-space backend decodes whatever its VAE produces.
        videos = self.backend.generate(env_ids=range(num_envs), actions=actions)

        # Reshape to match current_obs format: [num_envs, C, 1, T, H, W]
        x_samples = videos.unsqueeze(2).to(self.device, dtype=self.current_obs.dtype)

        # Update current observation: append new generated frames to the time dimension
        self.current_obs = torch.cat([self.current_obs, x_samples], dim=3)

        # Keep a sliding window of the frames anything still reads: the reward model scores the last
        # chunk and the observation is the last frame.
        max_frames = self.condition_frame_length + self.chunk
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

        last_frame = self.current_obs[:, :, 0, -1, :, :]  # [num_envs, 3, H, W]

        full_image = last_frame.permute(0, 2, 3, 1)  # [num_envs, H, W, 3]
        # Denormalize from [-1, 1] to [0, 255]
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image.float(), 0, 255)

        # Resize to match libero_env format
        if full_image.shape[1:3] != self.image_size:
            full_image = full_image.permute(0, 3, 1, 2)  # [num_envs, 3, H, W]
            full_image = F.interpolate(
                full_image, size=self.image_size, mode="bilinear", align_corners=False
            )
            full_image = full_image.permute(0, 2, 3, 1)  # [num_envs, H, W, 3]

        # Convert to uint8 tensor (keep as tensor, not numpy)
        full_image = full_image.to(torch.uint8)

        # Get states (dummy for now, can be extended)
        states = torch.zeros((num_envs, 16), device=self.device, dtype=torch.float32)

        # Wrap observation - format aligned with libero_env
        obs = {
            "main_images": full_image,  # [num_envs, H, W, 3]
            "wrist_images": None,  # Not available in world model
            "states": states,  # [num_envs, 16]
            "task_descriptions": self.task_descriptions,  # list of strings
        }

        return obs

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        """Restart the episodes that ended, leaving the other slots running."""
        final_obs = extracted_obs
        final_info = infos

        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        extracted_obs, infos = self.reset(env_idx=env_idx.tolist())

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones

        return extracted_obs, infos

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        """Advance one action chunk: [num_envs, chunk, action_dim]."""
        self.onload()
        self._infer_next_chunk_frames(policy_output_action)

        # Update elapsed steps (incremented after inference)
        self._elapsed_steps += self.chunk

        # Read the last frame from self.current_obs
        extracted_obs = self._wrap_obs()

        chunk_rewards = self._infer_next_chunk_rewards()
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        estimated_success = self._estimate_success_from_rewards(chunk_rewards)

        # Create terminations tensor: success is marked at the last step of chunk
        raw_chunk_terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        raw_chunk_terminations[:, -1] = estimated_success

        raw_chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps

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
        self.backend.offload()
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = recursive_to_device(self.current_obs, "cpu")
        self.prev_step_reward = self.prev_step_reward.cpu()
        self._elapsed_steps = self._elapsed_steps.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        self._clear_accelerator_cache()
        self._is_offloaded = True

    def onload(self):
        """Move models and runtime tensors back to execution device."""
        if not self._is_offloaded:
            return
        self.backend.onload()
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self._elapsed_steps = self._elapsed_steps.to(self.device)
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
            "elapsed_steps": self.elapsed_steps.cpu(),
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

        buffer = io.BytesIO()
        torch.save(env_state, buffer)
        return buffer.getvalue()
