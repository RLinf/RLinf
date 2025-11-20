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

"""Data collector for reward model training data during SAC training."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.io_struct import EnvOutput


class RewardDataCollector:
    """Collects frame-based reward training data during SAC training.
    
    Collects images and success_frame labels, organizing them into positive
    (success_once=1) and negative (success_once=0) directories.
    """

    def __init__(
        self,
        cfg: DictConfig,
        positive_dir: str,
        negative_dir: str,
        image_keys: List[str],
        max_positive_trajectories: Optional[int] = None,
        max_negative_trajectories: Optional[int] = None,
    ):
        """
        Args:
            cfg: Configuration object
            positive_dir: Directory to save positive samples (success_once=1)
            negative_dir: Directory to save negative samples (success_once=0)
            image_keys: List of image keys to collect (e.g., ["base_camera"])
            max_positive_trajectories: Maximum number of positive trajectories to save (None = unlimited)
            max_negative_trajectories: Maximum number of negative trajectories to save (None = unlimited)
        """
        self.cfg = cfg
        self.positive_dir = Path(positive_dir)
        self.negative_dir = Path(negative_dir)
        self.image_keys = image_keys
        self.max_positive_trajectories = max_positive_trajectories
        self.max_negative_trajectories = max_negative_trajectories

        # Create directories
        self.positive_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)

        # Counters
        self.positive_count = 0
        self.negative_count = 0
        
        # Buffer to store trajectory data for each environment
        # Key: env_idx (int), Value: List of step data dicts
        self.trajectory_buffers: Dict[int, List[Dict]] = {}
        # Key: env_idx (int), Value: success_once (bool or None)
        self.trajectory_success_once_dict: Dict[int, Optional[bool]] = {}

    def reset(self, env_idx: Optional[int] = None):
        """Reset trajectory buffer(s).
        
        Args:
            env_idx: If provided, reset only this environment's buffer. Otherwise reset all.
        """
        if env_idx is not None:
            if env_idx in self.trajectory_buffers:
                del self.trajectory_buffers[env_idx]
            if env_idx in self.trajectory_success_once_dict:
                del self.trajectory_success_once_dict[env_idx]
        else:
            self.trajectory_buffers = {}
            self.trajectory_success_once_dict = {}

    def _get_batch_size(self, env_output: EnvOutput) -> int:
        """Get batch size from env_output.
        
        Args:
            env_output: Environment output
            
        Returns:
            Batch size (number of parallel environments)
        """
        obs = env_output.obs
        
        # Try to infer batch size from observations
        if "images" in obs:
            images = obs["images"]
            if isinstance(images, dict):
                # If images is a dict, check first image
                first_key = list(images.keys())[0]
                first_image = images[first_key]
                if isinstance(first_image, (torch.Tensor, np.ndarray)):
                    if first_image.ndim >= 4:  # [B, C, H, W] or [B, T, C, H, W]
                        return first_image.shape[0]
            elif isinstance(images, (torch.Tensor, np.ndarray)):
                if images.ndim >= 4:
                    return images.shape[0]
        
        # Try to infer from success_frame
        if env_output.success_frame is not None:
            if isinstance(env_output.success_frame, torch.Tensor):
                if env_output.success_frame.dim() >= 1:
                    return env_output.success_frame.shape[0]
        
        # Try to infer from dones
        if env_output.dones is not None:
            if isinstance(env_output.dones, torch.Tensor):
                if env_output.dones.dim() >= 1:
                    return env_output.dones.shape[0]
        
        # Default to 1 if cannot infer
        return 1

    def add_env_output(
        self,
        env_output: EnvOutput,
        env_info: Dict,
        is_last_step: bool = False,
    ):
        """Add environment output to the trajectory buffer(s).

        Args:
            env_output: Environment output containing obs and success_frame
            env_info: Environment info containing success_once
            is_last_step: Whether this is the last step of the trajectory
        """
        # Get batch size
        batch_size = self._get_batch_size(env_output)
        
        # Debug: print batch info (commented out for cleaner output)
        # if batch_size > 0:
        #     print(f"[RewardDataCollector] add_env_output: batch_size={batch_size}, is_last_step={is_last_step}")
        #     # Print buffer status for each env
        #     for env_idx in range(min(batch_size, 5)):  # Only print first 5 to avoid spam
        #         buffer_len = len(self.trajectory_buffers.get(env_idx, []))
        #         success_once_val = self.trajectory_success_once_dict.get(env_idx, None)
        #         print(f"  env {env_idx}: buffer_len={buffer_len}, success_once={success_once_val}")
        
        # Extract success_once from env_info (per environment)
        success_once_list = [None] * batch_size
        if "success_once" in env_info:
            success_once_val = env_info["success_once"]
            if isinstance(success_once_val, torch.Tensor):
                if success_once_val.dim() == 1:
                    # [B]
                    for i in range(min(batch_size, success_once_val.shape[0])):
                        success_once_list[i] = bool(success_once_val[i].item() > 0)
                elif success_once_val.dim() == 0:
                    # Scalar - apply to all envs
                    val = bool(success_once_val.item() > 0)
                    success_once_list = [val] * batch_size
            elif isinstance(success_once_val, (int, float, bool)):
                # Scalar - apply to all envs
                val = bool(success_once_val)
                success_once_list = [val] * batch_size
            elif isinstance(success_once_val, np.ndarray):
                if success_once_val.ndim == 1:
                    for i in range(min(batch_size, success_once_val.shape[0])):
                        success_once_list[i] = bool(success_once_val[i] > 0)
                else:
                    val = bool(success_once_val.max() > 0)
                    success_once_list = [val] * batch_size

        # Extract observations and success_frame
        obs = env_output.obs
        success_frame = env_output.success_frame
        dones = env_output.dones

        # Process each environment in the batch
        for env_idx in range(batch_size):
            # Initialize buffer for this environment if needed
            if env_idx not in self.trajectory_buffers:
                self.trajectory_buffers[env_idx] = []
                # New trajectory: clear any stale success_once from previous trajectory
                # This ensures each trajectory starts with a clean state
                if env_idx in self.trajectory_success_once_dict:
                    # Debug: clearing stale success_once for env {env_idx} at start of new trajectory
                    # print(f"Clearing stale success_once for env {env_idx} at start of new trajectory. Previous value: {self.trajectory_success_once_dict[env_idx]}")
                    del self.trajectory_success_once_dict[env_idx]
            
            # Extract success_frame for this environment
            env_success_frame = None
            if success_frame is not None and isinstance(success_frame, torch.Tensor):
                if success_frame.dim() == 2:
                    # [B, chunk_steps] - take this env's row
                    if env_idx < success_frame.shape[0]:
                        env_success_frame = success_frame[env_idx]
                    else:
                        print(
                            f"WARNING: env_idx {env_idx} >= success_frame.shape[0] {success_frame.shape[0]}, "
                            f"cannot extract success_frame. success_frame.shape={success_frame.shape}, "
                            f"env_idx={env_idx}, batch_size={batch_size}"
                        )
                elif success_frame.dim() == 1:
                    # [B] - take this env's value
                    if env_idx < success_frame.shape[0]:
                        env_success_frame = success_frame[env_idx]
                    else:
                        print(
                            f"WARNING: env_idx {env_idx} >= success_frame.shape[0] {success_frame.shape[0]}, "
                            f"cannot extract success_frame. success_frame.shape={success_frame.shape}, "
                            f"env_idx={env_idx}, batch_size={batch_size}"
                        )
                else:
                    print(
                        f"WARNING: Unexpected success_frame dimension: {success_frame.dim()}, "
                        f"shape: {success_frame.shape} for env_idx {env_idx}, "
                        f"batch_size={batch_size}"
                    )
            elif success_frame is None:
                # This can happen for the initial reset step, but should not happen for regular steps
                step_idx = len(self.trajectory_buffers.get(env_idx, []))
                print(
                    f"WARNING: success_frame is None for env_idx {env_idx} at step {step_idx}. "
                    f"env_output.success_frame={env_output.success_frame}, "
                    f"env_output.dones.shape={env_output.dones.shape if env_output.dones is not None else None}, "
                    f"env_output.rewards.shape={env_output.rewards.shape if env_output.rewards is not None else None}. "
                    f"This may indicate a problem with data collection or environment state."
                )
            
            # Prepare data for this step and environment
            step_data = {
                "obs": obs,
                "success_frame": env_success_frame,
                "env_idx": env_idx,
            }
            
            # Store success_once for this environment's trajectory
            # Check if this is the first step of a new trajectory (buffer is empty BEFORE adding current step)
            # If so, validate success_once to avoid stale values from previous trajectory
            buffer_is_empty = len(self.trajectory_buffers[env_idx]) == 0
            if success_once_list[env_idx] is not None:
                if env_idx not in self.trajectory_success_once_dict:
                    # New trajectory: validate success_once to avoid stale values
                    # If buffer is empty (true new trajectory) and success_once=True but no success_frame,
                    # this may indicate a stale success_once from previous trajectory
                    # Only trust success_once if we have a corresponding success_frame
                    if buffer_is_empty:
                        # Check if we have a success_frame for this step
                        has_success_frame = False
                        if env_success_frame is not None:
                            if isinstance(env_success_frame, torch.Tensor):
                                if env_success_frame.dim() == 0:
                                    has_success_frame = env_success_frame.item() >= 0.5
                                elif env_success_frame.dim() == 1:
                                    has_success_frame = (env_success_frame >= 0.5).any().item()
                            elif isinstance(env_success_frame, (int, float, np.number)):
                                has_success_frame = float(env_success_frame) >= 0.5
                            elif isinstance(env_success_frame, np.ndarray):
                                has_success_frame = (env_success_frame >= 0.5).any()
                        
                        # Only trust success_once if we have a corresponding success_frame
                        # If success_once=True but no success_frame, this is likely stale
                        if success_once_list[env_idx] and not has_success_frame:
                            print(
                                f"WARNING: Ignoring stale success_once=True for env {env_idx} at start of new trajectory. "
                                f"No success_frame found. This likely indicates stale success_once from previous trajectory."
                            )
                            # Don't set success_once if it's likely stale
                            self.trajectory_success_once_dict[env_idx] = False
                        else:
                            # Trust success_once if we have success_frame or if success_once=False
                            self.trajectory_success_once_dict[env_idx] = success_once_list[env_idx]
                    else:
                        # Not a new trajectory, initialize normally
                        self.trajectory_success_once_dict[env_idx] = success_once_list[env_idx]
                else:
                    # Existing trajectory: accumulate success_once (if any step succeeded, mark the whole trajectory as positive)
                    # However, we should validate success_once to avoid stale values
                    # If success_once=True but no success_frame for this step, it's likely stale -> don't accumulate
                    if success_once_list[env_idx]:
                        # Check if we have a success_frame for this step
                        has_success_frame = False
                        if env_success_frame is not None:
                            if isinstance(env_success_frame, torch.Tensor):
                                if env_success_frame.dim() == 0:
                                    has_success_frame = env_success_frame.item() >= 0.5
                                elif env_success_frame.dim() == 1:
                                    has_success_frame = (env_success_frame >= 0.5).any().item()
                            elif isinstance(env_success_frame, (int, float, np.number)):
                                has_success_frame = float(env_success_frame) >= 0.5
                            elif isinstance(env_success_frame, np.ndarray):
                                has_success_frame = (env_success_frame >= 0.5).any()
                        
                        # Only accumulate success_once if we have a corresponding success_frame
                        # If success_once=True but no success_frame, this is likely stale -> don't accumulate
                        if has_success_frame:
                            self.trajectory_success_once_dict[env_idx] = (
                                self.trajectory_success_once_dict[env_idx] or True
                            )
                        # else: don't accumulate stale success_once
                    else:
                        # success_once=False, no need to accumulate
                        pass
            
            # Check if this environment's trajectory is done BEFORE adding to buffer
            # This way we can save the complete trajectory including the last step
            is_env_done = False
            if dones is not None:
                if isinstance(dones, torch.Tensor):
                    if dones.dim() == 2:
                        # [B, chunk_steps] - check if last chunk step is done
                        if env_idx < dones.shape[0]:
                            is_env_done = dones[env_idx, -1].item()
                            # Debug: print dones info for first few envs (commented out for cleaner output)
                            # if env_idx < 3:
                            #     print(f"    env {env_idx}: dones shape={dones.shape}, dones[env_idx]={dones[env_idx]}, is_env_done={is_env_done}")
                    elif dones.dim() == 1:
                        # [B] - check this env
                        if env_idx < dones.shape[0]:
                            is_env_done = bool(dones[env_idx].item())
                            # Debug: print dones info (commented out for cleaner output)
                            # if env_idx < 3:
                            #     print(f"    env {env_idx}: dones shape={dones.shape}, dones[env_idx]={dones[env_idx]}, is_env_done={is_env_done}")
            
            # Add step data to buffer (including the done status)
            step_data["is_done"] = is_env_done
            self.trajectory_buffers[env_idx].append(step_data)
            
            # Debug: print buffer status for first few envs (commented out for cleaner output)
            # buffer_len_before = len(self.trajectory_buffers[env_idx])
            # buffer_len_after = len(self.trajectory_buffers[env_idx])
            # if env_idx < 3:
            #     print(f"    env {env_idx}: added step, buffer_len: {buffer_len_before} -> {buffer_len_after}, is_done={is_env_done}")
            
            # If this environment's trajectory is done, save it immediately
            # Save to positive if success_once=True, negative if success_once=False
            if is_env_done:
                # Save the trajectory for this env (will be saved to positive or negative based on success_once)
                buffer_len = len(self.trajectory_buffers[env_idx])
                # Debug: print saving info (commented out for cleaner output)
                # env_success_once = self.trajectory_success_once_dict.get(env_idx, None)
                # print(f"[RewardDataCollector] Saving trajectory for env {env_idx}: buffer_len={buffer_len}, success_once={env_success_once}, is_done=True")
                if buffer_len <= 5:
                    print(f"WARNING: Saving trajectory with only {buffer_len} steps for env {env_idx} (trajectory done)")
                self.save_trajectory(env_idx)
        
        # At epoch end (is_last_step), only save trajectories that are actually done
        # Don't save trajectories that are still in progress
        # Note: is_last_step can be True even if trajectories are not done (e.g., last chunk step of epoch)
        # So we only save if the trajectory is actually done (checked via is_done flag)
        if is_last_step:
            # Debug: print epoch end info (commented out for cleaner output)
            # print(f"[RewardDataCollector] is_last_step=True, checking {len(self.trajectory_buffers)} env buffers")
            for env_idx in list(self.trajectory_buffers.keys()):
                if len(self.trajectory_buffers[env_idx]) > 0:
                    # Check if this env's trajectory is actually done
                    trajectory_buffer = self.trajectory_buffers[env_idx]
                    if len(trajectory_buffer) > 0:
                        # Check the last step's is_done flag
                        last_step_done = trajectory_buffer[-1].get("is_done", False)
                        # Debug: print env status (commented out for cleaner output)
                        # env_success_once = self.trajectory_success_once_dict.get(env_idx, None)
                        # buffer_len = len(trajectory_buffer)
                        # print(f"  env {env_idx}: buffer_len={buffer_len}, last_step_done={last_step_done}, success_once={env_success_once}")
                        if last_step_done:
                            # Only save if trajectory is actually done
                            buffer_len = len(trajectory_buffer)
                            # Debug: print saving info (commented out for cleaner output)
                            # print(f"  Saving trajectory for env {env_idx} at epoch end: buffer_len={buffer_len}")
                            if buffer_len <= 5:
                                logger.warning(f"Saving trajectory with only {buffer_len} steps for env {env_idx} (at epoch end, trajectory done)")
                            self.save_trajectory(env_idx)
                        # If not done, keep the buffer for next epoch (trajectory continues)
                        # Debug: print keeping buffer info (commented out for cleaner output)
                        # else:
                        #     print(f"  Keeping buffer for env {env_idx} (trajectory not done, continues next epoch)")

    def save_trajectory(self, env_idx: Optional[int] = None):
        """Save trajectory(ies) to positive or negative directory.
        
        Each saved file contains:
        - 'images': Dict of numpy arrays, keyed by image_key, shape [T, C, H, W]
        - 'labels': numpy array of success_frame labels, shape [T]
        where T is the number of frames in the trajectory.
        
        Args:
            env_idx: Index of the environment whose trajectory to save. If None, save all remaining trajectories.
        """
        if env_idx is None:
            # Save all remaining trajectories that are actually done
            # Don't save trajectories that are still in progress
            # Debug: print save_trajectory(None) info (commented out for cleaner output)
            # print(f"[RewardDataCollector] save_trajectory(None) called, checking {len(self.trajectory_buffers)} env buffers")
            for env_idx_to_save in list(self.trajectory_buffers.keys()):
                if len(self.trajectory_buffers[env_idx_to_save]) > 0:
                    trajectory_buffer = self.trajectory_buffers[env_idx_to_save]
                    # Check if this trajectory is actually done
                    if len(trajectory_buffer) > 0:
                        last_step_done = trajectory_buffer[-1].get("is_done", False)
                        if last_step_done:
                            # Only save if trajectory is actually done
                            # Debug: print saving info (commented out for cleaner output)
                            # print(f"  Saving trajectory for env {env_idx_to_save} (done)")
                            self.save_trajectory(env_idx_to_save)
                        # Keep buffer if not done
                        # Debug: print keeping buffer info (commented out for cleaner output)
                        # else:
                        #     print(f"  Keeping buffer for env {env_idx_to_save} (not done)")
            return
        
        if env_idx not in self.trajectory_buffers:
            return
        
        trajectory_buffer = self.trajectory_buffers[env_idx]
        if len(trajectory_buffer) == 0:
            return

        # Determine if this is a positive or negative trajectory
        # Check both success_once and success_frame from all frames
        success_once = self.trajectory_success_once_dict.get(env_idx, None)
        has_success_frame = False
        
        # Check all frames in the trajectory for success_frame >= 0.5
        for step_data in trajectory_buffer:
            success_frame = step_data.get("success_frame")
            if success_frame is not None:
                if isinstance(success_frame, torch.Tensor):
                    # Check if any element in success_frame tensor is >= 0.5
                    if success_frame.dim() == 0:
                        # Scalar
                        if success_frame.item() >= 0.5:
                            has_success_frame = True
                            break
                    elif success_frame.dim() == 1:
                        # [chunk_steps]
                        if (success_frame >= 0.5).any().item():
                            has_success_frame = True
                            break
                    elif success_frame.dim() > 1:
                        # Multi-dimensional, check all elements
                        if (success_frame >= 0.5).any().item():
                            has_success_frame = True
                            break
                elif isinstance(success_frame, (int, float, np.number)):
                    if float(success_frame) >= 0.5:
                        has_success_frame = True
                        break
                elif isinstance(success_frame, np.ndarray):
                    if (success_frame >= 0.5).any():
                        has_success_frame = True
                        break
        
        # A trajectory is positive if:
        # 1. success_once is True AND any frame has success_frame >= 0.5, OR
        # 2. any frame has success_frame >= 0.5 (primary criterion)
        # If success_once=True but no success_frame=1, this likely indicates stale success_once from previous trajectory
        # So we mark it as negative to avoid false positives
        if success_once is None:
            # No success_once info - use success_frame check
            is_positive = has_success_frame
        else:
            # Primary: check if any frame has success_frame=1
            # Secondary: if success_once=True, it should match has_success_frame
            # If success_once=True but has_success_frame=False, this is likely stale -> mark as negative
            if bool(success_once) and not has_success_frame:
                # Stale success_once from previous trajectory - mark as negative
                print(
                    f"WARNING: Trajectory for env {env_idx} has success_once=True but no frames with success_frame=1. "
                    f"This likely indicates stale success_once from previous trajectory. "
                    f"Marking as negative trajectory. Trajectory length: {len(trajectory_buffer)} steps."
                )
                is_positive = False
            elif not bool(success_once) and has_success_frame:
                # success_frame indicates success but success_once=False - trust success_frame
                print(
                    f"WARNING: Trajectory for env {env_idx} has success_once=False but contains frames with success_frame=1. "
                    f"Marking as positive trajectory based on success_frame."
                )
                is_positive = True
            else:
                # Both match: success_once and has_success_frame are consistent
                is_positive = bool(success_once) or has_success_frame
        
        # Check limits and set target directory
        if is_positive:
            # Check if we've reached the maximum number of positive trajectories
            if self.max_positive_trajectories is not None:
                if self.positive_count >= self.max_positive_trajectories:
                    self.reset(env_idx=env_idx)
                    return
            target_dir = self.positive_dir
            counter = self.positive_count
        else:
            # Check if we've reached the maximum number of negative trajectories
            if self.max_negative_trajectories is not None:
                if self.negative_count >= self.max_negative_trajectories:
                    self.reset(env_idx=env_idx)
                    return
            target_dir = self.negative_dir
            counter = self.negative_count

        # Collect all frames for each image key
        trajectory_data = {}
        all_labels = []

        # Process each step in the trajectory
        for step_idx, step_data in enumerate(trajectory_buffer):
            obs = step_data["obs"]
            success_frame = step_data["success_frame"]
            step_env_idx = step_data.get("env_idx", env_idx)

            # Extract labels from success_frame
            step_labels = []
            
            if success_frame is not None and isinstance(success_frame, torch.Tensor):
                # success_frame can be [chunk_steps] (already extracted for this env) or scalar
                if success_frame.dim() == 1:
                    # [chunk_steps] - take all chunk steps
                    for chunk_idx in range(success_frame.shape[0]):
                        label = float(success_frame[chunk_idx].item())
                        step_labels.append(label)
                elif success_frame.dim() == 0:
                    # Scalar
                    label = float(success_frame.item())
                    step_labels.append(label)
                else:
                    # Unexpected dimension
                    print(
                        f"WARNING: Unexpected success_frame dimension: {success_frame.dim()}, "
                        f"shape: {success_frame.shape} at step {step_idx} for env {env_idx}. "
                        f"Using default trajectory label."
                    )
                    # Default to trajectory label
                    label = 1.0 if is_positive else 0.0
                    step_labels.append(label)
            elif success_frame is None:
                # Default to trajectory label if no success_frame
                # This should not happen if success_frame was correctly extracted
                print(
                    f"WARNING: Positive trajectory (env_idx={env_idx}) has success_frame=None at step {step_idx}. "
                    f"Buffer length: {len(trajectory_buffer)}, is_positive={is_positive}. "
                    f"Using default label {1.0 if is_positive else 0.0}. "
                    f"This may indicate missing success_frame data in the trajectory."
                )
                label = 1.0 if is_positive else 0.0
                step_labels.append(label)
            else:
                # success_frame is not None but also not a Tensor
                print(
                    f"WARNING: success_frame has unexpected type: {type(success_frame)} at step {step_idx} for env {env_idx}. "
                    f"Using default trajectory label."
                )
                label = 1.0 if is_positive else 0.0
                step_labels.append(label)

            all_labels.extend(step_labels)

            # Extract images for each image key
            for key in self.image_keys:
                # Try to find the image in obs
                image = None
                possible_keys = [
                    f"images/{key}",
                    key,
                    f"obs/images/{key}",
                ]

                for possible_key in possible_keys:
                    if possible_key in obs:
                        image = obs[possible_key]
                        break

                if image is None:
                    # Try to find in nested structure - images might be a dict keyed by image_key
                    if "images" in obs:
                        images_dict = obs["images"]
                        if isinstance(images_dict, dict):
                            # If images is a dict, try to find the key
                            if key in images_dict:
                                image = images_dict[key]
                            # Also try with possible key variations
                            for possible_key_variant in [f"images/{key}", key]:
                                if possible_key_variant in images_dict:
                                    image = images_dict[possible_key_variant]
                                    break
                        elif isinstance(images_dict, (torch.Tensor, np.ndarray)):
                            # If images is a tensor/array directly, use it (for single image case)
                            if len(self.image_keys) == 1:
                                image = images_dict
                            else:
                                print(f"WARNING: Multiple image keys but images is not a dict. Available: {list(obs.keys())}")
                                continue

                if image is None:
                    print(f"WARNING: Image key {key} not found in observation. Available keys: {list(obs.keys())}")
                    # Debug: print available image keys (commented out for cleaner output)
                    # if "images" in obs:
                    #     if isinstance(obs["images"], dict):
                    #         print(f"  Available image keys in obs['images']: {list(obs['images'].keys())}")
                    continue

                # Convert image to numpy and ensure [C, H, W] format
                # Handle batch dimension: take the specific env
                image_np = self._extract_image(image, key_offset=step_env_idx)
                if image_np is None:
                    continue

                # Initialize trajectory_data for this key if needed
                if key not in trajectory_data:
                    trajectory_data[key] = []

                # Handle chunk_steps: if multiple labels, save multiple frames
                if len(step_labels) > 1:
                    # For chunk_steps, we need to handle multiple frames per step
                    # This assumes image represents the state at each chunk step
                    # For now, we'll duplicate the image for each chunk step
                    for _ in step_labels:
                        trajectory_data[key].append(image_np)
                else:
                    trajectory_data[key].append(image_np)

        # Convert lists to numpy arrays: [T, C, H, W]
        for key in trajectory_data:
            if len(trajectory_data[key]) > 0:
                trajectory_data[key] = np.stack(trajectory_data[key], axis=0).astype(np.float32)

        # Convert labels to numpy array: [T]
        all_labels = np.array(all_labels, dtype=np.float32)

        # Ensure labels match the number of frames
        # If there's a mismatch, pad or truncate labels
        if len(trajectory_data) > 0:
            first_key = list(trajectory_data.keys())[0]
            num_frames = trajectory_data[first_key].shape[0]
            
            if len(all_labels) != num_frames:
                # Adjust labels to match frame count
                if len(all_labels) < num_frames:
                    # Pad with last label
                    padding = np.full(num_frames - len(all_labels), all_labels[-1] if len(all_labels) > 0 else (1.0 if is_positive else 0.0))
                    all_labels = np.concatenate([all_labels, padding])
                else:
                    # Truncate
                    all_labels = all_labels[:num_frames]

        # Create trajectory data dictionary
        trajectory_dict = {
            'images': trajectory_data,
            'labels': all_labels,
        }

        # Save trajectory to single file
        filename = f"{counter:06d}.npy"
        filepath = target_dir / filename
        np.save(filepath, trajectory_dict, allow_pickle=True)

        # Debug: print trajectory info (only print warnings and periodic summaries)
        num_frames = all_labels.shape[0] if len(all_labels) > 0 else 0
        success_frames = np.sum(all_labels == 1) if len(all_labels) > 0 else 0
        num_steps_in_buffer = len(trajectory_buffer)
        
        # Only log warnings for suspicious trajectories
        if num_frames <= 5:
            print(
                f"WARNING: Saved {'positive' if is_positive else 'negative'} trajectory {counter:06d} (env {env_idx}) with only {num_frames} frames! "
                f"Buffer had {num_steps_in_buffer} steps, success_frames={success_frames}"
            )
        
        # Debug: print all saved trajectories (commented out for cleaner output)
        # print(
        #     f"[RewardDataCollector] Saved {'positive' if is_positive else 'negative'} trajectory {counter:06d} (env {env_idx}): "
        #     f"num_steps_in_buffer={num_steps_in_buffer}, num_frames={num_frames}, success_frames={success_frames}"
        # )

        # Update counter and reset this environment's buffer
        if is_positive:
            self.positive_count += 1
        else:
            self.negative_count += 1
        # Reset this environment's buffer and clear success_once
        # This ensures the next trajectory starts with a clean state
        self.reset(env_idx=env_idx)
        # Additionally, clear success_once_dict if it exists (should already be cleared by reset)
        # This is a safety measure to ensure no stale values persist
        if env_idx in self.trajectory_success_once_dict:
            print(f"WARNING: Found stale success_once for env {env_idx} after reset. Clearing it.")
            del self.trajectory_success_once_dict[env_idx]

        if (self.positive_count + self.negative_count) % 100 == 0:
            print(
                f"INFO: Collected {self.positive_count} positive and {self.negative_count} negative trajectories"
            )
        
        # Log warning if reaching limits
        if is_positive and self.max_positive_trajectories is not None:
            if self.positive_count >= self.max_positive_trajectories:
                print(f"WARNING: Reached maximum positive trajectories limit: {self.max_positive_trajectories}")
        elif not is_positive and self.max_negative_trajectories is not None:
            if self.negative_count >= self.max_negative_trajectories:
                print(f"WARNING: Reached maximum negative trajectories limit: {self.max_negative_trajectories}")

    def _extract_image(self, image, key_offset=0):
        """Extract and convert image to numpy array in [C, H, W] format.
        
        Args:
            image: Image tensor or array
            key_offset: Offset for batch dimension
            
        Returns:
            numpy array in [C, H, W] format, or None if extraction fails
        """
        if isinstance(image, torch.Tensor):
            # Handle batch dimension
            if image.dim() == 4:
                # [B, C, H, W] or [B, H, W, C], take item at key_offset
                if key_offset < image.shape[0]:
                    image = image[key_offset]
                else:
                    image = image[0]  # Fallback to first item
            
            # Now image is 3D: [C, H, W] or [H, W, C]
            if image.dim() == 3:
                # Check if it's [H, W, C] by looking at last dimension
                if image.shape[-1] == 3 or image.shape[-1] == 1:
                    # [H, W, C] format, convert to [C, H, W]
                    image = image.permute(2, 0, 1)
                # Otherwise assume it's already [C, H, W]
            elif image.dim() != 3:
                print(f"WARNING: Unexpected image shape after batch extraction: {image.shape}")
                return None
            
            image_np = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            image_np = image.copy()  # Make a copy to avoid modifying original
            # Ensure correct format
            if image_np.ndim == 4:
                # [B, C, H, W] or [B, H, W, C]
                if key_offset < image_np.shape[0]:
                    image_np = image_np[key_offset]
                else:
                    image_np = image_np[0]
            
            # Now image_np is 3D: [C, H, W] or [H, W, C]
            if image_np.ndim == 3:
                # Check if it's [H, W, C] by looking at last dimension
                if image_np.shape[-1] == 3 or image_np.shape[-1] == 1:
                    # [H, W, C] format, convert to [C, H, W]
                    image_np = np.transpose(image_np, (2, 0, 1))
                # Otherwise assume it's already [C, H, W]
            else:
                logger.warning(f"Unexpected image shape after batch extraction: {image_np.shape}")
                return None
        else:
            print(f"WARNING: Unsupported image type: {type(image)}")
            return None

        # Final check: ensure shape is [C, H, W]
        if image_np.ndim != 3:
            print(f"WARNING: Final image shape is not 3D: {image_np.shape}")
            return None
        
        # Verify it's [C, H, W] format
        if image_np.shape[0] not in [1, 3]:
            print(f"WARNING: Image channel dimension is not 1 or 3: {image_np.shape}")
            # Try to fix: if it's [H, W, C], transpose again
            if image_np.shape[-1] in [1, 3]:
                image_np = np.transpose(image_np, (2, 0, 1))

        return image_np.astype(np.float32)



def create_reward_data_collector(cfg: DictConfig) -> Optional[RewardDataCollector]:
    """Create a reward data collector if enabled in config.

    Args:
        cfg: Configuration object

    Returns:
        RewardDataCollector instance or None if not enabled
    """
    if not cfg.get("reward", {}).get("collect_data", False):
        return None

    reward_cfg = cfg.reward.get("data_collection", {})
    positive_dir = reward_cfg.get("positive_dir", "./reward_data/positive")
    negative_dir = reward_cfg.get("negative_dir", "./reward_data/negative")
    # Get image_keys from data_collection config, fallback to actor.model.image_keys
    image_keys = reward_cfg.get("image_keys", None)
    if image_keys is None:
        image_keys = cfg.actor.model.image_keys
    
    # Get trajectory limits
    max_positive_trajectories = reward_cfg.get("max_positive_trajectories", None)
    max_negative_trajectories = reward_cfg.get("max_negative_trajectories", None)

    return RewardDataCollector(
        cfg=cfg,
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        image_keys=image_keys,
        max_positive_trajectories=max_positive_trajectories,
        max_negative_trajectories=max_negative_trajectories,
    )

