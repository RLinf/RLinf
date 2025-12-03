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

"""Data collector for reward model training data."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.io_struct import EnvOutput


class RewardDataCollector:
    """Collects frame-based reward training data.

    Collects images and success_frame labels, organizing them into positive
    (success_once=1) and negative (success_once=0) directories.
    """

    def __init__(
        self,
        cfg: DictConfig,
        positive_dir: str,
        negative_dir: str,
        image_keys: list[str],
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
        self.trajectory_buffers: dict[int, list[dict]] = {}
        # Key: env_idx (int), Value: success_once (bool or None)
        self.trajectory_success_once_dict: dict[int, Optional[bool]] = {}

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

    def _extract_success_once_list(
        self, env_info: dict, batch_size: int
    ) -> list[Optional[bool]]:
        """Extract success_once from env_info for each environment.

        Args:
            env_info: Environment info containing success_once
            batch_size: Number of parallel environments

        Returns:
            List of success_once values (one per environment)
        """
        success_once_list = [None] * batch_size
        if "success_once" not in env_info:
            return success_once_list

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

        return success_once_list

    def _extract_env_success_frame(self, success_frame, env_idx: int, batch_size: int):
        """Extract success_frame for a specific environment.

        Args:
            success_frame: Success frame tensor from env_output
            env_idx: Environment index
            batch_size: Number of parallel environments

        Returns:
            Extracted success_frame for this environment, or None if not available
        """
        if success_frame is None:
            return None

        if not isinstance(success_frame, torch.Tensor):
            return None

        if success_frame.dim() == 2:
            # [B, chunk_steps] - take this env's row
            if env_idx < success_frame.shape[0]:
                return success_frame[env_idx]
        elif success_frame.dim() == 1:
            # [B] - take this env's value
            if env_idx < success_frame.shape[0]:
                return success_frame[env_idx]

        return None

    def _check_has_success_frame(self, env_success_frame) -> bool:
        """Check if success_frame indicates success (>= 0.5).

        Args:
            env_success_frame: Success frame value for an environment

        Returns:
            True if success_frame >= 0.5, False otherwise
        """
        if env_success_frame is None:
            return False

        if isinstance(env_success_frame, torch.Tensor):
            if env_success_frame.dim() == 0:
                return env_success_frame.item() >= 0.5
            elif env_success_frame.dim() == 1:
                return (env_success_frame >= 0.5).any().item()
            else:
                return (env_success_frame >= 0.5).any().item()
        elif isinstance(env_success_frame, (int, float, np.number)):
            return float(env_success_frame) >= 0.5
        elif isinstance(env_success_frame, np.ndarray):
            return (env_success_frame >= 0.5).any()

        return False

    def _update_trajectory_success_once(
        self,
        env_idx: int,
        success_once: Optional[bool],
        env_success_frame,
        buffer_is_empty: bool,
    ):
        """Update trajectory success_once status for an environment.

        Args:
            env_idx: Environment index
            success_once: Success_once value from env_info
            env_success_frame: Success frame for this environment
            buffer_is_empty: Whether trajectory buffer is empty (new trajectory)
        """
        if success_once is None:
            return

        has_success_frame = self._check_has_success_frame(env_success_frame)

        if env_idx not in self.trajectory_success_once_dict:
            # New trajectory: validate success_once to avoid stale values
            if buffer_is_empty and success_once and not has_success_frame:
                # Stale success_once from previous trajectory - ignore it
                self.trajectory_success_once_dict[env_idx] = False
            else:
                self.trajectory_success_once_dict[env_idx] = success_once
        else:
            # Existing trajectory: accumulate success_once if validated
            if success_once and has_success_frame:
                self.trajectory_success_once_dict[env_idx] = (
                    self.trajectory_success_once_dict[env_idx] or True
                )

    def _check_env_done(self, dones, env_idx: int) -> bool:
        """Check if an environment's trajectory is done.

        Args:
            dones: Done tensor from env_output
            env_idx: Environment index

        Returns:
            True if environment is done, False otherwise
        """
        if dones is None:
            return False

        if not isinstance(dones, torch.Tensor):
            return False

        if dones.dim() == 2:
            # [B, chunk_steps] - check if last chunk step is done
            if env_idx < dones.shape[0]:
                return bool(dones[env_idx, -1].item())
        elif dones.dim() == 1:
            # [B] - check this env
            if env_idx < dones.shape[0]:
                return bool(dones[env_idx].item())

        return False

    def _process_env_step(
        self,
        env_idx: int,
        obs,
        env_success_frame,
        dones,
        success_once: Optional[bool],
        batch_size: int,
    ):
        """Process a single step for one environment.

        Args:
            env_idx: Environment index
            obs: Observations
            env_success_frame: Success frame for this environment
            dones: Done tensor
            success_once: Success_once value for this environment
            batch_size: Number of parallel environments
        """
        # Initialize buffer for this environment if needed
        if env_idx not in self.trajectory_buffers:
            self.trajectory_buffers[env_idx] = []
            if env_idx in self.trajectory_success_once_dict:
                del self.trajectory_success_once_dict[env_idx]

        buffer_is_empty = len(self.trajectory_buffers[env_idx]) == 0

        # Prepare step data
        step_data = {
            "obs": obs,
            "success_frame": env_success_frame,
            "env_idx": env_idx,
        }

        # Update trajectory success_once status
        self._update_trajectory_success_once(
            env_idx, success_once, env_success_frame, buffer_is_empty
        )

        # Check if environment is done
        is_env_done = self._check_env_done(dones, env_idx)
        step_data["is_done"] = is_env_done

        # Add step data to buffer
        self.trajectory_buffers[env_idx].append(step_data)

        # Save trajectory if done
        if is_env_done:
            buffer_len = len(self.trajectory_buffers[env_idx])
            if buffer_len <= 5:
                print(
                    f"WARNING: Saving trajectory with only {buffer_len} steps for env {env_idx}"
                )
            self.save_trajectory(env_idx)

    def _save_done_trajectories_at_epoch_end(self):
        """Save all done trajectories at epoch end."""
        for env_idx in list(self.trajectory_buffers.keys()):
            if len(self.trajectory_buffers[env_idx]) > 0:
                trajectory_buffer = self.trajectory_buffers[env_idx]
                last_step_done = trajectory_buffer[-1].get("is_done", False)
                if last_step_done:
                    buffer_len = len(trajectory_buffer)
                    if buffer_len <= 5:
                        print(
                            f"WARNING: Saving trajectory with only {buffer_len} steps for env {env_idx} (at epoch end)"
                        )
                    self.save_trajectory(env_idx)

    def add_env_output(
        self,
        env_output: EnvOutput,
        env_info: dict,
        is_last_step: bool = False,
    ):
        """Add environment output to the trajectory buffer(s).

        Args:
            env_output: Environment output containing obs and success_frame
            env_info: Environment info containing success_once
            is_last_step: Whether this is the last step of the trajectory
        """
        batch_size = self._get_batch_size(env_output)
        success_once_list = self._extract_success_once_list(env_info, batch_size)

        obs = env_output.obs
        success_frame = env_output.success_frame
        dones = env_output.dones

        # Process each environment in the batch
        for env_idx in range(batch_size):
            env_success_frame = self._extract_env_success_frame(
                success_frame, env_idx, batch_size
            )
            self._process_env_step(
                env_idx,
                obs,
                env_success_frame,
                dones,
                success_once_list[env_idx],
                batch_size,
            )

        # At epoch end, save all done trajectories
        if is_last_step:
            self._save_done_trajectories_at_epoch_end()

    def _save_all_done_trajectories(self):
        """Save all done trajectories."""
        for env_idx_to_save in list(self.trajectory_buffers.keys()):
            if len(self.trajectory_buffers[env_idx_to_save]) > 0:
                trajectory_buffer = self.trajectory_buffers[env_idx_to_save]
                if len(trajectory_buffer) > 0:
                    last_step_done = trajectory_buffer[-1].get("is_done", False)
                    if last_step_done:
                        self.save_trajectory(env_idx_to_save)

    def _check_trajectory_has_success(self, trajectory_buffer: list[dict]) -> bool:
        """Check if trajectory contains any success frames.

        Args:
            trajectory_buffer: List of step data dictionaries

        Returns:
            True if trajectory contains any success frames, False otherwise
        """
        for step_data in trajectory_buffer:
            success_frame = step_data.get("success_frame")
            if success_frame is not None:
                if isinstance(success_frame, torch.Tensor):
                    if success_frame.dim() == 0:
                        if success_frame.item() >= 0.5:
                            return True
                    else:
                        if (success_frame >= 0.5).any().item():
                            return True
                elif isinstance(success_frame, (int, float, np.number)):
                    if float(success_frame) >= 0.5:
                        return True
                elif isinstance(success_frame, np.ndarray):
                    if (success_frame >= 0.5).any():
                        return True
        return False

    def _determine_trajectory_label(
        self, env_idx: int, trajectory_buffer: list[dict]
    ) -> bool:
        """Determine if trajectory is positive (success) or negative.

        Args:
            env_idx: Environment index
            trajectory_buffer: List of step data dictionaries

        Returns:
            True if trajectory is positive, False otherwise
        """
        success_once = self.trajectory_success_once_dict.get(env_idx, None)
        has_success_frame = self._check_trajectory_has_success(trajectory_buffer)

        if success_once is None:
            return has_success_frame

        # Validate consistency between success_once and has_success_frame
        if bool(success_once) and not has_success_frame:
            # Stale success_once from previous trajectory - mark as negative
            print(
                f"WARNING: Trajectory for env {env_idx} has success_once=True but no frames with success_frame=1. "
                f"Marking as negative trajectory. Trajectory length: {len(trajectory_buffer)} steps."
            )
            return False
        elif not bool(success_once) and has_success_frame:
            # Trust success_frame over success_once
            print(
                f"WARNING: Trajectory for env {env_idx} has success_once=False but contains frames with success_frame=1. "
                f"Marking as positive trajectory based on success_frame."
            )
            return True
        else:
            return bool(success_once) or has_success_frame

    def _extract_step_labels(
        self, success_frame, step_idx: int, env_idx: int, is_positive: bool
    ) -> list[float]:
        """Extract labels from success_frame for a single step.

        Args:
            success_frame: Success frame value for this step
            step_idx: Step index
            env_idx: Environment index
            is_positive: Whether trajectory is positive

        Returns:
            List of label values for this step
        """
        step_labels = []

        if success_frame is None:
            label = 1.0 if is_positive else 0.0
            return [label]

        if isinstance(success_frame, torch.Tensor):
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
                label = 1.0 if is_positive else 0.0
                step_labels.append(label)
        elif isinstance(success_frame, (int, float, np.number)):
            step_labels.append(float(success_frame))
        else:
            label = 1.0 if is_positive else 0.0
            step_labels.append(label)

        return step_labels

    def _find_image_in_obs(self, obs: dict, key: str):
        """Find image in observation dictionary.

        Args:
            obs: Observation dictionary
            key: Image key to find

        Returns:
            Image tensor/array or None if not found
        """
        possible_keys = [f"images/{key}", key, f"obs/images/{key}"]

        for possible_key in possible_keys:
            if possible_key in obs:
                return obs[possible_key]

        # Try nested structure
        if "images" in obs:
            images_dict = obs["images"]
            if isinstance(images_dict, dict):
                if key in images_dict:
                    return images_dict[key]
                for possible_key_variant in [f"images/{key}", key]:
                    if possible_key_variant in images_dict:
                        return images_dict[possible_key_variant]
            elif isinstance(images_dict, (torch.Tensor, np.ndarray)):
                if len(self.image_keys) == 1:
                    return images_dict

        return None

    def _collect_trajectory_data(
        self, trajectory_buffer: list[dict], env_idx: int, is_positive: bool
    ) -> dict:
        """Collect all frames and labels for a trajectory.

        Args:
            trajectory_buffer: List of step data dictionaries
            env_idx: Environment index
            is_positive: Whether trajectory is positive

        Returns:
            Dictionary with 'trajectory_data' (images) and 'all_labels' (labels)
        """
        trajectory_data = {}
        all_labels = []

        for step_idx, step_data in enumerate(trajectory_buffer):
            obs = step_data["obs"]
            success_frame = step_data["success_frame"]
            step_env_idx = step_data.get("env_idx", env_idx)

            step_labels = self._extract_step_labels(
                success_frame, step_idx, env_idx, is_positive
            )
            all_labels.extend(step_labels)

            # Extract images for each image key
            for key in self.image_keys:
                image = self._find_image_in_obs(obs, key)
                if image is None:
                    continue

                image_np = self._extract_image(image, key_offset=step_env_idx)
                if image_np is None:
                    continue

                if key not in trajectory_data:
                    trajectory_data[key] = []

                # Handle chunk_steps: duplicate image if multiple labels
                for _ in step_labels:
                    trajectory_data[key].append(image_np)

        # Convert lists to numpy arrays: [T, C, H, W]
        for key in trajectory_data:
            if len(trajectory_data[key]) > 0:
                trajectory_data[key] = np.stack(trajectory_data[key], axis=0).astype(
                    np.float32
                )

        # Convert labels to numpy array: [T]
        all_labels = np.array(all_labels, dtype=np.float32)

        # Ensure labels match the number of frames
        if len(trajectory_data) > 0:
            first_key = list(trajectory_data.keys())[0]
            num_frames = trajectory_data[first_key].shape[0]

            if len(all_labels) != num_frames:
                if len(all_labels) < num_frames:
                    padding = np.full(
                        num_frames - len(all_labels),
                        all_labels[-1] if len(all_labels) > 0 else 0.0,
                    )
                    all_labels = np.concatenate([all_labels, padding])
                else:
                    all_labels = all_labels[:num_frames]

        return {"trajectory_data": trajectory_data, "all_labels": all_labels}

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
            self._save_all_done_trajectories()
            return

        if env_idx not in self.trajectory_buffers:
            return

        trajectory_buffer = self.trajectory_buffers[env_idx]
        if len(trajectory_buffer) == 0:
            return

        is_positive = self._determine_trajectory_label(env_idx, trajectory_buffer)

        # Check limits and set target directory
        if is_positive:
            if self.max_positive_trajectories is not None:
                if self.positive_count >= self.max_positive_trajectories:
                    self.reset(env_idx=env_idx)
                    return
            target_dir = self.positive_dir
            counter = self.positive_count
        else:
            if self.max_negative_trajectories is not None:
                if self.negative_count >= self.max_negative_trajectories:
                    self.reset(env_idx=env_idx)
                    return
            target_dir = self.negative_dir
            counter = self.negative_count

        # Collect trajectory data
        collected = self._collect_trajectory_data(
            trajectory_buffer, env_idx, is_positive
        )
        trajectory_data = collected["trajectory_data"]
        all_labels = collected["all_labels"]

        # Create trajectory data dictionary
        trajectory_dict = {
            "images": trajectory_data,
            "labels": all_labels,
        }

        # Save trajectory to single file
        filename = f"{counter:06d}.npy"
        filepath = target_dir / filename
        np.save(filepath, trajectory_dict, allow_pickle=True)

        # Log warnings for suspicious trajectories
        num_frames = all_labels.shape[0] if len(all_labels) > 0 else 0
        if num_frames <= 5:
            print(
                f"WARNING: Saved {'positive' if is_positive else 'negative'} trajectory {counter:06d} (env {env_idx}) "
                f"with only {num_frames} frames!"
            )

        # Update counter and reset buffer
        if is_positive:
            self.positive_count += 1
        else:
            self.negative_count += 1

        self.reset(env_idx=env_idx)

        if env_idx in self.trajectory_success_once_dict:
            del self.trajectory_success_once_dict[env_idx]

        # Periodic logging
        if (self.positive_count + self.negative_count) % 100 == 0:
            print(
                f"INFO: Collected {self.positive_count} positive and {self.negative_count} negative trajectories"
            )

        # Log warning if reaching limits
        if is_positive and self.max_positive_trajectories is not None:
            if self.positive_count >= self.max_positive_trajectories:
                print(
                    f"WARNING: Reached maximum positive trajectories limit: {self.max_positive_trajectories}"
                )
        elif not is_positive and self.max_negative_trajectories is not None:
            if self.negative_count >= self.max_negative_trajectories:
                print(
                    f"WARNING: Reached maximum negative trajectories limit: {self.max_negative_trajectories}"
                )

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
                print(
                    f"WARNING: Unexpected image shape after batch extraction: {image.shape}"
                )
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
                print(
                    f"WARNING: Unexpected image shape after batch extraction: {image_np.shape}"
                )
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
