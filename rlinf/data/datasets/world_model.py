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


import glob
import os
from typing import Any, Callable, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class FuncRegistry:
    """A registry for functions."""

    def __init__(self):
        self._func_map: dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        """Register a function with a given name."""

        def decorator(func: Callable) -> Callable:
            self._func_map[name] = func
            return func

        return decorator

    def __getitem__(self, name: str) -> Callable:
        """Get a function by name."""
        return self._func_map[name]

    def keys(self) -> list:
        """Get all registered function names."""
        return list(self._func_map.keys())


FUNC_MAPPING = FuncRegistry()


@FUNC_MAPPING.register("first_frame")
def first_frame(**kwargs: Any) -> list[int]:
    """Return the index of the first frame."""
    return [0]


@FUNC_MAPPING.register("last_frame")
def last_frame(**kwargs: Any) -> list[int]:
    """Return the index of the last frame."""
    return [kwargs["episode_frame_idxs"][-1].item()]


@FUNC_MAPPING.register("closest_timestamp")
def closest_timestamp(**kwargs: Any) -> list[int]:
    """Return the index of the frame closest to the target timestamp."""
    target_timestamp = kwargs["target_timestamp"]
    closest_idx = torch.argmin(
        torch.abs(kwargs["episode_timestamps"] - target_timestamp)
    )
    return [closest_idx.item()]


@FUNC_MAPPING.register("first_n_frames")
def first_n_frames(**kwargs: Any) -> list[int]:
    """Return the indices of the first n frames."""
    n = kwargs["start_n_frames"]
    return list(range(n))


@FUNC_MAPPING.register("last_n_frames")
def last_n_frames(**kwargs: Any) -> list[int]:
    """Return the indices of the last n frames."""
    n = kwargs["target_n_frames"]
    return list(range(-n, 0))


class NpyTrajectoryDatasetWrapper(Dataset):
    """
    A wrapper for npy trajectory files to provide custom frame selection policies.
    Each npy file contains a trajectory (episode) with frames as dictionaries.

    Args:
        data_dir: The directory containing npy trajectory files.
        start_select_policy: The policy to select the start frames.
        target_select_policy: The policy to select the target frames.
        camera_names: List of camera names to use. If None, will use 'image' as default.
        target_timestamp: The target timestamp for the 'closest_timestamp' policy.
        start_n_frames: The number of start frames for the 'first_n_frames' policy.
        target_n_frames: The number of target frames for the 'last_n_frames' policy.
        camera_heights: Optional height to resize images.
        camera_widths: Optional width to resize images.
        action_key: Key to use for action dimension ('delta_action' or 'abs_action').
        state_key: Key to use for state ('init_ee_pose' or 'abs_action').
    """

    def __init__(
        self,
        data_dir: str,
        start_select_policy: str = "first_frame",
        target_select_policy: str = "last_frame",
        camera_names: Optional[list[str]] = None,
        target_timestamp: float = 10**4,
        start_n_frames: int = 1,
        target_n_frames: int = 1,
        camera_heights: Optional[int] = None,
        camera_widths: Optional[int] = None,
        action_key: str = "delta_action",
        state_key: str = "init_ee_pose",
    ):
        self.data_dir = data_dir
        self.action_key = action_key
        self.state_key = state_key

        # Find all npy files in the directory
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not npy_files:
            raise ValueError(f"No npy files found in {data_dir}")
        self.npy_files = npy_files

        # Load first file to infer data structure
        sample_data = np.load(npy_files[0], allow_pickle=True)
        sample_frame = sample_data[0]

        # Determine camera names
        if camera_names is None:
            # Default to 'image' if available, otherwise use first available image key
            if "image" in sample_frame:
                self.camera_names = ["image"]
            else:
                # Try to find image-like keys
                image_keys = [
                    k
                    for k in sample_frame.keys()
                    if isinstance(sample_frame[k], np.ndarray)
                    and len(sample_frame[k].shape) == 3
                ]
                if image_keys:
                    self.camera_names = [image_keys[0]]
                else:
                    raise ValueError("No image key found in trajectory data")
        else:
            self.camera_names = camera_names

        # Determine action dimension
        if action_key in sample_frame:
            self.action_dim = sample_frame[action_key].shape[0]
        elif "abs_action" in sample_frame:
            self.action_dim = sample_frame["abs_action"].shape[0]
        elif "delta_action" in sample_frame:
            self.action_dim = sample_frame["delta_action"].shape[0]
        else:
            raise ValueError(f"Action key '{action_key}' not found in trajectory data")

        # Set up image transforms
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        if camera_heights is not None and camera_widths is not None:
            self.image_transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((camera_heights, camera_widths)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.image_transforms = None

        # Validate and set policies
        assert start_select_policy in FUNC_MAPPING.keys(), (
            f"start_select_policy {start_select_policy} not in {FUNC_MAPPING.keys()}"
        )
        assert target_select_policy in FUNC_MAPPING.keys(), (
            f"target_select_policy {target_select_policy} not in {FUNC_MAPPING.keys()}"
        )
        self.start_select_policy = FUNC_MAPPING[start_select_policy]
        self.target_select_policy = FUNC_MAPPING[target_select_policy]
        self.target_timestamp = target_timestamp
        self.start_n_frames = start_n_frames
        self.target_n_frames = target_n_frames

        # Pre-compute timestamps for all episodes (using frame indices as timestamps)
        self._episode_lengths = []
        for npy_file in self.npy_files:
            data = np.load(npy_file, allow_pickle=True)
            self._episode_lengths.append(len(data))

    def __len__(self) -> int:
        """Return the total number of episodes (npy files)."""
        return len(self.npy_files)

    def _load_trajectory(self, index: int) -> np.ndarray:
        """Load a trajectory from an npy file."""
        return np.load(self.npy_files[index], allow_pickle=True)

    def _convert_frame_to_lerobot_format(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Convert a frame from npy format to LeRobot format."""
        converted = {}

        # Convert images
        for camera_name in self.camera_names:
            if camera_name in frame:
                image = frame[camera_name]
                # Convert numpy array to torch tensor
                if isinstance(image, np.ndarray):
                    # Ensure image is in HWC format
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        # Convert to CHW format for torch
                        image_tensor = (
                            torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                        )
                    else:
                        image_tensor = torch.from_numpy(image).float()

                    # Apply transforms if specified
                    if self.image_transforms is not None:
                        # Convert back to numpy for PIL
                        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(
                            np.uint8
                        )
                        image_tensor = self.image_transforms(image_np)

                    converted[camera_name] = image_tensor
                else:
                    converted[camera_name] = image
            else:
                # Try 'image' as fallback
                if "image" in frame:
                    image = frame["image"]
                    if isinstance(image, np.ndarray):
                        image_tensor = (
                            torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                        )
                        if self.image_transforms is not None:
                            image_np = (
                                image_tensor.permute(1, 2, 0).numpy() * 255
                            ).astype(np.uint8)
                            image_tensor = self.image_transforms(image_np)
                        converted[camera_name] = image_tensor

        # Convert state - prioritize init_ee_pose for initial frame state
        # First check for init_ee_pose directly (for initial frame)
        if "init_ee_pose" in frame:
            state = frame["init_ee_pose"]
            if isinstance(state, np.ndarray):
                converted["observation.state"] = torch.from_numpy(state).float()
            else:
                converted["observation.state"] = torch.tensor(
                    state, dtype=torch.float32
                )
        elif self.state_key in frame:
            state = frame[self.state_key]
            if isinstance(state, np.ndarray):
                converted["observation.state"] = torch.from_numpy(state).float()
            else:
                converted["observation.state"] = torch.tensor(
                    state, dtype=torch.float32
                )
        else:
            # Fallback: use abs_action if available
            if "abs_action" in frame:
                state = frame["abs_action"]
                if isinstance(state, np.ndarray):
                    converted["observation.state"] = torch.from_numpy(state).float()
                else:
                    converted["observation.state"] = torch.tensor(
                        state, dtype=torch.float32
                    )
            else:
                # Create empty state
                converted["observation.state"] = torch.zeros(1, dtype=torch.float32)

        # Convert task/instruction
        if "instruction" in frame:
            instruction = frame["instruction"]
            if isinstance(instruction, (bytes, np.bytes_)):
                instruction = instruction.decode("utf-8")
            converted["task"] = str(instruction)
        elif "task" in frame:
            task = frame["task"]
            if isinstance(task, (bytes, np.bytes_)):
                task = task.decode("utf-8")
            converted["task"] = str(task)
        else:
            raise ValueError(f"No instruction or task found in frame {frame}")

        return converted

    def _get_frame_indices(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get frame indices and timestamps for an episode."""
        num_frames = self._episode_lengths[index]
        # Use frame indices as timestamps (can be modified if actual timestamps are available)
        timestamps = np.arange(num_frames, dtype=np.float32)
        return torch.arange(num_frames), torch.from_numpy(timestamps)

    def _select_frames(
        self, policy: Callable, trajectory: np.ndarray, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Select frames from a trajectory using a policy."""
        indices = policy(**kwargs)
        start_index = kwargs["episode_frame_idxs"][0].item()
        selected_frames = []
        for idx in indices:
            frame_idx = int(start_index + idx)
            if 0 <= frame_idx < len(trajectory):
                frame = trajectory[frame_idx]
                converted_frame = self._convert_frame_to_lerobot_format(frame)
                selected_frames.append(converted_frame)
        return selected_frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get an item from the dataset.
        Args:
            index: The index of the episode (npy file).
        Returns:
            A dictionary containing the start items, target items, episode index, task, and dataset metadata.
        """
        trajectory = self._load_trajectory(index)
        episode_frame_idxs, episode_timestamps = self._get_frame_indices(index)

        policy_kwargs = {
            "episode_frame_idxs": episode_frame_idxs - episode_frame_idxs[0],
            "episode_timestamps": episode_timestamps.numpy(),
            "episode_reward": None,
            "target_timestamp": self.target_timestamp,
            "start_n_frames": self.start_n_frames,
            "target_n_frames": self.target_n_frames,
        }

        start_items = self._select_frames(
            self.start_select_policy, trajectory, **policy_kwargs
        )
        target_items = self._select_frames(
            self.target_select_policy, trajectory, **policy_kwargs
        )

        # Get task from first frame
        first_frame = self._convert_frame_to_lerobot_format(trajectory[0])
        task = first_frame.get("task", "")

        # Create simple metadata
        dataset_meta = {
            "episode_length": len(trajectory),
            "file_path": self.npy_files[index],
        }

        return {
            "start_items": start_items,
            "target_items": target_items,
            "episode_index": index,
            "task": task,
            "dataset_meta": dataset_meta,
        }
