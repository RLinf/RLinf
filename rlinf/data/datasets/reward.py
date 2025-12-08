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

"""Dataset for training reward classifier with trajectory-level data."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class RewardDataset(Dataset):
    """Dataset for training reward classifier with trajectory-level data.

    Expected data format:
    - Each .npy file contains a dictionary with:
      - 'images': Dict of numpy arrays, keyed by image_key, shape [T, C, H, W]
      - 'labels': numpy array of success_frame labels, shape [T]
    - positive_dir: Directory containing positive trajectory files (success_once=1)
    - negative_dir: Directory containing negative trajectory files (success_once=0)
    """

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        positive_dir: Optional[str] = None,
        negative_dir: Optional[str] = None,
        image_key: Optional[str] = None,
        image_size: Optional[list] = None,
        device: str = "cpu",
    ):
        """
        Args:
            cfg: Configuration dict (preferred). If provided, other args are ignored.
            positive_dir: Path to directory containing positive trajectory files
            negative_dir: Path to directory containing negative trajectory files
            image_key: Image key to use (e.g., "base_camera")
            image_size: Image size [C, H, W]
            device: Device to load data on
        """
        if cfg is not None:
            # Use config if provided
            self.positive_dir = Path(cfg.positive_dir)
            self.negative_dir = Path(cfg.negative_dir)
            self.image_key = cfg.image_key
            self.image_size = cfg.image_size
            self.device = getattr(cfg, "device", device)
        else:
            # Fallback to individual args
            if (
                positive_dir is None
                or negative_dir is None
                or image_key is None
                or image_size is None
            ):
                raise ValueError(
                    "Either cfg or all individual args (positive_dir, negative_dir, image_key, image_size) must be provided"
                )
            self.positive_dir = Path(positive_dir)
            self.negative_dir = Path(negative_dir)
            self.image_key = image_key
            self.image_size = image_size
            self.device = device

        if not self.positive_dir.exists():
            raise ValueError(f"Positive directory not found: {self.positive_dir}")
        if not self.negative_dir.exists():
            raise ValueError(f"Negative directory not found: {self.negative_dir}")

        # Collect all trajectory files from both directories
        self.samples = []

        # Statistics
        positive_traj_count = 0
        negative_traj_count = 0
        positive_dir_frame_count = 0  # Frames from positive_dir trajectories
        negative_dir_frame_count = 0  # Frames from negative_dir trajectories
        # Separate statistics for each directory
        positive_dir_label_1_count = 0  # Frames with label == 1 in positive_dir
        positive_dir_label_0_count = 0  # Frames with label == 0 in positive_dir
        negative_dir_label_1_count = 0  # Frames with label == 1 in negative_dir
        negative_dir_label_0_count = 0  # Frames with label == 0 in negative_dir

        # Load positive trajectories
        positive_files = sorted(self.positive_dir.glob("*.npy"))
        for traj_path in positive_files:
            traj_data = np.load(traj_path, allow_pickle=True).item()
            if (
                isinstance(traj_data, dict)
                and "images" in traj_data
                and "labels" in traj_data
            ):
                if self.image_key in traj_data["images"]:
                    # Each trajectory file contains multiple frames
                    num_frames = traj_data["images"][self.image_key].shape[0]
                    positive_dir_frame_count += num_frames
                    positive_traj_count += 1
                    for frame_idx in range(num_frames):
                        # Use frame's own label from success_frame (even in positive trajectories)
                        label = float(traj_data["labels"][frame_idx])
                        self.samples.append((traj_path, frame_idx, label))
                        if label >= 0.5:  # Treat >= 0.5 as positive
                            positive_dir_label_1_count += 1
                        else:
                            positive_dir_label_0_count += 1

        # Load negative trajectories
        negative_files = sorted(self.negative_dir.glob("*.npy"))
        for traj_path in negative_files:
            traj_data = np.load(traj_path, allow_pickle=True).item()
            if (
                isinstance(traj_data, dict)
                and "images" in traj_data
                and "labels" in traj_data
            ):
                if self.image_key in traj_data["images"]:
                    # Each trajectory file contains multiple frames
                    num_frames = traj_data["images"][self.image_key].shape[0]
                    negative_dir_frame_count += num_frames
                    negative_traj_count += 1
                    for frame_idx in range(num_frames):
                        # Use frame's own label from success_frame
                        label = float(traj_data["labels"][frame_idx])
                        self.samples.append((traj_path, frame_idx, label))
                        if label >= 0.5:  # Treat >= 0.5 as positive
                            negative_dir_label_1_count += 1
                        else:
                            negative_dir_label_0_count += 1

        self.num_samples = len(self.samples)

        # Calculate totals
        total_label_1_count = positive_dir_label_1_count + negative_dir_label_1_count
        total_label_0_count = positive_dir_label_0_count + negative_dir_label_0_count

        # Print statistics
        import sys

        print(f"\n{'=' * 60}", flush=True)
        print("Dataset Statistics:", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(
            f"Trajectories: {positive_traj_count} positive, {negative_traj_count} negative",
            flush=True,
        )
        print(f"\nFrom positive_dir ({self.positive_dir}):", flush=True)
        print(f"  Total frames: {positive_dir_frame_count}", flush=True)
        print(
            f"  Frames with label=1 (success): {positive_dir_label_1_count}", flush=True
        )
        print(
            f"  Frames with label=0 (failure): {positive_dir_label_0_count}", flush=True
        )
        if positive_dir_frame_count > 0:
            print(
                f"  Label ratio (1/0): {positive_dir_label_1_count}/{positive_dir_label_0_count} = {positive_dir_label_1_count / max(positive_dir_label_0_count, 1):.3f}",
                flush=True,
            )
        print(f"\nFrom negative_dir ({self.negative_dir}):", flush=True)
        print(f"  Total frames: {negative_dir_frame_count}", flush=True)
        print(
            f"  Frames with label=1 (success): {negative_dir_label_1_count}", flush=True
        )
        print(
            f"  Frames with label=0 (failure): {negative_dir_label_0_count}", flush=True
        )
        if negative_dir_frame_count > 0:
            print(
                f"  Label ratio (1/0): {negative_dir_label_1_count}/{negative_dir_label_0_count} = {negative_dir_label_1_count / max(negative_dir_label_0_count, 1):.3f}",
                flush=True,
            )
        print("\nOverall:", flush=True)
        print(f"  Total frames: {self.num_samples}", flush=True)
        print(
            f"  Total frames with label=1 (success): {total_label_1_count}", flush=True
        )
        print(
            f"  Total frames with label=0 (failure): {total_label_0_count}", flush=True
        )
        print(
            f"  Overall label ratio (1/0): {total_label_1_count}/{total_label_0_count} = {total_label_1_count / max(total_label_0_count, 1):.3f}",
            flush=True,
        )
        print(f"{'=' * 60}\n", flush=True)
        sys.stdout.flush()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Get a single sample (frame).

        Returns:
            images: Dict of image tensors, keyed by image_key, shape [C, H, W]
            label: Binary label tensor (0 or 1) from success_frame
        """
        traj_path, frame_idx, label = self.samples[idx]

        # Load trajectory data
        traj_data = np.load(traj_path, allow_pickle=True).item()

        if not isinstance(traj_data, dict) or "images" not in traj_data:
            raise ValueError(f"Invalid trajectory file format: {traj_path}")

        if self.image_key not in traj_data["images"]:
            raise ValueError(
                f"Image key {self.image_key} not found in trajectory file: {traj_path}"
            )

        # Extract frame image: [T, C, H, W] -> [C, H, W]
        frame_image = traj_data["images"][self.image_key][frame_idx]

        # Convert to torch tensor
        if isinstance(frame_image, np.ndarray):
            img = torch.from_numpy(frame_image).float()
        else:
            img = torch.tensor(frame_image, dtype=torch.float32)

        # Ensure correct shape [C, H, W]
        if img.dim() == 3:
            if img.shape[0] == self.image_size[0]:
                pass  # Already [C, H, W]
            else:
                img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        elif img.dim() == 4:
            img = img.squeeze(0)  # [1, C, H, W] -> [C, H, W]
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # Ensure correct size
        if img.shape != tuple(self.image_size):
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=self.image_size[1:], mode="bilinear"
            ).squeeze(0)

        images = {self.image_key: img}

        # Use label from success_frame (which may differ from trajectory label)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return images, label_tensor
