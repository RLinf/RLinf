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


from typing import Any, Callable, Optional
import os
import torch
import pdb
import torchvision.transforms as transforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.data.schema import EmbodimentTag


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


class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """
    A wrapper for the LeRobotDataset to provide custom frame selection policies.

    Args:
        repo_id: The repository ID of the dataset on the Hugging Face Hub.
        root: The root directory where the dataset is stored.
        start_select_policy: The policy to select the start frames.
        target_select_policy: The policy to select the target frames.
        target_timestamp: The target timestamp for the 'closest_timestamp' policy.
        start_n_frames: The number of start frames for the 'first_n_frames' policy.
        target_n_frames: The number of target frames for the 'last_n_frames' policy.
    """

    def __init__(
        self,
        repo_id: str,
        root: str,
        start_select_policy: str,
        target_select_policy: str,
        camera_names: list[str],
        target_timestamp: float = 10**4,
        start_n_frames: int = 1,
        target_n_frames: int = 1,
        revision: str = 'v2.0',
        camera_heights: Optional[int] = None,
        camera_widths: Optional[int] = None,
        episodes: list[int] | None = None,
        episodes_list: str | None = None,
    ):
        if camera_heights is None or camera_widths is None:
            image_transforms = None
        else:
            image_transforms = transforms.Compose(
                [
                    transforms.Resize((camera_heights, camera_widths)),
                    transforms.Lambda(
                        lambda img: (img * 255).byte()
                        if img.dtype == torch.float32
                        else img
                    ),
                ]
            )
        self.select_episodes = None
        if episodes_list is not None and os.path.exists(episodes_list):
            with open(episodes_list, 'r')as f:
                self.select_episodes = [int(line.strip())
                                        for line in f if line.strip()]
        self._lerobot_dataset = LeRobotDataset(
            repo_id,  root=root, revision=revision, image_transforms=image_transforms,
            force_cache_sync=False,
            video_backend='pyav', episodes=episodes
        )
        self.timestamps = torch.stack(
            [i for i in self._lerobot_dataset.hf_dataset["timestamp"]]
        ).numpy()
        self.action_dim = self._lerobot_dataset.features["action"]["shape"][0]
        for camera_name in camera_names:
            assert camera_name in self._lerobot_dataset.meta.camera_keys
        self.camera_names = camera_names
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths

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

    def __len__(self) -> int:
        """Return the total number of episodes."""
        return self._lerobot_dataset.meta.total_episodes

    def _get_frame_indices(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_index = self._lerobot_dataset.episode_data_index["from"][index]
        end_index = self._lerobot_dataset.episode_data_index["to"][index]
        return torch.arange(start_index, end_index), self.timestamps[
            start_index:end_index
        ]

    def _select_frames(self, policy: Callable, **kwargs: Any) -> list[dict[str, Any]]:
        indices = policy(**kwargs)
        start_index = kwargs["episode_frame_idxs"][0]
        return [self._lerobot_dataset[int(start_index + idx)] for idx in indices]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get an item from the dataset.

        Args:
            index: The index of the episode.

        Returns:
            A dictionary containing the start items, target items, episode index, task, and dataset metadata.
        """
        episode_frame_idxs, episode_timestamps = self._get_frame_indices(index)

        policy_kwargs = {
            "episode_frame_idxs": episode_frame_idxs - episode_frame_idxs[0],
            "episode_timestamps": episode_timestamps,
            "episode_reward": None,
            "target_timestamp": self.target_timestamp,
            "start_n_frames": self.start_n_frames,
            "target_n_frames": self.target_n_frames,
        }

        start_items = self._select_frames(
            self.start_select_policy, **policy_kwargs)
        target_items = self._select_frames(
            self.target_select_policy, **policy_kwargs)

        return {
            "start_items": start_items,
            "target_items": target_items,
            "episode_index": index,
            "task": self._lerobot_dataset[episode_frame_idxs[0].item()]["task"],
            "dataset_meta": self._lerobot_dataset.meta.episodes_stats[index],
        }


class LeRobotFirstFrameDataset(torch.utils.data.Dataset):
    """
    A simplified dataset that loads only pre-extracted first frames from images.

    This is a drop-in replacement for LeRobotDatasetWrapper when only the first
    frame of each episode is needed. It loads pre-extracted images from the
    first_image/ directory instead of decoding MP4 videos, resulting in
    significantly faster loading times and lower memory usage.

    Args:
        repo_id: The repository ID of the dataset on the Hugging Face Hub.
        root: The root directory where the dataset is stored.
        camera_names: List of camera keys to load (e.g., ['observation.images.image_0']).
        camera_heights: Target height for resizing images (optional).
        camera_widths: Target width for resizing images (optional).
        episodes: List of episode indices to load (optional, loads all if None).
        episodes_list: Path to file containing episode indices (optional).
    """

    def __init__(
        self,
        repo_id: str,
        root: str,
        camera_names: list[str],
        camera_heights: Optional[int] = None,
        camera_widths: Optional[int] = None,
        episodes: list[int] | None = None,
        episodes_list: str | None = None,
    ):
        from pathlib import Path
        import json
        from PIL import Image

        self.root = Path(root)
        self.repo_id = repo_id
        self.camera_names = camera_names
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths

        # Setup image transforms
        if camera_heights is None or camera_widths is None:
            self.image_transforms = None
        else:
            self.image_transforms = transforms.Compose([
                transforms.Resize((camera_heights, camera_widths)),
                transforms.Lambda(
                    lambda img: (img * 255).byte()
                    if img.dtype == torch.float32
                    else img
                ),
            ])

        # Load dataset metadata
        meta_dir = self.root / "meta"
        if not meta_dir.exists():
            raise ValueError(f"Dataset metadata directory not found: {meta_dir}")

        # Load info.json to get dataset info
        info_path = meta_dir / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.info = json.load(f)
        else:
            self.info = {}

        # Load parquet files to get episode information
        import pandas as pd
        data_dir = self.root / "data"

        if not data_dir.exists():
            raise ValueError(f"Dataset data directory not found: {data_dir}")

        # Find all parquet files
        parquet_files = sorted(data_dir.rglob("*.parquet"))

        if not parquet_files:
            raise ValueError(f"No parquet files found in {data_dir}")

        # Load all parquet files to build episode index
        all_data = []
        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)
            all_data.append(df)

        # Combine all data
        self.df = pd.concat(all_data, ignore_index=True)

        # Build episode data index
        if "episode_index" not in self.df.columns:
            raise ValueError("Episode index column not found in parquet files")

        # Get unique episodes and sort them
        self.episodes = sorted(self.df["episode_index"].unique())
        self.total_episodes = len(self.episodes)

        # Filter episodes if specified
        self.select_episodes = None
        if episodes is not None:
            self.select_episodes = episodes
            self.episodes = [ep for ep in self.episodes if ep in episodes]
            self.total_episodes = len(self.episodes)
        elif episodes_list is not None and Path(episodes_list).exists():
            with open(episodes_list, 'r') as f:
                self.select_episodes = [int(line.strip()) for line in f if line.strip()]
            self.episodes = [ep for ep in self.episodes if ep in self.select_episodes]
            self.total_episodes = len(self.episodes)

        # Build episode data index for fast lookup
        self.episode_data_index = {}
        for ep_idx in self.episodes:
            ep_mask = self.df["episode_index"] == ep_idx
            ep_data = self.df[ep_mask]
            start_idx = ep_data.index[0]
            end_idx = ep_data.index[-1] + 1
            self.episode_data_index[ep_idx] = (start_idx, end_idx)

        # Verify first_image directory exists
        self.first_image_dir = self.root / "first_image"
        if not self.first_image_dir.exists():
            raise ValueError(
                f"First image directory not found: {self.first_image_dir}\n"
                f"Please run the preprocessing script first:\n"
                f"  python toolkits/world_model/extract_first_frames.py --root {root}"
            )

        # Build mapping from episode_index to image paths
        self._build_image_mapping()

        # Get action dimension from dataframe
        if "action" in self.df.columns:
            self.action_dim = len(self.df["action"].iloc[0])
        else:
            self.action_dim = None

        # Store timestamps
        if "timestamp" in self.df.columns:
            self.timestamps = self.df["timestamp"].values
        else:
            self.timestamps = None

        self.Image = Image  # Store for use in __getitem__

    def _build_image_mapping(self):
        """Build mapping from episode_index to first frame image paths."""
        from pathlib import Path

        self.image_paths = {}

        # Find all chunk directories in first_image
        chunk_dirs = sorted([
            d for d in self.first_image_dir.iterdir()
            if d.is_dir() and d.name.startswith("chunk-")
        ])

        for chunk_dir in chunk_dirs:
            chunk_name = chunk_dir.name

            # Find all camera directories
            camera_dirs = [d for d in chunk_dir.iterdir() if d.is_dir()]

            for camera_dir in camera_dirs:
                camera_name = camera_dir.name

                if camera_name not in self.camera_names:
                    continue

                # Find all image files
                for img_file in camera_dir.iterdir():
                    if img_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        # Extract episode index from filename
                        # Format: episode_XXXXXX.png
                        episode_str = img_file.stem.replace("episode_", "")
                        try:
                            episode_idx = int(episode_str)

                            if episode_idx not in self.image_paths:
                                self.image_paths[episode_idx] = {}

                            self.image_paths[episode_idx][camera_name] = str(img_file)
                        except ValueError:
                            # Skip files that don't match episode_XXXXXX pattern
                            pass

    def __len__(self) -> int:
        """Return the total number of episodes."""
        return self.total_episodes

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get an item from the dataset.

        Args:
            index: The index of the episode (0 to total_episodes-1).

        Returns:
            A dictionary containing:
                - start_items: List with single frame dict containing images
                - target_items: List with single frame dict containing images
                - episode_index: Episode index
                - task: Task string
                - dataset_meta: Episode statistics
        """
        # Get actual episode index
        episode_idx = self.episodes[index]

        # Get episode data indices
        if episode_idx not in self.episode_data_index:
            raise ValueError(f"Episode {episode_idx} not found in episode data index")

        start_idx, end_idx = self.episode_data_index[episode_idx]
        episode_data = self.df.iloc[start_idx:end_idx]

        # Load first frame images for each camera
        frame_dict = {}

        for camera_name in self.camera_names:
            if camera_name not in self.image_paths.get(episode_idx, {}):
                raise ValueError(
                    f"Image not found for episode {episode_idx}, camera {camera_name}\n"
                    f"Expected path: {self.first_image_dir}/.../{camera_name}/episode_{episode_idx:06d}.png"
                )

            img_path = self.image_paths[episode_idx][camera_name]

            # Load image
            img = self.Image.open(img_path)

            # Convert to tensor
            img = transforms.ToTensor()(img)

            # Apply transforms if specified
            if self.image_transforms is not None:
                img = self.image_transforms(img)

            frame_dict[camera_name] = img

        # Get task from first row
        task = episode_data["task"].iloc[0] if "task" in episode_data.columns else ""

        # Get episode statistics (simplified version)
        dataset_meta = {
            "episode_index": episode_idx,
            "length": len(episode_data),
        }

        # Return in same format as LeRobotDatasetWrapper
        return {
            "start_items": [frame_dict],
            "target_items": [frame_dict],
            "episode_index": episode_idx,
            "task": task,
            "dataset_meta": dataset_meta,
        }


def build_lerobot_world_model_dataset(dataset_cfg: dict) -> torch.utils.data.Dataset:
    """
    Factory function to build a LeRobot world model dataset.

    This function creates either a LeRobotDatasetWrapper or LeRobotFirstFrameDataset
    based on the configuration. The LeRobotFirstFrameDataset is used when
    use_first_frame_dataset=True and provides much faster loading when only
    the first frame of each episode is needed.

    Args:
        dataset_cfg: Dictionary containing dataset configuration with keys:
            - repo_id: HuggingFace repository ID
            - root: Root directory of the dataset
            - camera_names: List of camera keys to load
            - camera_heights: Target height for images (optional)
            - camera_widths: Target width for images (optional)
            - episodes: List of episode indices to load (optional)
            - episodes_list: Path to file with episode indices (optional)
            - use_first_frame_dataset: If True, use LeRobotFirstFrameDataset (default: False)
            - start_select_policy: Frame selection policy for LeRobotDatasetWrapper
            - target_select_policy: Frame selection policy for LeRobotDatasetWrapper
            - revision: Dataset revision (for LeRobotDatasetWrapper)
            - target_timestamp: Target timestamp for closest_timestamp policy
            - start_n_frames: Number of frames for first_n_frames policy
            - target_n_frames: Number of frames for last_n_frames policy

    Returns:
        Dataset instance (either LeRobotDatasetWrapper or LeRobotFirstFrameDataset)

    Example:
        dataset_cfg = {
            "repo_id": "jesbu1/bridge_v2_lerobot",
            "root": "../datasets/bridge_orig_lerobot",
            "camera_names": ["observation.images.image_0"],
            "camera_heights": 256,
            "camera_widths": 256,
            "use_first_frame_dataset": True,
        }
        dataset = build_lerobot_world_model_dataset(dataset_cfg)
    """
    use_first_frame = dataset_cfg.get("use_first_frame_dataset", False)

    if use_first_frame:
        # Use LeRobotFirstFrameDataset for fast first-frame loading
        return LeRobotFirstFrameDataset(
            repo_id=dataset_cfg["repo_id"],
            root=dataset_cfg["root"],
            camera_names=dataset_cfg["camera_names"],
            camera_heights=dataset_cfg.get("camera_heights"),
            camera_widths=dataset_cfg.get("camera_widths"),
            episodes=dataset_cfg.get("episodes"),
            episodes_list=dataset_cfg.get("episodes_list"),
        )
    else:
        # Use original LeRobotDatasetWrapper
        return LeRobotDatasetWrapper(
            repo_id=dataset_cfg["repo_id"],
            root=dataset_cfg["root"],
            start_select_policy=dataset_cfg.get("start_select_policy", "first_frame"),
            target_select_policy=dataset_cfg.get("target_select_policy", "last_frame"),
            camera_names=dataset_cfg["camera_names"],
            target_timestamp=dataset_cfg.get("target_timestamp", 10**4),
            start_n_frames=dataset_cfg.get("start_n_frames", 1),
            target_n_frames=dataset_cfg.get("target_n_frames", 1),
            revision=dataset_cfg.get("revision", "v2.0"),
            camera_heights=dataset_cfg.get("camera_heights"),
            camera_widths=dataset_cfg.get("camera_widths"),
            episodes=dataset_cfg.get("episodes"),
            episodes_list=dataset_cfg.get("episodes_list"),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        type=str,
        default="jesbu1/bridge_v2_lerobot",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="../datasets/bridge_orig_lerobot",
    )
    parser.add_argument("--start_select_policy",
                        type=str, default="first_frame")
    parser.add_argument("--target_select_policy",
                        type=str, default="last_frame")
    args = parser.parse_args()

    dataset = LeRobotDatasetWrapper(
        repo_id=args.repo_id,
        root=args.root,
        start_select_policy=args.start_select_policy,
        target_select_policy=args.target_select_policy,
        camera_names=['observation.images.image_0']
    )

    data = dataset[0]
    print(data)
