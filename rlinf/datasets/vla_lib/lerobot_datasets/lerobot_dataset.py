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
Simplified PyTorch dataset implementation that follows OpenPI's exact logic.

This implementation matches the original OpenPI data loading pipeline while being
compatible with TRL and Transformers trainers.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset, Subset

# Import unified interface (which inherits from Dataset)
from rlinf.datasets.base_interface import UnifiedDatasetInterface

# Import decorator for auto-registration
from rlinf.datasets.factory import register_dataset

from .config import (
    DataConfigFactory,
    create_data_config_factory,
    create_data_config_factory_from_dict,
)
from .transforms import (
    DataTransformFn,
    Normalize,
    PromptFromLeRobotTask,
    compose,
    load_task_descriptions,
)

logger = logging.getLogger(__name__)


class TransformedDataset(Dataset):
    """Simple transformed dataset wrapper matching OpenPI's pattern."""

    def __init__(self, dataset: Dataset, transforms: list[DataTransformFn]):
        self._dataset = dataset
        self._transform = compose(transforms)

    def __getitem__(self, index):
        # Get sample from underlying dataset
        sample = self._dataset[index]

        # Apply transforms directly to tensor data
        # All transforms are tensor-aware and preserve device placement
        transformed_sample = self._transform(sample)

        return transformed_sample

    def __len__(self):
        return len(self._dataset)


@register_dataset("vla/lerobot", [r"lerobot/", r"data/.+", "lerobot"])
class LeRobotPyTorchDataset(UnifiedDatasetInterface):
    """Simple PyTorch dataset that follows OpenPI's exact logic."""

    def __init__(
        self,
        dataset_path: str | None = None,
        repo_id: str
        | None = None,  # Alias for dataset_path (for backward compatibility)
        action_horizon: int = 10,
        split: str = "train",
        data_config_factory: Optional[DataConfigFactory] = None,
        action_dim: Optional[int] = None,
        max_samples: Optional[int] = None,  # Limit dataset size for testing
        # Config-based creation parameters (alternative to data_config_factory)
        robot_type: Optional[str] = None,
        model_type: Optional[str] = None,
        default_prompt: Optional[str] = None,
        extra_delta_transform: bool = False,
        norm_stats_dir: Optional[str] = None,
        asset_id: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,  # Full config dict from YAML
        # Episode-level data scaling for data scaling experiments
        episode_percentage: Optional[
            float
        ] = None,  # Percentage of episodes to use (0-100)
        shuffle_episodes: bool = False,  # Whether to shuffle episodes before selecting
        episode_seed: int = 42,  # Seed for episode shuffling reproducibility
    ):
        """
        Initialize dataset following OpenPI's create_torch_dataset + transform_dataset pattern.

        Args:
            dataset_path: LeRobot dataset path or repository ID (unified interface)
            repo_id: Alias for dataset_path (backward compatibility)
            action_horizon: Number of future actions to predict
            split: Dataset split ("train", "test", etc.)
            data_config_factory: Factory for dataset configuration (legacy)
            action_dim: Action dimensionality
            max_samples: Maximum number of samples to use (for testing)
            robot_type: Robot type (libero, droid, aloha, generic). Auto-detected if None.
            model_type: Model type for normalization (pi0, pi05)
            default_prompt: Default prompt if not in dataset
            extra_delta_transform: Apply extra delta transform
            norm_stats_dir: Directory for normalization stats
            asset_id: Asset ID for stats lookup
            config: Full config dict from YAML (alternative to individual params)
            episode_percentage: Percentage of episodes to use (0-100). If None, use all episodes.
            shuffle_episodes: If True, randomly select episodes; if False, use first N episodes.
            episode_seed: Random seed for reproducible episode selection.
        """
        # Accept either dataset_path or repo_id for flexibility
        self.repo_id = dataset_path or repo_id
        self.max_samples = max_samples
        if self.repo_id is None:
            raise ValueError("Either 'dataset_path' or 'repo_id' must be provided")
        self.action_horizon = action_horizon
        self.split = split

        # Episode-level data scaling parameters
        self.episode_percentage = episode_percentage
        self.shuffle_episodes = shuffle_episodes
        self.episode_seed = episode_seed
        self._episode_indices = None  # Selected episode indices (for reference)
        self._sample_indices = None  # Sample indices to use (for index-based filtering)

        # Check if this is a local path or remote repo ID
        self.is_local = self._is_local_path(self.repo_id)

        # Load dataset metadata and create base dataset
        if self.is_local:
            # For local datasets, use just the folder name as repo_id and set root to the full path
            # Note: use absolute() instead of resolve() to avoid symlink resolution issues
            local_path = Path(self.repo_id).absolute()
            folder_name = local_path.name
            self.dataset_meta = LeRobotDatasetMetadata(folder_name, root=local_path)
        else:
            self.dataset_meta = LeRobotDatasetMetadata(self.repo_id)

        # Create data config - priority: data_config_factory > config dict > individual params
        if data_config_factory is not None:
            self.data_config = data_config_factory.create(action_dim=action_dim)
        elif config is not None:
            # Create from config dict (e.g., from Hydra YAML)
            factory = create_data_config_factory_from_dict(config)
            self.data_config = factory.create(
                action_dim=action_dim or config.get("action_dim", 32)
            )
        elif robot_type is not None or model_type is not None:
            # Create from individual parameters
            factory = create_data_config_factory(
                dataset_path=self.repo_id,
                robot_type=robot_type,
                model_type=model_type,
                default_prompt=default_prompt,
                extra_delta_transform=extra_delta_transform,
                norm_stats_dir=norm_stats_dir,
                asset_id=asset_id,
            )
            self.data_config = factory.create(action_dim=action_dim or 32)
        else:
            self.data_config = None

        # For delta_timestamps, we need to use the RAW dataset keys (before repack transforms)
        # The most common key in LeRobot datasets is "action" (singular)
        raw_action_keys = []
        if "action" in self.dataset_meta.features:
            raw_action_keys = ["action"]
        elif "actions" in self.dataset_meta.features:
            raw_action_keys = ["actions"]
        else:
            raise ValueError(
                f"No action key found in dataset metadata: {self.dataset_meta.features}"
            )

        # Calculate delta_timestamps using FPS from metadata (OpenPI pattern)
        delta_timestamps = {
            key: [t / self.dataset_meta.fps for t in range(action_horizon)]
            for key in raw_action_keys
        }
        logger.info(f"Delta timestamps: {delta_timestamps}")

        # Create base LeRobot dataset (load all episodes first)
        # We'll apply episode filtering via sample index mapping after loading
        if self.is_local:
            local_path = Path(self.repo_id).absolute()
            folder_name = local_path.name
            self.base_dataset = LeRobotDataset(
                folder_name,
                root=local_path,
                delta_timestamps=delta_timestamps,
                download_videos=False,
            )
        else:
            self.base_dataset = LeRobotDataset(
                self.repo_id, delta_timestamps=delta_timestamps, download_videos=False
            )

        # Compute episode filtering AFTER loading (to get episode_data_index)
        # This creates a sample index mapping for filtered access
        self._compute_episode_filtering()

        # Step 2: Add prompt from task if needed (following OpenPI pattern)
        # This must happen BEFORE the main transform pipeline (like in OpenPI data_loader.py)
        if self.data_config and getattr(self.data_config, "prompt_from_task", True):
            # Determine which tasks to use
            tasks_to_use = None

            if self.is_local:
                # For local datasets, try loading from our universal task loader
                tasks_to_use = load_task_descriptions(Path(self.repo_id).absolute())

            # If no local tasks found, try using metadata tasks (for remote datasets or fallback)
            if (
                not tasks_to_use
                and hasattr(self.dataset_meta, "tasks")
                and self.dataset_meta.tasks
            ):
                tasks_to_use = self.dataset_meta.tasks

            # Apply prompt transform if we have tasks
            if tasks_to_use:
                logger.info(f"Adding prompt transform with {len(tasks_to_use)} tasks")
                self.base_dataset = TransformedDataset(
                    self.base_dataset, [PromptFromLeRobotTask(tasks_to_use)]
                )

        # Step 3: Apply transform pipeline (following OpenPI's transform_dataset pattern)
        # This should wrap the dataset with TransformedDataset, not apply transforms in __getitem__
        transforms = self._create_transform_list()
        if transforms:
            self.base_dataset = TransformedDataset(self.base_dataset, transforms)

        num_used_episodes = (
            len(self._episode_indices)
            if self._episode_indices is not None
            else self.dataset_meta.total_episodes
        )
        num_samples = (
            len(self._sample_indices)
            if self._sample_indices is not None
            else len(self.base_dataset)
        )
        logger.info(f"Loaded dataset: {self.repo_id}")
        logger.info(f"  Type: {'Local' if self.is_local else 'Remote'}")
        logger.info(
            f"  Episodes: {self.dataset_meta.total_episodes} (using {num_used_episodes})"
        )
        logger.info(f"  Frames: {self.dataset_meta.total_frames}")
        logger.info(f"  FPS: {self.dataset_meta.fps}")
        logger.info(f"  Available keys: {list(self.dataset_meta.features.keys())}")
        logger.info(f"  Split '{split}': {num_samples} samples")

    def _is_local_path(self, path_or_id: str) -> bool:
        """Check if the input is a local path or a remote repo ID."""
        # Check multiple indicators of a local path
        path = Path(path_or_id)
        return (
            path.exists()  # Path exists
            or path.is_absolute()  # Absolute path
            or path_or_id.startswith("./")  # Relative path with ./
            or path_or_id.startswith("../")  # Relative path with ../
            or (
                path_or_id.startswith("data/") and "/" not in path_or_id[5:]
            )  # data/ prefix without additional slashes
            or (
                "/" not in path_or_id and not path_or_id.startswith("lerobot/")
            )  # No slashes and not a huggingface repo
        )

    def _compute_episode_filtering(self) -> None:
        """
        Compute episode filtering based on episode_percentage and shuffle_episodes.

        This creates a sample index mapping for filtered access. We use index-based
        filtering instead of LeRobot's episodes argument because the episodes argument
        doesn't handle non-sequential episode indices correctly when shuffling.
        """
        if self.episode_percentage is None or self.episode_percentage >= 100.0:
            # No filtering needed
            return

        if self.episode_percentage <= 0:
            raise ValueError(
                f"episode_percentage must be > 0, got {self.episode_percentage}"
            )

        # Get total number of episodes
        total_episodes = self.dataset_meta.total_episodes

        # Calculate number of episodes to use (must be at least 1)
        num_episodes_to_use = max(
            1, int(total_episodes * self.episode_percentage / 100.0)
        )

        # Get episode indices
        all_episode_indices = list(range(total_episodes))

        if self.shuffle_episodes:
            # Randomly select episodes with reproducible seed
            rng = np.random.default_rng(self.episode_seed)
            selected_episodes = rng.choice(
                all_episode_indices, size=num_episodes_to_use, replace=False
            )
            selected_episodes = set(selected_episodes.tolist())
        else:
            # Use first N episodes
            selected_episodes = set(all_episode_indices[:num_episodes_to_use])

        self._episode_indices = sorted(selected_episodes)

        # Get episode_data_index from the loaded dataset to map episodes to sample indices
        episode_data_index = self.base_dataset.episode_data_index

        # Compute sample indices for selected episodes
        sample_indices = []
        for ep_idx in self._episode_indices:
            start_idx = episode_data_index["from"][ep_idx].item()
            end_idx = episode_data_index["to"][ep_idx].item()
            sample_indices.extend(range(start_idx, end_idx))

        self._sample_indices = sample_indices

        logger.info("  Episode filtering applied:")
        logger.info(f"    Percentage: {self.episode_percentage}%")
        logger.info(
            f"    Episodes: {num_episodes_to_use}/{total_episodes} ({'shuffled' if self.shuffle_episodes else 'first N'})"
        )
        logger.info(f"    Sample indices: {len(self._sample_indices)}")

    def _create_transform_list(self) -> list[DataTransformFn]:
        """Create transform list following OpenPI's transform_dataset logic."""
        transforms = []

        if self.data_config is not None:
            # Add repack transforms (following OpenPI pattern)
            transforms.extend(self.data_config.repack_transforms.inputs)

            # Add data transforms
            transforms.extend(self.data_config.data_transforms.inputs)

            # Add normalization (following OpenPI pattern)
            if self.data_config.norm_stats is not None:
                transforms.append(
                    Normalize(
                        self.data_config.norm_stats,
                        self.data_config.use_quantile_norm,
                        skip_dims=self.data_config.action_norm_skip_dims,
                    )
                )

            # Add model transforms
            transforms.extend(self.data_config.model_transforms.inputs)

        return transforms

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._sample_indices is not None:
            base_len = len(self._sample_indices)
        else:
            base_len = len(self.base_dataset)
        if self.max_samples is not None:
            return min(base_len, self.max_samples)
        return base_len

    def _normalize_video_timestamps(self) -> None:
        """
        Fix video timestamp metadata to be relative to each mp4 file.

        The original LeRobot metadata may store cumulative timestamps across
        concatenated video files. torchcodec expects timestamps within each
        video file. We detect per-file durations, then wrap episode
        from/to timestamps into the valid range for that file.
        """
        logger = logging.getLogger(__name__)
        meta = getattr(self.base_dataset, "meta", None)
        fps = getattr(self.base_dataset, "fps", None)
        if meta is None or fps is None:
            return

        # Derive durations for every (video_key, file_index)
        video_keys = []
        for k in meta.features.keys():
            if k.startswith("observation.images."):
                video_keys.append(k)
        if not video_keys:
            return

        durations = {}  # (video_key, file_idx) -> duration_seconds
        for video_key in video_keys:
            file_indices = meta.episodes[f"videos/{video_key}/file_index"].unique()
            for file_idx in file_indices:
                # Find one episode that uses this file to recover the path
                ep_idx = int(
                    meta.episodes[
                        meta.episodes[f"videos/{video_key}/file_index"] == file_idx
                    ]["episode_index"].iloc[0]
                )
                file_path = self.base_dataset.root / meta.get_video_file_path(
                    ep_idx, video_key
                )
                cap = cv2.VideoCapture(str(file_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if frame_count <= 0:
                    logger.warning(
                        f"{video_key} file_index={file_idx} has invalid frame count ({frame_count}); "
                        "skip timestamp normalization for this file."
                    )
                    continue
                durations[(video_key, int(file_idx))] = frame_count / fps

        # Apply wrap-around per episode
        episodes = meta.episodes.copy()
        for video_key in video_keys:
            from_col = f"videos/{video_key}/from_timestamp"
            to_col = f"videos/{video_key}/to_timestamp"
            file_idx_col = f"videos/{video_key}/file_index"
            if from_col not in episodes.columns:
                continue
            for idx, row in episodes.iterrows():
                file_idx = int(row[file_idx_col])
                duration = durations.get((video_key, file_idx))
                if duration is None or duration <= 0:
                    continue
                new_from = float(row[from_col]) % duration
                # Use episode length (frames) to recompute to_timestamp
                ep_len = int(row["length"])
                new_to = new_from + max(ep_len - 1, 0) / fps
                episodes.at[idx, from_col] = new_from
                episodes.at[idx, to_col] = new_to

        meta.episodes = episodes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample from the dataset."""
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        idx = int(idx)
        # Map through sample_indices if episode filtering is applied
        if self._sample_indices is not None:
            idx = self._sample_indices[idx]
        return self.base_dataset[idx]

    def get_train_val_split(
        self, validation_split: Optional[float] = None
    ) -> tuple["LeRobotPyTorchDataset", "LeRobotPyTorchDataset"]:
        """
        Split the dataset into train and validation sets.

        Args:
            validation_split: Fraction of data for validation (default: 0.1)

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if validation_split is None:
            validation_split = 0.1

        total_len = len(self)
        val_len = int(total_len * validation_split)
        train_len = total_len - val_len

        # Create index-based splits
        indices = np.arange(total_len)
        np.random.shuffle(indices)

        train_indices = indices[:train_len]
        val_indices = indices[train_len:]

        # Create subset datasets
        train_ds = Subset(self, train_indices)
        val_ds = Subset(self, val_indices)

        return train_ds, val_ds

    def get_custom_tokens(self) -> Optional[list[str]]:
        """
        Get custom tokens that should be added to the tokenizer.

        VLA datasets typically don't need custom tokens.

        Returns:
            None
        """
        return None

    def get_source_name(self) -> str:
        """
        Get a readable source name for this dataset.

        Returns:
            Human-readable dataset source name (repo_id)
        """
        # Use the repo_id as the source name
        return self.repo_id.replace("/", "_").replace("-", "_").lower()


def create_lerobot_dataset(
    dataset_path: str | None = None,
    repo_id: str | None = None,  # Alias for dataset_path
    action_horizon: int = 10,
    split: str = "train",
    data_config_factory: Optional[DataConfigFactory] = None,
    action_dim: Optional[int] = None,
    # Config-based creation (preferred)
    robot_type: Optional[str] = None,
    model_type: Optional[str] = None,
    default_prompt: Optional[str] = None,
    extra_delta_transform: bool = False,
    norm_stats_dir: Optional[str] = None,
    asset_id: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    max_samples: Optional[int] = None,
    # Episode-level data scaling
    episode_percentage: Optional[float] = None,
    shuffle_episodes: bool = False,
    episode_seed: int = 42,
) -> LeRobotPyTorchDataset:
    """
    Create a LeRobot dataset following OpenPI's pattern.

    This function provides a clean interface for creating VLA datasets.
    Configuration can be provided via:
    1. config dict (from YAML/Hydra) - preferred
    2. Individual parameters (robot_type, model_type, etc.)
    3. data_config_factory (legacy)

    Args:
        dataset_path: Path or repo_id of the dataset
        repo_id: Alias for dataset_path
        action_horizon: Number of future actions to predict
        split: Dataset split
        data_config_factory: Legacy factory interface
        action_dim: Action dimensionality
        robot_type: Robot type (libero, droid, aloha, generic)
        model_type: Model type (pi0, pi05)
        default_prompt: Default prompt
        extra_delta_transform: Extra delta transform flag
        norm_stats_dir: Norm stats directory
        asset_id: Asset ID
        config: Full config dict from YAML
        max_samples: Max samples for testing
        episode_percentage: Percentage of episodes to use (0-100) for data scaling experiments
        shuffle_episodes: If True, randomly select episodes; if False, use first N episodes
        episode_seed: Random seed for reproducible episode selection

    Example:
        # From YAML config
        >>> dataset = create_lerobot_dataset(config=cfg.dataset, action_dim=32)

        # With explicit params
        >>> dataset = create_lerobot_dataset(
        ...     dataset_path="data/libero",
        ...     robot_type="libero",
        ...     model_type="pi05",
        ...     action_dim=32,
        ... )

        # With episode scaling (use 50% of episodes, randomly selected)
        >>> dataset = create_lerobot_dataset(
        ...     dataset_path="data/libero",
        ...     episode_percentage=50.0,
        ...     shuffle_episodes=True,
        ... )
    """
    return LeRobotPyTorchDataset(
        dataset_path=dataset_path or repo_id,
        action_horizon=action_horizon,
        split=split,
        data_config_factory=data_config_factory,
        action_dim=action_dim,
        robot_type=robot_type,
        model_type=model_type,
        default_prompt=default_prompt,
        extra_delta_transform=extra_delta_transform,
        norm_stats_dir=norm_stats_dir,
        asset_id=asset_id,
        config=config,
        max_samples=max_samples,
        episode_percentage=episode_percentage,
        shuffle_episodes=shuffle_episodes,
        episode_seed=episode_seed,
    )


def vla_data_collator(features):
    """
    Custom data collator for VLA training that handles multi-modal inputs.

    This collator handles:
    - Images (multiple cameras)
    - State vectors
    - Actions
    - Text prompts
    """
    batch = {}

    # Handle different types of inputs
    for key in features[0].keys():
        values = [f[key] for f in features]

        if key in ["image", "wrist_image"]:
            # Stack images
            batch[key] = torch.stack(values)
        elif key in ["state", "actions"]:
            # Stack numerical arrays
            batch[key] = torch.stack(values)
        elif key == "prompt":
            # Keep prompts as list for tokenization
            batch[key] = values
        else:
            # Default behavior
            batch[key] = (
                torch.stack(values) if isinstance(values[0], torch.Tensor) else values
            )

    return batch
