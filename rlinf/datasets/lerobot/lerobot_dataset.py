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

"""
Simplified PyTorch dataset implementation that follows OpenPI's exact logic.

This implementation matches the original OpenPI data loading pipeline while being
compatible with TRL and Transformers trainers.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset

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
        sample = self._dataset[index]
        return self._transform(sample)

    def __len__(self):
        return len(self._dataset)


class LeRobotPyTorchDataset(Dataset):
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
        self.repo_id = dataset_path or repo_id
        self.max_samples = max_samples
        if self.repo_id is None:
            raise ValueError("Either 'dataset_path' or 'repo_id' must be provided")
        self.action_horizon = action_horizon
        self.split = split

        self.episode_percentage = episode_percentage
        self.shuffle_episodes = shuffle_episodes
        self.episode_seed = episode_seed
        self._episode_indices = None  # Selected episode indices (for reference)
        self._sample_indices = None

        self.is_local = self._is_local_path(self.repo_id)

        if self.is_local:
            # Use absolute() instead of resolve() to avoid symlink resolution issues
            local_path = Path(self.repo_id).absolute()
            folder_name = local_path.name
            self.dataset_meta = LeRobotDatasetMetadata(folder_name, root=local_path)
        else:
            self.dataset_meta = LeRobotDatasetMetadata(self.repo_id)

        if data_config_factory is not None:
            self.data_config = data_config_factory.create(action_dim=action_dim)
        elif config is not None:
            factory = create_data_config_factory_from_dict(config)
            self.data_config = factory.create(
                action_dim=action_dim or config.get("action_dim", 32)
            )
        elif robot_type is not None or model_type is not None:
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

        # Use raw dataset keys (before repack transforms) for delta_timestamps
        raw_action_keys = []
        if "action" in self.dataset_meta.features:
            raw_action_keys = ["action"]
        elif "actions" in self.dataset_meta.features:
            raw_action_keys = ["actions"]
        else:
            raise ValueError(
                f"No action key found in dataset metadata: {self.dataset_meta.features}"
            )

        delta_timestamps = {
            key: [t / self.dataset_meta.fps for t in range(action_horizon)]
            for key in raw_action_keys
        }
        logger.info(f"Delta timestamps: {delta_timestamps}")

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

        self._compute_episode_filtering()

        # Prompt injection must happen before the main transform pipeline
        if self.data_config and getattr(self.data_config, "prompt_from_task", True):
            tasks_to_use = None

            if self.is_local:
                tasks_to_use = load_task_descriptions(Path(self.repo_id).absolute())

            if (
                not tasks_to_use
                and hasattr(self.dataset_meta, "tasks")
                and self.dataset_meta.tasks
            ):
                tasks_to_use = self.dataset_meta.tasks

            if tasks_to_use:
                logger.info(f"Adding prompt transform with {len(tasks_to_use)} tasks")
                self.base_dataset = TransformedDataset(
                    self.base_dataset, [PromptFromLeRobotTask(tasks_to_use)]
                )

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
        path = Path(path_or_id)
        return (
            path.exists()
            or path.is_absolute()
            or path_or_id.startswith("./")
            or path_or_id.startswith("../")
            or (path_or_id.startswith("data/") and "/" not in path_or_id[5:])
            or ("/" not in path_or_id and not path_or_id.startswith("lerobot/"))
        )

    def _compute_episode_filtering(self) -> None:
        """Compute episode filtering based on episode_percentage and shuffle_episodes.

        Uses index-based filtering instead of LeRobot's episodes argument because the
        episodes argument doesn't handle non-sequential episode indices correctly when shuffling.
        """
        if self.episode_percentage is None or self.episode_percentage >= 100.0:
            return

        if self.episode_percentage <= 0:
            raise ValueError(
                f"episode_percentage must be > 0, got {self.episode_percentage}"
            )

        total_episodes = self.dataset_meta.total_episodes
        num_episodes_to_use = max(
            1, int(total_episodes * self.episode_percentage / 100.0)
        )

        all_episode_indices = list(range(total_episodes))

        if self.shuffle_episodes:
            rng = np.random.default_rng(self.episode_seed)
            selected_episodes = rng.choice(
                all_episode_indices, size=num_episodes_to_use, replace=False
            )
            selected_episodes = set(selected_episodes.tolist())
        else:
            selected_episodes = set(all_episode_indices[:num_episodes_to_use])

        self._episode_indices = sorted(selected_episodes)

        episode_data_index = self.base_dataset.episode_data_index

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
            transforms.extend(self.data_config.repack_transforms.inputs)
            transforms.extend(self.data_config.data_transforms.inputs)

            if self.data_config.norm_stats is not None:
                transforms.append(
                    Normalize(
                        self.data_config.norm_stats,
                        self.data_config.use_quantile_norm,
                        skip_dims=self.data_config.action_norm_skip_dims,
                    )
                )

            transforms.extend(self.data_config.model_transforms.inputs)

        return transforms

    def __len__(self) -> int:
        if self._sample_indices is not None:
            base_len = len(self._sample_indices)
        else:
            base_len = len(self.base_dataset)
        if self.max_samples is not None:
            return min(base_len, self.max_samples)
        return base_len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        idx = int(idx)
        if self._sample_indices is not None:
            idx = self._sample_indices[idx]
        return self.base_dataset[idx]

