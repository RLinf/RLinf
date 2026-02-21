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
RL Dataset implementation for value learning.

This module extends the LeRobot dataset with RL-specific features:
1. History observations (past N timesteps)
2. Action/reward chunks for n-step learning
3. Next observation for bootstrapping
4. Precomputed returns
"""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from rlinf.datasets.factory import register_dataset

# Reuse VLA dataset transforms
from rlinf.datasets.vla_lib.lerobot_datasets.config import (
    DataConfigFactory,
    create_data_config_factory,
    create_data_config_factory_from_dict,
)
from rlinf.datasets.vla_lib.lerobot_datasets.lerobot_dataset import (
    LeRobotPyTorchDataset,
    TransformedDataset,
)
from rlinf.datasets.vla_lib.lerobot_datasets.transforms import (
    DataTransformFn,
    Normalize,
    PromptFromLeRobotTask,
    RepackTransform,
    compose,
    load_task_descriptions,
)

from .config import RLDataConfig, create_rl_config
from .io_processing.value_transforms import ReturnDiscretizer

logger = logging.getLogger(__name__)


@register_dataset("rl/lerobot", [r"rl/", r"critic/"])
class LeRobotRLDataset(LeRobotPyTorchDataset):
    """RL Dataset with temporal structure for value learning.

    Extends LeRobotDataset with:
    1. History observations (past N steps)
    2. Action/reward chunks (future H steps)
    3. Next observation for bootstrapping
    4. Precomputed returns

    Sample structure at timestep t:
        {
            # Current observation
            "state": tensor(obs_dim),
            "image": {"cam1": tensor(C,H,W), ...},

            # History (if history_length > 0)
            "history_state": tensor(N, obs_dim),
            "history_image": {"cam1": tensor(N, C,H,W), ...},
            "history_state_is_pad": tensor(N),  # True if padded

            # Future chunks
            "action_chunk": tensor(H, action_dim),
            "reward_chunk": tensor(H,),
            "action_chunk_is_pad": tensor(H),
            "reward_chunk_is_pad": tensor(H),

            # Terminal flag at t+H (for offline datasets)
            "done": tensor(1),  # True if episode ends at or before t+H

            # Bootstrapping
            "next_state": tensor(obs_dim),  # State at t+H
            "next_image": {"cam1": tensor(C,H,W), ...},
            "next_state_is_pad": bool,  # True if t+H is out of episode

            # Value targets
            "return": tensor(1),  # Precomputed return at t

            # Metadata
            "prompt": str,
            "episode_index": int,
            "frame_index": int,
        }
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        repo_id: str | None = None,
        # RL-specific configuration (required)
        rl_config: Optional[RLDataConfig] = None,
        # VLA dataset configuration
        split: str = "train",
        data_config_factory: Optional[DataConfigFactory] = None,
        action_dim: Optional[int] = None,
        robot_type: Optional[str] = None,
        model_type: Optional[str] = None,
        default_prompt: Optional[str] = None,
        extra_delta_transform: bool = False,
        norm_stats_dir: Optional[str] = None,
        asset_id: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        action_norm_skip_dims: Optional[dict[str, list[int]]] = None,
        # Episode filtering
        episode_percentage: Optional[float] = None,
        shuffle_episodes: bool = False,
        episode_seed: int = 42,
    ):
        """Initialize RL dataset.

        Args:
            dataset_path: LeRobot dataset path or repo ID
            repo_id: Alias for dataset_path
            rl_config: RL configuration (use create_rl_config() to build one)
            split: Dataset split
            data_config_factory: Factory for transforms
            action_dim: Action dimension
            robot_type: Robot type for auto-config
            model_type: Model type (pi0, pi05)
            default_prompt: Default prompt
            extra_delta_transform: Apply extra delta transform
            norm_stats_dir: Normalization stats directory
            asset_id: Asset ID
            config: Full config dict from YAML
            max_samples: Limit dataset size
            action_norm_skip_dims: Skip normalization for specific dimensions
            episode_percentage: Percentage of episodes to use
            shuffle_episodes: Random episode selection
            episode_seed: Seed for reproducibility
        """
        self.repo_id = dataset_path or repo_id
        if self.repo_id is None:
            raise ValueError("Either 'dataset_path' or 'repo_id' must be provided")

        self.max_samples = max_samples
        self.split = split

        # Episode filtering
        self.episode_percentage = episode_percentage
        self.shuffle_episodes = shuffle_episodes
        self.episode_seed = episode_seed
        self._episode_indices = None
        self._sample_indices = None

        # Check local vs remote
        self.is_local = self._is_local_path(self.repo_id)

        # Load metadata first (needed for auto-detecting history keys)
        if self.is_local:
            local_path = Path(self.repo_id).resolve()
            folder_name = local_path.name
            self.dataset_meta = LeRobotDatasetMetadata(folder_name, root=local_path)
        else:
            self.dataset_meta = LeRobotDatasetMetadata(self.repo_id)

        # Use provided rl_config or create default
        if rl_config is not None:
            self.rl_config = rl_config
            logger.info("Using provided rl_config")
        else:
            self.rl_config = create_rl_config(
                history_keys=self._auto_detect_history_keys(),
            )
            logger.info("Created default rl_config with auto-detected history_keys")

        # Auto-detect history keys if not provided in rl_config
        if self.rl_config.history_keys is None:
            detected_keys = self._auto_detect_history_keys()
            self.rl_config = replace(self.rl_config, history_keys=tuple(detected_keys))
            logger.info(f"Auto-detected history_keys: {detected_keys}")

        # Create data config for transforms
        self.data_config = self._create_data_config(
            data_config_factory=data_config_factory,
            config=config,
            robot_type=robot_type,
            model_type=model_type,
            default_prompt=default_prompt,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
            action_dim=action_dim,
            extra_delta_transform=extra_delta_transform,
            action_norm_skip_dims=action_norm_skip_dims,
        )

        # Generate delta_timestamps for temporal data
        delta_timestamps = self.rl_config.get_delta_timestamps(self.dataset_meta.fps)
        logger.info(f"RL Delta timestamps: {delta_timestamps}")

        # Create base LeRobot dataset with temporal structure
        if self.is_local:
            local_path = Path(self.repo_id).resolve()
            folder_name = local_path.name
            self.base_dataset = LeRobotDataset(
                folder_name,
                root=local_path,
                delta_timestamps=delta_timestamps,
                download_videos=False,
            )
        else:
            self.base_dataset = LeRobotDataset(
                self.repo_id,
                delta_timestamps=delta_timestamps,
                download_videos=False,
            )

        # Compute episode filtering
        self._compute_episode_filtering()

        # Add prompt transform
        self._add_prompt_transform()

        # Store VLA transforms to apply AFTER restructuring in __getitem__
        # This is critical because RL datasets have temporal structure (history + future)
        # and VLA transforms (like FrankaInputs) expect single-timestep data
        transforms = self._create_transform_list()
        self._vla_transform = compose(transforms) if transforms else None

        # Apply return discretization if enabled
        self.return_discretizer = None
        if self.rl_config.discretize_return:
            self.return_discretizer = self._create_return_discretizer()
            if self.return_discretizer:
                logger.info(
                    f"Return discretization enabled: {self.rl_config.num_return_bins} bins"
                )

        # Log dataset info
        self._log_dataset_info()

    def _auto_detect_history_keys(self):
        """Auto-detect observation keys for history from dataset features."""
        obs_keys = []
        for key in self.dataset_meta.features:
            if key.startswith("observation."):
                obs_keys.append(key)
        logger.info(f"History keys auto-detected: {obs_keys}")

        return obs_keys

    def _create_data_config(
        self,
        data_config_factory,
        config,
        robot_type,
        model_type,
        default_prompt,
        norm_stats_dir,
        asset_id,
        action_dim,
        extra_delta_transform,
        action_norm_skip_dims=None,
    ):
        """Create data configuration for transforms."""
        if data_config_factory is not None:
            return data_config_factory.create(action_dim=action_dim)
        elif config is not None:
            factory = create_data_config_factory_from_dict(config)
            return factory.create(action_dim=action_dim or config.get("action_dim", 32))
        elif robot_type is not None or model_type is not None:
            factory = create_data_config_factory(
                dataset_path=self.repo_id,
                robot_type=robot_type,
                model_type=model_type,
                default_prompt=default_prompt,
                extra_delta_transform=extra_delta_transform,
                norm_stats_dir=norm_stats_dir,
                asset_id=asset_id,
                action_norm_skip_dims=action_norm_skip_dims,
            )
            # For value training, normalization is required for state discretization
            # Only skip norm_stats if no norm_stats_dir is provided
            skip_norm = norm_stats_dir is None
            return factory.create(
                action_dim=action_dim or 32, skip_norm_stats=skip_norm
            )
        return None

    def _add_prompt_transform(self):
        """Add prompt extraction transform if needed."""
        if self.data_config and getattr(self.data_config, "prompt_from_task", True):
            tasks = None
            if self.is_local:
                tasks = load_task_descriptions(Path(self.repo_id).resolve())

            if (
                not tasks
                and hasattr(self.dataset_meta, "tasks")
                and self.dataset_meta.tasks
            ):
                tasks = self.dataset_meta.tasks

            if tasks:
                logger.info(f"Adding prompt transform with {len(tasks)} tasks")
                self.base_dataset = TransformedDataset(
                    self.base_dataset, [PromptFromLeRobotTask(tasks)]
                )

    def _create_transform_list(self) -> list[DataTransformFn]:
        """Create transform list following OpenPI's transform_dataset logic.

        For RL datasets, repack transforms are modified to pass through unmapped keys
        since RL restructuring creates additional keys like action_chunk, reward_chunk, etc.
        """
        transforms = []

        if self.data_config is not None:
            # Add repack transforms with passthrough_unmapped=True for RL
            # This ensures RL-specific keys (action_chunk, reward_chunk, etc.) are preserved
            for transform in self.data_config.repack_transforms.inputs:
                if isinstance(transform, RepackTransform):
                    transforms.append(
                        RepackTransform(transform.structure, passthrough_unmapped=True)
                    )
                else:
                    transforms.append(transform)

            # Add data transforms
            transforms.extend(self.data_config.data_transforms.inputs)

            # Add normalization (following OpenPI pattern)
            # Exclude 'return' from normalization since ReturnDiscretizer handles its own normalization
            if self.data_config.norm_stats is not None:
                norm_stats = self.data_config.norm_stats
                if (
                    self.rl_config.discretize_return
                    and self.rl_config.return_key in norm_stats
                ):
                    norm_stats = {
                        k: v
                        for k, v in norm_stats.items()
                        if k != self.rl_config.return_key
                    }
                transforms.append(
                    Normalize(
                        norm_stats,
                        self.data_config.use_quantile_norm,
                        skip_dims=self.data_config.action_norm_skip_dims,
                    )
                )

            # Add model transforms
            transforms.extend(self.data_config.model_transforms.inputs)

        return transforms

    def _create_return_discretizer(self) -> Optional[ReturnDiscretizer]:
        """Create return discretizer if enabled."""
        if not self.rl_config.discretize_return:
            return None

        # Common kwargs for discretizer
        common_kwargs = {
            "num_bins": self.rl_config.num_return_bins,
            "return_key": self.rl_config.return_key,
            "output_key": self.rl_config.return_token_key,
            "keep_continuous": self.rl_config.keep_continuous_return,
            "normalize_to_minus_one_zero": self.rl_config.normalize_to_minus_one_zero,
        }

        # Determine min/max values
        if (
            self.rl_config.return_min is not None
            and self.rl_config.return_max is not None
        ):
            return ReturnDiscretizer(
                return_min=self.rl_config.return_min,
                return_max=self.rl_config.return_max,
                **common_kwargs,
            )
        elif self.rl_config.return_norm_stats_path:
            return ReturnDiscretizer(
                norm_stats_path=Path(self.rl_config.return_norm_stats_path),
                **common_kwargs,
            )
        else:
            logger.warning(
                "Return discretization enabled but no min/max or norm_stats_path provided. "
                "Discretization will be skipped."
            )
            return None

    def _log_dataset_info(self):
        """Log dataset information."""
        num_episodes = (
            len(self._episode_indices)
            if self._episode_indices
            else self.dataset_meta.total_episodes
        )
        num_samples = (
            len(self._sample_indices)
            if self._sample_indices
            else len(self.base_dataset)
        )

        logger.info(f"Loaded RL dataset: {self.repo_id}")
        logger.info(f"  Type: {'Local' if self.is_local else 'Remote'}")
        logger.info(f"  Episodes: {num_episodes}/{self.dataset_meta.total_episodes}")
        logger.info(f"  FPS: {self.dataset_meta.fps}")
        logger.info(f"  History length: {self.rl_config.history_length}")
        logger.info(f"  Action horizon: {self.rl_config.action_horizon}")
        logger.info(f"  Valid samples: {num_samples}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single RL training sample.

        The sample is restructured from LeRobot's flat format to a hierarchical
        format suitable for value learning. LeRobot handles episode boundaries
        via clamping and provides `_is_pad` masks for temporal features.

        For next observations (include_next_obs=True), the same VLA transforms
        are applied to ensure identical processing (normalization, rot6d, etc.).
        """
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        # Map through valid sample indices
        if self._sample_indices is not None:
            idx = self._sample_indices[idx]

        # Get raw sample from LeRobot (with temporal structure from delta_timestamps)
        raw_sample = self.base_dataset[idx]

        # Restructure for RL: extract current obs, history, action chunks, etc.
        # This stores next obs in 'next_observation_raw' with same key structure
        rl_sample = self._restructure_sample(raw_sample)

        # Extract next obs before applying transforms (will process separately)
        next_obs_raw = rl_sample.pop("next_observation_raw", None)
        next_obs_is_pad = rl_sample.pop("next_observation_is_pad", False)

        # Apply VLA transforms to current observation
        if self._vla_transform is not None:
            rl_sample = self._vla_transform(rl_sample)

            # Apply same transforms to next observation
            if next_obs_raw is not None:
                # Add metadata needed by transforms
                if "prompt" in rl_sample:
                    next_obs_raw["prompt"] = rl_sample["prompt"]

                try:
                    processed_next = self._vla_transform(next_obs_raw)

                    # Store with same structure as current observation
                    rl_sample["next_observation"] = {
                        "images": processed_next.get(
                            "image", processed_next.get("images", {})
                        ),
                        "state": processed_next.get("state"),
                        "is_pad": next_obs_is_pad,
                    }
                except Exception as e:
                    if not getattr(self, "_logged_next_obs_warning", False):
                        logger.warning(f"Failed to process next observation: {e}")
                        self._logged_next_obs_warning = True
                    rl_sample["next_observation"] = {"is_pad": True}

        # Apply return discretization if enabled
        if self.return_discretizer is not None:
            rl_sample = self.return_discretizer(rl_sample)

        return rl_sample

    def _restructure_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Restructure LeRobot sample for RL training.

        Converts flat temporal arrays to structured format with:
        - Current observation
        - History observations
        - Future action/reward chunks
        - Next observation for bootstrapping

        All `_is_pad` masks from LeRobot are preserved and restructured
        alongside their corresponding data tensors.
        """
        rl_sample = {}

        N = self.rl_config.history_length
        H = self.rl_config.action_horizon

        # Metadata keys to pass through directly (no padding)
        metadata_keys = {
            "prompt",
            "task",
            "episode_index",
            "frame_index",
            "index",
            "timestamp",
            "task_index",
        }

        # Process each feature based on its role
        for key, value in sample.items():
            # Skip padding masks - they are handled alongside their data keys
            if key.endswith("_is_pad"):
                continue

            # Pass through metadata keys directly
            if key in metadata_keys:
                rl_sample[key] = value
                continue

            # Get corresponding padding mask if exists
            pad_key = f"{key}_is_pad"
            is_pad = sample.get(pad_key, None)

            # Handle temporal features (tensors with at least 1 dimension)
            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                if key in self.rl_config.action_keys:
                    self._extract_action_chunk(rl_sample, key, value, is_pad, N, H)
                elif key in self.rl_config.reward_keys:
                    self._extract_reward_chunk(rl_sample, key, value, is_pad, H)
                elif key == self.rl_config.done_key:
                    self._extract_done(rl_sample, key, value, is_pad)
                elif key == self.rl_config.return_key:
                    # Return is at current timestep only - preserve original key name
                    rl_sample[key] = value.squeeze() if value.dim() > 0 else value
                    if is_pad is not None:
                        rl_sample[f"{key}_is_pad"] = (
                            is_pad.squeeze() if is_pad.dim() > 0 else is_pad
                        )
                elif key in (self.rl_config.history_keys or []):
                    self._extract_obs_with_history(rl_sample, key, value, is_pad, N, H)
                else:
                    # Pass through other temporal features with their padding masks
                    rl_sample[key] = value
                    if is_pad is not None:
                        rl_sample[f"{key}_is_pad"] = is_pad
            else:
                # Non-tensor or scalar values pass through directly
                rl_sample[key] = value

        return rl_sample

    def _extract_action_chunk(
        self,
        rl_sample: dict,
        key: str,
        value: torch.Tensor,
        is_pad: Optional[torch.Tensor],
        N: int,
        H: int,
    ):
        """Extract action chunk from temporal tensor.

        Preserves the original key name so VLA transforms (repack) can find it.
        """
        if self.rl_config.include_history_actions:
            # value has shape (N + H, action_dim)
            # History: [:N], Future: [N:]
            if value.dim() >= 2 and value.shape[0] >= N + H:
                rl_sample[f"history_{key}"] = value[:N]
                rl_sample[key] = value[N : N + H]  # Preserve original key name
                if is_pad is not None:
                    rl_sample[f"history_{key}_is_pad"] = is_pad[:N]
                    rl_sample[f"{key}_is_pad"] = is_pad[N : N + H]
            else:
                rl_sample[key] = value[:H] if value.shape[0] >= H else value
                if is_pad is not None:
                    rl_sample[f"{key}_is_pad"] = (
                        is_pad[:H] if is_pad.shape[0] >= H else is_pad
                    )
        else:
            # value has shape (H, action_dim)
            rl_sample[key] = value[:H] if value.dim() >= 2 else value
            if is_pad is not None:
                rl_sample[f"{key}_is_pad"] = is_pad[:H] if is_pad.dim() >= 1 else is_pad

    def _extract_reward_chunk(
        self,
        rl_sample: dict,
        key: str,
        value: torch.Tensor,
        is_pad: Optional[torch.Tensor],
        H: int,
    ):
        """Extract reward chunk from temporal tensor.

        Preserves the original key name.
        """
        rl_sample[key] = (
            value[:H] if value.dim() >= 1 and value.shape[0] >= H else value
        )
        if is_pad is not None:
            rl_sample[f"{key}_is_pad"] = is_pad[:H] if is_pad.shape[0] >= H else is_pad

    def _extract_done(
        self,
        rl_sample: dict,
        key: str,
        value: torch.Tensor,
        is_pad: Optional[torch.Tensor],
    ):
        """Extract terminal done flag at t+H.

        For offline datasets, we only fetch done at the action horizon end (t+H).
        This indicates whether the episode terminates within the action chunk.
        Preserves the original key name.
        """
        # Done is a single value at t+H
        if value.dim() >= 1:
            rl_sample[key] = value[0] if value.shape[0] >= 1 else value
        else:
            rl_sample[key] = value

        if is_pad is not None:
            if is_pad.dim() >= 1:
                rl_sample[f"{key}_is_pad"] = (
                    is_pad[0] if is_pad.shape[0] >= 1 else is_pad
                )
            else:
                rl_sample[f"{key}_is_pad"] = is_pad

    def _extract_obs_with_history(
        self,
        rl_sample: dict,
        key: str,
        value: torch.Tensor,
        is_pad: Optional[torch.Tensor],
        N: int,
        H: int,
    ):
        """Extract observation with history and next_obs.

        For current observation, preserves the original key name so that
        VLA transforms (repack, robot-specific) can find the data.

        For next observation (include_next_obs=True), stores data in a nested
        'next_observation_raw' dict with the SAME key structure as current obs.
        This allows applying the same VLA transforms to next obs.

        For history, uses modified key names with 'history_' prefix.
        """
        # Expected shape: (N + 1 + 1, ...) = (history + current + next_obs, ...)
        total_steps = N + 1 + (1 if self.rl_config.include_next_obs else 0)

        if value.dim() < 1:
            rl_sample[key] = value
            return

        # Clean key name for history output
        clean_key = key.replace("observation.", "").replace(".", "_")

        if value.shape[0] >= total_steps:
            # History: [:N]
            if N > 0:
                rl_sample[f"history_{clean_key}"] = value[:N]
                if is_pad is not None:
                    rl_sample[f"history_{clean_key}_is_pad"] = is_pad[:N]

            # Current: [N] - preserve original key name for VLA transforms
            rl_sample[key] = value[N]

            # Next obs: [N+1] - store in nested dict with SAME key structure
            # This allows applying the same VLA transforms to next obs
            if self.rl_config.include_next_obs:
                if "next_observation_raw" not in rl_sample:
                    rl_sample["next_observation_raw"] = {}
                    rl_sample["next_observation_is_pad"] = False

                # Use same key as current obs so VLA transforms work identically
                rl_sample["next_observation_raw"][key] = value[N + 1]

                if is_pad is not None and is_pad[N + 1]:
                    rl_sample["next_observation_is_pad"] = True
        else:
            # Fallback: just use as-is (preserve original key)
            rl_sample[key] = value


def create_rl_dataset(
    dataset_path: str | None = None,
    repo_id: str | None = None,
    # RL config (either provide this OR the individual params below)
    rl_config: Optional[RLDataConfig] = None,
    # Individual RL params (used only if rl_config is None)
    history_length: int = 0,
    history_keys: Optional[list[str]] = None,
    action_horizon: int = 10,
    include_next_obs: bool = True,
    include_return: bool = True,
    gamma: float = 0.99,
    discretize_return: bool = False,
    num_return_bins: int = 201,
    return_norm_stats_path: Optional[str] = None,
    return_min: Optional[float] = None,
    return_max: Optional[float] = None,
    # VLA config
    split: str = "train",
    data_config_factory: Optional[DataConfigFactory] = None,
    action_dim: Optional[int] = None,
    robot_type: Optional[str] = None,
    model_type: Optional[str] = None,
    default_prompt: Optional[str] = None,
    extra_delta_transform: bool = False,
    norm_stats_dir: Optional[str] = None,
    asset_id: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    max_samples: Optional[int] = None,
    # Episode filtering
    episode_percentage: Optional[float] = None,
    shuffle_episodes: bool = False,
    episode_seed: int = 42,
) -> LeRobotRLDataset:
    """Create an RL dataset for value learning.

    Args:
        dataset_path: Path to LeRobot dataset
        repo_id: Alias for dataset_path
        rl_config: Complete RL config. If provided, individual RL params are ignored.
        history_length: Number of past observations (0 = current only)
        history_keys: Keys to include in history (auto-detected if None)
        action_horizon: Number of future actions/rewards
        include_next_obs: Fetch obs at t+H for bootstrapping
        include_return: Include precomputed return
        gamma: Discount factor
        discretize_return: Whether to discretize return into bins
        num_return_bins: Number of bins for discretization
        return_norm_stats_path: Path to norm_stats.json for min/max
        return_min: Override minimum return value
        return_max: Override maximum return value
        split: Dataset split
        data_config_factory: Factory for transforms
        action_dim: Action dimension
        robot_type: Robot type (franka, libero, etc.)
        model_type: Model type (pi0, pi05)
        default_prompt: Default prompt
        extra_delta_transform: Apply extra delta transform
        norm_stats_dir: Normalization stats directory
        asset_id: Asset ID
        config: Full config dict from YAML
        max_samples: Limit dataset size
        episode_percentage: Use subset of episodes
        shuffle_episodes: Random episode selection
        episode_seed: Seed for reproducibility

    Returns:
        LeRobotRLDataset instance
    """
    # Build rl_config from individual params if not provided
    if rl_config is None:
        rl_config = create_rl_config(
            history_length=history_length,
            history_keys=history_keys,
            action_horizon=action_horizon,
            include_next_obs=include_next_obs,
            include_return=include_return,
            gamma=gamma,
            discretize_return=discretize_return,
            num_return_bins=num_return_bins,
            return_norm_stats_path=return_norm_stats_path,
            return_min=return_min,
            return_max=return_max,
        )

    return LeRobotRLDataset(
        dataset_path=dataset_path or repo_id,
        rl_config=rl_config,
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
