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

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from rlinf.datasets.lerobot.config import (
    DataConfigFactory,
    create_data_config_factory,
    create_data_config_factory_from_dict,
)
from rlinf.datasets.lerobot.lerobot_dataset import (
    LeRobotPyTorchDataset,
    TransformedDataset,
)
from rlinf.datasets.lerobot.transforms import (
    DataTransformFn,
    Normalize,
    PromptFromLeRobotTask,
    RepackTransform,
    compose,
    load_task_descriptions,
)

from .config import RLDataConfig, create_rl_config
from .value_transforms import ReturnNormalizer

logger = logging.getLogger(__name__)


def load_return_stats_from_dataset(
    dataset_path: str | Path,
) -> tuple[float | None, float | None]:
    """Load return min/max from dataset's stats.json.

    Args:
        dataset_path: Path to LeRobot dataset

    Returns:
        Tuple of (return_min, return_max), or (None, None) if not found
    """
    import json

    stats_path = Path(dataset_path) / "meta" / "stats.json"
    if not stats_path.exists():
        return None, None

    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return_stats = stats.get("return", {})
        return return_stats.get("min"), return_stats.get("max")
    except (json.JSONDecodeError, KeyError):
        return None, None


def load_returns_sidecar(
    dataset_path: str | Path,
    returns_tag: str | None = None,
) -> dict[int, dict[str, np.ndarray]] | None:
    """Load ``meta/returns_{tag}.parquet`` sidecar written by compute_returns.py.

    Falls back to ``meta/returns.parquet`` when *returns_tag* is None.

    Returns:
        ``{episode_index: {"return": np.array, "reward": np.array}}``
        or None if sidecar does not exist.
    """
    import pyarrow.parquet as pq

    dataset_path = Path(dataset_path)
    sidecar_filename = (
        f"returns_{returns_tag}.parquet" if returns_tag else "returns.parquet"
    )
    sidecar_path = dataset_path / "meta" / sidecar_filename
    if not sidecar_path.exists():
        return None

    table = pq.read_table(str(sidecar_path))
    ep_col = table.column("episode_index").to_numpy()
    frame_col = table.column("frame_index").to_numpy()
    ret_col = table.column("return").to_numpy()
    rew_col = table.column("reward").to_numpy()

    sidecar: dict[int, dict[str, np.ndarray]] = {}
    for ep in np.unique(ep_col):
        mask = ep_col == ep
        frames = frame_col[mask]
        order = np.argsort(frames)
        sidecar[int(ep)] = {
            "return": ret_col[mask][order].astype(np.float32),
            "reward": rew_col[mask][order].astype(np.float32),
        }

    logger.info(f"Loaded returns sidecar: {sidecar_path} ({len(sidecar)} episodes)")
    return sidecar


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
        # Sidecar tag (e.g. returns_{tag}.parquet)
        tag: Optional[str] = None,
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

        self.tag = tag
        self.episode_percentage = episode_percentage
        self.shuffle_episodes = shuffle_episodes
        self.episode_seed = episode_seed
        self._episode_indices = None
        self._sample_indices = None

        self.is_local = self._is_local_path(self.repo_id)

        if self.is_local:
            local_path = Path(self.repo_id).resolve()
            folder_name = local_path.name
            self.dataset_meta = LeRobotDatasetMetadata(folder_name, root=local_path)
        else:
            self.dataset_meta = LeRobotDatasetMetadata(self.repo_id)

        if rl_config is not None:
            self.rl_config = rl_config
            logger.info("Using provided rl_config")
        else:
            self.rl_config = create_rl_config(
                history_keys=self._auto_detect_history_keys(),
            )
            logger.info("Created default rl_config with auto-detected history_keys")

        if self.rl_config.history_keys is None:
            detected_keys = self._auto_detect_history_keys()
            self.rl_config = replace(self.rl_config, history_keys=tuple(detected_keys))
            logger.info(f"Auto-detected history_keys: {detected_keys}")

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

        delta_timestamps = self.rl_config.get_delta_timestamps(self.dataset_meta.fps)

        # Load sidecar returns.parquet if it exists (written by compute_returns.py).
        # When loaded, remove return/reward from delta_timestamps so LeRobot
        # does not try to read these (possibly absent) columns from episode parquets.
        self._returns_sidecar = None
        if self.is_local:
            self._returns_sidecar = load_returns_sidecar(
                Path(self.repo_id).resolve(), self.tag
            )
            if self._returns_sidecar is not None:
                delta_timestamps.pop(self.rl_config.return_key, None)
                for rk in self.rl_config.reward_keys:
                    delta_timestamps.pop(rk, None)

        logger.info(f"RL Delta timestamps: {delta_timestamps}")

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

        self._compute_episode_filtering()
        self._add_prompt_transform()

        # VLA transforms are applied AFTER restructuring in __getitem__ because RL
        # datasets have temporal structure and VLA transforms expect single-timestep data
        transforms = self._create_transform_list()
        self._vla_transform = compose(transforms) if transforms else None

        self.return_normalizer = None
        if self.rl_config.normalize_return:
            self.return_normalizer = self._create_return_normalizer()

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
            skip_norm = norm_stats_dir is None
            return factory.create(
                action_dim=action_dim or 32, skip_norm_stats=skip_norm
            )
        return None

    def _add_prompt_transform(self):
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
        """Create transform list for RL datasets.

        Repack transforms use passthrough_unmapped=True so RL-specific keys
        (action_chunk, reward_chunk, etc.) are preserved through the pipeline.
        """
        transforms = []

        if self.data_config is not None:
            for transform in self.data_config.repack_transforms.inputs:
                if isinstance(transform, RepackTransform):
                    transforms.append(
                        RepackTransform(transform.structure, passthrough_unmapped=True)
                    )
                else:
                    transforms.append(transform)

            transforms.extend(self.data_config.data_transforms.inputs)

            # Exclude 'return' from normalization since ReturnNormalizer handles its own
            if self.data_config.norm_stats is not None:
                norm_stats = self.data_config.norm_stats
                if (
                    self.rl_config.normalize_return
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

            transforms.extend(self.data_config.model_transforms.inputs)

        return transforms

    def _create_return_normalizer(self) -> Optional[ReturnNormalizer]:
        if not self.rl_config.normalize_return:
            return None

        common_kwargs = {
            "return_key": self.rl_config.return_key,
            "keep_continuous": self.rl_config.keep_continuous_return,
            "normalize_to_minus_one_zero": self.rl_config.normalize_to_minus_one_zero,
        }

        if (
            self.rl_config.return_min is not None
            and self.rl_config.return_max is not None
        ):
            return ReturnNormalizer(
                return_min=self.rl_config.return_min,
                return_max=self.rl_config.return_max,
                **common_kwargs,
            )
        elif self.rl_config.return_norm_stats_path:
            return ReturnNormalizer(
                norm_stats_path=Path(self.rl_config.return_norm_stats_path),
                **common_kwargs,
            )
        else:
            logger.warning(
                "Return normalization enabled but no min/max or norm_stats_path "
                "provided. Normalization will be skipped."
            )
            return None

    def _log_dataset_info(self):
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

        if self._sample_indices is not None:
            idx = self._sample_indices[idx]

        raw_sample = self.base_dataset[idx]
        rl_sample = self._restructure_sample(raw_sample)

        next_obs_raw = rl_sample.pop("next_observation_raw", None)
        next_obs_is_pad = rl_sample.pop("next_observation_is_pad", False)

        if self._vla_transform is not None:
            rl_sample = self._vla_transform(rl_sample)

            if next_obs_raw is not None:
                if "prompt" in rl_sample:
                    next_obs_raw["prompt"] = rl_sample["prompt"]

                try:
                    processed_next = self._vla_transform(next_obs_raw)

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

        if self.return_normalizer is not None:
            rl_sample = self.return_normalizer(rl_sample)

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

        metadata_keys = {
            "prompt",
            "task",
            "episode_index",
            "frame_index",
            "index",
            "timestamp",
            "task_index",
        }

        for key, value in sample.items():
            if key.endswith("_is_pad"):
                continue

            if key in metadata_keys:
                rl_sample[key] = value
                continue

            pad_key = f"{key}_is_pad"
            is_pad = sample.get(pad_key, None)

            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                if key in self.rl_config.action_keys:
                    self._extract_action_chunk(rl_sample, key, value, is_pad, N, H)
                elif key in self.rl_config.reward_keys:
                    self._extract_reward_chunk(rl_sample, key, value, is_pad, H)
                elif key == self.rl_config.done_key:
                    self._extract_done(rl_sample, key, value, is_pad)
                elif key == self.rl_config.return_key:
                    rl_sample[key] = value.squeeze() if value.dim() > 0 else value
                    if is_pad is not None:
                        rl_sample[f"{key}_is_pad"] = (
                            is_pad.squeeze() if is_pad.dim() > 0 else is_pad
                        )
                elif key in (self.rl_config.history_keys or []):
                    self._extract_obs_with_history(rl_sample, key, value, is_pad, N, H)
                else:
                    rl_sample[key] = value
                    if is_pad is not None:
                        rl_sample[f"{key}_is_pad"] = is_pad
            else:
                rl_sample[key] = value

        if self._returns_sidecar is not None:
            ep_idx = int(rl_sample.get("episode_index", -1))
            frame_idx = int(rl_sample.get("frame_index", -1))
            if ep_idx in self._returns_sidecar:
                ep_data = self._returns_sidecar[ep_idx]

                rl_sample[self.rl_config.return_key] = torch.tensor(
                    ep_data["return"][frame_idx], dtype=torch.float32
                )

                ep_rewards = ep_data["reward"]
                ep_len = len(ep_rewards)
                for rk in self.rl_config.reward_keys:
                    reward_chunk = torch.zeros(H, dtype=torch.float32)
                    reward_is_pad = torch.ones(H, dtype=torch.bool)
                    end_idx = min(frame_idx + H, ep_len)
                    valid_len = end_idx - frame_idx
                    if valid_len > 0:
                        reward_chunk[:valid_len] = torch.from_numpy(
                            ep_rewards[frame_idx:end_idx].copy()
                        )
                        reward_is_pad[:valid_len] = False
                    rl_sample[rk] = reward_chunk
                    rl_sample[f"{rk}_is_pad"] = reward_is_pad

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
        """Extract action chunk, preserving the original key name for VLA transforms."""
        if self.rl_config.include_history_actions:
            if value.dim() >= 2 and value.shape[0] >= N + H:
                rl_sample[f"history_{key}"] = value[:N]
                rl_sample[key] = value[N : N + H]
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
        """Extract terminal done flag at t+H (episode terminates within action chunk)."""
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

        Current obs preserves the original key name for VLA transforms.
        Next obs is stored in 'next_observation_raw' with the same key structure
        so the same VLA transforms can be applied. History uses 'history_' prefix.
        """
        total_steps = N + 1 + (1 if self.rl_config.include_next_obs else 0)

        if value.dim() < 1:
            rl_sample[key] = value
            return

        clean_key = key.replace("observation.", "").replace(".", "_")

        if value.shape[0] >= total_steps:
            if N > 0:
                rl_sample[f"history_{clean_key}"] = value[:N]
                if is_pad is not None:
                    rl_sample[f"history_{clean_key}_is_pad"] = is_pad[:N]

            rl_sample[key] = value[N]

            if self.rl_config.include_next_obs:
                if "next_observation_raw" not in rl_sample:
                    rl_sample["next_observation_raw"] = {}
                    rl_sample["next_observation_is_pad"] = False

                rl_sample["next_observation_raw"][key] = value[N + 1]

                if is_pad is not None and is_pad[N + 1]:
                    rl_sample["next_observation_is_pad"] = True
        else:
            rl_sample[key] = value
