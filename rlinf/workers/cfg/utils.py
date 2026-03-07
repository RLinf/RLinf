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

"""Shared utilities for CFG (Classifier-Free Guidance) workers.

Contains data loading and dataset wrapper classes used by both FSDPCfgWorker
and DebugCFGFSDPActor to avoid code duplication.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    Observation as CFGObservation,
)

logger = logging.getLogger(__name__)


def cast_image_features(hf_dataset):
    """Cast image columns from struct to Image type for proper decoding.

    When parquet files store images as struct<bytes: binary, path: string>,
    we need to cast them to datasets.Image type for automatic decoding.

    Args:
        hf_dataset: HuggingFace dataset with struct-type image columns.

    Returns:
        Dataset with image columns cast to Image type.
    """
    from datasets import Image

    # Check if casting is needed
    features = hf_dataset.features
    needs_cast = False
    new_features = features.copy()

    for key, feat in features.items():
        # Check if this is a struct-type image (dict feature with 'bytes' field)
        # The feature type will be a dict like {'bytes': Value(...), 'path': Value(...)}
        if isinstance(feat, dict) and "bytes" in feat:
            new_features[key] = Image()
            needs_cast = True

    if needs_cast:
        from lerobot.datasets.utils import hf_transform_to_torch

        hf_dataset = hf_dataset.cast(new_features)
        hf_dataset.set_transform(hf_transform_to_torch)

    return hf_dataset


class DatasetWithAdvantage:
    """Wrapper to preserve advantage through OpenPI transform pipeline.

    OpenPI's RepackTransform removes all keys except required ones, which drops
    the advantage field. This wrapper pre-builds an index-to-advantage mapping
    at init time using efficient HF dataset column access (no image loading),
    avoiding the need to load each sample twice.

    Attributes:
        _transformed_dataset: Dataset after applying OpenPI transforms.
        _advantage_by_index: Pre-built mapping from sample index to advantage value.
        _base_dataset: Kept only as fallback when pre-building fails.
    """

    def __init__(
        self,
        base_dataset: Any,
        transformed_dataset: Any,
        advantages_lookup: dict[tuple[int, int], bool] | None = None,
    ):
        """Initialize DatasetWithAdvantage.

        Pre-builds index-to-advantage mapping to avoid loading each sample twice
        (once from base_dataset for advantage, once from transformed_dataset).

        Args:
            base_dataset: Base dataset with advantage field (from compute_advantages.py).
            transformed_dataset: Dataset after applying OpenPI transforms.
            advantages_lookup: Optional pre-loaded advantage lookup from
                meta/advantages_{tag}.parquet. If provided, advantage is read
                from this lookup instead of from the data parquet.
        """
        self._transformed_dataset = transformed_dataset
        self._advantage_by_index = self._build_advantage_index(
            base_dataset, advantages_lookup
        )
        # Keep base_dataset only as fallback when pre-building fails
        self._base_dataset = base_dataset if self._advantage_by_index is None else None

    @staticmethod
    def _get_hf_dataset(dataset: Any) -> Any:
        """Extract the underlying HuggingFace dataset from wrapped datasets.

        Traverses TransformedDataset wrappers to find the LeRobotDataset's
        hf_dataset, which allows efficient column access without image loading.
        """
        current = dataset
        while current is not None:
            if hasattr(current, "hf_dataset"):
                return current.hf_dataset
            elif hasattr(current, "_dataset"):
                current = current._dataset
            else:
                return None
        return None

    def _build_advantage_index(
        self,
        base_dataset: Any,
        advantages_lookup: dict[tuple[int, int], bool] | None,
    ) -> dict[int, bool] | None:
        """Build mapping from sample index to advantage value.

        Uses efficient column access on the underlying HF dataset to read
        episode_index/frame_index or advantage columns without loading images.

        Returns:
            Dict mapping sample index -> advantage (bool), or None if
            the HF dataset is not accessible (falls back to slow path).
        """
        hf_dataset = self._get_hf_dataset(base_dataset)
        if hf_dataset is None:
            logger.warning(
                "Cannot access underlying HF dataset, "
                "falling back to per-sample advantage loading (slower)."
            )
            return None

        if advantages_lookup is not None:
            # Efficient path: read episode_index and frame_index columns directly
            # (no image decoding, just integer columns)
            ep_indices = hf_dataset["episode_index"]
            frame_indices = hf_dataset["frame_index"]
            advantage_by_index = {}
            missing_keys = []
            for i in range(len(hf_dataset)):
                key = (int(ep_indices[i]), int(frame_indices[i]))
                if key in advantages_lookup:
                    advantage_by_index[i] = advantages_lookup[key]
                else:
                    missing_keys.append(key)
            if missing_keys:
                raise ValueError(
                    f"[DatasetWithAdvantage] {len(missing_keys)} samples not found "
                    f"in advantages lookup (first 5: {missing_keys[:5]}). "
                    f"The advantages parquet does not match this dataset. "
                    f"Re-run compute_advantages.py."
                )
            return advantage_by_index

        elif "advantage" in hf_dataset.column_names:
            # Fallback: read advantage column directly (no image decoding)
            advantages = hf_dataset["advantage"]
            return {i: bool(v) for i, v in enumerate(advantages)}

        else:
            raise ValueError(
                "[DatasetWithAdvantage] No advantage data found: "
                "advantages_lookup is None, and 'advantage' column not in dataset. "
                "Run compute_advantages.py first."
            )

    def __len__(self) -> int:
        return len(self._transformed_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get sample with advantage added.

        Only loads from _transformed_dataset once (no double data loading).

        Args:
            idx: Sample index.

        Returns:
            Transformed sample dict with 'advantage' field added.
        """
        sample = self._transformed_dataset[idx]

        if self._advantage_by_index is not None:
            if idx not in self._advantage_by_index:
                raise KeyError(
                    f"[DatasetWithAdvantage] Index {idx} not found in advantage index. "
                    f"Dataset size: {len(self._transformed_dataset)}, "
                    f"advantage index size: {len(self._advantage_by_index)}."
                )
            sample["advantage"] = self._advantage_by_index[idx]
        else:
            # Slow fallback: load from base dataset (only when HF dataset not accessible)
            base_sample = self._base_dataset[idx]
            if "advantage" not in base_sample:
                raise KeyError(
                    f"[DatasetWithAdvantage] 'advantage' key not found in base_sample "
                    f"at index {idx}. Run compute_advantages.py first."
                )
            advantage = base_sample["advantage"]
            if isinstance(advantage, torch.Tensor):
                advantage = bool(advantage.item())
            sample["advantage"] = advantage

        return sample


class CFGDataLoaderImpl:
    """DataLoader wrapper for CFG training.

    Yields (observation, actions, advantage) tuples for CFG model training.
    The advantage field is used to select positive or negative guidance.

    Attributes:
        _data_config: OpenPI data configuration.
        _data_loader: Underlying PyTorch DataLoader.
    """

    def __init__(self, data_config: Any, data_loader: Any):
        """Initialize CFGDataLoaderImpl.

        Args:
            data_config: OpenPI data configuration.
            data_loader: Underlying PyTorch DataLoader.
        """
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> Any:
        """Get data configuration."""
        return self._data_config

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        """Forward set_epoch to sampler and dataset for proper shuffling each epoch."""
        if hasattr(self._data_loader, "sampler") and hasattr(
            self._data_loader.sampler, "set_epoch"
        ):
            self._data_loader.sampler.set_epoch(epoch)
        # AdvantageMixtureDataset has set_epoch method
        if hasattr(self._data_loader, "dataset") and hasattr(
            self._data_loader.dataset, "set_epoch"
        ):
            self._data_loader.dataset.set_epoch(epoch)

    def __iter__(self):
        """Iterate over batches.

        Yields:
            Tuple of (observation, actions, advantage) for each batch.
        """
        for batch in self._data_loader:
            observation = CFGObservation.from_dict(batch)
            actions = batch["actions"]

            advantage = batch["advantage"]
            if not isinstance(advantage, torch.Tensor):
                advantage = torch.tensor(advantage, dtype=torch.bool)

            yield observation, actions, advantage
