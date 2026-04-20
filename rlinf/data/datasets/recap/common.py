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

"""Common dataloader and mixture dataset helpers for ReCap."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@runtime_checkable
class SizedDataset(Protocol):
    """Protocol for datasets that have both ``__len__`` and ``__getitem__``."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


def forward_set_epoch(data_loader: Any, epoch: int) -> None:
    """Forward ``set_epoch`` to the wrapped sampler and dataset when present."""
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)


class BaseDataLoaderImpl:
    """Lightweight wrapper around a torch dataloader with epoch forwarding."""

    def __init__(self, data_config: Any, data_loader: Any):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> Any:
        return self._data_config

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        forward_set_epoch(self._data_loader, epoch)


def _safe_hash(input_tuple) -> int:
    """Create a deterministic hash for seeding RNG."""
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)
    seed = int(sha256.hexdigest(), 16)
    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class ReCapMixtureDataset(Dataset):
    """Shared weighted-mixture sampling dataset for ReCap pipelines."""

    mixture_name = "ReCapMixtureDataset"

    def __init__(
        self,
        datasets: Sequence[tuple[SizedDataset, float]],
        mode: str = "train",
        balance_dataset_weights: bool = True,
        seed: int = 42,
    ):
        valid_datasets = []
        for ds, weight in datasets:
            if len(ds) == 0:
                logger.warning("Skipping empty dataset")
                continue
            valid_datasets.append((ds, weight))

        if not valid_datasets:
            raise ValueError("No valid (non-empty) datasets provided")

        self.datasets = [ds for ds, _ in valid_datasets]
        self._raw_weights = np.array([w for _, w in valid_datasets], dtype=np.float32)
        self.mode = mode
        self.balance_dataset_weights = balance_dataset_weights
        self.seed = seed

        self._dataset_lengths = np.array([len(ds) for ds in self.datasets])

        self._dataset_sampling_weights = self._raw_weights.copy()
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths

        weight_sum = self._dataset_sampling_weights.sum()
        if weight_sum <= 0 or np.isnan(weight_sum):
            logger.warning(f"Invalid weight sum {weight_sum}, using uniform weights")
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(
                self.datasets
            )
        else:
            self._dataset_sampling_weights /= weight_sum

        self._primary_indices = self._raw_weights == 1.0
        if not np.any(self._primary_indices):
            max_weight = self._raw_weights.max()
            self._primary_indices = self._raw_weights == max_weight

        self._epoch = 0
        self._log_initialization()

    def _log_initialization(self) -> None:
        logger.info(f"{self.mixture_name} initialized:")
        logger.info(f"  Datasets: {len(self.datasets)}")
        logger.info(f"  Total samples: {sum(self._dataset_lengths)}")
        logger.info(f"  Dataset lengths: {self._dataset_lengths.tolist()}")
        logger.info(f"  Raw weights: {self._raw_weights.tolist()}")
        logger.info(f"  Sampling weights: {self._dataset_sampling_weights.tolist()}")
        logger.info(f"  Mode: {self.mode}")

    @property
    def dataset_lengths(self) -> np.ndarray:
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        return self._dataset_sampling_weights

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        primary_lengths = self._dataset_lengths[self._primary_indices]
        primary_weights = self._dataset_sampling_weights[self._primary_indices]

        valid_mask = primary_weights > 0
        if not np.any(valid_mask):
            return int(self._dataset_lengths.sum())

        ratios = primary_lengths[valid_mask] / primary_weights[valid_mask]
        return int(ratios.max())

    def _sample_step(self, index: int) -> tuple[SizedDataset, int]:
        seed = (
            index
            if self.mode != "train"
            else _safe_hash((self._epoch, index, self.seed))
        )
        rng = np.random.default_rng(seed)
        ds_idx = rng.choice(len(self.datasets), p=self._dataset_sampling_weights)
        dataset = self.datasets[ds_idx]
        sample_idx = int(rng.integers(0, len(dataset)))
        return dataset, sample_idx

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset, sample_idx = self._sample_step(index)
        return dataset[sample_idx]
