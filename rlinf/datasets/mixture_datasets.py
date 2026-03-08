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
Mixture datasets for value and advantage training.

This module provides:
- AdvantageMixtureDataset: for CFG-RL training with binary advantage labels
- ValueMixtureDataset: for training value models on multiple datasets
"""

import hashlib
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from torch.utils.data import Dataset

from rlinf.utils.dist_utils import get_logger

logger = get_logger(__name__)


@runtime_checkable
class SizedDataset(Protocol):
    """Protocol for datasets that have both __len__ and __getitem__."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


def _safe_hash(input_tuple) -> int:
    """Create a deterministic hash for seeding RNG."""
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)
    seed = int(sha256.hexdigest(), 16)
    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class _MixtureBase(Dataset):
    """Shared logic for mixture datasets."""

    def __init__(
        self,
        datasets: Sequence[tuple[SizedDataset, float]],
        mode: str = "train",
        balance_dataset_weights: bool = True,
        seed: int = 42,
    ):
        # Filter out empty datasets
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
        if self.mode != "train":
            seed = index
        else:
            seed = _safe_hash((self._epoch, index, self.seed))

        rng = np.random.default_rng(seed)
        ds_idx = rng.choice(len(self.datasets), p=self._dataset_sampling_weights)
        dataset = self.datasets[ds_idx]
        sample_idx = int(rng.integers(0, len(dataset)))

        return dataset, sample_idx

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset, sample_idx = self._sample_step(index)
        return dataset[sample_idx]

    def get_custom_tokens(self) -> Optional[list[str]]:
        tokens = set()
        for ds in self.datasets:
            if hasattr(ds, "get_custom_tokens"):
                ds_tokens = ds.get_custom_tokens()
                if ds_tokens:
                    tokens.update(ds_tokens)
        return list(tokens) if tokens else None


class AdvantageMixtureDataset(_MixtureBase):
    """Mixture of multiple datasets with weighted sampling for CFG-RL training."""

    def __init__(
        self,
        datasets: Sequence[tuple[SizedDataset, float]],
        mode: str = "train",
        balance_dataset_weights: bool = True,
        seed: int = 42,
    ):
        super().__init__(datasets, mode, balance_dataset_weights, seed)

        logger.info("AdvantageMixtureDataset initialized:")
        logger.info(f"  Datasets: {len(self.datasets)}")
        logger.info(f"  Total samples: {sum(self._dataset_lengths)}")
        logger.info(f"  Dataset lengths: {self._dataset_lengths.tolist()}")
        logger.info(f"  Raw weights: {self._raw_weights.tolist()}")
        logger.info(f"  Sampling weights: {self._dataset_sampling_weights.tolist()}")
        logger.info(f"  Mode: {mode}")

    def get_train_val_split(
        self, validation_split: Optional[float] = None
    ) -> tuple["AdvantageMixtureDataset", "AdvantageMixtureDataset"]:
        if validation_split is None:
            validation_split = 0.1

        train_datasets = []
        val_datasets = []

        for ds, weight in zip(self.datasets, self._raw_weights):
            if hasattr(ds, "get_train_val_split"):
                train_ds, val_ds = ds.get_train_val_split(validation_split)
            else:
                total = len(ds)
                val_size = int(total * validation_split)
                indices = np.arange(total)
                np.random.shuffle(indices)
                from torch.utils.data import Subset

                val_ds = Subset(ds, indices[:val_size].tolist())
                train_ds = Subset(ds, indices[val_size:].tolist())

            train_datasets.append((train_ds, weight))
            val_datasets.append((val_ds, weight))

        train_mixture = AdvantageMixtureDataset(
            datasets=train_datasets,
            mode="train",
            balance_dataset_weights=self.balance_dataset_weights,
            seed=self.seed,
        )
        val_mixture = AdvantageMixtureDataset(
            datasets=val_datasets,
            mode="val",
            balance_dataset_weights=self.balance_dataset_weights,
            seed=self.seed,
        )

        return train_mixture, val_mixture

    def get_source_name(self) -> str:
        names = []
        for ds in self.datasets:
            if hasattr(ds, "get_source_name"):
                names.append(ds.get_source_name())
            elif hasattr(ds, "repo_id"):
                names.append(ds.repo_id)
        return "_".join(names[:3]) if names else "advantage_mixture"


class ValueMixtureDataset(_MixtureBase):
    """Mixture of multiple ValueDatasets with weighted sampling."""

    def __init__(
        self,
        datasets: Sequence[tuple[SizedDataset, float]],
        mode: str = "train",
        balance_dataset_weights: bool = True,
        seed: int = 42,
    ):
        super().__init__(datasets, mode, balance_dataset_weights, seed)

        logger.info("ValueMixtureDataset initialized:")
        logger.info(f"  Datasets: {len(self.datasets)}")
        logger.info(f"  Total samples: {sum(self._dataset_lengths)}")
        logger.info(f"  Dataset lengths: {self._dataset_lengths.tolist()}")
        logger.info(f"  Raw weights: {self._raw_weights.tolist()}")
        logger.info(f"  Sampling weights: {self._dataset_sampling_weights.tolist()}")
        logger.info(f"  Mode: {mode}")

    def get_train_val_split(
        self, validation_split: Optional[float] = None
    ) -> tuple["ValueMixtureDataset", "ValueMixtureDataset"]:
        if validation_split is None:
            validation_split = 0.1

        train_datasets = []
        val_datasets = []

        for ds, weight in zip(self.datasets, self._raw_weights):
            if hasattr(ds, "get_train_val_split"):
                train_ds, val_ds = ds.get_train_val_split(validation_split)
            else:
                total = len(ds)
                val_size = int(total * validation_split)
                indices = np.arange(total)
                np.random.shuffle(indices)
                from torch.utils.data import Subset

                val_ds = Subset(ds, indices[:val_size].tolist())
                train_ds = Subset(ds, indices[val_size:].tolist())

            train_datasets.append((train_ds, weight))
            val_datasets.append((val_ds, weight))

        train_mixture = ValueMixtureDataset(
            datasets=train_datasets,
            mode="train",
            balance_dataset_weights=self.balance_dataset_weights,
            seed=self.seed,
        )
        val_mixture = ValueMixtureDataset(
            datasets=val_datasets,
            mode="val",
            balance_dataset_weights=self.balance_dataset_weights,
            seed=self.seed,
        )

        return train_mixture, val_mixture

    def get_source_name(self) -> str:
        names = []
        for ds in self.datasets:
            if hasattr(ds, "get_source_name"):
                names.append(ds.get_source_name())
            elif hasattr(ds, "repo_id"):
                names.append(ds.repo_id)
        return "_".join(names[:3]) if names else "value_mixture"
