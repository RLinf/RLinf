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

"""DataLoader helpers for online DAgger LeRobot training."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from rlinf.data.datasets.dagger.dataset import RollingLeRobotDataset
from rlinf.utils.logging import get_logger

logger = get_logger()


class OnlineOfflineLeRobotDataset(Dataset):
    """Expose static offline and growing online datasets as one index space.

    Offline samples occupy ``[0, offline_size)`` and online samples occupy the
    remaining indices. The offline prefix is stable while the online dataset
    grows, so an existing DataLoader can safely sample newly appended online
    frames without rebuilding a ``ConcatDataset``.
    """

    def __init__(self, offline_dataset: Dataset, online_dataset: Dataset) -> None:
        self.offline_dataset = offline_dataset
        self.online_dataset = online_dataset

    @property
    def offline_size(self) -> int:
        return len(self.offline_dataset)

    @property
    def online_size(self) -> int:
        return len(self.online_dataset)

    def __len__(self) -> int:
        return self.offline_size + self.online_size

    @staticmethod
    def _training_fields(sample: Any) -> Any:
        if not isinstance(sample, dict):
            return sample
        exact_keys = {"state", "actions", "task", "image", "wrist_image"}
        return {
            key: value
            for key, value in sample.items()
            if key in exact_keys
            or key.startswith("wrist_image-")
            or key.startswith("extra_view_image")
        }

    def __getitem__(self, index: int):
        index = int(index)
        offline_size = self.offline_size
        if index < 0 or index >= offline_size + self.online_size:
            raise IndexError(index)
        if index < offline_size:
            sample = self.offline_dataset[index]
        else:
            sample = self.online_dataset[index - offline_size]
        return self._training_fields(sample)

    def __getitems__(self, indices: list[int]) -> list[Any]:
        return [self[index] for index in indices]


class RandomReplacementSampler(Sampler):
    """Sampler that randomly samples indices with replacement.

    Unlike DistributedSampler which iterates through the dataset without
    replacement, this sampler can sample the same index multiple times,
    making it suitable for small datasets with large batch sizes.

    This sampler is useful when you want to sample more data points than
    exist in the dataset (e.g., batch_size > dataset_size), which is common
    when using replay buffers or rolling datasets in RL training.

    Args:
        dataset: Dataset to sample from.
        num_samples: Number of samples to draw per epoch. If None, defaults
            to len(dataset). Can be set larger than len(dataset).
        seed: Random seed for reproducibility. If None, uses random state.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        if self.seed is not None:
            g.manual_seed(self.seed + self.epoch)

        indices = torch.randint(
            low=0,
            high=len(self.dataset),
            size=(self.num_samples,),
            generator=g,
            dtype=torch.int64,
        )

        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across epochs."""
        self.epoch = epoch


class DistributedRandomReplacementSampler(Sampler):
    """Distributed version of RandomReplacementSampler.

    Each rank samples from the full dataset with replacement, but uses
    a different random seed to ensure different samples across ranks.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_samples: int | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

        total_samples = num_samples if num_samples is not None else len(dataset)
        self.num_samples = total_samples // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * self.num_replicas + self.rank)

        indices = torch.randint(
            low=0,
            high=len(self.dataset),
            size=(self.num_samples,),
            generator=g,
            dtype=torch.int64,
        )

        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across epochs."""
        self.epoch = epoch


def build_dataloader_from_dataset(
    dataset: RollingLeRobotDataset,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 4,
    drop_last: bool = True,
    pin_memory: bool = True,
    use_random_replacement: bool = False,
    num_samples_per_epoch: int | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> DataLoader:
    """Build a :class:`DataLoader` from a :class:`RollingLeRobotDataset`.

    By default, uses :class:`~torch.utils.data.distributed.DistributedSampler`
    which samples without replacement. Set ``use_random_replacement=True`` to
    use :class:`RandomReplacementSampler` which samples with replacement,
    allowing batch sizes larger than the dataset size.
    """
    if use_random_replacement:
        if world_size > 1:
            sampler = DistributedRandomReplacementSampler(
                dataset,
                num_samples=num_samples_per_epoch,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
            )
        else:
            sampler = RandomReplacementSampler(
                dataset,
                num_samples=num_samples_per_epoch,
                seed=seed,
            )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    logger.info(
        "[build_dataloader_from_dataset] batch_size=%d, world_size=%d, "
        "rank=%d, sub_datasets=%d, total_frames=%d, sampler=%s, "
        "sampler_length=%d",
        batch_size,
        world_size,
        rank,
        len(dataset._sub_datasets),
        len(dataset),
        sampler.__class__.__name__,
        len(sampler),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        **kwargs,
    )


def build_combined_dataloader_from_datasets(
    online_dataset: Dataset,
    offline_dataset: Dataset,
    batch_size: int,
    num_samples_per_epoch: int,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 0,
    drop_last: bool = True,
    pin_memory: bool = True,
    seed: int = 42,
    **kwargs: Any,
) -> DataLoader:
    """Build one uniformly sampled pool from offline and online DAgger data."""
    dataset = OnlineOfflineLeRobotDataset(
        offline_dataset=offline_dataset,
        online_dataset=online_dataset,
    )
    if world_size > 1:
        sampler = DistributedRandomReplacementSampler(
            dataset,
            num_samples=num_samples_per_epoch,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
        )
    else:
        sampler = RandomReplacementSampler(
            dataset,
            num_samples=num_samples_per_epoch,
            seed=seed,
        )
    logger.info(
        "[build_combined_dataloader_from_datasets] batch_size=%d, world_size=%d, "
        "rank=%d, offline_samples=%d, online_samples=%d, total_samples=%d, "
        "sampler_length=%d",
        batch_size,
        world_size,
        rank,
        dataset.offline_size,
        dataset.online_size,
        len(dataset),
        len(sampler),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        **kwargs,
    )
