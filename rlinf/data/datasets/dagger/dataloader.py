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
