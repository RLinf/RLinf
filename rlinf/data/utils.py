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


from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from rlinf.utils.logging import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Random Replacement Samplers
# ---------------------------------------------------------------------------


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
        # Create generator with seed for reproducibility
        g = torch.Generator()
        if self.seed is not None:
            g.manual_seed(self.seed + self.epoch)

        # Sample indices with replacement
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

    This is useful for distributed training where each process needs to
    sample different data, but all processes sample from the same dataset
    with replacement.

    Args:
        dataset: Dataset to sample from.
        num_samples: Total number of samples across all ranks. Each rank
            will sample num_samples // num_replicas samples. If None,
            defaults to len(dataset).
        num_replicas: Number of distributed processes. If None, uses
            torch.distributed.get_world_size().
        rank: Rank of current process. If None, uses
            torch.distributed.get_rank().
        seed: Base random seed. Each rank uses seed + epoch * num_replicas + rank.
        shuffle: Unused, kept for API compatibility with DistributedSampler.
        valid_indices: When provided, sampling is restricted to this subset
            of logical indices (drawn uniformly with replacement from the
            list).  This is used for cache-aligned sampling when each rank
            only has a shard of the decoded cache.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_samples: int | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        shuffle: bool = True,
        valid_indices: list[int] | None = None,
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
        self.valid_indices = valid_indices

        # Total samples across all ranks
        total_samples = num_samples if num_samples is not None else len(dataset)

        # Samples per rank (divide evenly)
        self.num_samples = total_samples // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Create generator with rank-specific seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * self.num_replicas + self.rank)

        if self.valid_indices is not None and len(self.valid_indices) > 0:
            positions = torch.randint(
                low=0,
                high=len(self.valid_indices),
                size=(self.num_samples,),
                generator=g,
                dtype=torch.int64,
            )
            return iter([self.valid_indices[p] for p in positions.tolist()])

        # Sample indices with replacement
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


def batch_pad_to_fixed_len(
    batch: list[torch.Tensor],
    max_batch_len: int,
    pad_token: int,
    left_pad: bool = False,
) -> torch.Tensor:
    if left_pad:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),  # pad on the left
                        seq,
                    ]
                )
                for seq in batch
            ]
        )
    else:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        seq,
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),
                    ]
                )
                for seq in batch
            ]
        )
    return batch_pad


def build_dataloader_from_dataset(
    dataset: Dataset,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 4,
    drop_last: bool = True,
    pin_memory: bool = True,
    use_random_replacement: bool = False,
    num_samples_per_epoch: int | None = None,
    seed: int = 42,
    cache_aligned_indices: list[int] | None = None,
    **kwargs: Any,
) -> DataLoader:
    """Build a :class:`DataLoader` from a :class:`RollingLeRobotDataset`.

    By default, uses :class:`~torch.utils.data.distributed.DistributedSampler`
    which samples without replacement. Set ``use_random_replacement=True`` to
    use :class:`RandomReplacementSampler` which samples with replacement,
    allowing batch sizes larger than the dataset size.

    Args:
        dataset: The :class:`RollingLeRobotDataset` to wrap.
        batch_size: Number of samples per batch **per replica**.
        world_size: Total number of distributed replicas.  Defaults to ``1``.
        rank: Global rank of the current process.  Defaults to ``0``.
        num_workers: Number of DataLoader worker processes.
        drop_last: Drop the last incomplete batch.  Defaults to ``True``
            (recommended for distributed training).
        pin_memory: Pin CPU memory for faster GPU transfers.  Defaults to
            ``True``.
        use_random_replacement: If ``True``, use
            :class:`RandomReplacementSampler` which samples with replacement.
            If ``False`` (default), use :class:`DistributedSampler` which
            samples without replacement. Set to ``True`` when batch_size may
            exceed dataset size.
        num_samples_per_epoch: Number of samples per epoch when using random
            replacement sampling. If ``None``, defaults to ``len(dataset)``.
            Only used when ``use_random_replacement=True``.
        seed: Random seed for sampling. Only used when
            ``use_random_replacement=True``.
        cache_aligned_indices: When provided, the
            :class:`DistributedRandomReplacementSampler` restricts drawing to
            this subset of logical indices.  Used for cache-aligned sampling
            when each rank only holds a shard of the decoded cache.  Ignored
            when ``use_random_replacement=False`` or ``world_size <= 1``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~torch.utils.data.DataLoader`.

    Returns:
        A :class:`DataLoader` instance.  The sampler is accessible via
        ``loader.sampler``.  Typical training loop after refresh::

            dataset = build_rolling_lerobot_dataset(
                root_dir="logs/run/maniskill",
                chunk_size=16,
            )
            loader = build_dataloader_from_dataset(
                dataset,
                batch_size=64,
                world_size=world_size,
                rank=rank,
                use_random_replacement=True,
                num_samples_per_epoch=6400,
            )
            for epoch in range(100):
                loader.sampler.set_epoch(epoch)
                for batch in loader:
                    train(batch)
                n_new = dataset.refresh()
                if n_new:
                    loader = build_dataloader_from_dataset(
                        dataset,
                        batch_size=64,
                        world_size=world_size,
                        rank=rank,
                        use_random_replacement=True,
                        num_samples_per_epoch=6400,
                    )
    """
    if use_random_replacement:
        # Use random sampling with replacement
        if world_size > 1:
            sampler = DistributedRandomReplacementSampler(
                dataset,
                num_samples=num_samples_per_epoch,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                valid_indices=cache_aligned_indices,
            )
        else:
            sampler = RandomReplacementSampler(
                dataset,
                num_samples=num_samples_per_epoch,
                seed=seed,
            )
    else:
        # Use standard distributed sampler (without replacement)
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

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        **kwargs,
    )

    return loader
