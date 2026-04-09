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

"""Rolling LeRobot dataset loader for on-policy RL data collected during training.

The expected directory layout produced by the environment's data-collection
wrapper is::

    root_dir/
    ├── rank_0/
    │   ├── id_0/      <- completed LeRobot sub-dataset
    │   │   ├── meta/info.json
    │   │   ├── meta/episodes.jsonl
    │   │   └── data/chunk-000/episode_000000.parquet
    │   ├── id_64/
    │   ├── ...
    │   ├── id_512/    <- last safe sub-dataset
    │   └── id_576/    <- currently being written — EXCLUDED
    ├── rank_1/
    │   └── ...
    └── rank_N/
        └── ...

Sub-datasets are sorted numerically by the integer in their ``id_N`` folder
name.  The last ``skip_last_n`` sub-datasets **per rank** are excluded to avoid
reading files that the environment worker is still writing.

Each sub-dataset is loaded via :class:`lerobot.common.datasets.lerobot_dataset.LeRobotDataset`
with ``root=<sub_dataset_path>``.  Chunk (multi-frame) sampling for OpenPI /
DAgger is implemented through LeRobot's ``delta_timestamps`` mechanism.

Typical usage (recommended pattern with separate functions)::

    dataset = build_rolling_lerobot_dataset(
        root_dir="logs/20260402/maniskill",
        chunk_size=16,      # action-chunk window for OpenPI / DAgger
        skip_last_n=1,
    )
    loader = build_dataloader_from_dataset(
        dataset,
        batch_size=32,
        world_size=1,
        rank=0,
    )

    for epoch in range(num_epochs):
        loader.sampler.set_epoch(epoch)
        for batch in loader:
            # batch["state"]   shape (B, chunk_size, state_dim)
            # batch["actions"] shape (B, chunk_size, action_dim)
            # batch["image"]   shape (B, chunk_size, C, H, W)
            train_step(batch)

        # Pick up newly completed sub-datasets written since the last epoch.
        n_new = dataset.refresh()
        if n_new:
            # Rebuild only the dataloader, reusing the same dataset
            loader = build_dataloader_from_dataset(
                dataset,
                batch_size=32,
                world_size=1,
                rank=0,
            )

Alternative usage (convenience function)::

    dataset, loader = build_rolling_lerobot_dataloader(
        root_dir="logs/20260402/maniskill",
        batch_size=32,
        chunk_size=16,
        skip_last_n=1,
    )

    for epoch in range(num_epochs):
        loader.sampler.set_epoch(epoch)
        for batch in loader:
            train_step(batch)

        n_new = dataset.refresh()
        if n_new:
            loader = build_dataloader_from_dataset(
                dataset,
                batch_size=32,
                world_size=1,
                rank=0,
            )
"""

from __future__ import annotations

import bisect
import json
import re
import time
import asyncio
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from rlinf.utils.logging import get_logger

logger = get_logger()

# Parquet index / metadata columns that should not be chunked along the time
# axis — they are scalar per frame and have no meaningful "window" semantics.
_META_KEYS: frozenset[str] = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)


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

        # Total samples across all ranks
        total_samples = num_samples if num_samples is not None else len(dataset)

        # Samples per rank (divide evenly)
        self.num_samples = total_samples // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Create generator with rank-specific seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * self.num_replicas + self.rank)

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_lerobot_dataset(path: Path) -> bool:
    """Return True if *path* looks like a completed LeRobot sub-dataset."""
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def _extract_id(path: Path) -> float:
    """Extract the integer suffix from an ``id_N`` folder name.

    Returns ``float("inf")`` for names that do not match so that malformed
    directories sort after all valid ones.
    """
    m = re.search(r"(\d+)$", path.name)
    return float(m.group(1)) if m else float("inf")


def _discover_safe_datasets(root_dir: Path, skip_last_n: int) -> list[Path]:
    """Walk *root_dir* and return all safe (non-in-progress) sub-dataset paths.

    Structure assumed::

        root_dir/rank_N/id_M/   <- LeRobot sub-dataset

    For each ``rank_N`` subdirectory the ``id_M`` children are sorted by their
    numeric suffix and the last ``skip_last_n`` are excluded **per rank**.  All
    safe paths across all ranks are returned in a stable order (rank asc, id
    asc).

    Args:
        root_dir: Root directory containing one or more ``rank_N`` directories.
        skip_last_n: Number of latest sub-datasets to exclude per rank.

    Returns:
        Sorted list of ``Path`` objects pointing to safe sub-dataset roots.
    """
    safe: list[Path] = []

    if not root_dir.is_dir():
        logger.warning("[RollingLeRobotDataset] root_dir does not exist: %s", root_dir)
        return safe

    rank_dirs = sorted(
        (d for d in root_dir.iterdir() if d.is_dir()),
        key=_extract_id,
    )

    for rank_dir in rank_dirs:
        id_dirs = sorted(
            (d for d in rank_dir.iterdir() if d.is_dir() and _is_lerobot_dataset(d)),
            key=_extract_id,
        )
        cutoff = len(id_dirs) - skip_last_n
        safe.extend(id_dirs[:cutoff])

    return safe


def _build_delta_timestamps(info: dict, chunk_size: int, action_sequence_keys) -> dict[str, list[float]]:
    """Build a ``delta_timestamps`` dict for LeRobotDataset chunk sampling.

    All data features (i.e. keys that are not index/metadata columns) receive
    the same temporal window ``[0, 1/fps, …, (chunk_size-1)/fps]``.

    Args:
        info: Parsed ``meta/info.json`` content.
        chunk_size: Number of consecutive frames per sample.

    Returns:
        Dict mapping feature key → list of relative timestamps in seconds.
    """
    fps: float = info["fps"]
    timestamps = [i / fps for i in range(chunk_size)]
    data_keys = action_sequence_keys if action_sequence_keys is not None else []
    return {k: timestamps for k in data_keys}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RollingLeRobotDataset(Dataset):
    """PyTorch Dataset that loads frames from a rolling collection of LeRobot
    sub-datasets written incrementally during RL training.

    Each sub-dataset (``rank_N/id_M/``) is loaded via
    :class:`lerobot.common.datasets.lerobot_dataset.LeRobotDataset`.  Chunk
    sampling is implemented through LeRobot's ``delta_timestamps`` mechanism:
    each ``__getitem__`` call returns ``chunk_size`` consecutive frames, with
    boundary clamping and ``*_is_pad`` masks handled by LeRobot automatically.

    When ``chunk_size=1`` (default) no ``delta_timestamps`` are set and each
    sample is a single frame, backward-compatible with single-step RL training.

    Args:
        root_dir: Root directory containing ``rank_N/id_M/`` sub-datasets.
        skip_last_n: Number of latest sub-datasets to exclude per rank to
            avoid read/write conflicts with the writer.  Defaults to ``1``.
        chunk_size: Number of consecutive frames per sample.  Defaults to
            ``1`` (single-frame).  Set to the model's action-chunk horizon for
            OpenPI / DAgger training.
        delta_timestamps: Explicit ``delta_timestamps`` dict passed to each
            :class:`LeRobotDataset`.  When ``None`` and ``chunk_size > 1``,
            a dict is auto-generated from ``meta/info.json`` (all data keys,
            window ``[0, 1/fps, …, (chunk_size-1)/fps]``).  Ignored when
            ``chunk_size <= 1``.
        keys: If provided, ``__getitem__`` filters the returned dict to only
            these keys.  ``None`` returns every key from LeRobotDataset.
        image_transforms: Optional callable passed directly to each
            :class:`LeRobotDataset` as ``image_transforms``.
        min_datasets: Minimum number of safe sub-datasets required before the
            dataset is considered ready.  Construction blocks until this
            threshold is met.  Defaults to ``1``.
        wait_interval_s: Seconds to sleep between readiness polls when fewer
            than ``min_datasets`` sub-datasets are available.  Defaults to
            ``10.0``.
    """

    def __init__(
        self,
        root_dir: str | Path,
        skip_last_n: int = 1,
        chunk_size: int = 1,
        delta_timestamps: dict[str, list[float]] | None = None,
        keys: list[str] | None = None,
        image_transforms: Callable | None = None,
        min_datasets: int = 1,
        wait_interval_s: float = 10.0,
        action_sequence_keys: list[str] | None = ["actions"],
    ) -> None:
        self.root_dir = Path(root_dir)
        self.skip_last_n = skip_last_n
        self.chunk_size = chunk_size
        self._user_delta_timestamps = delta_timestamps
        self.keys = keys
        self.image_transforms = image_transforms
        self.min_datasets = min_datasets
        self.wait_interval_s = wait_interval_s
        self.action_sequence_keys = action_sequence_keys

        # Sub-datasets indexed so far.
        self._indexed_datasets: set[Path] = set()
        self._sub_datasets: list[Any] = []  # list[LeRobotDataset]

        # Prefix-sum of lengths for O(log n) index dispatch.
        # _cumulative_lengths[i] = sum of lengths of sub_datasets[0..i-1].
        self._cumulative_lengths: list[int] = [0]

        self._build_index(_discover_safe_datasets(self.root_dir, self.skip_last_n))

    # ------------------------------------------------------------------
    # Readiness gate
    # ------------------------------------------------------------------

    async def wait_until_ready(self) -> None:
        """Block until at least ``min_datasets`` safe sub-datasets are loaded.

        Polls ``root_dir`` every ``wait_interval_s`` seconds, indexing any
        newly completed sub-datasets on each iteration, until the threshold is
        reached.  Logs progress at each poll so the operator can see that the
        process is intentionally waiting for data to accumulate.
        """
        while len(self._sub_datasets) < self.min_datasets:
            logger.info(
                "[RollingLeRobotDataset] waiting for data: %d/%d sub-dataset(s) "
                "available, sleeping %.1fs ...",
                len(self._sub_datasets),
                self.min_datasets,
                self.wait_interval_s,
            )
            await asyncio.sleep(self.wait_interval_s)
            safe = _discover_safe_datasets(self.root_dir, self.skip_last_n)
            new_paths = [p for p in safe if p not in self._indexed_datasets]
            if new_paths:
                self._build_index(new_paths)

        logger.info(
            "[RollingLeRobotDataset] ready: %d sub-dataset(s), %d total frame(s)",
            len(self._sub_datasets),
            len(self),
        )
    
    def is_ready(self) -> bool:
        return len(self._sub_datasets) >= self.min_datasets

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _get_delta_timestamps(self, ds_path: Path) -> dict[str, list[float]] | None:
        """Return delta_timestamps for *ds_path*, or None for single-frame."""
        if self.chunk_size <= 1:
            return None
        if self._user_delta_timestamps is not None:
            return self._user_delta_timestamps
        with open(ds_path / "meta" / "info.json", encoding="utf-8") as f:
            info = json.load(f)
        return _build_delta_timestamps(info, self.chunk_size, self.action_sequence_keys)

    def _build_index(self, datasets: list[Path]) -> int:
        """Load *datasets* as LeRobotDataset instances and extend the index.

        Args:
            datasets: Sub-dataset root paths to load.

        Returns:
            Number of new sub-datasets successfully added.
        """
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        n_new = 0
        for ds_path in datasets:
            if ds_path in self._indexed_datasets:
                continue

            delta_timestamps = self._get_delta_timestamps(ds_path)

            try:
                sub_ds = LeRobotDataset(
                    repo_id=ds_path.name,
                    root=ds_path,
                    delta_timestamps=delta_timestamps,
                    image_transforms=self.image_transforms,
                    download_videos=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[RollingLeRobotDataset] failed to load sub-dataset %s: %s",
                    ds_path,
                    exc,
                )
                continue

            self._sub_datasets.append(sub_ds)
            self._cumulative_lengths.append(
                self._cumulative_lengths[-1] + len(sub_ds)
            )
            self._indexed_datasets.add(ds_path)
            n_new += 1

        return n_new

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> int:
        """Re-scan ``root_dir`` and load any newly completed sub-datasets.

        Call this periodically (e.g. after each training epoch) to pick up
        sub-datasets written since the last call.  The method is additive:
        existing sub-datasets are never removed or reloaded.

        Returns:
            Number of new sub-datasets added (0 if nothing changed).
        """
        safe = _discover_safe_datasets(self.root_dir, self.skip_last_n)
        new_paths = [p for p in safe if p not in self._indexed_datasets]
        if not new_paths:
            return 0
        n_new = self._build_index(new_paths)
        logger.info(
            "[RollingLeRobotDataset] refresh: +%d sub-dataset(s), total_frames=%d",
            n_new,
            len(self),
        )
        return n_new

    def __len__(self) -> int:
        return self._cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Binary search into prefix-sum to find the right sub-dataset.
        ds_idx = bisect.bisect_right(self._cumulative_lengths, idx) - 1
        local_idx = idx - self._cumulative_lengths[ds_idx]

        item: dict[str, Any] = self._sub_datasets[ds_idx][local_idx]

        if self.keys is not None:
            item = {k: v for k, v in item.items() if k in self.keys}

        return item


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_rolling_lerobot_dataset(
    root_dir: str | Path,
    skip_last_n: int = 1,
    chunk_size: int = 1,
    delta_timestamps: dict[str, list[float]] | None = None,
    keys: list[str] | None = None,
    image_transforms: Callable | None = None,
    min_datasets: int = 1,
    wait_interval_s: float = 10.0,
    action_sequence_keys: list[str] | None = ["actions"],
) -> RollingLeRobotDataset:
    """Build a :class:`RollingLeRobotDataset` for rolling data collection.

    Args:
        root_dir: Root directory containing ``rank_N/id_M/`` sub-datasets.
        skip_last_n: Latest sub-datasets to exclude per rank.  Defaults to
            ``1`` to avoid reading the sub-dataset currently being written.
        chunk_size: Consecutive frames per sample.  Defaults to ``1``
            (single-frame).  Set to the model's action-chunk horizon for
            OpenPI / DAgger training.
        delta_timestamps: Explicit ``delta_timestamps`` passed to each
            :class:`~lerobot.common.datasets.lerobot_dataset.LeRobotDataset`.
            Auto-generated from ``chunk_size`` and fps when ``None``.
        keys: Parquet column names to keep in each sample.  ``None`` keeps all
            keys returned by LeRobotDataset.
        image_transforms: Optional transform passed to each LeRobotDataset's
            ``image_transforms`` argument.
        min_datasets: Minimum number of safe sub-datasets required before the
            dataset returns.  Construction sleeps until the threshold is met.
            Defaults to ``1``.
        wait_interval_s: Seconds between readiness polls.  Defaults to ``10.0``.
        action_sequence_keys: List of keys to apply chunking to.  Defaults to
            ``["actions"]``.

    Returns:
        A :class:`RollingLeRobotDataset` instance.
    """
    dataset = RollingLeRobotDataset(
        root_dir=root_dir,
        skip_last_n=skip_last_n,
        chunk_size=chunk_size,
        delta_timestamps=delta_timestamps,
        keys=keys,
        image_transforms=image_transforms,
        min_datasets=min_datasets,
        wait_interval_s=wait_interval_s,
        action_sequence_keys=action_sequence_keys,
    )

    logger.info(
        "[build_rolling_lerobot_dataset] root_dir=%s, chunk_size=%d, "
        "skip_last_n=%d, sub_datasets=%d, total_frames=%d",
        root_dir,
        chunk_size,
        skip_last_n,
        len(dataset._sub_datasets),
        len(dataset),
    )

    return dataset


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


def build_rolling_lerobot_dataloader(
    root_dir: str | Path,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    skip_last_n: int = 1,
    chunk_size: int = 1,
    delta_timestamps: dict[str, list[float]] | None = None,
    keys: list[str] | None = None,
    image_transforms: Callable | None = None,
    min_datasets: int = 1,
    wait_interval_s: float = 10.0,
    num_workers: int = 4,
    drop_last: bool = True,
    pin_memory: bool = True,
    action_sequence_keys: list[str] | None = ["actions"],
    **kwargs: Any,
) -> tuple[RollingLeRobotDataset, DataLoader]:
    """Build a :class:`RollingLeRobotDataset` and a :class:`DataLoader` over it.

    A :class:`~torch.utils.data.distributed.DistributedSampler` is always
    created (``num_replicas=world_size``, ``rank=rank``), so set
    ``world_size=1`` / ``rank=0`` for single-process training.  Call
    ``loader.sampler.set_epoch(epoch)`` at the start of every epoch to reseed
    the sampler's shuffle.

    Returns both objects so the caller can invoke ``dataset.refresh()`` to
    pick up newly completed sub-datasets and then rebuild the loader.

    Args:
        root_dir: Root directory containing ``rank_N/id_M/`` sub-datasets.
        batch_size: Number of samples per batch **per replica**.
        world_size: Total number of distributed replicas.  Defaults to ``1``.
        rank: Global rank of the current process.  Defaults to ``0``.
        skip_last_n: Latest sub-datasets to exclude per rank.  Defaults to
            ``1`` to avoid reading the sub-dataset currently being written.
        chunk_size: Consecutive frames per sample.  Defaults to ``1``
            (single-frame).  Set to the model's action-chunk horizon for
            OpenPI / DAgger training.
        delta_timestamps: Explicit ``delta_timestamps`` passed to each
            :class:`~lerobot.common.datasets.lerobot_dataset.LeRobotDataset`.
            Auto-generated from ``chunk_size`` and fps when ``None``.
        keys: Parquet column names to keep in each sample.  ``None`` keeps all
            keys returned by LeRobotDataset.
        image_transforms: Optional transform passed to each LeRobotDataset's
            ``image_transforms`` argument.
        min_datasets: Minimum number of safe sub-datasets required before the
            dataset (and therefore this function) returns.  Construction sleeps
            until the threshold is met.  Defaults to ``1``.
        wait_interval_s: Seconds between readiness polls.  Defaults to ``10.0``.
        num_workers: Number of DataLoader worker processes.
        drop_last: Drop the last incomplete batch.  Defaults to ``True``
            (recommended for distributed training).
        pin_memory: Pin CPU memory for faster GPU transfers.  Defaults to
            ``True``.
        action_sequence_keys: List of keys to apply chunking to.  Defaults to
            ``["actions"]``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~torch.utils.data.DataLoader`.

    Returns:
        A ``(dataset, dataloader)`` tuple.  The sampler is accessible via
        ``loader.sampler``.  Typical training loop::

            dataset, loader = build_rolling_lerobot_dataloader(
                root_dir="logs/run/maniskill",
                batch_size=64,
                world_size=world_size,
                rank=rank,
                chunk_size=16,
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
                    )
    """
    dataset = build_rolling_lerobot_dataset(
        root_dir=root_dir,
        skip_last_n=skip_last_n,
        chunk_size=chunk_size,
        delta_timestamps=delta_timestamps,
        keys=keys,
        image_transforms=image_transforms,
        min_datasets=min_datasets,
        wait_interval_s=wait_interval_s,
        action_sequence_keys=action_sequence_keys,
    )

    loader = build_dataloader_from_dataset(
        dataset=dataset,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        **kwargs,
    )

    return dataset, loader
