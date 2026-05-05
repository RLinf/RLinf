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

Shard roots (``rank_N/id_M/``) are recorded in ``RollingLeRobotDataset._sub_datasets``
as :class:`~pathlib.Path` objects only.  A :class:`lerobot.common.datasets.lerobot_dataset.LeRobotDataset`
is opened on demand when serving a cache miss (or whenever the decoded cache is
disabled).  Chunk (multi-frame) sampling for OpenPI / DAgger uses LeRobot's
``delta_timestamps`` mechanism.
Optionally, ``require_all_intervene=True`` restricts the dataset to chunk
starts whose non-padded frames all have ``intervene_flag`` set (see
:class:`RollingLeRobotDataset`).

Typical usage (recommended pattern with separate functions)::

    dataset = build_rolling_lerobot_dataset(
        root_dir="logs/20260402/maniskill",
        chunk_size=16,  # action-chunk window for OpenPI / DAgger
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
"""

from __future__ import annotations

import bisect
import copy
import json
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rlinf.data.utils import build_dataloader_from_dataset
from rlinf.utils.logging import get_logger

logger = get_logger()

# Parquet index / metadata columns that should not be chunked along the time
# axis — they are scalar per frame and have no meaningful "window" semantics.
_META_KEYS: frozenset[str] = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)

CacheIngestMode = Literal["new_shards", "last_n", "both"]


def _deep_clone_sample(obj: Any) -> Any:
    """Clone a LeRobot sample for storage or hand-out (tensors detached/cloned)."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _deep_clone_sample(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_deep_clone_sample(x) for x in obj]
        return type(obj)(seq)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return copy.deepcopy(obj)


class DecodedTensorFifoCache:
    """FIFO ring storing decoded training samples keyed by global frame index.

    Each slot holds a full sample dict (tensors cloned; scalars / strings copied).
    When the ring wraps, the oldest slot is overwritten and its global index
    dropped from the lookup map.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(int(capacity), 1)
        self._lock = threading.Lock()
        self._slot_global: list[int | None] = [None] * self.capacity
        self._slot_payload: list[dict[str, Any] | None] = [None] * self.capacity
        self._global_to_slot: dict[int, int] = {}
        self._next_slot: int = 0
        self._hits: int = 0
        self._misses: int = 0

    def try_get(self, global_idx: int) -> dict[str, Any] | None:
        with self._lock:
            slot = self._global_to_slot.get(global_idx)
            if slot is None:
                return None
            self._hits += 1
            payload = self._slot_payload[slot]
            assert payload is not None
            return _deep_clone_sample(payload)

    def notify_miss(self) -> None:
        with self._lock:
            self._misses += 1

    def put(self, global_idx: int, item: dict[str, Any]) -> None:
        stored = _deep_clone_sample(item)
        with self._lock:
            if global_idx in self._global_to_slot:
                slot = self._global_to_slot[global_idx]
                self._slot_payload[slot] = stored
                return
            slot = self._next_slot
            old_g = self._slot_global[slot]
            if old_g is not None:
                del self._global_to_slot[old_g]
            self._slot_global[slot] = global_idx
            self._slot_payload[slot] = stored
            self._global_to_slot[global_idx] = slot
            self._next_slot = (slot + 1) % self.capacity

    def cached_indices(self) -> frozenset[int]:
        """Return the global frame indices currently stored in the cache."""
        with self._lock:
            return frozenset(self._global_to_slot.keys())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "decoded_cache_capacity": self.capacity,
                "decoded_cache_entries": len(self._global_to_slot),
                "decoded_cache_hits": self._hits,
                "decoded_cache_misses": self._misses,
            }


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


def _delta_offsets_for_sub_dataset(sub_ds: Any) -> list[int]:
    """Temporal offsets (in frames) for one LeRobot chunk sample.

    Mirrors :meth:`lerobot.common.datasets.lerobot_dataset.LeRobotDataset._get_query_indices`.
    """
    if getattr(sub_ds, "delta_indices", None) is None:
        return [0]
    first_key = next(iter(sub_ds.delta_indices))
    return [int(x) for x in sub_ds.delta_indices[first_key]]


def _hf_column_to_numpy_bool_1d(hf_dataset: Any, col: str) -> np.ndarray:
    """Load a per-row boolean column as ``(num_rows,)`` bool numpy array."""
    raw = hf_dataset[col]
    if isinstance(raw, torch.Tensor):
        t = raw
    else:
        t = torch.stack(list(raw))
    return t.reshape(-1).bool().numpy()


def _hf_column_to_numpy_int64_1d(hf_dataset: Any, col: str) -> np.ndarray:
    raw = hf_dataset[col]
    if isinstance(raw, torch.Tensor):
        t = raw
    else:
        t = torch.stack(list(raw))
    return t.reshape(-1).to(torch.int64).numpy()


def _compute_intervene_valid_local_indices(
    sub_ds: Any,
    intervene_flag_key: str,
) -> list[int]:
    """Return local frame indices whose chunk has ``intervene_flag=True`` on every non-padded step.

    Padding follows LeRobot's episode-local clamping: padded timesteps are ignored
    (no ``intervene_flag`` requirement on padded slots).

    If *intervene_flag_key* is absent from the HF dataset, logs a warning and
    treats every frame index as valid (same as no filtering for that shard).
    """
    hf = sub_ds.hf_dataset
    n = len(hf)
    if n == 0:
        return []

    if intervene_flag_key not in hf.column_names:
        logger.warning(
            "[RollingLeRobotDataset] require_all_intervene=True but column %r "
            "missing in %s; keeping all %d chunk starts for this shard.",
            intervene_flag_key,
            getattr(sub_ds, "root", "?"),
            n,
        )
        return list(range(n))

    flags = _hf_column_to_numpy_bool_1d(hf, intervene_flag_key)
    ep_idx = _hf_column_to_numpy_int64_1d(hf, "episode_index")
    ep_from = sub_ds.episode_data_index["from"].detach().cpu().numpy().astype(np.int64)
    ep_to = sub_ds.episode_data_index["to"].detach().cpu().numpy().astype(np.int64)
    deltas = np.array(_delta_offsets_for_sub_dataset(sub_ds), dtype=np.int64)

    idx_range = np.arange(n, dtype=np.int64)[:, None]
    raw = idx_range + deltas[None, :]
    ep_start = ep_from[ep_idx][:, None]
    ep_end = ep_to[ep_idx][:, None]
    is_pad = (raw < ep_start) | (raw >= ep_end)
    chunk_ok = np.ones(n, dtype=np.bool_)
    for j in range(deltas.shape[0]):
        padded = is_pad[:, j]
        rj = raw[:, j]
        # Padded slots are ignored; only index ``flags`` for in-range rows (``np.where`` is not lazy).
        step_ok = np.zeros(n, dtype=np.bool_)
        step_ok[padded] = True
        m = ~padded
        if m.any():
            step_ok[m] = flags[rj[m]]
        chunk_ok &= step_ok
    return np.nonzero(chunk_ok)[0].tolist()


@dataclass(frozen=True)
class _ShardIndexProbe:
    """Result of probing one LeRobot shard during :meth:`RollingLeRobotDataset._build_index`."""

    ds_path: Path
    ok: bool
    sub_ds: Any | None = None
    n_frames: int = 0
    num_episodes: int = 0
    intervene_locals: list[int] | None = None


def _build_delta_timestamps(
    info: dict, chunk_size: int, action_sequence_keys
) -> dict[str, list[float]]:
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
    return dict.fromkeys(data_keys, timestamps)


# ---------------------------------------------------------------------------
# In-memory Arrow store (used by RollingLeRobotDataset.in_memory_mode)
# ---------------------------------------------------------------------------


class InMemoryArrowStore:
    """Append-only in-memory frame store backed by per-episode Arrow tables.

    Provides chunk sampling with episode-boundary clamping and ``*_is_pad``
    masks that are **bit-for-bit identical** to
    :meth:`~lerobot.common.datasets.lerobot_dataset.LeRobotDataset.__getitem__`
    without any disk I/O.

    Each episode is stored as an independent HuggingFace
    :class:`datasets.Dataset` (Arrow table).  Appending a new episode is
    O(1) — no ``concatenate_datasets`` copy.  Random access requires an
    O(log N_episodes) binary search to locate the correct episode, then
    O(chunk_size) Arrow ``select`` rows for the chunk window.

    The HuggingFace ``Features`` schema is **inferred automatically** from
    the first episode's frame dicts, so the caller does not need to know the
    feature layout up front.

    Args:
        chunk_size: Consecutive frames per sample.  ``<= 1`` disables
            chunking (single-frame output with no ``delta_indices``).
        action_sequence_keys: Keys to apply chunk sampling to.  Determines
            which keys appear as ``[chunk_size, …]`` tensors and get
            companion ``*_is_pad`` masks.
        fps: Frames per second used for the ``timestamp`` metadata column.
        image_transforms: Optional callable applied to image tensors after
            ``hf_transform_to_torch`` converts PIL → float32 ``[C,H,W]``.
    """

    def __init__(
        self,
        chunk_size: int = 1,
        action_sequence_keys: list[str] | None = None,
        fps: int = 10,
        image_transforms: Callable | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._fps = max(fps, 1)
        self._image_transforms = image_transforms
        keys = action_sequence_keys or []
        self._delta_indices: dict[str, list[int]] = (
            {k: list(range(chunk_size)) for k in keys} if chunk_size > 1 else {}
        )
        self._episode_datasets: deque[Any] = deque()
        self._ep_from: deque[int] = deque()
        self._ep_to: deque[int] = deque()
        self._total_frames: int = 0
        self._hf_features: Any = None
        self._image_keys: set[str] = set()
        self._task_to_idx: dict[str, int] = {}
        self._tasks: dict[int, str] = {}
        self._hits: int = 0

    # ------------------------------------------------------------------
    # Schema inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_hf_features(frame: dict) -> tuple[Any, set[str]]:
        """Derive a HuggingFace ``Features`` schema from a raw frame dict.

        Returns the schema and the set of keys that hold image arrays
        (``uint8 [H, W, C]``).
        """
        import datasets as hf_datasets

        meta_features: dict[str, Any] = {
            "index": hf_datasets.Value("int64"),
            "episode_index": hf_datasets.Value("int64"),
            "frame_index": hf_datasets.Value("int64"),
            "timestamp": hf_datasets.Value("float32"),
            "task_index": hf_datasets.Value("int64"),
        }
        data_features: dict[str, Any] = {}
        image_keys: set[str] = set()

        for key, val in frame.items():
            if key == "task":
                continue
            if not isinstance(val, np.ndarray):
                continue
            if val.dtype == np.uint8 and val.ndim == 3:
                data_features[key] = hf_datasets.Image()
                image_keys.add(key)
            elif val.shape == (1,):
                dtype_str = "bool" if np.issubdtype(val.dtype, np.bool_) else str(val.dtype)
                data_features[key] = hf_datasets.Value(dtype_str)
            elif val.ndim == 1:
                dtype_str = "bool" if np.issubdtype(val.dtype, np.bool_) else str(val.dtype)
                data_features[key] = hf_datasets.Sequence(
                    length=val.shape[0],
                    feature=hf_datasets.Value(dtype_str),
                )
            elif val.ndim == 2:
                data_features[key] = hf_datasets.Array2D(
                    shape=tuple(val.shape), dtype=str(val.dtype)
                )

        return hf_datasets.Features({**meta_features, **data_features}), image_keys

    # ------------------------------------------------------------------
    # Episode addition
    # ------------------------------------------------------------------

    def add_episode(self, ep_frames: list[dict]) -> None:
        """Append *ep_frames* as a new episode; O(1) Arrow table creation.

        The HuggingFace features schema is inferred lazily from the first
        episode so callers do not need to specify it up front.

        Args:
            ep_frames: Per-step frame dicts as produced by
                :meth:`~rlinf.envs.wrappers.collect_episode.CollectEpisode._buffer_to_lerobot_ep`.
        """
        import PIL.Image as PILImage
        from datasets import Dataset
        from lerobot.common.datasets.utils import hf_transform_to_torch

        n = len(ep_frames)
        if n == 0:
            return

        if self._hf_features is None:
            self._hf_features, self._image_keys = self._infer_hf_features(ep_frames[0])

        ep_idx = len(self._ep_from)
        base = self._total_frames

        # Register task strings and build per-frame task_index array.
        task_indices: list[int] = []
        for frame in ep_frames:
            task_str = frame.get("task", "")
            if task_str not in self._task_to_idx:
                tidx = len(self._task_to_idx)
                self._task_to_idx[task_str] = tidx
                self._tasks[tidx] = task_str
            task_indices.append(self._task_to_idx[task_str])

        ep_dict: dict[str, Any] = {
            "index": np.arange(base, base + n, dtype=np.int64),
            "episode_index": np.full((n,), ep_idx, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "timestamp": np.arange(n, dtype=np.float32) / self._fps,
            "task_index": np.array(task_indices, dtype=np.int64),
        }

        for key in self._hf_features:
            if key in ("index", "episode_index", "frame_index", "timestamp", "task_index"):
                continue
            if key in self._image_keys:
                ep_dict[key] = [PILImage.fromarray(f[key]) for f in ep_frames]
            else:
                vals = [f.get(key) for f in ep_frames]
                if all(v is not None for v in vals):
                    stacked = np.stack(vals)
                    # Scalar (1,) — squeeze to 1-D so Arrow Value dtype matches.
                    ep_dict[key] = (
                        stacked.squeeze(1) if stacked.shape == (n, 1) else stacked
                    )

        ep_ds = Dataset.from_dict(ep_dict, features=self._hf_features)
        ep_ds.set_transform(hf_transform_to_torch)
        self._episode_datasets.append(ep_ds)
        self._ep_from.append(base)
        self._ep_to.append(base + n)
        self._total_frames += n

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes stored in this shard."""
        return len(self._episode_datasets)

    def __getitem__(self, local_idx: int) -> dict[str, Any]:
        """Fetch one sample with a chunk window identical to LeRobot's output.

        Args:
            local_idx: Frame index within this shard (0-based).

        Returns:
            Dict of tensors.  When ``chunk_size > 1``, keys listed in
            ``delta_indices`` are replaced by ``[chunk_size, …]`` tensors and
            companion ``*_is_pad`` bool tensors are added — matching
            :class:`~lerobot.common.datasets.lerobot_dataset.LeRobotDataset`
            ``__getitem__`` output exactly.
        """
        local_idx = max(0, min(local_idx, self._total_frames - 1))
        self._hits += 1
        ep_idx = bisect.bisect_right(self._ep_to, local_idx)
        ep_start = self._ep_from[ep_idx]
        ep_end = self._ep_to[ep_idx]
        local_frame = local_idx - ep_start

        ep_ds = self._episode_datasets[ep_idx]
        item: dict[str, Any] = ep_ds[local_frame]
        item["task"] = self._tasks.get(int(item["task_index"].item()), "")

        if self._image_transforms is not None:
            for k in self._image_keys:
                if k in item:
                    item[k] = self._image_transforms(item[k])

        if self._delta_indices:
            query_indices = {
                key: [
                    max(ep_start, min(ep_end - 1, local_idx + d)) for d in deltas
                ]
                for key, deltas in self._delta_indices.items()
            }
            padding = {
                f"{key}_is_pad": torch.BoolTensor(
                    [
                        (local_idx + d < ep_start) | (local_idx + d >= ep_end)
                        for d in deltas
                    ]
                )
                for key, deltas in self._delta_indices.items()
            }
            # All chunk indices are within the same episode — use local offsets.
            query_result = {
                key: torch.stack(
                    ep_ds.select([q - ep_start for q in q_idxs])[key]
                )
                for key, q_idxs in query_indices.items()
                if key in self._hf_features and key not in self._image_keys
            }
            item = {**item, **padding}
            for k, v in query_result.items():
                item[k] = v

        return item

    def stats(self) -> dict[str, Any]:
        """Return access counter and current store size.

        Returns:
            Dictionary with keys:

            * ``in_memory_store_episodes`` – number of episodes in this shard.
            * ``in_memory_store_frames`` – total frames in this shard.
            * ``in_memory_store_hits`` – number of ``__getitem__`` calls.
        """
        return {
            "in_memory_store_episodes": len(self._episode_datasets),
            "in_memory_store_frames": self._total_frames,
            "in_memory_store_hits": self._hits,
        }


def _compute_intervene_valid_local_indices_in_memory(
    store: InMemoryArrowStore,
    intervene_flag_key: str,
    chunk_size: int,
) -> list[int]:
    """Like :func:`_compute_intervene_valid_local_indices` for :class:`InMemoryArrowStore`.

    Uses the same padding and per-chunk flag rules as the LeRobot path so
    index construction matches disk-backed shards without opening
    :class:`~lerobot.common.datasets.lerobot_dataset.LeRobotDataset`.
    """
    n = len(store)
    if n == 0:
        return []

    for ep_ds in store._episode_datasets:
        if intervene_flag_key not in ep_ds.column_names:
            logger.warning(
                "[RollingLeRobotDataset] require_all_intervene=True but column %r "
                "missing in in-memory shard; keeping all %d chunk starts for this shard.",
                intervene_flag_key,
                n,
            )
            return list(range(n))

    flag_parts = [
        _hf_column_to_numpy_bool_1d(ep_ds, intervene_flag_key)
        for ep_ds in store._episode_datasets
    ]
    flags = np.concatenate(flag_parts, axis=0)
    assert int(flags.shape[0]) == n

    ep_from = np.array(list(store._ep_from), dtype=np.int64)
    ep_to = np.array(list(store._ep_to), dtype=np.int64)
    ep_idx = np.empty(n, dtype=np.int64)
    for ei in range(len(ep_from)):
        ep_idx[ep_from[ei] : ep_to[ei]] = ei

    if chunk_size <= 1:
        deltas = np.array([0], dtype=np.int64)
    else:
        deltas = np.arange(chunk_size, dtype=np.int64)

    idx_range = np.arange(n, dtype=np.int64)[:, None]
    raw = idx_range + deltas[None, :]
    ep_start = ep_from[ep_idx][:, None]
    ep_end = ep_to[ep_idx][:, None]
    is_pad = (raw < ep_start) | (raw >= ep_end)
    chunk_ok = np.ones(n, dtype=np.bool_)
    for j in range(int(deltas.shape[0])):
        padded = is_pad[:, j]
        rj = raw[:, j]
        step_ok = np.zeros(n, dtype=np.bool_)
        step_ok[padded] = True
        m = ~padded
        if m.any():
            step_ok[m] = flags[rj[m]]
        chunk_ok &= step_ok
    return np.nonzero(chunk_ok)[0].tolist()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RollingLeRobotDataset(Dataset):
    """PyTorch Dataset that loads frames from a rolling collection of LeRobot
    sub-datasets written incrementally during RL training.

    Each sub-dataset path (``rank_N/id_M/``) is stored in ``_sub_datasets``; a
    :class:`lerobot.common.datasets.lerobot_dataset.LeRobotDataset` is opened
    lazily when a sample is read (decoded-cache hit avoids opening).  Chunk
    sampling is implemented through LeRobot's ``delta_timestamps`` mechanism:
    each sample is ``chunk_size`` consecutive frames, with boundary clamping
    and ``*_is_pad`` masks handled by LeRobot automatically.  When the decoded
    FIFO cache is enabled, each cached entry is that full window (identical to
    a single ``__getitem__`` result), not a single raw frame.

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
        min_frames: Minimum total number of frames required before the dataset
            is considered ready.  Construction blocks until this threshold is
            met.  Defaults to ``1``.
        wait_interval_s: Seconds to sleep between readiness polls when fewer
            than ``min_frames`` total frames are available.  Defaults to
            ``10.0``.
        action_sequence_keys: Keys to apply ``delta_timestamps`` chunking to.
            Defaults to ``["actions"]``.
        enable_decoded_cache: If ``True``, maintain a FIFO in-memory cache of
            fully decoded samples (see :class:`DecodedTensorFifoCache`).
            Use ``num_workers=0`` on the DataLoader so workers see the same
            cache (forked workers do not share updates).
        decoded_cache_capacity: Number of samples in the FIFO ring.
        cache_ingest_mode: ``new_shards`` — on refresh, ingest only global
            indices from newly added sub-datasets; ``last_n`` — each refresh,
            ingest the last ``cache_last_n_frames`` indices; ``both`` —
            combine both.
        cache_last_n_frames: Tail length for ``last_n`` / ``both`` modes.
        cache_ingest_max_frames: Optional cap on how many frames to ingest per
            ``refresh()`` call (``None`` = no cap).
        require_all_intervene: When ``True``, only chunk starts where every
            **non-padded** frame in the temporal window has
            ``intervene_flag_key=True`` are exposed to the sampler (dataset
            length and indices are restricted accordingly).  For
            ``chunk_size=1`` this reduces to single-frame filtering.  When the
            flag column is missing from a shard, that shard falls back to all
            starts (with a warning).
        intervene_flag_key: Parquet / HF column name for the per-frame bool
            flag (default ``"intervene_flag"``).
        window_size: If set to a positive integer, the dataset length and
            sampling only cover the last ``window_size`` **logical** frames
            (i.e. frames actually exposed to the sampler).  When
            ``require_all_intervene`` is active, logical frames are the
            intervene-valid subset, so ``window_size=1000`` always means
            "the last 1000 usable samples" regardless of intervene density.
            Without intervene filtering logical == physical, so the behavior
            is equivalent to a physical-frame window.  Analogous to
            ``TrajectoryReplayBuffer.sample_window_size`` in
            :mod:`rlinf.data.replay_buffer` but counted in frames instead of
            trajectories.  ``None`` or ``0`` disables windowing (full history).
        index_load_workers: How many threads may open distinct sub-dataset
            paths in parallel during :meth:`refresh` / initial index build.
            ``1`` keeps the original sequential behavior.  Values ``> 1``
            use :class:`~concurrent.futures.ThreadPoolExecutor` so metadata
            and parquet-backed HuggingFace dataset construction can overlap
            across shards.  Multiprocessing is not used: child processes
            cannot return live :class:`~lerobot.common.datasets.lerobot_dataset.LeRobotDataset`
            handles to the parent for ``out_open_handles`` / decoded-cache
            ingest without reopening every shard.
        cache_ingest_rank: Rank of the current process for sharding cache
            ingest across distributed workers.  When ``cache_ingest_world_size
            > 1``, only every ``cache_ingest_world_size``-th index (offset
            by ``cache_ingest_rank``) is ingested, so each rank pre-fills a
            disjoint slice of the frame space — reducing total CPU memory by
            roughly ``cache_ingest_world_size`` x.  Defaults to ``0``.
        cache_ingest_world_size: Total number of distributed replicas
            participating in cache ingest sharding.  ``1`` (default) disables
            sharding and ingests all indices (original behavior).
    """

    def __init__(
        self,
        root_dir: str | Path,
        skip_last_n: int = 1,
        chunk_size: int = 1,
        delta_timestamps: dict[str, list[float]] | None = None,
        keys: list[str] | None = None,
        image_transforms: Callable | None = None,
        min_frames: int = 10,
        wait_interval_s: float = 10.0,
        action_sequence_keys: list[str] | None = ["actions"],
        # cache
        enable_decoded_cache: bool = False,
        decoded_cache_capacity: int = 8192,
        cache_ingest_mode: CacheIngestMode = "new_shards",
        cache_last_n_frames: int = 10_000,
        cache_ingest_max_frames: int | None = None,
        # check intervene
        require_all_intervene: bool = False,
        intervene_flag_key: str = "intervene_flag",
        window_size: int | None = None,
        index_load_workers: int = 1,
        cache_ingest_rank: int = 0,
        cache_ingest_world_size: int = 1,
        in_memory_mode: bool = False,
        fps: int = 10,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.skip_last_n = skip_last_n
        self.chunk_size = chunk_size
        self._user_delta_timestamps = delta_timestamps
        self.keys = keys
        self.image_transforms = image_transforms
        self.min_frames = min_frames
        self.wait_interval_s = wait_interval_s
        self.action_sequence_keys = action_sequence_keys
        self.cache_ingest_mode: CacheIngestMode = cache_ingest_mode
        self.cache_last_n_frames = int(cache_last_n_frames)
        self.cache_ingest_max_frames = cache_ingest_max_frames
        self.require_all_intervene = bool(require_all_intervene)
        self.intervene_flag_key = intervene_flag_key
        self.window_size = window_size
        self._index_load_workers = max(1, int(index_load_workers))
        self._cache_ingest_rank = int(cache_ingest_rank)
        self._cache_ingest_world_size = max(1, int(cache_ingest_world_size))
        self._window_physical_start: int = 0
        self._window_valid_slice_lo: int = 0
        self._valid_physical_indices: list[int] | None = None
        self._valid_physical_set: set[int] | None = None
        if self.require_all_intervene:
            self._valid_physical_indices = []
            self._valid_physical_set = set()
        self._decoded_cache: DecodedTensorFifoCache | None = None
        if enable_decoded_cache:
            self._decoded_cache = DecodedTensorFifoCache(decoded_cache_capacity)
        # Serializes refresh (index growth + cache ingest) vs __getitem__/__getitems__.
        self._rolling_access_lock = threading.RLock()

        # Sub-dataset **roots** indexed so far (paths only; no live LeRobot handles).
        self._indexed_datasets: set[Path] = set()
        self._sub_datasets: list[Path] = []
        # At most one open LeRobotDataset — reopened when ``__getitem__`` crosses shards.
        self._lerobot_open: Any | None = None
        self._lerobot_open_idx: int | None = None

        # Prefix-sum of lengths for O(log n) index dispatch.
        # _cumulative_lengths[i] = sum of lengths of sub_datasets[0..i-1].
        self._cumulative_lengths: list[int] = [0]

        # Running total of episodes across all loaded sub-datasets.
        self._total_episodes: int = 0

        # Shard cache: when ``in_memory_mode=True``, newly written shards are
        # kept in RAM (keyed by their disk path) so that
        # ``_load_item_from_lerobot`` can serve frames from RAM instead of
        # reading them back from disk.  Disk writes are kept as a persistence
        # sidecar.  When a shard scrolls fully out of ``window_size`` it is
        # evicted from RAM; the disk copy remains as a transparent fallback.
        self._shard_cache_enabled: bool = bool(in_memory_mode)
        self._in_memory_shards: dict[Path, InMemoryArrowStore] = {}
        # Config kept so per-shard InMemoryArrowStore instances can be built
        # in add_shard_to_memory without requiring the caller to repeat params.
        self._shard_cache_chunk_size: int = chunk_size
        self._shard_cache_fps: int = max(1, int(fps))
        self._shard_cache_action_keys: list[str] = list(action_sequence_keys or [])
        # Hit/miss counters for the shard-level cache lookup in
        # _load_item_from_lerobot (shard found in RAM vs. fell back to disk).
        self._shard_cache_hits: int = 0
        self._shard_cache_misses: int = 0
        self._build_index(_discover_safe_datasets(self.root_dir, self.skip_last_n))
        if self._decoded_cache is not None and self.cache_ingest_mode in (
            "last_n",
            "both",
        ):
            self._ingest_decoded_cache(0, 0, 0)

    # ------------------------------------------------------------------
    # Readiness gate
    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return len(self) >= self.min_frames

    def _num_physical_frames(self) -> int:
        """Total indexed frames (ignores intervene filter and window)."""
        return int(self._cumulative_lengths[-1])

    def _logical_to_physical(self, logical_idx: int) -> int:
        if self._valid_physical_indices is None:
            if self._window_enabled():
                return self._window_physical_start + int(logical_idx)
            return int(logical_idx)
        if self._window_enabled():
            lo = self._window_valid_slice_lo
            return int(self._valid_physical_indices[lo + int(logical_idx)])
        return int(self._valid_physical_indices[logical_idx])

    def _window_enabled(self) -> bool:
        return self.window_size is not None and int(self.window_size) > 0

    def _update_window_sampling_bounds(self) -> None:
        """Recompute window slice offsets so ``window_size`` caps **logical** frames.

        *Logical frames* are the frames actually exposed by ``__len__`` /
        ``__getitem__``: when ``require_all_intervene`` is active they are the
        intervene-valid subset; otherwise every physical frame is logical.

        Without intervene filtering (``_valid_physical_indices is None``),
        physical == logical, so ``_window_physical_start`` is set to
        ``max(0, total_physical - window_size)`` exactly as before.

        With intervene filtering, the last ``window_size`` entries of
        ``_valid_physical_indices`` are kept, making the user-facing dataset
        length deterministic (``min(num_valid, window_size)``).
        """
        n = self._num_physical_frames()
        if not self._window_enabled():
            self._window_physical_start = 0
            self._window_valid_slice_lo = 0
            return
        w = max(0, int(self.window_size))
        if self._valid_physical_indices is not None:
            self._window_valid_slice_lo = max(0, len(self._valid_physical_indices) - w)
            if self._window_valid_slice_lo < len(self._valid_physical_indices):
                self._window_physical_start = self._valid_physical_indices[
                    self._window_valid_slice_lo
                ]
            else:
                self._window_physical_start = n
        else:
            self._window_physical_start = max(0, n - w)
            self._window_valid_slice_lo = 0

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _get_delta_timestamps(self, ds_path: Path) -> dict[str, list[float]] | None:
        """Return delta_timestamps for *ds_path*, or None for single-frame.

        When ``delta_timestamps`` was not passed to the constructor, auto values
        are derived from each shard's ``meta/info.json`` (typically identical
        fps across a rolling run).  Not stored per shard — recomputed on each
        lazy :class:`LeRobotDataset` open.
        """
        if self.chunk_size <= 1:
            return None
        if self._user_delta_timestamps is not None:
            return self._user_delta_timestamps
        with open(ds_path / "meta" / "info.json", encoding="utf-8") as f:
            info = json.load(f)
        return _build_delta_timestamps(info, self.chunk_size, self.action_sequence_keys)

    def _probe_shard_for_index(self, ds_path: Path) -> _ShardIndexProbe:
        """Open *ds_path* once and return length / optional intervene mask (thread-safe per path).

        When ``in_memory_mode`` is enabled and *ds_path* is already present in
        :attr:`_in_memory_shards` (typically because :meth:`add_shard_to_memory`
        ran after the writer finalized the shard), metadata is taken from the
        in-memory store so :class:`~lerobot.common.datasets.lerobot_dataset.LeRobotDataset`
        is not opened for indexing.
        """
        ds_path = Path(ds_path)
        if self._shard_cache_enabled:
            store = self._in_memory_shards.get(ds_path)
            if store is not None:
                n_frames = len(store)
                num_episodes = store.num_episodes
                intervene_locals: list[int] | None = None
                if self.require_all_intervene:
                    intervene_locals = _compute_intervene_valid_local_indices_in_memory(
                        store,
                        self.intervene_flag_key,
                        self.chunk_size,
                    )
                return _ShardIndexProbe(
                    ds_path=ds_path,
                    ok=True,
                    sub_ds=None,
                    n_frames=n_frames,
                    num_episodes=num_episodes,
                    intervene_locals=intervene_locals,
                )

        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

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
            return _ShardIndexProbe(ds_path=ds_path, ok=False)
        n_frames = len(sub_ds)
        num_episodes = int(getattr(sub_ds, "num_episodes", 0))
        intervene_locals: list[int] | None = None
        if self.require_all_intervene:
            intervene_locals = _compute_intervene_valid_local_indices(
                sub_ds, self.intervene_flag_key
            )
        return _ShardIndexProbe(
            ds_path=ds_path,
            ok=True,
            sub_ds=sub_ds,
            n_frames=n_frames,
            num_episodes=num_episodes,
            intervene_locals=intervene_locals,
        )

    def _build_index(
        self,
        datasets: list[Path],
        out_open_handles: dict[Path, Any] | None = None,
    ) -> int:
        """Index *datasets*: open each briefly to measure length / intervene mask, then store paths only.

        When :attr:`_index_load_workers` is greater than ``1`` and there are
        multiple pending shards, openings run in a thread pool (I/O overlap).
        Merge into :attr:`_cumulative_lengths` stays sequential so global
        frame order matches *datasets* order.

        Args:
            datasets: Sub-dataset root paths to load.
            out_open_handles: When not ``None``, each successfully opened
                :class:`LeRobotDataset` is stored under its root path so a
                matching :meth:`_ingest_decoded_cache` call can reuse it instead
                of opening the same shard twice in one :meth:`refresh`.

        Returns:
            Number of new sub-datasets successfully added.
        """
        pending = [p for p in datasets if p not in self._indexed_datasets]
        n_new = 0
        if not pending:
            self._update_window_sampling_bounds()
            return 0

        workers = self._index_load_workers
        if workers > 1 and len(pending) > 1:
            max_w = min(workers, len(pending))
            with ThreadPoolExecutor(max_workers=max_w) as ex:
                probes: list[_ShardIndexProbe] = list(
                    ex.map(self._probe_shard_for_index, pending)
                )
        else:
            probes = [self._probe_shard_for_index(p) for p in pending]

        for probe in probes:
            if not probe.ok or probe.sub_ds is None:
                continue
            ds_path = probe.ds_path
            sub_ds = probe.sub_ds
            physical_base = self._cumulative_lengths[-1]
            n_frames = probe.n_frames
            self._sub_datasets.append(ds_path)
            self._cumulative_lengths.append(physical_base + n_frames)
            if (
                self.require_all_intervene
                and self._valid_physical_indices is not None
                and probe.intervene_locals is not None
            ):
                assert self._valid_physical_set is not None
                for local_i in probe.intervene_locals:
                    gidx = physical_base + int(local_i)
                    self._valid_physical_indices.append(gidx)
                    self._valid_physical_set.add(gidx)
            self._indexed_datasets.add(ds_path)
            self._total_episodes += probe.num_episodes
            n_new += 1
            if out_open_handles is not None:
                out_open_handles[ds_path] = sub_ds

        self._update_window_sampling_bounds()
        return n_new

    def _ensure_lerobot_open(self, ds_idx: int) -> Any:
        """Return a :class:`LeRobotDataset` for shard *ds_idx*, (re)opening if needed."""
        if self._lerobot_open_idx == ds_idx and self._lerobot_open is not None:
            return self._lerobot_open
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        root = self._sub_datasets[ds_idx]
        delta_timestamps = self._get_delta_timestamps(root)
        self._lerobot_open = LeRobotDataset(
            repo_id=root.name,
            root=root,
            delta_timestamps=delta_timestamps,
            image_transforms=self.image_transforms,
            download_videos=False,
        )
        self._lerobot_open_idx = ds_idx
        return self._lerobot_open

    def _load_item_from_open_lerobot(
        self, lerobot_ds: Any, local_idx: int
    ) -> dict[str, Any]:
        """Decode one sample from an already-open :class:`LeRobotDataset`."""
        item: dict[str, Any] = lerobot_ds[local_idx]
        if self.keys is not None:
            item = {k: v for k, v in item.items() if k in self.keys}
        return item

    def _load_item_from_lerobot(self, idx: int) -> dict[str, Any]:
        """Fetch one sample (including chunk windows) from the backing store.

        When ``in_memory_mode=True`` and the shard containing *idx* is cached,
        the :class:`InMemoryArrowStore` for that shard is used directly,
        avoiding any disk I/O.  Falls through to the LeRobot disk path when
        the shard is not cached (e.g. older shards evicted from RAM).
        Either way the returned dict is key-filtered by ``self.keys`` and has
        the same chunk / ``*_is_pad`` structure produced by
        :class:`~lerobot.common.datasets.lerobot_dataset.LeRobotDataset`.
        """
        ds_idx = bisect.bisect_right(self._cumulative_lengths, idx) - 1
        local_idx = idx - self._cumulative_lengths[ds_idx]
        if self._shard_cache_enabled:
            path = self._sub_datasets[ds_idx]
            store = self._in_memory_shards.get(path)
            if store is not None:
                self._shard_cache_hits += 1
                item = store[local_idx]
                if self.keys is not None:
                    item = {k: v for k, v in item.items() if k in self.keys}
                return item
            self._shard_cache_misses += 1
        lerobot_ds = self._ensure_lerobot_open(ds_idx)
        return self._load_item_from_open_lerobot(lerobot_ds, local_idx)

    def _ingest_physical_indices_sharded(
        self,
        physical_indices: list[int],
        reuse_open_by_path: dict[Path, Any] | None,
    ) -> None:
        """Fill the decoded cache for *physical_indices* with at most one open per shard.

        Indices are grouped by backing sub-dataset.  Each shard is opened once
        (or taken from *reuse_open_by_path* for paths just indexed in the same
        ``refresh()``), samples are read in sorted global order, then the lazy
        handle :attr:`_lerobot_open` is cleared so ingest does not retain
        extra LeRobot handles.
        """
        cache = self._decoded_cache
        if cache is None or not physical_indices:
            return
        by_shard: dict[int, list[int]] = {}
        for g in physical_indices:
            ds_idx = bisect.bisect_right(self._cumulative_lengths, g) - 1
            by_shard.setdefault(ds_idx, []).append(int(g))
        try:
            for ds_idx in sorted(by_shard.keys()):
                idxs = sorted(by_shard[ds_idx])
                path = self._sub_datasets[ds_idx]
                base = self._cumulative_lengths[ds_idx]
                reused = (
                    reuse_open_by_path is not None
                    and path in reuse_open_by_path
                    and reuse_open_by_path[path] is not None
                )
                if reused:
                    lerobot_ds = reuse_open_by_path[path]
                else:
                    lerobot_ds = self._ensure_lerobot_open(ds_idx)
                for gidx in idxs:
                    try:
                        local_idx = gidx - base
                        item = self._load_item_from_open_lerobot(lerobot_ds, local_idx)
                        cache.put(gidx, item)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[RollingLeRobotDataset] cache ingest failed at idx=%s: %s",
                            gidx,
                            exc,
                        )
        finally:
            self._lerobot_open = None
            self._lerobot_open_idx = None

    def _ingest_decoded_cache(
        self,
        physical_before: int,
        physical_after: int,
        n_new: int,
        reuse_open_by_path: dict[Path, Any] | None = None,
    ) -> None:
        """Populate FIFO cache according to ``cache_ingest_mode``.

        ``physical_before`` / ``physical_after`` are cumulative **physical**
        frame counts (unfiltered) immediately before vs. after a refresh that
        appended new shards; they define the half-open index range for the
        ``new_shards`` ingest path.

        Args:
            reuse_open_by_path: Optional map of sub-dataset root → already-open
                :class:`LeRobotDataset` built during :meth:`_build_index` in the
                same operation, so ingest does not open those paths a second time.
        """
        cache = self._decoded_cache
        if cache is None:
            return
        indices: list[int] = []
        mode = self.cache_ingest_mode
        if mode in ("new_shards", "both") and n_new > 0:
            indices.extend(range(int(physical_before), int(physical_after)))
        if mode in ("last_n", "both"):
            n_phys = self._num_physical_frames()
            tail = self.cache_last_n_frames
            start = max(0, n_phys - tail)
            indices.extend(range(start, n_phys))
        seen: set[int] = set()
        uniq: list[int] = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        if self._valid_physical_set is not None:
            uniq = [i for i in uniq if i in self._valid_physical_set]
        if self._cache_ingest_world_size > 1:
            uniq = uniq[self._cache_ingest_rank :: self._cache_ingest_world_size]
        lim = self.cache_ingest_max_frames
        if lim is not None:
            uniq = uniq[: int(lim)]
        self._ingest_physical_indices_sharded(uniq, reuse_open_by_path)

    # ------------------------------------------------------------------
    # Shard cache API
    # ------------------------------------------------------------------

    def add_shard_to_memory(
        self, path: str | Path, episodes: list[list[dict]]
    ) -> None:
        """Populate the in-memory shard cache for *path* with *episodes*.

        Should be called by the actor worker immediately after a shard is
        finalized on disk (via
        :meth:`~rlinf.data.lerobot_writer.LeRobotDatasetWriter.finalize`),
        so the next :meth:`__getitem__` call for any frame in that shard can
        be served from RAM instead of reading back from disk.

        Thread-safe.  No-op when ``in_memory_mode=False``.

        Args:
            path: Root path of the finalized shard, matching the ``repo_id``
                passed to
                :meth:`~rlinf.data.lerobot_writer.LeRobotDatasetWriter.create`.
            episodes: Ordered list of episodes in the shard; each episode is a
                ``list[dict]`` of per-step frame dicts as produced by
                :meth:`~rlinf.envs.wrappers.collect_episode.CollectEpisode._buffer_to_lerobot_ep`.
        """
        if not self._shard_cache_enabled or not episodes:
            return
        store = InMemoryArrowStore(
            chunk_size=self._shard_cache_chunk_size,
            action_sequence_keys=self._shard_cache_action_keys,
            fps=self._shard_cache_fps,
            image_transforms=self.image_transforms,
        )
        for ep_frames in episodes:
            if ep_frames:
                store.add_episode(ep_frames)
        path = Path(path)
        with self._rolling_access_lock:
            self._in_memory_shards[path] = store
            logger.debug(
                "[RollingLeRobotDataset] shard cached: %s (%d episodes, %d frames)",
                path.name,
                len(store._episode_datasets),
                len(store),
            )

    def _evict_stale_shards(self) -> int:
        """Remove shard stores from RAM whose frames are all before the window.

        Must be called under :attr:`_rolling_access_lock`, after
        :meth:`_update_window_sampling_bounds`.  Shards whose cumulative end
        frame index ``<= _window_physical_start`` lie fully outside the
        sampling window and their Arrow tables are freed from RAM; the disk
        copy remains as a transparent fallback.

        Returns:
            Number of shard stores evicted.
        """
        if not self._in_memory_shards or not self._window_enabled():
            return 0
        n_evicted = 0
        for ds_idx, path in enumerate(self._sub_datasets):
            shard_end = self._cumulative_lengths[ds_idx + 1]
            if shard_end <= self._window_physical_start:
                if self._in_memory_shards.pop(path, None) is not None:
                    n_evicted += 1
        if n_evicted:
            logger.debug(
                "[RollingLeRobotDataset] evicted %d shard(s) from shard cache "
                "(window_physical_start=%d)",
                n_evicted,
                self._window_physical_start,
            )
        return n_evicted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _refresh_impl(self, new_paths: list[Path]) -> int:
        """Shared implementation for :meth:`refresh` and :meth:`refresh_one`.

        Indexes *new_paths*, optionally ingests them into the decoded cache,
        and cleans up transient open handles.

        Returns:
            Number of new sub-datasets successfully added.
        """
        if not new_paths:
            return 0
        physical_before = self._num_physical_frames()
        n_new = 0
        reuse_handles: dict[Path, Any] = {}
        with self._rolling_access_lock:
            try:
                n_new = self._build_index(
                    new_paths,
                    out_open_handles=(
                        reuse_handles if self._decoded_cache is not None else None
                    ),
                )
                physical_after = self._num_physical_frames()
                if n_new > 0:
                    logger.info(
                        "[RollingLeRobotDataset] refresh: +%d sub-dataset(s), "
                        "physical_frames=%d total_logical=%d "
                        "windowed_logical=%d",
                        n_new,
                        physical_after,
                        self._total_logical_samples(),
                        len(self),
                    )
                self._evict_stale_shards()
                if self._decoded_cache is not None:
                    self._ingest_decoded_cache(
                        physical_before,
                        physical_after,
                        n_new,
                        reuse_open_by_path=reuse_handles or None,
                    )
            finally:
                reuse_handles.clear()
                if self._decoded_cache is not None:
                    self._lerobot_open = None
                    self._lerobot_open_idx = None
        return n_new

    def refresh(self) -> int:
        """Ingest newly available data.

        Re-scans ``root_dir`` for newly completed LeRobot sub-datasets,
        optionally ingests them into the decoded FIFO cache, and (when
        ``in_memory_mode=True``) evicts shard stores that have scrolled out
        of the rolling window.

        Returns:
            Number of new sub-datasets added; ``0`` if nothing changed.
        """
        safe = _discover_safe_datasets(self.root_dir, self.skip_last_n)
        new_paths = [p for p in safe if p not in self._indexed_datasets]
        return self._refresh_impl(new_paths)

    def refresh_one(self) -> int:
        """Ingest one new sub-dataset (at most) per call.

        Bounds I/O for tight polling loops.  In disk mode at most one new
        sub-dataset is indexed per call.

        Returns:
            Number of items added; ``0`` if nothing changed.
        """
        safe = _discover_safe_datasets(self.root_dir, self.skip_last_n)
        new_paths = [p for p in safe if p not in self._indexed_datasets]
        if not new_paths:
            return 0
        return self._refresh_impl(new_paths[:1])

    def _total_logical_samples(self) -> int:
        """Total logical samples across all shards, ignoring ``window_size``."""
        if self._valid_physical_indices is not None:
            return len(self._valid_physical_indices)
        return self._num_physical_frames()

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "num_sub_datasets": len(self._sub_datasets),
            "physical_frames": self._num_physical_frames(),
            "total_logical_samples": self._total_logical_samples(),
            "logical_samples": len(self),
            "total_frames": len(self),
            "total_episodes": self._total_episodes,
            "require_all_intervene": self.require_all_intervene,
            "window_size": self.window_size if self.window_size is not None else 0,
            "window_physical_start": self._window_physical_start,
            "shard_cache_enabled": self._shard_cache_enabled,
            "shard_cache_shards": len(self._in_memory_shards),
            "shard_cache_hits": self._shard_cache_hits,
            "shard_cache_misses": self._shard_cache_misses,
        }
        if self._decoded_cache is not None:
            stats.update(self._decoded_cache.stats())
        return stats

    def __len__(self) -> int:
        if self._valid_physical_indices is not None:
            if self._window_enabled():
                return len(self._valid_physical_indices) - self._window_valid_slice_lo
            return len(self._valid_physical_indices)
        n = int(self._cumulative_lengths[-1])
        if self._window_enabled():
            return max(0, n - self._window_physical_start)
        return n

    def get_cache_aligned_logical_indices(self) -> list[int] | None:
        """Return logical indices whose physical counterparts are in this rank's decoded cache.

        When ``cache_ingest_world_size > 1``, each rank only ingests a strided
        shard of physical frame indices.  The
        :class:`~rlinf.data.utils.DistributedRandomReplacementSampler`, however,
        draws logical indices uniformly — which can land on physical indices not
        present in this rank's cache, causing misses.

        This method returns the subset of logical indices ``[0, len(self))``
        whose physical counterparts are currently cached, so a cache-aligned
        sampler can restrict drawing to those indices only.

        Returns ``None`` when alignment is unnecessary (no decoded cache, or
        ``cache_ingest_world_size <= 1``).
        """
        if self._decoded_cache is None or self._cache_ingest_world_size <= 1:
            return None
        cached = self._decoded_cache.cached_indices()
        if not cached:
            return None
        n = len(self)
        aligned = [l for l in range(n) if self._logical_to_physical(l) in cached]
        return aligned if aligned else None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        with self._rolling_access_lock:
            physical = self._logical_to_physical(int(idx))
            cache = self._decoded_cache
            if cache is not None:
                hit = cache.try_get(physical)
                if hit is not None:
                    return hit
                cache.notify_miss()
            return self._load_item_from_lerobot(physical)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Batch fetch for DataLoader (one call per batch when supported)."""
        if not indices:
            return []
        with self._rolling_access_lock:
            physicals = [self._logical_to_physical(int(i)) for i in indices]
            cache = self._decoded_cache
            if cache is None:
                return [self._load_item_from_lerobot(p) for p in physicals]
            out: list[dict[str, Any]] = []
            for physical in physicals:
                hit = cache.try_get(physical)
                if hit is not None:
                    out.append(hit)
                else:
                    cache.notify_miss()
                    out.append(self._load_item_from_lerobot(physical))
            return out


# ---------------------------------------------------------------------------
# Preload wrapper
# ---------------------------------------------------------------------------


class PreloadRollingLeRobotDataset:
    """Prefetches batches from a DataLoader-backed :class:`RollingLeRobotDataset`.

    Uses exactly the same :func:`build_dataloader_from_dataset` call
    (``DistributedSampler`` or ``DistributedRandomReplacementSampler``) as the
    non-preload path, then adds a daemon thread that iterates through DataLoader
    epochs in the background and stores ready batches in a bounded queue.

    This means both paths share the same sampler semantics (distributed
    sharding, epoch-seeded shuffling, with/without replacement) and differ
    only in whether batches are fetched synchronously or pre-fetched.

    :meth:`__len__` mirrors ``len(data_loader)`` so the consumer can drive
    the same gradient-accumulation logic regardless of which path is active.

    When new sub-datasets arrive, :meth:`refresh` rebuilds the DataLoader
    with the updated dataset length.  The background thread detects the swap
    at the start of the next batch and seamlessly continues from the new
    loader.

    Args:
        dataset: The :class:`RollingLeRobotDataset` to prefetch from.
        batch_size: Forwarded to :func:`build_dataloader_from_dataset`.
        world_size: Number of distributed replicas.  Defaults to ``1``.
        rank: Rank of the current process.  Defaults to ``0``.
        prefetch_size: Maximum number of batches buffered in the queue.
            Defaults to ``5``.
        use_random_replacement: Passed to :func:`build_dataloader_from_dataset`.
            Defaults to ``True``.
        num_samples_per_epoch: Samples per epoch for the internal sampler.
        seed: Random seed forwarded to the sampler.
        num_workers: DataLoader worker processes.  Defaults to ``4``.
        **dataloader_kwargs: Extra kwargs forwarded to
            :func:`build_dataloader_from_dataset`.
    """

    def __init__(
        self,
        dataset: RollingLeRobotDataset,
        batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        prefetch_size: int = 5,
        use_random_replacement: bool = True,
        num_samples_per_epoch: int | None = None,
        seed: int = 42,
        num_workers: int = 4,
        **dataloader_kwargs: Any,
    ) -> None:
        assert prefetch_size > 0, f"{prefetch_size=} must be greater than 0"

        self.dataset = dataset
        # Cache all DataLoader construction kwargs for rebuild on refresh().
        self._dl_kwargs: dict[str, Any] = dict(
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            use_random_replacement=use_random_replacement,
            num_samples_per_epoch=num_samples_per_epoch,
            seed=seed,
            num_workers=num_workers,
            **dataloader_kwargs,
        )
        self.prefetch_size = prefetch_size

        self._stop_event = threading.Event()
        # Guards swaps of self._loader so the background thread sees a
        # consistent reference when refresh() installs a new DataLoader.
        self._loader_lock = threading.Lock()
        self._bg_epoch: int = 0
        self.preload_queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=prefetch_size
        )
        self.sample_thread: threading.Thread | None = None
        self._exception: Exception | None = None

        self._loader: DataLoader = self._build_loader()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_loader(self) -> DataLoader:
        """Build a fresh DataLoader using the cached construction kwargs.

        Recomputes cache-aligned indices from the dataset's current cache
        state so that each rebuilt loader (after a refresh) only draws from
        logical indices backed by this rank's decoded cache shard.
        """
        aligned = self.dataset.get_cache_aligned_logical_indices()
        return build_dataloader_from_dataset(
            dataset=self.dataset,
            cache_aligned_indices=aligned,
            **self._dl_kwargs,
        )

    def _sample_worker(self) -> None:
        """Background thread: iterate DataLoader epochs and enqueue batches.

        Holds a local reference to the current DataLoader.  When
        :meth:`refresh` installs a new one (under ``_loader_lock``), the
        thread detects the swap at the top of the loop and resets its
        iterator.  Epoch counter and sampler ``set_epoch`` calls are managed
        here so the background stream is always correctly shuffled.
        """
        current_loader: DataLoader | None = None
        loader_iter = None

        while not self._stop_event.is_set():
            if self.preload_queue.full():
                time.sleep(0.1)
                continue

            if not self.dataset.is_ready():
                time.sleep(3)
                continue

            # Pick up a rebuilt loader installed by refresh(), if any.
            with self._loader_lock:
                if self._loader is not current_loader:
                    current_loader = self._loader
                    loader_iter = None  # reset iterator for the new loader

            if loader_iter is None:
                if hasattr(current_loader.sampler, "set_epoch"):
                    current_loader.sampler.set_epoch(self._bg_epoch)
                loader_iter = iter(current_loader)

            try:
                batch = next(loader_iter)
            except StopIteration:
                # One DataLoader epoch exhausted: advance and restart.
                self._bg_epoch += 1
                if hasattr(current_loader.sampler, "set_epoch"):
                    current_loader.sampler.set_epoch(self._bg_epoch)
                loader_iter = iter(current_loader)
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    time.sleep(1)
                    continue
            except Exception as e:  # noqa: BLE001
                logger.error("[PreloadRollingLeRobotDataset] sampling error: %s", e)
                self._exception = e
                self._stop_event.set()
                break

            try:
                self.preload_queue.put(batch, timeout=1)
            except queue.Full:
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of batches per epoch — mirrors the internal DataLoader."""
        return len(self._loader)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Return an infinite iterator of prefetched batches.

        Starts the background sampling thread on the first call.

        Yields:
            Collated batch dict produced by the internal DataLoader.
        """
        if self.sample_thread is None:
            self.sample_thread = threading.Thread(
                target=self._sample_worker, daemon=True
            )
            self.sample_thread.start()

        while not self._stop_event.is_set():
            try:
                batch = self.preload_queue.get(timeout=1)
                yield batch
            except queue.Empty:
                if self._stop_event.is_set():
                    if self._exception is not None:
                        raise RuntimeError(
                            "Sampling thread failed"
                        ) from self._exception
                    break
                continue

    def refresh(self) -> int:
        """Refresh the dataset and rebuild the DataLoader if new data arrived.

        Calls :meth:`RollingLeRobotDataset.refresh`.  When new sub-datasets
        are discovered, rebuilds the internal :class:`~torch.utils.data.DataLoader`
        (updating sampler length to reflect the larger dataset) and stores it
        under ``_loader_lock`` so the background thread picks it up cleanly.

        Returns:
            Number of new sub-datasets added (0 if nothing changed).
        """
        n_new = self.dataset.refresh()
        if n_new > 0:
            with self._loader_lock:
                self._loader = self._build_loader()
        return n_new

    def refresh_one(self) -> int:
        """Like :meth:`refresh`, but loads at most one new sub-dataset.

        Delegates to :meth:`RollingLeRobotDataset.refresh_one` and rebuilds
        the DataLoader when a new shard is picked up.  Call repeatedly in a
        polling loop for incremental, bounded-cost refreshes.

        Returns:
            ``1`` if a new sub-dataset was added, ``0`` otherwise.
        """
        n_new = self.dataset.refresh_one()
        if n_new > 0:
            with self._loader_lock:
                self._loader = self._build_loader()
        return n_new

    def close(self) -> None:
        """Stop the background thread and release resources."""
        self._stop_event.set()
        thread_timeout = 10
        if self.sample_thread is not None and self.sample_thread.is_alive():
            self.sample_thread.join(timeout=thread_timeout)
            if self.sample_thread.is_alive():
                logger.warning(
                    "[PreloadRollingLeRobotDataset] sample thread still alive "
                    "after %d seconds, force killing",
                    thread_timeout,
                )

    def __del__(self) -> None:
        """Destructor that ensures the sampling thread is stopped."""
        if not self._stop_event.is_set():
            self.close()


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
    min_frames: int = 1,
    wait_interval_s: float = 10.0,
    action_sequence_keys: list[str] | None = ["actions"],
    enable_decoded_cache: bool = False,
    decoded_cache_capacity: int = 8192,
    cache_ingest_mode: CacheIngestMode = "new_shards",
    cache_last_n_frames: int = 10_000,
    cache_ingest_max_frames: int | None = None,
    require_all_intervene: bool = False,
    intervene_flag_key: str = "intervene_flag",
    window_size: int | None = None,
    index_load_workers: int = 1,
    cache_ingest_rank: int = 0,
    cache_ingest_world_size: int = 1,
    in_memory_mode: bool = False,
    fps: int = 10,
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
        min_frames: Minimum number of safe sub-datasets required before the
            dataset returns.  Construction sleeps until the threshold is met.
            Defaults to ``1``.
        wait_interval_s: Seconds between readiness polls.  Defaults to ``10.0``.
        action_sequence_keys: List of keys to apply chunking to.  Defaults to
            ``["actions"]``.
        enable_decoded_cache: Enable in-memory decoded FIFO cache.
        decoded_cache_capacity: Ring capacity for decoded samples.
        cache_ingest_mode: ``new_shards``, ``last_n``, or ``both``.
        cache_last_n_frames: Tail size for ``last_n`` / ``both``.
        cache_ingest_max_frames: Max ingests per ``refresh()`` (``None`` = unlimited).
        require_all_intervene: See :class:`RollingLeRobotDataset`.
        intervene_flag_key: Column name for the per-frame intervention flag.
        window_size: Caps the dataset to the last ``window_size`` **logical**
            frames.  See :class:`RollingLeRobotDataset`.
        index_load_workers: Parallel thread count for opening sub-datasets
            during index build (``1`` = sequential).  See
            :class:`RollingLeRobotDataset`.
        cache_ingest_rank: See :class:`RollingLeRobotDataset`.
        cache_ingest_world_size: See :class:`RollingLeRobotDataset`.

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
        min_frames=min_frames,
        wait_interval_s=wait_interval_s,
        action_sequence_keys=action_sequence_keys,
        enable_decoded_cache=enable_decoded_cache,
        decoded_cache_capacity=decoded_cache_capacity,
        cache_ingest_mode=cache_ingest_mode,
        cache_last_n_frames=cache_last_n_frames,
        cache_ingest_max_frames=cache_ingest_max_frames,
        require_all_intervene=require_all_intervene,
        intervene_flag_key=intervene_flag_key,
        window_size=window_size,
        index_load_workers=index_load_workers,
        cache_ingest_rank=cache_ingest_rank,
        cache_ingest_world_size=cache_ingest_world_size,
        in_memory_mode=in_memory_mode,
        fps=fps,
    )

    logger.info(
        "[build_rolling_lerobot_dataset] root_dir=%s, chunk_size=%d, "
        "skip_last_n=%d, sub_datasets=%d, logical_samples=%d, "
        "physical_frames=%d, decoded_cache=%s, require_all_intervene=%s, "
        "window_size=%s",
        root_dir,
        chunk_size,
        skip_last_n,
        len(dataset._sub_datasets),
        len(dataset),
        dataset._num_physical_frames(),
        enable_decoded_cache,
        require_all_intervene,
        window_size,
    )

    return dataset
