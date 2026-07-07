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

from __future__ import annotations

import bisect
import io
import json
import threading
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.utils.logging import get_logger

logger = get_logger()

# Parquet index / metadata columns that should not be chunked along the time
# axis — they are scalar per frame and have no meaningful "window" semantics.
_META_KEYS: frozenset[str] = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)


def _hf_column_to_numpy_bool_1d(hf_dataset: Any, col: str) -> np.ndarray:
    """Load a per-row boolean column as ``(num_rows,)`` bool numpy array."""
    raw = hf_dataset[col]
    if isinstance(raw, torch.Tensor):
        t = raw
    else:
        t = torch.stack(list(raw))
    return t.reshape(-1).bool().numpy()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_lerobot_tasks(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    return {
        int(row["task_index"]): str(row["task"])
        for row in _read_jsonl(path)
        if "task_index" in row and "task" in row
    }


def _episode_task(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks", [])
    if tasks:
        return str(tasks[0])
    task = episode.get("task", "")
    return str(task) if task is not None else ""


def _episode_parquet_path(
    shard_path: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
) -> Path:
    episode_index = int(episode["episode_index"])
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return shard_path / data_path.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )


def _load_lerobot_episode_frames(
    parquet_path: Path,
    shard_path: Path,
    feature_specs: dict[str, Any],
    task_by_index: dict[int, str],
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet_path).to_pylist()
    return [
        _lerobot_row_to_frame(
            row,
            shard_path,
            feature_specs,
            task_by_index,
        )
        for row in rows
    ]


def _lerobot_row_to_frame(
    row: dict[str, Any],
    shard_path: Path,
    feature_specs: dict[str, Any],
    task_by_index: dict[int, str],
) -> dict[str, Any]:
    task = ""
    if "task" in row and row["task"] is not None:
        task = str(row["task"])
    elif "task_index" in row and row["task_index"] is not None:
        task = task_by_index.get(int(row["task_index"]), "")

    frame: dict[str, Any] = {"task": task}
    for key, value in row.items():
        if key in _META_KEYS or key == "task" or value is None:
            continue
        if _looks_like_lerobot_image(value):
            image = _decode_lerobot_image(value, shard_path)
            if image is not None:
                frame[key] = image
            continue
        frame[key] = _lerobot_value_to_array(
            value,
            feature_specs.get(key, {}).get("dtype"),
        )
    return frame


def _looks_like_lerobot_image(value: Any) -> bool:
    if isinstance(value, dict):
        return "bytes" in value or "path" in value
    return isinstance(value, (bytes, str, Path))


def _decode_lerobot_image(value: Any, shard_path: Path) -> np.ndarray | None:
    import PIL.Image as PILImage

    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        image_path = value.get("path")
    else:
        raw_bytes = value if isinstance(value, bytes) else None
        image_path = value if isinstance(value, (str, Path)) else None

    if raw_bytes is not None:
        with PILImage.open(io.BytesIO(raw_bytes)) as image:
            return np.asarray(image.convert("RGB"))

    if image_path is None:
        return None

    path = Path(image_path)
    if not path.is_absolute():
        path = shard_path / path
    if not path.is_file():
        return None
    with PILImage.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _lerobot_value_to_array(value: Any, dtype: str | None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if dtype and dtype != "image":
        arr = arr.astype(dtype, copy=False)
    return arr


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
                dtype_str = (
                    "bool" if np.issubdtype(val.dtype, np.bool_) else str(val.dtype)
                )
                data_features[key] = hf_datasets.Value(dtype_str)
            elif val.ndim == 1:
                dtype_str = (
                    "bool" if np.issubdtype(val.dtype, np.bool_) else str(val.dtype)
                )
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

    def _prepare_episode_dataset(
        self,
        ep_frames: list[dict],
        base: int | None = None,
        episode_index: int | None = None,
    ) -> dict[str, Any]:
        import PIL.Image as PILImage
        from datasets import Dataset
        from lerobot.common.datasets.utils import hf_transform_to_torch

        n = len(ep_frames)
        if n == 0:
            return {}

        hf_features = self._hf_features
        image_keys = self._image_keys
        if hf_features is None:
            hf_features, image_keys = self._infer_hf_features(ep_frames[0])

        ep_idx = len(self._ep_from) if episode_index is None else int(episode_index)
        base = self._total_frames if base is None else int(base)
        task_to_idx = dict(self._task_to_idx)
        tasks = dict(self._tasks)

        # Register task strings and build per-frame task_index array.
        task_indices: list[int] = []
        for frame in ep_frames:
            task_str = frame.get("task", "")
            if task_str not in task_to_idx:
                tidx = len(task_to_idx)
                task_to_idx[task_str] = tidx
                tasks[tidx] = task_str
            task_indices.append(task_to_idx[task_str])

        ep_dict: dict[str, Any] = {
            "index": np.arange(base, base + n, dtype=np.int64),
            "episode_index": np.full((n,), ep_idx, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "timestamp": np.arange(n, dtype=np.float32) / self._fps,
            "task_index": np.array(task_indices, dtype=np.int64),
        }

        for key in hf_features:
            if key in (
                "index",
                "episode_index",
                "frame_index",
                "timestamp",
                "task_index",
            ):
                continue
            if key in image_keys:
                ep_dict[key] = [PILImage.fromarray(f[key]) for f in ep_frames]
            else:
                vals = [f.get(key) for f in ep_frames]
                if all(v is not None for v in vals):
                    stacked = np.stack(vals)
                    # Scalar (1,) — squeeze to 1-D so Arrow Value dtype matches.
                    ep_dict[key] = (
                        stacked.squeeze(1) if stacked.shape == (n, 1) else stacked
                    )

        ep_ds = Dataset.from_dict(ep_dict, features=hf_features)
        ep_ds.set_transform(hf_transform_to_torch)
        return {
            "dataset": ep_ds,
            "base": base,
            "num_frames": n,
            "hf_features": hf_features,
            "image_keys": image_keys,
            "task_to_idx": task_to_idx,
            "tasks": tasks,
        }

    def _commit_prepared_episode(self, prepared_episode: dict[str, Any]) -> int:
        if not prepared_episode:
            return 0

        base = int(prepared_episode["base"])
        if base != self._total_frames:
            raise ValueError("Prepared episode base no longer matches store length.")

        n = int(prepared_episode["num_frames"])
        self._hf_features = prepared_episode["hf_features"]
        self._image_keys = prepared_episode["image_keys"]
        self._task_to_idx = prepared_episode["task_to_idx"]
        self._tasks = prepared_episode["tasks"]
        self._episode_datasets.append(prepared_episode["dataset"])
        self._ep_from.append(base)
        self._ep_to.append(base + n)
        self._total_frames += n
        return n

    def add_episode(self, ep_frames: list[dict]) -> None:
        prepared_episode = self._prepare_episode_dataset(ep_frames)
        if not prepared_episode:
            return
        self._commit_prepared_episode(prepared_episode)

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
                key: [max(ep_start, min(ep_end - 1, local_idx + d)) for d in deltas]
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
                key: torch.stack(ep_ds.select([q - ep_start for q in q_idxs])[key])
                for key, q_idxs in query_indices.items()
                if key in self._hf_features and key not in self._image_keys
            }
            item = {**item, **padding}
            for k, v in query_result.items():
                item[k] = v

        return item


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
            met.  Defaults to ``10``.
        wait_interval_s: Seconds to sleep between readiness polls when fewer
            than ``min_frames`` total frames are available.  Defaults to
            ``10.0``.
        action_sequence_keys: Keys to apply ``delta_timestamps`` chunking to.
            Defaults to ``["actions"]``.
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
            trajectories. ``None`` or ``0`` disables windowing (full history).
        in_memory_mode: Must be ``True``.  Newly written shards are loaded into
            an in-memory Arrow store for training reads.
        fps: Dataset frame rate used when auto-generating ``delta_timestamps``.
            Defaults to ``10``.
    """

    def __init__(
        self,
        root_dir: str | Path,
        chunk_size: int = 1,
        delta_timestamps: dict[str, list[float]] | None = None,
        keys: list[str] | None = None,
        image_transforms: Callable | None = None,
        min_frames: int = 10,
        wait_interval_s: float = 10.0,
        action_sequence_keys: list[str] | None = ["actions"],
        # check intervene
        require_all_intervene: bool = False,
        intervene_flag_key: str = "intervene_flag",
        window_size: int | None = None,
        in_memory_mode: bool = False,
        fps: int = 10,
    ) -> None:
        if not in_memory_mode:
            raise ValueError(
                "RollingLeRobotDataset now supports only in_memory_mode=True. "
                "Archived LeRobot shards must be loaded into memory before training."
            )
        self.root_dir = Path(root_dir)
        self.chunk_size = chunk_size
        self.keys = keys
        self.image_transforms = image_transforms
        self.min_frames = min_frames
        self.wait_interval_s = wait_interval_s
        self.action_sequence_keys = action_sequence_keys
        self.require_all_intervene = bool(require_all_intervene)
        self.intervene_flag_key = intervene_flag_key
        self.window_size = window_size
        self._window_physical_start: int = 0
        self._window_valid_slice_lo: int = 0
        self._valid_physical_indices: list[int] | None = None
        self._valid_physical_set: set[int] | None = None
        if self.require_all_intervene:
            self._valid_physical_indices = []
            self._valid_physical_set = set()
        # Serializes in-memory index growth vs __getitem__/__getitems__.
        self._rolling_access_lock = threading.RLock()

        # Sub-dataset **roots** indexed so far (paths only; no live LeRobot handles).
        self._sub_datasets: list[Path] = []

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
        self._shard_cache_enabled: bool = True
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

    # ------------------------------------------------------------------
    # Readiness gate
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        print(f"rdy chk, self: {len(self)}, min: {self.min_frames}")
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

    def _load_item_from_open_lerobot(
        self, lerobot_ds: Any, local_idx: int
    ) -> dict[str, Any]:
        """Decode one sample from an already-open :class:`LeRobotDataset`."""
        item: dict[str, Any] = lerobot_ds[local_idx]
        if self.keys is not None:
            item = {k: v for k, v in item.items() if k in self.keys}
        return item

    def _load_item_from_lerobot(self, idx: int) -> dict[str, Any]:
        ds_idx = bisect.bisect_right(self._cumulative_lengths, idx) - 1
        local_idx = idx - self._cumulative_lengths[ds_idx]

        path = self._sub_datasets[ds_idx]
        store = self._in_memory_shards.get(path)
        if store is not None:
            self._shard_cache_hits += 1
            item = store[local_idx]
            if self.keys is not None:
                item = {k: v for k, v in item.items() if k in self.keys}
            return item
        self._shard_cache_misses += 1
        raise RuntimeError(
            f"Indexed in-memory LeRobot shard is missing: {path}. "
            "Training does not fall back to archived disk shards."
        )

    # ------------------------------------------------------------------
    # Shard cache API
    # ------------------------------------------------------------------

    def _new_in_memory_store(self) -> InMemoryArrowStore:
        return InMemoryArrowStore(
            chunk_size=self._shard_cache_chunk_size,
            action_sequence_keys=self._shard_cache_action_keys,
            fps=self._shard_cache_fps,
            image_transforms=self.image_transforms,
        )

    def append_episode_to_memory(self, path: str | Path, ep_frames: list[dict]) -> None:
        if not ep_frames:
            return
        path = Path(path)
        with self._rolling_access_lock:
            if path in self._in_memory_shards:
                ds_idx = self._sub_datasets.index(path)
                if ds_idx != len(self._sub_datasets) - 1:
                    raise ValueError(
                        "Can only append to the latest in-memory LeRobot shard."
                    )
                store = self._in_memory_shards[path]
            else:
                store = self._new_in_memory_store()
            old_len = len(store)
            episode_index = store.num_episodes

        # Build the expensive HuggingFace episode outside the sampling lock.
        prepared_episode = store._prepare_episode_dataset(
            ep_frames,
            base=old_len,
            episode_index=episode_index,
        )
        if not prepared_episode:
            return

        with self._rolling_access_lock:
            if path in self._in_memory_shards:
                ds_idx = self._sub_datasets.index(path)
                if ds_idx != len(self._sub_datasets) - 1:
                    raise ValueError(
                        "Can only append to the latest in-memory LeRobot shard."
                    )
                store = self._in_memory_shards[path]
                physical_base = self._cumulative_lengths[ds_idx]
            else:
                self._in_memory_shards[path] = store
                self._sub_datasets.append(path)
                physical_base = self._cumulative_lengths[-1]
                self._cumulative_lengths.append(physical_base)

            old_len = int(prepared_episode["base"])
            added_frames = store._commit_prepared_episode(prepared_episode)
            if added_frames <= 0:
                return

            self._cumulative_lengths[-1] += added_frames
            self._total_episodes += 1
            if self.require_all_intervene and self._valid_physical_indices is not None:
                assert self._valid_physical_set is not None
                ep_ds = prepared_episode["dataset"]
                if self.intervene_flag_key not in ep_ds.column_names:
                    logger.warning(
                        "[RollingLeRobotDataset] require_all_intervene=True but "
                        "column %r missing in appended in-memory episode; keeping "
                        "all %d chunk starts for this episode.",
                        self.intervene_flag_key,
                        added_frames,
                    )
                    valid_locals = range(old_len, old_len + added_frames)
                else:
                    flags = _hf_column_to_numpy_bool_1d(ep_ds, self.intervene_flag_key)
                    assert int(flags.shape[0]) == added_frames
                    deltas = np.arange(max(1, int(self.chunk_size)), dtype=np.int64)
                    offsets = np.arange(added_frames, dtype=np.int64)[:, None]
                    raw = offsets + deltas[None, :]
                    in_episode = raw < added_frames
                    step_ok = np.ones_like(in_episode, dtype=np.bool_)
                    step_ok[in_episode] = flags[raw[in_episode]]
                    valid_offsets = np.nonzero(step_ok.all(axis=1))[0]
                    valid_locals = (old_len + int(offset) for offset in valid_offsets)
                for local_i in valid_locals:
                    gidx = physical_base + int(local_i)
                    self._valid_physical_indices.append(gidx)
                    self._valid_physical_set.add(gidx)

            self._update_window_sampling_bounds()
            self._evict_stale_shards()
            logger.debug(
                "[RollingLeRobotDataset] episode appended: %s "
                "(+%d frames, physical_frames=%d, logical_samples=%d)",
                path.name,
                added_frames,
                self._num_physical_frames(),
                len(self),
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
    # Archived shard resume
    # ------------------------------------------------------------------

    @staticmethod
    def archived_shard_info(path: str | Path) -> dict[str, Any] | None:
        """Return basic metadata for a finalized LeRobot shard, or ``None``."""
        path = Path(path)
        info_path = path / "meta" / "info.json"
        episodes_path = path / "meta" / "episodes.jsonl"
        data_dir = path / "data"
        if (
            not info_path.is_file()
            or not episodes_path.is_file()
            or not data_dir.is_dir()
        ):
            return None

        try:
            with info_path.open() as f:
                info = json.load(f)
            episodes = _read_jsonl(episodes_path)
        except (OSError, json.JSONDecodeError):
            return None

        if not episodes:
            return None

        import pyarrow.parquet as pq

        parquet_paths = []
        total_frames = 0
        for episode in episodes:
            try:
                parquet_path = _episode_parquet_path(path, info, episode)
                if not parquet_path.is_file():
                    return None
                total_frames += int(pq.read_metadata(parquet_path).num_rows)
                parquet_paths.append(parquet_path)
            except Exception:  # noqa: BLE001
                return None

        if total_frames <= 0:
            return None

        return {
            "num_episodes": len(parquet_paths),
            "num_frames": total_frames,
        }

    def _load_archived_shard_store(
        self, path: str | Path
    ) -> tuple[Path, InMemoryArrowStore]:
        """Decode a shard and build its in-memory store without indexing it."""
        path, shard_episodes = self._load_archived_shard_episodes(path)
        store = self._new_in_memory_store()
        for ep_frames in shard_episodes:
            store.add_episode(ep_frames)
        return path, store

    def _append_valid_indices_for_store(
        self, store: InMemoryArrowStore, physical_base: int
    ) -> None:
        """Add intervene-filtered indices for one attached in-memory store."""
        if not self.require_all_intervene or self._valid_physical_indices is None:
            return

        assert self._valid_physical_set is not None
        for ep_ds, ep_from, ep_to in zip(
            store._episode_datasets,
            store._ep_from,
            store._ep_to,
            strict=True,
        ):
            if self.intervene_flag_key not in ep_ds.column_names:
                logger.warning(
                    "[RollingLeRobotDataset] require_all_intervene=True but "
                    "column %r missing in appended in-memory episode; keeping "
                    "all %d chunk starts for this episode.",
                    self.intervene_flag_key,
                    int(ep_to) - int(ep_from),
                )
                valid_locals = range(int(ep_from), int(ep_to))
            else:
                flags = _hf_column_to_numpy_bool_1d(ep_ds, self.intervene_flag_key)
                added_frames = int(ep_to) - int(ep_from)
                assert int(flags.shape[0]) == added_frames
                deltas = np.arange(max(1, int(self.chunk_size)), dtype=np.int64)
                offsets = np.arange(added_frames, dtype=np.int64)[:, None]
                raw = offsets + deltas[None, :]
                in_episode = raw < added_frames
                step_ok = np.ones_like(in_episode, dtype=np.bool_)
                step_ok[in_episode] = flags[raw[in_episode]]
                valid_offsets = np.nonzero(step_ok.all(axis=1))[0]
                valid_locals = (int(ep_from) + int(offset) for offset in valid_offsets)

            for local_i in valid_locals:
                gidx = int(physical_base) + int(local_i)
                self._valid_physical_indices.append(gidx)
                self._valid_physical_set.add(gidx)

    def load_archived_shards_staged(
        self, paths: Sequence[str | Path], num_workers: int = 0
    ) -> list[tuple[Path, InMemoryArrowStore]]:
        """Build archived shard stores without exposing them to sampling."""
        paths = [Path(path) for path in paths]
        if not paths:
            return []

        num_workers = int(num_workers)
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative.")

        if num_workers == 0:
            return [self._load_archived_shard_store(path) for path in paths]

        max_workers = min(num_workers, len(paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._load_archived_shard_store, paths))

    def publish_staged_resume_shards(
        self, staged_stores: Sequence[tuple[Path, InMemoryArrowStore]]
    ) -> tuple[int, int]:
        """Atomically expose fully loaded resume shards before live online data."""
        if not staged_stores:
            return 0, 0

        staged_items = [(Path(path), store) for path, store in staged_stores]
        total_episodes = sum(int(store.num_episodes) for _, store in staged_items)
        total_frames = sum(int(len(store)) for _, store in staged_items)
        if total_frames <= 0:
            return 0, 0

        with self._rolling_access_lock:
            existing_items = [
                (path, self._in_memory_shards[path])
                for path in self._sub_datasets
                if path in self._in_memory_shards
            ]
            existing_paths = {path for path, _ in existing_items}
            duplicate_paths = [
                path for path, _ in staged_items if path in existing_paths
            ]
            if duplicate_paths:
                raise ValueError(
                    f"Archived LeRobot shard already loaded: {duplicate_paths[0]}"
                )

            ordered_items = [*staged_items, *existing_items]
            self._sub_datasets = []
            self._in_memory_shards = {}
            self._cumulative_lengths = [0]
            self._total_episodes = 0
            if self.require_all_intervene:
                self._valid_physical_indices = []
                self._valid_physical_set = set()

            for path, store in ordered_items:
                num_frames = int(len(store))
                if num_frames <= 0:
                    continue

                physical_base = self._cumulative_lengths[-1]
                self._sub_datasets.append(path)
                self._in_memory_shards[path] = store
                self._cumulative_lengths.append(physical_base + num_frames)
                self._total_episodes += int(store.num_episodes)
                self._append_valid_indices_for_store(store, physical_base)

            self._update_window_sampling_bounds()
            self._evict_stale_shards()

        return total_episodes, total_frames

    @staticmethod
    def _load_archived_shard_episodes(
        path: str | Path,
    ) -> tuple[Path, list[list[dict[str, Any]]]]:
        """Decode archived shard episodes without mutating dataset indexes."""
        path = Path(path)
        info_path = path / "meta" / "info.json"
        episodes_path = path / "meta" / "episodes.jsonl"
        tasks = _read_lerobot_tasks(path / "meta" / "tasks.jsonl")

        with info_path.open() as f:
            info = json.load(f)
        episodes = _read_jsonl(episodes_path)
        feature_specs = info.get("features", {})

        shard_episodes = []
        for episode in episodes:
            parquet_path = _episode_parquet_path(path, info, episode)
            ep_frames = _load_lerobot_episode_frames(
                parquet_path=parquet_path,
                shard_path=path,
                feature_specs=feature_specs,
                task_by_index=tasks,
            )
            if not ep_frames:
                continue
            shard_episodes.append(ep_frames)

        return path, shard_episodes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        with self._rolling_access_lock:
            physical = self._logical_to_physical(int(idx))
            return self._load_item_from_lerobot(physical)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Batch fetch for DataLoader (one call per batch when supported)."""
        if not indices:
            return []
        with self._rolling_access_lock:
            physicals = [self._logical_to_physical(int(i)) for i in indices]
            return [self._load_item_from_lerobot(p) for p in physicals]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_rolling_lerobot_dataset(
    root_dir: str | Path,
    chunk_size: int = 1,
    delta_timestamps: dict[str, list[float]] | None = None,
    keys: list[str] | None = None,
    image_transforms: Callable | None = None,
    min_frames: int = 1,
    wait_interval_s: float = 10.0,
    action_sequence_keys: list[str] | None = ["actions"],
    require_all_intervene: bool = False,
    intervene_flag_key: str = "intervene_flag",
    window_size: int | None = None,
    in_memory_mode: bool = False,
    fps: int = 10,
) -> RollingLeRobotDataset:
    """Build a :class:`RollingLeRobotDataset` for rolling data collection.

    Args:
        root_dir: Root directory containing ``rank_N/id_M/`` sub-datasets.
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
        require_all_intervene: See :class:`RollingLeRobotDataset`.
        intervene_flag_key: Column name for the per-frame intervention flag.
        window_size: Caps the dataset to the last ``window_size`` **logical**
            frames.  See :class:`RollingLeRobotDataset`.
        in_memory_mode: Must be ``True`` for rolling in-memory shard loading.
        fps: Dataset frame rate used when auto-generating ``delta_timestamps``.

    Returns:
        A :class:`RollingLeRobotDataset` instance.
    """
    dataset = RollingLeRobotDataset(
        root_dir=root_dir,
        chunk_size=chunk_size,
        delta_timestamps=delta_timestamps,
        keys=keys,
        image_transforms=image_transforms,
        min_frames=min_frames,
        wait_interval_s=wait_interval_s,
        action_sequence_keys=action_sequence_keys,
        require_all_intervene=require_all_intervene,
        intervene_flag_key=intervene_flag_key,
        window_size=window_size,
        in_memory_mode=in_memory_mode,
        fps=fps,
    )

    logger.info(
        "[build_rolling_lerobot_dataset] root_dir=%s, chunk_size=%d, "
        "sub_datasets=%d, logical_samples=%d, "
        "physical_frames=%d, require_all_intervene=%s, window_size=%s",
        root_dir,
        chunk_size,
        len(dataset._sub_datasets),
        len(dataset),
        dataset._num_physical_frames(),
        require_all_intervene,
        window_size,
    )

    return dataset
