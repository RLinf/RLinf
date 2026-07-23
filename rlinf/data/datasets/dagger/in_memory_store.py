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

"""In-memory Arrow store for online DAgger LeRobot shards."""

from __future__ import annotations

import bisect
from collections import deque
from typing import Any, Callable

import numpy as np
import torch


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
