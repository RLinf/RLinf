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

"""Pair dataset + collator for ARM + ReWiND binary value learning.

Dataset contract (``__getitem__``):

    {
        "image_t":  {cam_name: np.ndarray[H, W, 3] uint8, ...},  # frame at t
        "image_tk": {cam_name: np.ndarray[H, W, 3] uint8, ...},  # frame at t+k
        "image_mask_t":  {cam_name: bool, ...},
        "image_mask_tk": {cam_name: bool, ...},
        "prompt": str,
        "label": int,                      # long bin index in [0, num_bins); binary: 0 = regress, 1 = progress
        "episode": int,
        "frame_idx_t": int,
        "frame_idx_tk": int,
    }

Each ``cam_name`` is a camera **view** (e.g. ``"base_0_rgb"``,
``"left_wrist_0_rgb"``). The time axis — frame_t vs frame_{t+k} — is a
separate structural axis: the collator runs the
``SteamProcessor`` once for frame_t and once for frame_{t+k}, then
stacks the two per-camera image tensors along a new ``num_frames`` dim so
the backbone receives a ``[B, num_cameras, num_frames, 3, H, W]`` tensor
per camera key.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .binning import (
    _scaled_signed_stride_to_bin,
    _signed_stride_to_bin,
)

logger = logging.getLogger(__name__)


def _isolate_hf_datasets_cache_for_process() -> None:
    """Optionally give each actor its own HuggingFace datasets cache directory."""
    if os.environ.get("RLINF_ISOLATE_HF_DATASETS_CACHE", "0").lower() in (
        "0",
        "false",
        "no",
    ):
        return

    if os.environ.get("RLINF_HF_DATASETS_CACHE_ISOLATED"):
        return

    base_cache = os.environ.get("HF_DATASETS_CACHE")
    if not base_cache:
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        base_cache = str(Path(hf_home) / "datasets")

    rank = os.environ.get("RANK", "norank")
    cache_dir = Path(base_cache) / f"rlinf_rank_{rank}_pid_{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ["RLINF_HF_DATASETS_CACHE_ISOLATED"] = "1"

    try:
        import datasets

        datasets.config.HF_DATASETS_CACHE = str(cache_dir)
    except ImportError:
        pass


# Camera-view aliases tried against raw LeRobot sample dicts. Callers pass
# a plain camera key (e.g. ``image``) and the dataset probes the standard
# LeRobot path templates.
_IMAGE_KEY_ALIASES = (
    "{key}",
    "observation/{key}",
    "observation.{key}",
    "observation.images.{key}",
    "observation/images/{key}",
)


def _resolve_alias(sample: dict, key: str, aliases: Sequence[str]) -> Any:
    """Return ``sample[alias]`` for the first alias template that matches."""
    for template in aliases:
        resolved = template.format(key=key)
        if resolved in sample:
            return sample[resolved]
    raise KeyError(
        f"Could not resolve key={key!r} in sample. Tried: "
        f"{[t.format(key=key) for t in aliases]}. "
        f"Available: {sorted(sample.keys())}"
    )


def _scalar_item(value: Any) -> Any:
    """Return a Python scalar from tensor/array/list scalar-like values."""
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar-like list/tuple, got {value!r}")
        return _scalar_item(value[0])
    return value


def _to_uint8_hwc(frame: Any) -> np.ndarray:
    """Normalise a frame to ``(H, W, 3)`` uint8 numpy."""
    if hasattr(frame, "convert"):  # PIL.Image duck-typed
        frame = np.asarray(frame)
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)

    if arr.ndim != 3:
        raise ValueError(f"expected a rank-3 frame, got shape={arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if float(arr.max()) <= 1.5:
            arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr.astype(np.uint8)


# ---------------------------------------------------------------------------
# Trajectory sources
# ---------------------------------------------------------------------------


class TrajectorySource:
    """Minimal interface that any concrete trajectory source must satisfy."""

    def num_episodes(self) -> int:
        raise NotImplementedError

    def episode_length(self, episode: int) -> int:
        raise NotImplementedError

    def get_view(
        self, episode: int, frame: int, camera_key: str
    ) -> Optional[np.ndarray]:
        """Return a ``(H, W, 3)`` uint8 frame, or ``None`` if the camera is absent."""
        raise NotImplementedError

    def get_view_from_sample(
        self,
        sample: dict,
        camera_key: str,
    ) -> Optional[np.ndarray]:
        """Return a view from an already-loaded raw sample."""
        del sample, camera_key
        raise NotImplementedError

    def get_raw_sample(self, episode: int, frame: int) -> dict:
        """Return the raw source sample for transform pipelines."""
        raise NotImplementedError

    def get_raw_pair(
        self,
        episode: int,
        frame_t: int,
        frame_tk: int,
        *,
        camera_keys: Sequence[str],
        video_transform: Optional[Callable[[torch.Tensor], Any]] = None,
    ) -> tuple[dict, dict]:
        """Return two raw samples, optionally optimized as one pair read."""
        del camera_keys, video_transform
        return (
            self.get_raw_sample(episode, frame_t),
            self.get_raw_sample(episode, frame_tk),
        )

    def get_prompt(self, episode: int, frame: int) -> Optional[str]:
        """Return the task / language instruction for a given frame.

        Implementations return ``None`` if no per-sample instruction is
        available; the caller is expected to fall back to a default.
        """
        return None

    def get_prompt_from_sample(
        self,
        sample: dict,
        episode: int,
        frame: int,
    ) -> Optional[str]:
        """Return prompt from an already-loaded raw sample."""
        del sample
        return self.get_prompt(episode, frame)

    def episode_is_success(self, episode: int) -> bool:
        """Return whether the episode should be treated as successful."""
        raise NotImplementedError


class _LeRobotSource(TrajectorySource):
    """LeRobot-backed source with lazy per-frame access."""

    def __init__(
        self,
        dataset_path: str,
        *,
        only_success: bool = True,
        dataset_type: str,
    ) -> None:
        _isolate_hf_datasets_cache_for_process()
        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )

        from rlinf.data.datasets.recap.utils import (
            decode_image_struct_batch,
            load_task_descriptions,
        )

        local_path = Path(dataset_path).absolute()
        self._dataset_label = str(local_path)
        self.meta = LeRobotDatasetMetadata(local_path.name, root=local_path)
        self.base = LeRobotDataset(
            local_path.name, root=local_path, download_videos=False
        )
        self._only_success = bool(only_success)
        self.dataset_type = dataset_type

        eps = self.base.episode_data_index
        self._ep_starts = [int(x) for x in eps["from"].tolist()]
        self._ep_ends = [int(x) for x in eps["to"].tolist()]
        if self.dataset_type == "sft":
            self._episode_success = None
        else:
            self._episode_success = (
                self._scan_episode_successes() if only_success else None
            )

        self.base.hf_dataset.set_transform(decode_image_struct_batch)

        self._tasks: dict[int, str] = load_task_descriptions(local_path)

    def num_episodes(self) -> int:
        return len(self._ep_starts)

    def episode_length(self, episode: int) -> int:
        return self._ep_ends[episode] - self._ep_starts[episode]

    def _sample(self, episode: int, frame: int) -> dict:
        global_idx = self._ep_starts[episode] + int(frame)
        return self.base[global_idx]

    def _metadata_sample(self, episode: int, frame: int) -> dict:
        """Read non-video per-frame metadata from the HF table."""
        global_idx = self._ep_starts[episode] + int(frame)
        raw_dataset = self.base.hf_dataset
        sample = {
            key: raw_dataset.data.column(key)[global_idx].as_py()
            for key in raw_dataset.column_names
        }
        task_idx = sample.get("task_index")
        if task_idx is not None:
            sample["task"] = self.meta.tasks[int(_scalar_item(task_idx))]
        return sample

    @staticmethod
    def _coerce_success_flag(raw: Any) -> bool:
        """Normalise a raw per-frame success flag to Python bool."""
        return bool(_scalar_item(raw))

    def _scan_episode_successes(self) -> list[bool]:
        """Read one representative frame row per episode from ``is_success``."""
        raw_dataset = self.base.hf_dataset
        if "is_success" not in raw_dataset.column_names:
            raise ValueError(
                "PairDataset(dataset_type='rollout', only_success=True) "
                "requires the LeRobot dataset to contain an 'is_success' "
                "column on representative frame rows."
            )

        # Read is_success directly from the Arrow table, bypassing
        # hf_dataset.set_transform. The transform loads every column and runs
        # hf_transform_to_torch, which crashes on non-image dict columns
        # (e.g. lerobot Video struct dicts that lack a "bytes" key).
        is_success_column = raw_dataset.data.column("is_success")

        episode_success: list[bool] = [
            self._coerce_success_flag(is_success_column[int(start)].as_py())
            for start, _end in zip(self._ep_starts, self._ep_ends)
        ]

        num_success = sum(bool(v) for v in episode_success)
        logger.info(
            "Scanned %d episode(s) in %s via one frame per episode from is_success; "
            "%d marked successful",
            len(episode_success),
            self._dataset_label,
            num_success,
        )
        return episode_success

    def get_view(
        self, episode: int, frame: int, camera_key: str
    ) -> Optional[np.ndarray]:
        sample = self._sample(episode, frame)
        return self.get_view_from_sample(sample, camera_key)

    def get_view_from_sample(
        self,
        sample: dict,
        camera_key: str,
    ) -> Optional[np.ndarray]:
        try:
            raw = _resolve_alias(sample, camera_key, _IMAGE_KEY_ALIASES)
        except KeyError:
            return None
        return _to_uint8_hwc(raw)

    def get_raw_sample(self, episode: int, frame: int) -> dict:
        return self._sample(episode, frame)

    def get_raw_pair(
        self,
        episode: int,
        frame_t: int,
        frame_tk: int,
        *,
        camera_keys: Sequence[str],
        video_transform: Optional[Callable[[torch.Tensor], Any]] = None,
    ) -> tuple[dict, dict]:
        """Read a frame pair while decoding each camera video only once.

        ``LeRobotDataset.__getitem__`` decodes every video camera for a
        single timestamp. For pair training, calling it twice opens/seeks
        each camera video twice. This path reads metadata from Arrow and
        queries ``[t, tk]`` timestamps together per camera, which keeps the
        sample schema identical while reducing random video decode overhead.
        """
        # 一次性读取同一个 episode 里的两帧 frame_t 和 frame_tk，并尽量优化视频解码，避免为每个时间点重复打开/seek 视频。
        raw_t = self._metadata_sample(episode, frame_t)
        raw_tk = self._metadata_sample(episode, frame_tk)

        video_keys = set(getattr(self.meta, "video_keys", []))
        camera_keys = tuple(camera_keys)
        requested_video_keys = [key for key in camera_keys if key in video_keys]
        if len(requested_video_keys) != len(camera_keys):
            return (
                self.get_raw_sample(episode, frame_t),
                self.get_raw_sample(episode, frame_tk),
            )
        if not requested_video_keys:
            return raw_t, raw_tk

        # 取出两帧对应的时间戳。LeRobot 的视频读取通常不是按 frame index 直接读，而是按 episode + timestamp 查询视频帧。
        timestamps = [
            float(_scalar_item(raw_t["timestamp"])),
            float(_scalar_item(raw_tk["timestamp"])),
        ]
        ep_idx = int(_scalar_item(raw_t.get("episode_index", episode)))
        for camera_key in requested_video_keys:
            # 一次查询这个 camera 的两个时间戳：
            frames = self.base._query_videos({camera_key: timestamps}, ep_idx)[
                camera_key
            ]
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            frame_t_tensor = frames[0]
            frame_tk_tensor = frames[1]
            raw_t[camera_key] = (
                video_transform(frame_t_tensor)
                if video_transform is not None
                else frame_t_tensor
            )
            raw_tk[camera_key] = (
                video_transform(frame_tk_tensor)
                if video_transform is not None
                else frame_tk_tensor
            )

        return raw_t, raw_tk

    def get_prompt(self, episode: int, frame: int) -> str:
        sample = self._sample(episode, frame)
        return self.get_prompt_from_sample(sample, episode, frame)

    def get_prompt_from_sample(
        self,
        sample: dict,
        episode: int,
        frame: int,
    ) -> str:
        task = sample.get("task")
        if isinstance(task, str) and task:
            return task
        ti = sample.get("task_index")
        if ti is None:
            raise RuntimeError(
                f"PairDataset: sample for episode={episode} frame={frame} in "
                f"{self._dataset_label!r} has no 'task' string and no "
                "'task_index' field; cannot resolve per-episode task instruction."
            )
        ti_int = ti.item() if isinstance(ti, torch.Tensor) else int(ti)
        if not self._tasks:
            raise RuntimeError(
                f"PairDataset: sample for episode={episode} frame={frame} in "
                f"{self._dataset_label!r} has task_index={ti_int} but the dataset "
                "has no meta/tasks.jsonl (or meta/tasks.parquet) to resolve the "
                "instruction."
            )
        prompt = self._tasks.get(int(ti_int))
        if not prompt:
            raise RuntimeError(
                f"PairDataset: episode={episode} frame={frame} in "
                f"{self._dataset_label!r} has task_index={ti_int} but it is not "
                f"present in meta/tasks.jsonl "
                f"(available indices: {sorted(self._tasks.keys())})."
            )
        return prompt

    def episode_is_success(self, episode: int) -> bool:
        if self.dataset_type == "sft":
            return True
        if self._episode_success is None:
            return True
        return bool(self._episode_success[episode])


# ---------------------------------------------------------------------------
# Shared pair-dataset base
# ---------------------------------------------------------------------------


class _BasePairDataset(Dataset):
    """Shared trajectory-pair indexing for the train and inference datasets.

    Holds the logic common to :class:`PairDataset` (training) and
    :class:`BinaryPairInferenceDataset` (inference): the flat-index →
    ``(episode, t, t+k)`` anchor mapping over eligible episodes, and the
    per-frame view / prompt loading. Subclasses are expected to set
    ``self._source`` (a :class:`TrajectorySource`), ``self._eligible`` (the
    trained/scored episode ids), ``self.camera_keys``, ``self.k`` and
    ``self.source_name``, then
    call :meth:`_init_anchor_index`. An optional ``self.prompt`` attribute, if
    present and non-``None``, is used as the per-sample instruction fallback.
    """

    @property
    def source(self) -> TrajectorySource:
        return self._source

    @property
    def eligible_episodes(self) -> list[int]:
        return list(self._eligible)

    @property
    def num_pair_positions(self) -> int:
        """Number of distinct ``(episode, t)`` anchors before label duplication."""
        return self._num_pair_positions

    def _init_anchor_index(self) -> None:
        """Build the cumulative anchor index over eligible episodes.

        Each eligible episode contributes ``T_ep - 1`` temporal anchors
        ``t ∈ [0, T_ep - 1)``; the cumulative sum lets ``__getitem__`` map a
        flat index to ``(eligible-slot, t)`` in ``O(log |eligible|)`` via
        :func:`numpy.searchsorted`.
        """
        # 把多个 episode 里的时间位置 t 展平成一个连续的一维索引空间，方便 __getitem__(idx) 用一个整数 idx 找到对应的 episode 和帧位置
        # 每个episode包含T-1个 anchor
        pair_positions_per_episode = np.array(
            [self._source.episode_length(ep) - 1 for ep in self._eligible],
            dtype=np.int64,
        )
        # 计算每个 episode 的 累计结束位置（区分不同episode的边界）。
        self._pair_position_ends = np.cumsum(pair_positions_per_episode)
        # 总锚点个数，*2即为总数据对的个数（正向&反向）
        self._num_pair_positions = int(self._pair_position_ends[-1])

    def _resolve_pair_position(self, pair_position: int) -> tuple[int, int, int]:
        """Map a pair-position index to ``(episode, t, t_plus_k)``.

        ``t_plus_k`` uses the boundary clamp: when ``t + k`` overruns the
        episode the second slot collapses to the last frame ``T - 1`` (stride
        degrades to ``T - 1 - t < k``).
        """
        # 是 _init_anchor_index() 的反向映射：把一维的 pair_position 还原成具体的 (episode, t, t_plus_k)。
        if pair_position < 0 or pair_position >= self._num_pair_positions:
            raise IndexError(pair_position)
        # 判断属于第几个episode，不是原始 episode id，而是 _eligible 列表里的位置。
        episode_slot = int(
            np.searchsorted(self._pair_position_ends, pair_position, side="right")
        )
        # 拿到当前 episode 在 flattened 空间里的起始偏移。
        prev_episode_end = (
            int(self._pair_position_ends[episode_slot - 1]) if episode_slot > 0 else 0
        )
        episode = int(self._eligible[episode_slot])
        # 当前 episode 的局部时间位置就是：
        t = int(pair_position - prev_episode_end)
        # 下一帧的episode内局部位置
        t_plus_k = min(t + self.k, self._source.episode_length(episode) - 1)
        return episode, t, t_plus_k

    def _load_views(
        self,
        episode: int,
        frame_idx: int,
        *,
        sample: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
        views: dict[str, np.ndarray] = {}
        masks: dict[str, bool] = {}
        for camera_key in self.camera_keys:
            view = (
                self._source.get_view_from_sample(sample, camera_key)
                if sample is not None
                else self._source.get_view(episode, frame_idx, camera_key)
            )
            if view is None:
                masks[camera_key] = False
            else:
                views[camera_key] = view
                masks[camera_key] = True
        return views, masks

    def _resolve_prompt(self, episode: int, frame_idx: int) -> str:
        """Return the per-sample task instruction; raises if missing."""
        prompt = self._source.get_prompt(episode, frame_idx)
        return self._prompt_or_fallback(prompt, episode, frame_idx)

    def _resolve_prompt_from_sample(
        self,
        sample: dict,
        episode: int,
        frame_idx: int,
    ) -> str:
        """Return the per-sample task instruction from an already-loaded sample."""
        # 整体作用可以概括为：
        # 从已加载 sample 中拿任务语言指令
        #     -> 优先使用 sample["task"]
        #     -> 否则用 sample["task_index"] 查任务表
        #     -> 如果仍然没有，就使用 Dataset 的 fallback prompt
        #     -> 如果没有 fallback，则报错

        # 它主要是为了两个目的：
        # 1. 保证每个训练样本都有 prompt
        # 2. 避免为了拿 prompt 重复读取同一帧 sample
        
        prompt = self._source.get_prompt_from_sample(sample, episode, frame_idx)
        return self._prompt_or_fallback(prompt, episode, frame_idx)

    def _prompt_or_fallback(
        self,
        prompt: Optional[str],
        episode: int,
        frame_idx: int,
    ) -> str:
        if prompt:
            return prompt
        fallback = getattr(self, "prompt", None)
        if fallback is None:
            raise RuntimeError(
                f"No per-episode task instruction for episode={episode} "
                f"frame={frame_idx} in {self.source_name!r} and no fallback "
                "prompt was provided."
            )
        return fallback


# ---------------------------------------------------------------------------
# Pair dataset
# ---------------------------------------------------------------------------


class PairDataset(_BasePairDataset):
    """Yields ``(frame_t, frame_{t+k})`` pairs with multi-view per frame.

    Args:
        dataset_path: LeRobot dataset path.
        camera_keys: Camera view names to load per frame. These match the
            processor's ``image_keys`` — the collator feeds images under
            exactly these keys, and the processor fills any missing ones
            with zero placeholders (mask=False). Default:
            ``("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")``.
        k: Forward pair stride.
        dataset_type: Must be explicitly provided and be either ``"sft"``
            or ``"rollout"``. ``sft`` datasets are treated as all-success
            episodes, so they do not require an ``is_success`` column.
        only_success: Must be explicitly provided. When ``True``, keeps only
            episodes whose per-frame ``is_success`` column marks the episode
            as successful (for LeRobot datasets this checks one representative
            frame row per episode rather than relying on episode-level
            metadata files). When ``False``, every episode is used regardless
            of outcome and no ``is_success`` column is required. ``sft``
            datasets are always all-success, so this only affects
            ``rollout`` datasets.
        min_episode_length: Optional override for the minimum-length
            floor (default ``k + 1``).
        length_scale_enabled: If ``True`` (multi-bin modes only), the signed
            stride is length-normalized before binning:
            ``scaled = signed_stride * L_max / L_ep``, where ``L_ep`` is the
            current episode length and ``L_max`` is the reference length. Binning
            stays on the ``±K`` axis (bin width ``2K/num_bins``, same resolution
            as the unscaled path); ``L_max`` only sets the scale factor, and
            ``|scaled| > K`` saturates into the extreme bin. An episode of length
            ``L_max`` reproduces the unscaled layout (``scale == 1``); shorter
            episodes push a fixed frame stride into a higher bin (and saturate
            earlier), longer episodes under-reach. No-op in binary mode
            (``num_bins == 2``), where positive scaling preserves the stride sign.
        length_scale_percentile: Percentile of eligible-episode lengths used as
            the reference length ``L_max`` when no explicit
            ``length_scale_reference`` is supplied (default ``90``; ``100`` ⇒
            true max). Acts as the pivot length: episodes near ``L_max`` use the
            full bin range, shorter ones saturate, longer ones under-reach — so
            a lower percentile (e.g. the median) saturates fewer episodes.
        length_scale_reference: Explicit ``L_max``. When ``None`` and
            ``length_scale_enabled``, it is computed per-dataset from
            ``length_scale_percentile``; callers wanting one global ``L_max``
            across a mixture inject it via :meth:`set_length_scale_reference`.
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        camera_keys: Sequence[str] = (
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        ),
        k: int = 4,
        dataset_type: Optional[str] = None,
        only_success: Optional[bool] = None,
        min_episode_length: Optional[int] = None,
        num_bins: int = 2,
        length_scale_enabled: bool = False,
        length_scale_percentile: float = 90.0,
        length_scale_reference: Optional[float] = None,
    ) -> None:
        self.camera_keys: tuple[str, ...] = tuple(camera_keys)
        if not self.camera_keys:
            raise ValueError("camera_keys must be non-empty")
        self.k = int(k)
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        # Mode switch. num_bins == 2 → legacy binary mode: fixed-stride k.
        # num_bins > 2 → multi-bin: sample i uniformly from [1, min(K, T-1-t)]
        # per-anchor at __getitem__ time. Both emit a long bin-index label
        # in [0, num_bins); the bin layout (_signed_stride_to_bin) places
        # regressive bins in [0, num_bins // 2) and progressive bins in
        # [num_bins // 2, num_bins), so binary degenerates to 0 = regress,
        # 1 = progress. 2K must be an integer multiple of num_bins so every
        # bin covers the same number of strides (uniform bin widths).
        self.num_bins = int(num_bins)
        if self.num_bins < 2 or self.num_bins % 2 != 0:
            raise ValueError(f"num_bins must be >= 2 and even, got {self.num_bins}")
        if self.num_bins > 2 and (2 * self.k) % self.num_bins != 0:
            raise ValueError(
                f"For num_bins={self.num_bins} in multi-bin mode, 2*k must be a "
                f"multiple of num_bins; got k={self.k} (2*k={2 * self.k})."
            )
        self.length_scale_enabled = bool(length_scale_enabled)
        self.length_scale_percentile = float(length_scale_percentile)
        # Resolved per-dataset below once eligible episodes are known; a global
        # mixture-wide L_max can override it via set_length_scale_reference.
        self._length_scale_reference: Optional[float] = (
            None if length_scale_reference is None else float(length_scale_reference)
        )
        if self.length_scale_enabled:
            if not (0.0 < self.length_scale_percentile <= 100.0):
                raise ValueError(
                    "length_scale_percentile must be in (0, 100], got "
                    f"{self.length_scale_percentile}"
                )
            if self.num_bins == 2:
                logger.warning(
                    "PairDataset length_scale_enabled has no effect in binary "
                    "mode (num_bins == 2): scaling by L_max / L_ep preserves the "
                    "stride sign, so labels are unchanged. Set num_bins > 2 for "
                    "length-scaled multi-bin labels."
                )
        self._rng: np.random.Generator | None = None
        self.source_name = str(dataset_path)
        if dataset_type is None:
            raise ValueError(
                "PairDataset requires an explicit dataset_type argument "
                "('sft' or 'rollout')."
            )
        self.dataset_type = str(dataset_type).lower()
        if self.dataset_type not in ("sft", "rollout"):
            raise ValueError(
                f"PairDataset dataset_type must be 'sft' or 'rollout', "
                f"got {dataset_type!r}."
            )
        if only_success is None:
            raise ValueError(
                "PairDataset requires an explicit only_success argument. "
                "Set only_success=true to keep only successful episodes, or "
                "only_success=false to use every episode (success and failure)."
            )
        self.only_success = bool(only_success)

        self._source = _LeRobotSource(
            dataset_path,
            only_success=self.only_success,
            dataset_type=self.dataset_type,
        )

        # Default floor: any episode with at least 2 frames can form a pair
        # (t=0, t+k clamped to T-1). Yamls can raise this if they want to
        # exclude episodes that can't supply at least one full stride-k pair.
        if min_episode_length is None:
            min_episode_length = 2
        self._min_episode_length = int(min_episode_length)
        total_eps = self._source.num_episodes()
        # 筛选可用episode（最少2帧，满足only_success条件）
        self._eligible = [
            ep
            for ep in range(total_eps)
            if self._source.episode_length(ep) >= self._min_episode_length
            and (not self.only_success or self._source.episode_is_success(ep))
        ]
        if not self._eligible:
            raise ValueError(
                f"No eligible episodes found with length >= {self._min_episode_length} "
                f"and only_success={self.only_success} "
                f"(dataset has {total_eps} episodes)."
            )

        # Enumerate temporal anchors t ∈ [0, T_ep - 1) per eligible episode.
        # For t in [0, T_ep - k) the pair is the regular (t, t+k); near the end
        # the second slot is clamped to T_ep - 1 (boundary pair, stride < k).
        # Each anchor is later duplicated into a positive and a negative sample.
        self._init_anchor_index()

        # Resolve the per-dataset length-scale reference (L_max) as the
        # configured percentile of eligible-episode lengths, unless an explicit
        # (e.g. global mixture-wide) reference was supplied to the constructor.
        if self.length_scale_enabled and self._length_scale_reference is None:
            self._length_scale_reference = self._compute_length_scale_reference()

        logger.info(
            "PairDataset: dataset_path=%s, episodes=%d eligible=%d, k=%d, "
            "num_bins=%d (%s mode), total_positions=%d, "
            "dataset_type=%s, only_success=%s, camera_keys=%s",
            self.source_name,
            total_eps,
            len(self._eligible),
            self.k,
            self.num_bins,
            "binary" if self.num_bins == 2 else "multi-bin",
            self._num_pair_positions,
            self.dataset_type,
            self.only_success,
            self.camera_keys,
        )

    def set_epoch(self, epoch: int) -> None:
        del epoch  # no RNG state, retained for DataLoader wrapper compat

    @property
    def length_scale_reference(self) -> Optional[float]:
        """The resolved ``L_max`` used to length-scale strides (``None`` if off)."""
        return self._length_scale_reference

    def eligible_episode_lengths(self) -> list[int]:
        """Return the length of every eligible (trained-on) episode."""
        return [int(self._source.episode_length(ep)) for ep in self._eligible]

    def _compute_length_scale_reference(self) -> float:
        """Percentile of eligible-episode lengths, floored at 1."""
        lengths = self.eligible_episode_lengths()
        ref = float(
            np.percentile(
                np.asarray(lengths, dtype=np.float64),
                self.length_scale_percentile,
            )
        )
        return max(1.0, ref)

    def set_length_scale_reference(self, reference: float) -> None:
        """Override ``L_max`` (e.g. a global mixture-wide value).

        No-op when length scaling is disabled, so callers can apply it
        unconditionally across a heterogeneous mixture.
        """
        if not self.length_scale_enabled:
            return
        if reference <= 0:
            raise ValueError(f"length_scale_reference must be > 0, got {reference}")
        self._length_scale_reference = float(reference)

    @staticmethod
    def compute_global_length_scale_reference(
        datasets: Sequence["PairDataset"],
        percentile: float,
    ) -> float:
        """Pool eligible-episode lengths across datasets and return the percentile.

        Used to derive a single mixture-wide ``L_max`` so a fixed frame stride
        maps to the same bin regardless of which dataset the episode came from.
        """
        lengths: list[int] = []
        for ds in datasets:
            lengths.extend(ds.eligible_episode_lengths())
        if not lengths:
            raise ValueError(
                "compute_global_length_scale_reference got no episodes to pool."
            )
        ref = float(
            np.percentile(np.asarray(lengths, dtype=np.float64), float(percentile))
        )
        return max(1.0, ref)

    def __len__(self) -> int:
        # Each temporal anchor contributes two labeled samples:
        #   positive: (t, t+k)
        #   negative: (t+k, t)，直接颠倒两帧即可
        return 2 * self._num_pair_positions

    def _decode_sample_index(self, idx: int) -> tuple[int, bool]:
        """Map a flat dataset index to ``(pair_position, is_positive)``."""
        if idx < 0:
            idx += len(self)
        if not (0 <= idx < len(self)):
            raise IndexError(idx)

        pair_position = idx // 2
        is_positive = (idx % 2) == 0
        return pair_position, is_positive

    def _rng_for_worker(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def _build_sample(
        self,
        *,
        episode: int,
        frame_idx_t: int,
        frame_idx_tk: int,
        prompt: str,
        label,
        raw_t: Optional[dict] = None,
        raw_tk: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Assemble the sample dict for a single labeled frame pair.

        ``label`` is a Python ``int`` bin index in ``[0, num_bins)``. The
        collator casts the batched column to ``torch.long``. Binary mode
        (``num_bins == 2``) degenerates to ``0 = regress``, ``1 =
        progress``, matching the multi-bin layout from
        :func:`_signed_stride_to_bin`.
        """
        views_t, mask_t = self._load_views(episode, frame_idx_t, sample=raw_t)
        views_tk, mask_tk = self._load_views(episode, frame_idx_tk, sample=raw_tk)

        sample: dict[str, Any] = {
            "image_t": views_t,
            "image_tk": views_tk,
            "image_mask_t": mask_t,
            "image_mask_tk": mask_tk,
            "prompt": prompt,
            "label": label,
            "episode": int(episode),
            "frame_idx_t": int(frame_idx_t),
            "frame_idx_tk": int(frame_idx_tk),
            "source_name": self.source_name,
        }

        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # 靠 idx 的奇偶区分两个不同顺序的样本；当于原本的正向样本【0，1，2】，通过*2后全部未偶数，就留出了奇数作为负向样本
        pair_position, is_positive = self._decode_sample_index(idx)
        episode, t, t_plus_k_binary = self._resolve_pair_position(pair_position)

        if self.num_bins == 2:
            # Binary path: fixed stride k with the existing boundary
            # clamp (t+k may degrade to T-1 near episode end). Labels are
            # long bin indices matching the multi-bin layout — 1 for
            # progress (positive stride), 0 for regress (negative stride).
            # 仅用0/1二值化标签作为每个数据对的标签
            if is_positive:
                frame_idx_t, frame_idx_tk = t, t_plus_k_binary
                label: Any = 1
            else:
                frame_idx_t, frame_idx_tk = t_plus_k_binary, t
                label = 0
        else:
            # Multi-bin path: sample i uniformly from [1, min(K, T-1-t)]
            # at getitem time so every anchor gets exposure to every
            # valid stride over enough epochs. No boundary clamp — the
            # emitted bin always matches the true stride.
            episode_length = self._source.episode_length(episode)
            max_valid_stride = min(self.k, episode_length - 1 - t)
            if max_valid_stride < 1:
                # Should not happen: _pair_position_ends enumerates only
                # t ≤ T-2, so episode_length - 1 - t ≥ 1. Fail-loud per
                # 78bc04dd rather than silently handle.
                raise RuntimeError(
                    f"PairDataset: no valid stride for episode={episode} "
                    f"t={t} episode_length={episode_length} (bug in anchor "
                    "enumeration)."
                )
            # 表示这一对帧的时间方向和时间距离，用来生成分类标签 label， |i|代表时间距离，正负号代表时间方向
            i = int(self._rng_for_worker().integers(low=1, high=max_valid_stride + 1))
            if is_positive:
                frame_idx_t, frame_idx_tk = t, t + i
                signed_stride = i
            else:
                frame_idx_t, frame_idx_tk = t + i, t
                signed_stride = -i
            if self.length_scale_enabled and self._length_scale_reference is not None:
                # Length-normalize the stride so a fixed frame jump maps to a
                # higher progress bin in shorter episodes:
                #   scaled = signed_stride * L_max / L_ep
                # L_max only sets the scale factor; binning stays on the ±K axis
                # (bin width 2K/num_bins, same resolution as the unscaled path),
                # with |scaled| > K saturating into the extreme bin. An episode
                # of length L_max reproduces the unscaled layout (scale == 1).
                scale = max(1.0, self._length_scale_reference / float(episode_length))
                label = _scaled_signed_stride_to_bin(
                    signed_stride * scale, self.k, self.num_bins
                )
            else:
                # 构建帧对的标签
                label = _signed_stride_to_bin(signed_stride, self.k, self.num_bins)

        raw_t, raw_tk = self._source.get_raw_pair(
            episode,
            frame_idx_t,
            frame_idx_tk,
            camera_keys=self.camera_keys,
        )
        # Preserve the previous behavior: language is resolved from the
        # anchor frame ``t`` even for reversed/negative pairs. One of the two
        # already-loaded frames is always the anchor, so this avoids an extra
        # source read.
        if frame_idx_t == t:
            prompt_sample = raw_t
        elif frame_idx_tk == t:
            prompt_sample = raw_tk
        else:
            prompt_sample = raw_t
        prompt = self._resolve_prompt_from_sample(prompt_sample, episode, t)

        return self._build_sample(
            episode=episode,
            frame_idx_t=frame_idx_t,
            frame_idx_tk=frame_idx_tk,
            prompt=prompt,
            label=label,
            raw_t=raw_t,
            raw_tk=raw_tk,
        )


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


@dataclass
class BinaryPairDataCollator:
    """Collator that produces the backbone's observation dict for binary pairs.

    Parallel to :class:`~rlinf.models.embodiment.value_model.recap.data_collator.\
ValueDataCollator`. Runs the :class:`SteamProcessor` **twice**
    — once for frame_t's multi-view images, once for frame_{t+k}'s — then
    stacks the per-camera outputs along a new ``num_frames`` axis. The
    backbone receives per-camera tensors of shape
    ``[B, num_frames, 3, H, W]``.

    Attributes:
        processor: :class:`SteamProcessor` with ``image_keys``
            matching the dataset's ``camera_keys``.
        max_length: Token padding length.
        train: If ``True``, the processor's image augmentations fire.
        num_bins: Matches the paired :class:`PairDataset`'s ``num_bins``.
            Used only for validation assertions — labels are always
            emitted as ``torch.long`` bin indices in ``[0, num_bins)``,
            so binary (``num_bins == 2``) and multi-bin (``num_bins >
            2``) share the same tensor dtype.
    """

    processor: Any
    max_length: int = 200
    train: bool = True
    num_bins: int = 2

    def _collect_per_camera(
        self,
        examples: list[dict[str, Any]],
        images_key: str,
        masks_key: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Gather per-camera image tensors at a single timestamp.

        Missing camera entries for a sample turn into zero placeholders so
        the processor always sees a rectangular camera dict; the returned
        mask records which samples actually had the view.
        """
        camera_keys = set()
        for ex in examples:
            camera_keys.update(ex[images_key].keys())
        if not camera_keys:
            return {}, {}

        bsize = len(examples)
        images_out: dict[str, torch.Tensor] = {}
        masks_out: dict[str, torch.Tensor] = {}
        for cam in sorted(camera_keys):
            frames: list[np.ndarray] = []
            mask_vec: list[bool] = []
            shapes: list[tuple[int, ...]] = []
            for ex in examples:
                v = ex[images_key].get(cam)
                if v is None:
                    frames.append(None)  # type: ignore[arg-type]
                    mask_vec.append(False)
                else:
                    frames.append(v)
                    shapes.append(tuple(int(dim) for dim in v.shape))
                    mask_vec.append(bool(ex[masks_key].get(cam, True)))

            unique_shapes = sorted(set(shapes))
            if len(unique_shapes) > 1:
                shape_examples = [
                    {
                        "source": ex.get("source_name", "unknown"),
                        "episode": ex.get("episode"),
                        "frame_idx_t": ex.get("frame_idx_t"),
                        "frame_idx_tk": ex.get("frame_idx_tk"),
                        "shape": None
                        if ex[images_key].get(cam) is None
                        else tuple(int(dim) for dim in ex[images_key][cam].shape),
                    }
                    for ex in examples
                ]
                raise ValueError(
                    "BinaryPairDataCollator saw incompatible raw image shapes "
                    f"for camera={cam!r} at {images_key!r}: {unique_shapes}. "
                    "PairDataset assumes camera tensors are already shape-aligned; "
                    "this usually means your train batch mixed datasets with "
                    "different raw resolutions for the same camera key. "
                    f"Examples: {shape_examples}"
                )

            if unique_shapes:
                h, w = unique_shapes[0][:2]
            else:
                h, w = 1, 1

            # Replace None entries with zero placeholders matching the first
            # real frame's spatial size. If no real frame exists for this
            # camera across the whole batch, fall back to 1x1.
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            stacked = torch.from_numpy(
                np.stack([f if f is not None else placeholder for f in frames])
            )
            images_out[cam] = stacked
            masks_out[cam] = torch.tensor(mask_vec, dtype=torch.bool)

        # Ensure every example yields the same bsize (torch stack invariant).
        for cam, t in images_out.items():
            if t.shape[0] != bsize:
                raise RuntimeError(f"Unexpected batch shape for cam={cam}: {t.shape}")
        return images_out, masks_out

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("BinaryPairDataCollator received an empty batch")

        prompts: list[str] = [ex["prompt"] for ex in examples]

        # Frame-t and frame-tk run through the processor independently so
        # image augmentations are sampled per frame and so the per-camera
        # masks can differ between the two timestamps (e.g. a wrist view
        # that blinks on/off mid-episode).
        images_t, masks_t = self._collect_per_camera(
            examples, "image_t", "image_mask_t"
        )
        images_tk, masks_tk = self._collect_per_camera(
            examples, "image_tk", "image_mask_tk"
        )

        processed_t = self.processor.image_processor(
            images=images_t,
            image_masks=masks_t,
            return_tensors="pt",
            train=self.train,
        )
        processed_tk = self.processor.image_processor(
            images=images_tk,
            image_masks=masks_tk,
            return_tensors="pt",
            train=self.train,
        )

        # After process_images, pixel_values is a dict[cam → [B, 3, H, W]]
        # covering the **processor's** image_keys (missing camera keys get
        # zero-filled with mask=False at that stage). Stacking along a new
        # dim=1 gives [B, num_frames, 3, H, W] per camera.
        pixel_values_t = processed_t["pixel_values"]
        pixel_values_tk = processed_tk["pixel_values"]
        camera_keys_out = sorted(set(pixel_values_t) | set(pixel_values_tk))
        if not camera_keys_out:
            raise RuntimeError(
                "Processor returned no camera views — check image_keys / "
                "camera_keys alignment between dataset and processor."
            )

        def _stack_over_time(cam: str) -> tuple[torch.Tensor, torch.Tensor]:
            v_t = pixel_values_t.get(cam)
            v_tk = pixel_values_tk.get(cam)
            if v_t is None or v_tk is None:
                raise RuntimeError(
                    f"Camera {cam!r} missing from one of the two per-frame "
                    "processor outputs — this indicates inconsistent batch "
                    "shapes; investigate the dataset sample schema."
                )
            img_stacked = torch.stack([v_t, v_tk], dim=1)  # [B, 2, 3, H, W]
            m_t = processed_t["image_masks"][cam]
            m_tk = processed_tk["image_masks"][cam]
            mask_stacked = torch.stack([m_t, m_tk], dim=1).to(torch.bool)
            return img_stacked, mask_stacked

        images_observation: dict[str, torch.Tensor] = {}
        masks_observation: dict[str, torch.Tensor] = {}
        for cam in camera_keys_out:
            img, mask = _stack_over_time(cam)
            images_observation[cam] = img
            masks_observation[cam] = mask

        processed_txt = self.processor.process_text(
            prompts=prompts,
            max_length=self.max_length,
            return_tensors="pt",
        )

        observation = {
            "images": images_observation,
            "image_masks": masks_observation,
            "tokenized_prompt": processed_txt["input_ids"],
            "tokenized_prompt_mask": processed_txt["attention_mask"].bool(),
        }
        episode = torch.tensor(
            [int(ex["episode"]) for ex in examples], dtype=torch.long
        )
        frame_idx_t = torch.tensor(
            [int(ex["frame_idx_t"]) for ex in examples], dtype=torch.long
        )
        frame_idx_tk = torch.tensor(
            [int(ex["frame_idx_tk"]) for ex in examples], dtype=torch.long
        )
        observation["episode"] = episode
        observation["frame_idx_t"] = frame_idx_t
        observation["frame_idx_tk"] = frame_idx_tk

        # Labels are always long bin indices in [0, num_bins). Binary
        # (num_bins == 2) uses 0 = regress, 1 = progress; multi-bin uses
        # the _signed_stride_to_bin layout. Both feed straight into
        # ``F.cross_entropy`` with no further remapping.
        labels = torch.tensor([int(ex["label"]) for ex in examples], dtype=torch.long)
        return {
            "observation": observation,
            "labels": labels,
            "episode": episode,
            "frame_idx_t": frame_idx_t,
            "frame_idx_tk": frame_idx_tk,
        }


class BinaryPairInferenceDataset(_BasePairDataset):
    """Yields one ``(frame_t, frame_{t+k})`` pair per anchor for inference.

    Differences vs :class:`PairDataset`:
        * No success-only filter — every episode contributes pairs (matches
          ``compute_advantages.py`` which scores every frame).
        * Forward direction only: ``image_t = frame_t``, ``image_tk = frame_{t+k}``,
          ``label = 0`` (placeholder so the existing collator works).
        * Boundary clamp identical to PairDataset: when ``t + k > T - 1`` the
          second slot is clamped to ``T - 1``.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        camera_keys: list[str],
        k: int,
        prompt: Optional[str],
        dataset_type: str,
        min_episode_length: Optional[int] = None,
    ) -> None:
        if dataset_type not in ("sft", "rollout"):
            raise ValueError(
                "BinaryPairInferenceDataset.dataset_type must be 'sft' or 'rollout', "
                f"got {dataset_type!r}"
            )
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not camera_keys:
            raise ValueError("camera_keys must be non-empty")

        self.k = int(k)
        self.camera_keys = tuple(camera_keys)
        self.prompt = prompt
        self.dataset_type = dataset_type
        self.source_name = str(dataset_path)

        # Iterate every episode regardless of is_success; _LeRobotSource with
        # only_success=False skips the success scan and treats all episodes
        # as eligible.
        self._source = _LeRobotSource(
            dataset_path,
            only_success=False,
            dataset_type=dataset_type,
        )

        if min_episode_length is None:
            min_episode_length = 2
        self._min_episode_length = int(min_episode_length)

        total_eps = self._source.num_episodes()
        self._eligible: list[int] = [
            ep
            for ep in range(total_eps)
            if self._source.episode_length(ep) >= self._min_episode_length
        ]
        if not self._eligible:
            raise ValueError(
                f"No eligible episodes in {dataset_path!r} with length >= "
                f"{self._min_episode_length} (dataset has {total_eps} episodes)."
            )

        # One anchor per t in [0, T - 2]; total = sum(T_ep - 1).
        self._init_anchor_index()

        logger.info(
            "BinaryPairInferenceDataset: source=%s, episodes=%d, k=%d, "
            "total_anchors=%d, dataset_type=%s, "
            "camera_keys=%s",
            self.source_name,
            len(self._eligible),
            self.k,
            self._num_pair_positions,
            self.dataset_type,
            self.camera_keys,
        )

    def __len__(self) -> int:
        return self._num_pair_positions

    def _build_sample_from_pair(
        self,
        *,
        episode: int,
        t: int,
        t_plus_k: int,
        raw_t: Optional[dict] = None,
        raw_tk: Optional[dict] = None,
    ) -> dict[str, Any]:
        prompt = (
            self._resolve_prompt_from_sample(raw_t, episode, t)
            if raw_t is not None
            else self._resolve_prompt(episode, t)
        )
        views_t, mask_t = self._load_views(episode, t, sample=raw_t)
        views_tk, mask_tk = self._load_views(
            episode,
            t_plus_k,
            sample=raw_tk,
        )
        sample: dict[str, Any] = {
            "image_t": views_t,
            "image_tk": views_tk,
            "image_mask_t": mask_t,
            "image_mask_tk": mask_tk,
            "prompt": prompt,
            "label": 0,  # placeholder; collator emits but inference ignores
            "episode": int(episode),
            "frame_idx_t": int(t),
            "frame_idx_tk": int(t_plus_k),
            "source_name": self.source_name,
        }

        return sample

    def _supports_batched_video_query(self) -> bool:
        # _source is always a _LeRobotSource (has .base, .meta and
        # _metadata_sample); only the lerobot base._query_videos fast path is
        # version-dependent, so that is the sole capability worth probing.
        if not hasattr(self._source.base, "_query_videos"):
            return False
        video_keys = set(getattr(self._source.meta, "video_keys", []))
        return bool(video_keys) and all(cam in video_keys for cam in self.camera_keys)

    def _getitems_batched_video(self, indices: list[int]) -> list[dict[str, Any]]:
        resolved: list[tuple[int, int, int, int]] = []
        by_episode: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for out_idx, idx in enumerate(indices):
            episode, t, t_plus_k = self._resolve_pair_position(int(idx))
            resolved.append((out_idx, episode, t, t_plus_k))
            by_episode[episode].append((out_idx, t, t_plus_k))

        out: list[Optional[dict[str, Any]]] = [None] * len(indices)
        for episode, episode_entries in by_episode.items():
            frame_indices = sorted(
                {
                    frame
                    for _out_idx, t, t_plus_k in episode_entries
                    for frame in (t, t_plus_k)
                }
            )
            metadata_by_frame = {
                frame: self._source._metadata_sample(episode, frame)
                for frame in frame_indices
            }
            timestamps = [
                float(_scalar_item(metadata_by_frame[frame]["timestamp"]))
                for frame in frame_indices
            ]
            ep_idx = int(
                _scalar_item(
                    metadata_by_frame[frame_indices[0]].get("episode_index", episode)
                )
            )

            frames_by_camera = self._source.base._query_videos(
                dict.fromkeys(self.camera_keys, timestamps),
                ep_idx,
            )
            frame_lookup: dict[tuple[str, int], Any] = {}
            for cam in self.camera_keys:
                frames = frames_by_camera[cam]
                if frames.ndim == 3:
                    frames = frames.unsqueeze(0)
                if int(frames.shape[0]) != len(frame_indices):
                    raise RuntimeError(
                        "_query_videos returned an unexpected number of frames for "
                        f"camera={cam!r}: got {int(frames.shape[0])}, expected "
                        f"{len(frame_indices)}"
                    )
                for pos, frame_idx in enumerate(frame_indices):
                    frame_lookup[(cam, frame_idx)] = frames[pos]

            for out_idx, t, t_plus_k in episode_entries:
                raw_t = dict(metadata_by_frame[t])
                raw_tk = dict(metadata_by_frame[t_plus_k])
                for cam in self.camera_keys:
                    raw_t[cam] = frame_lookup[(cam, t)]
                    raw_tk[cam] = frame_lookup[(cam, t_plus_k)]
                out[out_idx] = self._build_sample_from_pair(
                    episode=episode,
                    t=t,
                    t_plus_k=t_plus_k,
                    raw_t=raw_t,
                    raw_tk=raw_tk,
                )

        if any(sample is None for sample in out):
            missing = [i for i, sample in enumerate(out) if sample is None]
            raise RuntimeError(
                f"Batched video loader missed sample positions: {missing}"
            )
        return [sample for sample in out if sample is not None]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        if self._supports_batched_video_query():
            return self._getitems_batched_video([int(idx) for idx in indices])
        return [self[int(idx)] for idx in indices]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode, t, t_plus_k = self._resolve_pair_position(idx)
        raw_t, raw_tk = self._source.get_raw_pair(
            episode,
            t,
            t_plus_k,
            camera_keys=self.camera_keys,
        )
        return self._build_sample_from_pair(
            episode=episode,
            t=t,
            t_plus_k=t_plus_k,
            raw_t=raw_t,
            raw_tk=raw_tk,
        )


__all__ = [
    "BinaryPairDataCollator",
    "BinaryPairInferenceDataset",
    "PairDataset",
    "TrajectorySource",
]
