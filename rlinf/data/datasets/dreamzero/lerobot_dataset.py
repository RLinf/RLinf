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

"""DreamZero LeRobot datasets and mixture sampling for SFT."""

from __future__ import annotations

import bisect
import json
import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from rlinf.data.datasets.dreamzero.sampling_strategy import (
    DEFAULT_VIDEO_IN_CHUNK_OFFSETS,
    EmptyTemporalSampleError,
    MultiAnchorTemporalConfig,
    SamplingMode,
    TemporalIndices,
    build_fixed_window_offsets,
    require_multi_anchor_temporal_indices,
)
from rlinf.data.datasets.dreamzero.utils import (
    collate_ready_sample,
    discover_local_lerobot_episode_indices,
    droid_default_state_action_slices,
    infer_modality_json_from_features,
    infer_named_component_slices,
    load_modality_json,
    load_task_texts,
    probe_video_container_fps,
    safe_lang_text,
)
from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root
from rlinf.data.utils import safe_hash
from rlinf.utils.logging import get_logger

logger = get_logger()


class DreamZeroLeRobotDataset(Dataset):
    """Generic LeRobot-backed DreamZero SFT dataset.

    Loads LeRobot modalities, applies ``ComposedModalityTransform`` per sample
    (typically in DataLoader workers), and returns collate-ready numpy dicts.
    """

    def __init__(
        self,
        data_path: str | list[str],
        video_keys: list[str],
        state_keys: list[str],
        action_keys: list[str],
        language_keys: list[str],
        data_transform: Any,
        lazy_load: bool,
        num_frames: int,
        state_horizon: int,
        action_horizon: int,
        max_chunk_size: int,
        relative_action: bool = False,
        relative_action_keys: list[str] | None = None,
        pq_cache_max_episodes: int = 128,
        video_tolerance_s: float = 0.1,
        video_backend: str = "pyav",
        sampling_mode: SamplingMode = "multi_anchor",
        multi_anchor_resample_attempts: int = 8,
        macro_stride: int | None = None,
        video_in_chunk_offsets: tuple[int, ...] | list[int] | None = None,
    ):
        if isinstance(data_path, (list, tuple)):
            if len(data_path) == 0:
                raise ValueError(
                    "DreamZeroLeRobotDataset requires at least one data path."
                )
            data_path = data_path[0]
        self.data_path = str(data_path)
        self.lazy_load = bool(lazy_load)
        self.sampling_mode: SamplingMode = sampling_mode
        self.state_horizon = int(state_horizon)
        self.action_horizon = int(action_horizon)
        self.max_chunk_size = int(max_chunk_size)
        self.num_frames = int(num_frames)
        self.multi_anchor_resample_attempts = max(
            1, int(multi_anchor_resample_attempts)
        )
        if self.state_horizon <= 0:
            raise ValueError(f"state_horizon must be positive, got {state_horizon!r}")
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon!r}")
        if self.max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size!r}")
        self.relative_action = bool(relative_action)
        self.relative_action_keys = list(relative_action_keys or [])
        self.data_transform = data_transform

        self.video_keys = list(video_keys)
        self.state_keys = list(state_keys)
        self.action_keys = list(action_keys)
        self.language_keys = list(language_keys)
        if not self.video_keys or not self.state_keys or not self.action_keys:
            raise ValueError(
                "DreamZeroLeRobotDataset requires video/state/action modality keys; "
                f"got video={self.video_keys}, state={self.state_keys}, action={self.action_keys}"
            )
        if not self.language_keys:
            raise ValueError(
                "DreamZeroLeRobotDataset requires at least one language key."
            )

        self._root = resolve_lerobot_dataset_root(self.data_path)
        self._meta_dir = self._root / "meta"
        with open(self._meta_dir / "info.json") as f:
            self._info = json.load(f)
        self._fps = float(self._info.get("fps", 10))
        self._version = str(self._info.get("codebase_version", "v3.0"))
        self._tasks = load_task_texts(self._meta_dir)
        self._features = self._info.get("features") or {}
        self._modality_meta = load_modality_json(self._meta_dir)
        if not self._modality_meta:
            self._modality_meta = infer_modality_json_from_features(self._features)
            logger.info(
                "meta/modality.json not found under %s; inferred modality mapping from info.json features.",
                self._meta_dir,
            )
        self._source_video_key = self._build_source_video_key_map()
        self._state_components = self._build_component_sources("state", self.state_keys)
        self._action_components = self._build_component_sources(
            "action", self.action_keys
        )
        self._language_sources = self._build_language_sources()
        self._vector_source_keys = sorted(
            {
                source
                for source, _ in [
                    *self._state_components.values(),
                    *self._action_components.values(),
                ]
            }
        )

        self._use_image_parquet_tree = self._uses_image_parquet_storage()
        self._use_lazy_video_tree = bool(
            self.lazy_load
            and not self._use_image_parquet_tree
            and self._info.get("video_path")
            and self._root.exists()
        )
        self._video_backend = str(video_backend)
        if (
            self.sampling_mode == "multi_anchor"
            and not self.lazy_load
            and not self._use_image_parquet_tree
        ):
            raise ValueError(
                "DreamZeroLeRobotDataset sampling_mode='multi_anchor' requires lazy_load=True "
                "or a v2 image-parquet layout (no mp4 video_path)."
            )
        if self.sampling_mode == "fixed_window":
            self._fixed_window_temporal: TemporalIndices = build_fixed_window_offsets(
                self.num_frames,
                self.state_horizon,
                self.action_horizon,
                self.max_chunk_size,
            )
            self._multi_anchor_cfg = None
        else:
            self._fixed_window_temporal = None
            if macro_stride is None:
                resolved_macro_stride = self.action_horizon
            else:
                resolved_macro_stride = int(macro_stride)
            if video_in_chunk_offsets is None:
                resolved_video_offsets = DEFAULT_VIDEO_IN_CHUNK_OFFSETS
            else:
                resolved_video_offsets = tuple(int(x) for x in video_in_chunk_offsets)
            self._multi_anchor_cfg = MultiAnchorTemporalConfig(
                max_chunk_size=self.max_chunk_size,
                action_horizon=self.action_horizon,
                macro_stride=resolved_macro_stride,
                video_in_chunk_offsets=resolved_video_offsets,
            )
        if self._use_lazy_video_tree:
            self._init_lazy_map_style(pq_cache_max_episodes, video_tolerance_s)
        else:
            self._init_lerobot_or_v2_parquet()

    @staticmethod
    def _short_modality_key(key: str) -> str:
        return key.split(".", 1)[1] if "." in key else key

    def _uses_image_parquet_storage(self) -> bool:
        """Detect LeRobot trees where image frames live in parquet, not mp4 files."""
        source_keys = set(self._source_video_key.values())
        if not source_keys:
            return False
        source_features = [self._features.get(key) or {} for key in source_keys]
        has_video_feature = any(
            feature.get("dtype") == "video" for feature in source_features
        )
        if has_video_feature:
            return False
        all_image_features = all(
            feature.get("dtype") == "image" for feature in source_features
        )
        if not all_image_features:
            return False
        return (
            self._version.startswith("v2")
            or int(self._info.get("total_videos") or 0) == 0
            or not self._info.get("video_path")
        )

    def _modality_entry(self, modality: str, key: str) -> dict[str, Any] | None:
        entries = self._modality_meta.get(modality)
        if not isinstance(entries, dict):
            return None
        return entries.get(self._short_modality_key(key))

    def _build_source_video_key_map(self) -> dict[str, str]:
        image_features = [
            k for k in self._features if k.startswith("observation.images.")
        ]
        mapping: dict[str, str] = {}
        for transform_key in self.video_keys:
            short = self._short_modality_key(transform_key)
            entry = self._modality_entry("video", transform_key)
            if entry is not None and entry.get("original_key"):
                mapping[transform_key] = str(entry["original_key"])
                continue
            canonical = f"observation.images.{short}"
            if canonical in self._features:
                mapping[transform_key] = canonical
                continue
            # LeRobot v2 LIBERO stores image columns without the observation.images prefix.
            if short in self._features or short in ("image", "wrist_image"):
                mapping[transform_key] = short
                continue
            match = [k for k in image_features if k.endswith(f".{short}")]
            if match:
                mapping[transform_key] = match[0]
        if len(mapping) != len(self.video_keys):
            missing = [k for k in self.video_keys if k not in mapping]
            raise KeyError(
                f"Could not map transform video keys {missing} to LeRobot image features "
                f"under {self.data_path}; available={sorted(image_features)}"
            )
        return mapping

    def _default_vector_source_key(self, modality: str) -> str:
        candidates = (
            ("observation.state", "state")
            if modality == "state"
            else ("action", "actions")
        )
        for candidate in candidates:
            if candidate in self._features:
                return candidate
        return candidates[0]

    def _build_component_sources(
        self, modality: str, transform_keys: list[str]
    ) -> dict[str, tuple[str, slice]]:
        sources: dict[str, tuple[str, slice]] = {}
        missing_keys: list[str] = []
        for key in transform_keys:
            entry = self._modality_entry(modality, key)
            if entry is None:
                missing_keys.append(key)
                continue
            source = str(
                entry.get("original_key") or self._default_vector_source_key(modality)
            )
            start = int(entry.get("start", 0))
            end = entry.get("end")
            sources[key] = (source, slice(start, None if end is None else int(end)))
        if not missing_keys:
            return sources

        missing_short_keys = [self._short_modality_key(k) for k in missing_keys]
        if modality == "state":
            feature = (
                self._features.get("observation.state")
                or self._features.get("state")
                or {}
            )
        else:
            feature = (
                self._features.get("action") or self._features.get("actions") or {}
            )
        dim = int((feature.get("shape") or [0])[0] or 0)
        inferred = infer_named_component_slices(
            feature.get("names"), dim, missing_short_keys
        )
        if inferred is None and len(missing_short_keys) == 1:
            inferred = {missing_short_keys[0]: slice(0, dim or None)}
        if inferred is None:
            # Backward-compatible DROID fallback when meta names are missing.
            if set(missing_short_keys).issubset({"joint_position", "gripper_position"}):
                st_j, st_g, ac_j, ac_g = droid_default_state_action_slices()
                inferred = (
                    {"joint_position": st_j, "gripper_position": st_g}
                    if modality == "state"
                    else {"joint_position": ac_j, "gripper_position": ac_g}
                )
            else:
                raise ValueError(
                    f"Cannot infer {modality} component slices for keys={transform_keys} "
                    f"from feature names={feature.get('names')!r} dim={dim}"
                )
        source = self._default_vector_source_key(modality)
        for key in missing_keys:
            sources[key] = (source, inferred[self._short_modality_key(key)])
        return sources

    def _build_language_sources(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        annotations = self._modality_meta.get("annotation")
        for key in self.language_keys:
            subkey = self._short_modality_key(key)
            entry = annotations.get(subkey) if isinstance(annotations, dict) else None
            source = entry.get("original_key") if entry else None
            sources[key] = str(source or f"annotation.{subkey}")
        return sources

    def _init_lazy_map_style(
        self, pq_cache_max_episodes: int, video_tolerance_s: float
    ) -> None:
        if not self._root.exists():
            raise FileNotFoundError(
                f"DreamZero data_path must be local: {self.data_path}"
            )
        self._chunks_size = int(self._info.get("chunks_size") or 1000)
        self._data_tmpl = str(self._info.get("data_path") or "")
        self._video_tmpl = str(self._info.get("video_path") or "")
        if not self._data_tmpl:
            raise ValueError("meta/info.json missing data_path")

        meta_episode_indices: set[int] = set()
        episode_lengths: dict[int, int] = {}
        with open(self._meta_dir / "episodes.jsonl") as epf:
            for line in epf:
                if not line.strip():
                    continue
                obj = json.loads(line)
                ep_idx = int(obj.get("episode_index", 0))
                meta_episode_indices.add(ep_idx)
                for k in ("episode_length", "length", "num_frames", "num_steps"):
                    if obj.get(k) is not None:
                        episode_lengths[ep_idx] = int(obj[k])
                        break

        self._episodes = discover_local_lerobot_episode_indices(
            self._root, self._info, allowed_episode_indices=meta_episode_indices
        )
        self._episode_lengths = [
            episode_lengths[ep]
            if ep in episode_lengths
            else self._infer_episode_length_from_parquet(ep)
            for ep in self._episodes
        ]
        self._episode_starts = [0]
        total = 0
        for n in self._episode_lengths:
            total += int(n)
            self._episode_starts.append(total)
        self._total_frames = int(total)
        self._pq_cache: "OrderedDict[int, Any]" = OrderedDict()
        self._pq_cache_max_episodes = max(1, int(pq_cache_max_episodes))
        self._video_decode_fps_cache: "OrderedDict[str, float]" = OrderedDict()
        self._video_decode_fps_cache_max = 512
        self._video_tolerance_s = float(video_tolerance_s)
        if self._video_tolerance_s <= 0:
            raise ValueError(
                f"video_tolerance_s must be positive, got {video_tolerance_s!r}"
            )

    def _init_lerobot_or_v2_parquet(self) -> None:
        if self._use_image_parquet_tree:
            self._init_v2_image_parquet()
            return
        import lerobot.datasets.lerobot_dataset as lerobot_dataset

        if self.sampling_mode == "multi_anchor":
            delta_timestamps = {
                self._source_video_key[k]: [0.0] for k in self.video_keys
            }
            state_sources = {source for source, _ in self._state_components.values()}
            action_sources = {source for source, _ in self._action_components.values()}
            for source in state_sources | action_sources:
                delta_timestamps[source] = [0.0]
        else:
            fixed_window = self._fixed_window_temporal
            assert fixed_window is not None
            delta_timestamps = {
                self._source_video_key[k]: [t / self._fps for t in fixed_window.video]
                for k in self.video_keys
            }
            state_sources = {source for source, _ in self._state_components.values()}
            action_sources = {source for source, _ in self._action_components.values()}
            for source in state_sources:
                delta_timestamps[source] = [t / self._fps for t in fixed_window.state]
            for source in action_sources:
                delta_timestamps[source] = [t / self._fps for t in fixed_window.action]
        self.dataset = lerobot_dataset.LeRobotDataset(
            self.data_path,
            root=self._root,
            delta_timestamps=delta_timestamps,
            video_backend=self._video_backend,
        )
        self._use_v2_image_parquet = False

    def _init_v2_image_parquet(self) -> None:
        import pyarrow.parquet as pq

        data_root = self._root / "data"
        episodes_path = self._meta_dir / "episodes.jsonl"
        self._episodes_meta = []
        with open(episodes_path) as f:
            for line in f:
                if line.strip():
                    self._episodes_meta.append(json.loads(line))
        self._ep_frames = []
        self._ep_parquet_paths = []
        for ep in self._episodes_meta:
            ep_idx = int(ep["episode_index"])
            pq_path = (
                data_root
                / f"chunk-{ep_idx // 1000:03d}"
                / f"episode_{ep_idx:06d}.parquet"
            )
            table = pq.read_table(pq_path)
            self._ep_frames.append(len(table))
            self._ep_parquet_paths.append(pq_path)
        self._cumulative = np.cumsum(self._ep_frames)
        self._total_frames = int(self._cumulative[-1])
        self._pq_cache = {}
        self._use_v2_image_parquet = True

    def _read_v2_episode(self, ep_pos: int):
        if ep_pos not in self._pq_cache:
            import pyarrow.parquet as pq

            self._pq_cache[ep_pos] = pq.read_table(str(self._ep_parquet_paths[ep_pos]))
            if len(self._pq_cache) > 50:
                del self._pq_cache[next(iter(self._pq_cache))]
        return self._pq_cache[ep_pos]

    def _decode_v2_image(self, cell) -> np.ndarray:
        from io import BytesIO

        from PIL import Image

        raw = cell.as_py()
        if isinstance(raw, dict):
            raw = raw.get("bytes", raw)
        if isinstance(raw, bytes):
            return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        return np.asarray(raw)

    def _get_v2_image_sample(self, idx: int) -> dict[str, Any]:
        frame_in_ep, episode_index, ep_len = self._resolve_index_context(idx)
        ep_pos = int(np.searchsorted(self._cumulative, idx, side="right"))
        table = self._read_v2_episode(ep_pos)
        video_offsets, state_offsets, action_offsets = self._temporal_offsets_for_frame(
            frame_in_ep, episode_index, ep_len
        )
        video_idx, state_idx, action_idx = self._episode_frame_indices(
            frame_in_ep, ep_len, video_offsets, state_offsets, action_offsets
        )

        sample: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_index": frame_in_ep,
        }
        for transform_key, source_key in self._source_video_key.items():
            sample[transform_key] = np.stack(
                [
                    self._decode_v2_image(table.column(source_key)[int(i)])
                    for i in video_idx
                ],
                axis=0,
            )
        state_rows = state_idx.tolist()
        action_rows = action_idx.tolist()
        state_sources = {source for source, _ in self._state_components.values()}
        action_sources = {source for source, _ in self._action_components.values()}
        for source in state_sources:
            if source not in table.column_names:
                continue
            sample[source] = np.asarray(
                [table.column(source)[r].as_py() for r in state_rows], dtype=np.float32
            )
        for source in action_sources:
            if source not in table.column_names:
                continue
            sample[source] = np.asarray(
                [table.column(source)[r].as_py() for r in action_rows], dtype=np.float32
            )
        for key, source in self._language_sources.items():
            if source in table.column_names:
                sample[key] = table.column(source)[frame_in_ep].as_py()
        if "task" in table.column_names:
            sample["task"] = table.column("task")[frame_in_ep].as_py()
        elif "task_index" in table.column_names:
            sample["task_index"] = table.column("task_index")[frame_in_ep].as_py()
        return sample

    def _infer_episode_length_from_parquet(self, episode_index: int) -> int:
        import pyarrow.parquet as pq

        return int(
            pq.read_metadata(str(self._get_parquet_path(episode_index))).num_rows
        )

    def _get_parquet_path(self, episode_index: int) -> Path:
        ep_chunk = int(episode_index) // self._chunks_size
        rel = Path(
            self._data_tmpl.format(
                episode_chunk=ep_chunk, episode_index=int(episode_index)
            )
        )
        p = (self._root / rel).resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"Parquet file not found for episode {episode_index}: {p}"
            )
        return p

    def _get_video_path(self, episode_index: int, video_key: str) -> Path:
        ep_chunk = int(episode_index) // self._chunks_size
        rel = Path(
            self._video_tmpl.format(
                episode_chunk=ep_chunk,
                video_key=video_key,
                episode_index=int(episode_index),
            )
        )
        p = (self._root / rel).resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"Video file not found for episode {episode_index} key {video_key}: {p}"
            )
        return p

    def _decode_fps_for_video_file(self, video_path: Path) -> float:
        key = str(video_path.resolve())
        if key in self._video_decode_fps_cache:
            fps = self._video_decode_fps_cache.pop(key)
            self._video_decode_fps_cache[key] = fps
            return fps
        fps = float(probe_video_container_fps(video_path) or self._fps)
        self._video_decode_fps_cache[key] = fps
        if len(self._video_decode_fps_cache) > self._video_decode_fps_cache_max:
            self._video_decode_fps_cache.popitem(last=False)
        return fps

    def _get_episode_table(self, episode_index: int):
        episode_index = int(episode_index)
        if episode_index in self._pq_cache:
            tbl = self._pq_cache.pop(episode_index)
            self._pq_cache[episode_index] = tbl
            return tbl
        import pyarrow.parquet as pq

        p = self._get_parquet_path(episode_index)
        schema = set(pq.read_schema(str(p)).names)
        cols = [
            c
            for c in (
                "observation",
                *self._vector_source_keys,
                "task",
                "task_index",
                *self._language_sources.values(),
            )
            if c in schema
        ]
        tbl = pq.read_table(str(p), columns=list(dict.fromkeys(cols)))
        self._pq_cache[episode_index] = tbl
        if len(self._pq_cache) > self._pq_cache_max_episodes:
            self._pq_cache.popitem(last=False)
        return tbl

    @staticmethod
    def _clip_indices(indices: np.ndarray, length: int) -> np.ndarray:
        return np.clip(indices.astype(np.int64), 0, max(0, int(length) - 1))

    def _episode_frame_indices(
        self,
        frame_in_ep: int,
        ep_len: int,
        video_offsets: np.ndarray,
        state_offsets: np.ndarray,
        action_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map temporal sampling output to episode-local frame indices.

        ``fixed_window`` returns offsets relative to ``frame_in_ep``; ``multi_anchor``
        returns episode-local absolute indices.
        """
        if self.sampling_mode == "fixed_window":
            return (
                self._clip_indices(frame_in_ep + video_offsets, ep_len),
                self._clip_indices(frame_in_ep + state_offsets, ep_len),
                self._clip_indices(frame_in_ep + action_offsets, ep_len),
            )
        return (
            self._clip_indices(video_offsets, ep_len),
            self._clip_indices(state_offsets, ep_len),
            self._clip_indices(action_offsets, ep_len),
        )

    @staticmethod
    def _video_to_thwc_uint8(frames: Any) -> np.ndarray:
        """Match Groot VideoTransform: numpy (T, H, W, C) uint8.

        LeRobot ``decode_video_frames`` returns float32 (T, C, H, W) in [0, 1].
        Parquet/PIL paths usually already yield uint8 (T, H, W, C).
        """
        if torch.is_tensor(frames):
            arr = frames.detach().cpu().numpy()
        else:
            arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None, ...]
        elif arr.ndim == 5:
            # (B, T, C, H, W) -> (B, T, H, W, C)
            if arr.shape[2] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (0, 1, 3, 4, 2))
        elif arr.ndim == 4:
            if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32, copy=False)
            max_v = float(arr_f.max()) if arr_f.size else 0.0
            if max_v <= 1.0 + 1e-3:
                arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr_f, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _col_exists(table, name: str) -> bool:
        return hasattr(table, "column_names") and name in table.column_names

    @staticmethod
    def _read_list_column(table, name: str, indices: np.ndarray) -> np.ndarray:
        col = table.column(name)
        return np.asarray(
            [col[int(i)].as_py() for i in indices.tolist()], dtype=np.float32
        )

    @staticmethod
    def _read_struct_list_field(
        table, struct_col: str, field: str, indices: np.ndarray
    ) -> np.ndarray:
        col = table.column(struct_col)
        arr = col.chunk(0) if hasattr(col, "num_chunks") and col.num_chunks > 0 else col
        field_arr = arr.field(field)
        return np.asarray(
            [field_arr[int(i)].as_py() for i in indices.tolist()], dtype=np.float32
        )

    def _resolve_index_context(self, idx: int) -> tuple[int, int, int]:
        """Return ``(frame_in_ep, episode_index, episode_length)`` for a global index."""
        if self._use_lazy_video_tree:
            if idx < 0 or idx >= self._total_frames:
                raise IndexError(
                    f"Index {idx} out of range for dataset of len {self._total_frames}"
                )
            ep_pos = bisect.bisect_right(self._episode_starts, int(idx)) - 1
            ep_pos = max(0, min(ep_pos, len(self._episodes) - 1))
            frame_in_ep = int(idx) - int(self._episode_starts[ep_pos])
            episode_index = int(self._episodes[ep_pos])
            ep_len = int(self._episode_lengths[ep_pos])
            return frame_in_ep, episode_index, ep_len
        if getattr(self, "_use_v2_image_parquet", False):
            if idx < 0 or idx >= self._total_frames:
                raise IndexError(
                    f"Index {idx} out of range for dataset of len {self._total_frames}"
                )
            ep_pos = int(np.searchsorted(self._cumulative, idx, side="right"))
            frame_in_ep = (
                int(idx) if ep_pos == 0 else int(idx - self._cumulative[ep_pos - 1])
            )
            episode_index = int(
                self._episodes_meta[ep_pos].get("episode_index", ep_pos)
            )
            ep_len = int(self._ep_frames[ep_pos])
            return frame_in_ep, episode_index, ep_len
        raise RuntimeError(
            "_resolve_index_context requires lazy video tree or v2 image parquet in multi_anchor mode"
        )

    def _language_column_for_episode_table(self, table) -> str | None:
        for source in self._language_sources.values():
            if self._col_exists(table, source):
                return source
        for key in ("task_index", "task"):
            if self._col_exists(table, key):
                return key
        return None

    def _read_episode_language_labels(self, episode_index: int) -> np.ndarray:
        table = self._get_episode_table(episode_index)
        col = self._language_column_for_episode_table(table)
        if col is None:
            return np.zeros(table.num_rows, dtype=object)
        if col == "task_index":
            return np.asarray(
                [table.column(col)[i].as_py() for i in range(table.num_rows)],
                dtype=np.int64,
            )
        return np.asarray(
            [table.column(col)[i].as_py() for i in range(table.num_rows)],
            dtype=object,
        )

    def _temporal_offsets_for_frame(
        self, frame_in_ep: int, episode_index: int, ep_len: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.sampling_mode == "fixed_window":
            assert self._fixed_window_temporal is not None
            t = self._fixed_window_temporal
            return t.video, t.state, t.action
        assert self._multi_anchor_cfg is not None
        language = self._read_episode_language_labels(episode_index)
        temporal = require_multi_anchor_temporal_indices(
            frame_in_ep,
            language,
            ep_len,
            self._multi_anchor_cfg,
            episode_index=episode_index,
        )
        return temporal.video, temporal.state, temporal.action

    def _decode_video_frames(
        self,
        video_path: Path | str,
        timestamps: list[float],
        tolerance_s: float,
    ) -> Any:
        """Decode video frames, dispatching by ``self._video_backend``.

        decord is handled inline (returns numpy uint8 THWC) so we do not
        modify the vendored lerobot.  All other backends delegate to lerobot.
        """
        if self._video_backend == "decord":
            return self._decode_video_frames_decord(video_path, timestamps, tolerance_s)

        if self._video_backend == "torchcodec":
            return self._decode_video_frames_torchcodec(
                video_path, timestamps, tolerance_s
            )

        from lerobot.datasets.video_utils import (
            decode_video_frames as decode_video_frames_lerobot,
        )

        return decode_video_frames_lerobot(
            video_path,
            timestamps,
            tolerance_s,
            backend=self._video_backend,
        )

    @staticmethod
    def _decode_video_frames_decord(
        video_path: Path | str,
        timestamps: list[float],
        tolerance_s: float,
    ) -> np.ndarray:
        """Decode frames via decord; return numpy uint8 (T, H, W, C).

        No float conversion or /255 is performed here — the caller
        (:meth:`_build_modality_dict`) already normalises via
        :meth:`_video_to_thwc_uint8`.
        """

        import decord

        vr = decord.VideoReader(str(video_path), num_threads=1)
        total_frames = len(vr)
        average_fps = vr.get_avg_fps()
        frame_indices = [
            min(round(ts * average_fps), total_frames - 1) for ts in timestamps
        ]

        frames = vr.get_batch(frame_indices)
        arr = frames.asnumpy()  # (T, H, W, C) uint8
        return arr

    @staticmethod
    def _decode_video_frames_torchcodec(
        video_path: Path | str,
        timestamps: list[float],
        tolerance_s: float,
    ) -> np.ndarray:
        """Decode frames via torchcodec; return numpy uint8 (T, H, W, C).

        Mirrors ``lerobot.datasets.video_utils.decode_video_frames_torchcodec``
        but skips float32 normalisation (the caller normalises via
        :meth:`_video_to_thwc_uint8`) and returns uint8 THWC instead of
        float32 TCHW.

        The tolerance check (query vs loaded pts_seconds via cdist) is
        preserved verbatim from the LeRobot reference implementation so
        that approximate-seek mismatches are detected early.
        """

        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(
            str(video_path),
            device="cpu",
            seek_mode="approximate",
            dimension_order="NHWC",
        )
        loaded_frames = []
        loaded_ts = []

        # get metadata for frame information
        metadata = decoder.metadata
        average_fps = metadata.average_fps

        # convert timestamps to frame indices
        frame_indices = [round(ts * average_fps) for ts in timestamps]

        # retrieve frames based on indices
        frames_batch = decoder.get_frames_at(indices=frame_indices)

        for frame, pts in zip(
            frames_batch.data, frames_batch.pts_seconds, strict=False
        ):
            loaded_frames.append(frame)
            loaded_ts.append(pts.item())

        query_ts = torch.tensor(timestamps)
        loaded_ts = torch.tensor(loaded_ts)

        dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
        min_dists, argmin = dist.min(1)

        is_within_tol = min_dists < float(tolerance_s)
        if not is_within_tol.all():
            violators = min_dists[~is_within_tol]
            raise RuntimeError(
                f"[dreamzero] torchcodec tolerance violation: {violators.tolist()} > "
                f"tolerance_s={tolerance_s}. "
                f"video={video_path} "
                f"n={len(timestamps)} idx=[{min(frame_indices)}..{max(frame_indices)}]"
            )

        # Select the closest frame to each query timestamp (handles approximate-seek drift).
        tensor = torch.stack([loaded_frames[int(idx)] for idx in argmin])
        # tensor is now (T, H, W, C) uint8 on CPU, contiguous.

        # torchcodec returns FrameBatch with .data as a torch.Tensor.
        # On CPU with contiguous NHWC layout, .numpy(force=True) is a zero-copy view.
        arr = tensor.numpy(force=True)

        return arr

    def _materialize_parquet_sample(
        self,
        frame_in_ep: int,
        episode_index: int,
        ep_len: int,
        table: Any,
        *,
        decode_video: bool,
    ) -> dict[str, Any]:
        video_offsets, state_offsets, action_offsets = self._temporal_offsets_for_frame(
            frame_in_ep, episode_index, ep_len
        )
        video_idx, state_idx, action_idx = self._episode_frame_indices(
            frame_in_ep, ep_len, video_offsets, state_offsets, action_offsets
        )

        sample: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_index": frame_in_ep,
        }
        if decode_video:
            for transform_key, source_key in self._source_video_key.items():
                video_path = self._get_video_path(episode_index, source_key)
                fps = self._decode_fps_for_video_file(video_path)
                sample[transform_key] = self._decode_video_frames(
                    video_path,
                    [float(int(i)) / fps for i in video_idx.tolist()],
                    tolerance_s=self._video_tolerance_s,
                )

        for key in ("task", "task_index"):
            if self._col_exists(table, key):
                sample[key] = table.column(key)[int(frame_in_ep)].as_py()
        for key, source in self._language_sources.items():
            if self._col_exists(table, source):
                sample[key] = table.column(source)[int(frame_in_ep)].as_py()

        for source, _ in self._state_components.values():
            if source in sample:
                continue
            if self._col_exists(table, source):
                sample[source] = self._read_list_column(table, source, state_idx)
            elif source == "observation.state" and self._col_exists(
                table, "observation"
            ):
                sample[source] = self._read_struct_list_field(
                    table, "observation", "state", state_idx
                )
            else:
                raise KeyError(
                    f"episode parquet missing state source column {source!r}"
                )
        for source, _ in self._action_components.values():
            if source in sample:
                continue
            if not self._col_exists(table, source):
                raise KeyError(
                    f"episode parquet missing action source column {source!r}"
                )
            sample[source] = self._read_list_column(table, source, action_idx)
        return sample

    def _get_lazy_sample(self, idx: int) -> dict[str, Any]:
        frame_in_ep, episode_index, ep_len = self._resolve_index_context(idx)
        table = self._get_episode_table(episode_index)
        return self._materialize_parquet_sample(
            frame_in_ep,
            episode_index,
            ep_len,
            table,
            decode_video=True,
        )

    def __len__(self) -> int:
        if self._use_lazy_video_tree or getattr(self, "_use_v2_image_parquet", False):
            return int(self._total_frames)
        return len(self.dataset)

    def _resolve_task_text(self, sample: dict[str, Any]) -> str:
        task_text = sample.get("task")
        if task_text is None:
            task_text = self._tasks.get(int(sample.get("task_index", 0)), "")
        candidates = []
        for key in self.language_keys:
            if key in sample:
                text = safe_lang_text(sample[key], self._tasks)
                if text:
                    candidates.append(text)
        if candidates:
            return str(np.random.choice(candidates))
        return str(task_text or "")

    def _state_key_for_relative_action(self, action_key: str) -> str | None:
        """Pair ``action.<sub>`` with ``state.<sub>`` (groot convention for relative_action)."""
        short = self._short_modality_key(action_key)
        for sk in self.state_keys:
            if self._short_modality_key(sk) == short:
                return sk
        return None

    def _subtract_relative_action(
        self, value: np.ndarray, out: dict[str, Any], action_key: str
    ) -> np.ndarray:
        """Per-chunk anchor subtraction to match groot ``lerobot_sharded`` relative_action.

        In ``groot/vla/data/dataset/lerobot_sharded.py::_convert_to_relative_action``, each
        block of ``chunk_size`` action rows uses the trajectory state at the **first** row
        of that block as reference. Here ``chunk_size`` corresponds to ``action_horizon``,
        and state rows from ``_state_offsets`` are ordered as
        ``chunk_idx * action_horizon + state_idx``, so the anchor row for chunk ``c`` is
        index ``c * state_horizon`` in the stacked state tensor for the matching modality.
        """
        sk = self._state_key_for_relative_action(action_key)
        if sk is None or sk not in out:
            return value
        state_arr = np.asarray(out[sk], dtype=np.float32)
        if state_arr.ndim == 1:
            state_arr = state_arr[None, :]
        t_act, ad = value.shape
        n_st, sd = state_arr.shape
        if self.sampling_mode == "multi_anchor":
            assert self._multi_anchor_cfg is not None
            action_horizon = self._multi_anchor_cfg.action_horizon
            if t_act > 0 and t_act % action_horizon == 0 and n_st > 0:
                v = value.astype(np.float32, copy=True)
                d_ref = min(ad, sd)
                num_macro_chunks = t_act // action_horizon
                for c in range(num_macro_chunks):
                    rs = c * action_horizon
                    re = rs + action_horizon
                    anchor_idx = min(c * self.state_horizon, n_st - 1)
                    ref = state_arr[anchor_idx : anchor_idx + 1, :d_ref]
                    v[rs:re, :d_ref] = value[rs:re, :d_ref].astype(np.float32) - ref
                if d_ref < ad:
                    v[:, d_ref:] = value[:, d_ref:].astype(np.float32)
                return v
            return value.astype(np.float32) - state_arr[0:1, :ad]

        exp_act = self.max_chunk_size * self.action_horizon
        exp_st = self.max_chunk_size * self.state_horizon
        if (
            t_act == exp_act
            and n_st == exp_st
            and self.max_chunk_size > 0
            and self.action_horizon > 0
            and self.state_horizon > 0
        ):
            v = value.astype(np.float32, copy=True)
            d_ref = min(ad, sd)
            for c in range(self.max_chunk_size):
                r = c * self.state_horizon
                ref = state_arr[r : r + 1, :d_ref]
                rs = c * self.action_horizon
                re = rs + self.action_horizon
                v[rs:re, :d_ref] = value[rs:re, :d_ref].astype(np.float32) - ref
            if d_ref < ad:
                v[:, d_ref:] = value[:, d_ref:].astype(np.float32)
            return v
        return value.astype(np.float32) - state_arr[0:1, :ad]

    def _put_components(
        self,
        out: dict[str, Any],
        sample: dict[str, Any],
        components: dict[str, tuple[str, slice]],
        *,
        is_action: bool,
    ) -> None:
        for key, (source, sl) in components.items():
            raw = np.asarray(sample[source], dtype=np.float32)
            if raw.ndim == 1:
                raw = raw[None, :]
            value = raw[:, sl].astype(np.float32)
            if (
                is_action
                and self.relative_action
                and self._short_modality_key(key) in self.relative_action_keys
            ):
                value = self._subtract_relative_action(value, out, key)
            out[key] = value

    def _load_raw_sample(self, idx: int) -> dict[str, Any]:
        if self._use_lazy_video_tree:
            return self._get_lazy_sample(idx)
        if getattr(self, "_use_v2_image_parquet", False):
            return self._get_v2_image_sample(idx)
        return self.dataset[idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.multi_anchor_resample_attempts):
            try:
                raw = self._build_modality_dict(self._load_raw_sample(idx))
                transformed = self.data_transform(raw)
                return collate_ready_sample(transformed)
            except EmptyTemporalSampleError as exc:
                last_error = exc
                if self.sampling_mode != "multi_anchor":
                    raise
                if attempt + 1 >= self.multi_anchor_resample_attempts:
                    break
                idx = random.randint(0, max(0, len(self) - 1))
        raise EmptyTemporalSampleError(
            f"Failed to sample a valid multi_anchor index after "
            f"{self.multi_anchor_resample_attempts} attempts: {last_error}"
        )

    def _build_modality_dict(self, sample: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for transform_key, source_key in self._source_video_key.items():
            if transform_key in sample:
                raw_frames = sample[transform_key]
            else:
                raw_frames = sample[source_key]
            out[transform_key] = self._video_to_thwc_uint8(raw_frames)

        self._put_components(out, sample, self._state_components, is_action=False)
        self._put_components(out, sample, self._action_components, is_action=True)

        fallback_text = self._resolve_task_text(sample)
        wrote_language = False
        for key in self.language_keys:
            source = self._language_sources.get(key, key)
            value = sample.get(key, sample.get(source, ""))
            text = safe_lang_text(value, self._tasks) if value != "" else ""
            if text:
                out[key] = text
                wrote_language = True
        if not wrote_language:
            out[self.language_keys[0]] = fallback_text
        return out


def is_dreamzero_mixture_spec(data_paths: Any) -> bool:
    """Return True when ``data_paths`` is a list of per-dataset mixture specs."""
    if data_paths is None:
        return False
    if isinstance(data_paths, str):
        return False
    # OmegaConf ListConfig is not a list/tuple subclass.
    if not (isinstance(data_paths, (list, tuple)) or OmegaConf.is_list(data_paths)):
        return False
    if len(data_paths) == 0:
        raise ValueError(
            "DreamZero mixture spec must contain at least one dataset entry."
        )
    first = data_paths[0]
    return isinstance(first, Mapping) or OmegaConf.is_dict(first)


def _normalize_mixture_spec(data_paths: Sequence[Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, item in enumerate(data_paths):
        if isinstance(item, Mapping):
            spec = dict(item)
        elif OmegaConf.is_dict(item):
            spec = OmegaConf.to_container(item, resolve=True)  # type: ignore[arg-type]
            assert isinstance(spec, dict)
        else:
            raise ValueError(
                f"Invalid DreamZero mixture entry at index {index}: expected a dict-like "
                f"spec with dataset_path and weight, got {type(item)!r}."
            )
        specs.append(spec)
    return specs


def _resolve_dataset_paths(spec: dict[str, Any]) -> list[str]:
    paths = spec.get("dataset_path", spec.get("data_path"))
    if paths is None:
        raise ValueError(
            "Each DreamZero mixture spec entry must define 'dataset_path' or 'data_path'."
        )
    if isinstance(paths, str):
        return [paths]
    return [str(path) for path in paths]


def build_single_dreamzero_lerobot_dataset(
    *,
    data_path: str,
    cfg: DictConfig,
    embodiment_tag: str | None = None,
    metadata_json_path: str | None = None,
    eval_dataset: bool = False,
):
    """Build one :class:`DreamZeroLeRobotDataset` with an embodiment-specific transform."""
    from groot.vla.data.transform import ComposedModalityTransform

    from rlinf.data.datasets.dreamzero.data_transforms import (
        build_dreamzero_composed_transform,
        collect_dreamzero_dataset_keys,
        load_dreamzero_dataset_metadata,
    )

    data_cfg = cfg.data
    model_cfg = cfg.actor.model
    tokenizer_path = model_cfg.get("tokenizer_path", "google/umt5-xxl")
    tag = embodiment_tag or model_cfg.embodiment_tag

    model_overrides: dict[str, Any] = {"embodiment_tag": tag}
    if metadata_json_path is not None:
        model_overrides["metadata_json_path"] = metadata_json_path
    sub_model_cfg = OmegaConf.merge(model_cfg, model_overrides)

    metadata = load_dreamzero_dataset_metadata(sub_model_cfg)
    transform_on_gpu = bool(data_cfg.get("transform_on_gpu", False))
    data_transform = build_dreamzero_composed_transform(
        sub_model_cfg, tokenizer_path, transform_on_gpu=transform_on_gpu
    )

    assert isinstance(data_transform, ComposedModalityTransform), f"{data_transform=}"
    data_transform.set_metadata(metadata)
    if eval_dataset:
        data_transform.eval()
    else:
        data_transform.train()

    video_keys, state_keys, action_keys, language_keys = collect_dreamzero_dataset_keys(
        data_transform, tag
    )

    sampling_mode = data_cfg.get("sampling_mode", "multi_anchor")
    if sampling_mode not in ("multi_anchor", "fixed_window"):
        raise ValueError(
            f"Unsupported data.sampling_mode {sampling_mode!r}; "
            "use 'multi_anchor' or 'fixed_window'."
        )

    max_chunk_size = model_cfg.action_head_cfg.config.diffusion_model_cfg.max_chunk_size
    num_frames = model_cfg.action_head_cfg.config.num_frames
    state_horizon = model_cfg.get("state_horizon", 1)
    action_horizon = model_cfg.action_horizon
    macro_stride = data_cfg.get("macro_stride")
    if macro_stride is not None:
        macro_stride = int(macro_stride)
    video_in_chunk_offsets = data_cfg.get("video_in_chunk_offsets")
    if video_in_chunk_offsets is not None:
        video_in_chunk_offsets = tuple(int(x) for x in video_in_chunk_offsets)

    dataset = DreamZeroLeRobotDataset(
        data_path=data_path,
        video_keys=video_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        language_keys=language_keys,
        data_transform=data_transform,
        lazy_load=cfg.data.get("lazy_load", True),
        num_frames=num_frames,
        state_horizon=state_horizon,
        action_horizon=action_horizon,
        relative_action=bool(model_cfg.get("relative_action", False)),
        relative_action_keys=list(model_cfg.get("relative_action_keys", [])),
        pq_cache_max_episodes=cfg.data.get("parquet_cache_size", 128),
        video_tolerance_s=cfg.data.get("video_tolerance_s", 0.1),
        video_backend=data_cfg.get("video_backend", "pyav"),
        max_chunk_size=max_chunk_size,
        sampling_mode=sampling_mode,  # type: ignore[arg-type]
        multi_anchor_resample_attempts=data_cfg.get(
            "multi_anchor_resample_attempts", 8
        ),
        macro_stride=macro_stride,
        video_in_chunk_offsets=video_in_chunk_offsets,
    )
    multi_anchor_cfg = dataset._multi_anchor_cfg
    logger.info(
        "DreamZero LeRobot dataset: path=%s embodiment=%s sampling_mode=%s "
        "max_chunk_size=%s action_horizon=%s macro_stride=%s video_in_chunk_offsets=%s "
        "video_keys=%s state_keys=%s action_keys=%s language_keys=%s",
        data_path,
        tag,
        dataset.sampling_mode,
        dataset.max_chunk_size,
        dataset.action_horizon,
        None if multi_anchor_cfg is None else multi_anchor_cfg.macro_stride,
        None if multi_anchor_cfg is None else multi_anchor_cfg.video_in_chunk_offsets,
        dataset.video_keys,
        dataset.state_keys,
        dataset.action_keys,
        dataset.language_keys,
    )
    return dataset


class DreamZeroLeRobotMixtureDataset(Dataset):
    """Weighted mixture of :class:`DreamZeroLeRobotDataset` instances.

    Each sub-dataset keeps its own data transform so raw LeRobot samples are
    converted into a unified training format before collation. Sampling picks a
    dataset according to normalized weights, then delegates to that dataset's
    ``__getitem__``.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[Any, float]],
        training: bool = True,
        seed: int = 42,
    ):
        if not data_mixture:
            raise ValueError(
                "DreamZeroLeRobotMixtureDataset requires at least one dataset."
            )

        datasets: list[Any] = []
        raw_weights: list[float] = []
        for dataset, weight in data_mixture:
            if len(dataset) == 0:
                logger.warning("Skipping empty DreamZero LeRobot dataset: %s", dataset)
                continue
            datasets.append(dataset)
            raw_weights.append(float(weight))

        if not datasets:
            raise ValueError(
                "No valid (non-empty) DreamZero LeRobot datasets provided."
            )

        self.datasets = datasets
        self.training = bool(training)
        self.seed = int(seed)

        self._dataset_lengths = np.array(
            [len(dataset) for dataset in self.datasets], dtype=np.int64
        )
        self._raw_weights = np.array(raw_weights, dtype=np.float64)

        self._dataset_sampling_weights = self._raw_weights.copy()
        weight_sum = self._dataset_sampling_weights.sum()
        if weight_sum <= 0 or np.isnan(weight_sum):
            raise ValueError(
                f"Invalid DreamZero mixture sampling weights: sum={weight_sum!r}"
            )
        self._dataset_sampling_weights /= weight_sum

        self._epoch = 0
        self._log_initialization()

    def _log_initialization(self) -> None:
        logger.info("DreamZeroLeRobotMixtureDataset initialized:")
        logger.info("  Datasets: %s", len(self.datasets))
        logger.info("  Total samples: %s", int(self._dataset_lengths.sum()))
        logger.info("  Dataset lengths: %s", self._dataset_lengths.tolist())
        logger.info("  Raw weights: %s", self._raw_weights.tolist())
        logger.info("  Sampling weights: %s", self._dataset_sampling_weights.tolist())
        logger.info("  Training mode: %s", self.training)

    @property
    def dataset_lengths(self) -> np.ndarray:
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        return self._dataset_sampling_weights

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __str__(self) -> str:
        descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            descriptions.append(
                {
                    "Dataset": str(getattr(dataset, "data_path", dataset)),
                    "Sampling weight": float(weight),
                }
            )
        return yaml.dump({"DreamZero mixture dataset": descriptions})

    def _sample_step(self, index: int) -> tuple[Any, int]:
        if self.training:
            seed = safe_hash((self._epoch, index, self.seed))
            rng = np.random.default_rng(seed)
            dataset_index = int(
                rng.choice(len(self.datasets), p=self._dataset_sampling_weights)
            )
            dataset = self.datasets[dataset_index]
            sample_index = int(rng.integers(0, len(dataset)))
            return dataset, sample_index

        length_cumsum = np.cumsum(self._dataset_lengths)
        dataset_index = int(np.searchsorted(length_cumsum, index, side="right"))
        if dataset_index > 0:
            sample_index = index - int(length_cumsum[dataset_index - 1])
        else:
            sample_index = index
        return self.datasets[dataset_index], sample_index

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset, sample_index = self._sample_step(index)
        return dataset[sample_index]

    def __len__(self) -> int:
        if self.training:
            return int((self._dataset_lengths * self._dataset_sampling_weights).sum())
        return int(self._dataset_lengths.sum())


def build_dreamzero_mixture_dataset_from_spec(
    cfg: DictConfig,
    mixture_spec: Sequence[Any],
    eval_dataset: bool = False,
) -> tuple[DreamZeroLeRobotMixtureDataset, dict[str, int]]:
    """Build a mixture dataset and merged embodiment tag mapping from config."""
    from rlinf.data.datasets.dreamzero.data_transforms import (
        embodiment_tag_mapping_for_embodiment,
    )

    specs = _normalize_mixture_spec(mixture_spec)
    data_mixture: list[tuple[Any, float]] = []
    merged_tag_mapping: dict[str, int] = {}

    for spec in specs:
        paths = _resolve_dataset_paths(spec)
        weight = float(spec.get("weight", spec.get("dataset_weight", 1.0)))
        distribute_weights = bool(spec.get("distribute_weights", False))
        embodiment_tag = spec.get("embodiment_tag", cfg.actor.model.embodiment_tag)
        metadata_json_path = spec.get("metadata_json_path")
        tag_mapping_override = spec.get("embodiment_tag_mapping")

        datasets_for_spec = [
            build_single_dreamzero_lerobot_dataset(
                data_path=path,
                cfg=cfg,
                embodiment_tag=str(embodiment_tag),
                metadata_json_path=metadata_json_path,
                eval_dataset=eval_dataset,
            )
            for path in paths
        ]

        if distribute_weights and len(datasets_for_spec) > 1:
            lengths = np.array(
                [len(dataset) for dataset in datasets_for_spec], dtype=np.float64
            )
            relative_lengths = lengths / lengths.sum()
            for dataset, relative_length in zip(datasets_for_spec, relative_lengths):
                data_mixture.append((dataset, weight * float(relative_length)))
        else:
            per_dataset_weight = weight / len(datasets_for_spec)
            for dataset in datasets_for_spec:
                data_mixture.append((dataset, per_dataset_weight))

        merged_tag_mapping.update(
            embodiment_tag_mapping_for_embodiment(
                str(embodiment_tag),
                tag_mapping_override or cfg.actor.model.get("embodiment_tag_mapping"),
            )
        )

    mixture = DreamZeroLeRobotMixtureDataset(
        data_mixture=data_mixture,
        training=not eval_dataset,
        seed=int(cfg.actor.get("seed", cfg.data.get("seed", 42))),
    )
    return mixture, merged_tag_mapping
