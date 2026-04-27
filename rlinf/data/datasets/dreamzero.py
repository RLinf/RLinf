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

"""DreamZero SFT data utilities for LIBERO and OXE DROID (LeRobot).

Provides DreamZeroLiberoDataset / DreamZeroDroidDataset and DreamZeroCollator
that convert LeRobot data into the batch format expected by VLA.forward().

"""

import bisect
import json
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.utils.logging import get_logger

logger = get_logger()

# Shared cache across all dataset instances in the same process to avoid redundant
# parquet reads when DataLoader forks worker subprocesses.
_ADVANTAGE_CACHE: dict[str, dict[int, np.ndarray]] = {}

# Prompt template fed to the T5 tokenizer.
# {task} is the raw task string from tasks.jsonl, e.g. "put the white mug on the left plate".
LIBERO_PROMPT_TEMPLATE = (
    "A multi-view video shows that a robot {task} "
    "The video is split into two horizontal views: the left view shows "
    "the exterior camera and the right view shows the wrist camera. "
    "The robot {task}"
)
POSITIVE_GUIDANCE_PROMPT_TEMPLATE = "[POSITIVE][POSITIVE]\n" + LIBERO_PROMPT_TEMPLATE
NEGATIVE_GUIDANCE_PROMPT_TEMPLATE = "[NEGATIVE][NEGATIVE]\n" + LIBERO_PROMPT_TEMPLATE

# Prompt for DROID (aligned with groot DreamZero collate / rlinf_collate wording).
DROID_PROMPT_TEMPLATE = (
    "A multi-view video shows that a robot {task} "
    "The video is split into three views: The top view shows the camera view from the robot's wrist, "
    "the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view "
    "shows the camera view from the right exterior camera. During training, one of the two bottom exterior "
    "views may be a black screen (dropped view). The robot {task}"
)

# DreamZero projector id for OXE DROID (see rlinf.models.embodiment.dreamzero.transform_runtime).
DROID_EMBODIMENT_ID = 17


def _infer_droid_view_hw_from_model_cfg(model_cfg: Any) -> tuple[int, int]:
    """Infer DROID per-view resize target, compatible with WAN2.1/WAN2.2.

    Priority:
    1) Explicit override via cfg: dreamzero_droid_view_height/width
    2) Infer from model_path/config.json diffusion model signature
    3) Fallback to WAN2.1-style 176x320
    """
    h_override = model_cfg.get("dreamzero_droid_view_height", None)
    w_override = model_cfg.get("dreamzero_droid_view_width", None)
    if h_override is not None and w_override is not None:
        return int(h_override), int(w_override)

    model_path = Path(str(model_cfg.get("model_path", ""))).expanduser()
    cfg_path = model_path / "config.json"
    if not cfg_path.is_file():
        return (176, 320)

    try:
        with open(cfg_path) as f:
            cfg_json = json.load(f)
        ah_cfg = cfg_json.get("action_head_cfg", {}).get("config", {})
        dcfg = ah_cfg.get("diffusion_model_cfg", {})
        in_dim = int(dcfg.get("in_dim", -1))
        model_type = str(dcfg.get("model_type", "")).lower()
        frame_seqlen = int(dcfg.get("frame_seqlen", -1))
        if in_dim == 48 or model_type == "ti2v" or frame_seqlen == 50:
            return (160, 320)  # WAN2.2 default
    except Exception:
        logger.exception(
            "Failed to infer DROID per-view target from %s; fallback to 176x320.",
            cfg_path,
        )

    return (176, 320)


def _load_gear_stats(meta_dir: Path) -> dict[str, np.ndarray]:
    """Load normalization bounds from stats.json.

    Returns dict with keys like 'state.state' and 'action.actions',
    each mapping to {'q01': np.array, 'q99': np.array}.
    Falls back to min/max when q01/q99 are not available (matching
    DreamZero's LIBERO handling).

    The returned q01/q99 arrays have shape (D,) where D is the joint/action dim.
    Example: state q01/q99 both have shape (8,) for LIBERO's 8-DOF arm.
    """
    stats_path = meta_dir / "stats.json"
    if not stats_path.exists():
        return {}
    with open(stats_path) as f:
        raw = json.load(f)
    result = {}
    for key, val in raw.items():
        # Use q01/q99 if available; fall back to min/max
        lo = val.get("q01") or val.get("min")
        hi = val.get("q99") or val.get("max")
        if lo is not None and hi is not None:
            result[key] = {
                "q01": np.array(lo, dtype=np.float32),
                "q99": np.array(hi, dtype=np.float32),
            }
    return result


def q99_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Normalize x to [-1, 1] using q01/q99, matching DreamZero's Normalizer.

    Formula: 2 * (x - q01) / (q99 - q01) - 1, clamped to [-1, 1].
    Where q01 == q99, keep original value (avoid division by zero).

    Args:
        x:   (..., D)  raw values in physical units (radians, metres, etc.)
        q01: (D,)      lower bound (1st percentile or min)
        q99: (D,)      upper bound (99th percentile or max)
    Returns:
        (..., D)  values in [-1, 1]
    """
    denom = q99 - q01
    safe = denom != 0
    out = np.zeros_like(x)
    out[..., safe] = 2.0 * (x[..., safe] - q01[safe]) / denom[safe] - 1.0
    # For constant dimensions (denom==0) keep the raw value unchanged
    out[..., ~safe] = x[..., ~safe]
    return np.clip(out, -1.0, 1.0)


def _resolve_advantage_parquet_path(
    meta_dir: Path, advantage_parquet: str | None
) -> Path:
    adv_path = (
        Path(advantage_parquet)
        if advantage_parquet
        else (meta_dir / "advantages_test.parquet")
    )
    if not adv_path.is_absolute():
        adv_path = meta_dir / adv_path
    if not adv_path.exists():
        raise FileNotFoundError(
            f"CFG mode enabled but advantage parquet not found: {adv_path}"
        )
    return adv_path


def _load_advantage_map_cached(adv_path: Path) -> dict[int, np.ndarray]:
    """Load advantage parquet into {episode_index: bool_lookup[frame_index]} with caching."""
    import pandas as pd

    cache_key = str(adv_path.resolve())
    if cache_key in _ADVANTAGE_CACHE:
        return _ADVANTAGE_CACHE[cache_key]

    t0 = time.monotonic()
    df = pd.read_parquet(
        adv_path, columns=["episode_index", "frame_index", "advantage"]
    )
    required_cols = {"episode_index", "frame_index", "advantage"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Advantage parquet must contain columns {sorted(required_cols)}, got {list(df.columns)}"
        )

    advantage_map: dict[int, np.ndarray] = {}
    for ep_idx, ep_df in df.groupby("episode_index", sort=False):
        frame_idx = ep_df["frame_index"].to_numpy(dtype=np.int64)
        advantage = ep_df["advantage"].to_numpy(dtype=np.bool_)
        max_frame = int(frame_idx.max())
        lookup = np.zeros(max_frame + 1, dtype=np.bool_)
        lookup[frame_idx] = advantage
        advantage_map[int(ep_idx)] = lookup

    _ADVANTAGE_CACHE[cache_key] = advantage_map
    logger.info(
        "Loaded advantage parquet (%d rows, %d episodes) from %s in %.1fs",
        len(df),
        len(advantage_map),
        adv_path,
        time.monotonic() - t0,
    )
    return advantage_map


def _lookup_advantage_from_map(
    advantage_map: dict[int, np.ndarray],
    adv_path: Path | None,
    episode_index: int,
    frame_index: int,
    *,
    warn_oob: bool,
) -> bool:
    if episode_index not in advantage_map:
        raise KeyError(
            f"episode_index={episode_index} not found in advantage parquet {adv_path}"
        )
    values = advantage_map[int(episode_index)]
    fi = int(frame_index)
    if fi < 0 or fi >= len(values):
        if warn_oob:
            logger.warning(
                "frame_index=%d out of range [0, %d] for episode %d; clamping to boundary",
                fi,
                len(values) - 1,
                int(episode_index),
            )
        fi = min(max(fi, 0), len(values) - 1)
    return bool(values[fi])


def _probe_video_container_fps(video_path: Path) -> float | None:
    """Read average FPS from the video container (PyAV), not meta/info.json.

    Many RoboMIND / converted trees ship ``fps`` in info.json that does not match the
    muxed stream (e.g. meta 14 vs H.264 30). Using meta for ``index / fps`` decode times
    then violates lerobot's PTS tolerance against torchvision/pyav.
    """
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(video_path), mode="r") as container:
            streams = container.streams.video
            if not streams:
                return None
            st = streams[0]

            def _as_positive_fps(rate: Any) -> float | None:
                if rate is None:
                    return None
                try:
                    f = float(rate)
                except (TypeError, ValueError, ZeroDivisionError):
                    return None
                if 0.5 < f < 480.0:
                    return f
                return None

            for attr in ("average_frame_rate", "guessed_frame_rate", "average_rate"):
                f = _as_positive_fps(getattr(st, attr, None))
                if f is not None:
                    return f

            cc = getattr(st, "codec_context", None)
            if cc is not None:
                f = _as_positive_fps(getattr(cc, "framerate", None))
                if f is not None:
                    return f

            nb = int(getattr(st, "frames", 0) or 0)
            if nb > 0 and st.duration is not None and st.time_base is not None:
                try:
                    dur_s = float(st.duration * st.time_base)
                    if dur_s > 1e-3:
                        f = float(nb) / dur_s
                        if 0.5 < f < 480.0:
                            return f
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
    except OSError:
        return None
    except Exception:
        logger.debug("PyAV fps probe failed for %s", video_path, exc_info=True)
        return None
    return None


class DreamZeroLiberoDataset(Dataset):
    """Map LeRobot LIBERO samples to DreamZero training inputs.

    Supports both LeRobot v2 (images in parquet) and v3 (images in mp4).

    Frame sampling strategy (num_chunks=4, VIDEO_CHUNK_STRIDE=16):
        Each "chunk" samples 8 frames at offsets [0,2,4,6,8,10,12,14] relative to
        the chunk's start frame. The 4 chunks start at frames [0, 16, 32, 48], plus
        one extra boundary frame at offset 30+2=32... effectively 4*8+1 = 33 frames.

        Visual layout (frame indices relative to current timestep):
          chunk 0: frames  0, 2, 4, 6, 8,10,12,14
          chunk 1: frames 16,18,20,22,24,26,28,30
          chunk 2: frames 32,34,36,38,40,42,44,46
          chunk 3: frames 48,50,52,54,56,58,60,62
          extra:   frame  64          <- boundary anchor
          total:   33 frames

    State sampling: one frame per chunk start = frames [0, 16, 32, 48], giving 4 states.

    Action sampling: 64 consecutive frames starting at current timestep.
    """

    # 8 sub-frame offsets within each video chunk (stride-2 sampling for temporal coverage)
    VIDEO_CHUNK_OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14]
    # Distance in frames between consecutive chunk start points
    VIDEO_CHUNK_STRIDE = 16
    # Number of action steps per chunk (must equal VIDEO_CHUNK_STRIDE)
    ACTION_CHUNK_SIZE = 16

    def __init__(
        self,
        data_path: str | list[str],
        action_horizon: int = 64,  # Total action steps = ACTION_CHUNK_SIZE * num_chunks
        num_chunks: int = 4,  # Number of temporal chunks (matches dreamzero_num_chunks in config)
        max_action_dim: int = 32,  # Padding target for action dim (LIBERO uses 7, padded to 32)
        max_state_dim: int = 64,  # Padding target for state dim  (LIBERO uses 8, padded to 64)
        cfg_mode: bool = False,
        advantage_parquet: str | None = None,
        unconditional_prob: float = 0.3,
    ):
        if isinstance(data_path, (list, tuple)):
            if len(data_path) == 0:
                raise ValueError(
                    "DreamZeroLiberoDataset requires at least one data path."
                )
            data_path = data_path[0]
        self.data_path = str(data_path)
        self.num_chunks = num_chunks
        self.action_horizon = action_horizon
        self.video_frames_per_chunk = len(self.VIDEO_CHUNK_OFFSETS)  # 8
        # Total video frames returned per sample: 4 chunks * 8 frames + 1 boundary = 33
        self.video_horizon = self.video_frames_per_chunk * num_chunks + 1
        # One state snapshot per chunk start frame
        self.state_horizon = num_chunks  # 4
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.cfg_mode = bool(cfg_mode)
        self.advantage_parquet = advantage_parquet
        self.unconditional_prob = float(unconditional_prob)
        if not 0.0 <= self.unconditional_prob <= 1.0:
            raise ValueError(
                f"unconditional_prob must be in [0, 1], got {self.unconditional_prob}"
            )
        # Embodiment ID 21 is the LIBERO robot; selects per-robot weight matrices in the model
        self.embodiment_id = 21

        meta_dir = Path(self.data_path) / "meta"
        with open(meta_dir / "info.json") as f:
            info = json.load(f)
        self._fps = info.get("fps", 10)
        self._version = info.get("codebase_version", "v3.0")
        self._tasks = self._load_task_texts(meta_dir)

        # Load per-dimension normalization bounds: shape (D,) each
        gear_stats = _load_gear_stats(meta_dir)
        # Try both key naming conventions (v3: "state.state", v2: "state")
        st = gear_stats.get("state.state") or gear_stats.get("state") or {}
        ac = gear_stats.get("action.actions") or gear_stats.get("actions") or {}
        self._state_q01 = st.get("q01")  # shape (8,) for LIBERO
        self._state_q99 = st.get("q99")  # shape (8,)
        self._action_q01 = ac.get("q01")  # shape (7,) for LIBERO
        self._action_q99 = ac.get("q99")  # shape (7,)

        # Precompute frame index offsets for video, state, and action sampling
        # video_offsets: list of 33 relative frame offsets
        video_steps: list[int] = []
        for c in range(num_chunks):
            base = c * self.VIDEO_CHUNK_STRIDE
            video_steps.extend(base + o for o in self.VIDEO_CHUNK_OFFSETS)
        video_steps.append(video_steps[-1] + 2)  # extra boundary frame
        self._video_offsets = video_steps  # len=33
        # state_offsets: [0, 16, 32, 48] — one per chunk start
        self._state_offsets = [c * self.VIDEO_CHUNK_STRIDE for c in range(num_chunks)]
        # action_offsets: [0, 1, 2, ..., 63] — dense consecutive steps
        self._action_offsets = list(range(action_horizon))

        if self._version.startswith("v2"):
            self._init_v2()
        else:
            self._init_v3()

        self._advantage_map: dict[int, np.ndarray] = {}
        self._advantage_path: Path | None = None
        self._init_advantage_lookup(meta_dir)

    def _init_advantage_lookup(self, meta_dir: Path) -> None:
        if not self.cfg_mode:
            return
        adv_path = _resolve_advantage_parquet_path(meta_dir, self.advantage_parquet)
        self._advantage_map = _load_advantage_map_cached(adv_path)
        self._advantage_path = adv_path

    def _lookup_advantage(self, episode_index: int, frame_index: int) -> bool:
        return _lookup_advantage_from_map(
            self._advantage_map,
            self._advantage_path,
            episode_index,
            frame_index,
            warn_oob=True,
        )

    def _init_v3(self):
        """Initialize with LeRobot v3 dataset (mp4 videos).

        v3 stores images as mp4 video files per episode.
        delta_timestamps tells the lerobot loader which relative frame offsets to return.
        """
        import lerobot.datasets.lerobot_dataset as lerobot_dataset

        delta_timestamps = {
            "observation.images.image": [t / self._fps for t in self._video_offsets],
            "observation.images.wrist_image": [
                t / self._fps for t in self._video_offsets
            ],
            "observation.state": [t / self._fps for t in self._state_offsets],
            "action": [t / self._fps for t in self._action_offsets],
        }
        self.dataset = lerobot_dataset.LeRobotDataset(
            self.data_path,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )
        self._use_v2 = False

    def _init_v2(self):
        """Initialize with LeRobot v2 dataset (images stored as PNG bytes inside parquet).

        v2 stores each episode as a single parquet file. Each row is one timestep.
        Image columns contain raw PNG bytes (wrapped in a dict {'bytes': b'...'}).

        Parquet schema per row:
          image        struct{bytes: binary}  PNG-encoded (H=256, W=256, C=3)
          wrist_image  struct{bytes: binary}  PNG-encoded (H=256, W=256, C=3)
          state        list<float>            length 8  (joint angles)
          actions      list<float>            length 7  (joint deltas)
          task_index   int64
        """

        import pyarrow.parquet as pq

        data_root = Path(self.data_path) / "data"
        episodes_path = Path(self.data_path) / "meta" / "episodes.jsonl"

        self._episodes = []
        with open(episodes_path) as f:
            for line in f:
                if line.strip():
                    self._episodes.append(json.loads(line))

        # For each episode: count its frames and store its parquet path
        self._ep_frames = []
        self._ep_parquet_paths = []
        for ep in self._episodes:
            ep_idx = ep["episode_index"]
            chunk_idx = ep_idx // 1000
            pq_path = (
                data_root / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            )
            table = pq.read_table(pq_path)
            n_frames = len(table)
            self._ep_frames.append(n_frames)
            self._ep_parquet_paths.append(pq_path)

        # Cumulative frame count enables O(log N) episode lookup from global frame index
        self._cumulative = np.cumsum(self._ep_frames)
        self._total_frames = int(self._cumulative[-1])
        self._use_v2 = True
        # LRU-style parquet table cache (max 50 episodes) to avoid repeated I/O
        self._pq_cache: dict[int, "pq.Table"] = {}
        self._v2_img_keys = ("image", "wrist_image")
        self._v2_action_key = "actions"
        self._v2_state_key = "state"

    def _read_v2_episode(self, ep_idx: int):
        """Return cached pyarrow Table for episode ep_idx.

        Table shape: (n_frames, n_columns). Evicts oldest entry when cache exceeds 50.
        """
        if ep_idx not in self._pq_cache:
            import pyarrow.parquet as pq

            self._pq_cache[ep_idx] = pq.read_table(str(self._ep_parquet_paths[ep_idx]))
            if len(self._pq_cache) > 50:
                oldest = next(iter(self._pq_cache))
                del self._pq_cache[oldest]
        return self._pq_cache[ep_idx]

    def _decode_v2_image(self, cell) -> np.ndarray:
        """Decode one parquet image cell to a uint8 HWC numpy array.

        v2 image cells are pyarrow scalars containing PNG bytes in a dict.
        Output shape: (H=256, W=256, C=3), dtype=uint8.
        """
        from io import BytesIO

        from PIL import Image

        raw = cell.as_py()
        if isinstance(raw, dict):
            raw = raw.get("bytes", raw)
        if isinstance(raw, bytes):
            return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        return np.asarray(raw)

    def _get_v2_sample(self, idx: int) -> dict:
        """Retrieve one training sample from v2 parquet data by global frame index.

        Args:
            idx: global frame index in [0, total_frames)

        Returns dict with:
          observation.images.image      np.ndarray (33, 256, 256, 3) uint8
          observation.images.wrist_image np.ndarray (33, 256, 256, 3) uint8
          observation.state             np.ndarray (4, 8)   float32 — 4 chunk-start states
          action                        np.ndarray (64, 7)  float32 — 64-step action sequence
          task_index                    int

        Frame clamping: out-of-bounds offsets are clamped to [0, n_frames-1] (first_last padding).
        """
        # Locate which episode contains this global frame index
        ep_idx = int(np.searchsorted(self._cumulative, idx, side="right"))
        start = int(self._cumulative[ep_idx - 1]) if ep_idx > 0 else 0
        frame_in_ep = idx - start
        table = self._read_v2_episode(ep_idx)
        n = len(table)

        def clamp(offset):
            # Clamp to valid frame range — implements "first_last" boundary padding
            return min(max(frame_in_ep + offset, 0), n - 1)

        # Decode 33 video frames for main camera: shape (33, 256, 256, 3) uint8
        main_imgs = np.stack(
            [
                self._decode_v2_image(table.column("image")[clamp(o)])
                for o in self._video_offsets
            ],
            axis=0,
        )
        # Decode 33 video frames for wrist camera: shape (33, 256, 256, 3) uint8
        wrist_imgs = np.stack(
            [
                self._decode_v2_image(table.column("wrist_image")[clamp(o)])
                for o in self._video_offsets
            ],
            axis=0,
        )

        # State at 4 chunk-start frames: shape (4, 8) float32
        state_rows = [clamp(o) for o in self._state_offsets]
        state_col = table.column("state")
        state = np.array([state_col[r].as_py() for r in state_rows], dtype=np.float32)

        # Action sequence for 64 consecutive steps: shape (64, 7) float32
        action_rows = [clamp(o) for o in self._action_offsets]
        action_col = table.column("actions")
        action = np.array(
            [action_col[r].as_py() for r in action_rows], dtype=np.float32
        )

        task_idx = int(table.column("task_index")[frame_in_ep].as_py())

        return {
            "observation.images.image": main_imgs,
            "observation.images.wrist_image": wrist_imgs,
            "observation.state": state,
            "action": action,
            "task_index": task_idx,
            "episode_index": int(ep_idx),
            "frame_index": int(frame_in_ep),
        }

    @staticmethod
    def _load_task_texts(meta_dir: Path) -> dict[int, str]:
        """Build task_index -> instruction string mapping from tasks.jsonl or tasks.parquet."""
        import pandas as pd

        task_map: dict[int, str] = {}

        tasks_jsonl = meta_dir / "tasks.jsonl"
        if tasks_jsonl.exists():
            with open(tasks_jsonl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    task_id = int(entry.get("task_index", 0))
                    task_text = str(entry.get("task", ""))
                    task_map[task_id] = task_text
            if task_map:
                return task_map

        task_path = meta_dir / "tasks.parquet"
        if not task_path.exists():
            return {}

        tasks_df = pd.read_parquet(task_path)

        if list(tasks_df.columns) == ["task_index"] and tasks_df.index.dtype.kind in (
            "U",
            "O",
            "S",
        ):
            for text, row in tasks_df.iterrows():
                task_map[int(row["task_index"])] = str(text)
            return task_map

        text_col = None
        for candidate in ("task", "task_text", "language", "instruction", "prompt"):
            if candidate in tasks_df.columns:
                text_col = candidate
                break
        if text_col is None:
            cols = [c for c in tasks_df.columns if c != "task_index"]
            text_col = cols[0] if cols else None

        for _, row in tasks_df.iterrows():
            task_id = int(row.get("task_index", 0))
            if text_col is None:
                task_text = ""
            else:
                value = row.get(text_col, "")
                task_text = "" if value is None else str(value)
            task_map[task_id] = task_text
        return task_map

    @staticmethod
    def _to_hwc_uint8(image: Any) -> np.ndarray:
        """Convert image to HWC uint8, handling CHW float inputs.

        Input:  HWC uint8, or CHW float in [0,1]
        Output: (H, W, C) uint8 in [0, 255]
        """
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] == 3:
            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    def _build_video_grid(
        self, main_frames: np.ndarray, wrist_frames: np.ndarray
    ) -> np.ndarray:
        """Horizontally concatenate main and wrist views.

        Input:  main_frames  (T, H, W, 3) uint8   e.g. (33, 256, 256, 3)
                wrist_frames (T, H, W, 3) uint8   e.g. (33, 256, 256, 3)
        Output: (T, H, 2*W, 3) uint8              e.g. (33, 256, 512, 3)
                left half = main (exterior), right half = wrist camera
        """
        images = []
        for idx in range(main_frames.shape[0]):
            main = self._to_hwc_uint8(main_frames[idx])
            wrist = self._to_hwc_uint8(wrist_frames[idx])
            merged = np.concatenate([main, wrist], axis=1)  # concat along width dim
            images.append(merged)
        return np.stack(images, axis=0)

    @staticmethod
    def _augment_video(images: np.ndarray) -> np.ndarray:
        """Apply augmentations matching groot's LIBERO transform pipeline.

        Pipeline (same random params across all frames for temporal consistency):
          1. VideoCrop(scale=0.95):  random 95% crop, then resize back to original WxH
          2. VideoColorJitter:       brightness ±0.3, contrast ±0.4, saturation ±0.5, hue ±0.08

        Args:
            images: (T, H, W, C) uint8 in [0, 255]
        Returns:
            (T, H, W, C) uint8 in [0, 255]  — same spatial size, augmented appearance
        """
        T, H, W, C = images.shape

        import cv2

        # --- Step 1: Random crop (scale=0.95) then resize back to (H, W) ---
        crop_scale = 0.95
        crop_h, crop_w = int(H * crop_scale), int(W * crop_scale)
        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)
        images = images[:, top : top + crop_h, left : left + crop_w, :]
        # Resize each frame back to original resolution: (T, H, W, C)
        resized = np.stack(
            [
                cv2.resize(images[t], (W, H), interpolation=cv2.INTER_LINEAR)
                for t in range(T)
            ],
            axis=0,
        )

        result = resized.astype(np.float32)

        # --- Step 2: Brightness jitter — multiply all pixels by a scalar ---
        brightness = 1.0 + np.random.uniform(-0.3, 0.3)
        result = result * brightness

        # --- Step 3: Contrast jitter — scale deviation from per-frame mean ---
        # mean shape: (T, 1, 1, 1) so broadcast applies per-frame
        contrast = 1.0 + np.random.uniform(-0.4, 0.4)
        mean = result.mean(axis=(1, 2), keepdims=True)
        result = (result - mean) * contrast + mean

        # --- Step 4: Saturation jitter — lerp between grayscale and color ---
        saturation = 1.0 + np.random.uniform(-0.5, 0.5)
        # Luminance approximation: Y = 0.299R + 0.587G + 0.114B
        gray = (
            0.299 * result[..., 0:1]
            + 0.587 * result[..., 1:2]
            + 0.114 * result[..., 2:3]
        )
        result = (result - gray) * saturation + gray  # saturation=0 -> grayscale

        # --- Step 5: Hue shift (applied in HSV space) ---
        # hue_shift in [-0.08, 0.08] maps to [-14.4°, +14.4°] in OpenCV's [0,180] H range
        hue_shift = np.random.uniform(-0.08, 0.08)
        if abs(hue_shift) > 1e-4:
            for t in range(T):
                frame_bgr = cv2.cvtColor(
                    np.clip(result[t], 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV
                ).astype(np.float32)
                frame_bgr[..., 0] = (frame_bgr[..., 0] + hue_shift * 180.0) % 180.0
                result[t] = cv2.cvtColor(
                    frame_bgr.astype(np.uint8), cv2.COLOR_HSV2RGB
                ).astype(np.float32)

        return np.clip(result, 0, 255).astype(np.uint8)

    def __len__(self) -> int:
        return self._total_frames if self._use_v2 else len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one training sample for DreamZero.

        Output dict shapes (default config: num_chunks=4, action_horizon=64):
          images          (33, 256, 512, 3)  uint8   — side-by-side video (T, H, 2W, C)
          state           (4, 64)            float32 — normalized joint angles, zero-padded to max_state_dim
          state_mask      (4, 64)            bool    — True for real dimensions (first 8 cols for LIBERO)
          action          (64, 32)           float32 — normalized joint deltas, zero-padded to max_action_dim
          action_mask     (64, 32)           bool    — True for real dimensions (first 7 cols for LIBERO)
          embodiment_id   ()                 int64   — robot ID (21 = LIBERO)
          has_real_action ()                 bool    — always True for supervised data
          text            str                        — raw task instruction (tokenized in collator)
          (+ dummy zero tensors for lapa/segmentation fields used by other embodiments)
        """
        sample = self._get_v2_sample(idx) if self._use_v2 else self.dataset[idx]

        # ------------------------------------------------------------------ #
        # Images: concat main + wrist views, then augment                     #
        # Raw:     main  (33, 256, 256, 3) uint8                              #
        #          wrist (33, 256, 256, 3) uint8                              #
        # Merged:        (33, 256, 512, 3) uint8  (side-by-side)             #
        # After aug:     (33, 256, 512, 3) uint8  (same size, augmented)     #
        # ------------------------------------------------------------------ #
        main_frames = np.asarray(sample["observation.images.image"])
        wrist_frames = np.asarray(sample["observation.images.wrist_image"])
        if main_frames.ndim == 3:
            main_frames = main_frames[None, ...]  # (H,W,C) -> (1,H,W,C)
        if wrist_frames.ndim == 3:
            wrist_frames = wrist_frames[None, ...]
        images = self._build_video_grid(main_frames, wrist_frames).astype(np.uint8)
        images = self._augment_video(images)

        # ------------------------------------------------------------------ #
        # State: (4, 8) raw float32  ->  normalize  ->  pad to (4, 64)       #
        # Normalization: 2*(x-q01)/(q99-q01)-1, clamped to [-1,1]           #
        # Padding: zero-pad along dim=1 from 8 to max_state_dim=64           #
        # state_mask marks the first 8 columns as valid                       #
        # ------------------------------------------------------------------ #
        state = np.asarray(sample["observation.state"], dtype=np.float32)  # (4, 8)
        if state.ndim == 1:
            state = state[None, :]  # edge case: single timestep -> (1, 8)
        state = state[: self.state_horizon]
        if state.shape[0] < self.state_horizon:
            # "first_last" padding: repeat last row instead of filling zeros
            last = state[-1:]
            pad = np.repeat(last, self.state_horizon - state.shape[0], axis=0)
            state = np.concatenate([state, pad], axis=0)  # (4, 8)
        if self._state_q01 is not None and self._state_q99 is not None:
            state_dim_raw = state.shape[-1]
            sq01 = self._state_q01[:state_dim_raw]
            sq99 = self._state_q99[:state_dim_raw]
            state = q99_normalize(state, sq01, sq99)  # (4, 8) in [-1,1]
        # Zero-pad to (4, 64)
        state_pad = np.zeros((self.state_horizon, self.max_state_dim), dtype=np.float32)
        state_dim = min(state.shape[-1], self.max_state_dim)
        state_pad[:, :state_dim] = state[:, :state_dim]
        state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
        state_mask[:, :state_dim] = True  # mark real dims

        # ------------------------------------------------------------------ #
        # Action: (64, 7) raw float32  ->  normalize  ->  pad to (64, 32)    #
        # Same normalization and padding strategy as state                    #
        # action_mask marks the first 7 columns as valid                      #
        # ------------------------------------------------------------------ #
        action = np.asarray(sample["action"], dtype=np.float32)  # (64, 7)
        if action.ndim == 1:
            action = action[None, :]
        if action.shape[0] < self.action_horizon:
            # "first_last" padding: repeat last action step instead of zeros
            last = action[-1:]
            pad = np.repeat(last, self.action_horizon - action.shape[0], axis=0)
            action = np.concatenate([action, pad], axis=0)
        action = action[: self.action_horizon]  # (64, 7)
        if self._action_q01 is not None and self._action_q99 is not None:
            action_dim_raw = action.shape[-1]
            aq01 = self._action_q01[:action_dim_raw]
            aq99 = self._action_q99[:action_dim_raw]
            action = q99_normalize(action, aq01, aq99)  # (64, 7) in [-1,1]
        else:
            action = np.clip(action, -1.0, 1.0)
        # Zero-pad to (64, 32)
        action_pad = np.zeros(
            (self.action_horizon, self.max_action_dim), dtype=np.float32
        )
        action_dim = min(action.shape[-1], self.max_action_dim)
        action_pad[:, :action_dim] = action[:, :action_dim]
        action_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
        action_mask[:, :action_dim] = True  # mark real dims

        # Resolve task instruction string.
        task_text = sample.get("task")
        if task_text is None:
            task_idx = int(sample.get("task_index", 0))
            task_text = self._tasks.get(task_idx, "")
        task_text = str(task_text)

        prompt = task_text
        if self.cfg_mode:
            episode_index = sample.get("episode_index")
            frame_index = sample.get("frame_index")
            if episode_index is None or frame_index is None:
                raise KeyError(
                    "CFG mode requires episode_index and frame_index in each sample "
                    "to lookup per-frame advantage labels."
                )
            use_unconditional = np.random.random() < self.unconditional_prob
            if use_unconditional:
                prompt = LIBERO_PROMPT_TEMPLATE.format(task=task_text)
            else:
                advantage = self._lookup_advantage(int(episode_index), int(frame_index))
                if advantage:
                    prompt = POSITIVE_GUIDANCE_PROMPT_TEMPLATE.format(task=task_text)
                else:
                    prompt = NEGATIVE_GUIDANCE_PROMPT_TEMPLATE.format(task=task_text)

        return {
            # Core training fields
            "images": images,  # (33, 256, 512, 3) uint8
            "state": state_pad,  # (4, 64)           float32
            "state_mask": state_mask,  # (4, 64)           bool
            "action": action_pad,  # (64, 32)          float32
            "action_mask": action_mask,  # (64, 32)          bool
            "embodiment_id": np.int64(self.embodiment_id),  # ()  int64
            "has_real_action": np.bool_(True),  # ()                bool
            # Fields required by VLA.forward() signature but unused in SFT:
            "has_lapa_action": np.bool_(False),
            "is_cotrain_instance": np.bool_(False),
            "segmentation_target": np.zeros((2,), dtype=np.float32),
            "segmentation_target_mask": np.zeros((1,), dtype=np.float32),
            "lapa_action": np.zeros_like(action_pad),
            "lapa_action_mask": np.zeros_like(action_mask),
            # Text: raw string, tokenized by DreamZeroCollator
            "text": prompt,
        }


def _droid_default_state_action_slices() -> tuple[slice, slice, slice, slice]:
    """Slice ranges into convert_droid-style concatenated vectors.

    state: cartesian(6) + gripper(1) + joint(7)
    action: cartesian(6) + cartesian_vel(6) + gripper(1) + gripper_vel(1) + joint(7) + joint_vel(7)
    """
    st_joint = slice(7, 14)
    st_grip = slice(6, 7)
    ac_joint = slice(14, 21)
    ac_grip = slice(12, 13)
    return st_joint, st_grip, ac_joint, ac_grip


def _empty_slice() -> slice:
    return slice(0, 0)


def _infer_joint_grip_slices(
    names: list | dict | None, feature_dim: int | None = None
) -> tuple[slice, slice] | None:
    """Map concat-vector indices for joint_position and gripper_position from meta names."""
    if not names:
        return None
    # droid_100-style: names is a dict like {"motors": ["motor_0", ...]}.
    if isinstance(names, dict):
        if "joint_position" in names:
            joints = names.get("joint_position")
            n_joint = len(joints) if isinstance(joints, list) else 0
            if n_joint > 0:
                if "gripper_position" in names:
                    gripper = names.get("gripper_position")
                    n_grip = len(gripper) if isinstance(gripper, list) else 0
                    if n_grip > 0:
                        return slice(0, n_joint), slice(n_joint, n_joint + n_grip)
                return slice(0, n_joint), _empty_slice()
        motors = names.get("motors")
        if isinstance(motors, list) and len(motors) > 0:
            # Heuristic fallback:
            # - 8 dims -> (7 joint + 1 gripper), which matches official modality definition.
            # - otherwise keep joint-only.
            if len(motors) >= 8:
                return slice(0, 7), slice(7, 8)
            return slice(0, len(motors)), _empty_slice()
        return None

    # LeRobot convert_droid-style: names is a list of component keys with known widths.
    if all(isinstance(x, str) for x in names):
        plan = {
            "cartesian_position": 6,
            "cartesian_velocity": 6,
            "gripper_position": 1,
            "gripper_velocity": 1,
            "joint_position": 7,
            "joint_velocity": 7,
        }
        cursor = 0
        spans: dict[str, tuple[int, int]] = {}
        for key in names:
            if key not in plan:
                return None
            w = plan[key]
            spans[key] = (cursor, cursor + w)
            cursor += w
        if "joint_position" in spans and "gripper_position" in spans:
            j0, j1 = spans["joint_position"]
            g0, g1 = spans["gripper_position"]
            return slice(j0, j1), slice(g0, g1)
        # Joint-only representation (e.g. ["motor_0", ...] or ["joint_position"]).
        if "joint_position" in spans:
            j0, j1 = spans["joint_position"]
            return slice(j0, j1), _empty_slice()
        if feature_dim is not None and feature_dim > 0:
            return slice(0, feature_dim), _empty_slice()
        return None

    cursor = 0
    spans = {}
    for entry in names:
        if isinstance(entry, str):
            key = entry
            size = 1
        elif isinstance(entry, dict):
            key = str(entry.get("name", ""))
            shape = entry.get("shape")
            size = (
                int(np.prod(shape)) if shape is not None else int(entry.get("dim", 1))
            )
        else:
            continue
        spans[key] = (cursor, cursor + size)
        cursor += size
    need = ("joint_position", "gripper_position")
    if not all(k in spans for k in need):
        if "joint_position" in spans:
            j0, j1 = spans["joint_position"]
            return slice(j0, j1), _empty_slice()
        if feature_dim is not None and feature_dim > 0:
            return slice(0, feature_dim), _empty_slice()
        return None
    j0, j1 = spans["joint_position"]
    g0, g1 = spans["gripper_position"]
    return slice(j0, j1), slice(g0, g1)


def _safe_lang_text(value: Any, task_map: dict[int, str]) -> str:
    """Decode language field into a non-empty string when possible."""
    raw = value
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) == 0:
            return ""
        raw = raw[0]
        if hasattr(raw, "item"):
            raw = raw.item()
    if isinstance(raw, (int, np.integer)) and task_map:
        return str(task_map.get(int(raw), "")).strip()
    if raw is None:
        return ""
    return str(raw).strip()


def _discover_local_lerobot_episode_indices(
    root: Path, info: dict, allowed_episode_indices: set[int] | None = None
) -> list[int]:
    """Episode indices that have both parquet and all video files on disk.

    LeRobot 0.3.x otherwise checks ``range(total_episodes)`` from info.json; any missing
    file triggers Hub download (``get_safe_version``), which breaks offline machines even
    when ``data/`` already contains a subset of episodes.
    """
    root = root.resolve()
    data_root = root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"LeRobot dataset missing data/ directory: {data_root}."
            "Offline loading requires local data, videos, and meta to be aligned."
        )
    ep_re = re.compile(r"^episode_(\d+)\.parquet$")
    present: set[int] = set()
    for p in data_root.rglob("episode_*.parquet"):
        m = ep_re.match(p.name)
        if m:
            present.add(int(m.group(1)))
    if not present:
        raise FileNotFoundError(
            f"No episode_*.parquet found in {data_root}."
            "Please confirm data_path matches disk directory (e.g. data/chunk-000/episode_000000.parquet)."
        )
    chunks_size = int(info.get("chunks_size") or 1000)
    data_tmpl = info.get("data_path")
    video_tmpl = info.get("video_path")
    if not data_tmpl:
        raise ValueError("meta/info.json missing data_path")
    feats = info.get("features") or {}
    video_keys = [k for k, v in feats.items() if v.get("dtype") == "video"]
    complete: list[int] = []
    for ep_idx in sorted(present):
        ep_chunk = ep_idx // chunks_size
        rel_p = Path(data_tmpl.format(episode_chunk=ep_chunk, episode_index=ep_idx))
        if not (root / rel_p).is_file():
            continue
        if video_tmpl and video_keys:
            if not all(
                (
                    root
                    / Path(
                        video_tmpl.format(
                            episode_chunk=ep_chunk,
                            video_key=vk,
                            episode_index=ep_idx,
                        )
                    )
                ).is_file()
                for vk in video_keys
            ):
                continue
        complete.append(ep_idx)
    if allowed_episode_indices is not None:
        complete = [e for e in complete if e in allowed_episode_indices]
    if not complete:
        raise FileNotFoundError(
            f"Found parquet in {root}/data/, but no episode that satisfies "
            f"data_path and video_path (both in meta/episodes.jsonl) in info.json."
            f"({len(present)} parquet files on disk). Please fill in the corresponding videos/ or check if the paths match meta."
        )
    return complete


def _infer_droid_image_keys(info: dict) -> tuple[str, str, str]:
    feats = info.get("features") or {}
    candidates = [k for k in feats if k.startswith("observation.images.")]
    wrist_keys = [k for k in candidates if "wrist" in k]
    ext_keys = sorted([k for k in candidates if "wrist" not in k])

    def _lr_sort(keys: list[str]) -> list[str]:
        # Stable heuristic: prefer *_left* then *_right*, else lexicographic.
        def _rank(k: str) -> tuple[int, str]:
            kl = k.lower()
            if "left" in kl and "right" not in kl:
                return (0, k)
            if "right" in kl and "left" not in kl:
                return (1, k)
            return (2, k)

        return sorted(keys, key=_rank)

    # Common DROID layouts:
    # - 2 exterior + 1 wrist (canonical)
    # - 1 exterior (often top_cam) + 2 wrist (left/right). For DreamZero's 3-view grid, we can
    #   use exterior as the "wrist/top" slot and put the two wrists in the bottom tiles.
    if len(wrist_keys) == 1 and len(ext_keys) == 2:
        exts = _lr_sort(ext_keys)
        return exts[0], exts[1], wrist_keys[0]
    if len(wrist_keys) == 2 and len(ext_keys) == 1:
        wrists = _lr_sort(wrist_keys)
        return wrists[0], wrists[1], ext_keys[0]

    raise ValueError(
        "DROID dataset needs either (2 exterior + 1 wrist) or (1 exterior + 2 wrist) "
        "under observation.images.*; "
        f"found wrist={sorted(wrist_keys)} exterior={sorted(ext_keys)} candidates={sorted(candidates)}"
    )


class DreamZeroDroidDataset(Dataset):
    """Map LeRobot OXE DROID samples to DreamZero SFT tensors (layout matches groot OXE_DROID).

    Video: 33 frames (delta 0..32), stacked into a 2x2 layout (H*2 x W*2),
    matching DreamTransform._prepare_video for EmbodimentTag.OXE_DROID.

    State: one timestep, joint (7) + gripper (1) -> padded to (1, max_state_dim).

    Action: 24 steps (delta 0..23), joint + optional gripper slices -> padded to (24, max_action_dim).

    Optional relative joint targets: action_joint := action_joint - ref_joint (current state),
    same convention as groot LeRobotSingleDataset relative_action on ``joint_position``.
    """

    def __init__(
        self,
        data_path: str | list[str],
        lazy_load: bool = True,
        action_horizon: int = 24,
        num_video_frames: int = 33,
        total_action_steps: int = 96,
        state_horizon: int = 4,
        state_step: int = 24,
        max_action_dim: int = 32,
        max_state_dim: int = 64,
        cfg_mode: bool = False,
        advantage_parquet: str | None = None,
        unconditional_prob: float = 0.3,
        relative_action: bool = True,
        relative_action_keys: list[str] | None = None,
        droid_view_hw: tuple[int, int] = (176, 320),
        pq_cache_max_episodes: int = 128,
        video_tolerance_s: float = 0.1,
    ):
        if isinstance(data_path, (list, tuple)):
            if len(data_path) == 0:
                raise ValueError(
                    "DreamZeroDroidDataset requires at least one data path."
                )
            data_path = data_path[0]
        self.data_path = str(data_path)
        self.lazy_load = bool(lazy_load)
        self.action_horizon = int(action_horizon)
        self.total_action_steps = int(total_action_steps)
        self.num_video_frames = int(num_video_frames)
        self.state_horizon = int(state_horizon)
        self.state_step = int(state_step)
        self.max_action_dim = int(max_action_dim)
        self.max_state_dim = int(max_state_dim)
        self.cfg_mode = bool(cfg_mode)
        self.advantage_parquet = advantage_parquet
        self.unconditional_prob = float(unconditional_prob)
        self.relative_action = bool(relative_action)
        self.relative_action_keys = list(relative_action_keys or ["joint_position"])
        self.embodiment_id = DROID_EMBODIMENT_ID
        self._target_view_hw = (int(droid_view_hw[0]), int(droid_view_hw[1]))

        meta_dir = Path(self.data_path) / "meta"
        with open(meta_dir / "info.json") as f:
            info = json.load(f)
        self._fps = float(info.get("fps", 15))
        self._version = str(info.get("codebase_version", "v2.0"))
        feats = info.get("features") or {}
        self._raw_hw = feats.get("observation.images.exterior_image_1_left", {}).get(
            "shape", [180, 320, 3]
        )[:2]
        self._raw_hw = (int(self._raw_hw[0]), int(self._raw_hw[1]))

        st_feat = feats.get("observation.state") or {}
        act_feat = feats.get("action") or {}
        st_dim = int((st_feat.get("shape") or [0])[0] or 0)
        ac_dim = int((act_feat.get("shape") or [0])[0] or 0)
        st_slices = _infer_joint_grip_slices(st_feat.get("names"), st_dim)
        ac_slices = _infer_joint_grip_slices(act_feat.get("names"), ac_dim)
        if st_slices is None or ac_slices is None:
            if st_dim > 0 and ac_dim > 0 and st_dim <= 8 and ac_dim <= 8:
                self._st_j, self._st_g = slice(0, st_dim), _empty_slice()
                self._ac_j, self._ac_g = slice(0, ac_dim), _empty_slice()
                logger.info(
                    "DROID: using joint-only slices inferred from feature dims "
                    "(state=%d, action=%d).",
                    st_dim,
                    ac_dim,
                )
            else:
                self._st_j, self._st_g, self._ac_j, self._ac_g = (
                    _droid_default_state_action_slices()
                )
                logger.info(
                    "DROID: using default convert_droid slices "
                    "(meta feature names missing joint_position/gripper_position)."
                )
        else:
            self._st_j, self._st_g = st_slices
            self._ac_j, self._ac_g = ac_slices

        self._img_key_left, self._img_key_right, self._img_key_wrist = (
            _infer_droid_image_keys(info)
        )
        self._tasks = DreamZeroLiberoDataset._load_task_texts(meta_dir)

        gear_stats = _load_gear_stats(meta_dir)
        st_full = gear_stats.get("observation.state") or {}
        ac_full = gear_stats.get("action") or {}
        self._full_state_q01 = st_full.get("q01")
        self._full_state_q99 = st_full.get("q99")
        self._full_action_q01 = ac_full.get("q01")
        self._full_action_q99 = ac_full.get("q99")

        video_offsets = list(range(self.num_video_frames))
        state_offsets = [i * self.state_step for i in range(self.state_horizon)]
        action_offsets = list(range(self.total_action_steps))

        if self.lazy_load:
            # --- Map-style, episode-local loading (DreamZero official behavior) ---
            # Avoid `datasets.load_dataset("parquet")` which materializes a huge Arrow dataset and
            # is the source of "Generating train split ..." + OOM in multi-process jobs.
            self._root = Path(self.data_path).resolve()
            if not self._root.exists():
                raise FileNotFoundError(
                    f"DROID data_path must be a local path, got: {self.data_path}"
                )
            self._meta_dir = meta_dir
            self._info = info
            self._chunks_size = int(info.get("chunks_size") or 1000)
            self._data_tmpl = str(info.get("data_path") or "")
            self._video_tmpl = str(info.get("video_path") or "")
            if not self._data_tmpl:
                raise ValueError("meta/info.json missing data_path")

            feats = info.get("features") or {}
            self._video_keys = [
                k for k, v in feats.items() if v.get("dtype") == "video"
            ]

            # Episodes list: only those that exist on disk and in meta/episodes.jsonl.
            meta_episode_indices: set[int] = set()
            episode_lengths: dict[int, int] = {}
            with open(meta_dir / "episodes.jsonl") as _epf:
                for _line in _epf:
                    _line = _line.strip()
                    if not _line:
                        continue
                    obj = json.loads(_line)
                    ep_idx = int(obj.get("episode_index", 0))
                    meta_episode_indices.add(ep_idx)
                    # Try several common keys.
                    for k in ("episode_length", "length", "num_frames", "num_steps"):
                        if k in obj and obj[k] is not None:
                            try:
                                episode_lengths[ep_idx] = int(obj[k])
                                break
                            except Exception:
                                pass

            self._episodes: list[int] = _discover_local_lerobot_episode_indices(
                self._root, info, allowed_episode_indices=meta_episode_indices
            )
            if not self._episodes:
                raise FileNotFoundError(
                    f"No local episodes found under {self._root} that match meta/episodes.jsonl."
                )
            logger.info(
                "DROID lazy map-style: using %d local episodes (info total_episodes=%s).",
                len(self._episodes),
                info.get("total_episodes"),
            )

            # Episode -> length (fallback to parquet metadata when missing).
            self._episode_lengths: list[int] = []
            for ep_idx in self._episodes:
                n = episode_lengths.get(int(ep_idx))
                if n is None or n <= 0:
                    n = self._infer_episode_length_from_parquet(int(ep_idx))
                self._episode_lengths.append(int(n))

            # Prefix sum offsets for global indexing (idx -> (episode, frame_in_episode)).
            self._episode_starts: list[int] = [0]
            total = 0
            for n in self._episode_lengths:
                total += int(n)
                self._episode_starts.append(total)
            self._total_frames = int(total)

            # Cached episode parquet tables (LRU-ish using insertion order).
            self._pq_cache: "OrderedDict[int, Any]" = OrderedDict()
            self._pq_cache_max_episodes = max(1, int(pq_cache_max_episodes))

            # Per-mp4 decode FPS: info.json ``fps`` often disagrees with the muxed stream (e.g. 14 vs 30).
            self._video_decode_fps_cache: "OrderedDict[str, float]" = OrderedDict()
            self._video_decode_fps_cache_max = 512

            # Offsets used for sampling each modality around the base frame.
            self._video_offsets = np.asarray(video_offsets, dtype=np.int64)
            self._state_offsets = np.asarray(state_offsets, dtype=np.int64)
            self._action_offsets = np.asarray(action_offsets, dtype=np.int64)
            _tol = float(video_tolerance_s)
            if _tol <= 0.0:
                raise ValueError(
                    f"video_tolerance_s must be positive, got {video_tolerance_s!r}"
                )
            self._video_tolerance_s = _tol
            self._video_backend = "pyav"
            self._parquet_columns = [
                "timestamp",
                # Some datasets store state under a struct column "observation" with field "state"
                # instead of a flattened "observation.state" column.
                "observation",
                "observation.state",
                "action",
                "task",
                "task_index",
                "annotation.language.language_instruction",
                "annotation.language.language_instruction_2",
                "annotation.language.language_instruction_3",
            ]
        else:
            # --- LeRobotDataset loading (legacy behavior, non-map-style) ---
            from importlib.metadata import version as _pkg_version

            import lerobot.datasets.lerobot_dataset as lerobot_dataset
            from packaging.version import Version as _PkgVersion

            delta_timestamps = {
                self._img_key_left: [t / self._fps for t in video_offsets],
                self._img_key_right: [t / self._fps for t in video_offsets],
                self._img_key_wrist: [t / self._fps for t in video_offsets],
                "observation.state": [t / self._fps for t in state_offsets],
                "action": [t / self._fps for t in action_offsets],
            }
            data_path_obj = Path(self.data_path)
            if not data_path_obj.exists():
                raise FileNotFoundError(
                    f"DROID data_path must be a local path, got: {self.data_path}"
                )

            # lerobot 0.3.x: first arg is repo_id (name only), not a filesystem path. Local datasets must use
            # repo_id=<folder name> + root=<absolute path to that folder> so metadata loads from disk and
            # never hits Hub (get_safe_version). lerobot 0.4.x rejects codebase_version v2.0 in info.json.
            if _PkgVersion(_pkg_version("lerobot")) >= _PkgVersion("0.4.0"):
                raise RuntimeError(
                    "DreamZeroDroidDataset requires lerobot<0.4 for datasets with "
                    "meta/info.json codebase_version v2.0 (e.g. DreamZero-DROID-Data-mini). "
                    "Install lerobot==0.3.3 (see requirements/embodied/models/dreamzero.txt)."
                )
            # Only pass episodes that exist locally; info.json total_episodes can be full-dataset
            # size while this tree is a subset — otherwise lerobot asserts all files and hits Hub.
            meta_episode_indices: set[int] = set()
            with open(meta_dir / "episodes.jsonl") as _epf:
                for _line in _epf:
                    _line = _line.strip()
                    if not _line:
                        continue
                    meta_episode_indices.add(int(json.loads(_line)["episode_index"]))
            local_episodes = _discover_local_lerobot_episode_indices(
                data_path_obj, info, allowed_episode_indices=meta_episode_indices
            )
            logger.info(
                "DROID LeRobot: loading %d episodes present under data/ (info total_episodes=%s).",
                len(local_episodes),
                info.get("total_episodes"),
            )
            self.dataset = lerobot_dataset.LeRobotDataset(
                data_path_obj.name,
                root=str(data_path_obj.resolve()),
                episodes=local_episodes,
                delta_timestamps=delta_timestamps,
                video_backend="pyav",
            )

        self._advantage_map = {}
        self._advantage_path = None
        self._init_advantage_lookup(meta_dir)

    def _infer_episode_length_from_parquet(self, episode_index: int) -> int:
        import pyarrow.parquet as pq

        p = self._get_parquet_path(episode_index)
        md = pq.read_metadata(str(p))
        n = int(md.num_rows)
        if n <= 0:
            raise ValueError(f"episode_{episode_index:06d}.parquet has 0 rows: {p}")
        return n

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
        if not self._video_tmpl:
            raise FileNotFoundError("meta/info.json missing video_path")
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
        """FPS used for ``row_index / fps`` → decode timestamp; prefers container over meta."""
        key = str(video_path.resolve())
        if key in self._video_decode_fps_cache:
            fps = self._video_decode_fps_cache.pop(key)
            self._video_decode_fps_cache[key] = fps
            return fps
        probed = _probe_video_container_fps(video_path)
        fps = float(probed) if probed is not None else float(self._fps)
        if fps <= 0.0:
            fps = float(self._fps)
        self._video_decode_fps_cache[key] = fps
        if len(self._video_decode_fps_cache) > self._video_decode_fps_cache_max:
            self._video_decode_fps_cache.popitem(last=False)
        return fps

    def _get_episode_table(self, episode_index: int):
        # Small LRU cache keyed by episode_index.
        episode_index = int(episode_index)
        if episode_index in self._pq_cache:
            tbl = self._pq_cache.pop(episode_index)
            self._pq_cache[episode_index] = tbl
            return tbl

        import pyarrow.parquet as pq

        p = self._get_parquet_path(episode_index)
        # Only load columns we might touch; keeps per-episode memory down.
        # Some datasets do not contain optional columns like "task" (they might only have
        # task_index + tasks.jsonl), so intersect with the on-disk schema to avoid ArrowInvalid.
        # IMPORTANT: use Arrow schema column names, not parquet metadata schema names.
        # For list/fixed_size_list columns, `read_metadata(...).schema.names` can degrade into
        # repeated "element" entries, which breaks column projection decisions.
        schema_names = list(pq.read_schema(str(p)).names)
        schema_set = set(schema_names)
        cols = [c for c in self._parquet_columns if c in schema_set]
        # Ensure required columns are never accidentally dropped by projection logic.
        for required in ("timestamp", "action", "observation.state"):
            if required in schema_set and required not in cols:
                cols.append(required)
        # Keep deterministic order (pyarrow accepts list; order doesn't matter but helps debugging).
        cols = [c for i, c in enumerate(cols) if c not in cols[:i]]
        tbl = pq.read_table(str(p), columns=cols)
        self._pq_cache[episode_index] = tbl
        if len(self._pq_cache) > self._pq_cache_max_episodes:
            self._pq_cache.popitem(last=False)
        return tbl

    @staticmethod
    def _pick_state_column_from_schema(schema_names: list[str]) -> str | None:
        # Prefer the canonical flattened name.
        if "observation.state" in schema_names:
            return "observation.state"
        # Common struct flattening patterns in parquet.
        candidates = [
            n
            for n in schema_names
            if ("state" in n.split(".")[-1])
            and ("observation" in n.split(".")[0] or n.startswith("observation"))
        ]
        return candidates[0] if candidates else None

    @staticmethod
    def _clip_indices(indices: np.ndarray, length: int) -> np.ndarray:
        if length <= 0:
            return np.zeros_like(indices, dtype=np.int64)
        return np.clip(indices.astype(np.int64), 0, int(length) - 1)

    @staticmethod
    def _col_exists(table, name: str) -> bool:
        try:
            # `pyarrow.Table` should expose `column_names`; some derived objects can behave
            # oddly under column projection. Check schema too for robustness.
            if hasattr(table, "column_names") and name in table.column_names:
                return True
            if hasattr(table, "schema") and hasattr(table.schema, "names"):
                return name in table.schema.names
            return False
        except Exception:
            return False

    @staticmethod
    def _read_list_column(
        table, name: str, indices: np.ndarray, dtype=np.float32
    ) -> np.ndarray:
        col = table.column(name)
        out = []
        for i in indices.tolist():
            v = col[int(i)].as_py()
            out.append(v)
        return np.asarray(out, dtype=dtype)

    @staticmethod
    def _read_struct_list_field(
        table,
        struct_col: str,
        field: str,
        indices: np.ndarray,
        dtype=np.float32,
    ) -> np.ndarray:
        """Read a list/fixed_size_list field from a struct column."""
        col = table.column(struct_col)
        try:
            arr = (
                col.chunk(0)
                if hasattr(col, "num_chunks") and col.num_chunks > 0
                else col
            )
            # `arr` is a StructArray; extract the field as an Array-like
            field_arr = arr.field(field)
        except Exception as e:
            raise KeyError(f"Missing struct field {struct_col}.{field}: {e}") from e

        out = []
        for i in indices.tolist():
            out.append(field_arr[int(i)].as_py())
        return np.asarray(out, dtype=dtype)

    @staticmethod
    def _read_scalar_column(table, name: str, index: int) -> Any:
        return table.column(name)[int(index)].as_py()

    def _init_advantage_lookup(self, meta_dir: Path) -> None:
        if not self.cfg_mode:
            return
        adv_path = _resolve_advantage_parquet_path(meta_dir, self.advantage_parquet)
        self._advantage_map = _load_advantage_map_cached(adv_path)
        self._advantage_path = adv_path

    def _lookup_advantage(self, episode_index: int, frame_index: int) -> bool:
        return _lookup_advantage_from_map(
            self._advantage_map,
            self._advantage_path,
            episode_index,
            frame_index,
            warn_oob=False,
        )

    def _slice_norm(
        self,
        x: np.ndarray,
        full_q01: np.ndarray | None,
        full_q99: np.ndarray | None,
        sl: slice,
    ) -> np.ndarray:
        if full_q01 is None or full_q99 is None:
            return np.clip(x.astype(np.float32), -1.0, 1.0)
        q01 = full_q01[sl].astype(np.float32)
        q99 = full_q99[sl].astype(np.float32)
        return q99_normalize(x.astype(np.float32), q01, q99)

    def _build_droid_frame_grid(
        self, left: np.ndarray, right: np.ndarray, wrist: np.ndarray
    ) -> np.ndarray:
        """Stack three views into (H*2, W*2) per timestep (groot OXE_DROID layout)."""
        import cv2

        t = left.shape[0]
        out = []
        for i in range(t):
            a = DreamZeroLiberoDataset._to_hwc_uint8(left[i])
            b = DreamZeroLiberoDataset._to_hwc_uint8(right[i])
            w = DreamZeroLiberoDataset._to_hwc_uint8(wrist[i])
            # Per-view size is inferred from model config (WAN2.1/WAN2.2) or manual override.
            target_hw = self._target_view_hw
            if a.shape[:2] != target_hw:
                a = cv2.resize(a, target_hw[::-1], interpolation=cv2.INTER_LINEAR)
            if b.shape[:2] != target_hw:
                b = cv2.resize(b, target_hw[::-1], interpolation=cv2.INTER_LINEAR)
            if w.shape[:2] != target_hw:
                w = cv2.resize(w, target_hw[::-1], interpolation=cv2.INTER_LINEAR)
            h0, w0 = a.shape[0], a.shape[1]
            hb, wb = h0 * 2, w0 * 2
            canvas = np.zeros((hb, wb, 3), dtype=np.uint8)
            wrist_wide = np.repeat(w, 2, axis=1)
            canvas[:h0, :] = wrist_wide
            canvas[h0:, :w0] = a
            canvas[h0:, w0:] = b
            out.append(canvas)
        return np.stack(out, axis=0)

    def __len__(self) -> int:
        if self.lazy_load:
            return int(self._total_frames)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.lazy_load:
            # Map global frame index -> (episode_index, frame_in_episode)
            if idx < 0 or idx >= self._total_frames:
                raise IndexError(
                    f"Index {idx} out of range for dataset of len {self._total_frames}"
                )
            ep_pos = bisect.bisect_right(self._episode_starts, int(idx)) - 1
            ep_pos = max(0, min(ep_pos, len(self._episodes) - 1))
            ep_start = self._episode_starts[ep_pos]
            frame_in_ep = int(idx) - int(ep_start)
            episode_index = int(self._episodes[ep_pos])
            ep_len = int(self._episode_lengths[ep_pos])

            table = self._get_episode_table(episode_index)

            # Indices per modality (padded by clamping).
            video_idx = self._clip_indices(frame_in_ep + self._video_offsets, ep_len)
            state_idx = self._clip_indices(frame_in_ep + self._state_offsets, ep_len)
            action_idx = self._clip_indices(frame_in_ep + self._action_offsets, ep_len)

            # Decode times must track **muxed PTS**. (1) Parquet ``timestamp`` can disagree with
            # PTS (fixed earlier). (2) meta ``fps`` can disagree with the stream (e.g. info 14 vs
            # H.264 30); use PyAV-probed FPS per file, cached.
            left_p = self._get_video_path(episode_index, self._img_key_left)
            right_p = self._get_video_path(episode_index, self._img_key_right)
            wrist_p = self._get_video_path(episode_index, self._img_key_wrist)
            fl, fr, fw = (
                self._decode_fps_for_video_file(left_p),
                self._decode_fps_for_video_file(right_p),
                self._decode_fps_for_video_file(wrist_p),
            )
            vi = video_idx.tolist()
            ts_left = [float(int(i)) / fl for i in vi]
            ts_right = [float(int(i)) / fr for i in vi]
            ts_wrist = [float(int(i)) / fw for i in vi]

            from lerobot.datasets.video_utils import decode_video_frames

            left = decode_video_frames(
                left_p,
                ts_left,
                tolerance_s=self._video_tolerance_s,
                backend=self._video_backend,
            )
            right = decode_video_frames(
                right_p,
                ts_right,
                tolerance_s=self._video_tolerance_s,
                backend=self._video_backend,
            )
            wrist = decode_video_frames(
                wrist_p,
                ts_wrist,
                tolerance_s=self._video_tolerance_s,
                backend=self._video_backend,
            )
            sample: dict[str, Any] = {
                self._img_key_left: left,
                self._img_key_right: right,
                self._img_key_wrist: wrist,
                "episode_index": episode_index,
                "frame_index": frame_in_ep,
            }

            # Optional language/task fields (base frame only).
            for k in (
                "task",
                "task_index",
                "annotation.language.language_instruction",
                "annotation.language.language_instruction_2",
                "annotation.language.language_instruction_3",
            ):
                if self._col_exists(table, k):
                    sample[k] = self._read_scalar_column(table, k, frame_in_ep)

            # Vector modalities sampled with offsets.
            if self._col_exists(table, "observation.state"):
                state_arr = self._read_list_column(
                    table, "observation.state", state_idx, dtype=np.float32
                )
            elif self._col_exists(table, "observation"):
                # Struct column case: observation: struct<..., state: fixed_size_list<...>, ...>
                state_arr = self._read_struct_list_field(
                    table, "observation", "state", state_idx, dtype=np.float32
                )
            else:
                # Fallback: schema-driven column selection (handles variants like observation/state flattening)
                import pyarrow.parquet as pq

                p = self._get_parquet_path(episode_index)
                schema_names = list(pq.read_schema(str(p)).names)
                picked = self._pick_state_column_from_schema(schema_names)
                if picked is not None and picked != "observation":
                    tbl2 = pq.read_table(str(p), columns=[picked])
                    state_arr = self._read_list_column(
                        tbl2, picked, state_idx, dtype=np.float32
                    )
                else:
                    raise KeyError(
                        "episode parquet missing state; expected 'observation.state' or struct 'observation.state'. "
                        f"episode={episode_index} parquet={p} loaded_cols={getattr(table, 'column_names', None)} "
                        f"schema_cols={schema_names[:50]} (showing first 50)"
                    )
            if not self._col_exists(table, "action"):
                raise KeyError("episode parquet missing 'action' column")
            sample["observation.state"] = state_arr
            sample["action"] = self._read_list_column(
                table, "action", action_idx, dtype=np.float32
            )
        else:
            sample = self.dataset[idx]

        left = np.asarray(sample[self._img_key_left])
        right = np.asarray(sample[self._img_key_right])
        wrist = np.asarray(sample[self._img_key_wrist])
        for name, arr in (("left", left), ("right", right), ("wrist", wrist)):
            if arr.ndim == 3:
                arr = arr[None, ...]
            if name == "left":
                left = arr
            elif name == "right":
                right = arr
            else:
                wrist = arr

        images = self._build_droid_frame_grid(left, right, wrist).astype(np.uint8)
        images = DreamZeroLiberoDataset._augment_video(images)

        full_state = np.asarray(sample["observation.state"], dtype=np.float32)
        if full_state.ndim == 1:
            full_state = full_state[None, ...]
        if full_state.shape[0] < self.state_horizon:
            last = full_state[-1:]
            pad = np.repeat(last, self.state_horizon - full_state.shape[0], axis=0)
            full_state = np.concatenate([full_state, pad], axis=0)
        full_state = full_state[: self.state_horizon]
        joint_s = full_state[:, self._st_j].astype(np.float32)
        grip_s = full_state[:, self._st_g].astype(np.float32)

        full_action = np.asarray(sample["action"], dtype=np.float32)
        if full_action.ndim == 1:
            full_action = full_action[None, :]
        if full_action.ndim == 2 and full_action.shape[0] < self.total_action_steps:
            last = full_action[-1:]
            pad = np.repeat(
                last, self.total_action_steps - full_action.shape[0], axis=0
            )
            full_action = np.concatenate([full_action, pad], axis=0)
        full_action = full_action[: self.total_action_steps]

        joint_a = full_action[:, self._ac_j].astype(np.float32)
        grip_a = full_action[:, self._ac_g].astype(np.float32)

        if self.relative_action and "joint_position" in self.relative_action_keys:
            # Use the first state token as reference for all action steps.
            joint_a = joint_a - joint_s[0:1, :]

        joint_s_n = self._slice_norm(
            joint_s, self._full_state_q01, self._full_state_q99, self._st_j
        )
        grip_s_n = self._slice_norm(
            grip_s, self._full_state_q01, self._full_state_q99, self._st_g
        )
        state_norm = np.concatenate([joint_s_n, grip_s_n], axis=-1)

        joint_a_n = self._slice_norm(
            joint_a, self._full_action_q01, self._full_action_q99, self._ac_j
        )
        grip_a_n = self._slice_norm(
            grip_a.reshape(self.total_action_steps, -1),
            self._full_action_q01,
            self._full_action_q99,
            self._ac_g,
        )
        action_norm = np.concatenate([joint_a_n, grip_a_n], axis=-1)

        state_pad = np.zeros((self.state_horizon, self.max_state_dim), dtype=np.float32)
        sd = min(state_norm.shape[-1], self.max_state_dim)
        state_pad[:, :sd] = state_norm[:, :sd]
        state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
        state_mask[:, :sd] = True

        action_pad = np.zeros(
            (self.total_action_steps, self.max_action_dim), dtype=np.float32
        )
        ad = min(action_norm.shape[-1], self.max_action_dim)
        action_pad[:, :ad] = action_norm[:, :ad]
        action_mask = np.zeros(
            (self.total_action_steps, self.max_action_dim), dtype=bool
        )
        action_mask[:, :ad] = True

        task_text = sample.get("task")
        if task_text is None:
            task_idx = int(sample.get("task_index", 0))
            task_text = self._tasks.get(task_idx, "")
        task_text = str(task_text)
        lang_keys = [
            "annotation.language.language_instruction",
            "annotation.language.language_instruction_2",
            "annotation.language.language_instruction_3",
        ]
        lang_candidates: list[str] = []
        for lang_key in lang_keys:
            if lang_key not in sample:
                continue
            text = _safe_lang_text(sample[lang_key], self._tasks)
            if text:
                lang_candidates.append(text)
        # Match official training behavior: randomly sample one instruction if multiple are available.
        if lang_candidates:
            task_text = str(np.random.choice(lang_candidates))
        elif not task_text.strip():
            task_text = ""

        prompt = task_text
        if self.cfg_mode:
            episode_index = sample.get("episode_index")
            frame_index = sample.get("frame_index")
            if episode_index is None or frame_index is None:
                raise KeyError(
                    "CFG mode requires episode_index and frame_index in each sample."
                )
            use_unconditional = np.random.random() < self.unconditional_prob
            if use_unconditional:
                prompt = DROID_PROMPT_TEMPLATE.format(task=task_text)
            else:
                advantage = self._lookup_advantage(int(episode_index), int(frame_index))
                if advantage:
                    prompt = "[POSITIVE][POSITIVE]\n" + DROID_PROMPT_TEMPLATE.format(
                        task=task_text
                    )
                else:
                    prompt = "[NEGATIVE][NEGATIVE]\n" + DROID_PROMPT_TEMPLATE.format(
                        task=task_text
                    )

        return {
            "images": images,
            "state": state_pad,
            "state_mask": state_mask,
            "action": action_pad,
            "action_mask": action_mask,
            "embodiment_id": np.int64(self.embodiment_id),
            "has_real_action": np.bool_(True),
            "has_lapa_action": np.bool_(False),
            "is_cotrain_instance": np.bool_(False),
            "segmentation_target": np.zeros((2,), dtype=np.float32),
            "segmentation_target_mask": np.zeros((1,), dtype=np.float32),
            "lapa_action": np.zeros_like(action_pad),
            "lapa_action_mask": np.zeros_like(action_mask),
            "text": prompt,
        }


class DreamZeroCollator:
    """Collate DreamZero samples: stack tensors and tokenize text.

    Called by DataLoader to combine a list of __getitem__ outputs into one batch.

    Input:  list of B dicts, each from DreamZeroLiberoDataset.__getitem__
    Output: single dict with batched tensors:
      images            (B, 33, 256, 512, 3)  uint8   stacked video frames
      state             (B, 4, 64)            float32 stacked normalized states
      state_mask        (B, 4, 64)            bool
      action            (B, 64, 32)           float32 stacked normalized actions
      action_mask       (B, 64, 32)           bool
      embodiment_id     (B,)                  int64
      has_real_action   (B,)                  bool
      text              (B, 512)              int64   T5 token IDs, padded to max_seq_len
      text_attention_mask (B, 512)            int64   1 for real tokens, 0 for padding
    """

    def __init__(
        self,
        tokenizer_path: str,
        max_seq_len: int,
        cfg_mode: bool = False,
        sft_embodiment: str = "libero",
    ):
        from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
            HuggingfaceTokenizer,
        )

        # HuggingfaceTokenizer wraps the umt5-xxl tokenizer with fixed output length (512)
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=max_seq_len,  # output token IDs are padded/truncated to this length
            clean="whitespace",
        )
        self.cfg_mode = bool(cfg_mode)
        self.sft_embodiment = str(sft_embodiment).lower().replace("-", "_")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}

        # Stack all array fields along a new batch dimension (axis 0)
        for key in [
            "images",
            "state",
            "state_mask",
            "action",
            "action_mask",
            "embodiment_id",
            "has_real_action",
            "has_lapa_action",
            "is_cotrain_instance",
            "segmentation_target",
            "segmentation_target_mask",
            "lapa_action",
            "lapa_action_mask",
        ]:
            values = [f[key] for f in features]
            batch[key] = torch.as_tensor(np.stack(values, axis=0))

        # Tokenize text:
        # - SFT mode: raw task string -> LIBERO_PROMPT_TEMPLATE -> T5 tokens
        # - CFG mode: dataset already outputs final prompt string -> T5 tokens
        # text_ids shape: (B, 512) int32
        # text_mask shape: (B, 512) int32  (1=real token, 0=padding)
        raw_texts = [str(f["text"]) for f in features]
        if self.cfg_mode:
            text_values = raw_texts
        elif self.sft_embodiment in ("droid", "oxe_droid"):
            text_values = [
                DROID_PROMPT_TEMPLATE.format(task=t.lower()) for t in raw_texts
            ]
        else:
            text_values = [LIBERO_PROMPT_TEMPLATE.format(task=t) for t in raw_texts]
        text_ids, text_mask = self.tokenizer(
            text_values, return_mask=True, add_special_tokens=True
        )
        batch["text"] = torch.as_tensor(text_ids)  # (B, 512) int64
        batch["text_attention_mask"] = torch.as_tensor(text_mask)  # (B, 512) int64
        return batch


def build_dreamzero_sft_dataloader(
    cfg,
    world_size: int,
    rank: int,
    data_paths: str,
    eval_dataset: bool = False,
):
    """Build DreamZero SFT dataloader -- callable from FSDPVlaSftWorker.

    Uses DistributedSampler to shard data across GPUs:
      - Each of the 8 GPUs sees 1/8 of the dataset per epoch
      - micro_batch_size samples are returned per iteration per GPU
      - Global effective batch size = micro_batch_size * world_size * grad_accum_steps
    """
    model_cfg = cfg.actor.model
    tokenizer_path = model_cfg.get("tokenizer_path", "google/umt5-xxl")
    max_seq_len = int(model_cfg.get("max_seq_len", 512))
    max_action_dim = int(model_cfg.get("dreamzero_max_action_dim", 32))
    max_state_dim = int(model_cfg.get("dreamzero_max_state_dim", 64))
    cfg_mode = bool(model_cfg.get("cfg_mode", False))
    advantage_parquet = model_cfg.get("advantage_parquet", None)
    if advantage_parquet in ("", None):
        advantage_parquet = None
    unconditional_prob = float(model_cfg.get("unconditional_prob", 0.3))

    sft_embodiment = str(model_cfg.get("embodiment_tag", "libero_sim")).lower()
    if sft_embodiment in ("droid", "oxe_droid"):
        # Unified knobs for both libero and droid:
        # - dreamzero_action_horizon: per-block action steps
        # - dreamzero_num_chunks: number of temporal chunks/blocks
        # - dreamzero_num_video_frames: sampled video frames
        droid_action_horizon = int(model_cfg.get("dreamzero_action_horizon", 24))
        droid_num_chunks = int(model_cfg.get("dreamzero_num_chunks", 4))
        droid_video_frames = int(model_cfg.get("dreamzero_num_video_frames", 33))
        droid_total_action_steps = int(
            model_cfg.get(
                "dreamzero_total_action_steps",
                droid_action_horizon * droid_num_chunks,
            )
        )
        droid_state_horizon = int(
            model_cfg.get("dreamzero_state_horizon", droid_num_chunks)
        )
        droid_state_step = int(
            model_cfg.get("dreamzero_state_step", droid_action_horizon)
        )
        rel = bool(model_cfg.get("relative_action", True))
        rel_keys = list(model_cfg.get("relative_action_keys", ["joint_position"]))
        droid_view_hw = _infer_droid_view_hw_from_model_cfg(model_cfg)
        logger.info(
            "DreamZero DROID map-style: per-view resize target = %s", droid_view_hw
        )

        lazy_load = cfg.data.get("lazy_load", True)
        pq_cache_max_episodes = cfg.data.get("parquet_cache_size", 128)
        video_tolerance_s = cfg.data.get("video_tolerance_s", 0.1)

        dataset = DreamZeroDroidDataset(
            data_path=data_paths,
            lazy_load=lazy_load,
            action_horizon=droid_action_horizon,
            num_video_frames=droid_video_frames,
            total_action_steps=droid_total_action_steps,
            state_horizon=droid_state_horizon,
            state_step=droid_state_step,
            max_action_dim=max_action_dim,
            max_state_dim=max_state_dim,
            cfg_mode=cfg_mode,
            advantage_parquet=advantage_parquet,
            unconditional_prob=unconditional_prob,
            relative_action=rel,
            relative_action_keys=rel_keys,
            droid_view_hw=droid_view_hw,
            pq_cache_max_episodes=pq_cache_max_episodes,
            video_tolerance_s=video_tolerance_s,
        )
        collate_embodiment = "oxe_droid"
    else:
        action_chunk_size = int(
            model_cfg.get("dreamzero_action_horizon", 16)
        )  # steps per chunk
        num_chunks = int(model_cfg.get("dreamzero_num_chunks", 4))
        effective_action_horizon = (
            action_chunk_size * num_chunks
        )  # = 64 total action steps
        dataset = DreamZeroLiberoDataset(
            data_path=data_paths,
            action_horizon=effective_action_horizon,
            num_chunks=num_chunks,
            max_action_dim=max_action_dim,
            max_state_dim=max_state_dim,
            cfg_mode=cfg_mode,
            advantage_parquet=advantage_parquet,
            unconditional_prob=unconditional_prob,
        )
        collate_embodiment = "libero"
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )
    num_workers = int(cfg.data.get("num_workers", 4))
    prefetch_factor = int(cfg.data.get("prefetch_factor", 4))
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,  # samples per GPU per step
        sampler=sampler,
        drop_last=not eval_dataset,
        num_workers=num_workers,
        pin_memory=True,  # faster CPU->GPU transfer
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=DreamZeroCollator(
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            cfg_mode=cfg_mode,
            sft_embodiment=collate_embodiment,
        ),
    )
    return data_loader, {"num_samples": len(dataset)}
