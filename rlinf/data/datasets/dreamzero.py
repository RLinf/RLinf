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

"""DreamZero SFT data utilities for LIBERO.

Provides DreamZeroLiberoDataset and DreamZeroCollator that convert
LeRobot v3 LIBERO data into the batch format expected by VLA.forward().

Data flow summary (shapes shown for default config: num_chunks=4, action_horizon=64):
  Raw parquet/mp4
    -> __getitem__:
         images           (T=33, H=256, W=512, C=3)  uint8   side-by-side main+wrist
         state            (4, 64)                     float32 normalized joint angles, zero-padded
         state_mask       (4, 64)                     bool    True for real joint dims
         action           (64, 32)                    float32 normalized joint deltas, zero-padded
         action_mask      (64, 32)                    bool    True for real action dims
         text             str                         raw task description
    -> DreamZeroCollator:
         images           (B, 33, 256, 512, 3)        uint8
         state            (B, 4, 64)                  float32
         state_mask       (B, 4, 64)                  bool
         action           (B, 64, 32)                 float32
         action_mask      (B, 64, 32)                 bool
         text             (B, 512)                    int64   T5 token IDs
         text_attn_mask   (B, 512)                    int64   1=real token, 0=padding
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

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

    # Shared cache across all Dataset instances in the same process to avoid
    # redundant parquet reads when DataLoader forks worker subprocesses.
    _advantage_cache: dict[str, dict[int, np.ndarray]] = {}

    def _init_advantage_lookup(self, meta_dir: Path) -> None:
        """Load per-frame advantage labels for CFG prompt guidance.

        Uses a class-level cache keyed by the resolved parquet path so that
        multiple Dataset instances (e.g. from DataLoader worker forks) share
        the same parsed numpy arrays via copy-on-write memory.
        """
        if not self.cfg_mode:
            return

        import pandas as pd

        adv_path = (
            Path(self.advantage_parquet)
            if self.advantage_parquet
            else (meta_dir / "advantages_test.parquet")
        )
        if not adv_path.is_absolute():
            adv_path = meta_dir / adv_path
        if not adv_path.exists():
            raise FileNotFoundError(
                f"CFG mode enabled but advantage parquet not found: {adv_path}"
            )

        cache_key = str(adv_path.resolve())
        if cache_key in DreamZeroLiberoDataset._advantage_cache:
            self._advantage_map = DreamZeroLiberoDataset._advantage_cache[cache_key]
            self._advantage_path = adv_path
            return

        t0 = time.monotonic()
        df = pd.read_parquet(
            adv_path, columns=["episode_index", "frame_index", "advantage"]
        )
        required_cols = {"episode_index", "frame_index", "advantage"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Advantage parquet must contain columns {sorted(required_cols)}, "
                f"got {list(df.columns)}"
            )

        advantage_map: dict[int, np.ndarray] = {}
        for ep_idx, ep_df in df.groupby("episode_index", sort=False):
            frame_idx = ep_df["frame_index"].to_numpy(dtype=np.int64)
            advantage = ep_df["advantage"].to_numpy(dtype=np.bool_)
            max_frame = int(frame_idx.max())
            lookup = np.zeros(max_frame + 1, dtype=np.bool_)
            lookup[frame_idx] = advantage
            advantage_map[int(ep_idx)] = lookup

        DreamZeroLiberoDataset._advantage_cache[cache_key] = advantage_map
        self._advantage_map = advantage_map
        self._advantage_path = adv_path
        elapsed = time.monotonic() - t0
        logger.info(
            "Loaded advantage parquet (%d rows, %d episodes) from %s in %.1fs",
            len(df),
            len(advantage_map),
            adv_path,
            elapsed,
        )

    def _lookup_advantage(self, episode_index: int, frame_index: int) -> bool:
        """Return per-frame advantage label.

        Raises KeyError if the episode is missing entirely.
        Logs a warning (once per episode) and clamps if frame_index is out of range.
        """
        if episode_index not in self._advantage_map:
            raise KeyError(
                f"episode_index={episode_index} not found in advantage parquet "
                f"{self._advantage_path}"
            )
        values = self._advantage_map[episode_index]
        fi = int(frame_index)
        if fi < 0 or fi >= len(values):
            logger.warning(
                "frame_index=%d out of range [0, %d] for episode %d; clamping to boundary",
                fi,
                len(values) - 1,
                episode_index,
            )
            fi = min(max(fi, 0), len(values) - 1)
        return bool(values[fi])

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

    def __init__(self, tokenizer_path: str, max_seq_len: int, cfg_mode: bool = False):
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
    action_chunk_size = int(
        model_cfg.get("dreamzero_action_horizon", 16)
    )  # steps per chunk
    num_chunks = int(model_cfg.get("dreamzero_num_chunks", 4))
    effective_action_horizon = action_chunk_size * num_chunks  # = 64 total action steps
    max_action_dim = int(model_cfg.get("dreamzero_max_action_dim", 32))
    max_state_dim = int(model_cfg.get("dreamzero_max_state_dim", 64))
    cfg_mode = bool(model_cfg.get("cfg_mode", False))
    advantage_parquet = model_cfg.get("advantage_parquet", None)
    if advantage_parquet in ("", None):
        advantage_parquet = None
    unconditional_prob = float(model_cfg.get("unconditional_prob", 0.3))

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
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )
    num_workers = int(cfg.actor.get("dataloader_num_workers", 4))
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,  # samples per GPU per step
        sampler=sampler,
        shuffle=False,
        drop_last=not eval_dataset,
        num_workers=num_workers,
        pin_memory=True,  # faster CPU->GPU transfer
        persistent_workers=num_workers > 0,
        collate_fn=DreamZeroCollator(
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            cfg_mode=cfg_mode,
        ),
    )
    return data_loader, {"num_samples": len(dataset)}
