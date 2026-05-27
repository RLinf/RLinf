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

"""Helpers for DreamZero LeRobot dataset loading and collation."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.utils.logging import get_logger

logger = get_logger()


def load_task_texts(meta_dir: Path) -> dict[int, str]:
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


def probe_video_container_fps(video_path: Path) -> float | None:
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


def droid_default_state_action_slices() -> tuple[slice, slice, slice, slice]:
    """Slice ranges into convert_droid-style concatenated vectors.

    state: cartesian(6) + gripper(1) + joint(7)
    action: cartesian(6) + cartesian_vel(6) + gripper(1) + gripper_vel(1) + joint(7) + joint_vel(7)
    """
    st_joint = slice(7, 14)
    st_grip = slice(6, 7)
    ac_joint = slice(14, 21)
    ac_grip = slice(12, 13)
    return st_joint, st_grip, ac_joint, ac_grip


def load_modality_json(meta_dir: Path) -> dict[str, Any]:
    modality_path = meta_dir / "modality.json"
    if not modality_path.exists():
        return {}
    with open(modality_path, encoding="utf-8") as f:
        return json.load(f)


def flatten_leaves(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(flatten_leaves(v))
        return out
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def feature_component_spans(names: Any, feature_dim: int) -> dict[str, slice]:
    spans: dict[str, slice] = {}
    if not names:
        return spans

    if isinstance(names, dict):
        cursor = 0
        for key, values in names.items():
            width = len(flatten_leaves(values))
            if width <= 0:
                continue
            spans[str(key)] = slice(cursor, cursor + width)
            cursor += width
        return spans

    if isinstance(names, list) and all(isinstance(x, str) for x in names):
        component_widths = {
            "cartesian_position": 6,
            "cartesian_velocity": 6,
            "gripper_position": 1,
            "gripper_velocity": 1,
            "joint_position": 7,
            "joint_velocity": 7,
            "state": feature_dim,
            "actions": feature_dim,
        }
        cursor = 0
        for key in names:
            width = int(component_widths.get(key, 1))
            spans[str(key)] = slice(cursor, cursor + width)
            cursor += width
        return spans

    cursor = 0
    for entry in names if isinstance(names, list) else []:
        if isinstance(entry, str):
            key, width = entry, 1
        elif isinstance(entry, dict):
            key = str(entry.get("name", ""))
            shape = entry.get("shape")
            width = (
                int(np.prod(shape)) if shape is not None else int(entry.get("dim", 1))
            )
        else:
            continue
        if key:
            spans[key] = slice(cursor, cursor + width)
        cursor += width
    return spans


def infer_modality_json_from_features(features: dict[str, Any]) -> dict[str, Any]:
    """Best-effort modality metadata for LeRobot trees without meta/modality.json."""
    modality: dict[str, Any] = {
        "video": {},
        "state": {},
        "action": {},
        "annotation": {},
    }

    for source_key, feature in features.items():
        if not isinstance(feature, dict):
            continue
        if feature.get("dtype") == "video" or source_key.startswith(
            "observation.images."
        ):
            short = source_key.split("observation.images.", 1)[-1]
            modality["video"][short] = {"original_key": source_key}
        elif source_key in ("image", "wrist_image"):
            modality["video"][source_key] = {"original_key": source_key}
        elif source_key.startswith("annotation."):
            short = source_key.split("annotation.", 1)[-1]
            modality["annotation"][short] = {"original_key": source_key}

    for modality_name, source_candidates in (
        ("state", ("observation.state", "state")),
        ("action", ("action", "actions")),
    ):
        source_key = next((key for key in source_candidates if key in features), None)
        if source_key is None:
            continue
        feature = features.get(source_key) or {}
        feature_dim = int((feature.get("shape") or [0])[0] or 0)
        for key, span in feature_component_spans(
            feature.get("names"), feature_dim
        ).items():
            modality[modality_name][key] = {
                "original_key": source_key,
                "start": int(span.start or 0),
                "end": None if span.stop is None else int(span.stop),
            }

    return {key: value for key, value in modality.items() if value}


def safe_lang_text(value: Any, task_map: dict[int, str]) -> str:
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


def discover_local_lerobot_episode_indices(
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


def infer_named_component_slices(
    names: Any, feature_dim: int, wanted: list[str]
) -> dict[str, slice] | None:
    """Infer component slices from LeRobot feature names for generic DreamZero transforms."""
    if not wanted:
        return {}
    if not names:
        return None

    if isinstance(names, dict):
        cursor = 0
        spans: dict[str, slice] = {}
        for key, values in names.items():
            width = len(flatten_leaves(values))
            if width <= 0:
                continue
            spans[str(key)] = slice(cursor, cursor + width)
            cursor += width
        if all(k in spans for k in wanted):
            return {k: spans[k] for k in wanted}
        if len(wanted) == 1 and wanted[0] in ("state", "actions"):
            return {wanted[0]: slice(0, feature_dim)}
        return None

    if isinstance(names, list) and all(isinstance(x, str) for x in names):
        plan = {
            "cartesian_position": 6,
            "cartesian_velocity": 6,
            "gripper_position": 1,
            "gripper_velocity": 1,
            "joint_position": 7,
            "joint_velocity": 7,
            "state": feature_dim,
            "actions": feature_dim,
        }
        cursor = 0
        spans: dict[str, slice] = {}
        for key in names:
            width = int(plan.get(key, 1))
            spans[str(key)] = slice(cursor, cursor + width)
            cursor += width
        if all(k in spans for k in wanted):
            return {k: spans[k] for k in wanted}

    cursor = 0
    spans = {}
    for entry in names if isinstance(names, list) else []:
        if isinstance(entry, str):
            key, width = entry, 1
        elif isinstance(entry, dict):
            key = str(entry.get("name", ""))
            shape = entry.get("shape")
            width = (
                int(np.prod(shape)) if shape is not None else int(entry.get("dim", 1))
            )
        else:
            continue
        spans[key] = slice(cursor, cursor + width)
        cursor += width
    if all(k in spans for k in wanted):
        return {k: spans[k] for k in wanted}

    return None


def collate_ready_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert per-sample transform output to numpy for ``DreamZeroCollator`` stacking."""
    out: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        elif isinstance(value, np.generic):
            value = value.item() if value.ndim == 0 else np.asarray(value)
        out[key] = value
    return out
