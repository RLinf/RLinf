#!/usr/bin/env python3
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

"""Headless LeRobot parquet visualizer.

This script expands a LeRobot dataset into easy-to-read files:
1. one output folder per episode
2. one ``.jpg`` per image field per step
3. one ``.txt`` per step for non-image fields
4. one ``episode.txt`` for episode-level metadata
5. (optional) one ``.mp4`` video per image field per episode
6. (optional) one merged ``episode_merged.mp4`` when multiple image fields exist

Normal usage:
    python3 toolkits/lerobot/visualize_lerobot_dataset.py \\
        --dataset-path /path/to/collected_data \\
        --output-dir /path/to/output

With mp4 export:
    python3 toolkits/lerobot/visualize_lerobot_dataset.py \\
        --dataset-path /path/to/collected_data \\
        --output-dir /path/to/output \\
        --export-mp4 --mp4-fps 30

Multi-view merge layout (when >=2 image fields):
    - 2 views: horizontal strip (non-wrist left, wrist right)
    - 3-4 views: 2-column grid (row-major)
    - 5+ views: horizontal strip
    - all panels are resized to the same height; each panel is labeled with its field name
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

# Edit these two paths for your normal workflow.
DATASET_PATH = "collected_data"
OUTPUT_DIR = "collected_data_visualized"

JPEG_QUALITY = 95


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a LeRobot dataset into per-episode JPG/TXT files, "
            "optionally with per-view and merged MP4 videos."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_PATH,
        help="Path to a LeRobot dataset root or a single episode parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory where the visualized episode folders will be written.",
    )
    parser.add_argument(
        "--export-mp4",
        action="store_true",
        default=False,
        help="Export one .mp4 video per image field per episode (requires opencv-python). "
        "When multiple image fields exist, also exports episode_merged.mp4.",
    )
    parser.add_argument(
        "--mp4-fps",
        type=int,
        default=30,
        help="FPS for the exported mp4 videos (default: 30).",
    )
    return parser


def _require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install the same environment used for "
            "RLinf data collection, or run `pip install pyarrow`."
        ) from exc
    return pq


def _require_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: Pillow. Install the same environment used for "
            "RLinf data collection, or run `pip install Pillow`."
        ) from exc
    return Image


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: opencv-python. Run `pip install opencv-python`."
        ) from exc
    return cv2


def _image_bytes_to_array(raw_bytes: bytes, image_cls: Any) -> Any:
    """Convert raw image bytes to a numpy RGB array via Pillow."""
    import numpy as np

    if isinstance(raw_bytes, memoryview):
        raw_bytes = raw_bytes.tobytes()
    with image_cls.open(io.BytesIO(raw_bytes)) as image:
        return np.array(image.convert("RGB"))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_parquet_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        if dataset_path.suffix != ".parquet":
            raise SystemExit(f"Expected a parquet file, got: {dataset_path}")
        return [dataset_path]

    if not dataset_path.exists():
        raise SystemExit(f"Dataset path does not exist: {dataset_path}")

    default_data_dir = dataset_path / "data"
    if default_data_dir.exists():
        return sorted(default_data_dir.glob("chunk-*/episode_*.parquet"))

    return sorted(dataset_path.glob("**/*.parquet"))


def _build_episode_meta_map(meta_dir: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["episode_index"]): row
        for row in _load_jsonl(meta_dir / "episodes.jsonl")
        if "episode_index" in row
    }


def _build_task_map(meta_dir: Path) -> dict[int, str]:
    return {
        int(row["task_index"]): row["task"]
        for row in _load_jsonl(meta_dir / "tasks.jsonl")
        if "task_index" in row and "task" in row
    }


def _infer_episode_index(parquet_path: Path, rows: list[dict[str, Any]]) -> int:
    if rows and "episode_index" in rows[0]:
        return int(rows[0]["episode_index"])

    match = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
    if match:
        return int(match.group(1))

    raise ValueError(f"Unable to infer episode index from {parquet_path}")


def _is_image_struct(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value.keys()) >= {"bytes", "path"}
        and isinstance(value.get("path"), str)
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _save_image_struct(
    image_struct: dict[str, Any],
    output_path: Path,
    image_cls: Any,
) -> bool:
    raw_bytes = image_struct.get("bytes")
    if not raw_bytes:
        return False

    if isinstance(raw_bytes, memoryview):
        raw_bytes = raw_bytes.tobytes()

    with image_cls.open(io.BytesIO(raw_bytes)) as image:
        image.convert("RGB").save(output_path, format="JPEG", quality=JPEG_QUALITY)
    return True


def _write_text_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _step_stem(step_idx: int, row: dict[str, Any]) -> str:
    frame_index = row.get("frame_index")
    if isinstance(frame_index, int):
        return f"step_{frame_index:06d}"
    return f"step_{step_idx:06d}"


def _extract_image_keys(
    info_json: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[str]:
    feature_keys = []
    features = info_json.get("features", {})
    for key, value in features.items():
        if isinstance(value, dict) and value.get("dtype") == "image":
            feature_keys.append(key)

    if feature_keys:
        return feature_keys

    if not rows:
        return []

    return [key for key, value in rows[0].items() if _is_image_struct(value)]


def _sort_image_keys_for_layout(image_keys: list[str]) -> list[str]:
    """Order cameras for stitching: non-wrist first, then wrist, then by name."""

    def sort_key(key: str) -> tuple[int, str]:
        is_wrist = 1 if "wrist" in key.lower() else 0
        return (is_wrist, key)

    return sorted(image_keys, key=sort_key)


def _resize_to_height(frame: Any, target_height: int, cv2: Any) -> Any:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    target_width = max(1, int(width * target_height / height))
    return cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def _draw_camera_label(frame: Any, label: str, cv2: Any) -> Any:
    labeled = frame.copy()
    banner_h = 28
    cv2.rectangle(
        labeled, (0, 0), (labeled.shape[1], banner_h), (0, 0, 0), thickness=-1
    )
    cv2.putText(
        labeled,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def _concat_horizontal(frames: list[Any]) -> Any:
    import numpy as np

    return np.concatenate(frames, axis=1)


def _concat_vertical(frames: list[Any]) -> Any:
    import numpy as np

    return np.concatenate(frames, axis=0)


def _pad_row_to_width(row: Any, target_width: int) -> Any:
    import numpy as np

    if row.shape[1] >= target_width:
        return row
    pad = np.zeros((row.shape[0], target_width - row.shape[1], 3), dtype=row.dtype)
    return np.concatenate([row, pad], axis=1)


def _compose_multi_view_frame(panel_frames: list[tuple[str, Any]], cv2: Any) -> Any:
    """Stitch multiple camera frames into one RGB frame."""
    import numpy as np

    if not panel_frames:
        raise ValueError("panel_frames must not be empty")
    if len(panel_frames) == 1:
        return panel_frames[0][1]

    target_height = max(frame.shape[0] for _, frame in panel_frames)
    labeled = [
        _draw_camera_label(_resize_to_height(frame, target_height, cv2), key, cv2)
        for key, frame in panel_frames
    ]

    if len(labeled) == 2:
        return _concat_horizontal(labeled)

    if len(labeled) <= 4:
        cols = 2
        rows = []
        for start in range(0, len(labeled), cols):
            row_panels = labeled[start : start + cols]
            if len(row_panels) == 1:
                row_panels.append(np.zeros_like(row_panels[0]))
            rows.append(_concat_horizontal(row_panels))
        max_width = max(row.shape[1] for row in rows)
        return _concat_vertical([_pad_row_to_width(row, max_width) for row in rows])

    return _concat_horizontal(labeled)


def _collect_episode_frames(
    rows: list[dict[str, Any]],
    image_keys: list[str],
    image_cls: Any,
) -> dict[str, list[Any]]:
    frames: dict[str, list[Any]] = {key: [] for key in image_keys}
    for row in rows:
        for key in image_keys:
            value = row.get(key)
            if not _is_image_struct(value):
                continue
            raw_bytes = value.get("bytes")
            if not raw_bytes:
                continue
            frames[key].append(_image_bytes_to_array(raw_bytes, image_cls))
    return frames


def _write_video_file(
    frame_list: list[Any],
    output_path: Path,
    fps: int,
    cv2: Any,
) -> None:
    if not frame_list:
        return
    if fps <= 0:
        raise ValueError(f"mp4 fps must be positive, got {fps}")
    height, width = frame_list[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {output_path}. "
            "Check that opencv-python is installed and the output path is writable."
        )
    for frame in frame_list:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _keys_with_image_bytes(
    rows: list[dict[str, Any]],
    image_keys: list[str],
) -> list[str]:
    """Return image keys that have at least one non-empty bytes payload in rows."""
    active_keys: list[str] = []
    for key in image_keys:
        for row in rows:
            value = row.get(key)
            if _is_image_struct(value) and value.get("bytes"):
                active_keys.append(key)
                break
    return active_keys


def _build_merged_frames_from_rows(
    rows: list[dict[str, Any]],
    active_keys: list[str],
    image_cls: Any,
    cv2: Any,
) -> list[Any]:
    """Build one stitched frame per timestep, aligned by parquet row order."""
    import numpy as np

    reference_frames: dict[str, Any] = {}
    merged_frames: list[Any] = []
    for row in rows:
        panel_frames: list[tuple[str, Any]] = []
        for key in active_keys:
            value = row.get(key)
            if _is_image_struct(value):
                raw_bytes = value.get("bytes")
                if raw_bytes:
                    frame = _image_bytes_to_array(raw_bytes, image_cls)
                    reference_frames[key] = frame
                    panel_frames.append((key, frame))
                    continue
            if key in reference_frames:
                panel_frames.append((key, np.zeros_like(reference_frames[key])))

        if len(panel_frames) >= 2:
            merged_frames.append(_compose_multi_view_frame(panel_frames, cv2))
    return merged_frames


def _write_mp4_videos(
    rows: list[dict[str, Any]],
    image_keys: list[str],
    episode_dir: Path,
    fps: int,
    cv2: Any,
    image_cls: Any,
) -> tuple[dict[str, str], str | None]:
    """Write per-view mp4 files and optionally one merged multi-view mp4.

    Returns:
        (per_view_mp4_map, merged_mp4_name_or_none)
    """
    if not rows or not image_keys:
        return {}, None

    frames = _collect_episode_frames(rows, image_keys, image_cls)
    ordered_keys = _sort_image_keys_for_layout(image_keys)

    mp4_map: dict[str, str] = {}
    for key in image_keys:
        frame_list = frames[key]
        if not frame_list:
            continue
        mp4_name = f"episode_{key}.mp4"
        _write_video_file(frame_list, episode_dir / mp4_name, fps, cv2)
        mp4_map[key] = mp4_name

    merged_mp4_name: str | None = None
    if len(ordered_keys) >= 2:
        active_keys = _keys_with_image_bytes(rows, ordered_keys)
        if len(active_keys) >= 2:
            merged_frames = _build_merged_frames_from_rows(
                rows, active_keys, image_cls, cv2
            )
            if merged_frames:
                merged_mp4_name = "episode_merged.mp4"
                _write_video_file(
                    merged_frames, episode_dir / merged_mp4_name, fps, cv2
                )

    return mp4_map, merged_mp4_name


def _write_episode(
    parquet_path: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    episode_meta: dict[str, Any],
    task_map: dict[int, str],
    dataset_info: dict[str, Any],
    image_cls: Any,
    *,
    export_mp4: bool = False,
    mp4_fps: int = 30,
    cv2_module: Any = None,
) -> None:
    episode_index = _infer_episode_index(parquet_path, rows)
    episode_dir = output_dir / f"episode_{episode_index:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    image_keys = _extract_image_keys(dataset_info, rows)

    mp4_map: dict[str, str] = {}
    merged_mp4_name: str | None = None
    if export_mp4 and cv2_module is not None:
        mp4_map, merged_mp4_name = _write_mp4_videos(
            rows, image_keys, episode_dir, mp4_fps, cv2_module, image_cls
        )

    for step_idx, row in enumerate(rows):
        stem = _step_stem(step_idx, row)
        step_payload: dict[str, Any] = {
            "episode_index": episode_index,
            "step_index": step_idx,
            "source_parquet": str(parquet_path),
        }

        exported_images: dict[str, Any] = {}
        for key, value in row.items():
            if key in image_keys and _is_image_struct(value):
                image_name = f"{stem}_{key}.jpg"
                image_path = episode_dir / image_name
                exported = _save_image_struct(value, image_path, image_cls)
                exported_images[key] = {
                    "exported": exported,
                    "output_file": image_name if exported else None,
                    "source_path": value.get("path", ""),
                }
            else:
                step_payload[key] = _json_safe(value)

        task_index = row.get("task_index")
        if isinstance(task_index, int) and task_index in task_map:
            step_payload["task"] = task_map[task_index]

        step_payload["image_exports"] = exported_images
        _write_text_file(episode_dir / f"{stem}.txt", step_payload)

    episode_payload = {
        "episode_index": episode_index,
        "source_parquet": str(parquet_path),
        "num_steps": len(rows),
        "task": task_map.get(rows[0].get("task_index")) if rows else None,
        "episode_meta": _json_safe(episode_meta),
        "dataset_info": {
            "robot_type": dataset_info.get("robot_type"),
            "fps": dataset_info.get("fps"),
            "codebase_version": dataset_info.get("codebase_version"),
        },
        "image_keys": image_keys,
        "step_txt_pattern": "step_XXXXXX.txt",
        "step_image_pattern": "step_XXXXXX_<image_key>.jpg",
        "mp4_videos": mp4_map if mp4_map else None,
        "mp4_merged_video": merged_mp4_name,
        "mp4_layout": {
            "ordered_image_keys": _sort_image_keys_for_layout(image_keys),
            "rules": "2 views: horizontal; 3-4 views: 2-column grid; 5+ views: horizontal",
        }
        if export_mp4 and len(image_keys) >= 2
        else None,
    }
    _write_text_file(episode_dir / "episode.txt", episode_payload)


def main() -> None:
    args = _build_arg_parser().parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    pq = _require_pyarrow()
    image_cls = _require_pillow()
    cv2_module = _require_cv2() if args.export_mp4 else None

    parquet_files = _resolve_parquet_files(dataset_path)
    if not parquet_files:
        raise SystemExit(f"No parquet files found under: {dataset_path}")

    dataset_root = dataset_path if dataset_path.is_dir() else dataset_path.parents[2]
    meta_dir = dataset_root / "meta"
    dataset_info = _load_json(meta_dir / "info.json")
    episode_meta_map = _build_episode_meta_map(meta_dir)
    task_map = _build_task_map(meta_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset path: {dataset_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Episodes:     {len(parquet_files)}")
    if args.export_mp4:
        print(f"MP4 export:   enabled (fps={args.mp4_fps})")

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        if not rows:
            print(f"Skip empty parquet: {parquet_path}")
            continue

        episode_index = _infer_episode_index(parquet_path, rows)
        _write_episode(
            parquet_path=parquet_path,
            output_dir=output_dir,
            rows=rows,
            episode_meta=episode_meta_map.get(episode_index, {}),
            task_map=task_map,
            dataset_info=dataset_info,
            image_cls=image_cls,
            export_mp4=args.export_mp4,
            mp4_fps=args.mp4_fps,
            cv2_module=cv2_module,
        )
        print(f"Exported episode {episode_index:06d} from {parquet_path.name}")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
