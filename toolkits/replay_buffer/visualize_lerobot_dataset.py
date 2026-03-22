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

Normal usage:
    1. Edit ``DATASET_PATH`` and ``OUTPUT_DIR`` below.
    2. Run:
       python3 toolkits/replay_buffer/visualize_lerobot_dataset.py

Optional:
    python3 toolkits/replay_buffer/visualize_lerobot_dataset.py \
        --dataset-path /path/to/collected_data \
        --output-dir /path/to/output
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
        description="Export a LeRobot dataset into per-episode JPG/TXT files."
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


def _write_episode(
    parquet_path: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    episode_meta: dict[str, Any],
    task_map: dict[int, str],
    dataset_info: dict[str, Any],
    image_cls: Any,
) -> None:
    episode_index = _infer_episode_index(parquet_path, rows)
    episode_dir = output_dir / f"episode_{episode_index:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    image_keys = _extract_image_keys(dataset_info, rows)

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
    }
    _write_text_file(episode_dir / "episode.txt", episode_payload)


def main() -> None:
    args = _build_arg_parser().parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    pq = _require_pyarrow()
    image_cls = _require_pillow()

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
        )
        print(f"Exported episode {episode_index:06d} from {parquet_path.name}")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
