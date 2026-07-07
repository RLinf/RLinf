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

"""Generate DreamZero ``metadata.json`` from a local LeRobot v2 dataset.

The output format matches ``model_path/experiment_cfg/metadata.json`` used by
DreamZero training and inference.

Examples:

  # Libero (simple state/actions columns)
  python toolkits/lerobot/generate_dreamzero_metadata.py \
    --preset libero_sim \
    --dataset-root /path/libero \
    --output-metadata /path/to/output/metadata.json

  # DROID / OXE (split state/action via meta/modality.json)
  python toolkits/lerobot/generate_dreamzero_metadata.py \
    --preset oxe_droid \
    --dataset-root /path/droid \
    --output-metadata /path/to/output/metadata.json \
    --merge

  # Franka real-world pick-and-place (simple state/actions columns)
  python toolkits/lerobot/generate_dreamzero_metadata.py \
    --preset franka_pnp \
    --dataset-root /path/franka_pnp \
    --output-metadata /path/to/output/franka_pnp_metadata.json

  # Both presets in one metadata.json
  python toolkits/lerobot/generate_dreamzero_metadata.py \
    --preset libero_sim oxe_droid \
    --dataset-root /path/libero /path/droid \
    --output-metadata /path/to/output/metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# Preset registry: (embodiment_tag, state_key, action_key, video_keys, use_modality_json)
PRESETS: dict[str, dict[str, Any]] = {
    "libero_sim": {
        "embodiment_tag": "libero_sim",
        "state_key": "state",
        "action_key": "actions",
        "video_keys": ["image", "wrist_image"],
        "use_modality_json": False,
        "default_dataset_root": None,
    },
    "oxe_droid": {
        "embodiment_tag": "oxe_droid",
        "state_key": "observation.state",
        "action_key": "action",
        "video_keys": None,
        "use_modality_json": True,
        "default_dataset_root": None,
    },
    "franka_pnp": {
        "embodiment_tag": "franka_pnp",
        "state_key": "state",
        "action_key": "actions",
        "video_keys": [
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        "use_modality_json": False,
        "default_dataset_root": None,
    },
    "real_panda_single_arm": {
        "embodiment_tag": "real_panda_single_arm",
        "state_key": "state",
        "action_key": "actions",
        "video_keys": [
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        "use_modality_json": False,
        "default_dataset_root": None,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_episodes(meta_dir: Path) -> list[int]:
    episodes_path = meta_dir / "episodes.jsonl"
    episodes: list[int] = []
    with open(episodes_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            episodes.append(int(json.loads(line)["episode_index"]))
    if not episodes:
        raise ValueError(f"No episodes found in {episodes_path}")
    return episodes


def _episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size") or 1000)
    data_tmpl = info.get("data_path")
    if not data_tmpl:
        raise KeyError("meta/info.json missing data_path")
    rel = data_tmpl.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return root / rel


def _resolve_column(schema_names: list[str], preferred: str, aliases: list[str]) -> str:
    for name in [preferred, *aliases]:
        if name in schema_names:
            return name
    raise KeyError(
        f"Could not find column {preferred!r}; tried {[preferred, *aliases]} "
        f"in schema {schema_names}"
    )


def _column_to_2d_array(table: Any, column_name: str) -> np.ndarray:
    values = table.column(column_name).to_pylist()
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            f"{column_name} must be a 1D/2D vector column, got {arr.shape}"
        )
    return arr


def _compute_stats(arr: np.ndarray) -> dict[str, list[float]]:
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D array, got {arr.shape}")
    arr64 = arr.astype(np.float64, copy=False)
    return {
        "min": np.min(arr64, axis=0).tolist(),
        "max": np.max(arr64, axis=0).tolist(),
        "mean": np.mean(arr64, axis=0).tolist(),
        "std": np.std(arr64, axis=0).tolist(),
        "q01": np.quantile(arr64, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr64, 0.99, axis=0).tolist(),
    }


def _slice_stats(
    full_stats: dict[str, list[float]], start: int, end: int
) -> dict[str, list[float]]:
    return {key: values[start:end] for key, values in full_stats.items()}


def _feature_resolution(feature: dict[str, Any]) -> list[int]:
    shape = feature.get("shape") or []
    names = feature.get("names") or []
    if len(shape) < 2:
        raise ValueError(f"Cannot infer resolution from feature shape {shape}")
    if "height" in names and "width" in names:
        height = int(shape[names.index("height")])
        width = int(shape[names.index("width")])
        return [height, width]
    return [int(shape[0]), int(shape[1])]


def _video_metadata_from_feature(
    info: dict[str, Any],
    feature_key: str,
    *,
    short_name: str | None = None,
) -> dict[str, Any]:
    feature = (info.get("features") or {}).get(feature_key, {})
    video_info = feature.get("video_info") or feature.get("info") or {}
    fps = float(video_info.get("video.fps", info.get("fps", 10)))
    channels = 3
    if "channel" in (feature.get("names") or []):
        channels = (
            int(shape_val) if (shape_val := _shape_dim(feature, "channel")) else 3
        )
    elif len(feature.get("shape") or []) >= 3:
        channels = int(feature["shape"][-1])
    return {
        "resolution": _feature_resolution(feature),
        "channels": channels,
        "fps": fps,
    }


def _shape_dim(feature: dict[str, Any], dim_name: str) -> int | None:
    names = feature.get("names")
    shape = feature.get("shape")
    if not names or not shape or dim_name not in names:
        return None
    return int(shape[names.index(dim_name)])


def _load_modality_json(meta_dir: Path) -> dict[str, Any]:
    modality_path = meta_dir / "modality.json"
    if not modality_path.exists():
        raise FileNotFoundError(
            f"modality.json not found at {modality_path}; required for split state/action"
        )
    return _load_json(modality_path)


def _build_split_modalities(
    modality_cfg: dict[str, Any],
    *,
    kind: str,
    full_dim: int,
) -> dict[str, dict[str, Any]]:
    entries = modality_cfg.get(kind) or {}
    result: dict[str, dict[str, Any]] = {}
    for subkey, spec in entries.items():
        start = int(spec["start"])
        end = int(spec["end"])
        result[subkey] = {
            "absolute": True,
            "rotation_type": None,
            "shape": [end - start],
            "continuous": True,
        }
    covered = sum(v["shape"][0] for v in result.values())
    if covered != full_dim:
        raise ValueError(
            f"{kind} modality slices cover {covered} dims but column has {full_dim}"
        )
    return result


def _build_split_statistics(
    full_stats: dict[str, list[float]],
    modality_cfg: dict[str, Any],
    *,
    kind: str,
) -> dict[str, dict[str, list[float]]]:
    entries = modality_cfg.get(kind) or {}
    return {
        subkey: _slice_stats(full_stats, int(spec["start"]), int(spec["end"]))
        for subkey, spec in entries.items()
    }


def _video_modalities_from_modality(
    info: dict[str, Any],
    modality_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    video_cfg = modality_cfg.get("video") or {}
    result: dict[str, dict[str, Any]] = {}
    for short_name, spec in video_cfg.items():
        original_key = spec.get("original_key", short_name)
        result[short_name] = _video_metadata_from_feature(info, original_key)
    return result


def _infer_video_keys(info: dict[str, Any]) -> list[str]:
    """Pick LeRobot feature keys for video modalities from meta/info.json."""
    features = info.get("features") or {}
    video_keys = [key for key, feat in features.items() if feat.get("dtype") == "video"]
    if video_keys:
        return sorted(video_keys)
    for key in ("image", "wrist_image"):
        if key in features and features[key].get("dtype") == "image":
            video_keys.append(key)
    if video_keys:
        return video_keys
    raise ValueError(
        "Could not infer video_keys from meta/info.json; pass --video-keys explicitly"
    )


def _video_modalities_simple(
    info: dict[str, Any],
    video_keys: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for video_key in video_keys:
        short_name = video_key.removeprefix("observation.images.")
        result[short_name] = _video_metadata_from_feature(info, video_key)
    return result


def build_metadata(
    dataset_root: Path,
    *,
    embodiment_tag: str,
    state_key: str,
    action_key: str,
    video_keys: list[str] | None,
    use_modality_json: bool,
    max_episodes: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta_dir = dataset_root / "meta"
    info = _load_json(meta_dir / "info.json")
    modality_cfg = _load_modality_json(meta_dir) if use_modality_json else {}
    episodes = _load_episodes(meta_dir)
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]

    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []

    for episode_index in episodes:
        parquet_path = _episode_path(dataset_root, info, episode_index)
        schema_names = list(pq.read_schema(parquet_path).names)
        resolved_state_key = _resolve_column(
            schema_names,
            state_key,
            aliases=["observation.state", "state"],
        )
        resolved_action_key = _resolve_column(
            schema_names,
            action_key,
            aliases=["action", "actions"],
        )
        table = pq.read_table(
            parquet_path,
            columns=[resolved_state_key, resolved_action_key],
        )
        state_chunks.append(_column_to_2d_array(table, resolved_state_key))
        action_chunks.append(_column_to_2d_array(table, resolved_action_key))

    state = np.concatenate(state_chunks, axis=0)
    action = np.concatenate(action_chunks, axis=0)
    state_stats_full = _compute_stats(state)
    action_stats_full = _compute_stats(action)

    if use_modality_json:
        state_stats = _build_split_statistics(
            state_stats_full, modality_cfg, kind="state"
        )
        action_stats = _build_split_statistics(
            action_stats_full, modality_cfg, kind="action"
        )
        state_modalities = _build_split_modalities(
            modality_cfg, kind="state", full_dim=int(state.shape[-1])
        )
        action_modalities = _build_split_modalities(
            modality_cfg, kind="action", full_dim=int(action.shape[-1])
        )
        video_modalities = _video_modalities_from_modality(info, modality_cfg)
        lerobot_stats = {
            state_key: state_stats_full,
            action_key: action_stats_full,
        }
        for subkey, spec in (modality_cfg.get("state") or {}).items():
            start, end = int(spec["start"]), int(spec["end"])
            lerobot_stats[f"state.{subkey}"] = _slice_stats(
                state_stats_full, start, end
            )
        for subkey, spec in (modality_cfg.get("action") or {}).items():
            start, end = int(spec["start"]), int(spec["end"])
            lerobot_stats[f"action.{subkey}"] = _slice_stats(
                action_stats_full, start, end
            )
    else:
        if not video_keys:
            video_keys = _infer_video_keys(info)
        state_stats = {"state": state_stats_full}
        action_stats = {"actions": action_stats_full}
        state_modalities = {
            "state": {
                "absolute": True,
                "rotation_type": None,
                "shape": [int(state.shape[-1])],
                "continuous": True,
            }
        }
        action_modalities = {
            "actions": {
                "absolute": True,
                "rotation_type": None,
                "shape": [int(action.shape[-1])],
                "continuous": True,
            }
        }
        video_modalities = _video_modalities_simple(info, video_keys)
        lerobot_stats = {
            "state": state_stats_full,
            "actions": action_stats_full,
            "state.state": state_stats_full,
            "action.actions": action_stats_full,
        }

    metadata = {
        embodiment_tag: {
            "statistics": {
                "state": state_stats,
                "action": action_stats,
            },
            "modalities": {
                "video": video_modalities,
                "state": state_modalities,
                "action": action_modalities,
            },
            "embodiment_tag": embodiment_tag,
        }
    }
    return metadata, lerobot_stats


def _resolve_preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}; known presets: {known}")
    return PRESETS[name].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DreamZero metadata.json from LeRobot datasets."
    )
    parser.add_argument(
        "--preset",
        nargs="+",
        default=None,
        help=f"Built-in presets: {', '.join(sorted(PRESETS))}",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        nargs="+",
        default=None,
        help="One root per preset (or single root for manual mode).",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("/mnt/project_rlinf/guozhen/test_wam/test_dataset/metadata.json"),
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        nargs="*",
        default=None,
        help="Optional stats.json paths (one per dataset). Defaults to DATASET_ROOT/meta/stats.json.",
    )
    parser.add_argument(
        "--merge", action="store_true", help="Merge into existing output file."
    )
    parser.add_argument("--embodiment-tag", default=None)
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--action-key", default=None)
    parser.add_argument("--video-keys", nargs="+", default=None)
    parser.add_argument(
        "--use-modality-json",
        action="store_true",
        help="Split state/action using meta/modality.json.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    if args.preset:
        if args.dataset_root and len(args.dataset_root) not in (1, len(args.preset)):
            raise SystemExit(
                "Provide either one --dataset-root for all presets, "
                "or one root per preset."
            )
        combined: dict[str, Any] = {}
        if args.merge and args.output_metadata.exists():
            combined = _load_json(args.output_metadata)

        for idx, preset_name in enumerate(args.preset):
            preset = _resolve_preset(preset_name)
            if args.dataset_root:
                root = (
                    args.dataset_root[idx]
                    if len(args.dataset_root) == len(args.preset)
                    else args.dataset_root[0]
                )
            else:
                root = Path(preset["default_dataset_root"])

            metadata, lerobot_stats = build_metadata(
                root,
                embodiment_tag=preset["embodiment_tag"],
                state_key=preset["state_key"],
                action_key=preset["action_key"],
                video_keys=preset.get("video_keys"),
                use_modality_json=bool(preset["use_modality_json"]),
                max_episodes=args.max_episodes,
            )
            combined.update(metadata)

            stats_path = None
            if args.output_stats:
                stats_path = (
                    args.output_stats[idx] if idx < len(args.output_stats) else None
                )
            stats_path = stats_path or (root / "meta" / "stats.json")
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(lerobot_stats, f, indent=2)
                f.write("\n")
            tag = preset["embodiment_tag"]
            print(f"[{tag}] dataset_root={root}")
            print(f"[{tag}] wrote stats {stats_path}")

        args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_metadata, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=4)
            f.write("\n")
        print(f"Wrote merged metadata: {args.output_metadata}")
        print(f"embodiment keys: {list(combined.keys())}")
        return

    if not args.dataset_root:
        raise SystemExit("--dataset-root is required without --preset")
    if args.embodiment_tag is None:
        raise SystemExit("--embodiment-tag is required without --preset")

    root = args.dataset_root[0]
    metadata, lerobot_stats = build_metadata(
        root,
        embodiment_tag=args.embodiment_tag,
        state_key=args.state_key or "state",
        action_key=args.action_key or "actions",
        video_keys=args.video_keys,
        use_modality_json=args.use_modality_json,
        max_episodes=args.max_episodes,
    )

    output = args.output_metadata
    if args.merge and output.exists():
        existing = _load_json(output)
        existing.update(metadata)
        metadata = existing

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        f.write("\n")

    stats_path = (args.output_stats[0] if args.output_stats else None) or (
        root / "meta" / "stats.json"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(lerobot_stats, f, indent=2)
        f.write("\n")

    tag = args.embodiment_tag
    print(f"Wrote {output}")
    print(f"Wrote {stats_path}")
    print(f"embodiment_tag={tag}")


if __name__ == "__main__":
    main()
