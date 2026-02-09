# Copyright 2025 The RLinf Authors.
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

import argparse
import glob
import json
import os
import shutil


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _find_index_path(rank_dir: str) -> str:
    trajectory_path = os.path.join(rank_dir, "trajectory_index.json")
    if os.path.exists(trajectory_path):
        return trajectory_path
    legacy_path = os.path.join(rank_dir, "trajector_index.json")
    if os.path.exists(legacy_path):
        return legacy_path
    raise FileNotFoundError(
        f"trajectory_index.json/trajector_index.json not found in {rank_dir}"
    )


def _normalize_index_data(index_data: dict) -> tuple[dict[int, dict], list[int]]:
    trajectory_index = {
        int(k): v for k, v in index_data.get("trajectory_index", {}).items()
    }
    trajectory_id_list = [int(k) for k in index_data.get("trajectory_id_list", [])]
    for info in trajectory_index.values():
        if isinstance(info, dict) and "trajectory_id" in info:
            try:
                info["trajectory_id"] = int(info["trajectory_id"])
            except (TypeError, ValueError):
                pass
    return trajectory_index, trajectory_id_list


def _resolve_trajectory_path(
    rank_dir: str,
    trajectory_id: int,
    model_weights_id: str,
    ext: str,
) -> str:
    expected = os.path.join(
        rank_dir, f"trajectory_{trajectory_id}_{model_weights_id}{ext}"
    )
    if os.path.exists(expected):
        return expected

    pattern = os.path.join(rank_dir, f"trajectory_{trajectory_id}_*{ext}")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"trajectory file not found for id={trajectory_id} in {rank_dir}"
        )
    return candidates[0]


def merge_replay_buffers(source_path: str, save_path: str, move_files: bool) -> None:
    os.makedirs(save_path, exist_ok=True)

    rank_dirs = sorted(
        [
            os.path.join(source_path, name)
            for name in os.listdir(source_path)
            if name.startswith("rank_")
            and os.path.isdir(os.path.join(source_path, name))
        ]
    )
    if not rank_dirs:
        raise FileNotFoundError(f"No rank_* directories found in {source_path}")

    merged_index: dict[int, dict] = {}
    merged_id_list: list[int] = []
    total_samples = 0
    next_id = 0
    trajectory_format = None
    seed = None

    rank_data = []
    for rank_dir in rank_dirs:
        metadata = _load_json(os.path.join(rank_dir, "metadata.json"))
        if trajectory_format is None:
            trajectory_format = metadata.get("trajectory_format", "pt")
        elif trajectory_format != metadata.get("trajectory_format", "pt"):
            raise ValueError("trajectory_format mismatch across ranks")

        if seed is None and "seed" in metadata:
            seed = metadata["seed"]

        index_path = _find_index_path(rank_dir)
        index_data = _load_json(index_path)
        trajectory_index, _ = _normalize_index_data(index_data)
        rank_data.append((rank_dir, trajectory_index))

    ext = ".pt" if trajectory_format == "pt" else ".pkl"
    all_ids = sorted(
        {
            trajectory_id
            for _, trajectory_index in rank_data
            for trajectory_id in trajectory_index.keys()
        }
    )

    for trajectory_id in all_ids:
        for rank_dir, trajectory_index in rank_data:
            if trajectory_id not in trajectory_index:
                continue
            info = trajectory_index[trajectory_id]
            model_weights_id = info.get("model_weights_id", "")
            src_path = _resolve_trajectory_path(
                rank_dir, trajectory_id, model_weights_id, ext
            )

            new_id = next_id
            next_id += 1

            info["trajectory_id"] = new_id
            merged_index[new_id] = info
            merged_id_list.append(new_id)

            new_name = f"trajectory_{new_id}_{model_weights_id}{ext}"
            dst_path = os.path.join(save_path, new_name)
            if move_files:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

            total_samples += int(info.get("num_samples", 0))

    metadata_out = {
        "trajectory_format": trajectory_format or "pt",
        "size": len(merged_id_list),
        "total_samples": total_samples,
        "trajectory_counter": len(merged_id_list),
    }
    if seed is not None:
        metadata_out["seed"] = seed

    index_out = {
        "trajectory_index": merged_index,
        "trajectory_id_list": merged_id_list,
    }

    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata_out, f)

    trajectory_index_path = os.path.join(save_path, "trajectory_index.json")
    with open(trajectory_index_path, "w") as f:
        json.dump(index_out, f)

    legacy_index_path = os.path.join(save_path, "trajector_index.json")
    with open(legacy_index_path, "w") as f:
        json.dump(index_out, f)


def split_replay_buffer(
    buffer_dir: str, save_path: str, split_count: int, move_files: bool
) -> None:
    if split_count <= 0:
        raise ValueError("split_count must be > 0")
    os.makedirs(save_path, exist_ok=True)

    metadata = _load_json(os.path.join(buffer_dir, "metadata.json"))
    trajectory_format = metadata.get("trajectory_format", "pt")
    seed = metadata.get("seed", None)

    index_path = _find_index_path(buffer_dir)
    index_data = _load_json(index_path)
    trajectory_index, trajectory_id_list = _normalize_index_data(index_data)

    ext = ".pt" if trajectory_format == "pt" else ".pkl"

    selected_ids = trajectory_id_list[:split_count]
    if len(selected_ids) < split_count:
        raise ValueError(
            f"Only {len(selected_ids)} trajectories available, "
            f"cannot split {split_count}"
        )

    merged_index: dict[int, dict] = {}
    merged_id_list: list[int] = []
    total_samples = 0
    next_id = 0

    for trajectory_id in selected_ids:
        if trajectory_id not in trajectory_index:
            raise KeyError(f"Trajectory id {trajectory_id} not found in index")
        info = trajectory_index[trajectory_id]
        model_weights_id = info.get("model_weights_id", "")
        src_path = _resolve_trajectory_path(
            buffer_dir, trajectory_id, model_weights_id, ext
        )

        new_id = next_id
        next_id += 1

        info["trajectory_id"] = new_id
        merged_index[new_id] = info
        merged_id_list.append(new_id)

        new_name = f"trajectory_{new_id}_{model_weights_id}{ext}"
        dst_path = os.path.join(save_path, new_name)
        if move_files:
            shutil.move(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

        total_samples += int(info.get("num_samples", 0))

    metadata_out = {
        "trajectory_format": trajectory_format,
        "size": len(merged_id_list),
        "total_samples": total_samples,
        "trajectory_counter": len(merged_id_list),
    }
    if seed is not None:
        metadata_out["seed"] = seed

    index_out = {
        "trajectory_index": merged_index,
        "trajectory_id_list": merged_id_list,
    }

    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata_out, f)

    trajectory_index_path = os.path.join(save_path, "trajectory_index.json")
    with open(trajectory_index_path, "w") as f:
        json.dump(index_out, f)

    legacy_index_path = os.path.join(save_path, "trajector_index.json")
    with open(legacy_index_path, "w") as f:
        json.dump(index_out, f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge replay_buffer data across ranks or split a single buffer by trajectory count"
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=".",
        help="Directory containing rank_* subdirectories or a single buffer dir",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory to save merged replay_buffer",
    )
    parser.add_argument(
        "--split-count",
        type=int,
        default=None,
        help="If set, split a single buffer by trajectory count",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them",
    )
    args = parser.parse_args()

    if args.split_count is not None:
        split_replay_buffer(
            args.source_path,
            args.save_path,
            split_count=args.split_count,
            move_files=not args.copy,
        )
    else:
        merge_replay_buffers(args.source_path, args.save_path, move_files=not args.copy)


# python merge_or_split_replay_buffer.py --source-path /path/to/buffer --save-path /path/to/merge --copy
# python merge_or_split_replay_buffer.py --source-path /path/to/buffer --save-path /path/to/split --split-count 10 --copy


if __name__ == "__main__":
    main()
