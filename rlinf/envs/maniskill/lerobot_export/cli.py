# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.




"""
output_root/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   ├── chunk-001/
│   └── ...
├── videos/
│   ├── chunk-000/
│   │   └── observation.images.top/
│   │       ├── episode_000000.mp4
│   │       ├── episode_000001.mp4
│   │       └── ...
│   ├── chunk-001/
│   └── ...
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    ├── episodes_stats.jsonl
    └── stats.json

"""


import os
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
def process_episode_worker(
    npz_path: str,
    eps_idx: int,
    keep_last_n: int,
    fps: int,
    videos_dir: str,
    temp_dir: str,
    export_videos: bool,
    use_temp: bool,
    n_eps_per_shard: int,
) -> Dict[str, Any]:
    try:
        payload = read_episode_npz(npz_path)
        images = payload["image"]
        images_list = images if isinstance(images, list) else list(images)
        if len(images_list) == 0:
            return {"episode_index": eps_idx, "length": 0, "skip": True}
        h, w = images_list[0].size[1], images_list[0].size[0]
        action = np.asarray(payload["action"], dtype=np.float32)
        state = np.asarray(payload["state"], dtype=np.float32)
        instr_np = payload.get("instruction")
        task_text = str(instr_np.tolist()[0]) if isinstance(instr_np, np.ndarray) else str(instr_np)
        mask = apply_action_filter(action, keep_last_n=keep_last_n)
        keep_indices = np.nonzero(mask)[0]
        if keep_indices.size == 0:
            return {"episode_index": eps_idx, "length": 0, "skip": True}
        episode_len = int(keep_indices.size)
        tmp_path = None
        if use_temp:
            # Save filtered arrays to temp to avoid large IPC
            tmp_path = os.path.join(temp_dir, f"episode_{eps_idx:06d}.npz")
            np.savez_compressed(
                tmp_path,
                action=action[keep_indices].astype(np.float32),
                state=state[keep_indices].astype(np.float32),
                task_descriptions=task_text,
            )
        if export_videos:
            # Write video for kept frames in shard subfolder structure
            shard_id = eps_idx // n_eps_per_shard
            video_shard_dir = os.path.join(videos_dir, f"chunk-{shard_id:03d}")
            video_key_dir = os.path.join(video_shard_dir, "observation.images.top")
            ensure_dir(video_key_dir)
            video_out = os.path.join(video_key_dir, f"episode_{eps_idx:06d}.mp4")
            export_images = [images_list[i] for i in keep_indices.tolist()]
            save_video_from_pil_images(export_images, video_out, fps=fps)
        # Episode stats
        # Image stats: compute if we exported or if needed
        img_stats = {"count": 0, "mean": [], "std": [], "min": [], "max": []}
        try:
            export_images = [images_list[i] for i in keep_indices.tolist()]
            if len(export_images) > 0:
                img_arr = np.stack([np.array(im) for im in export_images], axis=0)
                img_stats = compute_image_channel_stats(img_arr)
        except Exception:
            pass
        ep_state_rs = RollingStats(8)
        ep_action_rs = RollingStats(7)
        ep_state_rs.update(state[keep_indices])
        ep_action_rs.update(action[keep_indices])
        result = {
            "episode_index": eps_idx,
            "length": episode_len,
            "task_descriptions": task_text,
            "tmp": tmp_path,
            "img_shape": [h, w, 3],
            "episode_stats": {
                "observation.state": ep_state_rs.finalize(),
                "actions": ep_action_rs.finalize(),
                "observation.images.top": img_stats,
            },
        }
        if not use_temp:
            # Return indices so coordinator can re-read original arrays
            result["indices"] = keep_indices.tolist()
        return result
    except Exception as e:
        return {"episode_index": eps_idx, "error": str(e), "skip": True}


# Support both package execution (-m) and direct script execution
try:
    from .io_utils import list_npz_files, read_episode_npz, ensure_dir, write_json, write_jsonl
    from .filtering import apply_action_filter
    from .stats import RollingStats, compute_image_channel_stats
    from .media import save_video_from_pil_images
    from .parquet_writer import write_episode_parquet
    from .metadata import (
        make_info_json,
        make_episode_row,
        make_task_row,
        make_episode_stats_row,
    )
except Exception:
    # Fallback for direct execution: python cli.py
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from io_utils import list_npz_files, read_episode_npz, ensure_dir, write_json, write_jsonl
    from filtering import apply_action_filter
    from stats import RollingStats, compute_image_channel_stats
    from media import save_video_from_pil_images
    from parquet_writer import write_episode_parquet
    from metadata import (
        make_info_json,
        make_episode_row,
        make_task_row,
        make_episode_stats_row,
    )


def build_feature_spec(image_shape_hw3: List[int], fps: int) -> Dict[str, Any]:
    h, w, c = image_shape_hw3
    return {
        "observation.state": {"dtype": "float32", "shape": [8], "names": ["state"]},
        "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
        "timestamp": {"dtype": "float32", "shape": [1],},
        "episode_index": {"dtype": "int64", "shape": [1],},
        "frame_index": {"dtype": "int64", "shape": [1],},
        "index": {"dtype": "int64", "shape": [1],},
        "next.done": {"dtype": "bool", "shape": [1],},
        "task_index": {"dtype": "int64", "shape": [1],},
        "observation.images.top": {
            "dtype": "video",
            "shape": [h, w, c],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": fps, "video.codec": "mp4v"},
        },
    }


def convert_one_split(
    npz_files: List[str],
    output_root: str,
    split_name: str,
    n_eps_per_shard: int,
    fps: int,
    keep_last_n: int,
    verbose: bool = True,
    num_workers: int = 1,
    export_videos: bool = True,
    use_temp: bool = True,
    keep_temp: bool = False,
    data_dir_override: Optional[str] = None,
    videos_dir_override: Optional[str] = None,
    meta_dir_override: Optional[str] = None,
    temp_dir_override: Optional[str] = None,
) -> None:
    if verbose:
        print(f"[convert_one_split] split={split_name} episodes={len(npz_files)} n_eps_per_shard={n_eps_per_shard} fps={fps} keep_last_n={keep_last_n}", flush=True)

    data_dir = data_dir_override if data_dir_override else os.path.join(output_root, split_name, "data")
    videos_dir = videos_dir_override if videos_dir_override else os.path.join(output_root, split_name, "videos")
    meta_dir = meta_dir_override if meta_dir_override else os.path.join(output_root, split_name, "meta")
    temp_dir = temp_dir_override if temp_dir_override else os.path.join(output_root, split_name, "_temp")
    ensure_dir(data_dir)
    ensure_dir(videos_dir)
    ensure_dir(meta_dir)
    if use_temp:
        ensure_dir(temp_dir)

    # Rolling stats across episodes for this split only
    rs_action = RollingStats(7)
    rs_state = RollingStats(8)
    episodes_rows: List[Dict[str, Any]] = []
    episode_stats_rows: List[Dict[str, Any]] = []
    tasks_map: Dict[str, int] = {}
    tasks_rows: List[Dict[str, Any]] = []

    global_index = 0
    total_videos = 0
    image_shape_hw3 = None

    results: List[Dict[str, Any]] = [None] * len(npz_files)
    if num_workers and num_workers > 1:
        if verbose:
            print(f"[convert_one_split] Parallel pre-processing with {num_workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {
                ex.submit(
                    process_episode_worker,
                    path,
                    i,
                    keep_last_n,
                    fps,
                    videos_dir,
                    temp_dir,
                    export_videos,
                    use_temp,
                    n_eps_per_shard,
                ): i
                for i, path in enumerate(npz_files)
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name}: preprocess"):
                res = fut.result()
                results[res["episode_index"]] = res
    else:
        for i, path in enumerate(tqdm(npz_files, desc=f"{split_name}: preprocess")):
            results[i] = process_episode_worker(path, i, keep_last_n, fps, videos_dir, temp_dir, export_videos, use_temp, n_eps_per_shard)

    # Determine image shape from first non-skip
    for res in results:
        if res and not res.get("skip") and res.get("img_shape"):
            image_shape_hw3 = res["img_shape"]
            break

    # Second phase: assign global indices, write parquet and aggregate stats
    current_start_index = 0
    for eps_idx, res in enumerate(tqdm(results, desc=f"{split_name}: write")):
        if res is None or res.get("skip"):
            if res is not None and res.get("error") and verbose:
                print(f"[convert_one_split] Skipping episode {eps_idx} due to error: {res['error']}", flush=True)
            continue
        task_text = res["task_descriptions"]
        episode_len = int(res["length"])
        ep_action = None
        ep_state = None
        if use_temp and res.get("tmp"):
            npz = np.load(res["tmp"])  # load filtered arrays
            ep_action = npz["action"]
            ep_state = npz["state"]
        else:
            # Re-read original npz and apply indices
            orig_npz = npz_files[eps_idx]
            payload = read_episode_npz(orig_npz)
            action = np.asarray(payload["action"], dtype=np.float32)
            state = np.asarray(payload["state"], dtype=np.float32)
            keep_indices = np.asarray(res.get("indices", []), dtype=np.int64)
            if keep_indices.size == 0:
                # fallback recompute
                keep_indices = np.nonzero(apply_action_filter(action, keep_last_n=keep_last_n))[0]
            ep_action = action[keep_indices]
            ep_state = state[keep_indices]
        # Update split-level rolling stats by merging episode stats (no need to hold arrays)
        def _merge_stats(rs: RollingStats, stats_dict: Dict[str, Any]) -> None:
            n_b = int(stats_dict.get("count", [0])[0])
            if n_b <= 0:
                return
            mean_b = np.asarray(stats_dict["mean"], dtype=np.float64)
            std_b = np.asarray(stats_dict["std"], dtype=np.float64)
            min_b = np.asarray(stats_dict["min"], dtype=np.float64)
            max_b = np.asarray(stats_dict["max"], dtype=np.float64)
            M2_b = (std_b ** 2) * max(n_b - 1, 0)
            # combine rs with (n_b, mean_b, M2_b)
            n_a = rs.count
            if n_a == 0:
                rs.count = n_b
                rs.mean = mean_b
                rs.M2 = M2_b
                rs.min = min_b
                rs.max = max_b
            else:
                delta = mean_b - rs.mean
                n = n_a + n_b
                rs.mean = rs.mean + delta * (n_b / n)
                rs.M2 = rs.M2 + M2_b + (delta ** 2) * (n_a * n_b / n)
                rs.count = n
                rs.min = np.minimum(rs.min, min_b)
                rs.max = np.maximum(rs.max, max_b)

        _merge_stats(rs_action, res["episode_stats"]["actions"])
        _merge_stats(rs_state, res["episode_stats"]["observation.state"])
        # Episode row/meta
        episodes_rows.append(make_episode_row(eps_idx, [task_text], episode_len))
        episode_stats_rows.append(make_episode_stats_row(eps_idx, res["episode_stats"]))
        # Task mapping
        task_index = tasks_map.setdefault(task_text, len(tasks_map))
        # Shard and parquet write
        shard_id = eps_idx // n_eps_per_shard
        shard_dir = os.path.join(data_dir, f"chunk-{shard_id:03d}")
        ensure_dir(shard_dir)
        parquet_path = os.path.join(shard_dir, f"episode_{eps_idx:06d}.parquet")
        rows: List[Dict[str, Any]] = []
        for local_frame_idx in range(episode_len):
            rows.append(
                {
                    "observation.state": ep_state[local_frame_idx].astype(np.float32).tolist(),
                    "actions": ep_action[local_frame_idx].astype(np.float32).tolist(),
                    "timestamp": float(local_frame_idx) / float(fps),
                    "episode_index": int(eps_idx),
                    "frame_index": int(local_frame_idx),
                    "index": int(current_start_index + local_frame_idx),
                    "next.done": bool(local_frame_idx == episode_len - 1),
                    "task_index": int(task_index),
                }
            )
        write_episode_parquet(parquet_path, rows)
        current_start_index += episode_len
        total_videos += 1
        # Delete per-episode temp file to reduce peak disk usage
        if use_temp and res.get("tmp"):
            try:
                os.remove(res["tmp"])
            except Exception:
                pass

    # Global stats (v2.0)
    stats_json = {
        "observation.state": rs_state.finalize(),
        "actions": rs_action.finalize(),
    }
    
    # Generate norm_stats.json in OpenPI format
    norm_stats_json = {
        "norm_stats": {
            "state": {
                "mean": rs_state.finalize()["mean"],
                "std": rs_state.finalize()["std"]
            },
            "actions": {
                "mean": rs_action.finalize()["mean"],
                "std": rs_action.finalize()["std"]
            }
        }
    }

    # Tasks jsonl
    for text, idx in sorted(tasks_map.items(), key=lambda x: x[1]):
        tasks_rows.append(make_task_row(idx, text))

    # Info.json
    if image_shape_hw3 is None:
        image_shape_hw3 = [224, 224, 3]
    features = build_feature_spec(image_shape_hw3, fps)
    split_key = split_name
    info = make_info_json(
        codebase_version="v2.1",
        robot_type="widowX",
        total_episodes=len(episodes_rows),
        total_frames=global_index,
        total_tasks=len(tasks_rows),
        total_videos=total_videos,
        total_chunks=len(episodes_rows) // n_eps_per_shard,
        chunks_size=n_eps_per_shard,
        fps=fps,
        splits={split_key: f"0:{len(episodes_rows)}"},
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/observation.images.top/episode_{episode_index:06d}.mp4",
        features=features,
    )

    if verbose:
        print(f"[convert_one_split] Writing metadata for split={split_name}", flush=True)
    # Write metadata files
    write_json(os.path.join(meta_dir, "info.json"), info)
    write_jsonl(os.path.join(meta_dir, "episodes.jsonl"), episodes_rows)
    write_jsonl(os.path.join(meta_dir, "tasks.jsonl"), tasks_rows)
    write_jsonl(os.path.join(meta_dir, "episodes_stats.jsonl"), episode_stats_rows)
    write_json(os.path.join(meta_dir, "stats.json"), stats_json)
    write_json(os.path.join(meta_dir, "norm_stats.json"), norm_stats_json)
    if verbose:
        print(f"[convert_one_split] Done split={split_name}: episodes={len(episodes_rows)} frames={stats_json['observation.state']['count']} videos={total_videos}", flush=True)
    # Remove temp dir unless asked to keep
    if use_temp and not keep_temp:
        try:
            shutil.rmtree(temp_dir)
            if verbose:
                print(f"[convert_one_split] Removed temp dir: {temp_dir}", flush=True)
        except Exception as e:
            if verbose:
                print(f"[convert_one_split] Failed to remove temp dir {temp_dir}: {e}", flush=True)


def convert(
    input_dir: str,
    output_root: str,
    n_eps_per_shard: int = 25,
    fps: int = 24,
    keep_last_n: int = 15,
    val_split_percent: float = 0.0,
    verbose: bool = True,
    num_workers: int = 1,
    export_videos: bool = True,
    use_temp: bool = True,
    keep_temp: bool = False,
) -> None:
    if verbose:
        print("[convert] Listing episodes...", flush=True)
    npz_files = list_npz_files(input_dir)
    if len(npz_files) == 0:
        raise RuntimeError(f"No .npz files found in {input_dir}")

    total_eps = len(npz_files)
    val_count = int(total_eps * max(0.0, min(1.0, val_split_percent)))
    train_count = total_eps - val_count

    if verbose:
        print(f"[convert] total_eps={total_eps} train={train_count} val={val_count} val_split_percent={val_split_percent}", flush=True)

    train_files = npz_files[:train_count]
    val_files = npz_files[train_count:] if val_count > 0 else []

    if verbose:
        print("[convert] Converting train split...", flush=True)
    if val_count == 0:
        # No validation: write directly under <output_root>/data
        convert_one_split(
            train_files,
            output_root=output_root,
            split_name="train",  # still used for meta/logging
            n_eps_per_shard=n_eps_per_shard,
            fps=fps,
            keep_last_n=keep_last_n,
            verbose=verbose,
            num_workers=num_workers,
            export_videos=export_videos,
            use_temp=use_temp,
            keep_temp=keep_temp,
            data_dir_override=os.path.join(output_root, "data"),
            videos_dir_override=os.path.join(output_root, "videos"),
            meta_dir_override=os.path.join(output_root, "meta"),
            temp_dir_override=os.path.join(output_root, "_temp"),
        )
    else:
        convert_one_split(
            train_files,
            output_root=output_root,
            split_name="train",
            n_eps_per_shard=n_eps_per_shard,
            fps=fps,
            keep_last_n=keep_last_n,
            verbose=verbose,
            num_workers=num_workers,
            export_videos=export_videos,
            use_temp=use_temp,
            keep_temp=keep_temp,
        )

    if len(val_files) > 0:
        if verbose:
            print("[convert] Converting val split...", flush=True)
        convert_one_split(
            val_files,
            output_root=output_root,
            split_name="val",
            n_eps_per_shard=n_eps_per_shard,
            fps=fps,
            keep_last_n=keep_last_n,
            verbose=verbose,
            num_workers=num_workers,
            export_videos=export_videos,
            use_temp=use_temp,
            keep_temp=keep_temp,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert ManiSkill .npz trajectories to LeRobot v2.1 dataset",
    )
    parser.add_argument(
        "--input_dir",
        default="/mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/data/maniskill/PutSpoonOnTableClothInScene-v1/150/data_append",
    )
    parser.add_argument(
        "--output_root",
        default="/mnt/mnt/public/liuzhihao/openpi-main/data/maniskill/PutSpoonOnTableClothInScene-v1/lerobot_v2_1",
    )
    parser.add_argument("--n_eps_per_shard", type=int, default=25)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--keep_last_n", type=int, default=15)
    parser.add_argument("--val_split_percent", type=float, default=0.0, help="Fraction of episodes for validation [0,1)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging and progress bars")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers for preprocessing")
    parser.add_argument("--export_videos", action="store_true", default=True, help="Export mp4 videos per episode")
    parser.add_argument("--no_export_videos", dest="export_videos", action="store_false")
    parser.add_argument("--use_temp", action="store_true", default=True, help="Store filtered arrays in temp to avoid IPC")
    parser.add_argument("--no_use_temp", dest="use_temp", action="store_false")
    parser.add_argument("--keep_temp", action="store_true", default=False, help="Keep temp files for debugging")
    args = parser.parse_args()

    if args.verbose:
        print("[main] Arguments:", flush=True)
        print({
            "input_dir": args.input_dir,
            "output_root": args.output_root,
            "n_eps_per_shard": args.n_eps_per_shard,
            "fps": args.fps,
            "keep_last_n": args.keep_last_n,
            "val_split_percent": args.val_split_percent,
            "verbose": args.verbose,
            "num_workers": args.num_workers,
            "export_videos": args.export_videos,
            "use_temp": args.use_temp,
            "keep_temp": args.keep_temp,
        }, flush=True)

    convert(
        input_dir=args.input_dir,
        output_root=args.output_root,
        n_eps_per_shard=args.n_eps_per_shard,
        fps=args.fps,
        keep_last_n=args.keep_last_n,
        val_split_percent=args.val_split_percent,
        verbose=args.verbose,
        num_workers=args.num_workers,
        export_videos=args.export_videos,
        use_temp=args.use_temp,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    main()


