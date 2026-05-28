#!/usr/bin/env python
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
"""
Convert raw LIBERO-Object HDF5 demos to a LeRobot v2.1 dataset with
LingBot-VA's `action_config` annotation injected into episodes.jsonl.

Source (download separately):
    datasets/libero_object_hdf5/libero_object/*.hdf5   (10 files, 50 demos each)

Output (LeRobot v2.1 layout, ready for select_subset.py + extract_latents.py):
    datasets/libero_object_hdf5_prepared/
        meta/{info,tasks,episodes,episodes_stats}.jsonl + info.json
        data/chunk-000/episode_NNNNNN.parquet
        videos/chunk-000/observation.images.agentview_rgb/episode_NNNNNN.mp4
        videos/chunk-000/observation.images.eye_in_hand_rgb/episode_NNNNNN.mp4

Key contracts (must match the eval client + the LingBot-VA loader):
  * BOTH cameras get a vertical flip (`frames[:, ::-1]`). HDF5 stores
    OpenGL-orientation frames; the eval client also flips sim output, so
    training data must be in display orientation to match inference.
  * Action stays in raw 7-dim LIBERO format with gripper in [-1, +1].
    Padding to the 30-dim standard layout happens at load time via
    `inverse_used_action_channel_ids` — DO NOT pre-pad here.
  * Camera keys must be exactly `observation.images.agentview_rgb` and
    `observation.images.eye_in_hand_rgb` (see va_libero_cfg.obs_cam_keys).
  * Task order is the LIBERO benchmark order 0..9 (matches §1's per-task
    SR table; do not reorder by HDF5 mtime / alphabetical filename).
  * Video encoding: libx264, yuv420p, CRF 18 at 20 fps (near-lossless
    for VAE input — extract_latents.py reads these mp4s back).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

os.environ.setdefault("MUJOCO_GL", "egl")  # avoid headless GL crash on benchmark import
from libero.libero import benchmark  # noqa: E402

FPS = 20
HEIGHT = 128
WIDTH = 128
CHUNK_SIZE = 1000  # one chunk holds all 500 episodes
CRF = 18
PIX_FMT = "yuv420p"
CODEC = "libx264"
CAM_KEYS = [
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
]
HDF5_CAM_KEYS = {  # LeRobot-side key -> HDF5-side path under data/demo_N/obs/
    "observation.images.agentview_rgb": "agentview_rgb",
    "observation.images.eye_in_hand_rgb": "eye_in_hand_rgb",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def find_hdf5_for_task(hdf5_dir: Path, task_name: str) -> Path:
    """LIBERO HDF5 filenames are `<task.name>_demo.hdf5`."""
    candidate = hdf5_dir / f"{task_name}_demo.hdf5"
    if candidate.exists():
        return candidate
    # Fallback: match by stripping `_demo` suffix and comparing slugs.
    slug = re.sub(r"\s+", "_", task_name.strip().lower())
    for p in sorted(hdf5_dir.glob("*_demo.hdf5")):
        if p.stem.replace("_demo", "") == slug:
            return p
    raise FileNotFoundError(
        f"No HDF5 found for task '{task_name}' under {hdf5_dir}"
    )


def encode_video_ffmpeg(frames: np.ndarray, out_path: Path, fps: int) -> None:
    """Pipe a (T,H,W,3) uint8 RGB array through ffmpeg at libx264 CRF 18."""
    assert frames.ndim == 4 and frames.shape[-1] == 3 and frames.dtype == np.uint8
    T, H, W, _ = frames.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{W}x{H}",
        "-framerate", str(fps),
        "-i", "-",
        "-c:v", CODEC,
        "-pix_fmt", PIX_FMT,
        "-crf", str(CRF),
        "-preset", "medium",
        "-an",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(input=np.ascontiguousarray(frames).tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def feature_stats(arr: np.ndarray) -> dict:
    """Per-dim {min,max,mean,std,count} for a (T, D) action/state array."""
    arr = arr.astype(np.float64, copy=False)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def video_stats_placeholder(channels: int = 3) -> dict:
    """Per-channel placeholder stats. LatentLeRobotDataset doesn't aggregate
    these at training time (self.episodes is None), so the values are not
    load-bearing — but the schema slot must be present."""
    return {
        "min": [0.0] * channels,
        "max": [1.0] * channels,
        "mean": [0.5] * channels,
        "std": [0.25] * channels,
        "count": [1],
    }


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def convert(src: Path, dst: Path, benchmark_name: str = "libero_object") -> None:
    if dst.exists():
        raise SystemExit(
            f"Destination exists: {dst}. Remove it first to avoid mixing runs."
        )
    bench_dict = benchmark.get_benchmark_dict()
    bench = bench_dict[benchmark_name]()
    n_tasks = bench.get_num_tasks()
    print(f"[convert] {benchmark_name}: {n_tasks} tasks")

    data_dir = dst / "data" / "chunk-000"
    meta_dir = dst / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    for cam in CAM_KEYS:
        (dst / "videos" / "chunk-000" / cam).mkdir(parents=True, exist_ok=True)

    # tasks.jsonl: one line per unique task language, deterministic order.
    task_languages: list[str] = []
    for t_idx in range(n_tasks):
        task = bench.get_task(t_idx)
        lang = getattr(task, "language", None) or task.name.replace("_", " ")
        task_languages.append(lang)
    tasks_path = meta_dir / "tasks.jsonl"
    with tasks_path.open("w") as f:
        for ti, lang in enumerate(task_languages):
            f.write(json.dumps({"task_index": ti, "task": lang}) + "\n")

    # Sweep: per task, per demo, emit parquet + 2 mp4s, accumulate metadata.
    episode_lines: list[dict] = []
    episode_stats_lines: list[dict] = []
    global_idx = 0      # cumulative frame index across all episodes
    episode_index = 0   # cumulative episode index
    total_frames = 0

    for t_idx in range(n_tasks):
        task = bench.get_task(t_idx)
        lang = task_languages[t_idx]
        hdf5_path = find_hdf5_for_task(src, task.name)
        print(f"[convert] task {t_idx}: {task.name}  ({hdf5_path.name})")

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(
                (k for k in f["data"].keys() if k.startswith("demo_")),
                key=lambda k: int(k.split("_")[1]),
            )
            for demo_key in demo_keys:
                grp = f["data"][demo_key]
                actions = np.array(grp["actions"], dtype=np.float32)  # (T, 7)
                T = actions.shape[0]
                # observation.state: 8-dim (joint_states 7 + gripper 1) is the
                # LIBERO convention. Fall back to whatever exists.
                if "obs/joint_states" in grp and "obs/gripper_states" in grp:
                    joint = np.array(grp["obs/joint_states"], dtype=np.float32)
                    gripper = np.array(grp["obs/gripper_states"], dtype=np.float32)
                    if gripper.ndim == 1:
                        gripper = gripper[:, None]
                    state = np.concatenate([joint, gripper], axis=1)
                elif "obs/ee_states" in grp:
                    state = np.array(grp["obs/ee_states"], dtype=np.float32)
                else:
                    state = np.zeros((T, 8), dtype=np.float32)

                # Cameras: HDF5 -> uint8 (T, H, W, 3) in OpenGL orientation.
                # Flip vertically so display orientation matches the eval client
                # (which also flips sim output before sending to the server).
                cam_frames: dict[str, np.ndarray] = {}
                for cam_key, hdf5_name in HDF5_CAM_KEYS.items():
                    arr = np.array(grp[f"obs/{hdf5_name}"])
                    if arr.dtype != np.uint8:
                        arr = arr.astype(np.uint8)
                    arr = np.ascontiguousarray(arr[:, ::-1])  # vertical flip
                    assert arr.shape == (T, HEIGHT, WIDTH, 3), (
                        f"unexpected {cam_key} shape {arr.shape} for {demo_key}"
                    )
                    cam_frames[cam_key] = arr

                # Write per-episode parquet.
                timestamps = (np.arange(T) / FPS).astype(np.float32)
                frame_indices = np.arange(T, dtype=np.int64)
                ep_df = pd.DataFrame({
                    "action": list(actions),
                    "observation.state": list(state),
                    "timestamp": timestamps,
                    "frame_index": frame_indices,
                    "episode_index": np.full(T, episode_index, dtype=np.int64),
                    "index": np.arange(global_idx, global_idx + T, dtype=np.int64),
                    "task_index": np.full(T, t_idx, dtype=np.int64),
                })
                ep_df.to_parquet(
                    data_dir / f"episode_{episode_index:06d}.parquet",
                    index=False,
                )

                # Write mp4s.
                for cam_key, frames in cam_frames.items():
                    out_mp4 = (
                        dst / "videos" / "chunk-000" / cam_key
                        / f"episode_{episode_index:06d}.mp4"
                    )
                    encode_video_ffmpeg(frames, out_mp4, fps=FPS)

                # episodes.jsonl line — with the LingBot-VA `action_config`
                # field. Single segment covering the whole episode.
                episode_lines.append({
                    "episode_index": episode_index,
                    "tasks": [lang],
                    "length": int(T),
                    "action_config": [{
                        "start_frame": 0,
                        "end_frame": int(T),
                        "action_text": lang,
                    }],
                })

                # episodes_stats.jsonl line.
                episode_stats_lines.append({
                    "episode_index": episode_index,
                    "stats": {
                        "action": feature_stats(actions),
                        "observation.state": feature_stats(state),
                        "observation.images.agentview_rgb": video_stats_placeholder(),
                        "observation.images.eye_in_hand_rgb": video_stats_placeholder(),
                        "timestamp": feature_stats(timestamps[:, None]),
                        "frame_index": feature_stats(frame_indices[:, None].astype(np.float32)),
                        "episode_index": feature_stats(
                            np.full((T, 1), episode_index, dtype=np.float32)
                        ),
                        "index": feature_stats(
                            np.arange(global_idx, global_idx + T,
                                      dtype=np.float32)[:, None]
                        ),
                        "task_index": feature_stats(
                            np.full((T, 1), t_idx, dtype=np.float32)
                        ),
                    },
                })

                episode_index += 1
                global_idx += T
                total_frames += T

    # Write episodes.jsonl + episodes_stats.jsonl
    with (meta_dir / "episodes.jsonl").open("w") as f:
        for line in episode_lines:
            f.write(json.dumps(line) + "\n")
    with (meta_dir / "episodes_stats.jsonl").open("w") as f:
        for line in episode_stats_lines:
            f.write(json.dumps(line) + "\n")

    # info.json — schema declaration LeRobot reads on dataset open.
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": episode_index,
        "total_frames": total_frames,
        "total_tasks": n_tasks,
        "total_videos": episode_index * len(CAM_KEYS),
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{episode_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": [
                    "ee_x", "ee_y", "ee_z",
                    "ee_rx", "ee_ry", "ee_rz",
                    "gripper",
                ],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state.shape[1]],
                "names": [f"s{i}" for i in range(state.shape[1])],
            },
            "observation.images.agentview_rgb": {
                "dtype": "video",
                "shape": [HEIGHT, WIDTH, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": float(FPS),
                    "video.codec": "h264",
                    "video.pix_fmt": PIX_FMT,
                    "video.is_depth_map": False,
                    "video.has_audio": False,
                },
            },
            "observation.images.eye_in_hand_rgb": {
                "dtype": "video",
                "shape": [HEIGHT, WIDTH, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": float(FPS),
                    "video.codec": "h264",
                    "video.pix_fmt": PIX_FMT,
                    "video.is_depth_map": False,
                    "video.has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with (meta_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    print(
        f"[convert] done. episodes={episode_index} frames={total_frames} "
        f"-> {dst}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        default=Path("datasets/libero_object_hdf5/libero_object"),
        help="Directory of `*_demo.hdf5` files",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("datasets/libero_object_hdf5_prepared"),
        help="Output LeRobot v2.1 dataset directory",
    )
    p.add_argument(
        "--benchmark",
        default="libero_object",
        help="LIBERO benchmark name (default: libero_object)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH; install it or use the imageio binary.")
    convert(args.src, args.dst, benchmark_name=args.benchmark)
