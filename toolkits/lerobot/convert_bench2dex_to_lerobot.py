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

"""Convert Bench2Dex replay HDF5 to LeRobot RGB and active-joint demonstrations.

Run with the Bench2Dex/OpenPI environment (LeRobot 0.3.3, dataset format v2.1).
The output includes norm_stats.json for OpenPI and meta/bench2dex.json recording
joint order and the retained frame ranges of each source episode.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

CAMERAS = ("cam_stereo_left", "cam_stereo_right", "cam_wrist_left", "cam_wrist_right")


def active_joint_indices(
    h5file: Any, dof_maps: dict
) -> tuple[str, list[str], list[int]]:
    """Select active names in the HDF5 runtime order, never URDF index order."""
    robot_key = h5file["meta/robot_key"].asstr()[()]
    names = h5file["robot/joint_names"].asstr()[:].tolist()
    spec = dof_maps["robots"][robot_key]
    if len(names) != len(set(names)) or set(names) != set(spec["full_joint_names"]):
        raise ValueError(
            f"Runtime joint names for {robot_key} do not match its DOF map."
        )
    for key in ("robot/qpos", "action/commanded"):
        if h5file[key].ndim != 2 or h5file[key].shape[1] != len(names):
            raise ValueError(f"{key} must have one column per runtime joint name.")
    active = set(spec["active_joint_names"])
    indices = [i for i, name in enumerate(names) if name in active]
    if len(indices) != spec["active_dof"]:
        raise ValueError(
            f"Expected {spec['active_dof']} active joints, got {len(indices)}."
        )
    return robot_key, names, indices


def valid_segments(h5file: Any) -> list[np.ndarray]:
    """Exclude invalid/nonfinite and homing frames, splitting at validity gaps."""
    qpos = np.asarray(h5file["robot/qpos"], dtype=np.float32)
    actions = np.asarray(h5file["action/commanded"], dtype=np.float32)
    if qpos.shape != actions.shape or qpos.ndim != 2:
        raise ValueError(
            f"Expected matching (T,J) state/action arrays, got {qpos.shape}/{actions.shape}."
        )
    total = len(qpos)
    valid = np.isfinite(actions).all(axis=1) & np.isfinite(qpos).all(axis=1)
    for key in ("frame_valid", "action/action_valid"):
        if key in h5file:
            flag = np.asarray(h5file[key]).astype(bool).reshape(-1)
            if flag.shape != (total,):
                raise ValueError(
                    f"Expected {total} validity flags at {key}, got {flag.shape}."
                )
            valid &= flag
    if "meta/homing_start_sim_step" not in h5file:
        raise ValueError(
            "Replay requires meta/homing_start_sim_step; an absent marker "
            "cannot establish whether the episode contains homing frames."
        )
    cutoff = int(h5file["meta/homing_start_sim_step"][()])
    # The official collector records -1 when no homing event occurred.
    if cutoff >= 0:
        if "time/sim_step" not in h5file:
            raise ValueError(
                "Homing marker requires time/sim_step for safe truncation."
            )
        steps = np.asarray(h5file["time/sim_step"]).reshape(-1)
        if steps.shape != (total,) or np.any(np.diff(steps) < 0):
            raise ValueError(
                "Expected monotonically increasing time/sim_step matching qpos."
            )
        valid &= steps < cutoff
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    return list(np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1))


def validate_timing(h5file: Any, segments: list[np.ndarray], fps: int) -> None:
    """Check the simulation clock of retained frames against the output frame rate."""
    if "meta/physics_dt" not in h5file or "time/sim_step" not in h5file:
        raise ValueError("Replay timing requires meta/physics_dt and time/sim_step.")
    physics_dt = float(h5file["meta/physics_dt"][()])
    if not np.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError(f"Expected positive physics_dt, got {physics_dt}.")
    if "meta/effective_fps" in h5file:
        recorded_fps = float(h5file["meta/effective_fps"][()])
        if not np.isclose(recorded_fps, fps, rtol=1e-5, atol=1e-8):
            raise ValueError(
                f"Requested {fps} FPS but replay effective_fps is {recorded_fps}."
            )
    steps = np.asarray(h5file["time/sim_step"])
    for frames in segments:
        intervals = np.diff(steps[frames]) * physics_dt
        if not np.allclose(intervals, 1.0 / fps, rtol=1e-5, atol=1e-8):
            raise ValueError(
                f"Retained simulation intervals {np.unique(intervals).tolist()} "
                f"do not match {fps} FPS; resampling is required."
            )


def decode_rgb(dataset: Any, frame: int) -> np.ndarray:
    """Decode the recorded JPEG or an uncompressed RGB frame without resizing."""
    from PIL import Image

    image = np.asarray(dataset[frame])
    if image.ndim == 3 and image.shape[-1] == 3:
        return image.astype(np.uint8)
    with Image.open(io.BytesIO(bytes(image))) as decoded:
        return np.asarray(decoded.convert("RGB"))


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    active_dof_map: Path,
    *,
    task: str,
    repo_id: str = "bench2dex/03_ability",
    fps: int = 20,
    image_writer_threads: int = 4,
    max_episodes: int | None = None,
    max_frames: int | None = None,
) -> dict:
    """Write four RGB views, active-joint states and absolute position actions.

    ``max_frames`` caps each source episode for smoke tests. Statistics use
    exactly the exported frames. Existing output directories are rejected.
    """
    import h5py
    import yaml
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.data.storage.lerobot import add_frame_to_dataset

    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive.")
    paths = sorted(input_dir.glob("*.hdf5"))
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise FileNotFoundError(f"No HDF5 episodes found in {input_dir}.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if fps <= 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("fps and max_frames must be positive.")
    if image_writer_threads <= 0:
        raise ValueError("image_writer_threads must be positive.")
    dof_maps = yaml.safe_load(active_dof_map.read_text())
    with h5py.File(paths[0]) as first:
        robot_key, full_names, active_indices = active_joint_indices(first, dof_maps)
        active_names = [full_names[i] for i in active_indices]
        segments = valid_segments(first)
        validate_timing(first, segments, fps)
        if not segments:
            raise ValueError(f"No valid frames in {paths[0]}.")
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(active_names),),
                "names": active_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(active_names),),
                "names": active_names,
            },
            "source_frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        }
        for camera in CAMERAS:
            image = decode_rgb(first[f"cameras/{camera}/rgb"], int(segments[0][0]))
            features[f"observation.images.{camera}"] = {
                "dtype": "image",
                "shape": image.shape,
                "names": ["height", "width", "channels"],
            }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        robot_type=robot_key,
        features=features,
        use_videos=False,
        image_writer_threads=image_writer_threads,
    )
    manifest = {
        "robot_key": robot_key,
        "full_joint_names": full_names,
        "active_joint_names": active_names,
        "active_indices": active_indices,
        "cameras": list(CAMERAS),
        "action_semantics": "absolute_joint_position_target",
        "joint_order": "runtime_joint_names",
        "fps": fps,
        "timebase": "time/sim_step * meta/physics_dt; retained intervals validated",
        "frame_filter": "frame_valid & action_valid & finite(state,action) & sim_step < homing_start",
        "invalid_gap_policy": "split_episode",
        "episodes": [],
        "max_frames_per_source": max_frames,
    }
    states, actions = [], []
    try:
        for path in paths:
            with h5py.File(path) as source:
                current_robot, current_names, indices = active_joint_indices(
                    source, dof_maps
                )
                if current_robot != robot_key or current_names != full_names:
                    raise ValueError(
                        f"Episode {path} changes robot or runtime joint order."
                    )
                segments = valid_segments(source)
                validate_timing(source, segments, fps)
                remaining = max_frames
                for frames in segments:
                    if remaining is not None:
                        frames = frames[:remaining]
                        remaining -= len(frames)
                    if not len(frames):
                        continue
                    episode_index = len(manifest["episodes"])
                    qpos = np.asarray(source["robot/qpos"])[frames][:, indices].astype(
                        np.float32
                    )
                    action = np.asarray(source["action/commanded"])[frames][
                        :, indices
                    ].astype(np.float32)
                    states.append(qpos)
                    actions.append(action)
                    for index, frame_index in enumerate(frames):
                        frame = {
                            "observation.state": qpos[index],
                            "action": action[index],
                            "source_frame_index": np.asarray(
                                [frame_index], dtype=np.int64
                            ),
                            "task": task,
                        }
                        for camera in CAMERAS:
                            frame[f"observation.images.{camera}"] = decode_rgb(
                                source[f"cameras/{camera}/rgb"], int(frame_index)
                            )
                        add_frame_to_dataset(dataset, frame)
                    dataset.save_episode()
                    if dataset.meta.info["codebase_version"] == "v2.1":
                        # This is a write-only flow; v2 otherwise retains every image.
                        dataset.hf_dataset = dataset.create_hf_dataset()
                    manifest["episodes"].append(
                        {
                            "episode_index": episode_index,
                            "source_file": path.name,
                            "source_frame_start": int(frames[0]),
                            "source_frame_stop": int(frames[-1]) + 1,
                            "length": len(frames),
                        }
                    )
    finally:
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        dataset.stop_image_writer()
    if not states:
        raise ValueError("No frames remain after validity and homing filtering.")
    norm_stats = {}
    for key, chunks in (("state", states), ("actions", actions)):
        values = np.concatenate(chunks).astype(np.float64)
        norm_stats[key] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    (output_dir / "norm_stats.json").write_text(
        json.dumps({"norm_stats": norm_stats}, indent=2) + "\n"
    )
    (output_dir / "meta/bench2dex.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    """Convert a directory of replay episodes using an explicit robot DOF map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-dof-map", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--repo-id", default="bench2dex/03_ability")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-frames", type=int)
    convert_dataset(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
