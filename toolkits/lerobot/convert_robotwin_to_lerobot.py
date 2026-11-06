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

"""Convert RoboTwin 2.0 HDF5 demonstrations into a LeRobot v3 dataset.

Each input directory is one extracted RoboTwin release, e.g.
``click_bell/aloha-agilex_clean_50``, containing ``data/episode<i>.hdf5`` and
``instructions/episode<i>.json``. Frame ``t`` stores the 14-dim joint state and
the three camera images at ``t`` and uses the joint state at ``t + 1`` as its
action, following the RoboTwin temporal contract. Frames cycle through the
episode's ``seen`` instructions.

Usage
-----
    python toolkits/lerobot/convert_robotwin_to_lerobot.py \\
        --raw-root click_bell/aloha-agilex_clean_50 \\
        --output-root datasets/robotwin2_lerobot/click_bell_clean_50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERA_MAP = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}
IMAGE_HW = (480, 640)
FPS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        nargs="+",
        help="Extracted RoboTwin release directories; all episodes are converted.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination LeRobot dataset directory; must not exist.",
    )
    return parser.parse_args()


def decode_frame(encoded: np.bytes_) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode a RoboTwin camera frame")
    height, width = IMAGE_HW
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def load_instructions(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        instructions = json.load(handle).get("seen", [])
    if not instructions:
        raise ValueError(f"No seen instructions in {path}")
    return instructions


def joint_vector(handle: h5py.File, index: int) -> np.ndarray:
    """Return ``[left_arm(6), left_gripper, right_arm(6), right_gripper]``."""
    return np.concatenate(
        [
            handle["joint_action/left_arm"][index],
            np.atleast_1d(handle["joint_action/left_gripper"][index]),
            handle["joint_action/right_arm"][index],
            np.atleast_1d(handle["joint_action/right_gripper"][index]),
        ]
    ).astype(np.float32)


def create_dataset(output_root: Path) -> LeRobotDataset:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": ["joints"]},
        "action": {"dtype": "float32", "shape": (14,), "names": ["joints"]},
    }
    for target_key in CAMERA_MAP.values():
        features[target_key] = {
            "dtype": "image",
            "shape": (3, *IMAGE_HW),
            "names": ["channels", "height", "width"],
        }
    return LeRobotDataset.create(
        repo_id=output_root.name,
        root=output_root,
        fps=FPS,
        robot_type="aloha",
        features=features,
        use_videos=False,
    )


def convert_episode(dataset: LeRobotDataset, raw_root: Path, episode_index: int) -> int:
    hdf5_path = raw_root / "data" / f"episode{episode_index}.hdf5"
    instructions = load_instructions(
        raw_root / "instructions" / f"episode{episode_index}.json"
    )
    with h5py.File(hdf5_path, "r") as handle:
        num_raw_frames = handle["joint_action/left_arm"].shape[0]
        if num_raw_frames < 2:
            raise ValueError(f"Episode has fewer than two frames: {hdf5_path}")
        for frame_index in range(num_raw_frames - 1):
            frame = {
                "observation.state": joint_vector(handle, frame_index),
                "action": joint_vector(handle, frame_index + 1),
                "task": instructions[frame_index % len(instructions)],
            }
            for source_key, target_key in CAMERA_MAP.items():
                encoded = handle[f"observation/{source_key}/rgb"][frame_index]
                frame[target_key] = decode_frame(encoded)
            dataset.add_frame(frame)
    dataset.save_episode(parallel_encoding=False)
    return num_raw_frames - 1


def main() -> None:
    args = parse_args()
    dataset = create_dataset(args.output_root.resolve())
    total_episodes = total_frames = 0
    for raw_root in args.raw_root:
        raw_root = raw_root.resolve()
        num_episodes = len(list((raw_root / "data").glob("episode*.hdf5")))
        if num_episodes == 0:
            raise FileNotFoundError(f"No episode*.hdf5 under {raw_root / 'data'}")
        for episode_index in range(num_episodes):
            total_frames += convert_episode(dataset, raw_root, episode_index)
        total_episodes += num_episodes
        print(f"Converted {num_episodes} episodes from {raw_root}", flush=True)
    dataset.finalize()
    print(
        f"Wrote {total_episodes} episodes / {total_frames} frames to "
        f"{args.output_root.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
