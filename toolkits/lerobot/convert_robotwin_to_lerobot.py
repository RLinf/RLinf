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

"""Convert RoboTwin HDF5 episodes to LeRobot v3 with next-frame joint actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("lerobot.datasets"):
        raise
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from rlinf.data.storage.lerobot import add_frame_to_dataset
from rlinf.utils.logging import get_logger

logger = get_logger()

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
    """Decode one RoboTwin camera frame as an RGB image."""
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode a RoboTwin camera frame")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            "shape": (*IMAGE_HW, 3),
            "names": ["height", "width", "channels"],
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
            add_frame_to_dataset(dataset, frame)
    dataset.save_episode()
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
        logger.info("Converted %d episodes from %s", num_episodes, raw_root)
    logger.info(
        "Wrote %d episodes / %d frames to %s",
        total_episodes,
        total_frames,
        args.output_root.resolve(),
    )


if __name__ == "__main__":
    main()
