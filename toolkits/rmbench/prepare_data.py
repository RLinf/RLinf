# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert RMBench demonstrations to LeRobot v2.1."""

import json
from io import BytesIO
from pathlib import Path

import datasets
import h5py
import numpy as np
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from openpi import transforms
from openpi.shared import normalize
from PIL import Image

from rlinf.data.storage.lerobot.writer import write_embedded_episode
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.utils.logging import get_logger

logger = get_logger()
FPS = 15
CAMERAS = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
JOINT_FIELDS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
    "action": {"dtype": "float32", "shape": (14,), "names": None},
    **{
        f"observation.images.{camera}": {
            "dtype": "image",
            "shape": (240, 320, 3),
            "names": ("height", "width", "channel"),
        }
        for camera in CAMERAS
    },
}


def _read_joints(group: h5py.Group) -> np.ndarray:
    return np.concatenate([group[key][:] for key in JOINT_FIELDS], axis=-1)


def _rgb_image(data: bytes) -> bytes:
    # XPolicyLab marks standard JPEGs with a COM segment before the scan data.
    offset = 2
    while offset + 4 <= len(data) and data[offset] == 0xFF:
        marker = data[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue
        if marker in (0xDA, 0xD9):
            break
        end = offset + 2 + int.from_bytes(data[offset + 2 : offset + 4], "big")
        if marker == 0xFE and data[offset + 4 : end] == b"XPL-RGB1":
            return data
        offset = end

    # Older RMBench JPEGs have the color channels reversed in the file.
    with Image.open(BytesIO(data)) as image:
        rgb = np.asarray(image.convert("RGB"))[:, :, ::-1]
    output = BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return output.getvalue()


def prepare_data(dataset_root: Path, output_dir: Path) -> None:
    """Write train/val LeRobot datasets and training-only OpenPI normalization."""
    tasks = sorted(dataset_root.glob("*/aloha_agilex/data"))
    if not tasks:
        raise ValueError(
            f"Expected RMBench task/aloha_agilex/data directories in {dataset_root}."
        )
    config = get_openpi_config("pi05_rmbench")
    transform = transforms.compose(config.data.policy_transforms().inputs)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    # Use the same Aloha transforms for norm stats without running image preprocessing.
    placeholder = {camera: np.zeros((1, 1, 3), dtype=np.uint8) for camera in CAMERAS}
    action_offsets = np.arange(config.model.action_horizon)
    datasets.disable_progress_bars()
    metadata = {
        split: LeRobotDatasetMetadata.create(
            repo_id=f"rmbench/{split}",
            fps=FPS,
            root=output_dir / split,
            robot_type="aloha_agilex",
            features=FEATURES,
            use_videos=False,
        )
        for split in ("train", "val")
    }
    for task in tqdm.tqdm(tasks, desc="Preparing RMBench tasks"):
        paths = sorted(task.glob("*.hdf5"))
        if len(paths) <= 5:
            raise ValueError(
                f"Expected more than 5 episodes in {task}, found {len(paths)}."
            )
        val_indices = set(np.random.default_rng(42).permutation(len(paths))[:5])
        for index, path in enumerate(paths):
            with h5py.File(path, "r") as episode:
                fps = episode["additional_info/frequency"][()]
                states = _read_joints(episode["state"])
                actions = _read_joints(episode["action"])
                prompt = json.loads(episode["instructions"][()])[0]
                images = {
                    f"observation.images.{camera}": [
                        _rgb_image(bits)
                        for bits in episode[f"vision/{source}/colors"][:]
                    ]
                    for camera, source in CAMERAS.items()
                }
            if fps != FPS:
                raise ValueError(f"Expected {FPS} Hz in {path}, got {fps} Hz.")
            frames = len(states)
            if (
                not frames
                or states.shape != (frames, 14)
                or actions.shape != (frames, 14)
            ):
                raise ValueError(
                    f"Expected aligned, nonempty 14D states and actions in {path}, "
                    f"got {states.shape} and {actions.shape}."
                )
            split = "val" if index in val_indices else "train"
            write_embedded_episode(
                metadata[split],
                {"observation.state": states, "action": actions},
                images,
                prompt,
            )
            if split == "train":
                _update_stats(
                    stats, transform, placeholder, states, actions, action_offsets
                )

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    normalize.save(output_dir, norm_stats)
    for split, info in metadata.items():
        logger.info(
            "%s: %d episodes, %d frames",
            split,
            info.total_episodes,
            info.total_frames,
        )
    logger.info(
        "Saved the LeRobot datasets and OpenPI norm_stats.json in %s", output_dir
    )


def _update_stats(stats, transform, images, states, actions, action_offsets) -> None:
    # Account for every action in every training window, including tail padding.
    for start in range(0, len(states), 256):
        anchors = np.arange(start, min(start + 256, len(states)))
        indices = np.minimum(anchors[:, None] + action_offsets, len(states) - 1)
        samples = [
            transform({"images": images, "state": states[anchor], "actions": chunk})
            for anchor, chunk in zip(anchors, actions[indices])
        ]
        stats["state"].update(np.stack([item["state"] for item in samples]))
        stats["actions"].update(np.concatenate([item["actions"] for item in samples]))


if __name__ == "__main__":
    tyro.cli(prepare_data)
