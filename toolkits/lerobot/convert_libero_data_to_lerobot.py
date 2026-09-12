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

"""Convert the public LIBERO RLDS demonstrations to StreamingVLA LeRobot data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.utils.logging import get_logger

_DEFAULT_RAW_DATASET_NAMES = (
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
)
_ACTION_DIM = 7


def _compute_action_states(actions: np.ndarray) -> np.ndarray:
    """Return the cumulative action trajectory at the start of each step."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != _ACTION_DIM:
        raise ValueError(
            f"Expected actions with shape (steps, {_ACTION_DIM}), got {actions.shape}."
        )
    initial_state = np.zeros((1, _ACTION_DIM), dtype=np.float32)
    return np.cumsum(
        np.concatenate([initial_state, actions], axis=0), axis=0, dtype=np.float32
    )[:-1]


def _load_lerobot_api() -> tuple[Any, Path]:
    try:
        from lerobot.datasets.lerobot_dataset import (
            HF_LEROBOT_HOME,
            LeRobotDataset,
        )
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import (
            HF_LEROBOT_HOME,
            LeRobotDataset,
        )
    return LeRobotDataset, Path(HF_LEROBOT_HOME)


def _create_dataset(
    repo_name: str,
    *,
    image_writer_threads: int,
    image_writer_processes: int,
) -> Any:
    """Create the StreamingVLA LeRobot schema."""
    lerobot_dataset, _ = _load_lerobot_api()
    return lerobot_dataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (_ACTION_DIM,),
                "names": ["actions"],
            },
            "action_states": {
                "dtype": "float32",
                "shape": (_ACTION_DIM,),
                "names": ["action_states"],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument(
        "--raw-dataset-names",
        nargs="+",
        default=list(_DEFAULT_RAW_DATASET_NAMES),
    )
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Convert all selected LIBERO suites into one LeRobot dataset."""
    args = _parse_args()
    if Path(args.repo_name).is_absolute() or args.repo_name in {"", "."}:
        raise ValueError("repo_name must be a non-empty relative Hugging Face repo id.")
    if args.image_writer_threads < 0 or args.image_writer_processes < 0:
        raise ValueError("Image writer counts must be non-negative.")

    _, lerobot_home = _load_lerobot_api()
    output_path = (lerobot_home / args.repo_name).resolve()
    home = lerobot_home.resolve()
    if output_path == home or home not in output_path.parents:
        raise ValueError(f"Refusing output path outside HF_LEROBOT_HOME: {output_path}")
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output dataset already exists at {output_path}; pass --overwrite "
                "to replace that exact dataset."
            )
        shutil.rmtree(output_path)

    try:
        import tensorflow_datasets as tfds
    except ImportError as error:
        raise ImportError(
            "LIBERO conversion requires tensorflow and tensorflow_datasets."
        ) from error
    from rlinf.data.storage.lerobot import add_frame_to_dataset

    dataset = _create_dataset(
        args.repo_name,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )
    logger = get_logger()
    episode_count = 0
    frame_count = 0
    for raw_dataset_name in args.raw_dataset_names:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=args.data_dir, split="train")
        for episode in raw_dataset:
            steps = list(episode["steps"].as_numpy_iterator())
            if not steps:
                logger.warning(f"Skipping empty LIBERO episode in {raw_dataset_name}")
                continue
            actions = np.stack(
                [np.asarray(step["action"], dtype=np.float32) for step in steps]
            )
            action_states = _compute_action_states(actions)
            for step, action, action_state in zip(
                steps, actions, action_states, strict=True
            ):
                language = step["language_instruction"]
                task = (
                    language.decode()
                    if isinstance(language, (bytes, bytearray))
                    else str(language)
                )
                add_frame_to_dataset(
                    dataset,
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": np.asarray(
                            step["observation"]["state"], dtype=np.float32
                        ),
                        "actions": action,
                        "action_states": action_state,
                        "task": task,
                    },
                )
                frame_count += 1
            dataset.save_episode()
            episode_count += 1

    logger.info(
        f"Converted {episode_count} LIBERO episodes and {frame_count} frames "
        f"to {output_path}"
    )
    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds", "streamingvla"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main()
