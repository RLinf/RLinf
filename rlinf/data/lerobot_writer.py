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

"""LeRobot dataset writer for saving rollout data."""

import gc
import io
import json
import os
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from rlinf.utils.logging import get_logger
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset



class LeRobotDatasetWriter:
    """
    Wrapper for LeRobotDataset that provides a simplified interface for writing episodes.

    Usage:
        writer = LeRobotDatasetWriter()
        writer.create(
            repo_id="my-dataset",
            robot_type="franka_panda",
            fps=5,
            features={...}
        )
        
        for episode_data in episodes:
            writer.add_episode(episode_data)
        
        writer.finalize(push_to_hub=False)
    """

    def __init__(self):
        """Initialize the writer."""
        self.dataset: LeRobotDataset | None = None
        self.logger = get_logger()


    
    def create(
        self,
        repo_id: str,
        robot_type: str = "franka_panda",
        fps: int = 5,
        features: dict[str, dict[str, Any]] | None = None,
        image_writer_threads: int = 10,
        image_writer_processes: int = 5,
        image_shape: tuple[int, int, int] = (256, 256, 3),
        state_dim: int = 8,
        action_dim: int = 7,
        has_wrist_image: bool = True,
        has_extra_view_image: bool = True,
    ) -> None:
        """
        Create a new LeRobot dataset.

        Args:
            repo_id: The identifier for the new LeRobot dataset
            robot_type: Robot type (default "franka_panda")
            fps: Frame rate (default 5)
            features: Feature schema dictionary defining the dataset structure. If None, auto-generated from dimensions.
            image_writer_threads: Number of threads for image writing
            image_writer_processes: Number of processes for image writing
            image_shape: Image shape (H, W, C) for auto-generated features
            state_dim: State dimension for auto-generated features
            action_dim: Action dimension for auto-generated features
            has_wrist_image: Whether to include wrist_image in auto-generated features
            has_extra_view_image: Whether to include extra_view_image in auto-generated features

        """
        if features is None:
            features = {
                "image": {
                    "dtype": "image",
                    "shape": list(image_shape),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (state_dim, ),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (action_dim, ),
                    "names": ["actions"],
                },
                "done": {
                    "dtype": "bool",
                    "shape": (1, ),
                    "names": ["done"],
                },
                "is_success": {
                    "dtype": "bool",
                    "shape": (1, ),
                    "names": ["is_success"],
                },
            }
            if has_wrist_image:
                features["wrist_image"] = {
                    "dtype": "image",
                    "shape": list(image_shape),
                    "names": ["height", "width", "channel"],
                }
            if has_extra_view_image:
                features["extra_view_image"] = {
                    "dtype": "image",
                    "shape": list(image_shape),
                    "names": ["height", "width", "channel"],
                }

        self.logger.info(f"Creating LeRobot dataset: repo_id={repo_id}, robot_type={robot_type}, fps={fps}")
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=robot_type,
            fps=fps,
            features=features,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )

    def add_episode(self, episode_data: list[dict[str, Any]]) -> None:
        """
        Add an episode to the dataset.

        Args:
            episode_data: List of frame dictionaries, where each frame contains:
                - image: np.ndarray [H, W, C]
                - wrist_image: np.ndarray [H, W, C] (optional)
                - state: np.ndarray [state_dim]
                - actions: np.ndarray [action_dim]
                - task: str (task instruction)
                - Any other fields defined in the features schema

        The frames will be automatically processed to include both the original
        image format and the observation.images format (transposed to [C, H, W]).
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not created. Call create() first.")

        if not episode_data:
            self.logger.warning("Empty episode_data provided, skipping.")
            return
        # breakpoint()
        for frame_data in episode_data:

            self.dataset.add_frame(frame_data)

        self.dataset.save_episode()
        self.logger.info(f"Saved episode with {len(episode_data)} frames, task: '{episode_data[0].get('task', 'N/A')}'")

    def finalize(self) -> None:
        """Finalize the dataset."""
        if self.dataset is None:
            raise RuntimeError("Dataset not created. Call create() first.")
        
        if hasattr(self.dataset, 'image_writer') and self.dataset.image_writer is not None:
            self.dataset.image_writer.wait_until_done()
        
        if hasattr(self.dataset, 'image_writer') and self.dataset.image_writer is not None:
            self.dataset.image_writer.stop()
            self.dataset.image_writer = None
        
        if hasattr(self.dataset, 'episode_buffer'):
            self.dataset.episode_buffer = None
        
        if hasattr(self.dataset, 'hf_dataset'):
            self.dataset.hf_dataset = None
        
        del self.dataset
        gc.collect()
        self.dataset = None
        self.logger.info("Dataset finalized.")

def merge_distributed_datasets(
    base_dir: str,
    output_dir: str,
    pattern: str = "*_stage*_rank*",
    robot_type: str = "panda",
    fps: int = 10,
) -> int:
    """
    Merge data directories saved by multiple distributed workers.

    In distributed training, each worker saves data to a different subdirectory:
        base_dir/
        ├── collected_data_stage0_rank0/
        ├── collected_data_stage0_rank1/
        ├── collected_data_stage1_rank0/
        └── ...

    This function merges all subdirectories into a single unified output directory.

    Args:
        base_dir: Parent directory containing multiple worker data directories
        output_dir: Output directory for the merged dataset
        pattern: Glob pattern to match worker directories
        robot_type: Robot type
        fps: Frame rate

    Returns:
        Total number of merged episodes

    Usage:
        merge_distributed_datasets(
            base_dir="/path/to/results/test_openpi",
            output_dir="/path/to/results/test_openpi/merged_data",
            pattern="collected_data_stage*_rank*"
        )
    """
    import glob

    logger = get_logger()
    logger.info(f"[merge] Starting merge from {base_dir} to {output_dir}")
    logger.info(f"[merge] Pattern: {pattern}")

    # Find all matching subdirectories
    search_pattern = os.path.join(base_dir, pattern)
    sub_dirs = sorted(glob.glob(search_pattern))

    if not sub_dirs:
        logger.warning(f"[merge] No directories found matching {search_pattern}")
        return 0

    logger.info(f"[merge] Found {len(sub_dirs)} directories to merge: {sub_dirs}")

    # Collect data from all episodes
    all_episodes_data = []  # [(parquet_path, episode_meta), ...]
    all_tasks = {}  # task_str -> task_index (globally re-indexed)

    for sub_dir in sub_dirs:
        # Check directory structure
        meta_dir = os.path.join(sub_dir, "meta")
        data_dir = os.path.join(sub_dir, "data")

        if not os.path.exists(meta_dir) or not os.path.exists(data_dir):
            logger.warning(
                f"[merge] Skipping {sub_dir}: missing meta or data directory"
            )
            continue

        # Read episodes.jsonl
        episodes_file = os.path.join(meta_dir, "episodes.jsonl")
        if not os.path.exists(episodes_file):
            logger.warning(f"[merge] Skipping {sub_dir}: missing episodes.jsonl")
            continue

        with open(episodes_file, "r") as f:
            for line in f:
                ep_meta = json.loads(line.strip())
                ep_idx = ep_meta["episode_index"]
                chunk_idx = ep_idx // 1000  # Assumes chunks_size=1000

                # Find the corresponding parquet file
                parquet_path = os.path.join(
                    data_dir, f"chunk-{chunk_idx:03d}", f"episode_{ep_idx:06d}.parquet"
                )

                if os.path.exists(parquet_path):
                    all_episodes_data.append((parquet_path, ep_meta, sub_dir))

                    # Collect tasks
                    for task in ep_meta.get("tasks", []):
                        if task not in all_tasks:
                            all_tasks[task] = len(all_tasks)
                else:
                    logger.warning(f"[merge] Parquet not found: {parquet_path}")

    logger.info(
        f"[merge] Collected {len(all_episodes_data)} episodes, {len(all_tasks)} unique tasks"
    )

    if not all_episodes_data:
        logger.warning("[merge] No episodes to merge")
        return 0

    # Read the first parquet to infer data dimensions
    first_table = pq.read_table(all_episodes_data[0][0])
    first_df = first_table.to_pandas()

    # Infer dimensions
    state_dim = len(first_df["state"].iloc[0])
    action_dim = len(first_df["actions"].iloc[0])
    image_shape = (256, 256, 3)  # Default; can be read from info.json

    # Try to read from info.json
    first_info_path = os.path.join(
        os.path.dirname(all_episodes_data[0][2]), "meta", "info.json"
    )
    if os.path.exists(first_info_path):
        with open(first_info_path, "r") as f:
            info = json.load(f)
            if "features" in info and "image" in info["features"]:
                image_shape = tuple(info["features"]["image"]["shape"])

    logger.info(
        f"[merge] Inferred dimensions: state_dim={state_dim}, action_dim={action_dim}, image_shape={image_shape}"
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data", "chunk-000"), exist_ok=True)

    # Re-index and copy parquet files
    global_frame_index = 0
    merged_episodes = []

    for new_ep_idx, (parquet_path, ep_meta, _) in enumerate(all_episodes_data):
        # Read original parquet
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        T = len(df)

        # Update indices
        df["episode_index"] = new_ep_idx
        df["index"] = range(global_frame_index, global_frame_index + T)

        # Update task_index (using global task mapping)
        task = ep_meta.get("tasks", ["unknown"])[0]
        df["task_index"] = all_tasks.get(task, 0)

        # Write new parquet
        chunk_idx = new_ep_idx // 1000
        chunk_dir = os.path.join(output_dir, "data", f"chunk-{chunk_idx:03d}")
        os.makedirs(chunk_dir, exist_ok=True)

        new_parquet_path = os.path.join(chunk_dir, f"episode_{new_ep_idx:06d}.parquet")

        # Rebuild table and write, preserving HuggingFace metadata
        new_table = pa.Table.from_pandas(df, preserve_index=False)

        # Add HuggingFace features metadata
        hf_features = {
            "info": {
                "features": {
                    "image": {"_type": "Image"},
                    "wrist_image": {"_type": "Image"},
                    "extra_view_image": {"_type": "Image"},
                    "state": {
                        "feature": {"dtype": "float32", "_type": "Value"},
                        "length": state_dim,
                        "_type": "Sequence",
                    },
                    "actions": {
                        "feature": {"dtype": "float32", "_type": "Value"},
                        "length": action_dim,
                        "_type": "Sequence",
                    },
                    "timestamp": {"dtype": "float32", "_type": "Value"},
                    "frame_index": {"dtype": "int64", "_type": "Value"},
                    "episode_index": {"dtype": "int64", "_type": "Value"},
                    "index": {"dtype": "int64", "_type": "Value"},
                    "task_index": {"dtype": "int64", "_type": "Value"},
                    "done": {"dtype": "bool", "_type": "Value"},
                    "is_success": {"dtype": "bool", "_type": "Value"},
                }
            }
        }
        new_schema = new_table.schema.with_metadata(
            {"huggingface": json.dumps(hf_features)}
        )
        new_table = new_table.cast(new_schema)
        pq.write_table(new_table, new_parquet_path)

        # Update episode metadata
        merged_episodes.append(
            {
                "episode_index": new_ep_idx,
                "tasks": ep_meta.get("tasks", []),
                "length": T,
                "is_success": ep_meta.get("is_success", False),
            }
        )

        global_frame_index += T

        if (new_ep_idx + 1) % 10 == 0:
            logger.info(
                f"[merge] Processed {new_ep_idx + 1}/{len(all_episodes_data)} episodes"
            )

    # Write meta files
    meta_dir = os.path.join(output_dir, "meta")

    # info.json
    total_episodes = len(merged_episodes)
    total_chunks = (total_episodes + 999) // 1000

    info = {
        "codebase_version": "v2.0",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": global_frame_index,
        "total_tasks": len(all_tasks),
        "total_videos": 0,
        "total_chunks": max(1, total_chunks),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "image": {
                "dtype": "image",
                "shape": list(image_shape),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": list(image_shape),
                "names": ["height", "width", "channel"],
            },
            "extra_view_image": {
                "dtype": "image",
                "shape": list(image_shape),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": [state_dim], "names": ["state"]},
            "actions": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": ["actions"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "done": {"dtype": "bool", "shape": [1], "names": None},
            "is_success": {"dtype": "bool", "shape": [1], "names": None},
        },
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # episodes.jsonl
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep in merged_episodes:
            f.write(json.dumps(ep) + "\n")

    # tasks.jsonl
    sorted_tasks = sorted(all_tasks.items(), key=lambda x: x[1])
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for task, task_index in sorted_tasks:
            f.write(json.dumps({"task_index": task_index, "task": task}) + "\n")

    logger.info(
        f"[merge] Merge complete: {total_episodes} episodes, {global_frame_index} frames -> {output_dir}"
    )

    return total_episodes
