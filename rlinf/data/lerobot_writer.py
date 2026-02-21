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

import io
import json
import logging
import os
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# Debug log directory (override via DATA_COLLECTOR_LOG_DIR env var)
DEBUG_LOG_DIR = os.environ.get(
    "DATA_COLLECTOR_LOG_DIR",
    os.path.join(os.getcwd(), "logs", "data_collector_logs"),
)
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
_writer_logger = logging.getLogger("LeRobotDatasetWriter")
_writer_logger.setLevel(logging.DEBUG)
if not _writer_logger.handlers:
    _file_handler = logging.FileHandler(
        os.path.join(DEBUG_LOG_DIR, "lerobot_writer_debug.log")
    )
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    _writer_logger.addHandler(_file_handler)
_writer_logger.info(f"LeRobot writer debug log initialized. Log dir: {DEBUG_LOG_DIR}")


class LeRobotDatasetWriter:
    """
    Save rollout data as a LeRobot-format dataset.

    Directory structure:
        root_dir/
        ├── meta/
        │   ├── info.json           # Dataset metadata
        │   ├── episodes.jsonl      # Episode information
        │   ├── tasks.jsonl         # Task list
        │   └── stats.json          # Statistics
        └── data/
            └── chunk-000/
                ├── episode_000000.parquet
                ├── episode_000001.parquet
                └── ...

    Parquet Schema:
        - image: struct<bytes: binary, path: string>
        - wrist_image: struct<bytes: binary, path: string>
        - state: fixed_size_list<float>[state_dim]
        - actions: fixed_size_list<float>[action_dim]
        - timestamp: float
        - frame_index: int64
        - episode_index: int64
        - index: int64
        - task_index: int64
        - is_success: bool

    Usage:
        writer = LeRobotDatasetWriter("./output_data")
        for episode in episodes:
            writer.add_episode(
                images=episode["images"],
                wrist_images=episode["wrist_images"],
                states=episode["states"],
                actions=episode["actions"],
                task=episode["task"],
                is_success=episode["is_success"],
            )
        writer.finalize()

    Incremental Mode:
        writer = LeRobotDatasetWriter("./output_data", use_incremental_stats=True)
        # Stats are computed incrementally using Welford's algorithm
        # No need to store all raw data in memory
    """

    def __init__(
        self,
        root_dir: str,
        robot_type: str = "panda",
        fps: int = 10,
        image_shape: tuple[int, int, int] = (256, 256, 3),
        state_dim: int = 8,
        action_dim: int = 7,
        chunks_size: int = 1000,
        codebase_version: str = "v2.0",
        use_incremental_stats: bool = False,
        stats_sample_ratio: float = 0.1,
    ):
        """
        Initialize the LeRobot dataset writer.

        Args:
            root_dir: Dataset root directory
            robot_type: Robot type (default "panda")
            fps: Frame rate (default 10)
            image_shape: Image shape (H, W, C)
            state_dim: State dimension
            action_dim: Action dimension
            chunks_size: Maximum number of episodes per chunk
            codebase_version: LeRobot version string
            use_incremental_stats: Whether to use incremental statistics (Welford's algorithm) to avoid storing all raw data
            stats_sample_ratio: Sampling ratio for statistics (images only, to reduce memory usage)
        """
        self.root_dir = root_dir
        self.robot_type = robot_type
        self.fps = fps
        self.image_shape = image_shape
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunks_size = chunks_size
        self.codebase_version = codebase_version
        self.use_incremental_stats = use_incremental_stats
        self.stats_sample_ratio = stats_sample_ratio

        # Internal state
        self._episodes: list[dict[str, Any]] = []  # Episode metadata
        self._tasks: dict[str, int] = {}  # task_str -> task_index
        self._global_frame_index = 0  # Global frame counter

        if use_incremental_stats:
            # Accumulators using Welford's online algorithm (no raw data storage)
            self._stats_accumulators: dict[str, dict[str, Any]] = {
                "image": {
                    "sum": None,
                    "sum_sq": None,
                    "count": 0,
                    "min": None,
                    "max": None,
                },
                "wrist_image": {
                    "sum": None,
                    "sum_sq": None,
                    "count": 0,
                    "min": None,
                    "max": None,
                },
                "state": {
                    "sum": None,
                    "sum_sq": None,
                    "count": 0,
                    "min": None,
                    "max": None,
                },
                "actions": {
                    "sum": None,
                    "sum_sq": None,
                    "count": 0,
                    "min": None,
                    "max": None,
                },
                "timestamp": {
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "count": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                },
                "frame_index": {
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "count": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                },
                "episode_index": {
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "count": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                },
                "index": {
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "count": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                },
                "task_index": {
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "count": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                },
            }
        else:
            # Original approach: accumulate raw data
            self._stats_accumulators: dict[str, dict[str, Any]] = {
                "image": {"values": []},
                "wrist_image": {"values": []},
                "state": {"values": []},
                "actions": {"values": []},
                "timestamp": {"values": []},
                "frame_index": {"values": []},
                "episode_index": {"values": []},
                "index": {"values": []},
                "task_index": {"values": []},
            }

        # Create directory structure
        self._create_directories()

    def _create_directories(self) -> None:
        """Create the dataset directory structure."""
        os.makedirs(os.path.join(self.root_dir, "meta"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "data"), exist_ok=True)

    def _get_chunk_index(self, episode_index: int) -> int:
        """Get the chunk index that the episode belongs to."""
        return episode_index // self.chunks_size

    def _get_chunk_dir(self, chunk_index: int) -> str:
        """Get the chunk directory path."""
        chunk_dir = os.path.join(self.root_dir, "data", f"chunk-{chunk_index:03d}")
        os.makedirs(chunk_dir, exist_ok=True)
        return chunk_dir

    def _get_task_index(self, task: str) -> int:
        """Get the task index, creating a new one if it does not exist."""
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        return self._tasks[task]

    def _encode_image_to_png(self, image: np.ndarray) -> bytes:
        """Encode a numpy image to PNG bytes."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def _create_image_struct(
        self, image: np.ndarray, frame_index: int
    ) -> dict[str, Any]:
        """Create a LeRobot image struct format."""
        return {
            "bytes": self._encode_image_to_png(image),
            "path": f"frame_{frame_index:06d}.png",
        }

    def add_episode(
        self,
        images: np.ndarray,
        wrist_images: np.ndarray | None,
        states: np.ndarray,
        actions: np.ndarray,
        task: str,
        is_success: bool,
        dones: np.ndarray | None = None,
    ) -> int:
        """
        Add an episode to the dataset.

        Args:
            images: Main camera images [T, H, W, C] uint8
            wrist_images: Wrist camera images [T, H, W, C] uint8 or None
            states: State vectors [T, state_dim] float32
            actions: Action vectors [T, action_dim] float32
            task: Task description string
            is_success: Whether this is a successful trajectory
            dones: Per-step done flags [T] bool; if None, auto-generated (True only at last step)

        Returns:
            episode_index: Index of the added episode
        """
        episode_index = len(self._episodes)
        chunk_index = self._get_chunk_index(episode_index)
        chunk_dir = self._get_chunk_dir(chunk_index)
        task_index = self._get_task_index(task)

        T = len(images)

        # If dones is not provided, auto-generate (True at the last step, False elsewhere)
        if dones is None:
            _writer_logger.warning(
                f"[add_episode] ep={episode_index}: dones is None, auto-generating"
            )
            dones = np.zeros(T, dtype=bool)
            dones[-1] = True

        # Log detailed input data information
        _writer_logger.info(
            f"[add_episode] ep={episode_index}: T={T}, "
            f"images={images.shape}, states={states.shape}, actions={actions.shape}, "
            f"dones={dones.shape}, is_success={is_success}, task='{task}'"
        )
        # Check dones values
        num_true_dones = np.sum(dones)
        _writer_logger.info(
            f"[add_episode] ep={episode_index}: dones check: "
            f"num_true={num_true_dones}, last_done={dones[-1]}, "
            f"first_5={dones[:5].tolist()}, last_5={dones[-5:].tolist()}"
        )

        # Build parquet data
        data = {
            "image": [],
            "wrist_image": [],
            "state": [],
            "actions": [],
            "timestamp": [],
            "frame_index": [],
            "episode_index": [],
            "index": [],
            "task_index": [],
            "done": [],  # Per-step done flag
            "is_success": [],
        }

        for t in range(T):
            # Image struct
            data["image"].append(self._create_image_struct(images[t], t))

            # Wrist image struct
            if wrist_images is not None:
                data["wrist_image"].append(
                    self._create_image_struct(wrist_images[t], t)
                )
            else:
                # Create an empty wrist image
                data["wrist_image"].append({"bytes": b"", "path": ""})

            # State and actions as lists (for fixed_size_list)
            data["state"].append(states[t].astype(np.float32).tolist())
            data["actions"].append(actions[t].astype(np.float32).tolist())

            # Metadata
            data["timestamp"].append(float(t) / self.fps)
            data["frame_index"].append(t)
            data["episode_index"].append(episode_index)
            data["index"].append(self._global_frame_index + t)
            data["task_index"].append(task_index)
            data["done"].append(bool(dones[t]))  # Per-step done flag
            data["is_success"].append(is_success)

        # Update stats accumulators
        self._update_stats_accumulators(images, wrist_images, states, actions, data)

        # Update global frame index
        self._global_frame_index += T

        # Create PyArrow schema
        schema = self._create_parquet_schema()

        # Create PyArrow table
        table = self._create_parquet_table(data, schema)

        # Write parquet file
        parquet_path = os.path.join(chunk_dir, f"episode_{episode_index:06d}.parquet")
        pq.write_table(table, parquet_path)
        _writer_logger.info(
            f"[add_episode] ep={episode_index}: Written to {parquet_path}"
        )

        # Verify written data
        _writer_logger.info(
            f"[add_episode] ep={episode_index}: Verifying written parquet..."
        )
        verify_table = pq.read_table(parquet_path)
        verify_df = verify_table.to_pandas()
        _writer_logger.info(
            f"[add_episode] ep={episode_index}: Verified columns: {list(verify_df.columns)}, "
            f"rows={len(verify_df)}"
        )
        # Check done column
        done_col = verify_df["done"].values
        _writer_logger.info(
            f"[add_episode] ep={episode_index}: done column: "
            f"dtype={done_col.dtype}, num_true={np.sum(done_col)}, last={done_col[-1]}"
        )

        # Record episode metadata
        self._episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": T,
                "is_success": is_success,
            }
        )

        return episode_index

    def _create_parquet_schema(self) -> pa.Schema:
        """Create a PyArrow schema with HuggingFace features metadata."""
        # Image struct type
        image_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

        schema = pa.schema(
            [
                ("image", image_struct),
                ("wrist_image", image_struct),
                ("state", pa.list_(pa.float32(), self.state_dim)),
                ("actions", pa.list_(pa.float32(), self.action_dim)),
                ("timestamp", pa.float32()),
                ("frame_index", pa.int64()),
                ("episode_index", pa.int64()),
                ("index", pa.int64()),
                ("task_index", pa.int64()),
                ("done", pa.bool_()),  # Per-step done flag
                ("is_success", pa.bool_()),
            ]
        )

        # Add HuggingFace features metadata so HuggingFace datasets can correctly parse image fields
        hf_features = {
            "info": {
                "features": {
                    "image": {"_type": "Image"},
                    "wrist_image": {"_type": "Image"},
                    "state": {
                        "feature": {"dtype": "float32", "_type": "Value"},
                        "length": self.state_dim,
                        "_type": "Sequence",
                    },
                    "actions": {
                        "feature": {"dtype": "float32", "_type": "Value"},
                        "length": self.action_dim,
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
        schema = schema.with_metadata({"huggingface": json.dumps(hf_features)})
        return schema

    def _create_parquet_table(
        self, data: dict[str, list], schema: pa.Schema
    ) -> pa.Table:
        """Create a PyArrow table."""
        arrays = {
            "image": pa.array(data["image"], type=schema.field("image").type),
            "wrist_image": pa.array(
                data["wrist_image"], type=schema.field("wrist_image").type
            ),
            "state": pa.array(data["state"], type=schema.field("state").type),
            "actions": pa.array(data["actions"], type=schema.field("actions").type),
            "timestamp": pa.array(data["timestamp"], type=pa.float32()),
            "frame_index": pa.array(data["frame_index"], type=pa.int64()),
            "episode_index": pa.array(data["episode_index"], type=pa.int64()),
            "index": pa.array(data["index"], type=pa.int64()),
            "task_index": pa.array(data["task_index"], type=pa.int64()),
            "done": pa.array(data["done"], type=pa.bool_()),
            "is_success": pa.array(data["is_success"], type=pa.bool_()),
        }
        return pa.table(arrays, schema=schema)

    def _update_stats_accumulators(
        self,
        images: np.ndarray,
        wrist_images: np.ndarray | None,
        states: np.ndarray,
        actions: np.ndarray,
        data: dict[str, list],
    ) -> None:
        """Update the statistics accumulators."""
        if self.use_incremental_stats:
            # Use incremental statistics (Welford's algorithm)
            self._update_stats_incremental(images, wrist_images, states, actions, data)
        else:
            # Original approach: accumulate raw data
            # Images: normalize to [0, 1] before computing statistics
            img_normalized = images.astype(np.float32) / 255.0
            self._stats_accumulators["image"]["values"].append(img_normalized)

            if wrist_images is not None:
                wrist_normalized = wrist_images.astype(np.float32) / 255.0
                self._stats_accumulators["wrist_image"]["values"].append(
                    wrist_normalized
                )

            self._stats_accumulators["state"]["values"].append(states)
            self._stats_accumulators["actions"]["values"].append(actions)
            self._stats_accumulators["timestamp"]["values"].extend(data["timestamp"])
            self._stats_accumulators["frame_index"]["values"].extend(
                data["frame_index"]
            )
            self._stats_accumulators["episode_index"]["values"].extend(
                data["episode_index"]
            )
            self._stats_accumulators["index"]["values"].extend(data["index"])
            self._stats_accumulators["task_index"]["values"].extend(data["task_index"])

    def _update_stats_incremental(
        self,
        images: np.ndarray,
        wrist_images: np.ndarray | None,
        states: np.ndarray,
        actions: np.ndarray,
        data: dict[str, list],
    ) -> None:
        """Incrementally update statistics using Welford's algorithm without storing raw data."""
        # Image sampling (use only a subset for statistics to reduce computation)
        T = len(images)
        sample_size = max(1, int(T * self.stats_sample_ratio))
        sample_indices = np.random.choice(T, sample_size, replace=False)
        sampled_images = images[sample_indices].astype(np.float32) / 255.0
        self._update_running_stats_array("image", sampled_images)

        if wrist_images is not None:
            sampled_wrist = wrist_images[sample_indices].astype(np.float32) / 255.0
            self._update_running_stats_array("wrist_image", sampled_wrist)

        # Use all data for state/action
        self._update_running_stats_array("state", states)
        self._update_running_stats_array("actions", actions)

        # Scalar statistics
        for key, values in [
            ("timestamp", data["timestamp"]),
            ("frame_index", data["frame_index"]),
            ("episode_index", data["episode_index"]),
            ("index", data["index"]),
            ("task_index", data["task_index"]),
        ]:
            self._update_running_stats_scalar(key, values)

    def _update_running_stats_array(self, key: str, data: np.ndarray) -> None:
        """Update array mean and variance using Welford's online algorithm (for image, state, actions)."""
        acc = self._stats_accumulators[key]
        n = data.shape[0]

        if n == 0:
            return

        # Flatten to 2D: [N, ...] -> [N, -1] for easier computation
        # For images: [N, H, W, C] -> we compute per-channel stats
        # For state/actions: [N, D] -> compute per-dimension stats
        if key in ["image", "wrist_image"]:
            # For images, compute per-channel mean/std
            # Reshape to [N, H*W, C] then mean over H*W
            data_reshaped = data.reshape(n, -1, data.shape[-1])  # [N, H*W, C]
            data_flat = data_reshaped.mean(axis=1)  # [N, C] - mean over spatial dims
        else:
            data_flat = data  # [N, D]

        if acc["count"] == 0:
            acc["sum"] = np.sum(data_flat, axis=0)
            acc["sum_sq"] = np.sum(data_flat**2, axis=0)
            acc["min"] = np.min(data_flat, axis=0)
            acc["max"] = np.max(data_flat, axis=0)
        else:
            acc["sum"] = acc["sum"] + np.sum(data_flat, axis=0)
            acc["sum_sq"] = acc["sum_sq"] + np.sum(data_flat**2, axis=0)
            acc["min"] = np.minimum(acc["min"], np.min(data_flat, axis=0))
            acc["max"] = np.maximum(acc["max"], np.max(data_flat, axis=0))
        acc["count"] += n

    def _update_running_stats_scalar(self, key: str, values: list) -> None:
        """Update scalar statistics using Welford's online algorithm."""
        acc = self._stats_accumulators[key]
        arr = np.array(values, dtype=np.float32)
        n = len(arr)

        if n == 0:
            return

        acc["sum"] += float(np.sum(arr))
        acc["sum_sq"] += float(np.sum(arr**2))
        acc["count"] += n
        acc["min"] = min(acc["min"], float(np.min(arr)))
        acc["max"] = max(acc["max"], float(np.max(arr)))

    def _compute_stats(self) -> dict[str, dict[str, Any]]:
        """Compute statistics for all fields."""
        if self.use_incremental_stats:
            return self._compute_stats_from_accumulators()
        else:
            return self._compute_stats_from_values()

    def _compute_stats_from_values(self) -> dict[str, dict[str, Any]]:
        """Compute statistics from accumulated raw data (original approach)."""
        stats = {}

        # Image stats (per channel)
        if self._stats_accumulators["image"]["values"]:
            all_images = np.concatenate(
                self._stats_accumulators["image"]["values"], axis=0
            )
            stats["image"] = self._compute_image_stats(all_images)

        # Wrist image stats
        if self._stats_accumulators["wrist_image"]["values"]:
            all_wrist = np.concatenate(
                self._stats_accumulators["wrist_image"]["values"], axis=0
            )
            stats["wrist_image"] = self._compute_image_stats(all_wrist)

        # State stats
        if self._stats_accumulators["state"]["values"]:
            all_states = np.concatenate(
                self._stats_accumulators["state"]["values"], axis=0
            )
            stats["state"] = self._compute_array_stats(all_states)

        # Actions stats
        if self._stats_accumulators["actions"]["values"]:
            all_actions = np.concatenate(
                self._stats_accumulators["actions"]["values"], axis=0
            )
            stats["actions"] = self._compute_array_stats(all_actions)

        # Scalar stats
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            values = self._stats_accumulators[key]["values"]
            if values:
                arr = np.array(values, dtype=np.float32)
                stats[key] = {
                    "mean": [float(np.mean(arr))],
                    "std": [float(np.std(arr))],
                    "max": [float(np.max(arr))],
                    "min": [float(np.min(arr))],
                }

        return stats

    def _compute_stats_from_accumulators(self) -> dict[str, dict[str, Any]]:
        """Compute statistics from Welford accumulators (incremental mode)."""
        stats = {}

        # Array-type statistics (image, wrist_image, state, actions)
        for key in ["image", "wrist_image", "state", "actions"]:
            acc = self._stats_accumulators[key]
            if acc["count"] == 0:
                continue

            n = acc["count"]
            mean = acc["sum"] / n
            variance = (acc["sum_sq"] / n) - (mean**2)
            std = np.sqrt(np.maximum(variance, 0))

            if key in ["image", "wrist_image"]:
                # Image stats need special format: per-channel with nested structure
                stats[key] = {
                    "mean": [[[float(m)]] for m in mean],  # [[[ ]]] per channel
                    "std": [[[float(s)]] for s in std],
                    "max": [[[float(m)]] for m in acc["max"]],
                    "min": [[[float(m)]] for m in acc["min"]],
                }
            else:
                # State/actions: per-dimension
                stats[key] = {
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "max": acc["max"].tolist(),
                    "min": acc["min"].tolist(),
                }

        # Scalar statistics
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            acc = self._stats_accumulators[key]
            if acc["count"] == 0:
                continue

            n = acc["count"]
            mean = acc["sum"] / n
            variance = (acc["sum_sq"] / n) - (mean**2)
            std = np.sqrt(max(variance, 0))

            stats[key] = {
                "mean": [float(mean)],
                "std": [float(std)],
                "max": [float(acc["max"])],
                "min": [float(acc["min"])],
            }

        return stats

    def _compute_image_stats(self, images: np.ndarray) -> dict[str, list]:
        """Compute image statistics (per channel)."""
        # images: [N, H, W, C]
        stats = {"mean": [], "std": [], "max": [], "min": []}
        for c in range(images.shape[-1]):
            channel = images[..., c]
            stats["mean"].append([[[float(np.mean(channel))]]])
            stats["std"].append([[[float(np.std(channel))]]])
            stats["max"].append([[[float(np.max(channel))]]])
            stats["min"].append([[[float(np.min(channel))]]])
        return stats

    def _compute_array_stats(self, arr: np.ndarray) -> dict[str, list]:
        """Compute array statistics (per dimension)."""
        # arr: [N, D]
        return {
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(),
        }

    def finalize(self) -> None:
        """
        Finalize dataset writing and generate all meta files.

        Generated files:
            - meta/info.json: Dataset metadata
            - meta/episodes.jsonl: Episode list
            - meta/tasks.jsonl: Task list
            - meta/stats.json: Statistics
        """
        _writer_logger.info(f"[finalize] Starting finalization for {self.root_dir}")
        _writer_logger.info(
            f"[finalize] Total episodes: {len(self._episodes)}, Total frames: {self._global_frame_index}, Total tasks: {len(self._tasks)}"
        )

        meta_dir = os.path.join(self.root_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        _writer_logger.info(f"[finalize] Meta dir: {meta_dir}")

        # Write info.json
        self._write_info_json(meta_dir)
        _writer_logger.info("[finalize] Written info.json")

        # Write episodes.jsonl
        self._write_episodes_jsonl(meta_dir)
        _writer_logger.info(
            f"[finalize] Written episodes.jsonl with {len(self._episodes)} episodes"
        )

        # Write tasks.jsonl
        self._write_tasks_jsonl(meta_dir)
        _writer_logger.info(
            f"[finalize] Written tasks.jsonl with {len(self._tasks)} tasks"
        )

        # Write stats.json
        self._write_stats_json(meta_dir)
        _writer_logger.info("[finalize] Written stats.json")
        _writer_logger.info(f"[finalize] Finalization complete for {self.root_dir}")

    def _write_info_json(self, meta_dir: str) -> None:
        """Write info.json."""
        total_episodes = len(self._episodes)
        total_chunks = (total_episodes + self.chunks_size - 1) // self.chunks_size

        info = {
            "codebase_version": self.codebase_version,
            "robot_type": self.robot_type,
            "total_episodes": total_episodes,
            "total_frames": self._global_frame_index,
            "total_tasks": len(self._tasks),
            "total_videos": 0,
            "total_chunks": max(1, total_chunks),
            "chunks_size": self.chunks_size,
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "image": {
                    "dtype": "image",
                    "shape": list(self.image_shape),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": list(self.image_shape),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": [self.state_dim],
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": [self.action_dim],
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

    def _write_episodes_jsonl(self, meta_dir: str) -> None:
        """Write episodes.jsonl."""
        with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
            for ep in self._episodes:
                f.write(json.dumps(ep) + "\n")

    def _write_tasks_jsonl(self, meta_dir: str) -> None:
        """Write tasks.jsonl."""
        # Sort by task_index
        sorted_tasks = sorted(self._tasks.items(), key=lambda x: x[1])
        with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
            for task, task_index in sorted_tasks:
                f.write(json.dumps({"task_index": task_index, "task": task}) + "\n")

    def _write_stats_json(self, meta_dir: str) -> None:
        """Write stats.json."""
        stats = self._compute_stats()
        with open(os.path.join(meta_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

    @property
    def num_episodes(self) -> int:
        """Return the number of added episodes."""
        return len(self._episodes)

    @property
    def num_frames(self) -> int:
        """Return the total number of frames."""
        return self._global_frame_index

    @property
    def num_tasks(self) -> int:
        """Return the number of unique tasks."""
        return len(self._tasks)


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

    _writer_logger.info(f"[merge] Starting merge from {base_dir} to {output_dir}")
    _writer_logger.info(f"[merge] Pattern: {pattern}")

    # Find all matching subdirectories
    search_pattern = os.path.join(base_dir, pattern)
    sub_dirs = sorted(glob.glob(search_pattern))

    if not sub_dirs:
        _writer_logger.warning(
            f"[merge] No directories found matching {search_pattern}"
        )
        return 0

    _writer_logger.info(
        f"[merge] Found {len(sub_dirs)} directories to merge: {sub_dirs}"
    )

    # Collect data from all episodes
    all_episodes_data = []  # [(parquet_path, episode_meta), ...]
    all_tasks = {}  # task_str -> task_index (globally re-indexed)

    for sub_dir in sub_dirs:
        # Check directory structure
        meta_dir = os.path.join(sub_dir, "meta")
        data_dir = os.path.join(sub_dir, "data")

        if not os.path.exists(meta_dir) or not os.path.exists(data_dir):
            _writer_logger.warning(
                f"[merge] Skipping {sub_dir}: missing meta or data directory"
            )
            continue

        # Read episodes.jsonl
        episodes_file = os.path.join(meta_dir, "episodes.jsonl")
        if not os.path.exists(episodes_file):
            _writer_logger.warning(
                f"[merge] Skipping {sub_dir}: missing episodes.jsonl"
            )
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
                    _writer_logger.warning(f"[merge] Parquet not found: {parquet_path}")

    _writer_logger.info(
        f"[merge] Collected {len(all_episodes_data)} episodes, {len(all_tasks)} unique tasks"
    )

    if not all_episodes_data:
        _writer_logger.warning("[merge] No episodes to merge")
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

    _writer_logger.info(
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
            _writer_logger.info(
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
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
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

    _writer_logger.info(
        f"[merge] Merge complete: {total_episodes} episodes, {global_frame_index} frames -> {output_dir}"
    )

    return total_episodes
