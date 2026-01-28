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


import copy
import json
import os
import pickle as pkl
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory


class TrajectoryCache:
    """FIFO cache for storing flattened trajectories."""

    def __init__(self, max_size: int = 5):
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.max_size = max_size

    def get(self, trajectory_id: str) -> Optional[dict]:
        return self.cache.get(trajectory_id)

    def put(self, trajectory_id: str, trajectory: dict):
        if trajectory_id not in self.cache:
            self.cache[trajectory_id] = trajectory
            # Evict oldest if cache is full
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
        else:
            # Update existing without changing position
            self.cache[trajectory_id] = trajectory

    def clear(self):
        self.cache.clear()


class TrajectoryReplayBuffer:
    """
    Simplified trajectory-based replay buffer.
    Directly stores batched trajectories (shape: [T, B, ...]) without splitting.
    Supports chunk-level sampling with caching.
    """

    def __init__(
        self,
        seed: Optional[int] = 1234,
        enable_cache: bool = True,
        cache_size: int = 5,
        storage_dir: str = "",
        storage_format: str = "pt",
        sample_window_size: int = 100,
        save_trajectories: bool = True,
    ):
        """
        Initialize trajectory-based replay buffer.

        Args:
            seed: Random seed for reproducibility
            storage_dir: Directory to store trajectories (None uses temp directory)
            enable_cache: Whether to enable trajectory caching
            cache_size: Maximum number of trajectories to cache in memory
            storage_format: Storage format ("pt", "pkl")
        """
        self.storage_format = storage_format
        self.enable_cache = enable_cache
        self.sample_window_size = sample_window_size
        self.save_trajectories = save_trajectories

        if not self.save_trajectories and not self.enable_cache:
            raise ValueError("save_trajectories=False requires enable_cache=True")
        if not self.save_trajectories:
            print(
                "[TrajectoryReplayBuffer] save_trajectories=False: "
                "checkpoint save/load is not supported."
            )

        # Storage directory
        assert storage_dir != "", "storage_dir is required"
        self.storage_dir = storage_dir
        if self.save_trajectories:
            os.makedirs(self.storage_dir, exist_ok=True)

        # Trajectory index: dict mapping trajectory_id to trajectory metadata
        # Each entry: {
        #   "num_samples": int,  # T * B (total samples in this trajectory)
        #   "trajectory_id": int,  # trajectory ID
        #   "max_episode_length": int,  # max episode length
        #   "shape": tuple,  # (T, B, ...)
        #   "model_weights_id": str,  # model weights ID
        # }
        self._trajectory_index: dict[int, dict] = {}
        self._trajectory_id_list: list[int] = []  # Ordered list of trajectory IDs
        self._trajectory_counter = 0  # Next trajectory ID to use
        self._index_version = 0

        # Flattened trajectory cache for fast sampling
        self._flat_trajectory_cache = (
            TrajectoryCache(cache_size) if enable_cache else None
        )

        # Async save executor
        self._save_executor = ThreadPoolExecutor(max_workers=1)
        self._index_lock = threading.Lock()

        # Cached window metadata for faster sampling
        self._window_cache_size = None
        self._window_cache_version = None
        self._window_cache_ids: list[int] = []
        self._window_cache_cumulative_ends: list[int] = []
        self._window_cache_cumulative_ends_tensor: Optional[torch.Tensor] = None
        self._window_cache_total_samples = 0

        # Buffer state
        self.size = 0  # Current number of trajectories
        self._total_samples = 0  # Total number of samples across all trajectories

        # Random seed
        self.seed = seed
        self.random_generator: Optional[torch.Generator] = None

        self._load_metadata()
        self._init_random_generator(self.seed)

    def _init_random_generator(self, seed):
        """(Re)initialize numpy and torch RNGs from self.seed."""
        np.random.seed(seed)
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(seed)

    def _get_trajectory_path(
        self,
        trajectory_id: int,
        model_weights_id: str,
        base_dir: Optional[str] = None,
    ) -> str:
        """Get file path for a trajectory."""
        ext = ".pt" if self.storage_format == "pt" else ".pkl"
        base_dir = base_dir or self.storage_dir
        return os.path.join(
            base_dir, f"trajectory_{trajectory_id}_{model_weights_id}{ext}"
        )

    def _get_metadata_path(self, base_dir: Optional[str] = None) -> str:
        """Get path to metadata file."""
        base_dir = base_dir or self.storage_dir
        return os.path.join(base_dir, "metadata.json")

    def _get_trajectory_index_path(self, base_dir: Optional[str] = None) -> str:
        """Get path to trajectory index file."""
        base_dir = base_dir or self.storage_dir
        return os.path.join(base_dir, "trajectory_index.json")

    def _save_metadata(self, save_path: Optional[str] = None):
        """Save metadata to disk."""
        with self._index_lock:
            metadata = {
                "storage_dir": self.storage_dir,
                "storage_format": self.storage_format,
                "size": self.size,
                "total_samples": self._total_samples,
                "trajectory_counter": self._trajectory_counter,
                "seed": self.seed,
            }
            with open(self._get_metadata_path(save_path), "w") as f:
                json.dump(metadata, f)

    def _load_metadata(self):
        """Load metadata from disk."""
        metadata_path = self._get_metadata_path()
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.storage_format = metadata.get("storage_format", self.storage_format)
            self.size = metadata.get("size", 0)
            self._total_samples = metadata.get("total_samples", 0)
            self._trajectory_counter = metadata.get("trajectory_counter", 0)
            self.seed = metadata.get("seed", self.seed)

            # Load trajectory index if exists
            index_path = self._get_trajectory_index_path()
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    index_data = json.load(f)
                self._trajectory_index = index_data.get("trajectory_index", {})
                self._trajectory_id_list = index_data.get("trajectory_id_list", [])

    def _save_trajectory_index(self, save_path: Optional[str] = None):
        """Save trajectory index to disk."""
        with self._index_lock:
            index_data = {
                "trajectory_index": copy.deepcopy(self._trajectory_index),
                "trajectory_id_list": list(self._trajectory_id_list),
            }
            with open(self._get_trajectory_index_path(save_path), "w") as f:
                json.dump(index_data, f)

    def _save_trajectory(
        self, trajectory: Trajectory, trajectory_id: int, model_weights_id: str
    ):
        """Save a single episode to disk as a dictionary."""
        trajectory_path = self._get_trajectory_path(trajectory_id, model_weights_id)

        # Convert Trajectory to dictionary for more stable storage
        trajectory_dict = {}
        for field_name in trajectory.__dataclass_fields__.keys():
            value = getattr(trajectory, field_name, None)
            if value is not None:
                trajectory_dict[field_name] = value

        if self.storage_format == "pt":
            torch.save(trajectory_dict, trajectory_path)
        else:
            with open(trajectory_path, "wb") as f:
                pkl.dump(trajectory_dict, f)

    def _load_trajectory(self, trajectory_id: int, model_weights_id: str) -> Trajectory:
        """Load a trajectory from disk and reconstruct Trajectory object."""

        # Get trajectory info from index
        if trajectory_id not in self._trajectory_index:
            raise ValueError(f"Trajectory {trajectory_id} not found in index")

        trajectory_info = self._trajectory_index[trajectory_id]

        trajectory_path = self._get_trajectory_path(
            trajectory_id,
            model_weights_id,
            base_dir=self.storage_dir,
        )

        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(f"Trajectory file not found at {trajectory_path}")

        # Load trajectory dictionary
        if self.storage_format == "pt":
            trajectory_dict = torch.load(trajectory_path, map_location="cpu")
        else:
            with open(trajectory_path, "rb") as f:
                trajectory_dict = pkl.load(f)

        # Reconstruct Trajectory object from dictionary
        trajectory = Trajectory(
            max_episode_length=trajectory_info["max_episode_length"]
        )
        for field_name, value in trajectory_dict.items():
            setattr(trajectory, field_name, value)

        return trajectory

    def add_trajectories(self, trajectories: list[Trajectory]):
        """
        Add trajectories to the buffer.
        Each trajectory is directly stored as-is (shape: [T, B, ...]).

        Args:
            trajectories: List of Trajectory objects, each with shape [T, B, ...]
                     where T*B is the total number of samples in the trajectory.
        """
        if not trajectories:
            return

        save_futures = []
        for trajectory in trajectories:
            model_weights_id = trajectory.model_weights_id
            trajectory_id = self._trajectory_counter

            # Calculate total samples: T * B
            if trajectory.prev_logprobs is not None:
                T, B = trajectory.prev_logprobs.shape[:2]
                num_samples = T * B
                trajectory_shape = trajectory.prev_logprobs.shape
            elif trajectory.rewards is not None:
                T, B = trajectory.rewards.shape[:2]
                num_samples = T * B
                trajectory_shape = trajectory.rewards.shape
            else:
                continue  # Skip empty trajectories

            # Save trajectory to disk if enabled
            if self.save_trajectories:
                # Save asynchronously to reduce I/O stalls
                save_futures.append(
                    self._save_executor.submit(
                        self._save_trajectory,
                        trajectory,
                        trajectory_id,
                        model_weights_id,
                    )
                )

            # Add to index
            with self._index_lock:
                trajectory_info = {
                    "num_samples": num_samples,
                    "trajectory_id": trajectory_id,
                    "max_episode_length": trajectory.max_episode_length,
                    "shape": tuple(trajectory_shape),
                    "model_weights_id": model_weights_id,
                }
                self._trajectory_index[trajectory_id] = trajectory_info
                self._trajectory_id_list.append(trajectory_id)

                # Update counters
                self._trajectory_counter += 1
                self.size += 1
                self._total_samples += num_samples
                self._index_version += 1

            if self._flat_trajectory_cache is not None:
                self._flat_trajectory_cache.put(
                    trajectory_id, self._flatten_trajectory(trajectory)
                )

        # Save metadata/index after all trajectory saves finish
        if self.save_trajectories:

            def _flush_metadata():
                for fut in save_futures:
                    fut.result()
                self._save_metadata()
                self._save_trajectory_index()

            self._save_executor.submit(_flush_metadata)

    def close(self, wait: bool = True):
        """Flush and shutdown async save executor."""
        if self._save_executor is not None:
            self._save_executor.shutdown(wait=wait)

    def sample(
        self,
        num_chunks: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Sample chunks (transitions) from the buffer.

        Args:
            num_chunks: Minimum number of chunks (transitions) to return

        Returns:
            Dictionary with rollout batch format [B, ...]
        """
        assert num_chunks > 0
        return self.sample_chunks(num_chunks)

    def sample_chunks(self, num_chunks: int) -> dict[str, torch.Tensor]:
        """
        Sample chunks (transitions) from the buffer.
        Each chunk is a single transition from any trajectory.

        Args:
            num_chunks: Number of chunks (transitions) to sample

        Returns:
            Dictionary with batch format [B, ...] where B = num_chunks
        """
        if self._total_samples == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        # Sample from the most recent trajectories (windowed)
        window_size = max(0, int(self.sample_window_size))
        with self._index_lock:
            if (
                self._window_cache_size == window_size
                and self._window_cache_version == self._index_version
            ):
                window_ids = self._window_cache_ids
                cumulative_ends = self._window_cache_cumulative_ends
                window_total_samples = self._window_cache_total_samples
            else:
                if window_size > 0:
                    window_ids = list(self._trajectory_id_list[-window_size:])
                else:
                    window_ids = list(self._trajectory_id_list)

                cumulative_ends = []
                running = 0
                for single_id in window_ids:
                    running += self._trajectory_index[single_id]["num_samples"]
                    cumulative_ends.append(running)
                window_total_samples = running

                self._window_cache_size = window_size
                self._window_cache_version = self._index_version
                self._window_cache_ids = window_ids
                self._window_cache_cumulative_ends = cumulative_ends
                self._window_cache_cumulative_ends_tensor = (
                    torch.as_tensor(cumulative_ends, dtype=torch.long)
                    if cumulative_ends
                    else None
                )
                self._window_cache_total_samples = window_total_samples

        if not window_ids:
            return {}

        if window_total_samples == 0:
            return {}

        if num_chunks > window_total_samples:
            num_chunks = window_total_samples

        # Sample chunk indices directly from total samples
        sample_ids = torch.randint(
            low=0,
            high=window_total_samples,
            size=(num_chunks,),
            generator=self.random_generator,
        )

        # Convert global sample indices to per-trajectory local indices
        if not window_ids:
            return {}

        grouped_indices: dict[str, list[tuple[int, int]]] = {}
        cumulative_ends_tensor = self._window_cache_cumulative_ends_tensor
        if cumulative_ends_tensor is None or cumulative_ends_tensor.numel() == 0:
            return {}

        # Vectorized bucketize to map sample_ids -> trajectory indices
        sample_ids_tensor = sample_ids.to(dtype=torch.long)
        bucket_indices = torch.bucketize(
            sample_ids_tensor, cumulative_ends_tensor, right=True
        )
        starts = torch.cat(
            [torch.zeros(1, dtype=torch.long), cumulative_ends_tensor[:-1]]
        )
        local_sample_indices = sample_ids_tensor - starts[bucket_indices]

        for idx_in_batch in range(sample_ids_tensor.numel()):
            idx = int(bucket_indices[idx_in_batch])
            if idx >= len(window_ids):
                continue
            trajectory_id = window_ids[idx]
            local_sample_idx = int(local_sample_indices[idx_in_batch])
            grouped_indices.setdefault(trajectory_id, []).append(
                (idx_in_batch, local_sample_idx)
            )

        # Load each trajectory once and extract multiple chunks (batched indexing)
        batch = None
        for trajectory_id, local_indices in grouped_indices.items():
            flat_trajectory = None
            if self._flat_trajectory_cache is not None:
                flat_trajectory = self._flat_trajectory_cache.get(trajectory_id)
            if flat_trajectory is None:
                model_weights_id = self._trajectory_index[trajectory_id][
                    "model_weights_id"
                ]
                trajectory = self._load_trajectory(trajectory_id, model_weights_id)
                flat_trajectory = self._flatten_trajectory(trajectory)
                if self._flat_trajectory_cache is not None:
                    self._flat_trajectory_cache.put(trajectory_id, flat_trajectory)

            if batch is None:
                batch = self._init_batch_from_flat(flat_trajectory, num_chunks)

            batch_indices = torch.as_tensor(
                [idx for idx, _ in local_indices], dtype=torch.long
            )
            local_indices_tensor = torch.as_tensor(
                [local_idx for _, local_idx in local_indices], dtype=torch.long
            )

            for key, value in flat_trajectory.items():
                if isinstance(value, torch.Tensor):
                    batch[key][batch_indices] = value[local_indices_tensor]
                elif isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, torch.Tensor):
                            batch[key][nested_key][batch_indices] = nested_value[
                                local_indices_tensor
                            ]

        return batch if batch is not None else {}

    def _flatten_trajectory(self, trajectory: Trajectory) -> dict:
        flat: dict[str, object] = {}
        tensor_fields = trajectory.__dataclass_fields__.keys()
        for field in tensor_fields:
            tensor = getattr(trajectory, field)
            if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                flat[field] = tensor.reshape(-1, *tensor.shape[2:])

        if trajectory.curr_obs:
            flat["curr_obs"] = {}
            for key, tensor in trajectory.curr_obs.items():
                if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                    flat["curr_obs"][key] = tensor.reshape(-1, *tensor.shape[2:])

        if trajectory.next_obs:
            flat["next_obs"] = {}
            for key, tensor in trajectory.next_obs.items():
                if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                    flat["next_obs"][key] = tensor.reshape(-1, *tensor.shape[2:])

        if trajectory.forward_inputs:
            flat["forward_inputs"] = {}
            for key, tensor in trajectory.forward_inputs.items():
                if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                    flat["forward_inputs"][key] = tensor.reshape(-1, *tensor.shape[2:])

        return flat

    def _extract_chunk_from_flat_trajectory(
        self, flat_trajectory: dict, idx: int
    ) -> dict:
        chunk: dict = {}
        for key, value in flat_trajectory.items():
            if isinstance(value, torch.Tensor):
                chunk[key] = value[idx]
            elif isinstance(value, dict):
                nested = {}
                for nested_key, tensor in value.items():
                    if isinstance(tensor, torch.Tensor):
                        nested[nested_key] = tensor[idx]
                if nested:
                    chunk[key] = nested
        return chunk

    def _init_batch_from_flat(self, flat_trajectory: dict, batch_size: int) -> dict:
        batch: dict[str, object] = {}
        for key, value in flat_trajectory.items():
            if isinstance(value, torch.Tensor):
                shape = (batch_size, *value.shape[1:])
                batch[key] = torch.empty(shape, dtype=value.dtype, device=value.device)
            elif isinstance(value, dict):
                nested_batch = {}
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, torch.Tensor):
                        shape = (batch_size, *nested_value.shape[1:])
                        nested_batch[nested_key] = torch.empty(
                            shape,
                            dtype=nested_value.dtype,
                            device=nested_value.device,
                        )
                if nested_batch:
                    batch[key] = nested_batch
        return batch

    def _merge_chunks_to_batch(self, chunks: list[dict]) -> dict[str, torch.Tensor]:
        """
        Merge a list of chunks into a batch dictionary.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Batch dictionary with shape [B, ...] where B = len(chunks)
        """
        if not chunks:
            return {}

        batch = {}
        first_chunk = chunks[0]

        for key, value in first_chunk.items():
            if isinstance(value, torch.Tensor):
                tensors = [chunk[key] for chunk in chunks if key in chunk]
                if tensors:
                    batch[key] = torch.stack(tensors, dim=0)  # [B, ...]
            elif isinstance(value, dict):
                # Handle nested dicts (obs, forward_inputs)
                nested_dicts = [chunk[key] for chunk in chunks if key in chunk]
                if nested_dicts:
                    all_keys = set()
                    for d in nested_dicts:
                        all_keys.update(d.keys())

                    nested_batch = {}
                    for nested_key in all_keys:
                        nested_tensors = [
                            d[nested_key]
                            for d in nested_dicts
                            if nested_key in d
                            and isinstance(d[nested_key], torch.Tensor)
                        ]
                        if nested_tensors:
                            nested_batch[nested_key] = torch.stack(
                                nested_tensors, dim=0
                            )  # [B, ...]
                    if nested_batch:
                        batch[key] = nested_batch

        return batch

    def __len__(self) -> int:
        """Return current buffer size (number of trajectories)."""
        return self.size

    @property
    def total_samples(self) -> int:
        """Return total number of samples across all trajectories."""
        return self._total_samples

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    async def is_ready_async(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    def clear(self):
        # Clear index
        self._trajectory_index.clear()
        self._trajectory_id_list.clear()

        # Clear cache
        if self._flat_trajectory_cache is not None:
            self._flat_trajectory_cache.clear()

        # Reset state
        self.size = 0
        self._total_samples = 0
        self._trajectory_counter = 0

    def get_stats(self) -> dict[str, float]:
        """Get buffer statistics."""
        stats = {
            "num_trajectories": self.size,
            "total_samples": self._total_samples,
            "cache_size": len(self._flat_trajectory_cache.cache)
            if self._flat_trajectory_cache
            else 0,
        }
        return stats

    def save_checkpoint(self, save_path: str):
        """
        Save buffer state (metadata and indices) to save_path.
        """
        if not self.save_trajectories:
            from rlinf.utils.logging import get_logger

            get_logger().warning(
                "save_trajectories=False: checkpoint save is not supported."
            )
            return
        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        # Save metadata and trajectory index into the specified directory
        self._save_metadata(save_path)
        self._save_trajectory_index(save_path)

    def load_checkpoint(
        self,
        load_path: str,
        is_distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Load buffer state from saved metadata.

        Args:
            load_path: Path to the directory containing metadata.json and trajectory_index.json
            is_distributed: If True, only load a portion of trajectories based on local_rank and world_size
            local_rank: Rank index (0-based) for partial loading. Only used when is_distributed=True
            world_size: Total number of ranks. Only used when is_distributed=True
        """
        metadata_path = os.path.join(load_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Update instance attributes from metadata
        self.storage_dir = metadata["storage_dir"]
        self.storage_format = metadata.get("storage_format", "pt")
        if "seed" in metadata:
            self.seed = metadata["seed"]
            self._init_random_generator(self.seed)

        # Load trajectory index and uuid list from save_path
        index_path = os.path.join(load_path, "trajectory_index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Trajectory index not found at {index_path}")

        with open(index_path, "r") as f:
            index_data = json.load(f)

        full_trajectory_index = index_data.get("trajectory_index", {})
        full_trajectory_id_list = index_data.get("trajectory_id_list", [])

        # Handle distributed loading
        if is_distributed:
            if local_rank < 0 or local_rank >= world_size:
                raise ValueError(
                    f"local_rank ({local_rank}) must be in range [0, {world_size})"
                )
            if world_size <= 0:
                raise ValueError(f"world_size ({world_size}) must be > 0")

            # Split trajectory_uuid_list into world_size parts
            total_trajectories = len(full_trajectory_id_list)
            trajectories_per_split = total_trajectories // world_size
            remainder = total_trajectories % world_size

            # Calculate start and end indices for this rank
            start_idx = local_rank * trajectories_per_split + min(local_rank, remainder)
            end_idx = (
                start_idx
                + trajectories_per_split
                + (1 if local_rank < remainder else 0)
            )

            # Extract the portion for this rank
            self._trajectory_id_list = full_trajectory_id_list[start_idx:end_idx]

            # Filter trajectory_index to only include trajectories in this rank's portion
            self._trajectory_index = {
                id: full_trajectory_index[id]
                for id in self._trajectory_id_list
                if id in full_trajectory_index
            }

            # Update size, total_samples, and trajectory_counter based on loaded portion
            self.size = len(self._trajectory_id_list)
            self._total_samples = sum(
                trajectory_info.get("num_samples", 0)
                for trajectory_info in self._trajectory_index.values()
            )
            # trajectory_counter should be set to the max trajectory_id in the loaded portion + 1
            if self._trajectory_index:
                max_trajectory_id = max(
                    trajectory_info.get("trajectory_id", 0)
                    for trajectory_info in self._trajectory_index.values()
                )
                self._trajectory_counter = max_trajectory_id + 1
            else:
                self._trajectory_counter = 0
        else:
            # Full load
            self._trajectory_index = full_trajectory_index
            self._trajectory_id_list = full_trajectory_id_list
            self.size = metadata.get("size", 0)
            self._total_samples = metadata.get("total_samples", 0)
            self._trajectory_counter = metadata.get("trajectory_counter", 0)

    def clear_cache(self):
        """Clear trajectory cache."""
        self.close()
        if self._flat_trajectory_cache is not None:
            self._flat_trajectory_cache.clear()
