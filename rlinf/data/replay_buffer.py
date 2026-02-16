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
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.utils.logging import get_logger


def clone_dict_of_tensors(obj):
    if torch.is_tensor(obj):
        return obj.cpu().contiguous().clone()
    elif isinstance(obj, int) or isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        return {k: clone_dict_of_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = [clone_dict_of_tensors(v) for v in obj]
        return type(obj)(t)
    else:
        return obj


class TrajectoryCache:
    """FIFO cache for storing flattened trajectories."""

    def __init__(self, max_size: int = 5):
        self.cache: dict[int, int] = {}
        self.max_size = max_size
        self._buffer: Optional[dict] = None
        self._traj_num_samples: Optional[int] = None
        self._traj_key_lengths: dict[int, dict] = {}
        self._last_slot = 0
        self._slot_to_id: dict[int, int] = {}

    def _get_key_lengths(self, trajectory: dict) -> dict:
        lengths: dict = {}
        has_tensor = False
        for key, value in trajectory.items():
            if isinstance(value, torch.Tensor):
                lengths[key] = int(value.shape[0])
                has_tensor = True
            elif isinstance(value, dict):
                nested = self._get_key_lengths(value)
                if nested:
                    lengths[key] = nested
                    has_tensor = True
        if not has_tensor:
            raise ValueError("Trajectory contains no tensor fields.")
        return lengths

    def _get_max_num_samples(self, lengths: dict) -> int:
        max_len = 0
        for value in lengths.values():
            if isinstance(value, dict):
                max_len = max(max_len, self._get_max_num_samples(value))
            else:
                max_len = max(max_len, int(value))
        return max_len

    def _alloc_buffer_like(self, trajectory: dict, total_samples: int) -> dict:
        buffer: dict = {}
        for key, value in trajectory.items():
            if isinstance(value, torch.Tensor):
                shape = (total_samples, *value.shape[1:])
                buffer[key] = torch.empty(shape, dtype=value.dtype, device=value.device)
            elif isinstance(value, dict):
                buffer[key] = self._alloc_buffer_like(value, total_samples)
            else:
                buffer[key] = value
        return buffer

    def _insert_into_buffer(self, trajectory: dict, buffer: dict, start: int) -> None:
        for key, value in trajectory.items():
            if isinstance(value, torch.Tensor):
                end = start + value.shape[0]
                buffer[key][start:end] = value
            elif isinstance(value, dict):
                self._insert_into_buffer(value, buffer[key], start)
            else:
                buffer[key] = value

    def _slice_from_buffer(self, buffer: dict, slc: slice) -> dict:
        sliced: dict = {}
        for key, value in buffer.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value[slc]
            elif isinstance(value, dict):
                sliced[key] = self._slice_from_buffer(value, slc)
            else:
                sliced[key] = value
        return sliced

    def _slice_from_buffer_with_lengths(
        self, buffer: dict, start: int, lengths: Optional[dict]
    ) -> dict:
        sliced: dict = {}
        for key, value in buffer.items():
            if isinstance(value, torch.Tensor):
                if lengths is None or key not in lengths:
                    end = start + self._traj_num_samples
                else:
                    end = start + int(lengths[key])
                sliced[key] = value[start:end]
            elif isinstance(value, dict):
                nested_lengths = None if lengths is None else lengths.get(key, None)
                sliced[key] = self._slice_from_buffer_with_lengths(
                    value, start, nested_lengths
                )
            else:
                sliced[key] = value
        return sliced

    def _copy_buffer_slice(
        self,
        src_buffer: dict,
        dst_buffer: dict,
        src_slc: slice,
        dst_slc: slice,
    ) -> None:
        for key, value in src_buffer.items():
            if isinstance(value, torch.Tensor):
                dst_buffer[key][dst_slc] = value[src_slc]
            elif isinstance(value, dict):
                self._copy_buffer_slice(value, dst_buffer[key], src_slc, dst_slc)
            else:
                dst_buffer[key] = value

    def _ensure_capacity(self, max_num_samples: int, trajectory: dict) -> None:
        if self._traj_num_samples is None:
            self._traj_num_samples = max_num_samples
            total_samples = self.max_size * self._traj_num_samples
            self._buffer = self._alloc_buffer_like(trajectory, total_samples)
            return
        if max_num_samples <= self._traj_num_samples:
            return

        # Grow slot length only when needed.
        old_slot_len = self._traj_num_samples
        new_slot_len = max_num_samples
        new_total_samples = self.max_size * new_slot_len
        new_buffer = self._alloc_buffer_like(trajectory, new_total_samples)

        if self._buffer is not None:
            for slot in self.cache.values():
                src_start = slot * old_slot_len
                src_end = src_start + old_slot_len
                dst_start = slot * new_slot_len
                dst_end = dst_start + old_slot_len
                self._copy_buffer_slice(
                    self._buffer,
                    new_buffer,
                    slice(src_start, src_end),
                    slice(dst_start, dst_end),
                )

        self._buffer = new_buffer
        self._traj_num_samples = new_slot_len

    def get(self, trajectory_id: int) -> Optional[dict]:
        if trajectory_id not in self.cache or self._buffer is None:
            return None
        slot = self.cache[trajectory_id]
        start = slot * self._traj_num_samples
        lengths = self._traj_key_lengths.get(trajectory_id)
        return self._slice_from_buffer_with_lengths(self._buffer, start, lengths)

    def get_buffer(self) -> Optional[dict]:
        return self._buffer

    def get_slot_length(self) -> Optional[int]:
        return self._traj_num_samples

    def put(self, trajectory_id: int, trajectory: dict):
        key_lengths = self._get_key_lengths(trajectory)
        max_num_samples = self._get_max_num_samples(key_lengths)
        self._ensure_capacity(max_num_samples, trajectory)

        if trajectory_id in self.cache:
            slot = self.cache[trajectory_id]
        else:
            slot = self._last_slot
            if slot in self._slot_to_id:
                evict_id = self._slot_to_id[slot]
                if evict_id in self.cache:
                    self.cache.pop(evict_id, None)
                self._traj_key_lengths.pop(evict_id, None)
            self._slot_to_id[slot] = trajectory_id
            self.cache[trajectory_id] = slot
            self._last_slot = (self._last_slot + 1) % self.max_size

        start = slot * self._traj_num_samples
        self._insert_into_buffer(trajectory, self._buffer, start)
        self._traj_key_lengths[trajectory_id] = key_lengths

    def clear(self):
        self.cache.clear()
        self._buffer = None
        self._traj_num_samples = None
        self._traj_key_lengths.clear()
        self._last_slot = 0
        self._slot_to_id.clear()


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
        sample_window_size: int = 100,
        auto_save: bool = False,
        auto_save_path: str = "",
        trajectory_format: str = "pt",
    ):
        """
        Initialize trajectory-based replay buffer.

        Args:
            seed: Random seed for reproducibility
            auto_save_path: Directory to store trajectories when auto_save=True
            enable_cache: Whether to enable trajectory caching
            cache_size: Maximum number of trajectories to cache in memory
            trajectory_format: Storage format ("pt", "pkl")
            sample_window_size: Number of trajectories to sample from for window cache
            auto_save: Whether to automatically save trajectories to disk
        """
        self.trajectory_format = trajectory_format
        self.enable_cache = enable_cache
        self.sample_window_size = sample_window_size
        self.auto_save = auto_save
        self.logger = get_logger()

        if not self.auto_save:
            self.logger.warning(
                f"auto_save is disabled, enabling cache with size {sample_window_size}"
            )
            self.enable_cache = True
            cache_size = sample_window_size

        # Auto-save path (only used when auto_save is enabled)
        if self.auto_save:
            assert auto_save_path != "", (
                "auto_save_path is required when auto_save is enabled"
            )
            self.logger.info(
                f"Created replay buffer with auto_save_path: {auto_save_path}"
            )
        self.auto_save_path = auto_save_path if self.auto_save else None
        if self.auto_save_path is not None:
            os.makedirs(self.auto_save_path, exist_ok=True)

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

        # Trajectory file path: dict mapping trajectory_id to trajectory file path
        # this enables each trajectory to be saved to or loaded from a separate file
        self._trajectory_file_path: dict[int, str] = {}

        self._trajectory_counter = 0  # Next trajectory ID to use
        self._index_version = 0

        # Flattened trajectory cache for fast sampling
        self._flat_trajectory_cache = (
            TrajectoryCache(cache_size) if enable_cache else None
        )

        # Async save executor for add_trajectories
        self._save_executor = ThreadPoolExecutor(max_workers=20)
        # Separate executor for checkpoint saves
        self._checkpoint_executor = ThreadPoolExecutor(max_workers=20)
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
        ext = ".pt" if self.trajectory_format == "pt" else ".pkl"
        base_dir = base_dir or self.auto_save_path
        return os.path.join(
            base_dir, f"trajectory_{trajectory_id}_{model_weights_id}{ext}"
        )

    def _get_metadata_path(self, base_dir: Optional[str] = None) -> str:
        """Get path to metadata file."""
        base_dir = base_dir or self.auto_save_path
        return os.path.join(base_dir, "metadata.json")

    def _get_trajectory_index_path(self, base_dir: Optional[str] = None) -> str:
        """Get path to trajectory index file."""
        base_dir = base_dir or self.auto_save_path
        return os.path.join(base_dir, "trajectory_index.json")

    def _save_metadata(self, save_path: Optional[str] = None):
        """Save metadata to disk."""
        save_path = save_path or self.auto_save_path
        with self._index_lock:
            metadata = {
                "trajectory_format": self.trajectory_format,
                "size": self.size,
                "total_samples": self._total_samples,
                "trajectory_counter": self._trajectory_counter,
                "seed": self.seed,
            }
            with open(self._get_metadata_path(save_path), "w") as f:
                json.dump(metadata, f)

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
        self,
        trajectory: Trajectory,
        trajectory_id: int,
        model_weights_id: str,
        save_dir: Optional[str] = None,
    ):
        """Save a single episode to disk as a dictionary."""
        trajectory_path = self._get_trajectory_path(
            trajectory_id, model_weights_id, base_dir=save_dir
        )

        # Convert Trajectory to dictionary for more stable storage
        trajectory_dict = {}
        for field_name in trajectory.__dataclass_fields__.keys():
            value = getattr(trajectory, field_name, None)
            if value is not None:
                trajectory_dict[field_name] = clone_dict_of_tensors(value)

        if self.trajectory_format == "pt":
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
            base_dir=self._trajectory_file_path[trajectory_id],
        )

        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(f"Trajectory file not found at {trajectory_path}")

        # Load trajectory dictionary
        if self.trajectory_format == "pt":
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
            if self.auto_save:
                # Save asynchronously to reduce I/O stalls
                save_futures.append(
                    self._save_executor.submit(
                        self._save_trajectory,
                        trajectory,
                        trajectory_id,
                        model_weights_id,
                    )
                )
                self._trajectory_file_path[trajectory_id] = self.auto_save_path

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
                    trajectory_id,
                    self._flatten_trajectory(trajectory),
                )

        # Save metadata/index after all trajectory saves finish
        if self.auto_save:

            def _flush_metadata():
                for fut in save_futures:
                    fut.result()
                self._save_metadata()
                self._save_trajectory_index()

            self._save_executor.submit(_flush_metadata)

    def _reshape_flat_for_save(self, value: object, T: int, B: int) -> object:
        if isinstance(value, torch.Tensor):
            if value.dim() >= 1 and value.shape[0] == T * B:
                return value.reshape(T, B, *value.shape[1:])
            return value
        if isinstance(value, dict):
            return {
                key: self._reshape_flat_for_save(val, T, B)
                for key, val in value.items()
            }
        return value

    def close(self, wait: bool = True):
        """Flush and shutdown async save executor."""
        if self._save_executor is not None:
            self._save_executor.shutdown(wait=wait)
        if self._checkpoint_executor is not None:
            self._checkpoint_executor.shutdown(wait=wait)

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

        # Vectorized sampling: use cache buffer directly, load misses and gather once.
        batch = None
        traj_ids_tensor = torch.as_tensor(
            [window_ids[int(idx)] for idx in bucket_indices], dtype=torch.long
        )
        batch_indices_tensor = torch.arange(num_chunks, dtype=torch.long)

        cached_mask = None
        cache = self._flat_trajectory_cache
        if cache is not None:
            cached_ids = list(cache.cache.keys())
            if cached_ids:
                cached_ids_tensor = torch.as_tensor(cached_ids, dtype=torch.long)
                cached_mask = torch.isin(traj_ids_tensor, cached_ids_tensor)
            else:
                cached_mask = torch.zeros_like(traj_ids_tensor, dtype=torch.bool)
        else:
            cached_mask = torch.zeros_like(traj_ids_tensor, dtype=torch.bool)

        # 1) Cache hits: gather from cache buffer.
        if torch.any(cached_mask):
            cache_buffer = cache.get_buffer() if cache is not None else None
            slot_len = cache.get_slot_length() if cache is not None else None
            if cache_buffer is not None and slot_len is not None:
                cached_traj_ids = traj_ids_tensor[cached_mask].tolist()
                cached_slots = torch.as_tensor(
                    [cache.cache[tid] for tid in cached_traj_ids], dtype=torch.long
                )
                cached_local = local_sample_indices[cached_mask]
                buffer_indices = cached_slots * slot_len + cached_local
                batch_indices = batch_indices_tensor[cached_mask]
                if batch is None:
                    batch = self._init_batch_from_buffer(cache_buffer, num_chunks)
                self._fill_batch_from_buffer_indices(
                    batch, cache_buffer, buffer_indices, batch_indices
                )

        # 2) Cache misses: load all, concat, then gather once.
        miss_mask = ~cached_mask
        if torch.any(miss_mask):
            miss_traj_ids = torch.unique(traj_ids_tensor[miss_mask]).tolist()
            miss_flats: list[dict] = []
            traj_offsets: dict[int, int] = {}
            cursor = 0
            for tid in miss_traj_ids:
                model_weights_id = self._trajectory_index[tid]["model_weights_id"]
                trajectory = self._load_trajectory(tid, model_weights_id)
                flat_trajectory = self._flatten_trajectory(trajectory)
                miss_flats.append(flat_trajectory)
                traj_offsets[tid] = cursor
                cursor += self._trajectory_index[tid]["num_samples"]

            concat_flat = self._concat_flat_trajectories(miss_flats)
            if batch is None:
                batch = self._init_batch_from_flat(concat_flat, num_chunks)

            miss_traj_ids_samples = traj_ids_tensor[miss_mask].tolist()
            miss_offsets = torch.as_tensor(
                [traj_offsets[tid] for tid in miss_traj_ids_samples], dtype=torch.long
            )
            miss_local = local_sample_indices[miss_mask]
            miss_buffer_indices = miss_offsets + miss_local
            miss_batch_indices = batch_indices_tensor[miss_mask]
            self._fill_batch_from_buffer_indices(
                batch, concat_flat, miss_buffer_indices, miss_batch_indices
            )

        return batch if batch is not None else {}

    def _flatten_trajectory(self, trajectory: Trajectory) -> dict:
        flat: dict[str, object] = {}
        tensor_fields = trajectory.__dataclass_fields__.keys()
        traj_len = int(trajectory.rewards.shape[0])

        for field in tensor_fields:
            tensor = getattr(trajectory, field)
            if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                if field in ["dones", "terminations", "truncations"]:
                    extra = int(tensor.shape[0] - traj_len)
                    if extra > 0:
                        assert traj_len % extra == 0, (
                            f"Trajectory length {traj_len} is not divisible by extra {extra} for field {field}"
                        )
                        epoch_len = traj_len // extra
                        tensor = tensor.reshape(
                            extra, epoch_len + 1, *tensor.shape[1:]
                        )[:, 1:]
                        tensor = tensor.reshape(traj_len, *tensor.shape[2:])
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

    def _init_batch_from_buffer(self, buffer: dict, batch_size: int) -> dict:
        batch: dict[str, object] = {}
        for key, value in buffer.items():
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

    def _fill_batch_from_buffer_indices(
        self,
        batch: dict,
        buffer: dict,
        buffer_indices: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> None:
        for key, value in buffer.items():
            if isinstance(value, torch.Tensor):
                batch[key][batch_indices] = value.index_select(0, buffer_indices)
            elif isinstance(value, dict):
                self._fill_batch_from_buffer_indices(
                    batch[key], value, buffer_indices, batch_indices
                )

    def _concat_flat_trajectories(self, flats: list[dict]) -> dict:
        if not flats:
            return {}
        out: dict = {}
        keys = flats[0].keys()
        for key in keys:
            if isinstance(flats[0][key], torch.Tensor):
                out[key] = torch.cat([f[key] for f in flats], dim=0)
            elif isinstance(flats[0][key], dict):
                nested_list = [f[key] for f in flats]
                out[key] = self._concat_flat_trajectories(nested_list)
        return out

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
        self._trajectory_file_path.clear()

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
        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        save_futures = []
        if not self.auto_save:
            cache = self._flat_trajectory_cache
            if cache is None:
                raise RuntimeError("auto_save=False requires cache to save checkpoint.")
            cached_ids = list(cache.cache.keys())
            for trajectory_id in cached_ids:
                flat = cache.get(trajectory_id)
                if flat is None:
                    continue
                info = self._trajectory_index.get(trajectory_id, None)
                if info is None:
                    continue
                shape = info.get("shape", None)
                if not shape or len(shape) < 2:
                    continue
                T, B = shape[:2]
                model_weights_id = info.get("model_weights_id", "")
                trajectory = Trajectory(
                    max_episode_length=info.get("max_episode_length", 0),
                    model_weights_id=model_weights_id,
                )
                for field_name in trajectory.__dataclass_fields__.keys():
                    if field_name in flat:
                        setattr(
                            trajectory,
                            field_name,
                            self._reshape_flat_for_save(flat[field_name], T, B),
                        )
                save_futures.append(
                    self._checkpoint_executor.submit(
                        self._save_trajectory,
                        trajectory,
                        trajectory_id,
                        model_weights_id,
                        save_dir=save_path,
                    )
                )
        else:
            for trajectory_id in self._window_cache_ids:
                model_weights_id = self._trajectory_index[trajectory_id][
                    "model_weights_id"
                ]
                trajectory_path = self._get_trajectory_path(
                    trajectory_id, model_weights_id
                )
                if not os.path.isfile(trajectory_path):
                    continue

                # copy trajectory file from trajectory_path to save_path
                target_path = os.path.join(save_path, os.path.basename(trajectory_path))
                save_futures.append(
                    self._checkpoint_executor.submit(
                        shutil.copyfile, trajectory_path, target_path
                    )
                )

        for fut in save_futures:
            fut.result()

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
        self.trajectory_format = metadata.get(
            "trajectory_format",
            self.trajectory_format,
        )
        if "seed" in metadata:
            self.seed = metadata["seed"]
            self._init_random_generator(self.seed)

        # Load trajectory index and uuid list from save_path
        index_path = os.path.join(load_path, "trajectory_index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Trajectory index not found at {index_path}")

        with open(index_path, "r") as f:
            index_data = json.load(f)

        full_trajectory_index = {
            int(k): v for k, v in index_data.get("trajectory_index", {}).items()
        }
        full_trajectory_id_list = [
            int(k) for k in index_data.get("trajectory_id_list", [])
        ]

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

            # Update trajectory file path
            for trajectory_id in self._trajectory_id_list:
                self._trajectory_file_path[trajectory_id] = load_path

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
            for trajectory_id in self._trajectory_id_list:
                self._trajectory_file_path[trajectory_id] = load_path
            self.size = metadata.get("size", 0)
            self._total_samples = metadata.get("total_samples", 0)
            self._trajectory_counter = metadata.get("trajectory_counter", 0)

        if self._flat_trajectory_cache is not None:
            self._flat_trajectory_cache.clear()
            if self._trajectory_id_list:
                max_cache = self._flat_trajectory_cache.max_size
                recent_ids = self._trajectory_id_list[-max_cache:]
                for trajectory_id in recent_ids:
                    model_weights_id = self._trajectory_index[trajectory_id][
                        "model_weights_id"
                    ]
                    trajectory = self._load_trajectory(trajectory_id, model_weights_id)
                    flat_trajectory = self._flatten_trajectory(trajectory)
                    self._flat_trajectory_cache.put(
                        trajectory_id,
                        flat_trajectory,
                    )

    def clear_cache(self):
        """Clear trajectory cache."""
        self.close()
        if self._flat_trajectory_cache is not None:
            self._flat_trajectory_cache.clear()


# python rlinf/data/replay_buffer.py --load-path /path/to/buffer --num-chunks 1024 --cache-size 10 --enable-cache


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple demo buffer load + sample test"
    )
    parser.add_argument(
        "--load-path",
        type=str,
        required=True,
        help="Checkpoint directory containing metadata.json and trajectory_index.json",
    )
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--sample-window-size", type=int, default=10)
    parser.add_argument("--cache-size", type=int, default=5)
    parser.add_argument("--enable-cache", action="store_true")
    args = parser.parse_args()

    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=args.enable_cache,
        cache_size=args.cache_size,
        sample_window_size=args.sample_window_size,
        auto_save=False,
        trajectory_format="pt",
    )

    buffer.load_checkpoint(
        args.load_path,
        is_distributed=False,
    )

    try:
        batch = buffer.sample(num_chunks=args.num_chunks)
    except RuntimeError as exc:
        print(f"[sample] failed: {exc}")
        raise SystemExit(1)

    print("[sample] keys:", list(batch.keys()))
