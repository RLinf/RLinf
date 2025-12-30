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


import os
import pickle as pkl
from typing import Optional

import numpy as np
import torch

from rlinf.scheduler import Channel
from rlinf.utils.nested_dict_process import cat_list_of_dict_tensor


def process_nested_dict_for_replay_buffer(nested_dict, rm_extra_done=True):
    ret_dict = {}
    num_data = None
    for key, value in nested_dict.items():
        if key in ["dones", "truncations", "terminations"] and rm_extra_done:
            value = value[1:]
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:]).cpu()
            if num_data is not None:
                assert num_data == ret_dict[key].shape[0], (
                    f"{key=}, {num_data=}, {ret_dict[key].shape[0]=}"
                )
            num_data = ret_dict[key].shape[0]
        elif isinstance(value, dict):
            ret_dict[key], num_data = process_nested_dict_for_replay_buffer(value)
    if len(ret_dict) > 0:
        assert num_data is not None
    return ret_dict, num_data


def get_zero_nested_dict(flattened_batch, capacity, with_batch_dim=True):
    buffer = {}
    for key, value in flattened_batch.items():
        if isinstance(value, torch.Tensor):
            if with_batch_dim:
                tgt_shape = (capacity, *value.shape[1:])
            else:
                tgt_shape = (capacity, *value.shape)
            buffer[key] = torch.zeros(tgt_shape, dtype=value.dtype, device="cpu")
        elif isinstance(value, dict):
            buffer[key] = get_zero_nested_dict(value, capacity, with_batch_dim)
        else:
            raise NotImplementedError
    return buffer


def truncate_nested_dict_by_capacity(nested_dict, capacity):
    ret_dict = {}
    for key, val in nested_dict.items():
        if isinstance(val, torch.Tensor):
            ret_dict[key] = val[-capacity:]
        elif isinstance(val, dict):
            ret_dict[key] = truncate_nested_dict_by_capacity(nested_dict, capacity)
        else:
            raise NotImplementedError
    return ret_dict


def sample_nested_batch(nested_dict, sample_ids):
    sample_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            sample_dict[key] = value[sample_ids]
        elif isinstance(value, dict):
            sample_dict[key] = sample_nested_batch(value, sample_ids)
        else:
            raise NotImplementedError
    return sample_dict


def insert_nested_batch(nested_dict, tgt_dict, insert_ids):
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            tgt_dict[key][insert_ids] = value
        elif isinstance(value, dict):
            tgt_dict[key] = insert_nested_batch(value, tgt_dict[key], insert_ids)
        else:
            raise NotImplementedError
    return tgt_dict


def shuffle_and_split_dict_to_chunk(data: dict, split_size, indice_ids):
    splited_list = [{} for _ in range(split_size)]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            split_vs = torch.chunk(value[indice_ids], split_size)
        elif isinstance(value, dict):
            split_vs = shuffle_and_split_dict_to_chunk(value, split_size, indice_ids)
        else:
            raise ValueError(f"{key=}, {type(value)} is not supported.")
        for split_id in range(split_size):
            splited_list[split_id][key] = split_vs[split_id]
    return splited_list


def clone_dict_and_get_size(nested_dict):
    ret_dict = {}
    size = None
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.clone()
            size = value.shape[0]
        elif isinstance(value, dict):
            ret_dict[key], size = clone_dict_and_get_size(value)
        else:
            raise NotImplementedError
    return ret_dict, size


class SACReplayBuffer:
    """
    Replay buffer for SAC algorithm using pre-allocated torch tensors.
    Implements a circular buffer for efficient memory usage.
    """

    def __init__(self, capacity: int, device: str = "cpu", seed: Optional[int] = None):
        """
        Initialize replay buffer.
        Args:
            capacity: Maximum number of transitions to store
            device: Device to output samples on (storage is always on CPU to save GPU memory)
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.device = device
        self.start = False

        # Storage: dictionary of pre-allocated tensors
        # Will be initialized lazily on first insertion
        self.buffer: dict[str, torch.Tensor] = {}

        self.pos = 0  # Next insertion index
        self.size = 0  # Current number of elements

        # Set random seed
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.random_generator = torch.Generator()
            self.random_generator.manual_seed(seed)
        else:
            self.random_generator = None

    @classmethod
    def create_from_demo(cls, demo_path, seed=None):
        if not os.path.exists(demo_path):
            raise FileNotFoundError(f"File {demo_path} not found")

        if demo_path.endswith(".pkl"):
            with open(demo_path, "rb") as f:
                data_ls = pkl.load(f)
        elif demo_path.endswith(".pt"):
            data_ls = torch.load(demo_path)

        # TODO: Possibly need to convert from jax to torch.
        instance = cls(capacity=len(data_ls), seed=seed)
        for data in data_ls:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            instance.add(data)
        return instance

    @classmethod
    def create_from_buffer(cls, buffer, seed):
        instance = cls(capacity=None, seed=seed)
        instance.buffer, size = clone_dict_and_get_size(buffer)
        instance.size = size
        instance.capacity = size
        return instance

    def _initialize_storage(
        self, flattened_batch: dict[str, torch.Tensor], with_batch_dim=True
    ):
        self.buffer = get_zero_nested_dict(
            flattened_batch, self.capacity, with_batch_dim
        )

    def add(self, data):
        if not self.buffer:
            self._initialize_storage(data, with_batch_dim=False)

        insert_nested_batch(data, self.buffer, self.pos)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _preprocess_rollout_batch(self, rollout_batch):
        if hasattr(self, "cfg"):
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                raise NotImplementedError

            # filter data by rewards
            if self.cfg.algorithm.get("filter_rewards", False):
                raise NotImplementedError

        flattened_batch, num_to_add = process_nested_dict_for_replay_buffer(
            rollout_batch
        )
        return flattened_batch, num_to_add

    def add_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor], extra_preprocess=True
    ):
        """
        Add a batch of transitions to the buffer.
        Handles flattening [T, B, ...] -> [T*B, ...] and circular insertion.
        """
        # 1. Flatten the batch: [n-chunk-steps, actor-bsz, ...] -> [num_samples, ...]

        if "prev_logprobs" in rollout_batch:
            rollout_batch.pop("prev_logprobs")
        if "prev_values" in rollout_batch:
            rollout_batch.pop("prev_values")

        if extra_preprocess:
            flattened_batch, num_to_add = self._preprocess_rollout_batch(rollout_batch)
        else:
            flattened_batch = rollout_batch
            num_to_add = flattened_batch["rewards"].shape[0]
        assert num_to_add > 0

        # 2. Lazy initialization of storage tensors on first call
        if not self.buffer:
            self._initialize_storage(flattened_batch)

        # 3. Handle case where incoming batch is larger than the entire capacity
        if num_to_add >= self.capacity:
            # Just take the last 'capacity' elements
            print(
                f"Warning: Adding batch size {num_to_add} >= capacity {self.capacity}. Overwriting entire buffer."
            )

            self.buffer = truncate_nested_dict_by_capacity(flattened_batch)
            self.pos = 0
            self.size = self.capacity
            return

        # 4. Circular buffer insertion
        start_idx = self.pos
        end_idx = start_idx + num_to_add

        # Use mod operation (%) to get circulated index.
        # [0, 1, 2, ..., capacity-1, capacity, capacity+1, ...]
        # -> [0, 1, 2, ..., capacity-1, 0, 1, ...]
        indices = torch.arange(start_idx, end_idx) % self.capacity

        # 5. Insert the batch
        insert_nested_batch(flattened_batch, self.buffer, indices)

        # 6. Update position and size
        self.pos = end_idx % self.capacity
        self.size = min(self.size + num_to_add, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        # Random sampling indices
        transition_ids = torch.randint(
            low=0, high=self.size, size=(batch_size,), generator=self.random_generator
        )

        batch = sample_nested_batch(self.buffer, transition_ids)
        return batch

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    async def is_ready_async(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    def clear(self):
        """Clear the buffer (reset pointers, keep memory allocated)."""
        self.pos = 0
        self.size = 0

    def get_stats(self) -> dict[str, float]:
        """Get buffer statistics."""
        stats = {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity if self.capacity > 0 else 0.0,
        }

        # Calculate reward statistics if available and buffer is not empty
        if self.size > 0 and "rewards" in self.buffer:
            # Only calculate stats on currently valid data
            valid_rewards = self.buffer["rewards"][: self.size]
            stats.update(
                {
                    "mean_reward": valid_rewards.mean().item(),
                    "std_reward": valid_rewards.std().item(),
                    "min_reward": valid_rewards.min().item(),
                    "max_reward": valid_rewards.max().item(),
                }
            )

        return stats

    def split_to_dict(self, num_splits, is_sequential=False):
        assert self.capacity % num_splits == 0

        all_ids = torch.arange(self.size).to(self.device)
        if not is_sequential:
            all_ids = torch.randperm(self.size, generator=self.random_generator).to(
                self.device
            )

        res_ls = shuffle_and_split_dict_to_chunk(
            self.buffer, split_size=num_splits, indice_ids=all_ids
        )
        return res_ls

    async def run(self, cfg, data_channel: Channel, split_num):
        self.start = True
        self.cfg = cfg
        while True:
            recv_list = []
            for _ in range(split_num):
                recv_list.append(await data_channel.get(async_op=True).async_wait())
            rollout_batch = cat_list_of_dict_tensor(recv_list)
            self.add_rollout_batch(rollout_batch, extra_preprocess=False)

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pkl.dump(self.buffer, f)
