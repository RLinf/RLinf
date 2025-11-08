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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

def concat_batch(data1, data2):
    batch = dict()
    for key, value in data1.items():
        if isinstance(value, torch.Tensor):
            batch[key] = torch.cat([data1[key], data2[key]], dim=0)
        elif isinstance(value, dict):
            batch[key] = concat_batch(data1[key], data2[key])
    return batch

class SACReplayBuffer:
    """
    Replay buffer for SAC algorithm using pre-allocated torch tensors.
    Implements a circular buffer for efficient memory usage.
    """

    def __init__(
        self,
        capacity: int,
        device: str = "cpu",
        seed: Optional[int] = None
    ):
        """
        Initialize replay buffer.
        Args:
            capacity: Maximum number of transitions to store
            device: Device to output samples on (storage is always on CPU to save GPU memory)
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.device = device
        
        # Storage: Dictionary of pre-allocated tensors
        # Will be initialized lazily on first insertion
        self.buffer: Dict[str, torch.Tensor] = {}
        
        self.pos = 0    # Next insertion index
        self.size = 0   # Current number of elements
        
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
        instance = cls(
            capacity=len(data_ls),
            seed=seed 
        )
        for data in data_ls:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            instance.add(data)
        return instance
    
    @classmethod
    def create_from_buffer(cls, buffer, seed):
        for key in buffer.keys():
            capacity = buffer[key].shape[0]
            break
        instance = cls(capacity=capacity, seed=seed)
        instance.buffer = dict()
        for key, value in buffer.items():
            instance.buffer[key] = value.clone()
        instance.size = instance.capacity
        return instance

    def _initialize_storage(self, flattened_batch: Dict[str, torch.Tensor], with_batch_dim=True):
        for key, value in flattened_batch.items():
            if with_batch_dim:
                tgt_shape = (self.capacity, *value.shape[1:])
            else:
                tgt_shape = (self.capacity, *value.shape)
            # Allocate fixed-size tensors on CPU
            self.buffer[key] = torch.zeros(
                tgt_shape, 
                dtype=value.dtype, 
                device='cpu'
            )

    def add(self, data):
        if not self.buffer:
            self._initialize_storage(data, with_batch_dim=False)
        
        for key, value in data.items():
            if key not in self.buffer:
                raise ValueError(f"Warning: Key '{key}' from rollout not in buffer storage. Skipping.")
            self.buffer[key][self.pos] = value
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_rollout_batch(self, rollout_batch: Dict[str, torch.Tensor]):
        """
        Add a batch of transitions to the buffer.
        Handles flattening [T, B, ...] -> [T*B, ...] and circular insertion.
        """
        # 1. Flatten the batch: [n-chunk-steps, actor-bsz, ...] -> [num_samples, ...]
        flattened_batch = {}
        num_to_add = None
        
        for key, value in rollout_batch.items():
            if key in ["prev_values", "dones"]:
                value = value[:-1]
            # Ensure value is on CPU for storage
            flat_val = value.reshape(-1, *value.shape[2:]).cpu()
            flattened_batch[key] = flat_val
            
            if num_to_add is None:
                num_to_add = flat_val.shape[0]
            else:
                assert num_to_add == flat_val.shape[0], \
                    f"Inconsistent batch sizes for key {key}, {num_to_add=}, {flat_val.shape[0]=}"

        assert num_to_add > 0

        # 2. Lazy initialization of storage tensors on first call
        if not self.buffer:
            self._initialize_storage(flattened_batch)

        # 3. Handle case where incoming batch is larger than the entire capacity
        if num_to_add >= self.capacity:
             # Just take the last 'capacity' elements
             print(f"Warning: Adding batch size {num_to_add} >= capacity {self.capacity}. Overwriting entire buffer.")
             for key, value in flattened_batch.items():
                 self.buffer[key][:] = value[-self.capacity:]
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
        for key, value in flattened_batch.items():
            if key not in self.buffer:
                raise ValueError(f"Warning: Key '{key}' from rollout not in buffer storage. Skipping.")
            self.buffer[key][indices] = value

        # 5. Update position and size
        self.pos = end_idx % self.capacity
        self.size = min(self.size + num_to_add, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        """
        if self.size == 0:
             raise RuntimeError("Cannot sample from an empty buffer.")
             
        # Random sampling indices
        transition_ids = torch.randint(
            low=0, high=self.size, size=(batch_size,),
            generator=self.random_generator
        )
        
        batch = {}
        for key, tensor in self.buffer.items():
            # Index into the storage tensor and move to target device (e.g., GPU)
            batch[key] = tensor[transition_ids].to(self.device)
            
        return batch

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    def clear(self):
        """Clear the buffer (reset pointers, keep memory allocated)."""
        self.pos = 0
        self.size = 0
        # Option: zero out buffer if needed, but usually just resetting size is enough
        # for key in self.buffer:
        #     self.buffer[key].zero_()

    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        stats = {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity if self.capacity > 0 else 0.0
        }

        # Calculate reward statistics if available and buffer is not empty
        if self.size > 0 and "rewards" in self.buffer:
            # Only calculate stats on currently valid data
            valid_rewards = self.buffer["rewards"][:self.size]
            stats.update({
                "mean_reward": valid_rewards.mean().item(),
                "std_reward": valid_rewards.std().item(),
                "min_reward": valid_rewards.min().item(),
                "max_reward": valid_rewards.max().item()
            })

        return stats
    
    def split_to_dict(self, num_splits, is_sequential=False):
        assert self.capacity % num_splits == 0
        each_split_size = self.capacity // num_splits

        all_ids = torch.arange(self.size).to(self.device)
        if not is_sequential:
            all_ids = torch.randperm(self.size, generator=self.random_generator).to(self.device)

        res_ls = []
        for split_id in range(num_splits):
            buffer = dict()
            select_idx = all_ids[split_id*each_split_size:(split_id+1)*each_split_size]
            for key in self.buffer:
                buffer[key] = self.buffer[key][select_idx].clone()
            res_ls.append(buffer)
        return res_ls        



class PrioritizedSACReplayBuffer(SACReplayBuffer):
    """
    Prioritized Experience Replay buffer for SAC.
    Samples transitions based on TD-error priorities.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        observation_keys: List[str] = None,
        device: str = "cpu",
        seed: Optional[int] = None
    ):
        """
        Initialize prioritized replay buffer.        
        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sampling step
            observation_keys: Keys for observation data
            device: Device to store tensors on
            seed: Random seed
        """
        super().__init__(capacity, observation_keys, device, seed)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

        # Priority storage (using numpy for efficiency)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.buffer_list = []  # Use list instead of deque for indexed access

    def add(self, transition: Dict[str, torch.Tensor]):
        """Add transition with maximum priority."""
        # Move to CPU
        cpu_transition = {}
        for key, value in transition.items():
            if isinstance(value, torch.Tensor):
                cpu_transition[key] = value.detach().cpu()
            elif isinstance(value, dict):
                cpu_transition[key] = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                cpu_transition[key] = value

        if len(self.buffer_list) < self.capacity:
            self.buffer_list.append(cpu_transition)
        else:
            self.buffer_list[self.position] = cpu_transition

        # Assign maximum priority to new transition
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.        
        Returns:
            Tuple of (batch, indices, importance_weights)
        """
        if len(self.buffer_list) < batch_size:
            raise ValueError(f"Buffer size {len(self.buffer_list)} < batch_size {batch_size}")

        # Calculate sampling probabilities
        buffer_size = len(self.buffer_list)
        priorities = self.priorities[:buffer_size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(buffer_size, batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (buffer_size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize by maximum weight

        # Get transitions
        transitions = [self.buffer_list[idx] for idx in indices]
        batch = self._collate_transitions(transitions)
        batch = self._move_to_device(batch, self.device)

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer_list)
