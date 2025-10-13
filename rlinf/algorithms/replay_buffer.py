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

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class SACReplayBuffer:
    """
    Replay buffer for SAC algorithm with support for embodied RL data.
    Stores transitions and provides random sampling for off-policy learning.
    """
    
    def __init__(
        self,
        capacity: int,
        observation_keys: List[str] = None,
        device: str = "cpu",
        seed: Optional[int] = None
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            observation_keys: Keys for observation data (e.g., ['input_ids', 'pixel_values'])
            device: Device to store tensors on
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.device = device
        self.observation_keys = observation_keys or [
            'input_ids', 'pixel_values', 'attention_mask'
        ]
        
        # Initialize storage
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def add(self, transition: Dict[str, torch.Tensor]):
        """
        Add a transition to the buffer.
        
        Args:
            transition: Dictionary containing:
                - observations: Dict with observation keys
                - actions: Action tensor
                - rewards: Reward tensor
                - next_observations: Dict with next observation keys  
                - dones: Done flags
                - action_tokens: Action tokens for embodied RL
                - logprobs: Log probabilities (optional)
        """
        # Move tensors to CPU for storage efficiency
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
        
        self.buffer.append(cpu_transition)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary containing batched transitions
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size {len(self.buffer)} < batch_size {batch_size}")
        
        # Random sampling
        transitions = random.sample(self.buffer, batch_size)
        
        # Collate transitions into batches
        batch = self._collate_transitions(transitions)
        
        # Move to target device
        batch = self._move_to_device(batch, self.device)
        
        return batch
    
    def _collate_transitions(self, transitions: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate list of transitions into batched tensors."""
        batch = {}
        
        # Handle observations
        if 'observations' in transitions[0]:
            batch['observations'] = {}
            for obs_key in self.observation_keys:
                if obs_key in transitions[0]['observations']:
                    obs_list = [t['observations'][obs_key] for t in transitions]
                    batch['observations'][obs_key] = torch.stack(obs_list, dim=0)
        
        # Handle next observations
        if 'next_observations' in transitions[0]:
            batch['next_observations'] = {}
            for obs_key in self.observation_keys:
                if obs_key in transitions[0]['next_observations']:
                    next_obs_list = [t['next_observations'][obs_key] for t in transitions]
                    batch['next_observations'][obs_key] = torch.stack(next_obs_list, dim=0)
        
        # Handle other keys
        for key in ['actions', 'rewards', 'dones', 'action_tokens', 'logprobs']:
            if key in transitions[0]:
                values = [t[key] for t in transitions]
                if isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values, dim=0)
                else:
                    batch[key] = torch.tensor(values)
        
        return batch
    
    def _move_to_device(self, batch: Dict, device: str) -> Dict:
        """Move batch tensors to target device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device)
            elif isinstance(value, dict):
                device_batch[key] = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                device_batch[key] = value
        return device_batch
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= min_size
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.position = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {"size": 0, "capacity": self.capacity, "utilization": 0.0}
        
        # Calculate basic stats
        stats = {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity
        }
        
        # Calculate reward statistics if available
        if self.buffer and 'rewards' in self.buffer[0]:
            rewards = [t['rewards'].item() if isinstance(t['rewards'], torch.Tensor) 
                      else t['rewards'] for t in self.buffer]
            stats.update({
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "min_reward": np.min(rewards),
                "max_reward": np.max(rewards)
            })
        
        return stats


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
