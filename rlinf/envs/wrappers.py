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

"""Environment wrappers for RLinf embodied RL."""

import logging
import os
import pickle
import time
from typing import Any, Optional

import gymnasium as gym  # type: ignore
import numpy as np

try:
    import torch  # type: ignore
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class DataCollectorWrapper(gym.Wrapper):
    """Data collection wrapper that saves each episode as a separate pkl file.
    
    Can be used in two modes:
    1. As gym.Wrapper: wrap an environment and collect data automatically
    2. Standalone: call collect() manually to add data
    
    Each pkl file contains:
        - frames: list of frame data (obs, action, reward, etc.)
        - grasp: whether grasp was successful
        - success: whether task was successful  
        - done: whether episode ended
        - metadata: episode statistics
    
    Attributes:
        save_dir: Directory to save data.
        max_frames: Maximum frames per episode (default 50).
        episode_count: Number of episodes collected.
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        save_dir: str = "./reward_data",
        max_frames: int = 50,
        target_episodes: int = 1000,
    ):
        """Initialize the data collector.
        
        Args:
            env: The environment to wrap (optional, can be None for standalone mode).
            save_dir: Directory to save collected data.
            max_frames: Maximum frames per episode.
            target_episodes: Target number of episodes to collect.
        """
        if env is not None:
            super().__init__(env)
        else:
            # Standalone mode - don't call super().__init__
            self.env = None
        
        self.save_dir = save_dir
        self.max_frames = max_frames
        self.target_episodes = target_episodes
        
        # Episode buffer
        self._current_episode: list[dict[str, Any]] = []
        self._episode_grasp = False
        self._episode_success = False
        
        # Statistics
        self.episode_count = 0
        self.success_episode_count = 0
        self.failure_episode_count = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(
            f"DataCollectorWrapper initialized. Save dir: {save_dir}, "
            f"Max frames: {max_frames}, Target episodes: {target_episodes}"
        )

    @property
    def is_full(self) -> bool:
        """Check if target episodes reached."""
        return self.episode_count >= self.target_episodes

    def step(self, action):
        """Execute one step and collect frame data.
        
        Args:
            action: The action to execute.
            
        Returns:
            Standard gym return values (obs, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract info flags
        is_success = self._get_bool(info, "success")
        is_grasp = self._get_bool(info, "is_grasped") or self._get_bool(info, "grasp")
        done = terminated or truncated
        
        # Update episode-level flags
        if is_grasp:
            self._episode_grasp = True
        if is_success:
            self._episode_success = True
        
        # Collect frame data
        frame = {
            "obs": self._to_numpy(obs),
            "action": self._to_numpy(action),
            "reward": self._to_scalar(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": done,
            "grasp": is_grasp,
            "success": is_success,
            "info": self._filter_info(info),
        }
        self._current_episode.append(frame)
        
        # Save episode when done
        if done:
            self._save_episode()
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and clear episode buffer."""
        if self.env is not None:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, info = None, {}
        
        # Reset episode buffer
        self._current_episode = []
        self._episode_grasp = False
        self._episode_success = False
        
        return obs, info

    def collect(
        self,
        obs: Any,
        action: Any,
        reward: Any,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> None:
        """Collect data manually (standalone mode).
        
        Args:
            obs: Observation.
            action: Action taken.
            reward: Reward received.
            terminated: Whether episode terminated.
            truncated: Whether episode was truncated.
            info: Info dict from environment.
        """
        # Extract info flags
        is_success = self._get_bool(info, "success")
        is_grasp = self._get_bool(info, "is_grasped") or self._get_bool(info, "grasp")
        done = terminated or truncated
        
        # Update episode-level flags
        if is_grasp:
            self._episode_grasp = True
        if is_success:
            self._episode_success = True
        
        # Collect frame data
        frame = {
            "obs": self._to_numpy(obs),
            "action": self._to_numpy(action),
            "reward": self._to_scalar(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": done,
            "grasp": is_grasp,
            "success": is_success,
            "info": self._filter_info(info),
        }
        self._current_episode.append(frame)
        
        # Save episode when done
        if done:
            self._save_episode()

    def new_episode(self) -> None:
        """Start a new episode (standalone mode)."""
        self._current_episode = []
        self._episode_grasp = False
        self._episode_success = False

    def _get_bool(self, info: dict, key: str) -> bool:
        """Extract boolean from info dict."""
        val = info.get(key, False)
        if torch is not None and isinstance(val, torch.Tensor):
            return bool(val.item() if val.numel() == 1 else val.any().item())
        return bool(val)

    def _to_scalar(self, data: Any) -> float:
        """Convert to scalar."""
        if torch is not None and isinstance(data, torch.Tensor):
            return float(data.item() if data.numel() == 1 else data.mean().item())
        elif isinstance(data, np.ndarray):
            return float(data.item() if data.size == 1 else data.mean())
        return float(data)

    def _to_numpy(self, data: Any) -> Any:
        """Convert data to numpy format for serialization."""
        if torch is not None and isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, dict):
            return {k: self._to_numpy(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._to_numpy(v) for v in data]
        return data

    def _filter_info(self, info: dict) -> dict:
        """Filter info dict to keep only serializable data."""
        filtered = {}
        for k, v in info.items():
            try:
                if torch is not None and isinstance(v, torch.Tensor):
                    filtered[k] = v.cpu().numpy()
                elif isinstance(v, (np.ndarray, int, float, str, bool, type(None))):
                    filtered[k] = v
                elif isinstance(v, dict):
                    filtered[k] = self._filter_info(v)
            except Exception:
                pass
        return filtered

    def _save_episode(self) -> None:
        """Save current episode to pkl file."""
        # Determine label from episode result
        label = "success" if self._episode_success else "failure"
        
        # Build episode data
        episode_data = {
            "frames": self._current_episode,
            "num_frames": len(self._current_episode),
            "grasp": self._episode_grasp,
            "success": self._episode_success,
            "done": True,
            "label": 1 if self._episode_success else 0,
            "metadata": {
                "episode_id": self.episode_count,
                "timestamp": time.time(),
                "max_frames": self.max_frames,
            },
        }
        
        # Create label subdirectory
        label_dir = os.path.join(self.save_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Save to pkl file
        filename = f"episode_{self.episode_count:06d}.pkl"
        filepath = os.path.join(label_dir, filename)
        
        with open(filepath, "wb") as f:
            pickle.dump(episode_data, f)
        
        # Update statistics
        self.episode_count += 1
        if self._episode_success:
            self.success_episode_count += 1
        else:
            self.failure_episode_count += 1
        
        # Periodic logging
        if self.episode_count % 10 == 0:
            logger.info(
                f"Saved episode {self.episode_count}: {label}, "
                f"frames={len(self._current_episode)}, grasp={self._episode_grasp}. "
                f"Total: {self.success_episode_count} success, {self.failure_episode_count} failure"
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Statistics dictionary.
        """
        return {
            "episode_count": self.episode_count,
            "success_episode_count": self.success_episode_count,
            "failure_episode_count": self.failure_episode_count,
            "is_full": self.is_full,
            "save_dir": self.save_dir,
        }
