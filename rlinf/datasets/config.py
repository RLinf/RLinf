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

"""
RL Dataset Configuration for value learning.

This module extends the VLA dataset configuration with RL-specific settings:
1. History length and history keys for past observations
2. Action/reward chunk for n-step learning
3. Next observation for bootstrapping
4. Return at current step
"""

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class RLDataConfig:
    """Configuration for RL dataset with temporal structure.

    This config defines how to construct training samples for value learning:

    For a sample at timestep t with history_length=N and action_horizon=H:
        - history_obs: o_{t-N}, ..., o_{t-1}  (N past observations)
        - current_obs: o_t
        - action_chunk: a_t, a_{t+1}, ..., a_{t+H-1}  (H actions)
        - reward_chunk: r_t, r_{t+1}, ..., r_{t+H-1}  (H rewards)
        - next_obs: o_{t+H}  (for bootstrapping)
        - done: d_{t+H}  (terminal flag at action horizon end)
        - return: G_t (precomputed return at t)
    """

    history_length: int = 0
    # If None, uses same keys as current observation
    history_keys: Optional[Sequence[str]] = None
    include_history_actions: bool = False

    action_horizon: int = 10
    action_keys: Sequence[str] = ("actions",)
    reward_keys: Sequence[str] = ("reward",)
    include_done: bool = True
    done_key: str = "done"

    include_next_obs: bool = True
    # If None, uses all current observation keys
    next_obs_keys: Optional[Sequence[str]] = None

    include_return: bool = True
    return_key: str = "return"

    normalize_return: bool = False
    # If None, return_min and return_max must be provided
    return_norm_stats_path: Optional[str] = None
    return_min: Optional[float] = None
    return_max: Optional[float] = None
    keep_continuous_return: bool = True
    normalize_to_minus_one_zero: bool = True

    gamma: float = 0.99

    def get_delta_timestamps(self, fps: int) -> dict[str, list[float]]:
        """Generate delta_timestamps dict for LeRobot dataset.

        This computes the time offsets needed to fetch:
        - History observations (negative offsets)
        - Current observation (0)
        - Future actions/rewards (positive offsets)
        - Next observation at t+H

        Args:
            fps: Dataset frames per second

        Returns:
            Dict mapping feature keys to list of delta timestamps
        """
        delta_timestamps = {}

        # Helper to convert frame offset to timestamp
        def frame_to_time(frame_offset: int) -> float:
            return frame_offset / fps

        # History + current observation timestamps: [-N/fps, ..., -1/fps, 0]
        history_times = [frame_to_time(-i) for i in range(self.history_length, -1, -1)]

        # Future timestamps for action/reward: [0, 1/fps, ..., (H-1)/fps]
        future_times = [frame_to_time(i) for i in range(self.action_horizon)]

        # Next obs timestamp: [H/fps]
        next_obs_time = [frame_to_time(self.action_horizon)]

        # Done timestamp at t+H (single value for offline datasets)
        done_times = [frame_to_time(self.action_horizon)]

        # Determine observation keys for history
        if self.history_keys:
            obs_keys = list(self.history_keys)
        else:
            obs_keys = []

        # Add observation keys with history + current
        for key in obs_keys:
            delta_timestamps[key] = history_times.copy()
            # Also add next_obs time if needed
            if self.include_next_obs:
                if self.next_obs_keys is None or key in self.next_obs_keys:
                    delta_timestamps[key] = history_times + next_obs_time

        # Add action keys
        for key in self.action_keys:
            action_times = future_times.copy()
            if self.include_history_actions:
                # Include history actions: [-N/fps, ..., -1/fps, 0, ..., (H-1)/fps]
                action_times = [
                    frame_to_time(-i) for i in range(self.history_length, 0, -1)
                ] + future_times
            delta_timestamps[key] = action_times

        # Add reward keys
        for key in self.reward_keys:
            delta_timestamps[key] = future_times.copy()

        # Add done key
        if self.include_done:
            delta_timestamps[self.done_key] = done_times

        # Add return key (only current timestep)
        if self.include_return:
            delta_timestamps[self.return_key] = [0.0]

        return delta_timestamps


def create_rl_config(
    history_length: int = 0,
    history_keys: Optional[list[str]] = None,
    action_horizon: int = 10,
    include_next_obs: bool = True,
    include_return: bool = True,
    gamma: float = 0.99,
    # Return normalization
    normalize_return: bool = False,
    return_norm_stats_path: Optional[str] = None,
    return_min: Optional[float] = None,
    return_max: Optional[float] = None,
    **kwargs,
) -> RLDataConfig:
    """Factory function to create RLDataConfig with sensible defaults.

    Args:
        history_length: Number of past observations to include
        history_keys: Keys for history observations (auto-detected if None)
        action_horizon: Number of future actions/rewards
        include_next_obs: Whether to fetch obs at t+H for bootstrapping
        include_return: Whether to include precomputed return
        gamma: Discount factor
        normalize_return: Whether to normalize return values
        return_norm_stats_path: Path to norm_stats.json for min/max
        return_min: Override minimum return value
        return_max: Override maximum return value

    Returns:
        Configured RLDataConfig instance
    """
    return RLDataConfig(
        history_length=history_length,
        history_keys=tuple(history_keys) if history_keys else None,
        action_horizon=action_horizon,
        include_next_obs=include_next_obs,
        include_return=include_return,
        gamma=gamma,
        normalize_return=normalize_return,
        return_norm_stats_path=return_norm_stats_path,
        return_min=return_min,
        return_max=return_max,
        **kwargs,
    )
