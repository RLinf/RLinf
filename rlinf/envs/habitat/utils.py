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

"""Utils for Habitat environments."""

import numpy as np
from habitat.core.env import RLEnv


class HabitatRLEnv(RLEnv):
    def __init__(self, config):
        super().__init__(config)

    def reset(self):
        observations = super().reset()
        
        metrics = self.habitat_env.get_metrics()
        self.previous_distance = metrics.get("geodesic_to_goal", 0)
        
        return observations

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        metrics = self.habitat_env.get_metrics()
        current_distance = metrics.get("geodesic_to_goal", 0)

        reward = self.previous_distance - current_distance
        
        reward -= 0.01
        
        if metrics.get("success", False):
            reward += 10.0
            
        self.previous_distance = current_distance
        
        return reward

    def get_done(self, observations):
        done = False
        # 1. if episode is over
        if self.habitat_env.episode_over:
            done = True
        # 2. if collision is detected
        metrics = self.habitat_env.get_metrics()
        if metrics.get("collisions", 0) > 0:
            done = True
            
        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        return info


def get_habitat_rgb_image(obs: dict) -> np.ndarray:
    """
    Extracts image from Habitat observations and preprocesses it.

    Args:
        obs: Observation dictionary from Habitat environment

    Returns:
        Preprocessed image as numpy array
    """
    # Habitat typically provides RGB observations
    if "rgb" in obs:
        img = obs["rgb"]
    elif "color_sensor" in obs:
        img = obs["color_sensor"]
    else:
        # Fallback: try to find any image-like observation
        for key in obs.keys():
            if isinstance(obs[key], np.ndarray) and len(obs[key].shape) == 3:
                img = obs[key]
                break
        else:
            raise ValueError("Could not find image observation in Habitat obs")

    # Ensure image is in RGB format (H, W, 3)
    if img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]  # Convert RGBA to RGB
    return img


def get_habitat_state(obs: dict) -> np.ndarray:
    """
    Extracts state information from Habitat observations.

    Args:
        obs: Observation dictionary from Habitat environment

    Returns:
        State vector as numpy array
    """
    # Habitat may provide different state information
    # Adapt based on your specific Habitat setup
    state_parts = []

    # Try to extract common state information
    if "gps" in obs:
        state_parts.append(obs["gps"].flatten())
    if "compass" in obs:
        state_parts.append(obs["compass"].flatten())
    if "pointgoal" in obs:
        state_parts.append(obs["pointgoal"].flatten())

    # If no specific state found, return empty array
    if len(state_parts) == 0:
        return np.array([])

    return np.concatenate(state_parts)


