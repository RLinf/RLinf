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


from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class WMBackendBase(gym.Env, ABC):
    """
    Abstract base class for a world model backend.
    This class defines the interface for a world model, which is responsible for
    simulating the environment and generating observations. Subclasses must
    implement the abstract methods defined here.
    """

    def __init__(self, cfg: dict[str, Any], dataset: Any, device: Any):
        """
        Initializes the world model backend.
        Args:
            cfg: Configuration dictionary.
            dataset: The dataset used by the world model.
            device: The device to run the model on.
        """
        self.cfg = cfg
        self.dataset = dataset
        self.device = device

        action_dim = self.dataset.action_dim
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self.camera_names = self.dataset.camera_names
        self.max_episode_steps = self.cfg["max_episode_steps"]
        self.current_step = 0

        self.batch_size = self.cfg["batch_size"]
        self.num_prompt_frames = self.cfg["num_prompt_frames"]

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Resets the environment to its initial state.
        Args:
            seed: The random seed for the environment.
            options: Additional options for resetting the environment.
        Returns:
            A tuple containing the initial observation and a dictionary of info.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Takes a step in the environment.
        Args:
            action: The action to take.
        Returns:
            A tuple containing the next observation, reward, terminated flag,
            truncated flag, and a dictionary of info.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self) -> Any:
        """Renders the environment."""
        raise NotImplementedError


class WorldModelBackend(WMBackendBase):
    """
    World model backend implementation.
    This class implements the world model backend interface using a specific model.
    """

    def __init__(self, cfg: dict[str, Any], dataset: Any, device: Any):
        super().__init__(cfg, dataset, device)

        self.model = self._load_model()
        self.episodes_current_frames: list[deque] = [
            deque(maxlen=self.num_prompt_frames) for _ in range(self.batch_size)
        ]

    def _get_latest_obs_from_deques(self) -> dict[str, Any]:
        """
        Retrieves the latest observation from the deques.
        Returns:
            A dictionary containing the latest observations and task descriptions.
        """
        latest_obs_list = [d[-1] for d in self.episodes_current_frames]

        images_and_states = {}
        for camera_name in self.camera_names:
            images_and_states[f"{camera_name}"] = torch.stack(
                [obs[f"{camera_name}"] for obs in latest_obs_list], axis=0
            )
        images_and_states["state"] = torch.stack(
            [obs["observation.state"] for obs in latest_obs_list], axis=0
        )

        task_descriptions = [obs["task"] for obs in latest_obs_list]

        return {
            "images_and_states": images_and_states,
            "task_descriptions": task_descriptions,
        }

    def _init_reset_state_ids(self, seed: int):
        """
        Initializes the reset state IDs.
        Args:
            seed: The random seed.
        """
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)
        self._reset_state_ids = torch.randint(
            0, len(self.dataset), (self.batch_size,), generator=self._generator
        )

    def _load_model(self) -> None:
        """Loads the world model."""
        pass

    def _load_reward_model(self) -> None:
        """Loads the reward model."""
        pass

    def _generate_next_frame(self, action: Any) -> Any:
        """
        Generates the next frame based on the given action.
        Args:
            action: The action to take.
        Returns:
            The generated next frame.
        """
        pass

    def _calc_step_reward(self):
        """Calculates the reward for the current step."""
        pass

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Resets the environment to its initial state.
        Args:
            seed: The random seed for the environment.
            options: Additional options for resetting the environment.
        Returns:
            A tuple containing the initial observation and a dictionary of info.
        """
        if seed is None:
            seed = 0

        if "episode_id" not in options:
            self._init_reset_state_ids(seed)
            options["episode_id"] = self._reset_state_ids
        if "env_idx" in options:
            env_idx = options["env_idx"]
            episode_ids = options["episode_id"][: len(env_idx)]
            for i, episode_id in zip(env_idx, episode_ids):
                self.episodes[i] = self.dataset[int(episode_id)]
                self.episodes_current_frames[i].clear()
                for frame in self.episodes[i]["start_items"]:
                    self.episodes_current_frames[i].append(frame)

            return self._get_latest_obs_from_deques(), {}

        episode_ids = options["episode_id"]
        self.episodes = [self.dataset[int(episode_id)] for episode_id in episode_ids]

        for i, episode in enumerate(self.episodes):
            self.episodes_current_frames[i].clear()
            for frame in episode["start_items"]:
                self.episodes_current_frames[i].append(frame)

        self.current_step = 0
        return self._get_latest_obs_from_deques(), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Takes a step in the environment.
        Args:
            action: The action to take.
        Returns:
            A tuple containing the next observation, reward, terminated flag,
            truncated flag, and a dictionary of info.
        """
        self.current_step += 1

        new_obs_list = self._generate_next_frame(action)
        for i in range(self.batch_size):
            self.episodes_current_frames[i].append(new_obs_list[i])

        info = {}

        return self._get_latest_obs_from_deques(), None, None, None, info

    def render(self) -> Any:
        # TODO: Implement rendering logic
        pass
