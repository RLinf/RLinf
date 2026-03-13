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


from collections import deque
from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces


class BaseFakeModelInference:
    """
    This class implements the world model inference using a fake model,
    the purpose is to define the interaction logic with the env interface.
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
        self.gen_num_image_each_step = self.cfg["gen_num_image_each_step"]

        self.model = self._load_model()
        self._init_episodes_structure()

    def _init_episodes_structure(self):
        """
        Initializes the episodes structure.
        episodes_history: pay attention that conditional frames and generated frames have different data structures.
        """

        self.episodes = [None] * self.batch_size
        self.episodes_latest_frames: deque = deque(
            [
                [{} for _ in range(self.batch_size)]
                for _ in range(self.gen_num_image_each_step)
            ],
            maxlen=self.gen_num_image_each_step,
        )
        self.episodes_history = [
            deque(maxlen=self.num_prompt_frames + self.max_episode_steps)
            for _ in range(self.batch_size)
        ]

    def _get_latest_obs_from_deques(self) -> dict[str, Any]:
        """
        Retrieves the latest observations from the episode deques for all batch environments.

        This method processes the latest frames from the episodes_latest_frames deque,
        organizing camera images and state information into a structured format suitable
        for model inference. It handles multi-camera setups and maintains task descriptions.

        Returns:
            A list of dictionaries, each containing:
                - images_and_states: Dictionary with camera images and state tensors
                - task_descriptions: List of task descriptions for each batch element
        """

        return_obs = []
        for episodes_latest_frame in self.episodes_latest_frames:
            if not episodes_latest_frame:
                continue
            for obs in episodes_latest_frame:
                assert obs, "episode_latest_frame cannot empty"

            images_and_states = {}
            for camera_name in self.camera_names:
                images_and_states[f"{camera_name}"] = torch.stack(
                    [obs[f"{camera_name}"] for obs in episodes_latest_frame], dim=0
                )
            images_and_states["state"] = torch.stack(
                [obs["observation.state"] for obs in episodes_latest_frame], dim=0
            )

            task_descriptions = [obs["task"] for obs in episodes_latest_frame]

            obs = {
                "images_and_states": images_and_states,
                "task_descriptions": task_descriptions,
            }
            return_obs.append(obs)

        return return_obs

    def _init_reset_state_ids(self, seed: int):
        """
        Initializes the reset state IDs for batch environment initialization.

        This method creates a random number generator and generates random episode IDs
        from the dataset for resetting the environments. It ensures deterministic
        behavior when a seed is provided.

        Args:
            seed: The random seed for deterministic episode selection.
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

    def _infer_next_frames(self, actions: torch.Tensor) -> Any:
        """
        (Initial implementation) Generates the next frames based on the given actions for all batch environments.

        This is a preliminary implementation for pipeline testing purposes. It generates
        random observations with the same structure as the latest observations, including
        camera images and state information for each batch element.

        Args:
            actions: The actions to take for each batch element.

        Returns:
            A list of new observation lists, each containing dictionaries with camera
            images, state tensors, and task descriptions for all batch elements.
        """

        latest_obs_list = self.episodes_latest_frames[-1]
        return_obs_list = []

        for _ in range(self.gen_num_image_each_step):
            new_obs_list = []
            for i in range(self.batch_size):
                new_obs = {}
                latest_obs = latest_obs_list[i]
                for camera_name in self.camera_names:
                    new_obs[f"{camera_name}"] = torch.randint(
                        0,
                        256,
                        latest_obs[f"{camera_name}"].shape,
                        dtype=torch.uint8,
                        device=latest_obs[f"{camera_name}"].device,
                    )

                state_tensor = latest_obs["observation.state"]
                new_obs["observation.state"] = torch.rand_like(state_tensor)

                new_obs["task"] = latest_obs["task"]
                new_obs_list.append(new_obs)
            return_obs_list.append(new_obs_list)

        return return_obs_list

    def _infer_next_rewards(
        self, new_obs_list: list[dict[str, Any]], info: dict[str, Any]
    ) -> torch.Tensor:
        """
        (Initial implementation) Infers the rewards for the next step based on the new observations.

        This is a preliminary implementation for pipeline testing purposes. It generates
        random reward values for each batch element independently.

        Args:
            new_obs_list: The list of new observations for each batch element.
            info: Additional information dictionary (currently unused).

        Returns:
            A tensor containing the inferred rewards for all batch elements.
        """
        return torch.rand(self.batch_size, dtype=torch.float32, device=self.device)

    def _calc_terminated(
        self,
        new_obs_list: list[dict[str, Any]],
        reward_list: torch.Tensor,
        info: dict[str, Any],
    ) -> torch.Tensor:
        """
        (Initial implementation) Calculates the terminated flags for all batch environments.

        This is a preliminary implementation for pipeline testing purposes. It generates
        random boolean values to determine if episodes should terminate for each batch
        element independently.

        Args:
            new_obs_list: The list of new observations for each batch element.
            reward_list: The tensor of rewards for each batch element.
            info: Additional information dictionary (currently unused).

        Returns:
            A boolean tensor indicating whether each batch element's episode is terminated.
        """
        return (
            torch.rand(self.batch_size, dtype=torch.float32,
                       device=self.device) > 0.5
        )

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

        def _padding_or_truncate_start_items(start_items: list[dict[str, Any]]):
            """Padding or truncate the start_items to the gen_num_image_each_step."""
            if len(start_items) < self.gen_num_image_each_step:
                start_items = start_items + [start_items[-1]] * (
                    self.gen_num_image_each_step - len(start_items)
                )
            elif len(start_items) > self.gen_num_image_each_step:
                start_items = start_items[-self.gen_num_image_each_step:]
            return start_items

        if seed is None:
            seed = 0

        options = options or {}

        if "episode_id" not in options:
            self._init_reset_state_ids(seed)
            options["episode_id"] = self._reset_state_ids
        if "env_idx" in options:
            env_idx = options["env_idx"]
            episode_ids = options["episode_id"][: len(env_idx)]
            for i, episode_id in zip(env_idx, episode_ids):
                self.episodes[i] = self.dataset[int(episode_id)]
                for j in range(self.gen_num_image_each_step):
                    self.episodes_latest_frames[j][i] = {}
                self.episodes_history[i].clear()

                start_items = _padding_or_truncate_start_items(
                    self.episodes[i]["start_items"]
                )
                for j, frame in enumerate(start_items):
                    self.episodes_latest_frames[j][i] = frame
                for frame in self.episodes[i]["start_items"]:
                    self.episodes_history[i].append(frame)

            return self._get_latest_obs_from_deques(), {}

        self._init_episodes_structure()

        episode_ids = options["episode_id"]
        self.episodes = [self.dataset[int(episode_id)]
                         for episode_id in episode_ids]
        assert len(self.episodes) == self.batch_size

        for i, episode in enumerate(self.episodes):
            start_items = _padding_or_truncate_start_items(
                episode["start_items"])
            for j, frame in enumerate(start_items):
                self.episodes_latest_frames[j][i] = frame
            for frame in self.episodes[i]["start_items"]:
                self.episodes_history[i].append(frame)

        self.current_step = 0
        return self._get_latest_obs_from_deques(), {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Takes a step in the environment for all batch elements.

        This method generates the next observations, calculates rewards, determines
        termination conditions, and updates the episode history for each batch element.
        It handles multiple frames per step and maintains proper episode tracking.

        Args:
            actions: The actions to take for each batch element.

        Returns:
            A tuple containing:
                - Latest observations from the updated deques
                - Stacked reward tensor for all generated frames
                - Stacked termination flags for all generated frames
                - Stacked truncation flags for all generated frames
                - List of info dictionaries for each generated frame
        """

        new_obs_list = self._infer_next_frames(actions)

        reward_list, terminated_list, truncated_list, info_list = [], [], [], []

        for i in range(self.gen_num_image_each_step):
            self.episodes_latest_frames.append(new_obs_list[i])
            info = {}
            reward_list.append(self._infer_next_rewards(new_obs_list[i], info))
            terminated_list.append(
                self._calc_terminated(new_obs_list[i], reward_list[i], info)
            )
            truncated_list.append(
                torch.zeros(self.batch_size, dtype=torch.bool,
                            device=self.device)
                if self.current_step <= self.max_episode_steps
                else torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            )
            info_list.append(info)

            for j in range(self.batch_size):
                step_data = {
                    "observation": new_obs_list[i][j],
                    "action": actions[j],
                    "reward": reward_list[i][j].item(),
                    "terminated": terminated_list[i][j].item(),
                    "truncated": truncated_list[i][j].item(),
                }
                self.episodes_history[j].append(step_data)

            self.current_step += 1

        return (
            self._get_latest_obs_from_deques(),
            reward_list,
            terminated_list,
            truncated_list,
            info_list,
        )
