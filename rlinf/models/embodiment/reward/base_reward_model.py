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

"""Base class for embodied reward models.

This module provides the abstract base class for all reward models used in
embodied reinforcement learning. Both image-based and video-based reward
models inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseRewardModel(ABC, nn.Module):
    """Root abstract class for all reward models.

    This class provides a common interface for both image-based (single-frame)
    and video-based (multi-frame) reward models. All concrete reward model
    implementations should inherit from either BaseImageRewardModel or
    BaseVideoRewardModel, which in turn inherit from this class.

    The reward model is designed to be framework-agnostic, working with
    PPO, SAC, GRPO, and other RL algorithms.

    Attributes:
        cfg: Configuration dictionary containing model parameters.
        device: The device (CPU/GPU) where the model is loaded.
        alpha: Scaling factor for reward values.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the base reward model.

        Args:
            cfg: Configuration dictionary containing:
                - alpha: Reward scaling factor (default: 1.0)
                - device: Target device for computation (default: "cuda")
        """
        super().__init__()
        self.cfg = cfg
        self.alpha = cfg.get("alpha", 1.0)
        self.device = torch.device(cfg.get("device", "cuda"))

    @abstractmethod
    def compute_reward(
        self,
        observations: dict[str, Any],
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Compute rewards from observations.

        This is the main interface method that all reward models must implement.
        The method takes environment observations and optionally task descriptions,
        and returns a tensor of reward values.

        Args:
            observations: Dictionary containing observation data. Expected keys:
                - 'images': Image tensor(s) from environment
                - 'states': Optional state vector
                - Other environment-specific observations
            task_descriptions: Optional list of task description strings,
                primarily used by VLM-based reward models.

        Returns:
            torch.Tensor: Tensor of shape [B] containing reward values for
                each sample in the batch, where B is the batch size.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.pt or .pth).
        """
        pass

    def scale_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Apply scaling factor to reward values.

        Args:
            reward: Raw reward tensor of shape [B].

        Returns:
            torch.Tensor: Scaled reward tensor of shape [B].
        """
        return reward * self.alpha

    def to_device(self, device: Optional[torch.device] = None) -> "BaseRewardModel":
        """Move model to specified device.

        Args:
            device: Target device. If None, uses self.device.

        Returns:
            Self for method chaining.
        """
        if device is not None:
            self.device = device
        return self.to(self.device)

    @property
    def model_type(self) -> str:
        """Return the type identifier of this reward model.

        Returns:
            String identifier for the model type (e.g., "image", "video").
        """
        return "base"
