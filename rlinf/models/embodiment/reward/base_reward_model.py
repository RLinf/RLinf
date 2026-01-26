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

"""Base class for reward models in embodied RL.

This module defines the abstract interface for reward models used in
reinforcement learning from human feedback (RLHF) or similar paradigms.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseRewardModel(nn.Module, ABC):
    """Abstract base class for reward models.

    This class defines the interface that all reward models must implement.
    Reward models are used to predict scalar rewards from observations,
    typically trained using pairwise comparison data (chosen vs rejected).

    Attributes:
        cfg: Configuration dictionary containing model parameters.
        device: The device (CPU/GPU) where the model is located.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the reward model.

        Args:
            cfg: Configuration dictionary containing:
                - checkpoint_path: Optional path to model checkpoint.
                - device: Device to place the model on (default: "cuda").
        """
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(cfg.get("device", "cuda"))

    @property
    def device(self) -> torch.device:
        """Return the device where the model is located."""
        # Get actual device from model parameters
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._device

    def to_device(self, device: Optional[torch.device] = None) -> "BaseRewardModel":
        """Move the model to the specified device.

        Args:
            device: Target device. If None, uses self._device.

        Returns:
            self for method chaining.
        """
        if device is not None:
            self._device = device
        self.to(self._device)
        return self

    @abstractmethod
    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Forward pass for training.

        This method should compute rewards and loss for training.
        The loss calculation should be encapsulated within this method
        following the DRY principle.

        Args:
            input_data: Input tensor, format depends on subclass.
            labels: Optional labels for supervised training.

        Returns:
            Dictionary containing at least:
                - "loss": Scalar loss tensor for backpropagation.
                - "accuracy": Classification accuracy (for pairwise models).
                - Additional metrics as needed.
        """
        pass

    @abstractmethod
    def compute_reward(
        self,
        observations: Any,
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Compute rewards for inference.

        This method is used during RL training to get reward signals.
        It should NOT compute gradients.

        Args:
            observations: Observation data (images, states, etc.).
            task_descriptions: Optional task descriptions for multi-task models.

        Returns:
            torch.Tensor: Reward tensor of shape [B] or [B, 1].
        """
        pass

    def scale_reward(
        self,
        rewards: torch.Tensor,
        scale: float = 1.0,
        shift: float = 0.0,
    ) -> torch.Tensor:
        """Apply scaling and shifting to rewards.

        Args:
            rewards: Raw reward tensor.
            scale: Multiplicative scale factor.
            shift: Additive shift.

        Returns:
            Scaled rewards: scale * rewards + shift.
        """
        return scale * rewards + shift

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the type identifier of this reward model.

        Returns:
            String identifier (e.g., "image", "video", "text").
        """
        pass
