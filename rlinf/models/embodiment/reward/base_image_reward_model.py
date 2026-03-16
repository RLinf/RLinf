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

"""Base class for image-based reward models.

This module provides the abstract base class for reward models that process
image observations. It handles common image preprocessing tasks like
normalization and channel ordering.
"""

from abc import abstractmethod
from typing import Any, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


class BaseImageRewardModel(BaseRewardModel):
    """Base class for image-based reward models.

    This class is designed for reward models that process image observations.
    It provides common image preprocessing utilities and defines the interface
    for image-based reward computation.

    Attributes:
        image_size: Expected input image size as [C, H, W].
        normalize: Whether to apply ImageNet normalization.
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, cfg: DictConfig):
        """Initialize the image reward model.

        Args:
            cfg: Configuration dictionary containing:
                - image_size: List of [C, H, W] for expected image dimensions.
                - normalize: Whether to apply ImageNet normalization (default: True).
        """
        super().__init__(cfg)
        self.image_size = cfg.get("image_size", [3, 224, 224])
        self.normalize = cfg.get("normalize", True)

        # Register normalization constants as buffers (will move with model)
        self.register_buffer(
            "_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for model input.

        Handles:
        1. Channel ordering (NHWC to NCHW if needed)
        2. Data type conversion (uint8 to float)
        3. Resize to expected image size
        4. ImageNet normalization (optional)

        Args:
            images: Input image tensor, either [B, H, W, C] or [B, C, H, W].

        Returns:
            torch.Tensor: Preprocessed image tensor of shape [B, C, H, W],
                normalized to ImageNet statistics if enabled.
        """
        # Handle channel-last format (NHWC -> NCHW)
        if images.dim() == 4 and images.shape[-1] in [1, 3, 4]:
            images = images.permute(0, 3, 1, 2)

        # Convert to float and normalize to [0, 1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.0:
            images = images / 255.0

        # Resize to expected image size if needed
        target_h, target_w = self.image_size[1], self.image_size[2]
        if images.shape[2] != target_h or images.shape[3] != target_w:
            images = F.interpolate(
                images,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        # Apply ImageNet normalization
        if self.normalize:
            images = (images - self._mean) / self._std

        return images.to(self.device)

    @abstractmethod
    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Forward pass for training.

        Args:
            input_data: Image tensor, typically (B, 2, C, H, W) for pairwise training.
            labels: Optional labels (usually not needed for pairwise loss).

        Returns:
            Dictionary with loss, accuracy, and other metrics.
        """
        pass

    @abstractmethod
    def compute_reward(
        self,
        observations: dict[str, Any],
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Compute rewards from image observations.

        Args:
            observations: Dictionary containing:
                - 'images' or 'main_images': Image tensor of shape [B, C, H, W].
            task_descriptions: Not used by image models.

        Returns:
            torch.Tensor: Reward tensor of shape [B].
        """
        pass

    @property
    def model_type(self) -> str:
        """Return the type identifier of this reward model."""
        return "image"
