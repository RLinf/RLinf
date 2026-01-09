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

"""Base class for single-frame image reward models.

This module provides the abstract base class for image-based reward models
that process individual frames independently. Suitable for fast inference
models like ResNet binary classifiers.
"""

from abc import abstractmethod
from typing import Any, Optional

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


class BaseImageRewardModel(BaseRewardModel):
    """Base class for single-frame image reward models.

    This class is designed for reward models that process individual frames
    independently, without temporal context. It is suitable for fast inference
    models such as binary classifiers (e.g., ResNet-based success detectors).

    The input images are expected to be single frames of shape [B, C, H, W],
    where:
        - B: Batch size
        - C: Number of channels (typically 3 for RGB)
        - H: Image height
        - W: Image width

    Subclasses should implement the compute_reward method to define the
    specific reward computation logic.

    Attributes:
        image_size: Expected input image size as [C, H, W].
        threshold: Classification threshold for binary reward (default: 0.5).
        use_soft_reward: If True, use probability as reward; if False, use
            binary 0/1 reward based on threshold.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the image reward model.

        Args:
            cfg: Configuration dictionary containing:
                - image_size: List of [C, H, W] for expected image dimensions.
                - threshold: Classification threshold (default: 0.5).
                - use_soft_reward: Whether to use soft probabilities (default: False).
                - checkpoint_path: Path to model checkpoint.
        """
        super().__init__(cfg)
        self.image_size = cfg.get("image_size", [3, 224, 224])
        self.threshold = cfg.get("threshold", 0.5)
        self.use_soft_reward = cfg.get("use_soft_reward", False)

    @abstractmethod
    def compute_reward(
        self,
        observations: dict[str, Any],
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Compute rewards from single-frame observations.

        Args:
            observations: Dictionary containing:
                - 'images': Image tensor of shape [B, C, H, W] or [B, H, W, C].
                - 'states': Optional state tensor of shape [B, state_dim].
            task_descriptions: Optional task descriptions (typically not used
                for image-based models, but included for interface consistency).

        Returns:
            torch.Tensor: Reward tensor of shape [B].
        """
        pass

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for model input.

        Handles channel ordering (NHWC to NCHW) and ImageNet normalization.

        Args:
            images: Input image tensor, either [B, H, W, C] or [B, C, H, W].

        Returns:
            torch.Tensor: Preprocessed image tensor of shape [B, C, H, W],
                with ImageNet normalization applied.
        """
        # Handle channel-last format (NHWC -> NCHW)
        if images.dim() == 4 and int(images.shape[-1]) in [1, 3, 4]:
            images = images.permute(0, 3, 1, 2)

        # Normalize to [0, 1] if needed
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        # Apply ImageNet normalization (same as training)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        return images.to(self.device)

    def apply_threshold(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Apply threshold to convert probabilities to binary rewards.

        Args:
            probabilities: Probability tensor of shape [B].

        Returns:
            torch.Tensor: If use_soft_reward is True, returns probabilities.
                Otherwise, returns binary tensor (0 or 1) based on threshold.
        """
        use_soft = (
            bool(self.use_soft_reward)
            if hasattr(self.use_soft_reward, "__bool__")
            else self.use_soft_reward
        )
        if use_soft:
            return probabilities
        return (probabilities >= self.threshold).float()

    @property
    def model_type(self) -> str:
        """Return the type identifier of this reward model."""
        return "image"
