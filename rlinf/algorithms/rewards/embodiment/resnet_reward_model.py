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

"""ResNet-based binary reward model for embodied RL.

This module implements a ResNet-based binary classifier for reward prediction,
similar to the HIL-SERL approach. It predicts success/failure from single
image frames for fast inference during online RL training.
"""

import os
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models.resnet import BasicBlock, ResNet

from rlinf.algorithms.rewards.embodiment.base_image_reward_model import (
    BaseImageRewardModel,
)


class MyGroupNorm(nn.GroupNorm):
    """GroupNorm with reordered parameters for ResNet compatibility.

    The standard GroupNorm expects (num_groups, num_channels), but ResNet's
    norm_layer interface expects (num_channels, ...). This wrapper reorders
    the parameters for compatibility.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(num_groups, num_channels, eps, affine, device, dtype)


class ResNet10Backbone(ResNet):
    """ResNet-10 backbone for feature extraction.

    A lightweight ResNet variant with [1, 1, 1, 1] block configuration,
    suitable for fast inference in real-time reward computation.

    Attributes:
        pre_pooling: If True, returns features before global average pooling.
    """

    def __init__(self, pre_pooling: bool = False):
        """Initialize ResNet-10 backbone.

        Args:
            pre_pooling: If True, skip the final avgpool layer and return
                spatial features of shape [B, 512, H', W'].
        """
        self.pre_pooling = pre_pooling
        super().__init__(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=1000,
            norm_layer=partial(MyGroupNorm, num_groups=4, eps=1e-5),
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without the final classification layer.

        Args:
            x: Input tensor of shape [B, 3, H, W].

        Returns:
            Feature tensor. Shape is [B, 512, H', W'] if pre_pooling=True,
            otherwise [B, 512, 1, 1].
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.pre_pooling:
            return x
        x = self.avgpool(x)
        return x


class ResNetRewardModel(BaseImageRewardModel):
    """ResNet-based binary reward model for success/failure prediction.

    This model uses a ResNet-10 backbone followed by a binary classification
    head to predict whether a task has been completed successfully. It is
    designed for fast inference during online RL training, similar to the
    HIL-SERL approach.

    The model can output either:
    - Soft rewards: Probability of success (0.0 to 1.0)
    - Binary rewards: 0 or 1 based on threshold

    Attributes:
        backbone: ResNet-10 feature extractor.
        classifier: Binary classification head.
        freeze_backbone: Whether to freeze backbone weights during training.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the ResNet reward model.

        Args:
            cfg: Configuration dictionary containing:
                - checkpoint_path: Path to model checkpoint.
                - image_size: Input image size as [C, H, W].
                - threshold: Classification threshold (default: 0.5).
                - use_soft_reward: Use probability as reward (default: False).
                - freeze_backbone: Freeze backbone weights (default: True).
                - hidden_dim: Hidden dimension for classifier (default: 256).
        """
        super().__init__(cfg)

        self.freeze_backbone = cfg.get("freeze_backbone", True)
        self.hidden_dim = cfg.get("hidden_dim", 256)

        # Build model architecture
        self._build_model()

        # Load checkpoint if provided
        checkpoint_path = cfg.get("checkpoint_path")
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_path)
            else:
                import logging
                logging.getLogger(__name__).warning(
                    f"Checkpoint not found at {checkpoint_path}, using random weights"
                )
        else:
            import logging
            logging.getLogger(__name__).info(
                "No checkpoint_path provided, using random weights (test mode)"
            )

        # Move to device
        self.to_device()

    def _build_model(self) -> None:
        """Build the ResNet backbone and classification head."""
        # ResNet-10 backbone (without final fc layer)
        self.backbone = ResNet10Backbone(pre_pooling=False)

        # Binary classification head
        # ResNet-10 outputs 512-dimensional features after avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Initialize classifier weights
        self._init_classifier_weights()

    def _init_classifier_weights(self) -> None:
        """Initialize classifier weights using Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_reward(
        self,
        observations: Dict[str, Any],
        task_descriptions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute binary rewards from image observations.

        Args:
            observations: Dictionary containing:
                - 'images' or 'main_images': Image tensor of shape [B, C, H, W]
                    or [B, H, W, C].
                - 'states': Optional state tensor (not used by this model).
            task_descriptions: Not used by this model, included for interface
                consistency.

        Returns:
            torch.Tensor: Reward tensor of shape [B]. Values are either
                probabilities (if use_soft_reward=True) or binary 0/1
                (if use_soft_reward=False).
        """
        # Get images from observations (avoid `or` which triggers tensor bool eval)
        images = observations.get("images")
        if images is None:
            images = observations.get("main_images")
        if images is None:
            raise ValueError("Observations must contain 'images' or 'main_images' key")

        # Preprocess images
        images = self.preprocess_images(images)

        # Forward pass
        freeze = bool(self.freeze_backbone) if hasattr(self.freeze_backbone, '__bool__') else self.freeze_backbone
        with torch.no_grad() if freeze else torch.enable_grad():
            features = self.backbone(images)

        # Classification
        probabilities = self.classifier(features).squeeze(-1)  # [B]

        # Apply threshold if needed
        rewards = self.apply_threshold(probabilities)

        # Scale rewards
        return self.scale_reward(rewards)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint.

        The checkpoint can contain either:
        - Full model state dict (with 'backbone' and 'classifier' keys)
        - Backbone-only state dict (for pretrained ResNet weights)

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Try to load full model first
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Try loading backbone only
            backbone_state_dict = {
                k.replace("backbone.", ""): v
                for k, v in state_dict.items()
                if k.startswith("backbone.") or not any(
                    k.startswith(prefix) for prefix in ["classifier.", "fc."]
                )
            }
            self.backbone.load_state_dict(backbone_state_dict, strict=False)

        # Freeze backbone if configured
        freeze = bool(self.freeze_backbone) if hasattr(self.freeze_backbone, '__bool__') else self.freeze_backbone
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for training.

        Args:
            images: Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Probability tensor of shape [B].
        """
        features = self.backbone(images)
        return self.classifier(features).squeeze(-1)

