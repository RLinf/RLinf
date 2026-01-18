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

"""ResNet-based reward model for embodied RL.

This module implements a ResNet-based reward model that uses binary
cross-entropy loss for training. It is designed for fast inference during
online RL training, similar to the HIL-SERL approach.
"""

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_image_reward_model import BaseImageRewardModel

logger = logging.getLogger(__name__)


class ResNetRewardModel(BaseImageRewardModel):
    """ResNet-based reward model using binary classification loss.

    This model uses a pretrained ResNet backbone followed by a linear head
    to output scalar rewards. It is trained using binary cross-entropy loss
    on individual images with success/fail labels.

    Training Input: (B, C, H, W) - batch of images with labels
    Inference Input: (B, C, H, W) - batch of single images

    Attributes:
        backbone: ResNet feature extractor with modified final layer.
        arch: Architecture name (e.g., "resnet18", "resnet50").
    """

    # Supported ResNet architectures
    SUPPORTED_ARCHS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    def __init__(self, cfg: DictConfig):
        """Initialize the ResNet reward model.

        Args:
            cfg: Configuration dictionary containing:
                - arch: ResNet architecture (default: "resnet18").
                - pretrained: Whether to use pretrained weights (default: True).
                - hidden_dim: Optional hidden dimension for MLP head.
                - dropout: Dropout rate for classification head (default: 0.1).
                - checkpoint_path: Optional path to load trained weights.
        """
        super().__init__(cfg)

        self.arch = cfg.get("arch", "resnet18")
        if self.arch not in self.SUPPORTED_ARCHS:
            raise ValueError(
                f"Unsupported architecture: {self.arch}. "
                f"Supported: {self.SUPPORTED_ARCHS}"
            )

        self.pretrained = cfg.get("pretrained", True)
        self.hidden_dim = cfg.get("hidden_dim", None)
        self.dropout_rate = cfg.get("dropout", 0.1)

        # Build model architecture
        self._build_model()

        # Load checkpoint if provided
        checkpoint_path = cfg.get("checkpoint_path")
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        elif checkpoint_path:
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}, using random weights"
            )

    def _build_model(self) -> None:
        """Build the ResNet backbone and reward head."""
        # Load pretrained ResNet backbone
        weights = "IMAGENET1K_V1" if self.pretrained else None
        self.backbone = getattr(models, self.arch)(weights=weights)

        # Get the number of features from the original fc layer
        num_features = self.backbone.fc.in_features

        # Replace the final fc layer with reward head
        if self.hidden_dim is not None:
            # MLP head with hidden layer
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            # Simple linear head
            self.backbone.fc = nn.Linear(num_features, 1)

        # Initialize weights
        self._init_head_weights()

    def _init_head_weights(self) -> None:
        """Initialize the reward head weights."""
        for module in self.backbone.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Forward pass for training with binary classification loss.

        Args:
            input_data: Image tensor of shape (B, C, H, W).
            labels: Binary labels (B,) where 1=success, 0=fail.

        Returns:
            Dictionary containing:
                - "loss": Binary cross entropy loss (scalar tensor).
                - "accuracy": Classification accuracy.
                - "logits": Raw model outputs (B,).
                - "probabilities": Sigmoid probabilities (B,).
        """
        # Input shape: (B, C, H, W)
        images = input_data

        # Preprocess images (normalization, etc.)
        images = self.preprocess_images(images)

        # Forward through backbone
        logits = self.backbone(images).squeeze(-1)  # (B,)

        # Compute probabilities
        probabilities = torch.sigmoid(logits)

        # Compute loss if labels provided
        if labels is not None:
            labels = labels.float().to(logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            # Compute accuracy
            predictions = (probabilities > 0.5).float()
            accuracy = (predictions == labels).float().mean()
        else:
            loss = torch.tensor(0.0, device=logits.device)
            accuracy = torch.tensor(0.0, device=logits.device)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "logits": logits,
            "probabilities": probabilities,
        }

    def compute_reward(
        self,
        observations: dict[str, Any],
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Compute rewards for inference.

        Args:
            observations: Dictionary containing:
                - 'images' or 'main_images': Image tensor of shape [B, C, H, W]
                    or [B, H, W, C].
            task_descriptions: Not used by this model.

        Returns:
            torch.Tensor: Reward tensor of shape [B].
        """
        # Get images from observations
        images = observations.get("images")
        if images is None:
            images = observations.get("main_images")
        if images is None:
            raise ValueError("Observations must contain 'images' or 'main_images' key")

        # Preprocess and compute rewards
        images = self.preprocess_images(images)

        with torch.no_grad():
            logits = self.backbone(images).squeeze(-1)  # (B,)
            # Return probabilities for binary classification
            rewards = torch.sigmoid(logits)

        return rewards

    def compute_reward_from_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Compute rewards directly from image tensor.

        Convenience method for inference when images are already tensors.

        Args:
            images: Image tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Reward tensor of shape [B].
        """
        images = self.preprocess_images(images)

        with torch.no_grad():
            rewards = self.backbone(images).squeeze(-1)

        return rewards

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint.

        Handles multiple checkpoint formats:
        1. Full model state dict
        2. State dict with 'state_dict' or 'model_state_dict' key
        3. Backbone-only weights
        4. SafeTensors format (.safetensors)

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint based on format
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path)
            logger.info(f"Loaded {len(state_dict)} keys from safetensors")
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
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

        # Remove common prefixes from keys (from DDP/FSDP training)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["module.", "_orig_mod.", "model."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            # Skip mean/std buffers (they are persistent=False, auto-created)
            if new_key in ["mean", "std", "_mean", "_std"]:
                continue
            new_state_dict[new_key] = v
        state_dict = new_state_dict

        # Try to load full model
        try:
            self.load_state_dict(state_dict, strict=True)
            logger.info("Loaded full model state dict")
        except RuntimeError as e:
            # Try loading backbone only
            logger.warning(f"Could not load full state dict: {e}")
            backbone_state = {
                k.replace("backbone.", ""): v
                for k, v in state_dict.items()
                if k.startswith("backbone.")
            }
            if backbone_state:
                self.backbone.load_state_dict(backbone_state, strict=False)
                logger.info("Loaded backbone state dict only")
            else:
                # Try non-strict loading
                logger.info("Trying non-strict loading...")
                self.load_state_dict(state_dict, strict=False)
                logger.info("Loaded state dict with non-strict mode")

    def save_checkpoint(self, checkpoint_path: str, **extra_info) -> None:
        """Save model checkpoint.

        Args:
            checkpoint_path: Path to save the checkpoint.
            **extra_info: Additional information to save (e.g., epoch, metrics).
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": dict(self.cfg),
            "arch": self.arch,
            **extra_info,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    @property
    def model_type(self) -> str:
        """Return the type identifier of this reward model."""
        return "resnet_image"
