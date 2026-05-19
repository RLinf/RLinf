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

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from omegaconf import DictConfig

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.reward.base_image_reward_model import BaseImageRewardModel


class ResNetRewardModel(BaseImageRewardModel):
    """ResNet-based reward model using binary classification loss.

    This model uses a pretrained ResNet backbone followed by a linear head
    to output scalar rewards. It is trained using binary cross-entropy loss
    on individual images with success/fail labels.

    Training Input: (B, C, H, W) - batch of images with labels
    Inference Input: observation dict containing ``main_images``

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
        """
        super().__init__(cfg)

        self.cfg = cfg

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

        self._load_model()

        torch_dtype = torch_dtype_from_precision(cfg.precision)
        self.to(torch_dtype)

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

    def _load_model(self):
        model_path = self.cfg.get("model_path", None)
        if model_path is not None:
            if model_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )

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

            self.load_state_dict(state_dict, strict=True)

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

    def compute_reward(self, observations: dict[str, Any]) -> torch.Tensor:
        """Compute rewards for inference.

        Args:
            observations: Observation dictionary containing ``main_images``.

        Returns:
            torch.Tensor: Reward tensor of shape [B].
        """
        images = observations.get("main_images", None)
        if images is None:
            raise ValueError(
                "Missing main_images in observations for ResNetRewardModel."
            )

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        model_parameter = next(self.parameters())
        images = images.to(device=model_parameter.device, dtype=model_parameter.dtype)

        images = self.preprocess_images(images)

        with torch.no_grad():
            logits = self.backbone(images).squeeze(-1)  # (B,)
            # Return probabilities for binary classification
            rewards = torch.sigmoid(logits)

        # Optional thresholding: keep consistent with prior worker behavior.
        threshold = self.cfg.get("reward_threshold", None)
        if threshold is not None:
            thr = float(threshold)
            rewards = torch.where(rewards > thr, rewards, torch.zeros_like(rewards))

        return rewards

    def load_from_path(self, model_path: str) -> None:
        """Load a ResNet reward checkpoint from a file path."""
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

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
        self.load_state_dict(new_state_dict, strict=True)
