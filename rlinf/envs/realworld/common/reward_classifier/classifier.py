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

"""Binary reward classifier using a frozen ResNet-10 backbone.

Architecture
------------
::

    image → ResNet-10 (frozen) → SpatialLearnedEmbeddings → 256-d bottleneck
         → Dense(256) → Dropout → LayerNorm → ReLU → Dense(1) → logit

The ResNet-10 backbone reuses the PyTorch implementation in
``rlinf.models.embodiment.modules.resnet_utils``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.modules.resnet_utils import (
    ResNet10,
    SpatialLearnedEmbeddings,
)


class RewardClassifier(nn.Module):
    """Binary image classifier for visual reward computation.

    Args:
        image_keys: Camera observation keys used as inputs (e.g.
            ``["wrist_1"]``).  Each key gets its own spatial-embedding
            pooling head; their outputs are concatenated before the
            classification head.
        pretrained_ckpt: Path to the pretrained ResNet-10 checkpoint.
            If ``None``, the backbone is randomly initialised.
        bottleneck_dim: Output dimensionality of each per-camera encoder.
        hidden_dim: Hidden size of the classification head.
        num_spatial_blocks: Number of spatial-learned-embedding features.
        image_size: ``(H, W)`` to which inputs are resized.
        dropout: Dropout probability in the classification head.
    """

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
        self,
        image_keys: list[str],
        pretrained_ckpt: Optional[str] = None,
        bottleneck_dim: int = 256,
        hidden_dim: int = 256,
        num_spatial_blocks: int = 8,
        image_size: tuple[int, int] = (128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_keys = list(image_keys)
        self.image_size = image_size

        # Frozen ResNet-10 backbone (shared across cameras)
        self.backbone = ResNet10(pre_pooling=True)
        if pretrained_ckpt is not None:
            state_dict = torch.load(pretrained_ckpt, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Per-camera spatial pooling + bottleneck
        # We do a single forward pass to determine the feature-map shape.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *image_size)
            feat = self.backbone(dummy)
            _, C, H, W = feat.shape

        self.pooling_heads = nn.ModuleDict()
        self.bottleneck_heads = nn.ModuleDict()
        for key in image_keys:
            self.pooling_heads[key] = SpatialLearnedEmbeddings(
                height=H, width=W, channel=C, num_features=num_spatial_blocks,
            )
            self.bottleneck_heads[key] = nn.Sequential(
                nn.Linear(C * num_spatial_blocks, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.Tanh(),
            )

        # Classification head
        total_embed_dim = bottleneck_dim * len(image_keys)
        self.classifier_head = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise uint8 ``[0, 255]`` images to ImageNet statistics.

        Expects ``(B, H, W, C)`` **or** ``(B, C, H, W)``.  Returns
        ``(B, C, H, W)`` float tensor.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # Channels-last → channels-first
        if x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        mean = self._IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
        std = self._IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        # Resize if necessary
        if x.shape[2:] != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        return x

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        *,
        return_prob: bool = False,
    ) -> torch.Tensor:
        """Compute classifier logits (or probabilities) from camera images.

        Args:
            observations: Mapping from ``image_key`` to image tensor.
                Each tensor can be ``uint8 [0, 255]`` or ``float [0, 1]``
                and shaped ``(B, H, W, 3)`` or ``(B, 3, H, W)``.
            return_prob: If ``True``, return ``sigmoid(logit)`` instead.

        Returns:
            Tensor of shape ``(B,)`` — per-sample logit or probability.
        """
        embeddings = []
        with torch.no_grad():
            for key in self.image_keys:
                img = self._preprocess(observations[key])
                feat = self.backbone(img)  # (B, C, H, W), frozen
                embeddings.append((key, feat))

        parts = []
        for key, feat in embeddings:
            pooled = self.pooling_heads[key](feat)
            bottleneck = self.bottleneck_heads[key](pooled)
            parts.append(bottleneck)

        combined = torch.cat(parts, dim=-1)
        logit = self.classifier_head(combined).squeeze(-1)  # (B,)
        if return_prob:
            return torch.sigmoid(logit)
        return logit


def load_reward_classifier(
    checkpoint_path: str,
    image_keys: list[str],
    pretrained_ckpt: Optional[str] = None,
    device: str | torch.device = "cpu",
) -> RewardClassifier:
    """Load a trained :class:`RewardClassifier` from a checkpoint.

    Args:
        checkpoint_path: Path to ``*.pt`` saved via ``torch.save(model.state_dict(), ...)``.
        image_keys: Camera observation keys the classifier was trained with.
        pretrained_ckpt: Path to the ResNet-10 pretrained weights (only
            needed if the backbone weights are not stored in the checkpoint).
        device: Target device.

    Returns:
        A ``RewardClassifier`` in eval mode on the requested device.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Support both raw state_dict and wrapped {"model_state_dict": ...} format
    state_dict = ckpt.get("model_state_dict", ckpt)
    model = RewardClassifier(image_keys=image_keys, pretrained_ckpt=pretrained_ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model
