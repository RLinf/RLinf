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

import os

import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.nature_cnn import (
    ResNet10,
    SpatialLearnedEmbeddings,
)


class BinaryRewardClassifier(nn.Module):
    """Frame-based binary classifier for reward prediction.
    
    This model takes a single frame (image observation) and predicts
    whether the task is completed successfully at that moment.
    """

    def __init__(
        self,
        image_keys: list,
        image_size: list,  # [c, h, w]
        hidden_dim: int = 256,
        num_spatial_blocks: int = 8,
        pretrained_encoder_path: str = None,
        use_pretrain: bool = True,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.image_keys = image_keys
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.num_spatial_blocks = num_spatial_blocks
        self.use_pretrain = use_pretrain
        self.freeze_encoder = freeze_encoder

        # Create encoders for each image key
        self.encoders = nn.ModuleDict()
        encoder_out_dim = 0
        
        sample_x = torch.randn(1, *image_size)
        for key in image_keys:
            encoder = self._create_encoder(sample_x)
            self.encoders[key] = encoder
            encoder_out_dim += encoder.out_dim

        # Binary classifier head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _create_encoder(self, sample_x):
        """Create a ResNet encoder with spatial learned embeddings."""
        encoder = ResNetEncoderWrapper(
            sample_x=sample_x,
            out_dim=self.hidden_dim,
            num_spatial_blocks=self.num_spatial_blocks,
            pretrained_encoder_path=getattr(self, 'pretrained_encoder_path', None),
            use_pretrain=self.use_pretrain,
            freeze_encoder=self.freeze_encoder,
        )
        return encoder

    def forward(self, images: dict, train: bool = False):
        """Forward pass.
        
        Args:
            images: Dict of image tensors, keyed by image_keys.
                   Each tensor should be [B, C, H, W]
            train: Whether in training mode.
        
        Returns:
            logits: [B, 1] tensor of binary classification logits
        """
        visual_features = []
        for key in self.image_keys:
            if key not in images:
                raise ValueError(f"Missing image key: {key}. Available keys: {list(images.keys())}")
            img = images[key]
            # Ensure images are in [B, C, H, W] format
            if img.dim() == 4:
                pass  # Already in correct format
            elif img.dim() == 3:
                img = img.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
            
            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            
            feature = self.encoders[key](img)
            visual_features.append(feature)
        
        # Concatenate features from all image keys
        combined_feature = torch.cat(visual_features, dim=-1)
        
        # Binary classification
        logits = self.classifier(combined_feature)
        return logits


class ResNetEncoderWrapper(nn.Module):
    """Wrapper for ResNet encoder that matches the serl_zhirui structure."""

    def __init__(
        self,
        sample_x: torch.Tensor,
        out_dim: int = 256,
        num_spatial_blocks: int = 8,
        pretrained_encoder_path: str = None,
        use_pretrain: bool = True,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_spatial_blocks = num_spatial_blocks
        self.use_pretrain = use_pretrain
        self.freeze_encoder = freeze_encoder
        self.pretrained_encoder_path = pretrained_encoder_path

        # ResNet backbone
        self.resnet_backbone = ResNet10(pre_pooling=self.use_pretrain)
        
        if self.use_pretrain and self.pretrained_encoder_path:
            self._load_pretrained_weights()
            if self.freeze_encoder:
                self._freeze_backbone_weights()

        # Get output dimensions from sample
        with torch.no_grad():
            sample_embed = self.resnet_backbone(sample_x)
            _, channel, height, width = sample_embed.shape

        # Pooling layer
        self.pooling_layer = SpatialLearnedEmbeddings(
            height=height,
            width=width,
            channel=channel,
            num_features=self.num_spatial_blocks,
        )
        self.dropout = nn.Dropout(0.1)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features=channel * self.num_spatial_blocks, out_features=self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.Tanh(),
        )

    def _load_pretrained_weights(self):
        """Load pretrained ResNet10 weights."""
        if self.pretrained_encoder_path and os.path.exists(self.pretrained_encoder_path):
            try:
                model_dict = torch.load(self.pretrained_encoder_path, map_location="cpu")
                self.resnet_backbone.load_state_dict(model_dict, strict=False)
                print(f"Loaded pretrained weights from {self.pretrained_encoder_path}")
            except Exception as e:
                print(f"Warning: Failed to load pretrained weights: {e}")
        else:
            print(f"Warning: Pretrained encoder path not found: {self.pretrained_encoder_path}")

    def _freeze_backbone_weights(self):
        """Freeze the ResNet backbone weights."""
        for p in self.resnet_backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        """Forward pass through encoder.
        
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            feature: Encoded feature tensor [B, out_dim]
        """
        x = self.resnet_backbone(x)

        if self.use_pretrain and self.freeze_encoder:
            x = x.detach()

        # Apply spatial learned embeddings pooling
        x = self.pooling_layer(x)
        x = self.dropout(x)

        # Final MLP
        x = self.mlp(x)
        return x

