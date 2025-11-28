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
from functools import partial

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

from .utils import make_mlp


class PlainConv(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_dim=256,
        pool_feature_map=False,
        last_act=True,  # True for ConvBody, False for CNN
        image_size=[128, 128],
    ):
        super().__init__()
        # assume input image size is 128x128 or 64x64

        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4)
            if image_size[0] == 128 and image_size[1] == 128
            else nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(64, 64, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(64 * 4 * 4, [out_dim], last_act=last_act)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class MyGroupNorm(nn.GroupNorm):
    """
    Reorganize the order of params to keep compatible to ResNet.
    """

    def __init__(
        self,
        num_channels,
        num_groups,
        eps=0.00001,
        affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__(num_groups, num_channels, eps, affine, device, dtype)


class ResNet10(ResNet):
    def __init__(self, pre_pooling=True):
        self.pre_pooling = pre_pooling
        super().__init__(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=1000,
            norm_layer=partial(MyGroupNorm, num_groups=4, eps=1e-5),
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove the last linear.
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


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=5):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(
            torch.randn(channel, height, width, num_features)
        )  # TODO: In SeRL, this is lecun_normal initialization

    def forward(self, features):
        """
        features: (B, C, H, W)
        """

        # expand to (B, C, H, W, F)
        weighted = features.unsqueeze(-1) * self.kernel.unsqueeze(0)
        # sum over H,W  -> (B, C, F)
        summed = weighted.sum(dim=(2, 3))
        # reshape -> (B, C*F)
        out = summed.reshape(features.shape[0], -1)

        return out


class ResNetEncoder(nn.Module):
    """ResNet encoder with spatial learned embeddings.

    This class provides a unified ResNet encoder that can be used in both
    policy networks and reward classifiers. It supports pretrained weights,
    freezing, and configurable spatial pooling.
    """

    def __init__(
        self,
        sample_x,
        out_dim=256,
        num_spatial_blocks=8,
        pretrained_encoder_path=None,
        use_pretrain=True,
        freeze_encoder=True,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.num_spatial_blocks = num_spatial_blocks
        self.use_pretrain = use_pretrain
        self.freeze_encoder = freeze_encoder
        self.pretrained_encoder_path = pretrained_encoder_path
        self.pooling_method = "spatial_learned_embeddings"

        # ResNet backbone
        self.resnet_backbone = ResNet10(pre_pooling=self.use_pretrain)

        if self.use_pretrain and self.pretrained_encoder_path:
            self._load_pretrained_weights()
            if self.freeze_encoder:
                self._freeze_backbone_weights()
        elif self.use_pretrain and not self.pretrained_encoder_path:
            # Backward compatibility: try default path
            default_path = "./resnet10_pretrained.pt"
            if os.path.exists(default_path):
                self.pretrained_encoder_path = default_path
                self._load_pretrained_weights()
                if self.freeze_encoder:
                    self._freeze_backbone_weights()

        # Get output dimensions from sample
        with torch.no_grad():
            sample_embed = self.resnet_backbone(sample_x)
            _, channel, height, width = sample_embed.shape

        # Pooling layer
        if self.pooling_method == "spatial_learned_embeddings":
            self.pooling_layer = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            )
            self.dropout = nn.Dropout(0.1)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=channel * self.num_spatial_blocks, out_features=self.out_dim
            ),
            nn.LayerNorm(self.out_dim),
            nn.Tanh(),
        )

    def _load_pretrained_weights(self):
        """Load pretrained ResNet10 weights."""
        if self.pretrained_encoder_path and os.path.exists(
            self.pretrained_encoder_path
        ):
            try:
                model_dict = torch.load(
                    self.pretrained_encoder_path, map_location="cpu"
                )
                self.resnet_backbone.load_state_dict(model_dict, strict=False)
                print(f"Loaded pretrained weights from {self.pretrained_encoder_path}")
            except Exception as e:
                print(f"Warning: Failed to load pretrained weights: {e}")
        else:
            if self.pretrained_encoder_path:
                print(
                    f"Warning: Pretrained encoder path not found: {self.pretrained_encoder_path}"
                )

    def _freeze_backbone_weights(self):
        """Freeze the ResNet backbone weights."""
        for p in self.resnet_backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        """Forward pass through encoder.

        Args:
            x: Input image tensor [B, C, H, W]

        Returns:
            feature: Encoded feature tensor [B, out_dim]
        """
        x = self.resnet_backbone(x)

        if self.use_pretrain and self.freeze_encoder:
            x = x.detach()

        if self.pooling_method == "spatial_learned_embeddings":
            x = self.pooling_layer(x)
            x = self.dropout(x)

        x = self.mlp(x)
        return x
