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

from functools import partial

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

from .utils import init_mlp_weights


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
    def __init__(self, sample_x, out_dim=256, encoder_cfg=None):
        super().__init__()

        self.out_dim = out_dim
        self.encoder_cfg = encoder_cfg

        self.num_spatial_blocks = 8
        self.pooling_method = "spatial_learned_embeddings"
        self.use_pretrain = True

        self.resnet_backbone = ResNet10(pre_pooling=self.use_pretrain)
        if self.use_pretrain:
            self._load_pretrained_weights()
            self._freeze_backbone_weights()

        sample_embed = self.resnet_backbone(sample_x)
        _, channel, height, width = sample_embed.shape
        # pooling
        if self.pooling_method == "spatial_learned_embeddings":
            self.pooling_layer = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            )
            self.dropout = nn.Dropout(0.1)

        # final linear
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=channel * self.num_spatial_blocks, out_features=self.out_dim
            ),
            nn.LayerNorm(self.out_dim),
            nn.Tanh(),
        )
        init_mlp_weights(self.mlp, nonlinearity="tanh")

    def _load_pretrained_weights(self):
        assert "ckpt_path" in self.encoder_cfg, (
            "Please use model_path and ckpt_name to specify the pretrained encoder weights path."
        )
        model_dict = torch.load(self.encoder_cfg["ckpt_path"])
        self.resnet_backbone.load_state_dict(model_dict)

    def _freeze_backbone_weights(self):
        for p in self.resnet_backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.resnet_backbone(x)

        if self.use_pretrain:
            x = x.detach()

        if self.pooling_method == "spatial_learned_embeddings":
            x = self.pooling_layer(x)
            x = self.dropout(x)

        x = self.mlp(x)
        return x
