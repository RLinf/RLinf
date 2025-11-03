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

import torch
import torch.nn as nn
from .utils import make_mlp
from functools import partial
from torchvision.models.resnet import ResNet, BasicBlock

class NatureCNN(nn.Module):
    def __init__(self, obs_dims):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256

        image_dims = obs_dims["image_dims"]
        in_channels = image_dims[0]

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            sample_obs = torch.rand(**image_dims)
            n_flatten = cnn(sample_obs).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in obs_dims:
            # for state data we simply pass it through a single linear layer
            state_size = obs_dims["state"]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class PlainConv(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_dim=256,
                 pool_feature_map=False,
                 last_act=True, # True for ConvBody, False for CNN
                 image_size=[128, 128]
                 ):
        super().__init__()
        # assume input image size is 128x128 or 64x64

        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4) if image_size[0] == 128 and image_size[1] == 128 else nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(64, 64, 1, padding=0, bias=True), nn.ReLU(inplace=True),
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
    def __init__(self, num_channels, num_groups, eps = 0.00001, affine = True, device=None, dtype=None):
        super().__init__(num_groups, num_channels, eps, affine, device, dtype)
    
class ResNet10(ResNet):
    def __init__(self, pre_pooling=True):
        self.pre_pooling = pre_pooling
        super().__init__(
            block=BasicBlock, 
            layers=[1, 1, 1, 1], 
            num_classes=1000, 
            norm_layer=partial(MyGroupNorm, num_groups=4, eps=1e-5)
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
        ) # TODO: In SeRL, this is lecun_normal initialization

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
    def __init__(self, sample_x, out_dim=256):
        super().__init__()

        self.out_dim = out_dim
        
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
            nn.Linear(in_features=channel*self.num_spatial_blocks, out_features=self.out_dim), 
            nn.LayerNorm(self.out_dim), 
            nn.Tanh()
        )

    def _load_pretrained_weights(self):
        pretrained_ckpt = "./resnet10_pretrained.pt"
        model_dict = torch.load(pretrained_ckpt)
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