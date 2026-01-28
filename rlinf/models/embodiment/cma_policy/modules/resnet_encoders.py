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

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import spaces
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from torch import Tensor

from rlinf.models.embodiment.cma_policy.modules.utils import single_frame_box_shape


class VlnResnetDepthEncoder(nn.Module):
    """Depth encoder using ResNet backbone, compatible with DDPPO pretrained weights."""

    def __init__(
        self,
        observation_space: spaces.Space = None,
        output_size: int = 128,
        checkpoint: str = "NONE",
        backbone: str = "resnet50",
        resnet_baseplanes: int = 32,
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
        depth_width: int = 256,  # For building observation_space from config
        depth_height: int = 256,
        min_depth: float = 0.0,
        max_depth: float = 10.0,
    ) -> None:
        super().__init__()

        # Build observation_space from parameters if not provided
        if observation_space is None:
            if depth_width is not None and depth_height is not None:
                try:
                    observation_space = spaces.Dict(
                        {
                            "depth": spaces.Box(
                                low=min_depth,
                                high=max_depth,
                                shape=(depth_height, depth_width, 1),
                                dtype=np.float32,
                            )
                        }
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not build observation_space from parameters: {e}"
                    )
                    observation_space = None
            else:
                raise ValueError(
                    "observation_space must be provided, or depth_width and depth_height must be specified"
                )

        # Default: use ResNetEncoder (like VLN-CE)
        self.visual_encoder = ResNetEncoder(
            spaces.Dict(
                {"depth": single_frame_box_shape(observation_space.spaces["depth"])}
            ),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE" and checkpoint is not None:
            try:
                # Use weights_only=False for compatibility with older checkpoints
                ddppo_weights = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )

                weights_dict = {}
                for k, v in ddppo_weights["state_dict"].items():
                    split_layer_name = k.split(".")[2:]
                    if (
                        len(split_layer_name) > 0
                        and split_layer_name[0] != "visual_encoder"
                    ):
                        continue

                    layer_name = (
                        ".".join(split_layer_name[1:])
                        if len(split_layer_name) > 1
                        else None
                    )
                    if layer_name:
                        weights_dict[layer_name] = v

                del ddppo_weights
                if weights_dict:
                    self.visual_encoder.load_state_dict(weights_dict, strict=False)
            except Exception as e:
                print(f"Warning: Could not load checkpoint {checkpoint}: {e}")

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )
            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations: dict[str, Tensor]) -> Tensor:
        """
        Args:
            observations: dict with "depth" key, shape [BATCH, HEIGHT, WIDTH, CHANNEL]
                          or [BATCH, CHANNEL, HEIGHT, WIDTH]
        Returns:
            [BATCH, OUTPUT_SIZE] or [BATCH, C, H, W] for spatial_output=True
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class TorchVisionResNet(nn.Module):
    """TorchVision ResNet pre-trained on ImageNet."""

    def __init__(
        self,
        output_size: int,
        resnet_version: str = "resnet50",
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
        single_spatial_filter: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_visual_inputs = normalize_visual_inputs
        self.spatial_output = spatial_output
        resnet = getattr(models, resnet_version)(pretrained=True)
        modules = list(resnet.children())
        self.resnet_layer_size = modules[-1].in_features
        self.cnn = nn.Sequential(*modules[:-1])

        for param in self.cnn.parameters():
            param.requires_grad_(trainable)
        self.cnn.train(trainable)

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.resnet_layer_size, output_size),
                nn.ReLU(),
            )
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))
                    return x

            if single_spatial_filter:
                self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
            self.cnn.avgpool = SpatialAvgPool()
            self.spatial_embeddings = nn.Embedding(4 * 4, 64)
            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

    def forward(self, observations: dict[str, Tensor]) -> Tensor:
        def normalize(imgs: Tensor) -> Tensor:
            """Normalizes a batch of images."""
            imgs = imgs.contiguous() / 255.0
            if self.normalize_visual_inputs:
                mean_norm = torch.tensor([0.485, 0.456, 0.406]).to(device=imgs.device)[
                    None, :, None, None
                ]
                std_norm = torch.tensor([0.229, 0.224, 0.225]).to(device=imgs.device)[
                    None, :, None, None
                ]
                return imgs.sub(mean_norm).div(std_norm)
            else:
                return imgs

        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            # Handle different input formats
            rgb = observations["rgb"]
            if len(rgb.shape) == 4 and rgb.shape[-1] == 3:
                # [B, H, W, C] -> [B, C, H, W]
                rgb = rgb.permute(0, 3, 1, 2)
            resnet_output = self.cnn(normalize(rgb))

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            return self.fc(resnet_output)


class TorchVisionResNet50(TorchVisionResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, resnet_version="resnet50", **kwargs)


class TorchVisionResNet18(TorchVisionResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, resnet_version="resnet18", **kwargs)
