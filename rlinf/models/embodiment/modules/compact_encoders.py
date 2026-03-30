# Copyright 2026 The RLinf Authors.
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
#
# Lightweight encoders inspired by dsrl_pi0 design.
# Replaces heavy ResNet18 + 1024-dim encoders with compact, efficient alternatives.

import math

import torch
import torch.nn as nn


class LightweightImageEncoder64(nn.Module):
    """
    Lightweight CNN encoder for 64x64 image features (matching dsrl_pi0).

    Architecture:
        - 4 Conv layers with (32, 32, 32, 32) filters
        - Strides: (2, 1, 1, 1) for downsampling
        - Bottleneck: compress to latent_dim

    Input: 64x64 images -> Output: latent_dim features
    Parameters: ~13K (much smaller than 224x224 version)

    Note: This is specifically designed for 64x64 inputs to match dsrl_pi0.
    """

    def __init__(self, num_images=1, latent_dim=64, image_size=64):
        super().__init__()
        self.num_images = num_images
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Calculate feature map size after convolutions
        # Input: [num_images * 3, 64, 64]
        # After stride 2: 32x32
        # After stride 1: remains 32x32
        final_h = image_size // 2  # 32
        final_w = image_size // 2  # 32
        self.flat_dim = 32 * final_h * final_w  # 32 * 32 * 32 = 32,768

        # CNN encoder (same architecture as LightweightImageEncoder)
        self.encoder = nn.Sequential(
            # Conv1: stride=2 for downsampling (64 -> 32)
            nn.Conv2d(num_images * 3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Conv2-4: stride=1, preserve spatial size (32x32)
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Bottleneck: compress to latent_dim
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using orthogonal initialization (like dsrl_pi0)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images):
        """
        Args:
            images: [B, N, C, H, W] where N=num_images (expected H=W=64)

        Returns:
            features: [B, latent_dim]
        """
        B, N, C, H, W = images.shape

        # Stack images: [B, N, C, H, W] -> [B, N*C, H, W]
        x = images.view(B, N * C, H, W)

        # Cast input to match encoder weight dtype (e.g., fp32 data -> bf16 weights)
        weight_dtype = self.encoder[0].weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)

        # CNN encoder
        x = self.encoder(x)  # [B, 32, 32, 32]

        # Bottleneck
        features = self.bottleneck(x)  # [B, latent_dim]

        return features


class LightweightImageEncoder(nn.Module):
    """
    Lightweight CNN encoder for image features, inspired by dsrl_pi0.

    Architecture:
        - 4 Conv layers with (32, 32, 32, 32) filters
        - Strides: (2, 1, 1, 1) for downsampling
        - Bottleneck: compress to latent_dim

    Parameters: ~50K (vs 11M for ResNet18)
    """

    def __init__(self, num_images=2, latent_dim=64, image_size=256):
        super().__init__()
        self.num_images = num_images
        self.latent_dim = latent_dim

        # Calculate feature map size after convolutions
        # Input: [num_images * 3, H, W]
        # After stride 2: H/2, W/2
        # After stride 1: remains same
        final_h = image_size // 2
        final_w = image_size // 2
        self.flat_dim = 32 * final_h * final_w

        # CNN encoder (similar to dsrl_pi0)
        self.encoder = nn.Sequential(
            # Conv1: stride=2 for downsampling
            nn.Conv2d(num_images * 3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Conv2-4: stride=1, preserve spatial size
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Bottleneck: compress to latent_dim
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using orthogonal initialization (like dsrl_pi0)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images):
        """
        Args:
            images: [B, N, C, H, W] where N=num_images

        Returns:
            features: [B, latent_dim]
        """
        B, N, C, H, W = images.shape

        # Stack images: [B, N, C, H, W] -> [B, N*C, H, W]
        x = images.view(B, N * C, H, W)

        # CNN encoder
        x = self.encoder(x)  # [B, 32, H/2, W/2]

        # Bottleneck
        features = self.bottleneck(x)  # [B, latent_dim]

        return features


class CompactStateEncoder(nn.Module):
    """
    Compact state encoder, inspired by dsrl_pi0's direct concatenation.

    Instead of projecting 32-dim state to 1024-dim, we keep it small.
    This matches dsrl_pi0's philosophy: state information is already compact.
    """

    def __init__(self, state_dim=32, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
        )

    def forward(self, state):
        """
        Args:
            state: [B, state_dim]

        Returns:
            features: [B, hidden_dim]
        """
        # Flatten if needed
        if state.dim() > 2:
            state = state.reshape(state.shape[0], -1)

        # Cast input to match encoder weight dtype
        weight_dtype = self.encoder[0].weight.dtype
        if state.dtype != weight_dtype:
            state = state.to(dtype=weight_dtype)

        return self.encoder(state)


class CompactQHead(nn.Module):
    """
    Compact Q-head for DSRL, inspired by dsrl_pi0's lightweight design.

    Architecture:
        - Input: [state_features + image_features + action_features]
        - Hidden: (128, 128, 128) to match dsrl_pi0
        - Output: Q-value

    Parameters: ~50K per Q-head (vs 1M in original ResNet18 version)
    """

    def __init__(
        self,
        state_dim=64,
        image_dim=64,
        action_dim=32,
        hidden_dims=(128, 128, 128),  # Match dsrl_pi0
        output_dim=1,
    ):
        super().__init__()

        input_dim = state_dim + image_dim + action_dim

        # Build MLP
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend(
                [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()]
            )
            in_dim = out_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize Q-head weights.

        Key points:
        1. Hidden layers: Use Kaiming Normal (fan_out) for ReLU activation
        2. Output layer: Use small Normal(0, 0.01) initialization
            - This ensures initial Q-values are close to 0
            - Prevents Q-value explosion during early training
        """
        for i, m in enumerate(self.net):
            if isinstance(m, nn.Linear):
                if i == len(self.net) - 1:  # Output layer
                    # Small init: std=0.01 so Q-values start near 0
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:  # Hidden layers
                    # Kaiming Normal for ReLU activations
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, state_features, image_features, actions):
        """
        Args:
            state_features: [B, state_dim]
            image_features: [B, image_dim]
            actions: [B, action_dim]

        Returns:
            q_values: [B, 1]
        """
        x = torch.cat([state_features, image_features, actions], dim=-1)
        # Cast input to match network weight dtype
        weight_dtype = self.net[0].weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)
        q_values = self.net(x)
        return q_values


class CompactMultiQHead(nn.Module):
    """
    Multiple compact Q-heads for DSRL (10 Q-networks).

    Total parameters: ~500K (vs 10M in original ResNet18 version)
    """

    def __init__(
        self,
        state_dim=64,
        image_dim=64,
        action_dim=32,
        hidden_dims=(128, 128, 128),  # Match dsrl_pi0
        num_q_heads=10,
        output_dim=1,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads

        self.q_heads = nn.ModuleList(
            [
                CompactQHead(state_dim, image_dim, action_dim, hidden_dims, output_dim)
                for _ in range(num_q_heads)
            ]
        )

    def forward(self, state_features, image_features, actions):
        """
        Args:
            state_features: [B, state_dim]
            image_features: [B, image_dim]
            actions: [B, action_dim]

        Returns:
            q_values: [B, num_q_heads]
        """
        q_values = []
        for q_head in self.q_heads:
            q_values.append(q_head(state_features, image_features, actions))

        return torch.cat(q_values, dim=-1)
