# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SigLIP vision encoder for PyTorch, aligned with JAX models/siglip.py.

Implements the So400m/14 ViT variant with pool_type="none" used by Pi0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _str_to_dtype


def posemb_sincos_2d(
    h: int, w: int, width: int, temperature: float = 10000.0
) -> torch.Tensor:
    """2D sine-cosine positional embedding following MoCo v3 logic."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    assert width % 4 == 0, "Width must be multiple of 4 for sincos posemb"
    omega = torch.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = torch.einsum("m,d->md", y.flatten().float(), omega)
    x = torch.einsum("m,d->md", x.flatten().float(), omega)

    pe = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    return pe[None, :, :]  # (1, h*w, width)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(self, dim: int, mlp_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        mlp_dim = mlp_dim or 4 * dim
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    def __init__(
        self, dim: int, num_heads: int, mlp_dim: int | None = None, dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = MlpBlock(dim, mlp_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to norm weight dtype for FSDP1 mixed precision compatibility
        norm_dtype = self.norm1.weight.dtype
        y = self.norm1(x.to(norm_dtype))
        y, _ = self.attn(y, y, y)
        y = self.dropout1(y)
        x = x + y

        y = self.norm2(x.to(norm_dtype))
        y = self.mlp(y)
        y = self.dropout2(y)
        x = x + y
        return x


class Encoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [Encoder1DBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.gradient_checkpointing = use_gradient_checkpointing
        # Whether the activation checkpoint uses reentrant autograd. Configurable
        # via Pi0.gradient_checkpointing_enable(gradient_checkpointing_kwargs=...).
        self.gradient_checkpointing_use_reentrant = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=self.gradient_checkpointing_use_reentrant
                )
            else:
                x = layer(x)
        x = self.norm(x.to(self.norm.weight.dtype))
        return x


class SigLIPViT(nn.Module):
    """SigLIP Vision Transformer for Pi0. Uses So400m/14 by default with pool_type="none".

    Mimics JAX behavior: stem + pos_embed run in float32, then cast to dtype_mm for encoder.
    """

    def __init__(
        self,
        variant: str = "So400m/14",
        pool_type: str = "none",
        num_classes: int = 0,
        use_gradient_checkpointing: bool = False,
        dtype_mm: str = "float32",
    ):
        super().__init__()

        # Decode variant
        params = _decode_variant(variant)
        self.width = params["width"]
        self.depth = params["depth"]
        self.num_heads = params["num_heads"]
        self.mlp_dim = params["mlp_dim"]
        self.patch_size = params.get("patch_size", (16, 16))
        self.pool_type = pool_type
        self.dtype_mm = (
            _str_to_dtype(dtype_mm) if isinstance(dtype_mm, str) else dtype_mm
        )

        # Patch embedding (Conv2d)
        self.stem = nn.Conv2d(
            3,
            self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=True,
        )

        # Positional embedding — stored as learnable parameter (matching JAX)
        # For 224x224 input with patch_size=16: 14x14 patches = 256 tokens
        # For 224x224 input with patch_size=14: 16x16 patches = 256 tokens
        num_patches = (224 // self.patch_size[0]) * (224 // self.patch_size[1])
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, self.width))

        # Encoder
        self.encoder = Encoder(
            dim=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=0.0,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Output head: projects width -> num_classes (used when num_classes matches PaliGemma width)
        if num_classes > 0:
            self.head = nn.Linear(self.width, num_classes)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            self.head = None

        # CLS token for pool_type="tok"
        if pool_type == "tok":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.width))

        # MAP head for pool_type="map"
        if pool_type == "map":
            self.map_head = MAPHead(self.width, self.num_heads, self.mlp_dim)

        # Determine expected input size for pos emb
        # Default input is 224x224, so features = 14x14 with patch_size=16
        self._init_weights()

    def _init_weights(self):
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.stem.weight)
        if not hasattr(self, "_pos_emb_cache"):
            self._pos_emb_cache = {}

    def _get_pos_emb(
        self, h: int, w: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Get or compute positional embeddings for given feature map size."""
        key = (h, w)
        if key not in self._pos_emb_cache:
            pe = posemb_sincos_2d(h, w, self.width)
            self._pos_emb_cache[key] = pe
        return self._pos_emb_cache[key].to(dtype=dtype, device=device)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            image: (B, H, W, 3) in [-1, 1] float32

        Returns:
            tokens: (B, num_patches, width) - all patch tokens when pool_type="none"
            _: placeholder for compatibility (None)
        """
        # --- Stem + pos_embed in float32 (matching JAX) ---
        # image is (B, H, W, C) -> (B, C, H, W) for Conv2d
        x = image.permute(0, 3, 1, 2).float()

        # Patch extraction in float32 — use explicit F.conv2d because FSDP2 may
        # cast stem parameters to bfloat16, but JAX stem always runs in float32.
        x = F.conv2d(
            x,
            self.stem.weight.float(),
            self.stem.bias.float() if self.stem.bias is not None else None,
            stride=self.stem.stride,
            padding=self.stem.padding,
        )  # (B, width, h, w)
        B, C, h, w = x.shape
        x = x.reshape(B, C, h * w).permute(0, 2, 1)  # (B, h*w, width)

        # Add position embedding in float32
        x = x + self.pos_embedding.float()

        if self.pool_type == "tok":
            cls_tokens = self.cls_token.float().expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # --- Cast to model dtype for encoder (matching JAX x.astype(self.dtype_mm)) ---
        x = x.to(self.dtype_mm)

        # Encoder
        x = self.encoder(x)

        if self.pool_type == "map":
            x = self.map_head(x)
        elif self.pool_type == "gap":
            x = x.mean(dim=1)
        elif self.pool_type == "tok":
            x = x[:, 0]
        # pool_type == "none": keep all tokens

        # Apply output head projection (no tanh — JAX num_classes head doesn't apply activation)
        if self.head is not None:
            x = self.head(x)

        return x, None


class MAPHead(nn.Module):
    """Multihead Attention Pooling head."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int | None = None):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.probe)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MlpBlock(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        probe = self.probe.expand(B, -1, -1)
        x, _ = self.attn(probe, x, x)
        y = self.norm(x)
        x = x + self.mlp(y)
        return x[:, 0]


def _decode_variant(variant: str) -> dict:
    """Converts a string like "B" or "B/32" into a params dict."""
    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
