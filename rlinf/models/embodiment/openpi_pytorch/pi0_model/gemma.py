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

"""Gemma backbone for PyTorch, aligned with JAX models/gemma.py.

Supports:
- Multiple experts sharing self-attention but with independent FFN
- RMSNorm (regular and adaptive/adaRMS)
- GQA (Grouped Query Attention) with RoPE
- KV caching for efficient inference
- LoRA for attention and FFN layers
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from . import lora
from .lora import FeedForward as LoRAFeedForward
from .utils import _str_to_dtype, gelu_glu

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": lora.LoRAConfig(rank=16, alpha=16.0),
                "ffn": lora.LoRAConfig(rank=16, alpha=16.0),
            },
        )
    if variant == "gemma_2b_lora_32":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": lora.LoRAConfig(rank=32, alpha=32.0),
                "ffn": lora.LoRAConfig(rank=32, alpha=32.0),
            },
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": lora.LoRAConfig(rank=32, alpha=32.0),
                "ffn": lora.LoRAConfig(rank=32, alpha=32.0),
            },
        )
    raise ValueError(f"Unknown variant: {variant}")


class RMSNorm(nn.Module):
    """RMSNorm with optional adaptive mode (adaRMS)."""

    def __init__(self, dim: int, adaptive: bool = False):
        super().__init__()
        self.dim = dim
        self.adaptive = adaptive
        if not self.adaptive:
            self.scale = nn.Parameter(torch.zeros(dim))
        else:
            self.ada_modulation = nn.Linear(dim, dim * 3)
            nn.init.zeros_(self.ada_modulation.weight)
            nn.init.zeros_(self.ada_modulation.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: input tensor of shape (..., dim)
            cond: optional conditioning for adaptive RMSNorm, shape (..., dim)

        Returns:
            (normalized_output, gate) where gate is None for regular RMSNorm
        """
        dtype = x.dtype
        x_float = x.float()
        var = torch.mean(x_float**2, dim=-1, keepdim=True)
        normed = x_float * torch.rsqrt(var + 1e-6)

        if not self.adaptive:
            scale = 1.0 + self.scale.float()
            normed = normed * scale
            return normed.to(dtype), None

        # adaptive RMSNorm
        target_dtype = self.ada_modulation.weight.dtype
        modulation = self.ada_modulation(cond.to(target_dtype))  # (..., 3*dim)
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        if x.dim() == 3:
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
            gate = gate.unsqueeze(-2)

        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


class Embedder(nn.Module):
    """Embedder module."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x * math.sqrt(self.embed_dim)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.weight)


# TODO: try use sdpa_attn
class Attention(nn.Module):
    """Multi-expert Grouped Query Attention with RoPE and LoRA."""

    def __init__(self, configs: Sequence[Config]):
        super().__init__()
        self.expert_configs = configs
        self.num_heads = configs[0].num_heads
        self.num_kv_heads = configs[0].num_kv_heads
        self.head_dim = configs[0].head_dim
        self.num_query_groups = self.num_heads // self.num_kv_heads

        # Q projections per expert
        self.q_proj = nn.ModuleList()
        self.k_proj = nn.ModuleList()
        self.v_proj = nn.ModuleList()
        self.o_proj = nn.ModuleList()

        for config in configs:
            lora_cfg = config.lora_configs.get("attn")
            # TODO: add lora attn support
            if lora_cfg is not None:
                raise NotImplementedError
            if config.num_kv_heads == config.num_heads:
                # Combined QKV projection
                self.q_proj.append(
                    nn.Linear(
                        config.width, 3 * config.num_heads * config.head_dim, bias=False
                    )
                )
                self.k_proj.append(None)  # handled by q_proj
                self.v_proj.append(None)
            else:
                self.q_proj.append(
                    nn.Linear(
                        config.width, config.num_heads * config.head_dim, bias=False
                    )
                )
                self.k_proj.append(
                    nn.Linear(
                        config.width, config.num_kv_heads * config.head_dim, bias=False
                    )
                )
                self.v_proj.append(
                    nn.Linear(
                        config.width, config.num_kv_heads * config.head_dim, bias=False
                    )
                )

            self.o_proj.append(
                nn.Linear(config.num_heads * config.head_dim, config.width, bias=False)
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for i, config in enumerate(self.expert_configs):
            # Q projection
            nn.init.normal_(self.q_proj[i].weight, std=1.0 / math.sqrt(config.width))
            if self.k_proj[i] is not None:
                nn.init.normal_(
                    self.k_proj[i].weight, std=1.0 / math.sqrt(config.width)
                )
                nn.init.normal_(
                    self.v_proj[i].weight, std=1.0 / math.sqrt(config.width)
                )
            # O projection
            nn.init.normal_(
                self.o_proj[i].weight,
                std=1.0 / math.sqrt(config.num_heads * config.head_dim),
            )

    def forward(
        self,
        xs: list[torch.Tensor | None],
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
        """Multi-expert attention forward.

        Concatenates all expert inputs along the sequence dimension (like JAX),
        computes unified self-attention, then splits outputs back per expert.

        Args:
            xs: list of inputs, one per expert (or None)
            positions: (B, T) positional indices covering full concatenated sequence
            attn_mask: (B, 1, T, S) attention mask
            kv_cache: optional (k, v) cache tuple

        Returns:
            (outputs, new_kv_cache)
        """
        dtype = next(x.dtype for x in xs if x is not None)

        q_parts, k_parts, v_parts = [], [], []
        for i, x in enumerate(xs):
            if x is None:
                q_parts.append(None)
                k_parts.append(None)
                v_parts.append(None)
                continue

            B, T, _ = x.shape
            if self.k_proj[i] is None:
                qkv = self.q_proj[i](x)
                qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
                q, k, v = qkv.unbind(dim=2)
            else:
                q = self.q_proj[i](x).reshape(B, T, self.num_heads, self.head_dim)
                k = self.k_proj[i](x).reshape(B, T, self.num_kv_heads, self.head_dim)
                v = self.v_proj[i](x).reshape(B, T, self.num_kv_heads, self.head_dim)

            q_parts.append(q)
            k_parts.append(k)
            v_parts.append(v)

        # Concatenate along sequence dimension (dim=1) — matching JAX jnp.concatenate(axis=1)
        q = torch.cat([q for q in q_parts if q is not None], dim=1)
        k = torch.cat([k for k in k_parts if k is not None], dim=1)
        v = torch.cat([v for v in v_parts if v is not None], dim=1)

        # Apply RoPE
        q = _apply_rope(q, positions=positions)
        k = _apply_rope(k, positions=positions)

        # KV cache: concatenate along sequence dim
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)

        new_kv_cache = (k, v)

        # Apply mask: shape (B, 1, T, S) -> broadcast to (B, K, G, T, S)
        if attn_mask.dim() == 4:
            attn_mask = attn_mask[:, 0:1, :, :]  # (B, 1, T, S)

        # GQA einsum pattern matching JAX:
        q = q * (self.head_dim**-0.5)

        # q: (B, T, num_heads, H) -> rearrange to (B, T, K, G, H)
        # k: (B, S, num_kv_heads, H) -> stays (B, S, K, H)
        K = self.num_kv_heads
        G = self.num_heads // K

        q_r = q.reshape(q.shape[0], q.shape[1], K, G, self.head_dim)
        k_r = k.reshape(k.shape[0], k.shape[1], K, self.head_dim)
        v_r = v.reshape(v.shape[0], v.shape[1], K, self.head_dim)

        # einsum "BTKGH,BSKH->BKGTS"
        logits = torch.einsum("BTKGH,BSKH->BKGTS", q_r.float(), k_r.float())

        # Align mask to logits shape: logits is (B, K, G, T, S), mask is (B, 1, T, S)
        # We need mask to be (B, 1, 1, T, S) so it broadcasts to (B, K, G, T, S)
        big_neg = -2.3819763e38
        mask_for_logits = attn_mask[:, :, None, :, :].expand_as(logits).bool()
        masked_logits = torch.where(
            mask_for_logits,
            logits,
            torch.tensor(big_neg, dtype=logits.dtype, device=logits.device),
        )

        probs = F.softmax(masked_logits, dim=-1).to(dtype)

        # einsum "BKGTS,BSKH->BTKGH"
        encoded = torch.einsum("BKGTS,BSKH->BTKGH", probs, v_r.to(dtype))
        encoded = encoded.reshape(
            encoded.shape[0], encoded.shape[1], K * G, self.head_dim
        )
        # encoded: (B, T_total, num_heads, head_dim)

        # Split back to per-expert outputs
        outputs = []
        start = 0
        for i, x in enumerate(xs):
            if x is not None:
                end = start + x.shape[1]
                expert_out = encoded[:, start:end].reshape(B, x.shape[1], -1)
                expert_out = self.o_proj[i](expert_out)
                outputs.append(expert_out)
                start = end
            else:
                outputs.append(None)

        return outputs, new_kv_cache


class FeedForward(nn.Module):
    """Feed forward module."""

    def __init__(self, features: int, hidden_dim: int):
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.w_gating = nn.Parameter(torch.empty(2, features, hidden_dim))
        self.w_linear = nn.Parameter(torch.empty(hidden_dim, features))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w_gating[0], std=1.0 / math.sqrt(self.features))
        nn.init.normal_(self.w_gating[1], std=1.0 / math.sqrt(self.features))
        nn.init.normal_(self.w_linear, std=1.0 / math.sqrt(self.hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        ff_gate = torch.matmul(x, self.w_gating[0].to(dtype))
        ff1 = torch.matmul(x, self.w_gating[1].to(dtype))
        activations = gelu_glu(ff_gate, ff1)
        outputs = torch.matmul(activations, self.w_linear.to(dtype))
        return outputs


class Block(nn.Module):
    """Transformer block with multi-expert attention and FFN."""

    def __init__(
        self,
        configs: Sequence[Config],
        adarms: Sequence[bool] | bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.configs = configs
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if isinstance(adarms, bool):
            adarms = [adarms] * len(configs)
        self.adarms = list(adarms)

        self.attn = Attention(configs)
        self.pre_attention_norms = nn.ModuleList(
            [RMSNorm(c.width, adaptive=adarms[i]) for i, c in enumerate(configs)]
        )
        self.pre_ffw_norms = nn.ModuleList(
            [RMSNorm(c.width, adaptive=adarms[i]) for i, c in enumerate(configs)]
        )

        # FFN: use LoRA version if lora config is present, else standard
        self.mlps = nn.ModuleList()
        for config in configs:
            lora_cfg = config.lora_configs.get("ffn")
            if lora_cfg is not None:
                self.mlps.append(
                    LoRAFeedForward(config.width, config.mlp_dim, lora_cfg)
                )
            else:
                self.mlps.append(FeedForward(config.width, config.mlp_dim))

    def forward(
        self,
        xs: list[torch.Tensor | None],
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        adarms_cond: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            xs: per-expert inputs
            kv_cache: optional KV cache
            positions: (B, T) position indices
            attn_mask: attention mask
            adarms_cond: per-expert adaptive RMS norm conditioning (None = regular)

        Returns:
            (outputs, new_kv_cache)
        """
        # Pre-attention norm
        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x_norm, gate = self.pre_attention_norms[i](x, adarms_cond[i])
                pre_attn.append(x_norm)
                gates.append(gate)
            else:
                pre_attn.append(None)
                gates.append(None)

        # Attention
        post_attn, kv_cache = self.attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = [self.dropout(p) if p is not None else None for p in post_attn]

        # Gated residual for attention
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, post_attn, gates, strict=True)
        ]

        # Pre-FFN norm
        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x_norm, gate = self.pre_ffw_norms[i](x, adarms_cond[i])
                x_ffn = self.mlps[i](x_norm)
                out.append(x_ffn)
                gates.append(gate)
            else:
                out.append(None)
                gates.append(None)

        # Dropout and gated residual for FFN
        out = [self.dropout(o) if o is not None else None for o in out]
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, out, gates, strict=True)
        ]

        return xs, kv_cache


class Module(nn.Module):
    """Gemma transformer with multi-expert support."""

    def __init__(
        self,
        configs: Sequence[Config],
        embed_dtype: str = "bfloat16",
        adarms: Sequence[bool] | bool = False,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.configs = configs
        self.embed_dtype = _str_to_dtype(embed_dtype)
        if isinstance(adarms, bool):
            adarms = [adarms] * len(configs)
        self.adarms = list(adarms)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=configs[0].width,
        )

        self.layers = nn.ModuleList(
            [
                Block(configs, adarms=self.adarms, dropout=dropout)
                for _ in range(configs[0].depth)
            ]
        )

        self.final_norms = nn.ModuleList(
            [RMSNorm(c.width, adaptive=self.adarms[i]) for i, c in enumerate(configs)]
        )

        self.gradient_checkpointing = use_gradient_checkpointing
        # Whether the activation checkpoint uses reentrant autograd. Configurable
        # via Pi0.gradient_checkpointing_enable(gradient_checkpointing_kwargs=...).
        self.gradient_checkpointing_use_reentrant = False

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed token indices."""
        return self.embedder.encode(tokens).to(self.embed_dtype)

    def forward(
        self,
        embedded: Sequence[torch.Tensor | None],
        positions: torch.Tensor,
        mask: torch.Tensor,
        adarms_cond: Sequence[torch.Tensor | None] | None = None,
        *,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
        """Full transformer forward pass.

        Args:
            embedded: list of embedded tokens, one per expert (or None)
            positions: (B, T) position indices
            mask: (B, T, S) attention mask (bool)
            adarms_cond: per-expert adaptive conditioning (or None)
            kv_cache: optional KV cache for inference

        Returns:
            (outputs, new_kv_cache)
        """
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        # Expand mask: (B, T, S) -> (B, 1, T, S)
        mask = mask.unsqueeze(1)

        xs = list(embedded)

        # Cast all embedded tokens to embed_dtype, matching JAX:
        #   embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        xs = [e.to(self.embed_dtype) if e is not None else None for e in xs]

        # KV cache: per-layer cache for inference. During training, no cache is used.
        # In JAX's nn.scan, each layer gets its own element from the cache list.
        # We replicate this with per-layer caches.
        if kv_cache is None:
            layer_kv_caches = [None] * len(self.layers)
        else:
            layer_kv_caches = (
                list(kv_cache)
                if isinstance(kv_cache, list | tuple)
                else [kv_cache] * len(self.layers)
            )

        new_layer_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv = layer_kv_caches[i] if i < len(layer_kv_caches) else None
            if self.gradient_checkpointing and self.training:
                xs, new_kv = torch.utils.checkpoint.checkpoint(
                    layer,
                    xs,
                    layer_kv,
                    positions,
                    mask,
                    adarms_cond,
                    use_reentrant=self.gradient_checkpointing_use_reentrant,
                )
            else:
                xs, new_kv = layer(xs, layer_kv, positions, mask, adarms_cond)
            new_layer_kv_caches.append(new_kv)

        # Return the list of per-layer KV caches
        kv_cache = tuple(new_layer_kv_caches)

        # Final norm
        outputs = []
        for i in range(len(self.configs)):
            if xs[i] is not None:
                out, _ = self.final_norms[i](xs[i], adarms_cond[i])
                outputs.append(out)
            else:
                outputs.append(None)

        return outputs, kv_cache


@torch.compile
def _apply_rope(
    x: torch.Tensor, *, positions: torch.Tensor, max_wavelength: float = 10000.0
) -> torch.Tensor:
    """Apply RoPE to input tensor."""
    head_dim = x.shape[-1]
    freq_exponents = (2.0 / head_dim) * torch.arange(
        head_dim // 2, dtype=torch.float32, device=x.device
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].float() / timescale[None, None, :]
    radians = radians[..., None, :]  # (B, T, 1, head_dim//2)

    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return res.to(x.dtype)


@torch.compile
def _fused_gated_residual(x, y, gate):
    """Fuse the gated residual ``x + y * gate`` into a single kernel."""
    return x + y * gate


def _gated_residual(x, y, gate):
    if x is None:
        return None
    if gate is None:
        return x + y
    return _fused_gated_residual(x, y, gate)
