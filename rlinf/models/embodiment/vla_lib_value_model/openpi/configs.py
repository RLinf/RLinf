# Copyright 2024 Big Vision Authors.
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

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""
import dataclasses
from typing import Literal


@dataclasses.dataclass
class Config:
    """Gemma action expert configuration.
    
    Parameter count formula (without embeddings, as action expert sets embed_tokens=None):
    
    Per layer:
      - Attention:
          q_proj: width * (num_heads * head_dim)
          k_proj: width * (num_kv_heads * head_dim)
          v_proj: width * (num_kv_heads * head_dim)
          o_proj: (num_heads * head_dim) * width
      - MLP (GeGLU):
          gate_proj: width * mlp_dim
          up_proj:   width * mlp_dim
          down_proj: mlp_dim * width
      - RMSNorm (weight only):
          input_layernorm:          width
          post_attention_layernorm: width
    
    Total params = depth * (
        width * (num_heads * head_dim) * 2 +       # q_proj + o_proj
        width * (num_kv_heads * head_dim) * 2 +    # k_proj + v_proj
        width * mlp_dim * 3 +                       # gate + up + down
        width * 2                                   # layernorms
    ) + width                                       # final norm
    """
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


Variant = Literal[
    "dummy",
    "gemma_1m",
    "gemma_50m", "gemma_100m", "gemma_150m", "gemma_300m", "gemma_2b",
]


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
    if variant == "gemma_1m":
        # ~1.21M params, lightweight 4-layer expert for fast iteration / debugging
        return Config(
            width=128,
            depth=4,
            mlp_dim=448,
            num_heads=1,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_50m":
        # ~56M params (head_dim=256 required for cross-attention with gemma_2b backbone)
        return Config(
            width=384,
            depth=18,
            mlp_dim=1536,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_100m":
        # ~110M params (head_dim=256 required for cross-attention with gemma_2b backbone)
        return Config(
            width=512,
            depth=18,
            mlp_dim=2048,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_150m":
        # ~165M params (head_dim=256 required for cross-attention with gemma_2b backbone)
        return Config(
            width=640,
            depth=18,
            mlp_dim=2560,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_300m":
        # ~311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        # ~1.98B params
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    raise ValueError(f"Unknown variant: {variant}")
