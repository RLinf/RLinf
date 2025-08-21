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

from dataclasses import dataclass
from typing import Optional, Union

import torch

# Type alias, supports scalar or tensor (only for length parameters)
TensorOrScalar = Union[int, float, torch.Tensor]


def lmhead_flops(hidden_size, vocab_size, seq_length: TensorOrScalar):
    """Calculate language model head FLOPs, seq_length supports tensor input, batch_size=1"""
    return 2 * hidden_size * vocab_size * seq_length


def qkv_project_flops(
    hidden_size, num_attn_heads, num_kv_heads, seq_length: TensorOrScalar
):
    """Calculate QKV projection FLOPs, seq_length supports tensor input, batch_size=1"""
    hidden_size_kv = hidden_size // (num_attn_heads // num_kv_heads)
    return 2 * seq_length * hidden_size * (hidden_size + 2 * hidden_size_kv)


def wo_projection_flops(hidden_size, seq_length: TensorOrScalar):
    """Calculate output projection FLOPs, seq_length supports tensor input, batch_size=1"""
    return 2 * seq_length * hidden_size * hidden_size


def attention_score_flops(hidden_size, seq_length: TensorOrScalar):
    """Calculate attention score FLOPs, seq_length supports tensor input, batch_size=1"""
    return 4 * hidden_size * seq_length * seq_length


def mlp_flops(hidden_size, mlp_intermediate_size, seq_length: TensorOrScalar):
    """Calculate MLP FLOPs, seq_length supports tensor input, batch_size=1"""
    return 6 * seq_length * hidden_size * mlp_intermediate_size


def rmsnorm_flops(hidden_size, seq_length: TensorOrScalar):
    """Calculate RMSNorm FLOPs, seq_length supports tensor input, batch_size=1"""
    return 4 * seq_length * hidden_size


@dataclass
class ModelConfig:
    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    num_query_groups: Optional[int] = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    ffn_hidden_size: Optional[int] = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    override_vocab_size: int = 0


class FLOPSCalculator:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def flops_generate(
        self, prompt_length: TensorOrScalar, decode_length: TensorOrScalar
    ):
        """Generation phase FLOPs calculation, prompt_length and decode_length support tensor, batch_size=1"""
        prefill_decode_flops = self._calculate_prefill_flops(
            prompt_length=prompt_length
        ) + self._calculate_decode_flops(
            prompt_length=prompt_length, decode_length=decode_length
        )

        return prefill_decode_flops

    def flops_inference(self, seq_length: TensorOrScalar):
        """Inference phase FLOPs calculation, seq_length supports tensor, batch_size=1"""
        prefill_total_flops = self._calculate_prefill_flops(prompt_length=seq_length)

        return prefill_total_flops

    def _calculate_prefill_flops(self, prompt_length: TensorOrScalar):
        """Prefill phase FLOPs calculation, prompt_length supports tensor, batch_size=1"""
        L = self.model_config.num_layers
        H = self.model_config.hidden_size
        n_h = self.model_config.num_attention_heads
        n_qg = self.model_config.num_query_groups
        n_kv = n_h // n_qg
        I = self.model_config.ffn_hidden_size
        V = self.model_config.override_vocab_size

        qkv = qkv_project_flops(H, n_h, n_kv, prompt_length)
        attn = attention_score_flops(H, prompt_length)
        wo = wo_projection_flops(H, prompt_length)
        mlp_part = mlp_flops(H, I, prompt_length)
        norms = 2 * rmsnorm_flops(H, prompt_length)
        final_norm = rmsnorm_flops(H, prompt_length)
        lm_head = lmhead_flops(H, V, prompt_length)

        prefill_flops = L * (qkv + attn + wo + mlp_part + norms) + final_norm + lm_head

        return prefill_flops

    def _calculate_decode_flops(
        self, prompt_length: TensorOrScalar, decode_length: TensorOrScalar
    ):
        """Decode phase FLOPs calculation, prompt_length and decode_length support tensor, batch_size=1"""
        L = self.model_config.num_layers
        H = self.model_config.hidden_size
        n_h = self.model_config.num_attention_heads
        n_qg = self.model_config.num_query_groups
        n_kv = n_h // n_qg
        I = self.model_config.ffn_hidden_size
        V = self.model_config.override_vocab_size

        # Convert to tensor for vectorized computation
        if isinstance(prompt_length, torch.Tensor):
            prompt_length = prompt_length.float()
        if isinstance(decode_length, torch.Tensor):
            decode_length = decode_length.float()

        qkv = qkv_project_flops(H, n_h, n_kv, 1)
        wo = wo_projection_flops(H, 1)
        mlp = mlp_flops(H, I, 1)
        norms = 2 * rmsnorm_flops(H, 1)
        lm_head = lmhead_flops(H, V, 1)

        # Fixed FLOPs per decode step multiplied by decode_length, plus attention computation
        decode_flops = (
            decode_length * (L * (qkv + wo + mlp + norms) + lm_head)
            + 4 * L * H * (decode_length + 2 * prompt_length) * decode_length / 2
        )

        return decode_flops
