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
    r"""
    Calculates the theoretical FLOPs for Qwen-series Transformer models.

    This class provides a framework for estimating the Floating Point Operations
    required for both the prefill and decode phases of a Qwen-style language model.
    The calculations are based on a standard Transformer architecture featuring
    Grouped-Query Attention and a SwiGLU MLP block.

    The primary purpose is to aid in performance modeling, hardware requirement
    analysis, and understanding the computational cost of different model configurations
    and input sizes.

    Note:
    - Embedding lookups are considered memory-bound and are assigned 0 FLOPs.

    Attributes:
    - L: The number of transformer layers in the model.
    - H: The hidden size (embedding dimension).
    - H\_kv: The hidden size for Key and Value heads in GQA. If using MHA,
        H_kv will be equal to H. If using MQA, H_kv = H / num_heads.
    - I: The intermediate size of the MLP feed-forward network.
    - V: The vocabulary size.

    **FLOPs Calculation Breakdown:**

    The total FLOPs are calculated by summing the contributions of each major
    component within the transformer architecture, scaled by the number of layers.

    Variable Definitions:
    - B: Batch size
    - S: Sequence length
    - L: Number of layers
    - H: Hidden size
    - H_kv: KV hidden size
    - I: MLP intermediate size
    - V: Vocabulary size

    **Attention Block**:
    -   **QKV & Output Projections**: This involves matrix multiplications for
        generating Query, Key, Value, and the final output projection.
        - Q proj: `2 * B * S * H * H`
        - K, V proj: `2 * (2 * B * S * H * H_kv)`
        - O proj: `2 * B * S * H * H`
        - Total: `4*B*S*H^2 + 4*B*S*H*H_kv`
    -   **Attention Score Calculation**: The computation of `Q @ K^T`.
        This is approximated as `4 * B * S^2 * H` to account for both the
        GEMM and subsequent softmax operations.

    **MLP Block (SwiGLU)**:
    -   The structure is `down_proj(SiLU(gate_proj(x)) * up_proj(x))`.
    -   `gate_proj` and `up_proj` are `(B,S,H) x (H,I)` GEMMs.
    -   `down_proj` is an `(B,S,I) x (I,H)` GEMM.
    -   Total: `2*B*S*H*I (gate) + 2*B*S*H*I (up) + 2*B*S*I*H (down) = 6*B*S*H*I`

    **RMSNorm**:
    -   Involves calculating the mean square root of a vector and normalizing.
    -   Approximated as `4 * B * S * H` FLOPs per application. Each layer has
        two RMSNorms (before Attention and MLP).
    -   Total per layer: `8 * B * S * H`

    **LM Head**:
    -   A final projection from the hidden state to the vocabulary logits.
    -   This is an `(B,S,H) x (H,V)` GEMM.
    -   Total: `2 * B * S * H * V`

    **Final Aggregated Formulas:**

    -   **Prefill Phase (`S = S_prompt`)**:
        All tokens in the prompt are processed in parallel.
        ```
        FLOPS_prefill = L * (4*B*S_prompt*H*(H + H_kv + S_prompt) + 6*B*S_prompt*H*I + 8*B*S_prompt*H)
                        + 2*B*S_prompt*H*V
        ```

    -   **Decode Step (`S = 1`, uses KV Cache of length `S_cache`)**:
        A single new token is generated, attending to all previous `S_cache` tokens.
        The main difference is in the Attention block, where GEMMs are smaller.
        - QKV Proj (for the new token): `4*B*H*(H + H_kv)`
        - Attention Score (new Q vs all cached K): `4*B*H*S_cache`
        ```
        FLOPS_decode_step = L * (4*B*H*(H + H_kv) + 4*B*H*S_cache + 6*B*H*I + 8*B*H)
                            + 2*B*H*V

        Simplified:
        FLOPS_decode_step = 4*B*L*H*(H + H_kv) + 6*B*L*H*I + 8*B*L*H + 2*B*H*V + 4*B*L*H*S_cache
        ```
    """

    @staticmethod
    def lmhead_flops(hidden_size, vocab_size, seq_length: TensorOrScalar):
        """Calculate language model head FLOPs, seq_length supports tensor or scalar input"""
        return 2 * hidden_size * vocab_size * seq_length

    @staticmethod
    def qkv_project_flops(
        hidden_size, num_attn_heads, num_kv_heads, seq_length: TensorOrScalar
    ):
        """Calculate QKV projection FLOPs, seq_length supports tensor or scalar input"""
        hidden_size_kv = hidden_size // (num_attn_heads // num_kv_heads)
        return 2 * seq_length * hidden_size * (hidden_size + 2 * hidden_size_kv)

    @staticmethod
    def wo_projection_flops(hidden_size, seq_length: TensorOrScalar):
        """Calculate output projection FLOPs, seq_length supports tensor or scalar input"""
        return 2 * seq_length * hidden_size * hidden_size

    @staticmethod
    def attention_score_flops(hidden_size, seq_length: TensorOrScalar):
        """Calculate attention score FLOPs, seq_length supports tensor or scalar input"""
        return 4 * hidden_size * seq_length * seq_length

    @staticmethod
    def mlp_flops(hidden_size, mlp_intermediate_size, seq_length: TensorOrScalar):
        """Calculate MLP FLOPs, seq_length supports tensor or scalar input"""
        return 6 * seq_length * hidden_size * mlp_intermediate_size

    @staticmethod
    def rmsnorm_flops(hidden_size, seq_length: TensorOrScalar):
        """Calculate RMSNorm FLOPs, seq_length supports tensor or scalar inputs"""
        return 4 * seq_length * hidden_size

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def flops_generate(
        self, prompt_length: TensorOrScalar, decode_length: TensorOrScalar
    ):
        """Generation phase FLOPs calculation, prompt_length and decode_length support tensor or scalar input"""
        prefill_decode_flops = self._calculate_prefill_flops(
            prompt_length=prompt_length
        ) + self._calculate_decode_flops(
            prompt_length=prompt_length, decode_length=decode_length
        )

        return prefill_decode_flops

    def flops_inference(self, seq_length: TensorOrScalar):
        """Inference phase FLOPs calculation, seq_length supports tensor or scalar input"""
        prefill_total_flops = self._calculate_prefill_flops(prompt_length=seq_length)

        return prefill_total_flops

    def _calculate_prefill_flops(self, prompt_length: TensorOrScalar):
        """Prefill phase FLOPs calculation, prompt_length supports tensor or scalar input"""
        L = self.model_config.num_layers
        H = self.model_config.hidden_size
        n_h = self.model_config.num_attention_heads
        n_qg = self.model_config.num_query_groups
        n_kv = n_h // n_qg
        I = self.model_config.ffn_hidden_size
        V = self.model_config.override_vocab_size

        qkv = FLOPSCalculator.qkv_project_flops(H, n_h, n_kv, prompt_length)
        attn = FLOPSCalculator.attention_score_flops(H, prompt_length)
        wo = FLOPSCalculator.wo_projection_flops(H, prompt_length)
        mlp_part = FLOPSCalculator.mlp_flops(H, I, prompt_length)
        norms = 2 * FLOPSCalculator.rmsnorm_flops(H, prompt_length)
        final_norm = FLOPSCalculator.rmsnorm_flops(H, prompt_length)
        lm_head = FLOPSCalculator.lmhead_flops(H, V, prompt_length)

        prefill_flops = L * (qkv + attn + wo + mlp_part + norms) + final_norm + lm_head

        return prefill_flops

    def _calculate_decode_flops(
        self, prompt_length: TensorOrScalar, decode_length: TensorOrScalar
    ):
        """Decode phase FLOPs calculation, prompt_length and decode_length support tensor or scalar input"""
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

        qkv = FLOPSCalculator.qkv_project_flops(H, n_h, n_kv, 1)
        wo = FLOPSCalculator.wo_projection_flops(H, 1)
        mlp = FLOPSCalculator.mlp_flops(H, I, 1)
        norms = 2 * FLOPSCalculator.rmsnorm_flops(H, 1)
        lm_head = FLOPSCalculator.lmhead_flops(H, V, 1)

        # Fixed FLOPs per decode step multiplied by decode_length, plus attention computation
        decode_flops = (
            decode_length * (L * (qkv + wo + mlp + norms) + lm_head)
            + 4 * L * H * (decode_length + 2 * prompt_length) * decode_length / 2
        )

        return decode_flops
