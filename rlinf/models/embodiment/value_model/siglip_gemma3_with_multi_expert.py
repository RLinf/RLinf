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

"""SiglipGemma3WithMultiExpert: SigLIP2 + Gemma3 270M backbone with independent experts.

Architecture differences from PaliGemmaWithMultiExpertModel (2B):
  - VLM backbone: SigLIP2-so400m (1152-dim) + Gemma3-270M (640-dim, head_dim=256)
  - Image projection: fresh nn.Linear(1152, 640) (not built-in like PaliGemma)
  - KV cache format: DynamicCache (Gemma3) instead of tuple (Gemma1/PaliGemma)
  - Gemma3 uses sliding-window attention (window=512); interleaved forward not supported
  - Expert: GemmaForCausalLM (Gemma1) with head_dim=256 to match Gemma3 KV head_dim

Forward modes (two-stage only; no interleaved):
  Mode A — inputs_embeds=[prefix, None], use_cache=True
    → Gemma3 processes prefix, returns DynamicCache
  Mode B — inputs_embeds=[None, suffix], past_key_values=DynamicCache
    → GemmaForCausalLM expert cross-attends to Gemma3 KV cache
  (Modes A+B together implement the "two-stage" forward used by ValueCriticModel)

Gradient flow:
  freeze_vlm=True  : Gemma3 params frozen; KV cache detached in _forward_expert_two_stage
  freeze_vlm=False : Gemma3 params trainable; no detach → gradients flow through KV into Gemma3
"""

import logging
from typing import Literal

import torch
from torch import nn
from transformers import Gemma3ForCausalLM, GemmaForCausalLM, SiglipVisionModel
from transformers.cache_utils import DynamicCache
from transformers.models.auto import CONFIG_MAPPING

from .paligemma_with_multi_expert import _requires_uniform_dtype

logger = logging.getLogger(__name__)


class SiglipGemma3WithMultiExpert(nn.Module):
    """SigLIP2 + Gemma3 270M VLM backbone with independent GemmaForCausalLM experts.

    Loads SigLIP2 and Gemma3 from separate pretrained paths. The image projection
    (1152→640) is freshly initialized and trained from scratch.

    Expert models are GemmaForCausalLM (Gemma1 architecture) with head_dim=256 to
    match Gemma3 270M's KV cache head dimension. They consume Gemma3's DynamicCache
    in Mode B forward.

    Args:
        expert_configs: Dict mapping expert names to their Config objects.
            Each config must have head_dim=256 to match Gemma3 KV cache.
        siglip_path: Path to pretrained SigLIP2 vision model.
        gemma3_path: Path to pretrained Gemma3 270M language model.
        freeze_vision_encoder: If True, freeze SigLIP2 parameters.
        freeze_vlm: If True, freeze Gemma3 parameters (train experts only).
        precision: Model dtype, "bfloat16" or "float32".
        trainable_experts: List of expert names to train. None = all trainable.
    """

    def __init__(
        self,
        expert_configs: dict,
        siglip_path: str,
        gemma3_path: str,
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = False,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        trainable_experts: list[str] | None = None,
    ):
        super().__init__()

        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_vlm = freeze_vlm
        self.expert_names = list(expert_configs.keys())
        self.trainable_experts = (
            trainable_experts if trainable_experts is not None else self.expert_names
        )

        logger.info(
            f"Creating SiglipGemma3WithMultiExpert: experts={self.expert_names}, "
            f"freeze_vision_encoder={freeze_vision_encoder}, freeze_vlm={freeze_vlm}"
        )

        # Load SigLIP2 vision encoder
        logger.info(f"  Loading SigLIP2 from {siglip_path}")
        self.vision_tower = SiglipVisionModel.from_pretrained(siglip_path)
        siglip_hidden = self.vision_tower.config.hidden_size  # 1152

        # Load Gemma3 270M language model
        logger.info(f"  Loading Gemma3 from {gemma3_path}")
        self.gemma3 = Gemma3ForCausalLM.from_pretrained(gemma3_path)
        gemma3_hidden = self.gemma3.config.hidden_size  # 640

        # Fresh image projection: SigLIP2 1152-dim → Gemma3 640-dim
        self.multi_modal_proj = nn.Linear(siglip_hidden, gemma3_hidden, bias=True)
        nn.init.normal_(self.multi_modal_proj.weight, std=0.02)
        nn.init.zeros_(self.multi_modal_proj.bias)
        logger.info(
            f"  Fresh projection: {siglip_hidden} → {gemma3_hidden} (randomly initialized)"
        )

        # Validate expert head_dim against Gemma3
        gemma3_head_dim = self.gemma3.config.head_dim
        for name, cfg in expert_configs.items():
            if cfg.head_dim != gemma3_head_dim:
                raise ValueError(
                    f"Expert '{name}' has head_dim={cfg.head_dim} but Gemma3 270M "
                    f"has head_dim={gemma3_head_dim}. They must match for KV cache "
                    f"cross-attention. Use a *_hd256 expert config variant."
                )

        # Create expert models (GemmaForCausalLM, Gemma1 architecture)
        self.experts = nn.ModuleDict()
        for name, expert_config in expert_configs.items():
            expert_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=expert_config.head_dim,
                hidden_size=expert_config.width,
                intermediate_size=expert_config.mlp_dim,
                num_attention_heads=expert_config.num_heads,
                num_hidden_layers=expert_config.depth,
                num_key_value_heads=expert_config.num_kv_heads,
                vocab_size=257152,
                hidden_activation="gelu_pytorch_tanh",
                torch_dtype="float32",
            )
            expert = GemmaForCausalLM(config=expert_config_hf)
            expert.model.embed_tokens = None
            self.experts[name] = expert
            logger.info(
                f"  Expert '{name}': width={expert_config.width}, "
                f"depth={expert_config.depth}, head_dim={expert_config.head_dim}"
            )

        self._apply_precision(precision)
        self._set_requires_grad()

    # =========================================================================
    # Embedding Methods (compatible with VLMObservationEncoder.embed_prefix)
    # =========================================================================

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed images: SigLIP2 → projection → [B, num_patches, 640].

        Args:
            image: [B, 3, H, W] image tensor.

        Returns:
            [B, 256, 640] image embeddings in Gemma3 hidden space.
        """
        feats = self.vision_tower(
            pixel_values=image
        ).last_hidden_state  # [B, 256, 1152]
        return self.multi_modal_proj(feats)  # [B, 256, 640]

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens using Gemma3's embedding table.

        Args:
            tokens: [B, L] token ids.

        Returns:
            [B, L, 640] token embeddings.
        """
        return self.gemma3.model.embed_tokens(tokens)

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list | None = None,
        use_cache: bool | None = None,
        expert_name: str | None = None,
        **kwargs,  # accept & ignore: adarms_cond, detach_prefix_for_suffix, etc.
    ):
        """Two-stage forward for SigLIP+Gemma3 backbone.

        Unlike PaliGemmaWithMultiExpertModel, interleaved forward is NOT supported
        because Gemma3's sliding-window attention is incompatible with layer-wise
        interleaving. Use _forward_expert_two_stage in ValueCriticModel instead.

        Args:
            attention_mask: [B, 1, Q, K] 4D attention mask.
            position_ids: [B, seq_len] position indices.
            past_key_values: DynamicCache from a prior Mode A call (for Mode B).
            inputs_embeds: [prefix_embs, suffix_embs] where one element may be None.
                - [prefix, None]: Mode A — Gemma3 prefix forward, returns DynamicCache
                - [None, suffix]: Mode B — expert forward with past DynamicCache
            use_cache: Whether to return KV cache (only used in Mode A).
            expert_name: Which expert to run in Mode B.
            **kwargs: Ignored keyword args (adarms_cond, etc.) for API compatibility.

        Returns:
            ([prefix_hidden, suffix_hidden], past_key_values)
            - Mode A: ([last_hidden_state, None], DynamicCache)
            - Mode B: ([None, last_hidden_state], None)
        """
        prefix_embs, suffix_embs = inputs_embeds

        # Mode A: prefix-only — Gemma3 processes prefix and caches KV.
        # NOTE: Gemma3TextModel only creates DynamicCache when `not self.training`
        # (transformers source: `if use_cache and past_key_values is None and not self.training`).
        # Fix: pre-create an empty DynamicCache so the condition is bypassed in training mode.
        if suffix_embs is None:
            if past_key_values is None:
                past_key_values = DynamicCache()
            out = self.gemma3.model(
                inputs_embeds=prefix_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            return [out.last_hidden_state, None], out.past_key_values

        # Mode B: suffix-only — expert cross-attends to Gemma3's DynamicCache
        if prefix_embs is None:
            expert_name = self._resolve_expert_name(expert_name)
            expert = self.experts[expert_name]
            out = expert.model(
                inputs_embeds=suffix_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,
            )
            return [None, out.last_hidden_state], None

        # Interleaved (both prefix and suffix) — not supported for Gemma3
        raise ValueError(
            "SiglipGemma3WithMultiExpert does not support interleaved forward "
            "(both prefix and suffix in a single call). Gemma3's sliding-window "
            "attention is incompatible with layer-wise interleaving. "
            "Use two-stage forward via _forward_expert_two_stage in ValueCriticModel "
            "(always active when backbone_variant='siglip_gemma3')."
        )

    # =========================================================================
    # Train / requires_grad
    # =========================================================================

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            self.vision_tower.eval()

        if self.freeze_vlm:
            for param in self.gemma3.parameters():
                param.requires_grad = False
            self.gemma3.eval()

        for name in self.expert_names:
            if name not in self.trainable_experts:
                for param in self.experts[name].parameters():
                    param.requires_grad = False
                self.experts[name].eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.vision_tower.eval()
        if self.freeze_vlm:
            self.gemma3.eval()
        for name in self.expert_names:
            if name not in self.trainable_experts:
                self.experts[name].eval()
        return self

    # =========================================================================
    # Precision
    # =========================================================================

    def _apply_precision(self, precision: Literal["bfloat16", "float32"]):
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid precision: {precision}")

        self.to(dtype=torch.bfloat16)

        if _requires_uniform_dtype():
            logger.info(
                "Parameter sharding detected (FSDP/Zero-3): using uniform bfloat16"
            )
            return

        logger.info(
            "Applying mixed precision: bf16 backbone + fp32 for selected layers"
        )
        params_to_keep_float32 = [
            "embeddings.patch_embedding.weight",
            "embeddings.patch_embedding.bias",
            "embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(sel in name for sel in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _resolve_expert_name(self, expert_name: str | None) -> str:
        if expert_name is not None:
            if expert_name not in self.expert_names:
                raise ValueError(
                    f"Unknown expert: {expert_name}. Available: {self.expert_names}"
                )
            return expert_name
        if len(self.expert_names) == 1:
            return self.expert_names[0]
        raise ValueError(
            f"expert_name must be specified when multiple experts exist: {self.expert_names}"
        )
