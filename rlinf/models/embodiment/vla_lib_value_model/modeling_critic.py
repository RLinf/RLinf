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

"""
ValueCriticModel for RLinf.

Copied from vla_lib with minimal modifications.
Supports three forward modes:
1. VLM mode: Predict "Value: X" tokens with CE loss
2. Expert mode: Gemma expert + [CLS] -> continuous/distributional value
3. Dual mode: Both VLM + expert objectives
"""

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.modeling_outputs import ModelOutput

# Import base classes from vla_lib
from vla_lib.models.vlas.openpi.configs import get_config
from vla_lib.models.vlas.openpi.modeling_pi0 import make_att_2d_masks
from vla_lib.models.vlas.openpi05.configuration_pi05 import PI05Config
from vla_lib.models.vlas.openpi05.modeling_pi05 import PI05FlowMatching
from vla_lib.models.vlas.openpi05.paligemma_with_multi_expert import (
    PaliGemmaWithMultiExpertModel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Value Head
# =============================================================================


class ValueHead(nn.Module):
    """Value prediction head with learnable CLS embedding and projection."""

    def __init__(
        self,
        hidden_size: int,
        num_bins: int,
        loss_type: str,
        v_min: float,
        v_max: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.loss_type = loss_type

        # Learnable CLS embedding
        self.cls_embedding = nn.Embedding(1, hidden_size)
        nn.init.normal_(self.cls_embedding.weight, std=0.02)

        # Value projection
        if loss_type == "mse":
            self.value_proj = nn.Linear(hidden_size, 1)
            self.register_buffer("atoms", None, persistent=False)
            self.num_bins = None
        else:
            self.value_proj = nn.Linear(hidden_size, num_bins)
            self.register_buffer(
                "atoms", torch.linspace(v_min, v_max, num_bins), persistent=False
            )
            self.num_bins = num_bins
            self.v_min = v_min
            self.v_max = v_max
            self.delta_z = (v_max - v_min) / (num_bins - 1)

    def get_cls_embedding(self, batch_size: int) -> Tensor:
        """Get CLS embedding expanded to batch size. Returns [B, 1, hidden_size]."""
        # Use embedding lookup instead of reading `.weight` directly.
        # Under FSDP, `.weight` can be a view into a flattened parameter and may
        # trigger autograd view/inplace conflicts during backward.
        cls_token_ids = torch.zeros(
            batch_size, 1, dtype=torch.long, device=self.cls_embedding.weight.device
        )
        return self.cls_embedding(cls_token_ids)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states to value logits."""
        # Ensure dtype matches for FSDP mixed precision compatibility
        hidden_states = hidden_states.to(self.value_proj.weight.dtype)
        return self.value_proj(hidden_states)


# =============================================================================
# Output
# =============================================================================


@dataclass
class CriticOutput(ModelOutput):
    """Output for critic models."""

    loss: Optional[torch.FloatTensor] = None
    predicted_values: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    probs: Optional[torch.FloatTensor] = None
    atoms: Optional[torch.FloatTensor] = None
    expert_loss: Optional[torch.FloatTensor] = None
    language_loss: Optional[torch.FloatTensor] = None
    language_logits: Optional[torch.FloatTensor] = None
    language_token_acc: Optional[torch.FloatTensor] = None
    language_loss_mask: Optional[torch.BoolTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    # Categorical loss metrics
    cat_acc_best: Optional[torch.FloatTensor] = None  # accuracy hitting best target bin
    cat_acc_neighbor: Optional[torch.FloatTensor] = None  # accuracy hitting l or u
    cat_mae: Optional[torch.FloatTensor] = None  # MAE in value space


# =============================================================================
# Config
# =============================================================================


class PI05CriticConfig(PI05Config):
    """Configuration for PI05 Critic models."""

    def __init__(
        self,
        critic_expert_variant: str = "gemma_100m",
        critic_forward_mode: Literal["vlm", "expert", "dual"] = "expert",
        expert_loss_type: Literal["mse", "categorical", "distributional"] = "mse",
        num_bins: int = 201,
        v_min: float = -1.0,
        v_max: float = 0.0,
        vlm_loss_weight: float = 1.0,
        expert_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.critic_expert_variant = critic_expert_variant
        self.critic_forward_mode = critic_forward_mode
        self.expert_loss_type = expert_loss_type
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.vlm_loss_weight = vlm_loss_weight
        self.expert_loss_weight = expert_loss_weight


# =============================================================================
# ValueCriticModel
# =============================================================================


class ValueCriticModel(PI05FlowMatching):
    """Value function V(s) inheriting PI05 infrastructure.

    Supports:
    - VLM mode: token prediction via inherited _forward_vlm
    - Expert mode: Gemma expert + [CLS] -> value prediction
    - Dual mode: both VLM + expert objectives
    """

    def __init__(self, config: PI05CriticConfig):
        nn.Module.__init__(self)
        self.config = config
        self.pi05 = True
        self.critic_forward_mode = getattr(config, "critic_forward_mode", "expert")
        self.expert_loss_type = getattr(config, "expert_loss_type", "mse")

        paligemma_config = get_config(config.paligemma_variant)
        expert_config = get_config(config.critic_expert_variant)

        logger.info(
            f"Creating ValueCritic: expert={config.critic_expert_variant}, "
            f"mode={self.critic_forward_mode}, loss_type={self.expert_loss_type}"
        )

        # Build expert configs based on forward mode
        if self.critic_forward_mode == "vlm":
            expert_configs = {}
            use_adarms = [False, {}]
            logger.info("  VLM-only mode: no value expert created")
        else:
            expert_configs = {"value": expert_config}
            use_adarms = [False, {"value": False}]
            logger.info(
                f"  Expert mode: creating 'value' expert with {config.critic_expert_variant}"
            )
        self.paligemma_with_expert = PaliGemmaWithMultiExpertModel(
            vlm_config=paligemma_config,
            expert_configs=expert_configs,
            use_adarms=use_adarms,
            precision=config.dtype,
            freeze_vision_encoder=getattr(config, "freeze_vision_encoder", False),
            freeze_vlm=getattr(config, "freeze_vlm", False),
        )

        self.gradient_checkpointing_enabled = False
        self._expert_config = expert_config

        # Value head (only for expert/dual modes)
        if self.critic_forward_mode != "vlm":
            self.expert_width = expert_config.width
            self.value_head = ValueHead(
                hidden_size=expert_config.width,
                num_bins=config.num_bins,
                loss_type=self.expert_loss_type,
                v_min=config.v_min,
                v_max=config.v_max,
            )
            if self.expert_loss_type != "mse":
                self.num_bins = config.num_bins
                self.v_min = config.v_min
                self.v_max = config.v_max
                self.delta_z = (config.v_max - config.v_min) / (config.num_bins - 1)
        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    @property
    def _no_split_modules(self) -> list[str]:
        if self.paligemma_with_expert.freeze_vlm:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
                "ValueHead",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
                # TODO: This is because ValueCriticModel uses a different value head class from RL, and current FSDP cannot wrap it. Fix this later.
                "ValueHead",
            ]
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "lm_head",
            "cls_embedding",
        ]

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            True
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        for expert in self.paligemma_with_expert.experts.values():
            expert.model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for ValueCritic")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            False
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        for expert in self.paligemma_with_expert.experts.values():
            expert.model.gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for ValueCritic")

    def get_cls_embedding(self, batch_size: int) -> Tensor:
        """Get CLS embedding from ValueHead."""
        return self.value_head.get_cls_embedding(batch_size)

    def embed_suffix(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """Create suffix with [CLS] token for value prediction."""
        cls_emb = self.get_cls_embedding(batch_size)
        pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=cls_emb.device)
        ar_mask = torch.ones(batch_size, 1, dtype=torch.long, device=cls_emb.device)
        return cls_emb, pad_mask, ar_mask

    def forward(
        self, observation, target_values=None, target_distribution=None, **kwargs
    ) -> CriticOutput:
        """Forward pass with auto mode detection.

        Args:
            observation: Observation dict from data collator
            target_values: Target values [B] for mse/categorical loss
            target_distribution: Target probability distribution [B, num_bins]
        """
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            token_ar_mask,
            token_loss_mask,
            _,
        ) = self._preprocess_observation(observation)

        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        # VLM-only mode
        if self.critic_forward_mode == "vlm":
            result = self._forward_vlm(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                token_ar_mask,
                token_loss_mask,
            )
            return CriticOutput(
                loss=result.loss,
                language_loss=result.language_loss,
                language_logits=result.logits,
                language_token_acc=result.language_token_acc,
                language_loss_mask=result.language_loss_mask,
            )

        # Expert or Dual mode
        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(batch_size)

        stop_gradient = getattr(self.config, "stop_gradient_to_vlm", False)
        values, hidden_states, logits, probs, prefix_out = self._forward_expert(
            prefix_embs,
            prefix_pad_masks,
            prefix_ar_masks,
            suffix_embs,
            suffix_pad_masks,
            suffix_ar_masks,
            stop_gradient_to_vlm=stop_gradient,
        )

        # Compute losses
        expert_loss = None
        cat_metrics = None
        if target_values is not None or target_distribution is not None:
            expert_loss, cat_metrics = self._compute_expert_loss(
                values, logits, target_values, target_distribution
            )

        language_loss, language_acc, language_logits = None, None, None
        if self.critic_forward_mode == "dual" and token_loss_mask.any():
            language_loss, language_acc, language_logits = self._compute_ce_loss(
                prefix_out[:, :-1],
                prefix_pad_masks,
                lang_tokens,
                token_loss_mask,
                truncated_input=False,
            )

        # Combine losses
        total_loss = None
        if expert_loss is not None or language_loss is not None:
            total_loss = torch.zeros(batch_size, device=device)
            if expert_loss is not None:
                total_loss = total_loss + self.config.expert_loss_weight * expert_loss
            if language_loss is not None:
                total_loss = total_loss + self.config.vlm_loss_weight * language_loss
            total_loss = total_loss.mean()

        return CriticOutput(
            loss=total_loss,
            predicted_values=values,
            logits=logits,
            probs=probs,
            atoms=self.value_head.atoms,
            expert_loss=expert_loss.mean() if expert_loss is not None else None,
            language_loss=language_loss,
            language_logits=language_logits,
            language_token_acc=language_acc,
            language_loss_mask=token_loss_mask[:, 1:]
            if self.critic_forward_mode == "dual"
            else None,
            hidden_states=hidden_states,
            cat_acc_best=cat_metrics["acc_best"] if cat_metrics else None,
            cat_acc_neighbor=cat_metrics["acc_neighbor"] if cat_metrics else None,
            cat_mae=cat_metrics["mae"] if cat_metrics else None,
        )

    def _forward_expert(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_ar_masks,
        suffix_embs,
        suffix_pad_masks,
        suffix_ar_masks,
        stop_gradient_to_vlm: bool = False,
    ):
        """Forward through VLM + value expert."""
        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        if model_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        # Interleaved path may hit view/inplace errors when VLM is frozen under
        # FSDP. Use robust two-stage forward in that case.
        if getattr(self.paligemma_with_expert, "freeze_vlm", False):
            prefix_out, suffix_out = self._forward_expert_two_stage(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_ar_masks=prefix_ar_masks,
                suffix_embs=suffix_embs,
                suffix_pad_masks=suffix_pad_masks,
                suffix_ar_masks=suffix_ar_masks,
                stop_gradient_to_vlm=stop_gradient_to_vlm,
            )
            cls_hidden = suffix_out[:, -1, :].to(
                self.value_head.value_proj.weight.dtype
            )
            values, logits, probs = self._compute_value_from_hidden(cls_hidden)
            return values, cls_hidden, logits, probs, prefix_out

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        ar_masks = torch.cat([prefix_ar_masks, suffix_ar_masks], dim=1)
        attn_mask = make_att_2d_masks(pad_masks, ar_masks)
        attn_mask_4d = self._prepare_attention_masks_4d(attn_mask)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(
            prefix_embs, suffix_embs, attn_mask_4d, position_ids, detach_kv
        ):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, None],
                expert_name="value",
                detach_prefix_for_suffix=detach_kv,
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            attn_mask_4d,
            position_ids,
            stop_gradient_to_vlm,
        )

        cls_hidden = suffix_out[:, -1, :].to(self.value_head.value_proj.weight.dtype)
        values, logits, probs = self._compute_value_from_hidden(cls_hidden)
        return values, cls_hidden, logits, probs, prefix_out

    def _forward_expert_two_stage(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_ar_masks,
        suffix_embs,
        suffix_pad_masks,
        suffix_ar_masks,
        stop_gradient_to_vlm: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Two-stage expert forward with KV cache."""
        # Phase 1: prefill frozen VLM to get KV cache.
        prefix_attn = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        prefix_attn_4d = self._prepare_attention_masks_4d(prefix_attn)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        (prefix_out, _), past_kv = self.paligemma_with_expert.forward(
            attention_mask=prefix_attn_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        if stop_gradient_to_vlm and past_kv is not None:
            past_kv = tuple(
                tuple(
                    t.detach() if isinstance(t, torch.Tensor) else t for t in layer_kv
                )
                for layer_kv in past_kv
            )

        # Phase 2: run value expert with cached prefix keys/values.
        batch_size = prefix_pad_masks.shape[0]
        prefix_len, suffix_len = prefix_pad_masks.shape[1], suffix_pad_masks.shape[1]
        prefix_2d = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_attn = make_att_2d_masks(suffix_pad_masks, suffix_ar_masks)
        full_attn_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_2d, suffix_attn], dim=2)
        )
        suffix_pos = (
            prefix_pad_masks.sum(dim=-1)[:, None]
            + torch.cumsum(suffix_pad_masks, dim=1)
            - 1
        )

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=full_attn_4d,
            position_ids=suffix_pos,
            past_key_values=past_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, None],
            expert_name="value",
        )
        return prefix_out, suffix_out

    def _compute_value_from_hidden(self, cls_hidden):
        """Compute value from [CLS] hidden state."""
        if self.expert_loss_type == "mse":
            return self.value_head(cls_hidden).squeeze(-1), None, None
        else:
            logits = self.value_head(cls_hidden)
            probs = F.softmax(logits, dim=-1)
            values = (probs * self.value_head.atoms).sum(dim=-1)
            return values, logits, probs

    def _compute_expert_loss(
        self, values, logits, target_values, target_distribution=None
    ):
        """Compute expert value loss.

        Returns:
            Tuple of (loss, cat_metrics) where cat_metrics is a dict with
            categorical metrics (acc_best, acc_neighbor, mae) or None for mse.
        """
        if self.expert_loss_type == "mse":
            return F.mse_loss(values, target_values, reduction="none"), None
        elif self.expert_loss_type == "categorical":
            return self._compute_categorical_loss(logits, target_values)
        else:
            if target_distribution is not None:
                loss = -(target_distribution * F.log_softmax(logits, dim=-1)).sum(
                    dim=-1
                )
                return loss, None
            return self._compute_categorical_loss(logits, target_values)

    def _compute_categorical_loss(self, logits, target_values):
        """Compute categorical loss (Dirac delta projection onto bins).

        Returns:
            Tuple of (loss, metrics_dict) where metrics_dict contains:
            - acc_best: accuracy of predicting the bin with highest target probability
            - acc_neighbor: accuracy of predicting either l or u bin
            - mae: mean absolute error in value space
        """
        target_values = target_values.clamp(self.v_min, self.v_max)
        b = (target_values - self.v_min) / self.delta_z
        l = b.floor().long().clamp(0, self.num_bins - 1)
        u = b.ceil().long().clamp(0, self.num_bins - 1)

        d_to_l, d_to_u = b - l.float(), u.float() - b
        same_bin = l == u
        d_to_l = torch.where(same_bin, torch.zeros_like(d_to_l), d_to_l)
        d_to_u = torch.where(same_bin, torch.ones_like(d_to_u), d_to_u)

        batch_size = target_values.shape[0]
        target_probs = torch.zeros(
            batch_size, self.num_bins, device=target_values.device
        )
        batch_idx = torch.arange(batch_size, device=target_values.device)
        target_probs[batch_idx, l] += d_to_u
        target_probs[batch_idx, u] += d_to_l

        loss = -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)

        # Compute metrics
        pred_bin = logits.argmax(dim=-1)  # predicted bin index

        # 1. Accuracy metrics
        # Best target bin: l if d_to_u >= d_to_l (l gets more probability), else u
        best_target_bin = torch.where(d_to_u >= d_to_l, l, u)
        acc_best = (pred_bin == best_target_bin).float().mean()
        acc_neighbor = ((pred_bin == l) | (pred_bin == u)).float().mean()

        # 2. MAE: distance to nearest target bin * delta_z
        dist_to_l = (pred_bin - l).abs()
        dist_to_u = (pred_bin - u).abs()
        min_dist = torch.min(dist_to_l, dist_to_u).float()
        mae = (min_dist * self.delta_z).mean()

        metrics = {
            "acc_best": acc_best,
            "acc_neighbor": acc_neighbor,
            "mae": mae,
        }

        return loss, metrics

    @torch.no_grad()
    def predict(self, observation) -> CriticOutput:
        """Inference with KV cache."""
        if self.critic_forward_mode == "vlm":
            raise ValueError("predict() not supported for VLM mode")

        (images, img_masks, lang_tokens, lang_masks, token_ar_mask, _, _) = (
            self._preprocess_observation(observation)
        )
        batch_size = lang_tokens.shape[0]

        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(batch_size)

        # Phase 1: Prefill VLM
        prefix_attn = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        prefix_attn_4d = self._prepare_attention_masks_4d(prefix_attn)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_kv = self.paligemma_with_expert.forward(
            attention_mask=prefix_attn_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Phase 2: Expert with cache
        prefix_len, suffix_len = prefix_pad_masks.shape[1], suffix_pad_masks.shape[1]
        prefix_2d = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_attn = make_att_2d_masks(suffix_pad_masks, suffix_ar_masks)
        full_attn_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_2d, suffix_attn], dim=2)
        )
        suffix_pos = (
            prefix_pad_masks.sum(dim=-1)[:, None]
            + torch.cumsum(suffix_pad_masks, dim=1)
            - 1
        )

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=full_attn_4d,
            position_ids=suffix_pos,
            past_key_values=past_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, None],
            expert_name="value",
        )

        cls_hidden = suffix_out[:, -1, :].to(self.value_head.value_proj.weight.dtype)
        values, logits, probs = self._compute_value_from_hidden(cls_hidden)
        return CriticOutput(
            predicted_values=values,
            logits=logits,
            probs=probs,
            atoms=self.value_head.atoms,
            hidden_states=cls_hidden,
        )


__all__ = ["ValueHead", "CriticOutput", "PI05CriticConfig", "ValueCriticModel"]
