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

"""PaliGemma with multiple experts sharing the same VLM backbone.

PaliGemmaWithMultiExpertModel supports multiple named experts (e.g., action, value)
sharing one PaliGemma VLM backbone with interleaved attention and optional
knowledge insulation (stop gradient from expert to VLM).

Forward modes:
- Prefix only: Process VLM and optionally cache KV for later expert use
- Prefix + expert: Interleaved attention between VLM and a specific expert
- Expert only: Use cached KV from VLM, process only expert suffix
"""

import logging
import os
from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

logger = logging.getLogger(__name__)


def check_transformers_replace():
    """Check if AdaRMS support is present in transformers. Raises error if not."""
    from transformers.models.gemma import modeling_gemma

    if not hasattr(modeling_gemma, "_gated_residual"):
        raise RuntimeError(
            "\n" + "=" * 70 + "\n"
            "ERROR: transformers_replace is NOT installed!\n"
            "=" * 70 + "\n\n"
            "PI0.5 requires custom AdaRMS modifications in transformers.\n\n"
            "Run this command from the project root:\n\n"
            "    cp -r ./rlinf/models/embodiment/value_model/openpi/transformers_replace/* \\\n"
            "        $(python -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/\n\n"
            + "="
            * 70
        )


# Verify custom transformers modifications are installed
check_transformers_replace()


def _requires_uniform_dtype() -> bool:
    """Check if the distributed training method requires uniform parameter dtype.

    Only parameter-sharding methods (FSDP, DeepSpeed Zero-3) require uniform dtype.
    DDP and DeepSpeed Zero-1/2 replicate parameters and can use mixed dtypes.

    Detection:
    - FSDP: FSDP_USE_ORIG_PARAMS or ACCELERATE_USE_FSDP env vars
    - DeepSpeed Zero-3: ACCELERATE_DEEPSPEED_ZERO_STAGE=3
    """
    # FSDP detection
    if os.environ.get("ACCELERATE_USE_FSDP", "").lower() in ("1", "true"):
        return True
    if os.environ.get("FSDP_USE_ORIG_PARAMS") is not None:
        return True

    # DeepSpeed Zero-3 detection
    zero_stage = os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "")
    if zero_stage == "3":
        return True

    return False


class PaliGemmaWithMultiExpertModel(nn.Module):
    """PaliGemma with multiple Gemma experts sharing the VLM backbone.

    Args:
        vlm_config: Config for the PaliGemma VLM backbone
        expert_configs: Dict mapping expert names to their configs
            e.g., {"action": action_config, "critic": critic_config}
        use_adarms: Dict mapping expert names to their adarms settings
            e.g., {"action": True, "critic": False}
            VLM always uses adarms=False
        precision: Model precision ("bfloat16" or "float32")
        freeze_vision_encoder: If True, freeze vision encoder parameters
        freeze_vlm: If True, freeze entire VLM (train experts only)
        trainable_experts: List of expert names to train. If None, all experts are trainable.
    """

    def __init__(
        self,
        vlm_config,
        expert_configs: dict,
        use_adarms: list | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = False,
        trainable_experts: list[str] | None = None,
    ):
        super().__init__()

        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_vlm = freeze_vlm
        self.expert_names = list(expert_configs.keys())
        self.trainable_experts = (
            trainable_experts if trainable_experts is not None else self.expert_names
        )

        logger.info(f"Creating multi-expert model with experts: {self.expert_names}")
        logger.info(f"Trainable experts: {self.trainable_experts}")
        logger.info(
            f"Freeze vision encoder: {freeze_vision_encoder}, Freeze VLM: {freeze_vlm}"
        )

        if use_adarms is None:
            use_adarms = [False, dict.fromkeys(self.expert_names, False)]

        # Create PaliGemma VLM
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = (
            vlm_config.width if use_adarms[0] else None
        )
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)

        # Create multiple experts
        self.experts = nn.ModuleDict()
        for name, expert_config in expert_configs.items():
            expert_use_adarms = use_adarms[1].get(name, False)
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
                use_adarms=expert_use_adarms,
                adarms_cond_dim=expert_config.width if expert_use_adarms else None,
            )
            expert = GemmaForCausalLM(config=expert_config_hf)
            expert.model.embed_tokens = None
            self.experts[name] = expert

        self.to_bfloat16_for_selected_params(precision)
        self.set_requires_grad()

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        if _requires_uniform_dtype():
            logger.info(
                "Parameter sharding detected (FSDP/Zero-3): using uniform bfloat16 dtype"
            )
            return

        logger.info(
            "Applying mixed precision: bf16 backbone + fp32 for selected layers"
        )
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list | tuple[tuple[torch.FloatTensor, ...], ...] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        detach_prefix_for_suffix: bool = False,
        expert_name: str | None = None,
    ):
        """Forward pass through VLM and optionally one expert.

        Args:
            inputs_embeds: [prefix_embeds, suffix_embeds] where:
                - prefix_embeds: VLM embeddings (images + language)
                - suffix_embeds: Expert suffix embeddings (or None for VLM-only)
            adarms_cond: [vlm_cond, expert_cond] conditioning tensors
            detach_prefix_for_suffix: If True, stop gradient from expert to VLM
            expert_name: Which expert to use for suffix. Required if suffix_embeds is not None.

        Returns:
            ([prefix_output, suffix_output], past_key_values)
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # VLM-only mode
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
            return [prefix_output, suffix_output], prefix_past_key_values

        # Resolve expert_name with backwards compatibility
        expert_name = self._resolve_expert_name(expert_name)

        # Expert-only mode (with cached KV from VLM)
        if inputs_embeds[0] is None:
            expert = self.experts[expert_name]
            suffix_output = expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1],
            )
            prefix_output = None
            suffix_past_key_values = suffix_output.past_key_values
            suffix_output = suffix_output.last_hidden_state
            return [prefix_output, suffix_output], suffix_past_key_values

        # Interleaved VLM + expert mode
        expert = self.experts[expert_name]

        return self._forward_interleaved(
            inputs_embeds,
            attention_mask,
            position_ids,
            adarms_cond,
            detach_prefix_for_suffix,
            expert,
        )

    def _forward_interleaved(
        self,
        inputs_embeds: list[torch.FloatTensor],
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        adarms_cond: list[torch.Tensor],
        detach_prefix_for_suffix: bool,
        expert: GemmaForCausalLM,
    ):
        """Interleaved forward through VLM and one expert."""
        models = [self.paligemma.language_model, expert.model]
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        use_gradient_checkpointing = (
            hasattr(expert.model, "gradient_checkpointing")
            and expert.model.gradient_checkpointing
            and self.training
        ) or (
            hasattr(self, "gradient_checkpointing")
            and self.gradient_checkpointing
            and self.training
        )

        if self.training and hasattr(expert.model, "gradient_checkpointing"):
            if not expert.model.gradient_checkpointing:
                expert.model.gradient_checkpointing = True
            use_gradient_checkpointing = True

        def compute_layer_complete(
            layer_idx,
            inputs_embeds,
            attention_mask,
            position_ids,
            adarms_cond,
            detach_prefix_kv,
        ):
            # inputs_embeds: list of 2 tensors
            #   [0] prefix (VLM): [B, prefix_len, hidden_dim]
            #   [1] suffix (action expert): [B, suffix_len, hidden_dim]
            # attention_mask: [B, num_heads, total_len, total_len] where total_len = prefix_len + suffix_len
            query_states = []
            key_states = []
            value_states = []
            gates = []

            for i, hidden_states in enumerate(inputs_embeds):
                # hidden_states: [B, seq_len, hidden_dim] where seq_len is prefix_len or suffix_len
                layer = models[i].layers[layer_idx]
                hidden_states, gate = layer.input_layernorm(
                    hidden_states, cond=adarms_cond[i]
                )
                gates.append(gate)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                # After view: [B, seq_len, num_heads, head_dim]
                # After transpose(1,2): [B, num_heads, seq_len, head_dim]
                query_state = (
                    layer.self_attn.q_proj(hidden_states)
                    .view(hidden_shape)
                    .transpose(1, 2)
                )
                key_state = (
                    layer.self_attn.k_proj(hidden_states)
                    .view(hidden_shape)
                    .transpose(1, 2)
                )
                value_state = (
                    layer.self_attn.v_proj(hidden_states)
                    .view(hidden_shape)
                    .transpose(1, 2)
                )

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # After loop:
            # query_states[0]: [B, num_heads, prefix_len, head_dim]
            # query_states[1]: [B, num_heads, suffix_len, head_dim]
            prefix_query, _suffix_query = query_states[0], query_states[1]
            _prefix_key, _suffix_key = key_states[0], key_states[1]
            prefix_value, suffix_value = value_states[0], value_states[1]
            prefix_len = prefix_query.shape[2]

            # Concatenate on dim=2 (sequence dim) for rotary embedding
            # query_states_cat: [B, num_heads, prefix_len + suffix_len, head_dim]
            query_states_cat = torch.cat(query_states, dim=2)
            key_states_cat = torch.cat(key_states, dim=2)

            dummy_tensor = torch.zeros(
                query_states_cat.shape[0],
                query_states_cat.shape[2],
                query_states_cat.shape[-1],
                device=query_states_cat.device,
                dtype=query_states_cat.dtype,
            )
            cos, sin = self.paligemma.model.language_model.rotary_emb(
                dummy_tensor, position_ids
            )
            # Apply rotary: shape unchanged [B, num_heads, total_len, head_dim]
            query_states_cat, key_states_cat = modeling_gemma.apply_rotary_pos_emb(
                query_states_cat, key_states_cat, cos, sin, unsqueeze_dim=1
            )

            # Split back after rotary embedding (for knowledge insulation path)
            # prefix_query_rot: [B, num_heads, prefix_len, head_dim]
            # suffix_query_rot: [B, num_heads, suffix_len, head_dim]
            prefix_query_rot = query_states_cat[:, :, :prefix_len, :]
            suffix_query_rot = query_states_cat[:, :, prefix_len:, :]
            prefix_key_rot = key_states_cat[:, :, :prefix_len, :]
            suffix_key_rot = key_states_cat[:, :, prefix_len:, :]

            batch_size = query_states_cat.shape[0]
            scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling
            head_dim = self.paligemma.language_model.layers[
                layer_idx
            ].self_attn.head_dim

            if detach_prefix_kv:
                # Knowledge insulation: stop gradient only for cross-attention path
                # Following Pi0 paper Eq. (5) and (6):
                # - P_bb (backbone self-attn): FULL gradients to Q_b, K_b, V_b
                # - P_ab (action attending to backbone): sg on K_b, V_b only
                # - P_aa (action self-attn): FULL gradients to Q_a, K_a, V_a

                # 1. Backbone self-attention: P_bb @ V_b (full gradients)
                # prefix_mask: [B, num_heads, prefix_len, prefix_len]
                prefix_mask = attention_mask[:, :, :prefix_len, :prefix_len]
                # Input Q,K,V: [B, num_heads, prefix_len, head_dim]
                # Output: [B, prefix_len, num_heads * head_dim]
                prefix_att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    prefix_query_rot,
                    prefix_key_rot,
                    prefix_value,
                    prefix_mask,
                    scaling,
                )

                # 2. Action attention: P_ab @ sg(V_b) + P_aa @ V_a
                # Keys for suffix: [sg(prefix_key), suffix_key]
                # suffix_keys_combined: [B, num_heads, prefix_len + suffix_len, head_dim]
                suffix_keys_combined = torch.cat(
                    [prefix_key_rot.detach(), suffix_key_rot], dim=2
                )
                # suffix_values_combined: [B, num_heads, prefix_len + suffix_len, head_dim]
                suffix_values_combined = torch.cat(
                    [prefix_value.detach(), suffix_value], dim=2
                )
                # suffix_mask: [B, num_heads, suffix_len, total_len]
                suffix_mask = attention_mask[:, :, prefix_len:, :]
                # Input Q: [B, num_heads, suffix_len, head_dim]
                # Input K,V: [B, num_heads, total_len, head_dim]
                # Output: [B, suffix_len, num_heads * head_dim]
                suffix_att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    suffix_query_rot,
                    suffix_keys_combined,
                    suffix_values_combined,
                    suffix_mask,
                    scaling,
                )

                # Concatenate on dim=1 (sequence dimension after attention transpose)
                # prefix_att_output: [B, prefix_len, num_heads * head_dim]
                # suffix_att_output: [B, suffix_len, num_heads * head_dim]
                # att_output: [B, prefix_len + suffix_len, num_heads * head_dim]
                att_output = torch.cat([prefix_att_output, suffix_att_output], dim=1)
            else:
                # Standard unified attention (no stop gradient)
                # value_states_cat: [B, num_heads, total_len, head_dim]
                value_states_cat = torch.cat([prefix_value, suffix_value], dim=2)
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states_cat,
                    key_states_cat,
                    value_states_cat,
                    attention_mask,
                    scaling,
                )

            # Reshape for o_proj: [B, total_len, 8 * head_dim]
            att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

            outputs_embeds = []
            start_pos = 0
            for i, hidden_states in enumerate(inputs_embeds):
                # hidden_states: [B, seq_len, hidden_dim] (original input before layernorm)
                layer = models[i].layers[layer_idx]
                end_pos = start_pos + hidden_states.shape[1]

                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                # att_output[:, start_pos:end_pos]: [B, seq_len, num_heads * head_dim]
                # out_emb after o_proj: [B, seq_len, hidden_dim]
                out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                # Residual: out_emb = hidden_states + out_emb * gate
                out_emb = modeling_gemma._gated_residual(
                    hidden_states, out_emb, gates[i]
                )
                after_first_residual = out_emb.clone()
                out_emb, gate = layer.post_attention_layernorm(
                    out_emb, cond=adarms_cond[i]
                )
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)

                out_emb = layer.mlp(out_emb)
                out_emb = modeling_gemma._gated_residual(
                    after_first_residual, out_emb, gate
                )
                # out_emb: [B, seq_len, hidden_dim]
                outputs_embeds.append(out_emb)
                start_pos = end_pos

            # outputs_embeds: [[B, prefix_len, hidden], [B, suffix_len, hidden]]
            return outputs_embeds

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    detach_prefix_for_suffix,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                inputs_embeds = compute_layer_complete(
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    detach_prefix_for_suffix,
                )

        def compute_final_norms(inputs_embeds, adarms_cond):
            outputs_embeds = []
            for i, hidden_states in enumerate(inputs_embeds):
                out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                outputs_embeds.append(out_emb)
            return outputs_embeds

        if use_gradient_checkpointing:
            outputs_embeds = torch.utils.checkpoint.checkpoint(
                compute_final_norms,
                inputs_embeds,
                adarms_cond,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

        prefix_output = outputs_embeds[0]
        suffix_output = outputs_embeds[1]
        prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values

    def set_requires_grad(self):
        """Set requires_grad for parameters based on freezing configuration."""
        if self.freeze_vision_encoder:
            for param in self.paligemma.vision_tower.parameters():
                param.requires_grad = False
            self.paligemma.vision_tower.eval()

        if self.freeze_vlm:
            for param in self.paligemma.parameters():
                param.requires_grad = False
            self.paligemma.eval()

        for name in self.expert_names:
            if name not in self.trainable_experts:
                for param in self.experts[name].parameters():
                    param.requires_grad = False
                self.experts[name].eval()

    def train(self, mode: bool = True):
        """Override train method to respect freezing configuration."""
        super().train(mode)

        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.freeze_vlm:
            self.paligemma.eval()

        for name in self.expert_names:
            if name not in self.trainable_experts:
                self.experts[name].eval()

    def get_expert(self, name: str) -> GemmaForCausalLM:
        """Get a specific expert by name."""
        return self.experts[name]

    def _resolve_expert_name(self, expert_name: str | None) -> str:
        """Resolve expert name with backwards compatibility.

        If expert_name is None:
        - Returns "action" if it exists (backwards compat with PaliGemmaWithExpertModel)
        - Returns the only expert if there's exactly one
        - Raises error otherwise
        """
        if expert_name is not None:
            if expert_name not in self.expert_names:
                raise ValueError(
                    f"Unknown expert: {expert_name}. Available: {self.expert_names}"
                )
            return expert_name

        if "action" in self.expert_names:
            return "action"
        if len(self.expert_names) == 1:
            return self.expert_names[0]
        raise ValueError(
            f"expert_name must be specified when multiple experts exist: {self.expert_names}"
        )

    # =========================================================================
    # Backwards Compatibility with PaliGemmaWithExpertModel
    # =========================================================================

    @property
    def gemma_expert(self) -> GemmaForCausalLM | None:
        """Backwards compatibility: returns 'action' expert if it exists."""
        return self.experts.get("action")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict with backwards compatibility for PaliGemmaWithExpertModel.

        Remaps 'gemma_expert.*' keys to 'experts.action.*' if 'action' expert exists.
        Ignores 'gemma_expert.*' keys if no 'action' expert is present.
        """
        remapped_state_dict = {}
        ignored_keys = []

        for key, value in state_dict.items():
            if key.startswith("gemma_expert."):
                if "action" in self.expert_names:
                    new_key = key.replace("gemma_expert.", "experts.action.", 1)
                    remapped_state_dict[new_key] = value
                    logger.debug(f"Remapped key: {key} -> {new_key}")
                else:
                    ignored_keys.append(key)
            else:
                remapped_state_dict[key] = value

        if ignored_keys:
            logger.info(
                f"Ignored {len(ignored_keys)} 'gemma_expert.*' keys from checkpoint "
                f"(no 'action' expert in model)"
            )

        result = super().load_state_dict(
            remapped_state_dict, strict=strict, assign=assign
        )

        # Log loading summary
        model_keys = set(self.state_dict().keys())
        loaded_keys = set(remapped_state_dict.keys())

        matched_keys = model_keys & loaded_keys
        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys

        logger.info("Weight loading summary:")
        logger.info(f"  Loaded: {len(matched_keys)} parameters")
        if missing_keys:
            # Group by module prefix for cleaner output
            missing_prefixes = {k.split(".")[0] for k in missing_keys}
            logger.info(
                f"  Missing (will be randomly initialized): {len(missing_keys)} params from {missing_prefixes}"
            )
        if unexpected_keys:
            unexpected_prefixes = {k.split(".")[0] for k in unexpected_keys}
            logger.info(
                f"  Unexpected (ignored): {len(unexpected_keys)} params from {unexpected_prefixes}"
            )
        if ignored_keys:
            logger.info(f"  Ignored gemma_expert keys: {len(ignored_keys)}")

        return result

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False, **kwargs):
        """Get state dict with optional backwards compatibility format.

        By default, uses new format (experts.{name}.*).
        Set PALIGEMMA_COMPAT_STATE_DICT=1 env var to use old format (gemma_expert.*).
        """
        sd = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars, **kwargs
        )

        if (
            os.environ.get("PALIGEMMA_COMPAT_STATE_DICT", 0) == "1"
            and "action" in self.expert_names
        ):
            compat_sd = {}
            for key, value in sd.items():
                if key.startswith(f"{prefix}experts.action."):
                    new_key = key.replace(
                        f"{prefix}experts.action.", f"{prefix}gemma_expert.", 1
                    )
                    compat_sd[new_key] = value
                else:
                    compat_sd[key] = value
            return compat_sd

        return sd
