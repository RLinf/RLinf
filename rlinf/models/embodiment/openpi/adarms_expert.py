# Copyright 2025 Shirui Chen.
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

"""AdaRMS-compatible expert helpers for OpenPI pi0.5 in RLinf."""

from __future__ import annotations

from types import MethodType

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm


class AdaRMSGemmaRMSNorm(GemmaRMSNorm):
    """Gemma RMSNorm variant that exposes OpenPI pi0.5 AdaRMS weights."""

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__(dim=dim, eps=eps)
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            del self.weight
            self.register_parameter("weight", None)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.dense = None

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = x.dtype
        normed_inputs = self._norm(x.float())

        if self.dense is None or cond is None:
            if self.weight is not None:
                normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None

        modulation = self.dense(cond)
        if x.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed_inputs = normed_inputs * (1.0 + scale.float()) + shift.float()
        return normed_inputs.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        if self.dense is None:
            return f"{self.dim}, eps={self.eps}"
        return f"{self.dim}, eps={self.eps}, adaptive=True, cond_dim={self.cond_dim}"


def _gated_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    if gate is None:
        return residual + update
    return residual + update * gate


def _ensure_gated_residual_support() -> None:
    if not hasattr(modeling_gemma, "_gated_residual"):
        modeling_gemma._gated_residual = _gated_residual  # type: ignore[attr-defined]


def _normalize_with_optional_cond(
    norm_module: nn.Module,
    hidden_states: torch.Tensor,
    cond: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cond is None:
        outputs = norm_module(hidden_states)
    else:
        outputs = norm_module(hidden_states, cond=cond)
    if torch.is_tensor(outputs):
        return outputs, None
    return outputs


def replace_gemma_adarms_norms(gemma_model: nn.Module) -> None:
    """Replace expert Gemma norms with AdaRMS-capable modules."""

    hidden_size = gemma_model.config.hidden_size
    eps = gemma_model.config.rms_norm_eps
    cond_dim = getattr(gemma_model.config, "adarms_cond_dim", None) or hidden_size

    for layer in gemma_model.layers:
        layer.input_layernorm = AdaRMSGemmaRMSNorm(
            hidden_size, eps=eps, cond_dim=cond_dim
        )
        layer.post_attention_layernorm = AdaRMSGemmaRMSNorm(
            hidden_size, eps=eps, cond_dim=cond_dim
        )
    gemma_model.norm = AdaRMSGemmaRMSNorm(hidden_size, eps=eps, cond_dim=cond_dim)


def _manual_gemma_model_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.Tensor | None = None,
    use_cache: bool | None = None,
    adarms_cond: torch.Tensor | None = None,
    **kwargs,
) -> BaseModelOutputWithPast:
    """Run the Gemma expert with AdaRMS/gated residual semantics."""

    del kwargs
    use_cache = (
        use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
    )
    if inputs_embeds is None:
        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")
        if self.embed_tokens is None:
            raise ValueError("embed_tokens is unavailable for this expert model.")
        hidden_states = self.embed_tokens(input_ids)
    else:
        hidden_states = inputs_embeds

    batch_size, seq_len = hidden_states.shape[:2]
    if use_cache and past_key_values is None:
        from transformers.cache_utils import DynamicCache

        past_key_values = DynamicCache(config=self.config)

    if (
        len(self.layers) > 0
        and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    ):
        hidden_states = hidden_states.to(dtype=torch.bfloat16)

    cache_position = None
    if past_key_values is not None:
        past_seen_tokens = past_key_values.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_len,
            device=hidden_states.device,
        )

    if position_ids is None:
        if cache_position is None:
            position_ids = torch.arange(
                seq_len, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
        else:
            position_ids = cache_position.unsqueeze(0)

    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    gradient_checkpointing = (
        getattr(self, "gradient_checkpointing", False) and self.training
    )
    returned_cache = past_key_values if use_cache else None

    def _run_layer(layer, layer_hidden_states):
        residual = layer_hidden_states
        layer_hidden_states, gate = _normalize_with_optional_cond(
            layer.input_layernorm, layer_hidden_states, adarms_cond
        )
        attn_output, _ = layer.self_attn(
            hidden_states=layer_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=bool(use_cache),
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        layer_hidden_states = _gated_residual(residual, attn_output, gate)

        residual = layer_hidden_states
        layer_hidden_states, gate = _normalize_with_optional_cond(
            layer.post_attention_layernorm, layer_hidden_states, adarms_cond
        )
        mlp_dtype = layer.mlp.up_proj.weight.dtype
        if layer_hidden_states.dtype != mlp_dtype:
            layer_hidden_states = layer_hidden_states.to(dtype=mlp_dtype)
        layer_hidden_states = layer.mlp(layer_hidden_states)
        return _gated_residual(residual, layer_hidden_states, gate)

    for layer in self.layers:
        if gradient_checkpointing:
            hidden_states = torch.utils.checkpoint.checkpoint(
                _run_layer,
                layer,
                hidden_states,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            hidden_states = _run_layer(layer, hidden_states)

    hidden_states, _ = _normalize_with_optional_cond(
        self.norm, hidden_states, adarms_cond
    )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=returned_cache,
    )


def _manual_pali_gemma_with_expert_forward(
    self,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: list[torch.Tensor | None] | None = None,
    use_cache: bool | None = None,
    adarms_cond: list[torch.Tensor | None] | None = None,
):
    if inputs_embeds is None:
        raise ValueError("inputs_embeds must be provided.")
    if adarms_cond is None:
        adarms_cond = [None, None]

    if inputs_embeds[1] is None:
        prefix_output = self.paligemma.language_model.forward(
            inputs_embeds=inputs_embeds[0],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return [prefix_output.last_hidden_state, None], prefix_output.past_key_values

    if inputs_embeds[0] is None:
        suffix_output = self.gemma_expert.model.forward(
            inputs_embeds=inputs_embeds[1],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond[1],
        )
        return [None, suffix_output.last_hidden_state], None

    models = [self.paligemma.language_model, self.gemma_expert.model]
    num_layers = self.paligemma.config.text_config.num_hidden_layers
    use_gradient_checkpointing = (
        getattr(self.gemma_expert.model, "gradient_checkpointing", False)
        and self.training
    ) or (getattr(self, "gradient_checkpointing", False) and self.training)

    def compute_layer_complete(
        layer_idx,
        prefix_hidden_states,
        suffix_hidden_states,
        attention_mask,
        position_ids,
        prefix_cond,
        suffix_cond,
    ):
        layer_inputs = [prefix_hidden_states, suffix_hidden_states]
        conds = [prefix_cond, suffix_cond]
        query_states = []
        key_states = []
        value_states = []
        gates = []
        for i, hidden_states in enumerate(layer_inputs):
            layer = models[i].layers[layer_idx]
            hidden_states, gate = _normalize_with_optional_cond(
                layer.input_layernorm, hidden_states, conds[i]
            )
            gates.append(gate)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_state = (
                layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            key_state = (
                layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            value_state = (
                layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        query_states = torch.cat(query_states, dim=2)
        key_states = torch.cat(key_states, dim=2)
        value_states = torch.cat(value_states, dim=2)
        cos, sin = self.paligemma.model.language_model.rotary_emb(
            query_states, position_ids
        )
        query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling
        att_output, _ = modeling_gemma.eager_attention_forward(
            self.paligemma.language_model.layers[layer_idx].self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
        batch_size = att_output.shape[0]
        att_output = att_output.reshape(
            batch_size, -1, att_output.shape[2] * att_output.shape[3]
        )

        outputs_embeds = []
        start_pos = 0
        for i, hidden_states in enumerate(layer_inputs):
            layer = models[i].layers[layer_idx]
            end_pos = start_pos + hidden_states.shape[1]
            layer_attn_output = att_output[:, start_pos:end_pos]
            if layer_attn_output.dtype != layer.self_attn.o_proj.weight.dtype:
                layer_attn_output = layer_attn_output.to(
                    dtype=layer.self_attn.o_proj.weight.dtype
                )
            out_emb = layer.self_attn.o_proj(layer_attn_output)
            out_emb = _gated_residual(hidden_states, out_emb, gates[i])
            residual = out_emb
            out_emb, gate = _normalize_with_optional_cond(
                layer.post_attention_layernorm, out_emb, conds[i]
            )
            mlp_dtype = layer.mlp.up_proj.weight.dtype
            if out_emb.dtype != mlp_dtype:
                out_emb = out_emb.to(dtype=mlp_dtype)
            out_emb = layer.mlp(out_emb)
            outputs_embeds.append(_gated_residual(residual, out_emb, gate))
            start_pos = end_pos

        return outputs_embeds[0], outputs_embeds[1]

    prefix_hidden_states, suffix_hidden_states = inputs_embeds
    prefix_cond, suffix_cond = adarms_cond
    for layer_idx in range(num_layers):
        if use_gradient_checkpointing:

            def custom_forward(prefix_states, suffix_states):
                return compute_layer_complete(
                    layer_idx,
                    prefix_states,
                    suffix_states,
                    attention_mask,
                    position_ids,
                    prefix_cond,
                    suffix_cond,
                )

            prefix_hidden_states, suffix_hidden_states = (
                torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    prefix_hidden_states,
                    suffix_hidden_states,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            )
        else:
            prefix_hidden_states, suffix_hidden_states = compute_layer_complete(
                layer_idx,
                prefix_hidden_states,
                suffix_hidden_states,
                attention_mask,
                position_ids,
                prefix_cond,
                suffix_cond,
            )

    prefix_output, _ = _normalize_with_optional_cond(
        self.paligemma.language_model.norm, prefix_hidden_states, prefix_cond
    )
    suffix_output, _ = _normalize_with_optional_cond(
        self.gemma_expert.model.norm, suffix_hidden_states, suffix_cond
    )
    return [prefix_output, suffix_output], None


def enable_openpi_adarms_expert(paligemma_with_expert: nn.Module) -> None:
    """Enable AdaRMS-compatible expert execution on an OpenPI PaliGemma wrapper."""

    enable_openpi_transformers_compat(paligemma_with_expert, enable_adarms=True)


def enable_openpi_transformers_compat(
    paligemma_with_expert: nn.Module,
    *,
    enable_adarms: bool = False,
) -> None:
    """Patch OpenPI's legacy custom forwards onto vanilla Transformers models.

    OpenPI's PyTorch runtime was authored against a vendored Transformers 4.53
    fork that added AdaRMS-aware Gemma norms and a bespoke PaliGemma+expert
    forward path. RLinf runs against upstream Transformers 4.57 instead, so we
    install the required forward shims at runtime and only swap in AdaRMS norms
    when the checkpoint actually expects them.
    """

    _ensure_gated_residual_support()
    expert_model = paligemma_with_expert.gemma_expert.model
    prefix_model = paligemma_with_expert.paligemma.language_model
    if enable_adarms:
        replace_gemma_adarms_norms(expert_model)
    prefix_model.forward = MethodType(_manual_gemma_model_forward, prefix_model)
    expert_model.forward = MethodType(_manual_gemma_model_forward, expert_model)
    paligemma_with_expert.forward = MethodType(
        _manual_pali_gemma_with_expert_forward, paligemma_with_expert
    )
