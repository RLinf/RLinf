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
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch

from rlinf.config import SupportedModel


class TransformType(Enum):
    SPLIT_QKV = "split_qkv"
    SPLIT_QKV_BIAS = "split_qkv_bias"
    SPLIT_EXPERT_FC1 = "split_expert_fc1"
    SPLIT_NONE = "split_none"


class TransformFunc:
    @staticmethod
    def _split_gqa_tensor(
        tensor: torch.Tensor, new_statedict: dict, weight_names: list[str], config
    ) -> None:
        hidden_size = config.model_config.hidden_size
        num_attention_heads = config.model_config.num_attention_heads
        num_query_groups = config.model_config.num_query_groups or num_attention_heads
        head_dim = config.model_config.kv_channels
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        target_tp = config.reshard_tp_size
        assert num_query_groups % target_tp == 0, (
            "num_query_groups must be divisible by reshard_tp_size"
        )
        local_num_query_groups = num_query_groups // target_tp

        # heads per query group
        assert num_attention_heads % num_query_groups == 0, (
            "num_attention_heads must be divisible by num_query_groups"
        )
        q_heads_per_group = num_attention_heads // num_query_groups

        num_channel_qkv = q_heads_per_group + 2

        if tensor.ndim == 2:
            # Weight: [out_features, in_features]
            out_features, in_features = tensor.shape
            expected_out = local_num_query_groups * num_channel_qkv * head_dim
            assert out_features == expected_out, (
                f"Unexpected fused QKV weight shape {tensor.shape}, expect "
                f"[{expected_out}, {in_features}] (local groups={local_num_query_groups})"
            )

            qkv = tensor.view(
                local_num_query_groups, num_channel_qkv, head_dim, in_features
            )
            q, k, v = torch.split(
                qkv, [q_heads_per_group, 1, 1], dim=1
            )  # shapes: [G, qh, D, In], [G,1,D,In], [G,1,D,In]
            q_full = q.reshape(-1, in_features).contiguous()
            k_full = k.reshape(-1, in_features).contiguous()
            v_full = v.reshape(-1, in_features).contiguous()
        else:
            # Bias: [out_features]
            out_features = tensor.shape[0]
            expected_out = local_num_query_groups * num_channel_qkv * head_dim
            assert out_features == expected_out, (
                f"Unexpected fused QKV bias shape {tensor.shape}, expect "
                f"[{expected_out}] (local groups={local_num_query_groups})"
            )

            qkv = tensor.view(local_num_query_groups, num_channel_qkv, head_dim)
            q, k, v = torch.split(qkv, [q_heads_per_group, 1, 1], dim=1)
            q_full = q.reshape(-1).contiguous()
            k_full = k.reshape(-1).contiguous()
            v_full = v.reshape(-1).contiguous()

        # Save to target names
        new_statedict[weight_names[0]] = q_full.clone()
        new_statedict[weight_names[1]] = k_full.clone()
        new_statedict[weight_names[2]] = v_full.clone()

    @staticmethod
    def split_expert_fc1(
        linear_fc1: torch.Tensor, new_statedict: dict, weight_names: list[str], config
    ) -> None:
        assert weight_names is not None and len(weight_names) == 2, (
            f"split_expert_fc1 transform expects two weight names, got {weight_names}"
        )

        weight_chunk = torch.chunk(linear_fc1, 2, dim=0)
        gate_proj = weight_chunk[0]
        up_proj = weight_chunk[1]

        new_statedict[weight_names[0]] = gate_proj.clone()
        new_statedict[weight_names[1]] = up_proj.clone()

    @staticmethod
    def split_none(
        tensor: torch.Tensor, new_statedict: dict, weight_names: list[str]
    ) -> None:
        assert weight_names is not None and len(weight_names) == 1, (
            f"split_none transform expects one weight name, got {weight_names}"
        )
        new_statedict[weight_names[0]] = tensor.clone()


@dataclass
class ConvertorRule:
    pattern: re.Pattern
    transform: TransformType
    targets: list[str]
    post: Optional[Callable] = None


class BaseConvertor:
    def __init__(self, config, strict: bool = False):
        self.cfg = config
        self.strict = strict
        self.rules = self.build_rules()

    def map_name(self, name: str) -> Optional[tuple[TransformType, list[str]]]:
        def _get_targets_from_match(templates: list[str], m: re.Match) -> list[str]:
            gd = m.groupdict()
            out = []
            for t in templates:
                if "{" in t and "}" in t:
                    out.append(t.format(**gd))
                else:
                    out.append(m.expand(t))
            return out

        for r in self.rules:
            m = r.pattern.fullmatch(name)
            if not m:
                continue
            targets = r.targets
            if r.post:
                targets = r.post(targets, m)
            full_names = _get_targets_from_match(targets, m)
            return r.transform, full_names
        return None

    def convert(self, state_dict: dict) -> dict:
        converted = {}
        for k, v in state_dict.items():
            mapped = self.map_name(k)
            if mapped is None:
                if self.strict:
                    raise KeyError(f"Unmapped key {k}")
                continue
            transform, targets = mapped
            if transform in (TransformType.SPLIT_QKV, TransformType.SPLIT_QKV_BIAS):
                TransformFunc._split_gqa_tensor(v, converted, targets, self.cfg)
            elif transform == TransformType.SPLIT_EXPERT_FC1:
                TransformFunc.split_expert_fc1(v, converted, targets, self.cfg)
            elif transform == TransformType.SPLIT_NONE:
                TransformFunc.split_none(v, converted, targets)
            else:
                raise ValueError(f"Unknown transform type {transform}")
        return converted

    def build_rules(self) -> list[ConvertorRule]:
        """
        Should be implemented in subclass to build the conversion rules.
        """
        raise NotImplementedError


class Qwen25Convertor(BaseConvertor):
    def build_rules(self) -> list[ConvertorRule]:
        LID = r"(?P<i>\d+)"
        WB = r"(?P<wb>weight|bias)"

        return [
            # embeddings
            ConvertorRule(
                re.compile(r"embedding\.word_embeddings\.weight$"),
                TransformType.SPLIT_NONE,
                [r"model.embed_tokens.weight"],
            ),
            # final_layernorm
            ConvertorRule(
                re.compile(r"decoder\.final_layernorm\.weight$"),
                TransformType.SPLIT_NONE,
                [r"model.norm.weight"],
            ),
            # lm_head
            ConvertorRule(
                re.compile(r"output_layer\.weight$"),
                TransformType.SPLIT_NONE,
                [r"lm_head.weight"],
            ),
            # attn qkv norm
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_qkv\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.input_layernorm.weight"],
            ),
            # attn qkv weights/bias
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_qkv\.{WB}$"
                ),
                TransformType.SPLIT_QKV,
                [
                    r"model.layers.\g<i>.self_attn.q_proj.\g<wb>",
                    r"model.layers.\g<i>.self_attn.k_proj.\g<wb>",
                    r"model.layers.\g<i>.self_attn.v_proj.\g<wb>",
                ],
            ),
            # attn o proj
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_proj\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.self_attn.o_proj.\g<wb>"],
            ),
            # mlp gate/up (tp_gather_fn already split fused fc1)
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.gate_proj\.{WB}$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.gate_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.up_proj\.{WB}$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.up_proj.\g<wb>"],
            ),
            # mlp fc2
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.linear_fc2\.{WB}$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.down_proj.\g<wb>"],
            ),
            # mlp norms
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.mlp\.linear_fc1\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.post_attention_layernorm.weight"],
            ),
        ]


class Qwen25VLConvertor(BaseConvertor):
    def _build_vision_rules(self) -> list[ConvertorRule]:
        B = r"(?P<i>\d+)"
        WB = r"(?P<wb>weight|bias)"
        HF_V_PREFIX = "model.visual"
        HF_V_DECODER_PREFIX = f"{HF_V_PREFIX}.blocks"
        MG_V_PREFIX = "vision_model"
        MG_V_DECODER_PREFIX = rf"{MG_V_PREFIX}\.decoder\.layers"

        vision_rules = [
            # vision patch embed
            ConvertorRule(
                re.compile(rf"^{MG_V_PREFIX}\.patch_embed\.proj\.weight$"),
                TransformType.SPLIT_NONE,
                [f"{HF_V_PREFIX}.patch_embed.proj.weight"],
            ),
            # final layer norm
            ConvertorRule(
                re.compile(rf"^{MG_V_PREFIX}\.decoder\.final_layernorm\.weight$"),
                TransformType.SPLIT_NONE,
                [f"{HF_V_PREFIX}.merger.ln_q.weight"],
            ),
            # attn norm
            ConvertorRule(
                re.compile(
                    rf"^{MG_V_DECODER_PREFIX}\.{B}\.self_attention\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [f"{HF_V_DECODER_PREFIX}" + r".\g<i>.norm1.weight"],
            ),
            # attn qkv
            ConvertorRule(
                re.compile(
                    rf"^{MG_V_DECODER_PREFIX}\.{B}\.self_attention\.linear_qkv\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [f"{HF_V_DECODER_PREFIX}" + r".\g<i>.attn.qkv.\g<wb>"],
            ),
            # attn proj
            ConvertorRule(
                re.compile(
                    rf"^{MG_V_DECODER_PREFIX}\.{B}\.self_attention\.linear_proj\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [f"{HF_V_DECODER_PREFIX}" + r".\g<i>.attn.proj.\g<wb>"],
            ),
            # mlp gate/up (tp_gather_fn already split fused fc1)
            ConvertorRule(
                re.compile(rf"^{MG_V_DECODER_PREFIX}\.{B}\.mlp\.gate_proj\.{WB}$"),
                TransformType.SPLIT_NONE,
                [f"{HF_V_DECODER_PREFIX}" + r".\g<i>.mlp.gate_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(rf"^{MG_V_DECODER_PREFIX}\.{B}\.mlp\.up_proj\.{WB}$"),
                TransformType.SPLIT_NONE,
                [f"{HF_V_DECODER_PREFIX}" + r".\g<i>.mlp.up_proj.\g<wb>"],
            ),
            # mlp fc2
            ConvertorRule(
                re.compile(rf"^{MG_V_DECODER_PREFIX}\.{B}\.mlp\.linear_fc2\.{WB}$"),
                TransformType.SPLIT_NONE,
                [f"{HF_V_DECODER_PREFIX}" + r".\g<i>.mlp.down_proj.\g<wb>"],
            ),
            # mlp norm
            ConvertorRule(
                re.compile(
                    rf"^{MG_V_DECODER_PREFIX}\.{B}\.mlp\.linear_fc1\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [f"{HF_V_DECODER_PREFIX}" + r".\g<i>.norm2.weight"],
            ),
        ]
        return vision_rules

    def _build_llm_rules(self) -> list[ConvertorRule]:
        B = r"(?P<i>\d+)"
        WB = r"(?P<wb>weight|bias)"
        HF_LLM_PREFIX = "model.language_model"
        MG_LLM_PREFIX = "language_model"
        MG_LLM_DECODER_PREFIX = rf"{MG_LLM_PREFIX}\.decoder\.layers"

        llm_rules = [
            # embeddings
            ConvertorRule(
                re.compile(rf"^{MG_LLM_PREFIX}\.embed_tokens\.weight$"),
                TransformType.SPLIT_NONE,
                [f"{HF_LLM_PREFIX}.embedding.weight"],
            ),
            # final_layernorm
            ConvertorRule(
                re.compile(rf"^{MG_LLM_PREFIX}\.final_layernorm\.weight$"),
                TransformType.SPLIT_NONE,
                [f"{HF_LLM_PREFIX}.norm.weight"],
            ),
            # attn norm
            ConvertorRule(
                re.compile(
                    rf"^{MG_LLM_DECODER_PREFIX}\.{B}\.self_attention\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [f"{HF_LLM_PREFIX}" + r".decoder.layers.\g<i>.input_layernorm.weight"],
            ),
            # attn qkv
            ConvertorRule(
                re.compile(
                    rf"^{MG_LLM_DECODER_PREFIX}\.{B}\.self_attention\.linear_qkv\.{WB}$"
                ),
                TransformType.SPLIT_QKV,
                [
                    f"{HF_LLM_PREFIX}"
                    + r".decoder.layers.\g<i>.self_attn.q_proj.\g<wb>",
                    f"{HF_LLM_PREFIX}"
                    + r".decoder.layers.\g<i>.self_attn.k_proj.\g<wb>",
                    f"{HF_LLM_PREFIX}"
                    + r".decoder.layers.\g<i>.self_attn.v_proj.\g<wb>",
                ],
            ),
            # attn proj
            ConvertorRule(
                re.compile(
                    rf"^{MG_LLM_DECODER_PREFIX}\.{B}\.self_attention\.linear_proj\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [f"{HF_LLM_PREFIX}" + r".decoder.layers.\g<i>.self_attn.o_proj.\g<wb>"],
            ),
            # mlp gate/up (tp_gather_fn already split fused fc1)
            ConvertorRule(
                re.compile(rf"^{MG_LLM_DECODER_PREFIX}\.{B}\.mlp\.gate_proj\.{WB}$"),
                TransformType.SPLIT_NONE,
                [f"{HF_LLM_PREFIX}" + r".decoder.layers.\g<i>.mlp.gate_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(rf"^{MG_LLM_DECODER_PREFIX}\.{B}\.mlp\.up_proj\.{WB}$"),
                TransformType.SPLIT_NONE,
                [f"{HF_LLM_PREFIX}" + r".decoder.layers.\g<i>.mlp.up_proj.\g<wb>"],
            ),
            # mlp fc2
            ConvertorRule(
                re.compile(rf"^{MG_LLM_DECODER_PREFIX}\.{B}\.mlp\.linear_fc2\.{WB}$"),
                TransformType.SPLIT_NONE,
                [f"{HF_LLM_PREFIX}" + r".decoder.layers.\g<i>.mlp.down_proj.\g<wb>"],
            ),
            # mlp norm
            ConvertorRule(
                re.compile(
                    rf"^{MG_LLM_DECODER_PREFIX}\.{B}\.mlp\.linear_fc1\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [
                    f"{HF_LLM_PREFIX}"
                    + r".decoder.layers.\g<i>.post_attention_layernorm.weight"
                ],
            ),
        ]
        return llm_rules

    def _build_projector_rules(self) -> list[ConvertorRule]:
        HF_PROJECTOR_PREFIX = "model.visual.merger"
        MG_PROJECTOR_PREFIX = "vision_model.protection.encoder"
        WB = r"(?P<wb>weight|bias)"

        projector_rules = [
            # projector fc1
            ConvertorRule(
                re.compile(rf"^{MG_PROJECTOR_PREFIX}\.linear_fc1\.{WB}$"),
                TransformType.SPLIT_NONE,
                [f"{HF_PROJECTOR_PREFIX}" + r".mlp.0.\g<wb>"],
            ),
            # projector fc2
            ConvertorRule(
                re.compile(rf"^{MG_PROJECTOR_PREFIX}\.linear_fc2\.{WB}$"),
                TransformType.SPLIT_NONE,
                [f"{HF_PROJECTOR_PREFIX}" + r".mlp.2.\g<wb>"],
            ),
        ]
        return projector_rules

    def build_rules(self) -> list[ConvertorRule]:
        rules = []
        rules.extend(self._build_vision_rules())
        rules.extend(self._build_llm_rules())
        rules.extend(self._build_projector_rules())
        return rules


class Qwen3BaseConvertor(BaseConvertor):
    def build_rules(self) -> list[ConvertorRule]:
        LID = r"(?P<i>\d+)"
        WB = r"(?P<wb>weight|bias)"

        return [
            # embeddings
            ConvertorRule(
                re.compile(r"embedding\.word_embeddings\.weight$"),
                TransformType.SPLIT_NONE,
                [r"model.embed_tokens.weight"],
            ),
            # final_layernorm
            ConvertorRule(
                re.compile(r"decoder\.final_layernorm\.weight$"),
                TransformType.SPLIT_NONE,
                [r"model.norm.weight"],
            ),
            # lm_head
            ConvertorRule(
                re.compile(r"output_layer\.weight$"),
                TransformType.SPLIT_NONE,
                [r"lm_head.weight"],
            ),
            # attn qkv norm
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_qkv\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.input_layernorm.weight"],
            ),
            # attn qkv weights/bias
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_qkv\.{WB}$"
                ),
                TransformType.SPLIT_QKV,
                [
                    r"model.layers.\g<i>.self_attn.q_proj.\g<wb>",
                    r"model.layers.\g<i>.self_attn.k_proj.\g<wb>",
                    r"model.layers.\g<i>.self_attn.v_proj.\g<wb>",
                ],
            ),
            # attn q layernorm weight/bias
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.q_layernorm\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [
                    r"model.layers.\g<i>.self_attn.q_norm.\g<wb>",
                ],
            ),
            # attn k layernorm weight/bias
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.k_layernorm\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [
                    r"model.layers.\g<i>.self_attn.k_norm.\g<wb>",
                ],
            ),
            # attn o proj
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_proj\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.self_attn.o_proj.\g<wb>"],
            ),
        ]


class Qwen3DenseConvertor(Qwen3BaseConvertor):
    def build_rules(self) -> list[ConvertorRule]:
        LID = r"(?P<i>\d+)"
        WB = r"(?P<wb>weight|bias)"

        return [
            *super().build_rules(),
            # mlp gate/up (tp_gather_fn already split fused fc1)
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.gate_proj\.{WB}$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.gate_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.up_proj\.{WB}$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.up_proj.\g<wb>"],
            ),
            # mlp fc2
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.linear_fc2\.{WB}$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.down_proj.\g<wb>"],
            ),
            # mlp norms
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.mlp\.linear_fc1\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.post_attention_layernorm.weight"],
            ),
        ]


class Qwen35Convertor(BaseConvertor):
    """Convert target TP shards of Qwen3.5 VL, including its MTP branch."""

    _LAYER = re.compile(
        r"(?:language_model\.)?(?:decoder\.layers\.(?P<i>\d+)|"
        r"mtp\.layers\.(?P<m>\d+)\.mtp_model_layer)\.(?P<name>.+)"
    )
    _VISION_QKV = re.compile(
        r"vision_model\.decoder\.layers\.(?P<i>\d+)\.self_attention\.linear_qkv\.(?P<wb>weight|bias)"
    )

    def __init__(self, config, strict: bool = False):
        model, tp = config.model_config, config.reshard_tp_size
        vision = config.vision_config
        assert vision is not None, "Qwen3.5 VL conversion requires vision_config."
        sizes = (
            model.num_attention_heads,
            model.linear_num_key_heads,
            model.linear_num_value_heads,
            model.ffn_hidden_size,
        )
        assert tp >= 1 and all(size % tp == 0 for size in sizes), (
            f"Qwen3.5 language head and FFN dimensions {sizes} must be divisible by target TP={tp}."
        )
        groups = model.num_query_groups
        assert groups % tp == 0 or tp % groups == 0, (
            f"Qwen3.5 KV groups={groups} and target TP={tp} must divide one another."
        )
        vision_tp = config.rollout_vision_tp_size
        vision_sizes = (
            vision.num_heads,
            vision.intermediate_size,
            vision.hidden_size * vision.spatial_merge_size**2,
        )
        assert all(size % vision_tp == 0 for size in vision_sizes), (
            f"Qwen3.5 vision dimensions {vision_sizes} must be divisible by vision TP={vision_tp}."
        )
        super().__init__(config, strict)

    def build_rules(self) -> list[ConvertorRule]:
        rules = []

        def add(source: str, target: str) -> None:
            rules.append(
                ConvertorRule(re.compile(source), TransformType.SPLIT_NONE, [target])
            )

        root = r"(?:language_model\.)?"
        add(
            root + r"embedding\.word_embeddings\.weight",
            "model.language_model.embed_tokens.weight",
        )
        add(
            root + r"decoder\.final_layernorm\.weight",
            "model.language_model.norm.weight",
        )
        add(root + r"output_layer\.weight", "lm_head.weight")
        layer_names = {
            "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
            "self_attention.in_proj.layer_norm_weight": "input_layernorm.weight",
            "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
            "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
            "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
            "self_attention.A_log": "linear_attn.A_log",
            "self_attention.dt_bias": "linear_attn.dt_bias",
            "self_attention.conv1d.weight": "linear_attn.conv1d.weight",
            "self_attention.out_proj.weight": "linear_attn.out_proj.weight",
            "mlp.gate_proj.weight": "mlp.gate_proj.weight",
            "mlp.up_proj.weight": "mlp.up_proj.weight",
            "mlp.linear_fc2.weight": "mlp.down_proj.weight",
            "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        }
        for source, target in layer_names.items():
            add(
                root + r"decoder\.layers\.(?P<i>\d+)\." + re.escape(source),
                rf"model.language_model.layers.\g<i>.{target}",
            )
            if "in_proj" not in source and "linear_attn" not in target:
                add(
                    root
                    + r"mtp\.layers\.(?P<i>\d+)\.mtp_model_layer\."
                    + re.escape(source),
                    rf"mtp.layers.\g<i>.{target}",
                )
        for source, target in {
            "eh_proj": "fc",
            "enorm": "pre_fc_norm_embedding",
            "hnorm": "pre_fc_norm_hidden",
            "final_layernorm": "norm",
        }.items():
            add(root + rf"mtp\.layers\.0\.{source}\.weight", f"mtp.{target}.weight")

        vision_names = {
            "self_attention.linear_proj": "attn.proj",
            "mlp.linear_fc1": "mlp.linear_fc1",
            "mlp.linear_fc2": "mlp.linear_fc2",
        }
        for source, target in vision_names.items():
            add(
                r"vision_model\.decoder\.layers\.(?P<i>\d+)\."
                + re.escape(source)
                + r"\.(?P<wb>weight|bias)",
                rf"model.visual.blocks.\g<i>.{target}.\g<wb>",
            )
        for source, target in (
            ("self_attention.linear_qkv", "norm1"),
            ("mlp.linear_fc1", "norm2"),
        ):
            add(
                r"vision_model\.decoder\.layers\.(?P<i>\d+)\."
                + re.escape(source)
                + r"\.layer_norm_(?P<wb>weight|bias)",
                rf"model.visual.blocks.\g<i>.{target}.\g<wb>",
            )
        add(
            r"vision_model\.patch_embed\.proj\.(?P<wb>weight|bias)",
            r"model.visual.patch_embed.proj.\g<wb>",
        )
        add(r"vision_model\.pos_embed\.weight", "model.visual.pos_embed.weight")
        add(
            r"vision_model\.merger\.patch_norm\.(?P<wb>weight|bias)",
            r"model.visual.merger.norm.\g<wb>",
        )
        add(
            r"vision_model\.merger\.(?P<fc>linear_fc[12])\.(?P<wb>weight|bias)",
            r"model.visual.merger.\g<fc>.\g<wb>",
        )
        return rules

    def convert(self, state_dict: dict) -> dict:
        converted, renamed = {}, {}
        for key, tensor in state_dict.items():
            layer = self._LAYER.fullmatch(key)
            vision = self._VISION_QKV.fullmatch(key)
            if vision:
                heads = (
                    self.cfg.vision_config.num_heads // self.cfg.rollout_vision_tp_size
                )
                dim = (
                    self.cfg.vision_config.hidden_size
                    // self.cfg.vision_config.num_heads
                )
                q, k, v = tensor.reshape(heads, 3, dim, *tensor.shape[1:]).unbind(1)
                converted[
                    f"model.visual.blocks.{vision['i']}.attn.qkv.{vision['wb']}"
                ] = torch.cat(
                    [part.reshape(-1, *tensor.shape[1:]) for part in (q, k, v)], dim=0
                ).contiguous()
                continue
            if layer:
                prefix = (
                    f"mtp.layers.{layer['m']}."
                    if layer["m"] is not None
                    else f"model.language_model.layers.{layer['i']}."
                )
                if layer["name"] == "self_attention.linear_qkv.weight":
                    converted.update(self._split_gated_qkv(prefix, tensor))
                    continue
                if layer["name"] == "self_attention.in_proj.weight":
                    converted.update(self._split_gdn_in_proj(prefix, tensor))
                    continue
                if layer["name"] == "self_attention.out_norm.weight":
                    converted[prefix + "linear_attn.norm.weight"] = tensor + 1
                    continue
            renamed[key] = tensor
        converted.update(super().convert(renamed))
        return converted

    def _split_gated_qkv(self, prefix: str, tensor: torch.Tensor) -> dict:
        model = self.cfg.model_config
        groups = max(1, model.num_query_groups // self.cfg.reshard_tp_size)
        heads = model.num_attention_heads // max(
            model.num_query_groups, self.cfg.reshard_tp_size
        )
        dim, hidden = model.kv_channels, model.hidden_size
        qg, k, v = tensor.reshape(groups, 2 * heads + 2, dim, hidden).split(
            [2 * heads, 1, 1], dim=1
        )
        qg = qg.reshape(groups, 2, heads, dim, hidden).transpose(1, 2)
        return {
            prefix + "self_attn.q_proj.weight": qg.reshape(-1, hidden).contiguous(),
            prefix + "self_attn.k_proj.weight": k.reshape(-1, hidden).contiguous(),
            prefix + "self_attn.v_proj.weight": v.reshape(-1, hidden).contiguous(),
        }

    def _split_gdn_in_proj(self, prefix: str, tensor: torch.Tensor) -> dict:
        model, tp = self.cfg.model_config, self.cfg.reshard_tp_size
        qk = model.linear_key_head_dim * model.linear_num_key_heads // tp
        v = model.linear_value_head_dim * model.linear_num_value_heads // tp
        heads = model.linear_num_value_heads // tp
        parts = tensor.split([2 * qk + v, v, heads, heads], dim=0)
        return {
            prefix + f"linear_attn.in_proj_{name}.weight": part.contiguous()
            for name, part in zip(("qkv", "z", "b", "a"), parts, strict=True)
        }


class Qwen3MoEConvertor(Qwen3BaseConvertor):
    def build_rules(self) -> list[ConvertorRule]:
        LID = r"(?P<i>\d+)"
        EID = r"(?P<ei>\d+)"
        WB = r"(?P<wb>weight|bias)"

        return [
            *super().build_rules(),
            # mlp expert fc1
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.mlp\.experts\.local_experts\.{EID}\.linear_fc1\.{WB}$"
                ),
                TransformType.SPLIT_EXPERT_FC1,
                [
                    r"model.layers.\g<i>.mlp.experts.\g<ei>.gate_proj.\g<wb>",
                    r"model.layers.\g<i>.mlp.experts.\g<ei>.up_proj.\g<wb>",
                ],
            ),
            # mlp expert fc2
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.mlp\.experts\.local_experts\.{EID}\.linear_fc2\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.experts.\g<ei>.down_proj.\g<wb>"],
            ),
            # pre_mlp_layernorms
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.pre_mlp_layernorm\.{WB}$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.post_attention_layernorm.\g<wb>"],
            ),
            # router weight
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.router\.{WB}$"),
                TransformType.SPLIT_NONE,
                [
                    r"model.layers.\g<i>.mlp.gate.\g<wb>",
                ],
            ),
        ]


class DeepseekV3Convertor(BaseConvertor):
    """mg2hf convertor for DeepSeek-V3 text backbone.

    Architecture: Multi-Latent Attention (MLA) + MoE with a shared expert and
    routed experts (TE grouped-gemm layout, converted to local_experts by
    moe_te_group_to_seq upstream). All MLA projections and norms map 1:1
    (SPLIT_NONE); TP sharding for them is handled by tp_gather_fn_deepseek_v3.
    Fused fc1 (dense / shared / routed-expert) is decomposed into gate+up.
    """

    def build_rules(self) -> list[ConvertorRule]:
        LID = r"(?P<i>\d+)"
        EID = r"(?P<ei>\d+)"
        WB = r"(?P<wb>weight|bias)"
        # MTP layer index = num_hidden_layers (sglang nextn loads as model.layers.{NL}).
        NL = self.cfg.model_config.num_layers

        return [
            # ---- model-level ----
            ConvertorRule(
                re.compile(r"embedding\.word_embeddings\.weight$"),
                TransformType.SPLIT_NONE,
                [r"model.embed_tokens.weight"],
            ),
            ConvertorRule(
                re.compile(r"decoder\.final_layernorm\.weight$"),
                TransformType.SPLIT_NONE,
                [r"model.norm.weight"],
            ),
            ConvertorRule(
                re.compile(r"output_layer\.weight$"),
                TransformType.SPLIT_NONE,
                [r"lm_head.weight"],
            ),
            # MLA attention (all SPLIT_NONE; TP handled by tp_gather_fn)
            # Moonlight (q_lora_rank=None): standard q_proj, no low-rank decomposition.
            # DeepSeek-V3 (q_lora_rank=1536) does NOT have linear_q_proj.
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_q_proj\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.self_attn.q_proj.weight"],
            ),
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_q_down_proj\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.self_attn.q_a_proj.weight"],
            ),
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_q_up_proj\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.self_attn.q_b_proj.weight"],
            ),
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_q_up_proj\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.self_attn.q_a_layernorm.weight"],
            ),
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_kv_down_proj\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.self_attn.kv_a_proj_with_mqa.weight"],
            ),
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_kv_up_proj\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.self_attn.kv_b_proj.weight"],
            ),
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_kv_up_proj\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.self_attn.kv_a_layernorm.weight"],
            ),
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_proj\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.self_attn.o_proj.\g<wb>"],
            ),
            # ---- layer norms ----
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.input_layernorm\.weight$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.input_layernorm.weight"],
            ),
            # MoE-layer post-attention norm
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.pre_mlp_layernorm\.weight$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.post_attention_layernorm.weight"],
            ),
            # dense-layer post-attention norm (fused into fc1 layernorm)
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.mlp\.linear_fc1\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.post_attention_layernorm.weight"],
            ),
            # ---- dense MLP (dense FFN layer) ----
            # tp_gather_fn already split fused linear_fc1 into gate_proj/up_proj
            # per-rank (unified fc1 branch), so these just need a rename.
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.gate_proj\.{WB}$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.gate_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.up_proj\.{WB}$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.up_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.linear_fc2\.{WB}$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.down_proj.\g<wb>"],
            ),
            # ---- shared expert MLP ----
            # shared_experts.linear_fc1 is split by tp_gather_fn (unified fc1
            # branch) into gate_proj/up_proj; the SPLIT_NONE rules below handle
            # them. linear_fc2 (down_proj) just renames.
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.mlp\.shared_experts\.linear_fc2\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.shared_experts.down_proj.\g<wb>"],
            ),
            # Pre-split shared_experts gate/up (DPA path: tp_reshard pre-splits
            # fused fc1 into gate_proj/up_proj to avoid slice-before-split; these
            # just need a rename, no further split).
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.mlp\.shared_experts\.gate_proj\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.shared_experts.gate_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.mlp\.shared_experts\.up_proj\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.shared_experts.up_proj.\g<wb>"],
            ),
            # ---- routed experts (after moe_te_group_to_seq -> local_experts.{EID}) ----
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.mlp\.experts\.local_experts\.{EID}\.linear_fc1\.{WB}$"
                ),
                TransformType.SPLIT_EXPERT_FC1,
                [
                    r"model.layers.\g<i>.mlp.experts.\g<ei>.gate_proj.\g<wb>",
                    r"model.layers.\g<i>.mlp.experts.\g<ei>.up_proj.\g<wb>",
                ],
            ),
            ConvertorRule(
                re.compile(
                    rf"decoder\.layers\.{LID}\.mlp\.experts\.local_experts\.{EID}\.linear_fc2\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.experts.\g<ei>.down_proj.\g<wb>"],
            ),
            # ---- router ----
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.router\.weight$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.gate.weight"],
            ),
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.router\.expert_bias$"),
                TransformType.SPLIT_NONE,
                [r"model.layers.\g<i>.mlp.gate.e_score_correction_bias"],
            ),
            # ---- MTP (mtp.layers.0): GLM-4.7-Flash ships MTP (num_nextn_predict_layers=1);
            # DeepSeek-V3 ran with MTP off so these were absent. mtp_model_layer mirrors a
            # regular MLA+MoE decoder layer (same internal names) -> model.layers.{NL}.*;
            # sglang's nextn load_weights remaps model.layers.{NL} -> model/model.decoder.
            # final_layernorm has no HF checkpoint slot; sglang skips it (no matching param).
            ConvertorRule(
                re.compile(r"mtp\.layers\.0\.enorm\.weight$"),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.enorm.weight"],
            ),
            ConvertorRule(
                re.compile(r"mtp\.layers\.0\.hnorm\.weight$"),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.hnorm.weight"],
            ),
            ConvertorRule(
                re.compile(r"mtp\.layers\.0\.eh_proj\.weight$"),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.eh_proj.weight"],
            ),
            ConvertorRule(
                re.compile(r"mtp\.layers\.0\.final_layernorm\.weight$"),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.final_layernorm.weight"],
            ),
            ConvertorRule(
                re.compile(
                    r"mtp\.layers\.0\.mtp_model_layer\.input_layernorm\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.input_layernorm.weight"],
            ),
            ConvertorRule(
                re.compile(
                    r"mtp\.layers\.0\.mtp_model_layer\.pre_mlp_layernorm\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.post_attention_layernorm.weight"],
            ),
            ConvertorRule(
                re.compile(
                    r"mtp\.layers\.0\.mtp_model_layer\.self_attention\.linear_q_down_proj\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.self_attn.q_a_proj.weight"],
            ),
            ConvertorRule(
                re.compile(
                    r"mtp\.layers\.0\.mtp_model_layer\.self_attention\.linear_q_up_proj\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.self_attn.q_b_proj.weight"],
            ),
            ConvertorRule(
                re.compile(
                    r"mtp\.layers\.0\.mtp_model_layer\.self_attention\.linear_q_up_proj\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.self_attn.q_a_layernorm.weight"],
            ),
            ConvertorRule(
                re.compile(
                    r"mtp\.layers\.0\.mtp_model_layer\.self_attention\.linear_kv_down_proj\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.self_attn.kv_a_proj_with_mqa.weight"],
            ),
            ConvertorRule(
                re.compile(
                    r"mtp\.layers\.0\.mtp_model_layer\.self_attention\.linear_kv_up_proj\.weight$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.self_attn.kv_b_proj.weight"],
            ),
            ConvertorRule(
                re.compile(
                    r"mtp\.layers\.0\.mtp_model_layer\.self_attention\.linear_kv_up_proj\.layer_norm_weight$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.self_attn.kv_a_layernorm.weight"],
            ),
            ConvertorRule(
                re.compile(
                    rf"mtp\.layers\.0\.mtp_model_layer\.self_attention\.linear_proj\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.self_attn.o_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(
                    rf"mtp\.layers\.0\.mtp_model_layer\.mlp\.shared_experts\.linear_fc2\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.mlp.shared_experts.down_proj.\g<wb>"],
            ),
            # Pre-split MTP shared_experts gate/up (tp_reshard unified fc1 branch
            # splits shared linear_fc1 into gate_proj/up_proj; just rename).
            ConvertorRule(
                re.compile(
                    rf"mtp\.layers\.0\.mtp_model_layer\.mlp\.shared_experts\.gate_proj\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.mlp.shared_experts.gate_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(
                    rf"mtp\.layers\.0\.mtp_model_layer\.mlp\.shared_experts\.up_proj\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.mlp.shared_experts.up_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(
                    rf"mtp\.layers\.0\.mtp_model_layer\.mlp\.experts\.local_experts\.{EID}\.linear_fc1\.{WB}$"
                ),
                TransformType.SPLIT_EXPERT_FC1,
                [
                    rf"model.layers.{NL}.mlp.experts.\g<ei>.gate_proj.\g<wb>",
                    rf"model.layers.{NL}.mlp.experts.\g<ei>.up_proj.\g<wb>",
                ],
            ),
            ConvertorRule(
                re.compile(
                    rf"mtp\.layers\.0\.mtp_model_layer\.mlp\.experts\.local_experts\.{EID}\.linear_fc2\.{WB}$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.mlp.experts.\g<ei>.down_proj.\g<wb>"],
            ),
            ConvertorRule(
                re.compile(r"mtp\.layers\.0\.mtp_model_layer\.mlp\.router\.weight$"),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.mlp.gate.weight"],
            ),
            ConvertorRule(
                re.compile(
                    r"mtp\.layers\.0\.mtp_model_layer\.mlp\.router\.expert_bias$"
                ),
                TransformType.SPLIT_NONE,
                [rf"model.layers.{NL}.mlp.gate.e_score_correction_bias"],
            ),
        ]

    def convert(self, state_dict: dict) -> dict:
        converted = super().convert(state_dict)
        # on sglang 0.4.6 kv_a_proj_with_mqa must arrive before q_a_proj
        # not functionally required on 0.5.12
        kv = {k: v for k, v in converted.items() if "kv_a_proj_with_mqa" in k}
        qa = {k: v for k, v in converted.items() if "q_a_proj" in k and "kv_a" not in k}
        if kv and qa:
            rest = {k: v for k, v in converted.items() if k not in kv and k not in qa}
            converted = {**kv, **rest, **qa}
        return converted


_MG2HF_CONVERTOR_REGISTRY = {
    SupportedModel.QWEN2_5: Qwen25Convertor,
    SupportedModel.QWEN2_5_VL: Qwen25VLConvertor,
    SupportedModel.QWEN3: Qwen3DenseConvertor,
    SupportedModel.QWEN3_5: Qwen35Convertor,
    SupportedModel.QWEN3_MOE: Qwen3MoEConvertor,
    SupportedModel.DEEPSEEK_V3: DeepseekV3Convertor,
    # GLM-4.7-Flash: same MLA+MoE HF weight layout as DeepSeek-V3 (MLA q_a/kv_a/
    # kv_b projections, experts.E.{gate,up,down}_proj, shared_experts, router
    # gate+e_score_correction_bias). MTP is off (mtp_num_layers=None). Reuse the
    # DeepSeek-V3 convertor (verify mg<->hf at e2e weight dump).
    SupportedModel.GLM4_MOE_LITE: DeepseekV3Convertor,
}


def get_mg2hf_convertor(model_type: str, config, strict: bool = False) -> BaseConvertor:
    model_type = SupportedModel(model_type)
    if model_type not in _MG2HF_CONVERTOR_REGISTRY:
        raise ValueError(f"No converter registered for {model_type.value}")
    convertor_cls = _MG2HF_CONVERTOR_REGISTRY[model_type]
    return convertor_cls(config=config, strict=strict)
