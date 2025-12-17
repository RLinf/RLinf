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

from rlinf.config import SupportedModel, get_supported_model


class TransformType(Enum):
    SPLIT_QKV = "split_qkv"
    SPLIT_QKV_BIAS = "split_qkv_bias"
    SPLIT_FC1 = "split_fc1"
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
    def split_fc1(
        linear_fc1: torch.Tensor, new_statedict: dict, weight_names: list[str], config
    ) -> None:
        assert weight_names is not None and len(weight_names) == 2, (
            f"split_fc1 transform expects two weight names, got {weight_names}"
        )

        tp_size = config.model_config.tensor_model_parallel_size
        target_tp = config.reshard_tp_size
        split_size = linear_fc1.shape[0] // (tp_size // target_tp)
        linear_fc1_slice = torch.split(linear_fc1, split_size, dim=0)

        gate_proj_shards = []
        up_proj_shards = []
        for weight in linear_fc1_slice:
            assert weight.shape[0] % 2 == 0, (
                f"linear_fc1 weight shape {weight.shape} is not even along dim 0"
            )
            weight_chunk = torch.chunk(weight, 2, dim=0)
            gate_proj_shards.append(weight_chunk[0])
            up_proj_shards.append(weight_chunk[1])
        gate_proj = torch.cat(gate_proj_shards, dim=0)
        up_proj = torch.cat(up_proj_shards, dim=0)

        new_statedict[weight_names[0]] = gate_proj.clone()
        new_statedict[weight_names[1]] = up_proj.clone()

    @staticmethod
    def split_expert_fc1(
        linear_fc1: torch.Tensor, new_statedict: dict, weight_names: list[str], config
    ) -> None:
        assert weight_names is not None and len(weight_names) == 2, (
            f"split_fc1 transform expects two weight names, got {weight_names}"
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
            elif transform == TransformType.SPLIT_FC1:
                TransformFunc.split_fc1(v, converted, targets, self.cfg)
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
            # mlp fc1
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.linear_fc1\.{WB}$"),
                TransformType.SPLIT_FC1,
                [
                    r"model.layers.\g<i>.mlp.gate_proj.\g<wb>",
                    r"model.layers.\g<i>.mlp.up_proj.\g<wb>",
                ],
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
            # mlp fc1
            ConvertorRule(
                re.compile(rf"^{MG_V_DECODER_PREFIX}\.{B}\.mlp\.linear_fc1\.{WB}$"),
                TransformType.SPLIT_FC1,
                [
                    f"{HF_V_DECODER_PREFIX}" + r".\g<i>.mlp.gate_proj.\g<wb>",
                    f"{HF_V_DECODER_PREFIX}" + r".\g<i>.mlp.up_proj.\g<wb>",
                ],
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
            # mlp fc1
            ConvertorRule(
                re.compile(rf"^{MG_LLM_DECODER_PREFIX}\.{B}\.mlp\.linear_fc1\.{WB}$"),
                TransformType.SPLIT_FC1,
                [
                    f"{HF_LLM_PREFIX}" + r".decoder.layers.\g<i>.mlp.gate_proj.\g<wb>",
                    f"{HF_LLM_PREFIX}" + r".decoder.layers.\g<i>.mlp.up_proj.\g<wb>",
                ],
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
            # mlp fc1
            ConvertorRule(
                re.compile(rf"decoder\.layers\.{LID}\.mlp\.linear_fc1\.{WB}$"),
                TransformType.SPLIT_FC1,
                [
                    r"model.layers.\g<i>.mlp.gate_proj.\g<wb>",
                    r"model.layers.\g<i>.mlp.up_proj.\g<wb>",
                ],
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


_MG2HF_CONVERTOR_REGISTRY = {
    SupportedModel.QWEN2_5: Qwen25Convertor,
    SupportedModel.QWEN2_5_VL: Qwen25VLConvertor,
    SupportedModel.QWEN3: Qwen3DenseConvertor,
    SupportedModel.QWEN3_MOE: Qwen3MoEConvertor,
}


def get_mg2hf_convertor(model_type: str, config, strict: bool = False) -> BaseConvertor:
    model_type = get_supported_model(model_type)
    if model_type not in _MG2HF_CONVERTOR_REGISTRY:
        raise ValueError(f"No convertor registered for {model_type.value}")
    convertor_cls = _MG2HF_CONVERTOR_REGISTRY[model_type]
    return convertor_cls(config=config, strict=strict)
