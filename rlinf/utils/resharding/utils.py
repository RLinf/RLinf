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
from enum import Enum
from typing import List, Tuple

import torch
from megatron.core import parallel_state


def get_convert_fn(model_arch: str):
    if model_arch == "qwen2.5":
        return TransformFunc.convert_mega_qwen2_5_to_hf
    else:
        raise NotImplementedError(
            f"get_convert_fn for model_arch {model_arch} is not implemented"
        )


def get_tp_reshard_fn(model_arch: str):
    if model_arch == "qwen2.5":
        return tp_reshard_fn_qwen2_5
    else:
        raise NotImplementedError(
            f"get_tp_reshard_fn for model_arch {model_arch} is not implemented"
        )


def get_pp_reshard_fn(model_arch: str):
    if model_arch == "qwen2.5":
        return pp_reshard_fn_qwen2_5
    else:
        raise NotImplementedError(
            f"get_pp_reshard_fn for model_arch {model_arch} is not implemented"
        )


###########################
# convert fn implementation
###########################


class TransformType(Enum):
    SPLIT_QKV = "split_qkv"
    SPLIT_QKV_BIAS = "split_qkv_bias"
    SPLIT_FC1 = "split_fc1"
    SPLIT_NONE = "split_none"


class TransformFunc:
    @staticmethod
    def _split_gqa_tensor(
        tensor: torch.Tensor, new_statedict: dict, weight_names: List[str], config
    ) -> None:
        """
        Private helper to split a GQA-combined tensor (weight or bias).
        """
        hidden_size = config.model_config.hidden_size
        num_attention_heads = config.model_config.num_attention_heads
        num_key_value_heads = (
            config.model_config.num_query_groups or num_attention_heads
        )
        head_dim = hidden_size // num_attention_heads

        tp_size = config.model_config.tensor_model_parallel_size

        assert num_key_value_heads % tp_size == 0, (
            "num_key_value_heads must be divisible by tensor parallel size"
        )

        q_heads_per_rank = num_attention_heads // tp_size
        kv_heads_per_rank = num_key_value_heads // tp_size

        q_shard_size = q_heads_per_rank * head_dim
        k_shard_size = kv_heads_per_rank * head_dim
        v_shard_size = kv_heads_per_rank * head_dim

        shard_size = q_shard_size + k_shard_size + v_shard_size

        q_shards, k_shards, v_shards = [], [], []

        # [Qi,Ki,Vi]
        for shard in tensor.split(shard_size, dim=0):
            # Qi, Ki, Vi
            q_shard, k_shard, v_shard = shard.split(
                [q_shard_size, k_shard_size, v_shard_size], dim=0
            )
            q_shards.append(q_shard)
            k_shards.append(k_shard)
            v_shards.append(v_shard)

        # cat
        q_full = torch.cat(q_shards, dim=0)
        k_full = torch.cat(k_shards, dim=0)
        v_full = torch.cat(v_shards, dim=0)

        # saved
        new_statedict[weight_names[0]] = q_full.clone()
        new_statedict[weight_names[1]] = k_full.clone()
        new_statedict[weight_names[2]] = v_full.clone()

    @staticmethod
    def split_fc1(
        linear_fc1: torch.Tensor, new_statedict: dict, weight_names: List[str], config
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
    def split_none(
        tensor: torch.Tensor, new_statedict: dict, weight_names: List[str]
    ) -> None:
        assert weight_names is not None and len(weight_names) == 1, (
            f"split_none transform expects one weight name, got {weight_names}"
        )
        new_statedict[weight_names[0]] = tensor.clone()

    @staticmethod
    def mega_name_qwen2_5_to_hf(name: str) -> Tuple[TransformType, List[str]]:
        """
        Convert qwen2_5 model weight megatron name to hf name and do shape transform if needed.

        Args:
            name (str): megatron model weight name

        Returns:
            (TransformType, List[str]): transform type and the corresponding hf model weight name
        """
        if "embedding.word_embeddings.weight" in name:
            return (TransformType.SPLIT_NONE, ["model.embed_tokens.weight"])
        if "decoder.final_layernorm.weight" in name:
            return (TransformType.SPLIT_NONE, ["model.norm.weight"])
        if "output_layer.weight" in name:
            return (TransformType.SPLIT_NONE, ["lm_head.weight"])
        layer_id, suffix = TransformFunc.extract_layer_info(name)
        assert layer_id is not None, f"Cannot extract layer info from {name}"
        result_pattern = "model.layers.{}.{}"
        nmap = {
            "self_attention.linear_proj.weight": (
                TransformType.SPLIT_NONE,
                ["self_attn.o_proj.weight"],
            ),
            "self_attention.linear_qkv.layer_norm_weight": (
                TransformType.SPLIT_NONE,
                ["input_layernorm.weight"],
            ),
            "self_attention.linear_qkv.weight": (
                TransformType.SPLIT_QKV,
                [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
            ),
            "self_attention.linear_qkv.bias": (
                TransformType.SPLIT_QKV_BIAS,
                [
                    "self_attn.q_proj.bias",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.bias",
                ],
            ),
            "mlp.linear_fc1.layer_norm_weight": (
                TransformType.SPLIT_NONE,
                ["post_attention_layernorm.weight"],
            ),
            "mlp.linear_fc1.weight": (
                TransformType.SPLIT_FC1,
                ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            ),
            "mlp.linear_fc2.weight": (
                TransformType.SPLIT_NONE,
                ["mlp.down_proj.weight"],
            ),
        }

        assert suffix in nmap, f"Cannot find mapping for {suffix}"

        transform_type, suffixes = nmap[suffix]
        return (
            transform_type,
            [result_pattern.format(layer_id, suffix) for suffix in suffixes],
        )

    @staticmethod
    def convert_mega_qwen2_5_to_hf(model_state_dict: dict, config) -> dict:
        new_statedict = {}
        for name, param in model_state_dict.items():
            transform_type, hf_names = TransformFunc.mega_name_qwen2_5_to_hf(name)
            if transform_type == TransformType.SPLIT_QKV:
                TransformFunc._split_gqa_tensor(param, new_statedict, hf_names, config)
            elif transform_type == TransformType.SPLIT_QKV_BIAS:
                TransformFunc._split_gqa_tensor(param, new_statedict, hf_names, config)
            elif transform_type == TransformType.SPLIT_FC1:
                TransformFunc.split_fc1(param, new_statedict, hf_names, config)
            elif transform_type == TransformType.SPLIT_NONE:
                TransformFunc.split_none(param, new_statedict, hf_names)
            else:
                raise NotImplementedError(
                    f"Transform type {transform_type} not implemented"
                )
        return new_statedict

    @staticmethod
    def extract_layer_info(s):
        pattern = r"layers\.(\d+)\.(.+)"
        match = re.search(pattern, s)
        if match:
            return match.group(1), match.group(2)
        return None, None


##############################
# tp reshard fn implementation
##############################


def _gather_tp_group_tensor_and_reshard(tensor, dim, merge_factor, tp_group):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(merge_factor)]

    torch.distributed.all_gather(gathered_tensors, tensor, group=tp_group)

    resharded_tensor = torch.cat(gathered_tensors, dim=dim)

    return resharded_tensor


def tp_reshard_fn_qwen2_5(model_state_dict, merge_factor, tp_group):
    for k, v in model_state_dict.items():
        if (
            "rotary_pos_emb.inv_freq" in k
            or "linear_qkv.layer_norm_weight" in k
            or "mlp.linear_fc1.layer_norm_weight" in k
            or "final_layernorm.weight" in k
        ):
            model_state_dict[k] = v.clone()
            continue

        dim = 0
        if "self_attention.linear_proj.weight" in k or "mlp.linear_fc2.weight" in k:
            dim = 1
        model_state_dict[k] = _gather_tp_group_tensor_and_reshard(
            v, dim, merge_factor, tp_group
        )
    return model_state_dict


##############################
# pp reshard fn implementation
##############################


def _gather_pp_group_tensor_and_reshard(
    model_state_dict, key, pp_src_idx, group, dtype
):
    tensor = model_state_dict.get(key)
    if tensor is not None:
        tensor_shape = [tensor.shape]
    else:
        tensor_shape = [None]

    torch.distributed.broadcast_object_list(tensor_shape, pp_src_idx, group=group)

    if tensor_shape[0] is None:
        return None
    if torch.distributed.get_rank() != pp_src_idx:
        tensor = torch.empty(tensor_shape[0], dtype=dtype).cuda()

    torch.distributed.broadcast(tensor.contiguous(), pp_src_idx, group=group)
    return tensor


def pp_reshard_fn_qwen2_5(model_state_dict, pp_group, dtype):
    pp_first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
    pp_last_rank = parallel_state.get_pipeline_model_parallel_last_rank()

    key = "decoder.final_layernorm.weight"
    tensor = _gather_pp_group_tensor_and_reshard(
        model_state_dict, key, pp_last_rank, pp_group, dtype
    )
    if tensor is not None:
        model_state_dict[key] = tensor.clone()

    key = "decoder.final_layernorm.bias"
    tensor = _gather_pp_group_tensor_and_reshard(
        model_state_dict, key, pp_last_rank, pp_group, dtype
    )
    if tensor is not None:
        model_state_dict[key] = tensor.clone()

    key = "embedding.word_embeddings.weight"
    tensor = _gather_pp_group_tensor_and_reshard(
        model_state_dict, key, pp_first_rank, pp_group, dtype
    )
    if tensor is not None:
        model_state_dict[key] = tensor.clone()

    key = "output_layer.weight"
    tensor = _gather_pp_group_tensor_and_reshard(
        model_state_dict, key, pp_last_rank, pp_group, dtype
    )
    if tensor is not None:
        model_state_dict[key] = tensor.clone()
    return model_state_dict
