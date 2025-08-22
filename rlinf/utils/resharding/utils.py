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

import torch
from megatron.core import parallel_state


def get_convert_fn(model_arch: str):
    if model_arch == "qwen2.5":
        return convert_fn_qwen2_5
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


def _convert_qwen2_5_weight(
    para_skip_convert,
    name: str,
    weight: torch.Tensor,
    q_slice,
    k_slice,
    v_slice,
    tp_size,
    target_tp,
    head_size,
    hidden_size,
) -> torch.Tensor:
    """
    Convert weight from megatron to vLLM format
    """
    for skip in para_skip_convert:
        if skip in name:
            return weight

    if "linear_qkv" in name:
        if "weight" in name:
            weight = weight.reshape(-1, head_size, hidden_size)
            max_dim0 = weight.size(0)
            q_slice = q_slice[q_slice < max_dim0]
            k_slice = k_slice[k_slice < max_dim0]
            v_slice = v_slice[v_slice < max_dim0]

            q = weight[q_slice]
            k = weight[k_slice]
            v = weight[v_slice]

            return torch.cat([q, k, v], dim=0).reshape((-1, weight.size(-1)))
        elif "bias" in name:
            weight = weight.reshape((-1, head_size))
            max_dim0 = weight.size(0)

            q_slice = q_slice[q_slice < max_dim0]
            k_slice = k_slice[k_slice < max_dim0]
            v_slice = v_slice[v_slice < max_dim0]

            qbias = weight[q_slice]
            kbias = weight[k_slice]
            vbias = weight[v_slice]

            return torch.cat([qbias, kbias, vbias], dim=0).reshape((-1))
        else:
            raise RuntimeError(
                f"convert_weight: Unknown weight type in linear_qkv: {name}"
            )
    elif "linear_fc1" in name:
        # reshape gate & up
        split_num = tp_size // target_tp * 2
        shard = weight.reshape((split_num, -1, weight.shape[-1]))
        gate_slice = torch.arange(0, split_num, 2, dtype=torch.long)
        up_slice = torch.arange(1, split_num, 2, dtype=torch.long)

        return torch.cat([shard[gate_slice], shard[up_slice]], dim=0).reshape(
            (
                -1,
                weight.shape[-1],
            )
        )
    else:
        raise RuntimeError(f"convert_weight: Unknown weight name: {name}")


def extract_layer_info(s):
    pattern = r"layers\.(\d+)\.(.+)"
    match = re.search(pattern, s)
    if match:
        return match.group(1), match.group(2)
    return None, None


def mega_name_to_vllm_name_qwen2(name: str):
    if "embedding.word_embeddings.weight" in name:
        return "model.embed_tokens.weight"
    if "decoder.final_layernorm.weight" in name:
        return "model.norm.weight"
    if "output_layer.weight" in name:
        return "lm_head.weight"
    layer_id, suffix = extract_layer_info(name)

    assert layer_id is not None, f"Cannot extract layer info from {name}"
    result_pattern = "model.layers.{}.{}"
    nmap = {
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "self_attention.linear_qkv.weight": "self_attn.qkv_proj.weight",
        "self_attention.linear_qkv.bias": "self_attn.qkv_proj.bias",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "mlp.linear_fc1.weight": "mlp.gate_up_proj.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    }

    assert suffix in nmap, f"Cannot find mapping for {suffix}"

    return result_pattern.format(layer_id, nmap[suffix])


def convert_fn_qwen2_5(model_state_dict, config):
    para_skip_convert = [
        "word_embeddings",
        "layer_norm_weight",
        "linear_proj",
        "linear_fc2",
        "final_layernorm",
        "output_layer",
        "rotary_pos_emb.inv_freq",
        "linear_qkv.layer_norm_weight",
        "mlp.linear_fc1.layer_norm_weight",
        "final_layernorm.weight",
    ]

    tp_size = config.model_config.tensor_model_parallel_size
    target_tp = config.reshard_tp_size

    hidden_size = config.model_config.hidden_size
    head_size = (
        config.model_config.kv_channels
        or hidden_size // config.model_config.num_attention_heads
    )
    head_num = config.model_config.num_attention_heads
    num_query_groups = config.model_config.num_query_groups or head_num

    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    q_slice = torch.cat(
        [
            torch.arange(
                (heads_per_group + 2) * i,
                (heads_per_group + 2) * i + heads_per_group,
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    converted_model_state_dict = {}
    for name, param in model_state_dict.items():
        converted_model_state_dict[mega_name_to_vllm_name_qwen2(name)] = (
            _convert_qwen2_5_weight(
                para_skip_convert,
                name,
                param,
                q_slice,
                k_slice,
                v_slice,
                tp_size,
                target_tp,
                head_size,
                hidden_size,
            )
        )

    return converted_model_state_dict


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
