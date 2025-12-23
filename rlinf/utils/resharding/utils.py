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


import torch
from megatron.core import parallel_state

from rlinf.config import SupportedModel, get_supported_model


def get_tp_reshard_fn(model_type: str):
    model_type = get_supported_model(model_type)
    if model_type == SupportedModel.QWEN2_5:
        return tp_reshard_fn_qwen2_5
    elif model_type == SupportedModel.QWEN3:
        return tp_reshard_fn_qwen3_dense
    elif model_type == SupportedModel.QWEN3_MOE:
        return tp_reshard_fn_qwen3_moe
    else:
        raise NotImplementedError(
            f"get_tp_reshard_fn for model_type {model_type} is not implemented"
        )


def get_tpe_reshard_fn(model_type: str):
    model_type = get_supported_model(model_type)
    if model_type == SupportedModel.QWEN3_MOE:
        return tpe_reshard_fn_qwen3_moe
    else:
        raise NotImplementedError(
            f"get_tpe_reshard_fn for model_type {model_type} is not implemented"
        )


def get_pp_reshard_fn(model_type: str):
    model_type = get_supported_model(model_type)
    if model_type == SupportedModel.QWEN2_5:
        return pp_reshard_fn_qwen2_5
    elif model_type == SupportedModel.QWEN3:
        return pp_reshard_fn_qwen3_dense
    elif model_type == SupportedModel.QWEN3_MOE:
        return pp_reshard_fn_qwen3_moe
    else:
        raise NotImplementedError(
            f"get_pp_reshard_fn for model_type {model_type} is not implemented"
        )


##############################
# tp reshard fn implementation
##############################


def _gather_tp_group_tensor_and_reshard(tensor, dim, merge_factor, tp_group):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(merge_factor)]

    torch.distributed.all_gather(gathered_tensors, tensor, group=tp_group)

    resharded_tensor = torch.cat(gathered_tensors, dim=dim)

    return resharded_tensor


def tp_reshard_fn_qwen2_5(model_state_dict, merge_factor, tp_group):
    # Parameters that should skip TP resharding (just clone)
    param_skip_tp_reshard = [
        "linear_qkv.layer_norm_weight",
        "mlp.linear_fc1.layer_norm_weight",
        "final_layernorm.weight",
    ]

    # Parameters that need to be gathered on dim=0
    param_reshard_column_parallel_linear = [
        "word_embeddings.weight",
        "output_layer.weight",
        "self_attention.linear_qkv.weight",
        "self_attention.linear_qkv.bias",
        "mlp.linear_fc1.weight",
    ]

    # Parameters that need to be gathered on dim=1
    param_reshard_row_parallel_linear = [
        "self_attention.linear_proj.weight",
        "mlp.linear_fc2.weight",
    ]

    for k, v in model_state_dict.items():
        if any(param in k for param in param_skip_tp_reshard):
            model_state_dict[k] = v.clone()
            continue

        if any(param in k for param in param_reshard_column_parallel_linear):
            dim = 0
        elif any(param in k for param in param_reshard_row_parallel_linear):
            dim = 1
        else:
            assert False, f"Unknown parameter: {k}"

        model_state_dict[k] = _gather_tp_group_tensor_and_reshard(
            v, dim, merge_factor, tp_group
        )

    return model_state_dict


def tp_reshard_fn_qwen3_dense(model_state_dict, merge_factor, tp_group):
    # Parameters that should skip TP resharding (just clone)
    param_skip_tp_reshard = [
        "linear_qkv.layer_norm_weight",
        "linear_fc1.layer_norm_weight",
        "final_layernorm.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
        "pre_mlp_layernorm.weight",
        "router.weight",
    ]

    # Parameters that need to be gathered on dim=0
    param_reshard_column_parallel_linear = [
        "word_embeddings.weight",
        "output_layer.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
    ]

    # Parameters that need to be gathered on dim=1
    param_reshard_row_parallel_linear = [
        "self_attention.linear_proj.weight",
        "mlp.linear_fc2.weight",
    ]

    for k, v in model_state_dict.items():
        if any(param in k for param in param_skip_tp_reshard):
            model_state_dict[k] = v.clone()
            continue

        if any(param in k for param in param_reshard_column_parallel_linear):
            dim = 0
        elif any(param in k for param in param_reshard_row_parallel_linear):
            dim = 1
        else:
            assert False, f"Unknown parameter: {k}"

        model_state_dict[k] = _gather_tp_group_tensor_and_reshard(
            v, dim, merge_factor, tp_group
        )

    return model_state_dict


def tp_reshard_fn_qwen3_moe(model_state_dict, merge_factor, tp_group):
    # Parameters that should skip TP resharding (just clone)
    param_skip_tp_reshard = [
        "linear_qkv.layer_norm_weight",
        "linear_fc1.layer_norm_weight",
        "final_layernorm.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
        "pre_mlp_layernorm.weight",
        "router.weight",
    ]

    # MoE model resharding the mlp weight in tpe_reshard_fn
    # Parameters that need to be gathered on dim=0
    param_reshard_column_parallel_linear = [
        "word_embeddings.weight",
        "output_layer.weight",
        "self_attention.linear_qkv.weight",
    ]

    # Parameters that need to be gathered on dim=1
    param_reshard_row_parallel_linear = [
        "self_attention.linear_proj.weight",
    ]

    # Parameters that need to skip in tp resharding
    param_reshard_skip_weight = [
        "linear_fc1.weight",
        "linear_fc2.weight",
    ]

    for k, v in model_state_dict.items():
        if any(param in k for param in param_skip_tp_reshard):
            model_state_dict[k] = v.clone()
            continue

        if any(param in k for param in param_reshard_column_parallel_linear):
            dim = 0
        elif any(param in k for param in param_reshard_row_parallel_linear):
            dim = 1
        elif any(param in k for param in param_reshard_skip_weight):
            continue
        else:
            assert False, f"Unknown parameter: {k}"

        model_state_dict[k] = _gather_tp_group_tensor_and_reshard(
            v, dim, merge_factor, tp_group
        )

    return model_state_dict


##############################
# tpe reshard fn implementation
##############################


def tpe_reshard_fn_qwen3_moe(
    model_state_dict, tpe_size, tpe_group, rollout_tp_size, dst_tp_rank
):
    for key, value in model_state_dict.items():
        if "linear_fc1.weight" in key:
            dim = 0
        elif "linear_fc2.weight" in key:
            dim = 1
        else:
            continue
        if tpe_size != 1:
            value = _gather_tp_group_tensor_and_reshard(value, dim, tpe_size, tpe_group)
        if dim == 0:
            # for the fc1 weight, we need to split it into two parts gate weight and up weight
            tpe_split_size = value.shape[dim] // tpe_size
            tpe_value_slice = torch.split(value, tpe_split_size, dim=dim)

            gate_proj_shards = []
            up_proj_shards = []

            for i, weight in enumerate(tpe_value_slice):
                weight_chunk = torch.chunk(weight, 2, dim=0)
                gate_proj_shards.append(weight_chunk[0])
                up_proj_shards.append(weight_chunk[1])

            gate_weight = torch.cat(gate_proj_shards, dim=dim)
            up_weight = torch.cat(up_proj_shards, dim=dim)

            rollout_split_size = gate_weight.shape[dim] // rollout_tp_size
            gate_value_slice = torch.split(gate_weight, rollout_split_size, dim=dim)
            up_value_slice = torch.split(up_weight, rollout_split_size, dim=dim)

            model_state_dict[key] = torch.cat(
                [gate_value_slice[dst_tp_rank], up_value_slice[dst_tp_rank]],
                dim=0,
            ).contiguous()
            del gate_weight, up_weight, gate_value_slice, up_value_slice, value
        else:
            rollout_split_size = value.shape[dim] // rollout_tp_size
            value_slice = torch.split(value, rollout_split_size, dim=dim)
            model_state_dict[key] = value_slice[dst_tp_rank].contiguous()
            del value

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


def gather_pp_group_tensor_and_reshard(
    model_state_dict, keys_with_ranks, pp_group, dtype
):
    """Helper function to reshard multiple keys."""
    for key, target_rank in keys_with_ranks:
        tensor = _gather_pp_group_tensor_and_reshard(
            model_state_dict, key, target_rank, pp_group, dtype
        )
        if tensor is not None:
            model_state_dict[key] = tensor.clone()
    return model_state_dict


def _pp_reshard_fn_Qwen_model(model_state_dict, pp_group, dtype):
    """Common resharding logic for Qwen models."""
    pp_first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
    pp_last_rank = parallel_state.get_pipeline_model_parallel_last_rank()

    keys_with_ranks = [
        ("embedding.word_embeddings.weight", pp_first_rank),
        ("decoder.final_layernorm.weight", pp_last_rank),
        ("decoder.final_layernorm.bias", pp_last_rank),
        ("output_layer.weight", pp_last_rank),
    ]

    return gather_pp_group_tensor_and_reshard(
        model_state_dict, keys_with_ranks, pp_group, dtype
    )


def pp_reshard_fn_qwen2_5(model_state_dict, pp_group, dtype):
    """Reshard pipeline parallel weights for Qwen2.5 models."""
    return _pp_reshard_fn_Qwen_model(model_state_dict, pp_group, dtype)


def pp_reshard_fn_qwen3_dense(model_state_dict, pp_group, dtype):
    """Reshard pipeline parallel weights for Qwen3 dense models."""
    return _pp_reshard_fn_Qwen_model(model_state_dict, pp_group, dtype)


def pp_reshard_fn_qwen3_moe(model_state_dict, pp_group, dtype):
    """Reshard pipeline parallel weights for Qwen3 MoE models."""
    return _pp_reshard_fn_Qwen_model(model_state_dict, pp_group, dtype)
