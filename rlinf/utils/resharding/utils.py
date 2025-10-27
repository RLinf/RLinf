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


def get_tp_reshard_fn(model_arch: str):
    if model_arch == "qwen2.5":
        return tp_reshard_fn_qwen2_5
    elif model_arch == "qwen3_moe":
        return tp_reshard_fn_qwen3_moe
    else:
        raise NotImplementedError(
            f"get_tp_reshard_fn for model_arch {model_arch} is not implemented"
        )


def get_tpe_reshard_fn(model_arch: str):
    if model_arch == "qwen3_moe":
        return tpe_reshard_fn_qwen3_moe
    else:
        raise NotImplementedError(
            f"get_tpe_reshard_fn for model_arch {model_arch} is not implemented"
        )


def get_pp_reshard_fn(model_arch: str):
    if model_arch == "qwen2.5":
        return pp_reshard_fn_qwen2_5
    elif model_arch == "qwen3_moe":
        return pp_reshard_fn_qwen3_moe
    else:
        raise NotImplementedError(
            f"get_pp_reshard_fn for model_arch {model_arch} is not implemented"
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


def tp_reshard_fn_qwen3_moe(model_state_dict, merge_factor, tp_group):
    for k, v in model_state_dict.items():
        if (
            "rotary_pos_emb.inv_freq" in k
            or "linear_qkv.layer_norm_weight" in k
            or "linear_fc1.layer_norm_weight" in k
            or "final_layernorm.weight" in k
            or "q_layernorm.weight" in k
            or "k_layernorm.weight" in k
            or "pre_mlp_layernorm.weight" in k
            or "router.weight" in k
        ):
            model_state_dict[k] = v.clone()
            continue

        if "self_attention.linear_proj.weight" in k:
            dim = 1
            model_state_dict[k] = _gather_tp_group_tensor_and_reshard(
                v, dim, merge_factor, tp_group
            )
    return model_state_dict


##############################
# tpe reshard fn implementation
##############################


def tpe_reshard_fn_qwen3_moe(
    model_state_dict, tpe_size, tpe_group, rollout_tp_size, dst_rank
):
    rollout_dp_rank, rollout_tp_rank = dst_rank

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
                [gate_value_slice[rollout_tp_rank], up_value_slice[rollout_tp_rank]],
                dim=0,
            ).contiguous()
            del gate_weight, up_weight, gate_value_slice, up_value_slice, value
        else:
            rollout_split_size = value.shape[dim] // rollout_tp_size
            value_slice = torch.split(value, rollout_split_size, dim=dim)
            model_state_dict[key] = value_slice[rollout_tp_rank].contiguous()
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


def pp_reshard_fn_qwen3_moe(model_state_dict, pp_group, dtype):
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
