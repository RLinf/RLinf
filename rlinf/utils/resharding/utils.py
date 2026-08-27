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

from rlinf.config import SupportedModel


def get_tp_reshard_fn(model_type: str):
    model_type = SupportedModel(model_type)
    if model_type == SupportedModel.QWEN2_5:
        return tp_reshard_fn_qwen2_5
    elif model_type == SupportedModel.QWEN3:
        return tp_reshard_fn_qwen3_dense
    elif model_type == SupportedModel.QWEN3_MOE:
        return tp_reshard_fn_qwen3_moe
    elif model_type in (SupportedModel.DEEPSEEK_V3, SupportedModel.GLM4_MOE_LITE):
        return tp_reshard_fn_deepseek_v3
    else:
        raise NotImplementedError(
            f"get_tp_reshard_fn for model_type {model_type} is not implemented"
        )


def get_tpe_reshard_fn(model_type: str):
    model_type = SupportedModel(model_type)
    if model_type == SupportedModel.QWEN3_MOE:
        return tpe_reshard_fn_qwen3_moe
    elif model_type in (SupportedModel.DEEPSEEK_V3, SupportedModel.GLM4_MOE_LITE):
        return tpe_reshard_fn_deepseek_v3
    else:
        raise NotImplementedError(
            f"get_tpe_reshard_fn for model_type {model_type} is not implemented"
        )


def get_ep_reshard_fn(model_type: str):
    model_type = SupportedModel(model_type)
    if model_type in (SupportedModel.DEEPSEEK_V3, SupportedModel.GLM4_MOE_LITE):
        return ep_reshard_fn_deepseek_v3
    else:
        raise NotImplementedError(
            f"get_ep_reshard_fn for model_type {model_type} is not implemented"
        )


def get_pp_reshard_fn(model_type: str):
    model_type = SupportedModel(model_type)
    if model_type == SupportedModel.QWEN2_5:
        return pp_reshard_fn_qwen2_5
    elif model_type == SupportedModel.QWEN3:
        return pp_reshard_fn_qwen3_dense
    elif model_type == SupportedModel.QWEN3_MOE:
        return pp_reshard_fn_qwen3_moe
    elif model_type in (SupportedModel.DEEPSEEK_V3, SupportedModel.GLM4_MOE_LITE):
        return pp_reshard_fn_deepseek_v3
    else:
        raise NotImplementedError(
            f"get_pp_reshard_fn for model_type {model_type} is not implemented"
        )


##############################
# tp reshard fn implementation
##############################


def all_gather_tensor(tensor, dim, group):
    """All-gather tensor across the given process group, cat along dim.

    Uses group.world_size (not an external merge_factor) to size the output
    list, avoiding mismatch errors.
    """
    world_size = torch.distributed.get_world_size(group)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=dim)


def reshard_tensor_by_rank(tensor, dim, rank, world_size):
    """Slice tensor on dim to rank's 1/world_size share.

    narrow() returns a view; for a dim>0 slice (row-parallel weights) the view
    is non-contiguous, which P2P send (collective_group._check_tensor_contiguous)
    rejects. .contiguous() makes row-parallel slices contiguous; for dim=0
    slices (column-parallel) and 1-D tensors narrow already yields a contiguous
    block so .contiguous() is a no-op.
    """
    full_size = tensor.shape[dim]
    shard_size = full_size // world_size
    return tensor.narrow(dim, rank * shard_size, shard_size).contiguous()


def gather_and_reshard_tensor(tensor, dim, group, dst_rank, dst_world_size):
    """All-gather across group, then narrow to dst_rank's 1/dst_world_size share.

    dst_rank/dst_world_size are required: the caller always specifies the target
    shard. For no-slice (all_gather result is final), pass dst_rank=0,
    dst_world_size=1.
    """
    full = all_gather_tensor(tensor, dim, group)
    result = reshard_tensor_by_rank(full, dim, dst_rank, dst_world_size)
    return result


def reshard_fused_fc1(tensor, tp_group, dst_rank, dst_world_size):
    """Reshard a fused gate+up fc1 (TE GLU stride-2): per-rank split gate/up,
    separate all_gather + slice. Returns (gate, up). Works for any model
    where the actor rank holds [gate_i, up_i] interleaved (stride=2)."""
    local_gate, local_up = torch.chunk(tensor, 2, dim=0)
    gate = gather_and_reshard_tensor(local_gate, 0, tp_group, dst_rank, dst_world_size)
    up = gather_and_reshard_tensor(local_up, 0, tp_group, dst_rank, dst_world_size)
    return gate, up


def tp_reshard_fn_qwen2_5(model_state_dict, tp_group, dst_rank, dst_world_size):
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

    for k, v in list(model_state_dict.items()):
        if any(param in k for param in param_skip_tp_reshard):
            model_state_dict[k] = v.clone()
            continue

        if any(param in k for param in param_reshard_column_parallel_linear):
            dim = 0
        elif any(param in k for param in param_reshard_row_parallel_linear):
            dim = 1
        else:
            assert False, f"Unknown parameter: {k}"

        # Fused fc1: per-rank split gate/up, separate all_gather + slice.
        # Output gate_proj/up_proj keys -> convertor SPLIT_NONE.
        if "linear_fc1" in k:
            gate, up = reshard_fused_fc1(v, tp_group, dst_rank, dst_world_size)
            model_state_dict[k.replace("linear_fc1", "gate_proj")] = gate
            model_state_dict[k.replace("linear_fc1", "up_proj")] = up
            del model_state_dict[k]
            continue

        model_state_dict[k] = gather_and_reshard_tensor(
            v, dim, tp_group, dst_rank, dst_world_size
        )

    return model_state_dict


def tp_reshard_fn_qwen3_dense(model_state_dict, tp_group, dst_rank, dst_world_size):
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

    for k, v in list(model_state_dict.items()):
        if any(param in k for param in param_skip_tp_reshard):
            model_state_dict[k] = v.clone()
            continue

        if any(param in k for param in param_reshard_column_parallel_linear):
            dim = 0
        elif any(param in k for param in param_reshard_row_parallel_linear):
            dim = 1
        else:
            assert False, f"Unknown parameter: {k}"

        # Fused fc1: per-rank split gate/up, separate all_gather + slice.
        if "linear_fc1" in k:
            gate, up = reshard_fused_fc1(v, tp_group, dst_rank, dst_world_size)
            model_state_dict[k.replace("linear_fc1", "gate_proj")] = gate
            model_state_dict[k.replace("linear_fc1", "up_proj")] = up
            del model_state_dict[k]
            continue

        model_state_dict[k] = gather_and_reshard_tensor(
            v, dim, tp_group, dst_rank, dst_world_size
        )

    return model_state_dict


def tp_reshard_fn_qwen3_moe(model_state_dict, tp_group, dst_rank, dst_world_size):
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

    for k, v in list(model_state_dict.items()):
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

        model_state_dict[k] = gather_and_reshard_tensor(
            v, dim, tp_group, dst_rank, dst_world_size
        )

    return model_state_dict


def tp_reshard_fn_deepseek_v3(model_state_dict, tp_group, dst_rank, dst_world_size):
    # DeepSeek-V3 / Kimi K2 / GLM-4.7-Flash text backbone: MLA attention + MoE.
    # TP reshard handles MLA projections and dense/shared MLP; routed expert
    # fc1/fc2 are skipped here (sliced by tpe_reshard_fn_deepseek_v3).
    # dst_rank/dst_world_size: always required — the gathered tensor is sliced
    # to dst_rank's 1/dst_world_size share. For no-slice (non-DPA where the
    # subgroup all_gather already produces the right shard), pass (0, 1).

    # Replicated / non-TP params: clone (no gather).
    # q_down_proj / kv_down_proj are replicated in TE despite the spec marking them
    # column-parallel (dump-confirmed: 1536 / 576 are not divided by TP).
    param_skip_tp_reshard = [
        "linear_q_up_proj.layer_norm_weight",
        "linear_kv_up_proj.layer_norm_weight",
        "linear_q_down_proj.weight",
        "linear_kv_down_proj.weight",
        "input_layernorm.weight",
        "pre_mlp_layernorm.weight",
        "linear_fc1.layer_norm_weight",
        "final_layernorm.weight",
        "router.weight",
        "router.expert_bias",
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
    ]

    param_reshard_column_parallel_linear = [
        "word_embeddings.weight",
        "output_layer.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
        "mlp.linear_fc1.weight",
        "shared_experts.linear_fc1.weight",
    ]

    param_reshard_row_parallel_linear = [
        "self_attention.linear_proj.weight",
        "mlp.linear_fc2.weight",
        "shared_experts.linear_fc2.weight",
    ]

    param_reshard_skip_weight = [
        "linear_fc1.weight",
        "linear_fc2.weight",
    ]

    for k, v in list(model_state_dict.items()):
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
        # Unified fused fc1 (dense + shared): per-rank split gate/up, separate
        # all_gather + slice. Output gate_proj/up_proj keys -> convertor
        # SPLIT_NONE. Routed-expert fc1 (local_experts) excluded (block, ETP=1).
        if "linear_fc1" in k and "local_experts" not in k:
            gate, up = reshard_fused_fc1(v, tp_group, dst_rank, dst_world_size)
            model_state_dict[k.replace("linear_fc1", "gate_proj")] = gate
            model_state_dict[k.replace("linear_fc1", "up_proj")] = up
            del model_state_dict[k]
            continue
        model_state_dict[k] = gather_and_reshard_tensor(
            v, dim, tp_group, dst_rank, dst_world_size
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
            value = all_gather_tensor(value, dim, tpe_group)
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


def tpe_reshard_fn_deepseek_v3(
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
            value = all_gather_tensor(value, dim, tpe_group)
        if dim == 0:
            # fc1: split fused gate+up, then slice per rollout tp rank.
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


def ep_reshard_fn_deepseek_v3(
    expert_params, rollout_ep_size, dst_ep_rank, num_moe_experts
):
    """EP-distribute: select this rollout ep rank's subset of FULL experts.
    Used when rollout_ep_size > 1.
    i.e. `...mlp.experts.local_experts.{G}.linear_fc1.weight` with GLOBAL expert
    id G in [0, num_moe_experts).
    Extensible to dp>1+ep: generalize dst_ep_rank to a (dp_rank, ep_rank) pair
    and recompute `start`/`end` here (the only place that needs to change).
    """
    experts_per_rank = num_moe_experts // rollout_ep_size
    start = dst_ep_rank * experts_per_rank
    end = start + experts_per_rank
    out = {}
    for key, val in expert_params.items():
        if "local_experts." not in key:
            continue
        g = int(key.split("local_experts.")[1].split(".")[0])
        if start <= g < end:
            out[key] = val
    return out


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


def pp_reshard_fn_deepseek_v3(model_state_dict, pp_group, dtype):
    """Reshard pipeline parallel weights for DeepSeek-V3 models."""
    return _pp_reshard_fn_Qwen_model(model_state_dict, pp_group, dtype)
