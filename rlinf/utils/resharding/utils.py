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


import math

import torch

from rlinf.config import SupportedModel


def get_tp_gather_fn(model_type: str):
    model_type = SupportedModel(model_type)
    if model_type == SupportedModel.QWEN2_5:
        return tp_gather_fn_qwen2_5
    elif model_type == SupportedModel.QWEN3:
        return tp_gather_fn_qwen3_dense
    elif model_type == SupportedModel.QWEN3_MOE:
        return tp_gather_fn_qwen3_moe
    elif model_type in (SupportedModel.DEEPSEEK_V3,):
        return tp_gather_fn_deepseek_v3
    elif model_type == SupportedModel.GLM4_MOE_LITE:
        return tp_gather_fn_deepseek_v3  # GLM-4.7-Flash: reuse DeepSeek-V3 MLA+MoE layout (verify at e2e)
    else:
        raise NotImplementedError(
            f"get_tp_gather_fn for model_type {model_type} is not implemented"
        )


def get_tpe_gather_fn(model_type: str):
    model_type = SupportedModel(model_type)
    if model_type == SupportedModel.QWEN3_MOE:
        return tpe_gather_fn_qwen3_moe
    elif model_type in (SupportedModel.DEEPSEEK_V3,):
        return tpe_gather_fn_deepseek_v3
    elif model_type == SupportedModel.GLM4_MOE_LITE:
        return tpe_gather_fn_deepseek_v3  # GLM-4.7-Flash: reuse DeepSeek-V3 MLA+MoE layout (verify at e2e)
    else:
        raise NotImplementedError(
            f"get_tpe_gather_fn for model_type {model_type} is not implemented"
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
    assert full_size % world_size == 0, (
        f"reshard_tensor_by_rank: dim {dim} size {full_size} not divisible "
        f"by world_size {world_size} (rank={rank})"
    )
    shard_size = full_size // world_size
    return tensor.narrow(dim, rank * shard_size, shard_size).contiguous()


# The tp_gather_fn_* family is the gather half of the TP reshard: it
# all_gathers each parameter across the actor TP group and tags it with a
# narrow spec; narrow_to_target (mcore_weight_reshard.py) does the per-target
# slicing. All per-model knowledge (the name lists below) stays here, so that
# narrow stays model-agnostic. See gather_full_model for the spec format.
#
# ``cat`` labels the destination shard grid: "attn" / "dense" / "shared" for
# the DPA split; non-DPA callers pass "attn", the single grid.
#
# ``tp_gather_is_identity`` means the only target wants exactly this rank's
# shard, so every all_gather is skipped and the local shard is cloned instead.


def _gather_tp_param(value, dim, tp_group, cat, tp_gather_is_identity):
    """Gather one TP-sharded param to full and tag its narrow spec."""
    if tp_gather_is_identity:
        return value.clone(), None
    return all_gather_tensor(value, dim, tp_group), ("slice", dim, cat)


def _gather_fused_fc1(value, tp_group, tp_gather_is_identity):
    """Split a fused gate+up fc1 into the full (gate, up) pair across the TP group.

    Each rank's local fc1 weight is [gate_i; up_i] — two contiguous halves
    (mcore's apply_swiglu_sharded_factory chunks it the same way).
    """
    local_gate, local_up = torch.chunk(value, 2, dim=0)
    if tp_gather_is_identity:
        return local_gate.clone(), local_up.clone()
    return (
        all_gather_tensor(local_gate, 0, tp_group),
        all_gather_tensor(local_up, 0, tp_group),
    )


def _apply_layer_offset(key, layer_offset):
    """Shift the decoder layer number in ``key`` by ``layer_offset``.

    Only ``decoder.layers.*`` keys are shifted. A pipeline-stage offset applies
    to decoder layers alone; mcore leaves MTP layers un-offset, so ``mtp.*``
    keys keep their own numbering.
    """
    if layer_offset == 0 or not key.startswith("decoder.layers."):
        return key
    _, _, layer_num, remaining = key.split(".", 3)
    return f"decoder.layers.{int(layer_num) + layer_offset}.{remaining}"


def pp_merge_params(params, pp_group, layer_offset=0):
    """Merge a parameter dict across pipeline-parallel ranks.

    Every rank broadcasts the entries it owns and receives the rest, so the
    result holds the union of all stages' entries. Each source applies
    ``layer_offset`` to its own ``decoder.layers.*`` keys, so receivers never
    have to infer a peer's global layer number from a fixed stride, which
    breaks under uneven PP splits and unaligned buckets.

    Communication is one all_gather_object for metadata plus one broadcast per
    (source, dtype) flat buffer, instead of one all_gather per tensor.

    Returns the input unchanged when pp_size == 1. Otherwise every value,
    including this rank's own, is a view into a freshly allocated flat buffer.

    Args:
        params: local parameter dict (values on GPU).
        pp_group: pipeline-parallel process group.
        layer_offset: this rank's pipeline-stage layer offset, from mcore's
            get_transformer_layer_offset.
    """
    pp_size = torch.distributed.get_world_size(pp_group)
    if pp_size == 1:
        return params

    pp_rank = torch.distributed.get_rank(pp_group)
    group_ranks = torch.distributed.get_process_group_ranks(pp_group)
    device = torch.cuda.current_device()

    own_params = {
        _apply_layer_offset(key, layer_offset): value for key, value in params.items()
    }
    own_meta = {
        key: (tuple(value.shape), value.dtype) for key, value in own_params.items()
    }
    all_meta = [None] * pp_size
    torch.distributed.all_gather_object(all_meta, own_meta, group=pp_group)

    merged_params = {}
    for src_index, src_meta in enumerate(all_meta):
        # Broadcast order must be identical on every rank: each one groups
        # the same src_meta in insertion order and walks it the same way.
        # Never sort the keys or dedupe them through a set here; that silently
        # breaks broadcast symmetry across ranks.
        dtype_groups = {}
        for key, (shape, dtype) in src_meta.items():
            dtype_groups.setdefault(dtype, []).append((key, shape))

        for dtype, src_entries in dtype_groups.items():
            flat_numel = sum(math.prod(shape) for _, shape in src_entries)
            flat_buffer = torch.empty(flat_numel, dtype=dtype, device=device)
            if src_index == pp_rank:
                offset = 0
                for key, shape in src_entries:
                    entry_numel = math.prod(shape)
                    flat_buffer[offset : offset + entry_numel].copy_(
                        own_params[key].reshape(-1)
                    )
                    offset += entry_numel
            # torch.distributed.broadcast takes the source as a global rank.
            torch.distributed.broadcast(
                flat_buffer, src=group_ranks[src_index], group=pp_group
            )
            offset = 0
            for key, shape in src_entries:
                entry_numel = math.prod(shape)
                # A view, not a copy: the flat buffer stays alive as long as
                # any of its views, and is freed together with them.
                merged_params[key] = flat_buffer[offset : offset + entry_numel].view(
                    shape
                )
                offset += entry_numel
    return merged_params


def tp_gather_fn_qwen2_5(
    model_state_dict, tp_group, cat, split_fc1=True, tp_gather_is_identity=False
):
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

    full_state_dict = {}
    for k, v in model_state_dict.items():
        if any(param in k for param in param_skip_tp_reshard):
            # Replicated param: clone so it never aliases a live parameter.
            full_state_dict[k] = (v.clone(), None)
            continue

        if any(param in k for param in param_reshard_column_parallel_linear):
            dim = 0
        elif any(param in k for param in param_reshard_row_parallel_linear):
            dim = 1
        else:
            assert False, f"Unknown parameter: {k}"

        # Fused fc1: split gate/up per rank, then gather each half to full.
        # split_fc1=False (mcore-format inference reshard) keeps the fused
        # linear_fc1 key instead, through the generic path below.
        if split_fc1 and "linear_fc1" in k:
            gate, up = _gather_fused_fc1(v, tp_group, tp_gather_is_identity)
            spec = None if tp_gather_is_identity else ("slice", 0, cat)
            full_state_dict[k.replace("linear_fc1", "gate_proj")] = (gate, spec)
            full_state_dict[k.replace("linear_fc1", "up_proj")] = (up, spec)
            continue

        full_state_dict[k] = _gather_tp_param(
            v, dim, tp_group, cat, tp_gather_is_identity
        )

    return full_state_dict


def tp_gather_fn_qwen3_dense(
    model_state_dict, tp_group, cat, split_fc1=True, tp_gather_is_identity=False
):
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

    full_state_dict = {}
    for k, v in model_state_dict.items():
        if any(param in k for param in param_skip_tp_reshard):
            # Replicated param: clone so it never aliases a live parameter.
            full_state_dict[k] = (v.clone(), None)
            continue

        if any(param in k for param in param_reshard_column_parallel_linear):
            dim = 0
        elif any(param in k for param in param_reshard_row_parallel_linear):
            dim = 1
        else:
            assert False, f"Unknown parameter: {k}"

        # Fused fc1: split gate/up per rank, then gather each half to full.
        # split_fc1=False (mcore-format inference reshard) keeps the fused
        # linear_fc1 key instead, through the generic path below.
        if split_fc1 and "linear_fc1" in k:
            gate, up = _gather_fused_fc1(v, tp_group, tp_gather_is_identity)
            spec = None if tp_gather_is_identity else ("slice", 0, cat)
            full_state_dict[k.replace("linear_fc1", "gate_proj")] = (gate, spec)
            full_state_dict[k.replace("linear_fc1", "up_proj")] = (up, spec)
            continue

        full_state_dict[k] = _gather_tp_param(
            v, dim, tp_group, cat, tp_gather_is_identity
        )

    return full_state_dict


def tp_gather_fn_qwen3_moe(
    model_state_dict, tp_group, cat, split_fc1=True, tp_gather_is_identity=False
):
    # split_fc1 is unused: qwen3_moe has no dense MLP, and routed-expert
    # fc1/fc2 never reach this fn (they live in expert_params). The parameter
    # only keeps the tp_gather_fn_* signature uniform.
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
    ]

    # Parameters that need to be gathered on dim=1
    param_reshard_row_parallel_linear = [
        "self_attention.linear_proj.weight",
    ]

    full_state_dict = {}
    for k, v in model_state_dict.items():
        if any(param in k for param in param_skip_tp_reshard):
            # Replicated param: clone so it never aliases a live parameter.
            full_state_dict[k] = (v.clone(), None)
            continue

        if any(param in k for param in param_reshard_column_parallel_linear):
            dim = 0
        elif any(param in k for param in param_reshard_row_parallel_linear):
            dim = 1
        else:
            assert False, f"Unknown parameter: {k}"

        full_state_dict[k] = _gather_tp_param(
            v, dim, tp_group, cat, tp_gather_is_identity
        )

    return full_state_dict


def tp_gather_fn_deepseek_v3(
    model_state_dict, tp_group, cat, split_fc1=True, tp_gather_is_identity=False
):
    # DeepSeek-V3 / Kimi K2 / GLM-4.7-Flash text backbone: MLA attention + MoE.
    # This fn covers the MLA projections and the dense/shared MLP; routed
    # experts live in expert_params and never reach it.

    # Replicated / non-TP params: clone, no gather. q_down_proj and
    # kv_down_proj are replicated in TE even though the spec marks them
    # column-parallel (confirmed by dump: 1536 / 576 are not divided by TP).
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
    ]

    param_reshard_column_parallel_linear = [
        "word_embeddings.weight",
        "output_layer.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
        "mlp.linear_fc1.weight",
        "shared_experts.linear_fc1.weight",
        # eh_proj is column-parallel in mcore. Only the mcore-format target
        # reaches here; the sglang-format case is intercepted below.
        "eh_proj.weight",
    ]

    param_reshard_row_parallel_linear = [
        "self_attention.linear_proj.weight",
        "mlp.linear_fc2.weight",
        "shared_experts.linear_fc2.weight",
    ]

    full_state_dict = {}
    for k, v in model_state_dict.items():
        if any(param in k for param in param_skip_tp_reshard):
            # Replicated param: clone so it never aliases a live parameter.
            full_state_dict[k] = (v.clone(), None)
            continue
        if k.endswith("eh_proj.weight") and split_fc1:
            # sglang target: deepseek_nextn uses a replicated nn.Linear, so
            # gather to full and never slice. The identity fast path does not
            # apply, since the target wants the full tensor rather than this
            # rank's shard. An mcore-format target falls through instead.
            full_state_dict[k] = (all_gather_tensor(v, 0, tp_group), None)
            continue
        if any(param in k for param in param_reshard_column_parallel_linear):
            dim = 0
        elif any(param in k for param in param_reshard_row_parallel_linear):
            dim = 1
        else:
            assert False, f"Unknown parameter: {k}"

        # Unified fused fc1 (dense + shared): split gate/up per rank, then
        # gather each half to full. split_fc1=False (mcore-format inference
        # reshard) keeps the fused key instead.
        if split_fc1 and "linear_fc1" in k:
            gate, up = _gather_fused_fc1(v, tp_group, tp_gather_is_identity)
            spec = None if tp_gather_is_identity else ("slice", 0, cat)
            full_state_dict[k.replace("linear_fc1", "gate_proj")] = (gate, spec)
            full_state_dict[k.replace("linear_fc1", "up_proj")] = (up, spec)
            continue

        full_state_dict[k] = _gather_tp_param(
            v, dim, tp_group, cat, tp_gather_is_identity
        )
    return full_state_dict


##############################
# tpe reshard fn implementation
##############################


# The tpe_gather_fn_* family is the gather half of the routed-expert reshard:
# it gathers expert weights across the expert-tensor-parallel (tpe) group and
# tags each with a narrow spec, always under the "expert_tp" grid.
#
#   fc1: full fused [gate; up], tagged ("fused_glu", "expert_tp"). The key
#       stays fused; convert_fn renames it to gate/up later.
#   fc2: full row-parallel tensor, tagged ("slice", 1, "expert_tp").
#
# The TP identity fast path does not apply here, because the tpe all_gather is
# a separate collective under separate conditions.


def tpe_gather_fn_qwen3_moe(expert_params, tpe_size, tpe_group):
    full_params = {}
    for key, value in expert_params.items():
        if "linear_fc1.weight" in key:
            if tpe_size == 1:
                # No tpe split: the local value is already the full fused
                # [gate; up].
                full_params[key] = (value, ("fused_glu", "expert_tp"))
                continue
            value = all_gather_tensor(value, 0, tpe_group)
            # Each tpe rank's fused shard is [gate_i; up_i], so the gathered
            # tensor is [g0; u0; g1; u1; ...]. Regroup it into [full_gate;
            # full_up] so that narrow can chunk the two halves apart.
            tpe_split_size = value.shape[0] // tpe_size

            gate_proj_shards = []
            up_proj_shards = []
            for weight in torch.split(value, tpe_split_size, dim=0):
                gate_proj_shard, up_proj_shard = torch.chunk(weight, 2, dim=0)
                gate_proj_shards.append(gate_proj_shard)
                up_proj_shards.append(up_proj_shard)

            gate_weight = torch.cat(gate_proj_shards, dim=0)
            up_weight = torch.cat(up_proj_shards, dim=0)
            full_params[key] = (
                torch.cat([gate_weight, up_weight], dim=0),
                ("fused_glu", "expert_tp"),
            )
        elif "linear_fc2.weight" in key:
            if tpe_size != 1:
                value = all_gather_tensor(value, 1, tpe_group)
            full_params[key] = (value, ("slice", 1, "expert_tp"))
        else:
            # Neither expert fc1 nor fc2 (none exist in practice today):
            # clone so it never aliases a live parameter, since narrow passes
            # None-spec entries straight through.
            full_params[key] = (value.clone(), None)
    return full_params


def tpe_gather_fn_deepseek_v3(expert_params, tpe_size, tpe_group):
    # Same logic as tpe_gather_fn_qwen3_moe. DeepSeek-V3 routed experts
    # usually run with tpe_size == 1, so the gather is skipped, but fc1 and
    # fc2 are still tagged for narrow's per-target rollout-TP slicing.
    full_params = {}
    for key, value in expert_params.items():
        if "linear_fc1.weight" in key:
            if tpe_size == 1:
                # No tpe split: the local value is already the full fused
                # [gate; up].
                full_params[key] = (value, ("fused_glu", "expert_tp"))
                continue
            value = all_gather_tensor(value, 0, tpe_group)
            # Each tpe rank's fused shard is [gate_i; up_i], so the gathered
            # tensor is [g0; u0; g1; u1; ...]. Regroup it into [full_gate;
            # full_up] so that narrow can chunk the two halves apart.
            tpe_split_size = value.shape[0] // tpe_size

            gate_proj_shards = []
            up_proj_shards = []
            for weight in torch.split(value, tpe_split_size, dim=0):
                gate_proj_shard, up_proj_shard = torch.chunk(weight, 2, dim=0)
                gate_proj_shards.append(gate_proj_shard)
                up_proj_shards.append(up_proj_shard)

            gate_weight = torch.cat(gate_proj_shards, dim=0)
            up_weight = torch.cat(up_proj_shards, dim=0)
            full_params[key] = (
                torch.cat([gate_weight, up_weight], dim=0),
                ("fused_glu", "expert_tp"),
            )
        elif "linear_fc2.weight" in key:
            if tpe_size != 1:
                value = all_gather_tensor(value, 1, tpe_group)
            full_params[key] = (value, ("slice", 1, "expert_tp"))
        else:
            # Neither expert fc1 nor fc2 (none exist in practice today):
            # clone so it never aliases a live parameter, since narrow passes
            # None-spec entries straight through.
            full_params[key] = (value.clone(), None)
    return full_params
