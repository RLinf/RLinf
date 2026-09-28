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


import functools
import inspect

import torch
from megatron.core import parallel_state
from megatron.core.transformer.transformer_layer import (
    get_transformer_layer_offset,
)
from megatron.training.training import unwrap_model

from .reshard_config import ReshardConfig
from .utils import all_gather_tensor, pp_merge_params, reshard_tensor_by_rank


@functools.cache
def _layer_offset_has_vp_stage():
    # Probed on first use rather than at import: the docs build mocks megatron,
    # and a mocked function has no signature to inspect.
    return "vp_stage" in inspect.signature(get_transformer_layer_offset).parameters


def _layer_offset(config, vp_stage):
    """Layer offset of one virtual chunk, across megatron-core versions.

    megatron-core 0.15 onwards takes vp_stage as an argument. Older versions
    read the virtual pipeline rank from global parallel state instead, so they
    only report the offset of the chunk that is currently active.
    """
    if _layer_offset_has_vp_stage():
        return get_transformer_layer_offset(config, vp_stage=vp_stage)
    return get_transformer_layer_offset(config)


class MegatronCoreWeightReshard:
    def __init__(self, config: ReshardConfig):
        self.config = config
        self.bucket_capacity = self.config.bucket_capacity
        if not _layer_offset_has_vp_stage():
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
            assert vp_size is None or vp_size == 1, (
                f"virtual pipeline size {vp_size} requires megatron-core >= 0.15; "
                "this version cannot report the layer offset of a chunk other "
                "than the one currently active in parallel state"
            )
        # When the only target wants exactly this rank's TP shard, the gather
        # clones local shards instead of all_gathering. Set by the actor worker.
        self.tp_gather_is_identity = False

    def set_tp_gather_is_identity(self, tp_gather_is_identity):
        self.tp_gather_is_identity = tp_gather_is_identity

    def divide_model_to_bucket(self, model):
        bucket_capacity = self.bucket_capacity
        model_bucket_list = []
        model_bucket = {}

        current_capacity = 0
        model = unwrap_model(model)
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        if vp_size is None:
            for key, val in model[0].state_dict().items():
                if "_extra_state" in key:
                    continue
                model_bucket[key] = val

                if "decoder.layers" in key:
                    current_capacity += val.numel() * val.element_size()

                if current_capacity >= bucket_capacity:
                    model_bucket_list.append(model_bucket)
                    current_capacity = 0
                    model_bucket = {}
        else:
            for idx, model_chunk in enumerate(model):
                for key, val in model_chunk.state_dict().items():
                    if "_extra_state" in key:
                        continue
                    model_bucket[key] = (val, idx)

                    if "decoder.layers" in key:
                        current_capacity += val.numel() * val.element_size()

                    if current_capacity >= bucket_capacity:
                        model_bucket_list.append(model_bucket)
                        current_capacity = 0
                        model_bucket = {}

        if len(model_bucket) > 0:
            model_bucket_list.append(model_bucket)

        if self._needs_pp_merge(
            parallel_state.get_tensor_model_parallel_world_size(),
            parallel_state.get_pipeline_model_parallel_world_size(),
        ):
            model_bucket_list = self._pad_buckets_to_stage_max(model_bucket_list)
        return model_bucket_list

    def _needs_pp_merge(self, tp_size, pp_size):
        """Whether the PP-stage merge collectives run.

        Derived from parallel_state and ReshardConfig at both call sites, so
        every rank in the group agrees and the merge stays symmetric.
        """
        return pp_size > 1 and (
            self.config.reshard_tp_size != tp_size
            or self.config.reshard_pp_size != pp_size
        )

    def _pad_buckets_to_stage_max(self, model_bucket_list):
        """Pad the bucket list to the PP-group max bucket count.

        Stages fill bucket_capacity at different rates and can end up with
        different bucket counts, but the merge runs once per bucket and needs
        them equal. A padded bucket joins the merge and is sent like any other.
        """
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        bucket_count = torch.tensor(
            [len(model_bucket_list)], device=torch.cuda.current_device()
        )
        torch.distributed.all_reduce(
            bucket_count, op=torch.distributed.ReduceOp.MAX, group=pp_group
        )
        max_bucket_count = int(bucket_count.item())
        while len(model_bucket_list) < max_bucket_count:
            model_bucket_list.append({})
        return model_bucket_list

    def gather_and_reshard_model(
        self,
        bucket_weight,
        dst_tp_rank,
        dst_ep_rank=None,
        dst_lm_head_rank=None,
        dst_shared_rank=None,
    ):
        """Gather once and narrow once, for the single-target inference reshard.

        The rollout path calls the two halves directly so that one gather can
        serve several targets.
        """
        full_sd = self.gather_full_model(bucket_weight)
        return self.narrow_to_target(
            full_sd,
            dst_tp_rank,
            dst_ep_rank,
            dst_lm_head_rank=dst_lm_head_rank,
            dst_shared_rank=dst_shared_rank,
        )

    def gather_full_model(self, bucket_weight):
        """Gather one bucket's parameters into full tensors.

        Takes no destination argument, so the result can be reused for every
        target. Returns ``{key: (full_tensor, narrow_spec)}``, where the spec
        tells narrow_to_target how to slice the tensor for one destination:
          ``None``                 pass through as-is
          ``("slice", dim, cat)``  slice on ``dim`` for the shard grid ``cat``
          ``("fused_glu", cat)``   tensor is [full_gate; full_up]: slice each
                                   half, then cat them back
          ``("expert_filter",)``   keep only this ep rank's experts
        """

        def _get_layer_index(split_key):
            for index, key in enumerate(split_key):
                if key == "layers":
                    return index + 1
            raise ValueError(f"Unknown layer name format: {split_key}")

        def _get_expert_index(split_key):
            for index, key in enumerate(split_key):
                if key == "local_experts":
                    return index + 1
            raise ValueError(f"Unknown expert name format: {split_key}")

        def rename_layer_num(param_name, layer_num):
            split_key = param_name.split(".")
            layer_index = int(_get_layer_index(split_key))
            split_key[layer_index] = str(layer_num)
            return ".".join(split_key)

        def rename_expert_layer_num(param_name, expert_num):
            split_key = param_name.split(".")
            expert_index = int(_get_expert_index(split_key))
            split_key[expert_index] = str(expert_num)
            return ".".join(split_key)

        def get_layer_num(param_name):
            split_key = param_name.split(".")
            layer_index = int(_get_layer_index(split_key))
            return int(split_key[layer_index])

        def get_expert_num(param_name):
            split_key = param_name.split(".")
            expert_index = int(_get_expert_index(split_key))
            return int(split_key[expert_index])

        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        ep_group = parallel_state.get_expert_model_parallel_group()
        tpe_size = parallel_state.get_expert_tensor_parallel_world_size()
        tpe_group = parallel_state.get_expert_tensor_parallel_group()
        num_moe_experts = self.config.model_config.num_moe_experts

        if not vp_size:
            vp_size = 1

        reshard_pp_model = self._needs_pp_merge(
            parallel_state.get_tensor_model_parallel_world_size(), pp_size
        )

        # This rank's pipeline-stage layer offset, from mcore so that uneven
        # stages and custom layouts are covered. The merge applies it on the
        # source side, so each source reports its true global layer number.
        stage_layer_offset = _layer_offset(self.config.model_config, 0)
        # Per-chunk offset relative to this stage's base, since interleaved
        # chunks repeat the same local layer number.
        vp_chunk_layer_offset = [
            _layer_offset(self.config.model_config, vp_stage) - stage_layer_offset
            for vp_stage in range(vp_size)
        ]

        model_level_params = {}
        tl_params = {}
        expert_params = {}

        if vp_size > 1:  # consolidate params across model chunks
            for key, (val, idx) in bucket_weight.items():
                if "_extra_state" in key:
                    continue
                if torch.is_tensor(val):
                    if key.startswith("decoder.layers."):
                        # Renumber to the chunk-relative layer number.
                        # mtp.* layers carry no chunk or stage offset.
                        key = rename_layer_num(
                            key, get_layer_num(key) + vp_chunk_layer_offset[idx]
                        )
                    # shared_experts is EP-replicated (not EP-sharded)
                    if (
                        num_moe_experts is not None
                        and "experts" in key
                        and "shared_experts" not in key
                    ):
                        expert_params[key] = val
                    elif "decoder.layers" in key:
                        tl_params[key] = val
                    else:
                        model_level_params[key] = val
        else:
            for key, val in bucket_weight.items():
                if "_extra_state" in key:
                    continue
                if torch.is_tensor(val):
                    if (
                        num_moe_experts is not None
                        and "experts" in key
                        and "shared_experts" not in key
                    ):
                        expert_params[key] = val
                    elif "decoder.layers" in key:
                        tl_params[key] = val
                    else:
                        model_level_params[key] = val

        # param split after routing: shared_experts must be in tl, not expert.
        n_shared_exp = sum(1 for key in expert_params if "shared_experts" in key)
        assert n_shared_exp == 0, (
            "shared_experts leaked into expert_params; would crash EP gather"
        )

        if reshard_pp_model:
            # Every stage broadcasts its entries under global layer numbers.
            # Without the merge, keys keep the local numbers that a receiver
            # with the same pp layout expects.
            tl_params = pp_merge_params(
                tl_params, pp_group, layer_offset=stage_layer_offset
            )

        full_sd = {}
        if num_moe_experts is not None:
            # in MoE model, if use the te group gemm, we need to convert the weight type from te group to seq group
            if self.config.moe_grouped_gemm == "te":
                from rlinf.utils.ckpt_convertor.megatron_convertor.utils.mg_moe_groupgemm import (
                    moe_te_group_to_seq,
                )

                expert_params = moe_te_group_to_seq(expert_params)
            else:
                assert self.config.moe_grouped_gemm in [None], (
                    f"now the rlinf just support moe_grouped_gemm to be None or 'te', got {self.config.moe_grouped_gemm}"
                )

            if ep_size > 1:
                # gather experts across ep ranks
                experts_per_chunk = num_moe_experts // ep_size
                ep_gathered_params = {}
                for key, val in expert_params.items():
                    weight_list = [torch.zeros_like(val) for _ in range(ep_size)]
                    torch.distributed.all_gather(weight_list, val, group=ep_group)
                    for idx in range(ep_size):
                        key2 = rename_expert_layer_num(
                            key, get_expert_num(key) + idx * experts_per_chunk
                        )
                        ep_gathered_params[key2] = weight_list[idx]
                expert_params = ep_gathered_params

            # Merge before the per-target filter, so the exchanged metadata
            # stays target-independent. Filtering first would make the
            # surviving key set depend on dst_ep_rank.
            if reshard_pp_model:
                expert_params = pp_merge_params(
                    expert_params, pp_group, layer_offset=stage_layer_offset
                )

            if self.config.rollout_ep_size > 1:
                assert tpe_size == 1, (
                    "EP reshard (rollout_ep_size > 1) expects no "
                    f"tensor-parallel-expert (tpe_size == 1), got tpe_size={tpe_size}"
                )
                # Experts stay whole; narrow_to_target picks each target's
                # subset.
                for key, val in expert_params.items():
                    full_sd[key] = (val, ("expert_filter",))
            else:
                # Experts are sharded by rollout TP instead.
                full_sd.update(
                    self.config.tpe_gather_fn(expert_params, tpe_size, tpe_group)
                )

        if reshard_pp_model:
            # Model-level params live on a single stage; the merge broadcasts
            # whichever stage owns each key, mtp.* included. layer_offset is 0
            # because these keys carry no decoder layer number.
            model_level_params = pp_merge_params(model_level_params, pp_group)

        # NOTE (wyq): Always reshard TP model even when tp_size == reshard_tp_size.
        # When tp_size == reshard_tp_size, resharding is equivalent to copying.
        # The rollout engine may load incorrect weights if not copied before offloading.
        full_tp_group = parallel_state.get_tensor_model_parallel_group()

        if self.config.enable_dp_attention:
            # DPA: attention TP differs from the full engine TP, so weight
            # types land on different destination grids, named by cat.
            attn_dict = {}
            dense_mlp_dict = {}
            shared_experts_dict = {}
            vocab_dict = {}
            # word_embeddings follows attn TP while output_layer follows
            # lm_head TP: sglang gates them on two independent flags,
            # is_dp_attention_enabled() and enable_dp_lm_head.
            for key, val in {**model_level_params, **tl_params}.items():
                if key.endswith("output_layer.weight"):
                    vocab_dict[key] = val
                elif "shared_experts" in key:
                    shared_experts_dict[key] = val
                elif "mlp.linear_fc" in key:
                    dense_mlp_dict[key] = val
                else:
                    attn_dict[key] = val
            # Attention: full TP all_gather + slice to 1/reshard_tp.
            full_sd.update(
                self.config.tp_gather_fn(
                    attn_dict,
                    full_tp_group,
                    "attn",
                    split_fc1=self.config.split_fc1,
                )
            )
            # Dense MLP: full TP all_gather + slice to 1/effective_dense_tp.
            full_sd.update(
                self.config.tp_gather_fn(
                    dense_mlp_dict,
                    full_tp_group,
                    "dense",
                    split_fc1=self.config.split_fc1,
                )
            )
            # Shared experts: full TP all_gather + slice to 1/rollout_full_tp.
            full_sd.update(
                self.config.tp_gather_fn(
                    shared_experts_dict,
                    full_tp_group,
                    "shared",
                    split_fc1=self.config.split_fc1,
                )
            )
            # lm_head (output_layer): full TP all_gather + slice.
            for key, val in vocab_dict.items():
                full_sd[key] = (
                    all_gather_tensor(val, 0, full_tp_group),
                    ("slice", 0, "lm_head"),
                )
        else:
            # Non-DPA: one shard grid for every weight.
            full_sd.update(
                self.config.tp_gather_fn(
                    {**model_level_params, **tl_params},
                    full_tp_group,
                    "attn",
                    split_fc1=self.config.split_fc1,
                    tp_gather_is_identity=self.tp_gather_is_identity,
                )
            )

        return full_sd

    def narrow_to_target(
        self,
        full_sd,
        dst_tp_rank,
        dst_ep_rank=None,
        dst_lm_head_rank=None,
        dst_shared_rank=None,
    ):
        """Slice a gathered full state dict for one target. Pure local.

        The gather half already encoded every per-model rule into the specs.
        Always builds a new dict, since full_sd is shared across targets, and
        never clones, since the gather half guarantees that None-spec entries
        do not alias live parameters.
        """
        reshard_tp_size = self.config.reshard_tp_size
        effective_dense_tp = self.config.rollout_moe_dense_tp_size or reshard_tp_size
        # cat label -> (rank, world_size), the only dst-dependent step here.
        # "expert_tp" resolves like "attn" but may diverge later.
        cat_to_dst = {
            "attn": (dst_tp_rank, reshard_tp_size),
            "dense": (dst_tp_rank % effective_dense_tp, effective_dense_tp),
            "shared": (dst_shared_rank, self.config.rollout_full_tp_size),
            "lm_head": (dst_lm_head_rank, self.config.rollout_lm_head_tp_size),
            "expert_tp": (dst_tp_rank, reshard_tp_size),
        }

        num_moe_experts = self.config.model_config.num_moe_experts
        experts_per_rank = (
            num_moe_experts // self.config.rollout_ep_size
            if num_moe_experts is not None and self.config.rollout_ep_size > 1
            else None
        )

        out = {}
        for key, (tensor, spec) in full_sd.items():
            if spec is None:
                out[key] = tensor
            elif spec[0] == "slice":
                _, dim, cat = spec
                dst_rank, dst_world_size = cat_to_dst[cat]
                out[key] = reshard_tensor_by_rank(tensor, dim, dst_rank, dst_world_size)
            elif spec[0] == "fused_glu":
                # [full_gate; full_up]: slice each half, then cat back.
                _, cat = spec
                dst_rank, dst_world_size = cat_to_dst[cat]
                gate, up = tensor.chunk(2, dim=0)
                out[key] = torch.cat(
                    [
                        reshard_tensor_by_rank(gate, 0, dst_rank, dst_world_size),
                        reshard_tensor_by_rank(up, 0, dst_rank, dst_world_size),
                    ],
                    dim=0,
                )
            else:
                # Keep only this ep rank's experts, matched on the global
                # expert id in the key.
                assert spec == ("expert_filter",)
                assert experts_per_rank is not None, (
                    "expert_filter spec requires rollout_ep_size > 1 and a "
                    "MoE model, so dst_ep_rank must not be None"
                )
                if "local_experts." not in key:
                    continue
                expert_num = int(key.split("local_experts.")[1].split(".")[0])
                start = dst_ep_rank * experts_per_rank
                if start <= expert_num < start + experts_per_rank:
                    out[key] = tensor

        if self.config.convert_fn is not None:
            out = self.config.convert_fn(out)
        return out
