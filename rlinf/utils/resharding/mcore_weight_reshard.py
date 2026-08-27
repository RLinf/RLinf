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
from megatron.training.training import unwrap_model

from .reshard_config import ReshardConfig
from .utils import gather_and_reshard_tensor


class MegatronCoreWeightReshard:
    def __init__(self, config: ReshardConfig):
        self.config = config
        self.bucket_capacity = self.config.bucket_capacity

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
        return model_bucket_list

    def gather_and_reshard_model(
        self,
        bucket_weight,
        dst_tp_rank,
        dst_ep_rank=None,
        dst_lm_head_rank=None,
        dst_shared_rank=None,
    ):
        """
        Accumulate all vp model chunks together, and reshard model (i.e) gather all pp ranks
        if required and return the final model state dict

        Args:
            dst_tp_rank: rollout attention tp rank (for attention/dense TP slice).
            dst_ep_rank: rollout ep rank (for EP expert selection). None when no
                MoE EP (rollout_ep_size==1 or non-MoE model); in that case the
                tpe_reshard_fn path (TP-slice) is used instead of ep_reshard_fn.
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

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        ep_group = parallel_state.get_expert_model_parallel_group()
        tpe_size = parallel_state.get_expert_tensor_parallel_world_size()
        tpe_group = parallel_state.get_expert_tensor_parallel_group()

        if not vp_size:
            vp_size = 1

        reshard_pp_model = False
        reshard_tp_model = True
        reshard_ep_model = False

        # NOTE (wyq): Always reshard TP model even when tp_size == reshard_tp_size.
        # When tp_size == reshard_tp_size, resharding is equivalent to copying.
        # The rollout engine may load incorrect weights if not copied before offloading.
        if (
            self.config.reshard_tp_size != tp_size
            or self.config.reshard_pp_size != pp_size
        ):
            if pp_size > 1:
                reshard_pp_model = True

        if (
            ep_size > 1 or tpe_size > 1
        ) and self.config.model_config.num_moe_experts is not None:
            reshard_ep_model = True
            experts_per_chunk = self.config.model_config.num_moe_experts // ep_size

        layers_per_pp = self.config.model_config.num_layers // pp_size
        layers_per_chunk = layers_per_pp // vp_size

        model_state_dict = {}
        model_level_params = {}
        tl_params = {}
        expert_params = {}

        if vp_size > 1:  # consolidate params across model chunks
            for key, (val, idx) in bucket_weight.items():
                if "_extra_state" in key:
                    continue
                if torch.is_tensor(val):
                    if "layers" in key:
                        key2 = rename_layer_num(
                            key,
                            get_layer_num(key) + idx * pp_size * layers_per_chunk,
                        )
                        # shared_experts is EP-replicated (not EP-sharded)
                        if (
                            reshard_ep_model
                            and "experts" in key
                            and "shared_experts" not in key
                        ):
                            expert_params[key2] = val
                        else:
                            tl_params[key2] = val
                    else:
                        model_level_params[key] = val
        else:
            for key, val in bucket_weight.items():
                if "_extra_state" in key:
                    continue
                if torch.is_tensor(val):
                    if (
                        reshard_ep_model
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

        if vp_size > 1 or reshard_pp_model:
            # gather layers across pp ranks
            gathered_params = {}
            for key, val in tl_params.items():
                weight_list = [torch.zeros_like(val) for _ in range(pp_size)]
                torch.distributed.all_gather(weight_list, val, group=pp_group)
                for idx in range(pp_size):
                    layer_num = get_layer_num(key) + idx * layers_per_chunk
                    key2 = rename_layer_num(key, layer_num)
                    if not reshard_pp_model:  # Save only layers of 1 single PP stage
                        layers_start = layers_per_pp * pp_rank
                        layers_end = layers_per_pp * (pp_rank + 1) - 1
                        if layer_num >= layers_start and layer_num <= layers_end:
                            key2 = rename_layer_num(key, layer_num % layers_per_pp)
                            gathered_params[key2] = weight_list[idx]
                    else:
                        gathered_params[key2] = weight_list[idx]
            tl_params = gathered_params

        if self.config.model_config.num_moe_experts is not None:
            # in MoE model, if use the te group gemm, we need to convert the weight type from te group to seq group
            if self.config.moe_grouped_gemm == "te":
                from rlinf.utils.ckpt_convertor.megatron_convertor.utils.mg_moe_groupgemm import (
                    moe_te_group_to_seq,
                )

                if reshard_ep_model:
                    expert_params = moe_te_group_to_seq(expert_params)
                else:
                    tl_params = moe_te_group_to_seq(tl_params)
            else:
                assert self.config.moe_grouped_gemm in [None], (
                    f"now the rlinf just support moe_grouped_gemm to be None or 'te', got {self.config.moe_grouped_gemm}"
                )

            if reshard_ep_model:
                # gather experts across ep ranks
                ep_gathered_params = {}
                for key, val in expert_params.items():
                    weight_list = [torch.zeros_like(val) for _ in range(ep_size)]
                    torch.distributed.all_gather(weight_list, val, group=ep_group)
                    for idx in range(ep_size):
                        key2 = rename_expert_layer_num(
                            key, get_expert_num(key) + idx * experts_per_chunk
                        )
                        ep_gathered_params[key2] = weight_list[idx]

                # reshard experts for the rollout layout.
                #   ep_reshard_fn selects this rank's slice by global expert id.
                if (
                    self.config.rollout_ep_size > 1
                    and self.config.ep_reshard_fn is not None
                ):
                    assert tpe_size == 1, (
                        "EP reshard (rollout_ep_size > 1) expects no "
                        f"tensor-parallel-expert (tpe_size == 1), got tpe_size={tpe_size}"
                    )
                    ep_gathered_params = self.config.ep_reshard_fn(
                        ep_gathered_params,
                        self.config.rollout_ep_size,
                        dst_ep_rank,
                        self.config.model_config.num_moe_experts,
                    )
                else:
                    ep_gathered_params = self.config.tpe_reshard_fn(
                        ep_gathered_params,
                        tpe_size,
                        tpe_group,
                        self.config.reshard_tp_size,
                        dst_tp_rank,
                    )

                # gather experts across pp ranks
                pp_gathered_params = {}
                for key, val in ep_gathered_params.items():
                    weight_list = [torch.zeros_like(val) for _ in range(pp_size)]
                    torch.distributed.all_gather(weight_list, val, group=pp_group)
                    for idx in range(pp_size):
                        layer_num = get_layer_num(key) + idx * layers_per_chunk
                        key2 = rename_layer_num(key, layer_num)
                        if (
                            not reshard_pp_model
                        ):  # Save only layers of 1 single PP stage
                            layers_start = layers_per_pp * pp_rank
                            layers_end = layers_per_pp * (pp_rank + 1) - 1
                            if layer_num >= layers_start and layer_num <= layers_end:
                                key2 = rename_layer_num(key, layer_num % layers_per_pp)
                                pp_gathered_params[key2] = weight_list[idx]
                        else:
                            pp_gathered_params[key2] = weight_list[idx]

                expert_params = pp_gathered_params
            else:
                tl_params = self.config.tpe_reshard_fn(
                    tl_params,
                    tpe_size,
                    tpe_group,
                    self.config.reshard_tp_size,
                    dst_tp_rank,
                )

        model_state_dict.update(model_level_params)
        model_state_dict.update(tl_params)
        model_state_dict.update(expert_params)

        reshard_dtype = self.config.model_config.params_dtype

        if reshard_pp_model:
            model_state_dict = self.config.pp_reshard_fn(
                model_state_dict, pp_group, reshard_dtype
            )

        if reshard_tp_model:
            full_tp_group = parallel_state.get_tensor_model_parallel_group()
            rollout_full_tp = self.config.rollout_full_tp_size

            if self.config.enable_dp_attention:
                # DPA: attn TP (reshard_tp_size) != full engine TP
                attn_dict = {}
                dense_mlp_dict = {}
                shared_experts_dict = {}
                vocab_dict = {}
                has_lm_head_dst = dst_lm_head_rank is not None
                for key, val in model_state_dict.items():
                    if has_lm_head_dst and (
                        key.endswith("output_layer.weight")
                        or key.endswith("eh_proj.weight")
                    ):
                        vocab_dict[key] = val
                    elif "shared_experts" in key:
                        shared_experts_dict[key] = val
                    elif "mlp.linear_fc" in key:
                        dense_mlp_dict[key] = val
                    else:
                        attn_dict[key] = val
                # Attention: full TP all_gather + slice to 1/reshard_tp.
                attn_dict = self.config.tp_reshard_fn(
                    attn_dict,
                    full_tp_group,
                    dst_tp_rank,
                    self.config.reshard_tp_size,
                )
                # Dense MLP: full TP all_gather + slice to 1/effective_dense_tp.
                effective_dense_tp = (
                    self.config.rollout_moe_dense_tp_size or self.config.reshard_tp_size
                )
                dense_dst_rank = dst_tp_rank % effective_dense_tp
                dense_mlp_dict = self.config.tp_reshard_fn(
                    dense_mlp_dict,
                    full_tp_group,
                    dense_dst_rank,
                    effective_dense_tp,
                )
                # Shared experts: full TP all_gather + slice to 1/rollout_full_tp.
                shared_experts_dict = self.config.tp_reshard_fn(
                    shared_experts_dict,
                    full_tp_group,
                    dst_shared_rank,
                    rollout_full_tp,
                )
                # lm_head (output_layer) + MTP eh_proj: full TP all_gather + slice.
                lm_head_tp = self.config.rollout_lm_head_tp_size
                for key, val in vocab_dict.items():
                    if key.endswith("eh_proj.weight"):
                        vocab_dict[key] = gather_and_reshard_tensor(
                            val, 0, full_tp_group, 0, 1
                        )
                    else:  # output_layer / lm_head
                        vocab_dict[key] = gather_and_reshard_tensor(
                            val,
                            0,
                            full_tp_group,
                            dst_lm_head_rank,
                            lm_head_tp,
                        )
                model_state_dict = {
                    **attn_dict,
                    **dense_mlp_dict,
                    **shared_experts_dict,
                    **vocab_dict,
                }
            else:
                # Non-DPA: full TP all_gather + slice to 1/reshard_tp.
                model_state_dict = self.config.tp_reshard_fn(
                    model_state_dict,
                    full_tp_group,
                    dst_tp_rank,
                    self.config.reshard_tp_size,
                )

        if self.config.convert_fn is not None:
            model_state_dict = self.config.convert_fn(model_state_dict)

        return model_state_dict
