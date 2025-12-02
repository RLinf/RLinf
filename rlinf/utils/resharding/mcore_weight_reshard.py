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

from .reshard_config import ReshardConfig


class MegatronCoreWeightReshard:
    def __init__(self, config: ReshardConfig):
        self.config = config

        assert (
            self.config.model_config.tensor_model_parallel_size
            >= self.config.reshard_tp_size
            and self.config.model_config.tensor_model_parallel_size
            % self.config.reshard_tp_size
            == 0
        ), (
            f"Invalid tensor model parallel size {self.config.model_config.tensor_model_parallel_size} "
            f"and reshard tp size {self.config.reshard_tp_size}. "
        )
        self.tp_subgroups = {}
        self.merge_factor = (
            self.config.model_config.tensor_model_parallel_size
            // self.config.reshard_tp_size
        )
        self._create_tp_subgroups()

    def _create_tp_subgroups(self):
        self.world_size = torch.distributed.get_world_size()
        num_groups = self.world_size // self.merge_factor
        all_subgroups = [
            list(range(i * self.merge_factor, (i + 1) * self.merge_factor))
            for i in range(num_groups)
        ]

        for subgroup_ranks in all_subgroups:
            key = tuple(subgroup_ranks)
            if key not in self.tp_subgroups:
                self.tp_subgroups[key] = parallel_state.create_group(
                    subgroup_ranks, backend="nccl"
                )

    def _get_tp_subgroup(self, subgroup_ranks):
        """
        Retrieve an existing communication subgroup.
        """
        key = tuple(subgroup_ranks)
        if key in self.tp_subgroups:
            return self.tp_subgroups[key]

        raise ValueError(
            f"Subgroup {key} does not exist! Please call _create_tp_subgroups() to create this subgroup first."
        )

    def gather_and_reshard_model(self, model, dst_rank):
        """
        Accumulate all vp model chunks together, and reshard model (i.e) gather all pp ranks
        if required and return the final model state dict
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

        tl_params = {}
        expert_params = {}
        model_level_params = {}
        if vp_size > 1:  # consolidate params across model chunks
            for idx, model_chunk in enumerate(model):
                for key, val in model_chunk.state_dict().items():
                    if "_extra_state" in key:
                        continue
                    if torch.is_tensor(val):
                        if "layers" in key:
                            key2 = rename_layer_num(
                                key,
                                get_layer_num(key) + idx * pp_size * layers_per_chunk,
                            )
                            if reshard_ep_model and "experts" in key:
                                expert_params[key2] = val
                            else:
                                tl_params[key2] = val
                        else:
                            model_level_params[key] = val
        else:
            for key, val in model[0].state_dict().items():
                if "_extra_state" in key:
                    continue
                if torch.is_tensor(val):
                    if reshard_ep_model and "experts" in key:
                        expert_params[key] = val
                    elif "decoder.layers" in key:
                        tl_params[key] = val
                    else:
                        model_level_params[key] = val

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
                from toolkits.ckpt_convertor.utils.mg_moe_groupgemm import (
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

                # reshard experts across tpe ranks
                ep_gathered_params = self.config.tpe_reshard_fn(
                    ep_gathered_params,
                    tpe_size,
                    tpe_group,
                    self.config.reshard_tp_size,
                    dst_rank,
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
                    dst_rank,
                )

        model_state_dict = model_level_params
        model_state_dict.update(tl_params)
        model_state_dict.update(expert_params)

        reshard_dtype = self.config.model_config.params_dtype

        if reshard_pp_model:
            model_state_dict = self.config.pp_reshard_fn(
                model_state_dict, pp_group, reshard_dtype
            )

        if reshard_tp_model:
            rank = torch.distributed.get_rank()
            group_index = rank // self.merge_factor
            subgroup_ranks = list(
                range(
                    group_index * self.merge_factor,
                    (group_index + 1) * self.merge_factor,
                )
            )
            tp_sub_group = self._get_tp_subgroup(subgroup_ranks)
            model_state_dict = self.config.tp_reshard_fn(
                model_state_dict, self.merge_factor, tp_sub_group
            )

        if self.config.convert_fn is not None:
            model_state_dict = self.config.convert_fn(model_state_dict)

        return model_state_dict
