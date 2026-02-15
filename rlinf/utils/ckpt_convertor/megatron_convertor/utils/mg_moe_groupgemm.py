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


def moe_seq_to_te_group(state_dict):
    key_moe_grouped = ".mlp.experts."
    key_moe_local = ".mlp.experts.local_experts."
    key_local_linear_fc1 = "linear_fc1.weight"
    key_local_linear_fc2 = "linear_fc2.weight"
    key_grouped_linear_fc1 = "linear_fc1.weight"
    key_grouped_linear_fc2 = "linear_fc2.weight"
    te_endswith = "_extra_state"

    for key in list(state_dict.keys()):
        if key_moe_local not in key:
            continue
        key_index = key.find(key_moe_local)
        key_prefix = key[:key_index]
        expert_index_start = key_index + len(key_moe_local)
        expert_index_end = key.find(".", expert_index_start)
        expert_index = int(key[expert_index_start:expert_index_end])
        if key.endswith(key_local_linear_fc1):
            state_dict[
                f"{key_prefix}{key_moe_grouped}{key_grouped_linear_fc1}{expert_index}"
            ] = state_dict.pop(key)
        elif key.endswith(key_local_linear_fc2):
            state_dict[
                f"{key_prefix}{key_moe_grouped}{key_grouped_linear_fc2}{expert_index}"
            ] = state_dict.pop(key)
        elif key.endswith(te_endswith):
            continue
        else:
            assert False, (
                f"key {key} not end with {key_local_linear_fc1} {key_local_linear_fc2} {te_endswith}"
            )
    return state_dict


def moe_te_group_to_seq(state_dict):
    key_moe_grouped = ".mlp.experts.linear"
    key_moe_local = ".mlp.experts.local_experts."
    key_local_linear_fc1 = "linear_fc1.weight"
    key_local_linear_fc2 = "linear_fc2.weight"
    key_local_weight = "weight"

    pop_linear_fc1 = {}
    pop_linear_fc2 = {}
    for key in list(state_dict.keys()):
        if key_moe_grouped not in key:
            continue
        if key.find(key_local_linear_fc1) != -1:
            assert key not in pop_linear_fc1
            pop_linear_fc1[key] = state_dict.pop(key)
        elif key.find(key_local_linear_fc2) != -1:
            assert key not in pop_linear_fc2
            pop_linear_fc2[key] = state_dict.pop(key)

    for key, weight in pop_linear_fc1.items():
        key_index = key.find(key_moe_grouped)
        key_prefix = key[:key_index]
        expert_index = key.find(key_local_weight)
        # find the local expert index 6 is 'weight' length
        expert_prefix = key[expert_index + 6 :]
        state_dict[
            f"{key_prefix}{key_moe_local}{expert_prefix}.{key_local_linear_fc1}"
        ] = weight

    for key, weight in pop_linear_fc2.items():
        key_index = key.find(key_moe_grouped)
        key_prefix = key[:key_index]
        expert_index = key.find(key_local_weight)
        # find the local expert index 6 is 'weight' length
        expert_prefix = key[expert_index + 6 :]
        state_dict[
            f"{key_prefix}{key_moe_local}{expert_prefix}.{key_local_linear_fc2}"
        ] = weight

    return state_dict


def moe_seq_to_group(state_dict, num_local_experts, glu):
    key_moe_grouped = ".mlp.experts."
    key_moe_local = ".mlp.experts.local_experts."
    key_local_linear_fc1 = "linear_fc1.weight"
    key_local_linear_fc2 = "linear_fc2.weight"
    key_grouped_linear_fc1 = "weight1"
    key_grouped_linear_fc2 = "weight2"

    pop_linear_fc1 = {}
    pop_linear_fc2 = {}
    for key in list(state_dict.keys()):
        if key_moe_local not in key:
            continue
        key_index = key.find(key_moe_local)
        key_prefix = key[:key_index]
        expert_index_start = key_index + len(key_moe_local)
        expert_index_end = key.find(".", expert_index_start)
        expert_index = int(key[expert_index_start:expert_index_end])
        if key.endswith(key_local_linear_fc1):
            if key_prefix not in pop_linear_fc1:
                pop_linear_fc1[key_prefix] = [None for _ in range(num_local_experts)]
            assert pop_linear_fc1[key_prefix][expert_index] is None
            pop_linear_fc1[key_prefix][expert_index] = state_dict.pop(key)
        elif key.endswith(key_local_linear_fc2):
            if key_prefix not in pop_linear_fc2:
                pop_linear_fc2[key_prefix] = [None for _ in range(num_local_experts)]
            assert pop_linear_fc2[key_prefix][expert_index] is None
            pop_linear_fc2[key_prefix][expert_index] = state_dict.pop(key)

    for key_prefix, value_list in pop_linear_fc1.items():
        if glu:
            weight = torch.stack(value_list)
            weight = weight.transpose(1, 2)
            weight = weight.view(num_local_experts, weight.shape[1], 2, -1)
            weight = weight.transpose(0, 1).transpose(1, 2)
            weight = weight.reshape(weight.shape[0], -1)
            key = f"{key_prefix}{key_moe_grouped}{key_grouped_linear_fc1}"
        else:
            weight = torch.stack(value_list, dim=0)
            weight = weight.transpose(1, 2)
            weight = weight.transpose(0, 1)
            weight = weight.reshape(weight.shape[0], -1)
            key = f"{key_prefix}{key_moe_grouped}{key_grouped_linear_fc1}"
        state_dict[f"{key_prefix}{key_moe_grouped}{key_grouped_linear_fc1}"] = weight

    for key_prefix, value_list in pop_linear_fc2.items():
        weight = torch.stack(value_list, dim=0)
        weight = weight.transpose(1, 2)
        weight = weight.reshape(-1, weight.shape[-1])
        key = f"{key_prefix}{key_moe_grouped}{key_grouped_linear_fc2}"
        state_dict[f"{key_prefix}{key_moe_grouped}{key_grouped_linear_fc2}"] = weight

    return state_dict


def moe_group_to_seq(state_dict, num_local_experts, glu):
    key_moe_grouped = ".mlp.experts."
    key_moe_local = ".mlp.experts.local_experts."
    key_local_linear_fc1 = "linear_fc1.weight"
    key_local_linear_fc2 = "linear_fc2.weight"
    key_grouped_linear_fc1 = "weight1"
    key_grouped_linear_fc2 = "weight2"

    pop_linear_fc1 = {}
    pop_linear_fc2 = {}
    for key in list(state_dict.keys()):
        if key_moe_grouped not in key:
            continue
        key_index = key.find(key_moe_grouped)
        key_prefix = key[:key_index]
        if key.endswith(key_grouped_linear_fc1):
            assert key_prefix not in pop_linear_fc1
            pop_linear_fc1[key_prefix] = state_dict.pop(key)
        elif key.endswith(key_grouped_linear_fc2):
            assert key_prefix not in pop_linear_fc2
            pop_linear_fc2[key_prefix] = state_dict.pop(key)

    for key_prefix, value in pop_linear_fc1.items():
        if glu:
            weight = value.view(value.shape[0], 2, num_local_experts, -1)
            weight = weight.transpose(0, 2).transpose(2, 3)
            weight = weight.reshape(-1, weight.shape[-1])
            weight_list = torch.chunk(weight, num_local_experts, dim=0)
        else:
            weight = value.view(value.shape[0], num_local_experts, -1)
            weight = weight.transpose(0, 1).transpose(1, 2)
            weight = weight.reshape(-1, weight.shape[-1])
            weight_list = torch.chunk(weight, num_local_experts, dim=0)
        for i, weight in enumerate(weight_list):
            state_dict[f"{key_prefix}{key_moe_local}{i}.{key_local_linear_fc1}"] = (
                weight
            )

    for key_prefix, value in pop_linear_fc2.items():
        weight = value.view(num_local_experts, -1, value.shape[-1])
        weight = weight.transpose(1, 2)
        weight = weight.reshape(-1, weight.shape[-1])
        weight_list = torch.chunk(weight, num_local_experts, dim=0)
        for i, weight in enumerate(weight_list):
            state_dict[f"{key_prefix}{key_moe_local}{i}.{key_local_linear_fc2}"] = (
                weight
            )

    return state_dict
