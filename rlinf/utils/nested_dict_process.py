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


def copy_dict_tensor(next_extracted_obs: dict):
    ret = {}
    for key, value in next_extracted_obs.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value.clone()
        elif isinstance(value, dict):
            ret[key] = copy_dict_tensor(value)
        else:
            raise ValueError(f"{key=}, {type(value)} is not supported.")
    return ret


def put_tensor_device(data_dict, device):
    if data_dict is None:
        return None

    if isinstance(data_dict, torch.Tensor):
        return data_dict.value.to(device=device).contiguous()
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_device(value, device)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(device=device).contiguous()
    return data_dict


def split_dict_to_chunk(data: dict, split_size, dim=0):
    splited_list = [{} for _ in range(split_size)]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            split_vs = torch.chunk(value, split_size, dim=dim)
        elif value is None:
            split_vs = [None for _ in range(split_size)]
        elif isinstance(value, dict):
            split_vs = split_dict_to_chunk(value, split_size, dim)
        else:
            raise ValueError(f"{key=}, {type(value)} is not supported.")
        for split_id in range(split_size):
            splited_list[split_id][key] = split_vs[split_id]
    return splited_list


def concat_batch(data1, data2):
    batch = {}
    for key, value in data1.items():
        if isinstance(value, torch.Tensor):
            if key not in data2:
                # NOTE: NO WARNING FOR THE CASE THAT DATA2 DOES NOT CONTAIN SOME KEYS IN DATA1
                continue
            batch[key] = torch.cat([data1[key], data2[key]], dim=0)
        elif isinstance(value, dict):
            batch[key] = concat_batch(data1[key], data2[key])
    return batch


def stack_list_of_dict_tensor(list_of_dict: list, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = {}
    for key in keys:
        _v0 = list_of_dict[0][key]
        if isinstance(_v0, torch.Tensor):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = torch.stack(v_list, dim=dim)
        elif isinstance(_v0, dict):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = stack_list_of_dict_tensor(v_list)
        elif _v0 is None:
            pass
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")
    return ret


def cat_list_of_dict_tensor(list_of_dict: list, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = {}
    for key in keys:
        _v0 = list_of_dict[0][key]
        if isinstance(_v0, torch.Tensor):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = torch.cat(v_list, dim=dim)
        elif isinstance(_v0, dict):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = cat_list_of_dict_tensor(v_list)
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")
    return ret
