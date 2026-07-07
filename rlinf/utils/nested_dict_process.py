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

from typing import Any

import numpy as np
import torch

# Keys that we have already warned about in concat_batch, so each missing key
# only produces a single warning per process (avoid log spam in the replay /
# demo batch pipeline).
_CONCAT_BATCH_WARNED_KEYS: set[str] = set()


def update_nested_cfg(base_cfg, override_cfg):
    for key, value in override_cfg.items():
        if (
            key in base_cfg
            and isinstance(base_cfg[key], dict)
            and isinstance(value, dict)
        ):
            update_nested_cfg(base_cfg[key], value)
        else:
            base_cfg[key] = value
    return base_cfg


def copy_dict_tensor(next_extracted_obs: dict):
    """
    Recursively clones all torch tensors in a dict.
    """
    ret = {}
    for key, value in next_extracted_obs.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value.clone()
        elif isinstance(value, dict):
            ret[key] = copy_dict_tensor(value)
        else:
            ret[key] = value
    return ret


def clone_nested_to_cpu(value: Any):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone_nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested_to_cpu(item) for item in value)
    return value


def put_tensor_device(data_dict, device):
    if data_dict is None:
        return None

    if isinstance(data_dict, torch.Tensor):
        return data_dict.to(device=device).contiguous()
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_device(value, device)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(device=device).contiguous()
    return data_dict


def _split_list_by_sizes(value: list, split_sizes: list[int] | int) -> list[list]:
    if isinstance(split_sizes, int):
        chunks = split_sizes
        chunk_size = (len(value) + chunks - 1) // chunks if chunks else 0
        split_sizes = [
            min(chunk_size, len(value) - i * chunk_size)
            if i * chunk_size < len(value)
            else 0
            for i in range(chunks)
        ]
    out, i = [], 0
    for n in split_sizes:
        out.append(value[i : i + n])
        i += n
    return out


def split_dict_to_chunk(data: dict, split_size, dim=0):
    splited_list = [{} for _ in range(split_size)]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            split_vs = [
                chunk.contiguous() for chunk in torch.chunk(value, split_size, dim=dim)
            ]
        elif isinstance(value, list):
            assert dim == 0, f"List field only supports dim=0, got {dim}."
            split_vs = _split_list_by_sizes(value, split_size)
        elif value is None:
            split_vs = [None for _ in range(split_size)]
        elif isinstance(value, dict):
            split_vs = split_dict_to_chunk(value, split_size, dim)
        else:
            raise ValueError(f"{key=}, {type(value)} is not supported.")
        for split_id in range(split_size):
            splited_list[split_id][key] = (
                split_vs[split_id].contiguous()
                if isinstance(split_vs[split_id], torch.Tensor)
                else split_vs[split_id]
            )
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
            # NOTE: added this for dealing with different keys in demo data.
            if key not in data2:
                if key not in _CONCAT_BATCH_WARNED_KEYS:
                    _CONCAT_BATCH_WARNED_KEYS.add(key)
                    # Lazy import to avoid pulling rlinf.scheduler.worker (and
                    # its heavy deps) at module import time. This only runs
                    # once per missing key, inside a worker where that import
                    # is essentially free.
                    from rlinf.utils.logging import get_logger

                    get_logger().warning(
                        "concat_batch: key '%s' not found in data2 (value type: %s), "
                        "skipping. This warning is only emitted once per key.",
                        key,
                        type(value).__name__,
                    )
                continue
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
        if _v0 is None:
            continue

        v_list = [d[key] for d in list_of_dict]

        if isinstance(_v0, torch.Tensor):
            ret[key] = torch.cat(v_list, dim=dim)
        elif isinstance(_v0, np.ndarray):
            ret[key] = np.concatenate([v for v in v_list if v is not None], axis=dim)
        elif isinstance(_v0, list):
            assert dim == 0, f"{key=} is list, dim !=0 is not supported!"
            ret[key] = [item for sub in v_list if sub is not None for item in sub]
        elif isinstance(_v0, dict):
            ret[key] = cat_list_of_dict_tensor(v_list, dim=dim)
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")

    return ret


def split_dict(
    batch: dict[str, Any],
    split_sizes: list[int],
    dim: int = 0,
) -> list[dict[str, Any]]:
    """Split one batch dict into size-specified sub-batches.

    Tensor values are chunked on ``dim``; list values are sliced proportionally;
    nested dict values are split recursively.

    Args:
        batch: Dict.
        split_sizes: Batch sizes for each destination rank.
        dim: Tensor dimension to split. Defaults to 0.

    Returns:
        A list of splited batches, one item per destination rank.
    """
    count = len(split_sizes)
    total_size = sum(split_sizes)
    splitted_batches = [{} for _ in range(count)]
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.shape[dim] == total_size, (
                f"Tensor field '{key}' expected split dim size {total_size}, "
                f"got {value.shape[dim]} on dim {dim}."
            )
            splitted_values = torch.split(value, split_sizes, dim=dim)
            for i in range(count):
                splitted_batches[i][key] = splitted_values[i].contiguous()
        elif isinstance(value, list):
            assert dim == 0, f"List field '{key}' only supports dim=0, got {dim}."
            length = len(value)
            assert length == total_size, (
                f"List field '{key}' expected length {total_size}, got {length}."
            )
            begin = 0
            for i, size in enumerate(split_sizes):
                splitted_batches[i][key] = value[begin : begin + size]
                begin += size
        elif isinstance(value, dict):
            splitted_sub_batches = split_dict(value, split_sizes, dim=dim)
            for i in range(count):
                splitted_batches[i][key] = splitted_sub_batches[i]
        else:
            for i in range(count):
                splitted_batches[i][key] = value

    return splitted_batches
