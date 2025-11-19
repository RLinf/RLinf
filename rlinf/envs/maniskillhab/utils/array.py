import copy
from typing import Sequence, Union

import numpy as np
import torch

from mani_skill.utils.structs.types import Array


def all_equal(array: list):
    return len(set(array)) == 1


def all_same_type(array: list):
    return len(set(map(type, array))) == 1


def tensor_intersection(a, b):
    a_cat_b, counts = torch.cat([a, b]).unique(return_counts=True)
    return a_cat_b[torch.where(counts.gt(1))]


def tensor_intersection_idx(a, b):
    """
    Given sorted ascending tensors a, b without repeated elements, for elements in tensor_intersection(a, b), give their index in a

    Args:
        a: tensor (1-D), sorted ascending, no repeated values
        b: tensor (1-D), sorted ascending, no repeated values
    """
    insert_pos = torch.clamp(torch.searchsorted(a, b), max=len(a) - 1)
    mask = (insert_pos < len(a)) & (a[insert_pos] == b)
    return insert_pos[mask]


def recursive_deepcopy(data):
    if isinstance(data, dict):
        return dict((k, recursive_deepcopy(v)) for k, v in data.items())
    if isinstance(data, list):
        return [recursive_deepcopy(x) for x in data]
    if isinstance(data, set):
        return set(recursive_deepcopy(list(data)))
    if isinstance(data, torch.Tensor):
        return data.clone()
    if isinstance(data, np.ndarray):
        return data.copy()
    return copy.deepcopy(data)


def recursive_slice(obs, slice, inplace=False):
    if isinstance(obs, dict):
        if inplace:
            for k, v in obs.items():
                obs[k] = recursive_slice(v, slice)
            return obs
        return dict((k, recursive_slice(v, slice)) for k, v in obs.items())
    else:
        return obs[slice]


def to_tensor(
    array: Union[torch.Tensor, np.array, Sequence], device=None, dtype=torch.float
):
    """
    Maps any given sequence to a torch tensor on the CPU/GPU. If physx gpu is not enabled then we use CPU, otherwise GPU, unless specified
    by the device argument

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None and to_tensor will put the device on the GPU if physx is enabled
            and CPU otherwise

    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v, device=device) for k, v in array.items()}
    if isinstance(array, np.ndarray):
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
        ret = torch.from_numpy(array)
    elif isinstance(array, list) and isinstance(array[0], np.ndarray):
        ret = torch.from_numpy(np.array(array))
    elif isinstance(array, torch.Tensor):
        ret = array
    elif isinstance(array, list) and isinstance(array[0], torch.Tensor):
        ret = torch.stack(array)
    else:
        ret = torch.Tensor(array)

    if not isinstance(dtype, str):
        ret = ret.to(dtype)
    elif dtype == "float":
        ret = ret.float()
    elif dtype == "double":
        ret = ret.double()
    elif dtype == "int":
        ret = ret.int()

    if device is not None:
        ret = ret.to(device)

    return ret


def to_numpy(array: Union[Array, Sequence], dtype=None) -> np.ndarray:
    if isinstance(array, (dict)):
        return {k: to_numpy(v, dtype=dtype) for k, v in array.items()}
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy().astype(dtype)
    if isinstance(array, np.ndarray):
        return array.astype(dtype)
    if (
        isinstance(array, bool)
        or isinstance(array, str)
        or isinstance(array, float)
        or isinstance(array, int)
    ):
        return array

    return np.array(array, dtype=dtype)
