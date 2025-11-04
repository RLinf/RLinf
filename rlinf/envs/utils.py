import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union


def to_tensor(
    array: Union[Dict, torch.Tensor, np.ndarray, List, Any], device: str = "cpu"
) -> Union[Dict, torch.Tensor]:
    """
    Copied from ManiSkill!
    Maps any given sequence to a torch tensor on the CPU/GPU. If physx gpu is not enabled then we use CPU, otherwise GPU, unless specified
    by the device argument

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None and to_tensor will put the device on the GPU if physx is enabled
            and CPU otherwise

    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v, device=device) for k, v in array.items()}
    elif isinstance(array, torch.Tensor):
        ret = array.to(device)
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
        elif array.dtype == np.uint32:
            array = array.astype(np.int64)
        ret = torch.tensor(array).to(device)
    else:
        if isinstance(array, list) and isinstance(array[0], np.ndarray):
            array = np.array(array)
        ret = torch.tensor(array, device=device)
    if ret.dtype == torch.float64:
        ret = ret.to(torch.float32)
    return ret


def list_of_dict_to_dict_of_list(
    list_of_dict: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    """
    Convert a list of dictionaries to a dictionary of lists.

    Args:
        list_of_dict: List of dictionaries with same keys

    Returns:
        Dictionary where each key maps to a list of values
    """
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output
