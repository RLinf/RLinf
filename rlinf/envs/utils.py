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

from typing import Any, Union

import numpy as np
import torch


def to_tensor(
    array: Union[dict, torch.Tensor, np.ndarray, list, Any], device: str = "cpu"
) -> Union[dict, torch.Tensor]:
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
    list_of_dict: list[dict[str, Any]],
) -> dict[str, list[Any]]:
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


def extract_success_frame_from_infos(
    infos: dict[str, Any], chunk_rewards: torch.Tensor
) -> torch.Tensor:
    """
    Extract success_frame from environment infos dictionary.

    Priority: use success_frame if available (contains per-step success for all chunk steps),
    otherwise fall back to success (only represents last step).

    Args:
        infos: Dictionary containing environment information, potentially with "success_frame" or "success" keys
        chunk_rewards: Reward tensor with shape [num_envs] or [num_envs, chunk_steps] used to determine output shape

    Returns:
        success_frame: Tensor with shape [num_envs] or [num_envs, chunk_steps] containing success signals
    """
    success_frame = None

    if "success_frame" in infos:
        # If environment already provides success_frame per step (from chunk_step)
        # This is the preferred source as it contains success for each step in the chunk
        success_frame = infos["success_frame"]
        if isinstance(success_frame, torch.Tensor):
            pass  # Already a tensor, keep as is
        elif isinstance(success_frame, (list, tuple)):
            success_frame = torch.tensor(
                success_frame, dtype=torch.float32, device=chunk_rewards.device
            )
    elif "success" in infos:
        # Fallback: use success if success_frame is not available
        # If success is a tensor with shape [num_envs], expand to [num_envs, chunk_steps] if needed
        success_tensor = infos["success"]
        if isinstance(success_tensor, torch.Tensor):
            if success_tensor.dim() == 1:
                # [num_envs] -> [num_envs, chunk_steps]
                # For chunk_step, we need success for each step
                # Check if chunk_rewards has chunk dimension
                if chunk_rewards.dim() == 2 and chunk_rewards.shape[1] > 1:
                    # Expand to match chunk_steps: repeat the last value for all steps
                    success_frame = success_tensor.unsqueeze(1).expand(
                        -1, chunk_rewards.shape[1]
                    )
                else:
                    success_frame = (
                        success_tensor.unsqueeze(1)
                        if chunk_rewards.dim() == 2
                        else success_tensor
                    )
            else:
                success_frame = success_tensor
        elif isinstance(success_tensor, (list, tuple)):
            # Convert list to tensor
            success_frame = torch.tensor(
                success_tensor, dtype=torch.float32, device=chunk_rewards.device
            )

    if success_frame is None:
        # Default to zeros if no success info available
        if chunk_rewards.dim() == 2:
            success_frame = torch.zeros(
                chunk_rewards.shape[0],
                chunk_rewards.shape[1],
                dtype=torch.float32,
                device=chunk_rewards.device,
            )
        else:
            success_frame = torch.zeros(
                chunk_rewards.shape[0],
                dtype=torch.float32,
                device=chunk_rewards.device,
            )

    return success_frame
