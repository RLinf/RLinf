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

import hashlib
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn


def encode_phrases_to_2d_tensor(phrases, pad_value=-1, fixed_length=200):
    original_max_length = max(len(phrase) for phrase in phrases) if phrases else 0
    if original_max_length > fixed_length:
        raise ValueError(
            f"Original max length {original_max_length} is greater than fixed length {fixed_length}"
        )
    max_length = fixed_length if fixed_length is not None else original_max_length

    padded_encoded = []
    for phrase in phrases:
        code_points = [ord(c) for c in phrase]
        truncated = code_points[:max_length]
        padded = truncated + [pad_value] * (max_length - len(truncated))
        padded_encoded.append(padded)

    tensor_2d = torch.tensor(padded_encoded, dtype=torch.int32)
    return tensor_2d


def decode_2d_tensor_to_phrases(tensor, pad_value=-1):
    code_points_list = tensor.cpu().numpy().tolist()

    phrases = []
    for codes in code_points_list:
        filtered_codes = [c for c in codes if c != pad_value]
        phrase = "".join([chr(c) for c in filtered_codes])
        phrases.append(phrase)
    return phrases


def hash_array_or_tensor(input_data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(input_data, torch.Tensor):
        # Detach (if needed), move to CPU, and ensure contiguous memory
        data_cpu = input_data.detach().cpu().contiguous()
        # Convert to NumPy array (for consistent byte extraction)
        if data_cpu.dtype == torch.bfloat16:
            data_cpu = data_cpu.float()
        data_np = data_cpu.numpy()
    elif isinstance(input_data, np.ndarray):
        # Ensure contiguous memory (mimic PyTorch's .contiguous())
        data_np = np.ascontiguousarray(input_data)
    else:
        raise TypeError(
            f"Unsupported type: {type(input_data)}. Use torch.Tensor or np.ndarray."
        )

    data_bytes = data_np.tobytes()
    sha256_hash = hashlib.sha256(
        data_bytes
    ).digest()  # Directly get bytes (avoids hex conversion)

    return torch.tensor([int(byte) for byte in sha256_hash], dtype=torch.uint8)


def replace_dropout_with_identity(model):
    """
    Find all dropout layers and replace them with nn.Identity()
    """
    for name, module in list(model.named_modules()):
        if isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout),
        ):
            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                child_name = name.rsplit(".", 1)[1]
                parent_module = dict(model.named_modules())[parent_name]
            else:
                parent_module = model
                child_name = name
            setattr(parent_module, child_name, nn.Identity())


# Helper functions
def unsqueeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)  # Fixed
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Fixed: only remove batch dim
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Fixed: only remove batch dim
        else:
            squeezed_data[k] = v
    return squeezed_data
