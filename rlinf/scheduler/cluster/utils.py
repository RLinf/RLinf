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


import dataclasses
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Optional, Protocol

import torch

# Type for a single tensor field value in a dataclass (used for send/recv).
TensorFieldValue = (
    torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor, ...]
    | dict[str, torch.Tensor]
)
# Metadata for flatten/unflatten: (field_name, 'tensor'|'list'|'tuple'|'dict', None|length|list_of_keys).
DataclassTensorFieldsMetadata = list[
    tuple[str, str, Optional[int] | Optional[list[str]]]
]


class DataclassProtocol(Protocol):
    """Protocol for dataclasses to enable type checking."""

    __dataclass_fields__: dict
    __dataclass_params__: dict
    __post_init__: Optional[Callable]


def parse_rank_config(
    rank_config: str | int,
    available_ranks: Optional[list[int]] = None,
    rank_type: Optional[str] = None,
) -> list[int]:
    """Parse a rank configuration string into a list of ranks.

    Args:
        rank_config (str | int): The rank configuration string, e.g., "0-3,5,7-9" or "all".
        available_ranks (Optional[list[int]]): The list of available ranks.
        rank_type (Optional[str]): The type of rank being parsed (for error messages).

    Returns:
        list[int]: The list of ranks.
    """
    ranks = set()
    if available_ranks is not None:
        available_ranks = sorted(available_ranks)
    # If the rank config is a single number
    # Omegaconf will parse it as an integer instead of a string
    rank_config = str(rank_config)
    if rank_config.lower() == "all":
        assert available_ranks is not None, (
            'When rank_config is "all", available_ranks must be provided.'
        )
        ranks = list(set(available_ranks))
    else:
        # First split by comma
        rank_ranges = rank_config.split(",")
        for rank_range in rank_ranges:
            rank_range = rank_range.strip()
            if rank_range == "":
                continue
            # Then split by hyphen to get the start and end of the range
            rank_range = rank_range.split("-")
            try:
                if len(rank_range) == 1:
                    start_rank = int(rank_range[0])
                    end_rank = start_rank
                elif len(rank_range) == 2:
                    start_rank = int(rank_range[0])
                    end_rank = int(rank_range[1])
                else:
                    raise ValueError
            except (ValueError, IndexError):
                raise ValueError(
                    f'Invalid rank format {rank_config} for {rank_type}, expected format: "a,b,c-d" or "all"'
                )
            assert end_rank >= start_rank, (
                f"Start rank {start_rank} must be less than or equal to end rank {end_rank} in rank config {rank_config} for {rank_type}."
            )
            if available_ranks is not None:
                assert available_ranks[0] <= start_rank <= available_ranks[-1], (
                    f'Start rank {start_rank} in rank config string "{rank_config}" must be within the available {rank_type if rank_type is not None else ""} ranks {available_ranks}.'
                )
                assert available_ranks[0] <= end_rank <= available_ranks[-1], (
                    f'End rank {end_rank} in rank config string "{rank_config}" must be within the available {rank_type if rank_type is not None else ""} ranks {available_ranks}.'
                )
            ranks.update(range(start_rank, end_rank + 1))
    ranks = list(ranks)
    return sorted(ranks)


def dataclass_arg_check(
    dataclass: DataclassProtocol,
    kwargs: dict,
    no_check_unknown: bool = False,
    error_suffix: str = "",
):
    """Check if the kwargs contain only valid fields for the given dataclass.

    Args:
        dataclass (DataclassProtocol): The dataclass to check against.
        kwargs (dict): The keyword arguments to check.
        no_check_unknown (bool): Whether to skip checking for unknown fields.
        error_suffix (str): Additional error message suffix.
    """
    args = set(kwargs.keys())
    valid_args = set(dataclass.__dataclass_fields__.keys())

    missing_args = valid_args - args
    unknown_args = args - valid_args
    missing_required_args = []
    for missing_arg in missing_args:
        field_info = dataclass.__dataclass_fields__[missing_arg]
        if (
            field_info.default is dataclasses.MISSING
            and field_info.default_factory is dataclasses.MISSING
        ):
            missing_required_args.append(missing_arg)

    assert not missing_required_args, (
        f"Missing fields '{missing_required_args}' detected {error_suffix}. Only got: {kwargs.keys()}."
    )
    if not no_check_unknown:
        assert not unknown_args, (
            f"Unknown fields '{unknown_args}' detected {error_suffix}. Valid fields are: {valid_args}."
        )

    return missing_required_args, unknown_args, valid_args


def extract_dataclass_tensor_fields(
    obj: Any,
) -> tuple[
    dict[str, TensorFieldValue], list[torch.Tensor], DataclassTensorFieldsMetadata
]:
    """Extract fields of a dataclass that are tensors or list/tuple/dict of tensors.

    Supported field types:
        - torch.Tensor
        - list[torch.Tensor] (all elements must be tensors)
        - tuple[torch.Tensor, ...] (all elements must be tensors)
        - dict[str, torch.Tensor] (all values must be tensors)

    Returns:
        (fields_dict, tensors_list, metadata): fields_dict maps field names to their value(s);
        tensors_list is a flat list of all tensors in field order for send/wire format;
        metadata describes each field's kind for unflatten on recv.
    """
    if not is_dataclass(obj):
        return {}, [], []
    result: dict[str, TensorFieldValue] = {}
    tensors_list: list[torch.Tensor] = []
    metadata: DataclassTensorFieldsMetadata = []
    for f in fields(obj):
        val = getattr(obj, f.name)
        if isinstance(val, torch.Tensor):
            result[f.name] = val
            tensors_list.append(val)
            metadata.append((f.name, "tensor", None))
        elif isinstance(val, (list, tuple)) and all(
            isinstance(item, torch.Tensor) for item in val
        ):
            # Preserve list vs tuple; flatten/unflatten will distinguish for wire format.
            result[f.name] = val
            tensors_list.extend(val)
            kind = "list" if isinstance(val, list) else "tuple"
            metadata.append((f.name, kind, len(val)))
        elif isinstance(val, dict) and all(
            isinstance(v, torch.Tensor) for v in val.values()
        ):
            result[f.name] = val
            keys = list(val.keys())
            tensors_list.extend(val[k] for k in keys)
            metadata.append((f.name, "dict", keys))
    return result, tensors_list, metadata


def unflatten_dataclass_tensor_fields(
    metadata: DataclassTensorFieldsMetadata,
    flat_tensors: list[torch.Tensor],
) -> dict[str, TensorFieldValue]:
    """Reconstruct a dict of tensor fields from metadata and flat tensor list (from recv)."""
    result: dict[str, TensorFieldValue] = {}
    idx = 0
    for name, kind, extra in metadata:
        if kind == "tensor":
            result[name] = flat_tensors[idx]
            idx += 1
        elif kind == "list":
            n = extra if isinstance(extra, int) else 0
            result[name] = flat_tensors[idx : idx + n]
            idx += n
        elif kind == "tuple":
            n = extra if isinstance(extra, int) else 0
            result[name] = tuple(flat_tensors[idx : idx + n])
            idx += n
        elif kind == "dict":
            keys = extra if isinstance(extra, list) else []
            result[name] = dict(zip(keys, flat_tensors[idx : idx + len(keys)]))
            idx += len(keys)
        else:
            raise ValueError(f"Unknown metadata kind for field {name}: {kind}")
    if idx != len(flat_tensors):
        raise ValueError(
            f"Metadata consumed {idx} tensors but flat list has {len(flat_tensors)}"
        )
    return result
