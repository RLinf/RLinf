# Copyright 2026 The RLinf Authors.
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

"""Compatibility fixes for supported FSDP1 versions."""

import torch
from packaging import version
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp._common_utils import clean_tensor_name
from torch.distributed.fsdp._runtime_utils import (
    _cast_buffers_to_dtype_and_device,
    _get_buffers_and_dtypes_for_computation,
)


def apply_buffer_dtype_compatibility(model: FSDP) -> None:
    """Preserve computation and checkpoint buffer dtypes for parameterless roots.

    Before PyTorch 2.7, root pre-forward restores every buffer to its original
    dtype when the root has no parameter handle. FSDP also skips checkpoint
    dtype restoration for these roots, including in newer versions.

    Args:
        model: The fully constructed root FSDP1 wrapper.
    """
    if model._handle is not None or not any(
        torch.is_floating_point(buffer) for buffer in model.buffers()
    ):
        return

    def restore_buffer_dtypes(module: nn.Module, args: tuple) -> None:
        if not model._is_root or any(
            handle._force_full_precision for handle in model._all_handles
        ):
            return
        buffers, dtypes = _get_buffers_and_dtypes_for_computation(model, model)
        _cast_buffers_to_dtype_and_device(buffers, dtypes, model.compute_device)

    def restore_checkpoint_dtypes(
        module: nn.Module, state_dict: dict, prefix: str, local_metadata: dict
    ) -> None:
        if (
            model._state_dict_type == StateDictType.LOCAL_STATE_DICT
            and model.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            return
        buffers, dtypes = _get_buffers_and_dtypes_for_computation(model, model)
        mixed_precision_buffers = {
            id(buffer) for buffer, dtype in zip(buffers, dtypes) if dtype is not None
        }
        for name, buffer in model.named_buffers():
            if id(buffer) not in mixed_precision_buffers:
                continue
            name = clean_tensor_name(name)
            dtype = model._buffer_name_to_orig_dtype[name]
            key = prefix + name
            buffer = state_dict.get(key)
            if (
                buffer is not None
                and torch.is_floating_point(buffer)
                and buffer.dtype != dtype
            ):
                state_dict[key] = buffer.to(dtype=dtype)

    if version.parse(torch.__version__) < version.parse("2.7.0"):
        # The wrapped module's hook runs after FSDP's native root pre-forward.
        model.module.register_forward_pre_hook(restore_buffer_dtypes, prepend=True)
    model._register_state_dict_hook(restore_checkpoint_dtypes)
