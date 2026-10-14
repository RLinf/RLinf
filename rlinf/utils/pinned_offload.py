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

"""Pinned-memory offload context manager for accelerator-agnostic model offload."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

import torch


_patch_lock = Lock()
_active_contexts = 0
_original_tensor_to = torch.Tensor.to


def _pinned_tensor_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    with _patch_lock:
        original_to = _original_tensor_to

    device = None
    dtype = None

    if args:
        first = args[0]
        # Tensor.to also accepts a dtype or another tensor as the first argument.
        if isinstance(first, (str, torch.device, int)):
            device = first
            if len(args) >= 2 and isinstance(args[1], (torch.dtype, type(None))):
                dtype = args[1]
        elif isinstance(first, torch.dtype):
            dtype = first
    if device is None:
        device = kwargs.get("device")
    if dtype is None:
        dtype = kwargs.get("dtype")

    if (
        device is not None
        and dtype is None
        and torch.device(device).type == "cpu"
        and self.device.type != "cpu"
    ):
        buffer = torch.empty_like(self, device="cpu", pin_memory=True)
        buffer.copy_(self)
        return buffer

    return original_to(self, *args, **kwargs)


@contextmanager
def pinned_offload_context() -> Iterator[None]:
    """Use pinned memory for accelerator-to-CPU ``Tensor.to`` calls.

    The patch is process-wide while any context is active. Context entries and
    exits are reference-counted so overlapping contexts restore ``Tensor.to``
    after the last context exits. CPU-to-accelerator copies are unchanged.

    The returned buffers become the tensors' CPU storage and remain pinned for
    the lifetime of those tensors. The host-memory cost is therefore the full
    offloaded tensor size per worker (about 22.5 GiB for the tested Wan2.2-5B
    pipeline).
    """
    global _active_contexts, _original_tensor_to

    with _patch_lock:
        if _active_contexts == 0:
            _original_tensor_to = torch.Tensor.to
            torch.Tensor.to = _pinned_tensor_to
        _active_contexts += 1
    try:
        yield
    finally:
        with _patch_lock:
            _active_contexts -= 1
            if _active_contexts == 0 and torch.Tensor.to is _pinned_tensor_to:
                torch.Tensor.to = _original_tensor_to
