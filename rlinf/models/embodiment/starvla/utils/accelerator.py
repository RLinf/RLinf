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

"""Accelerator-neutral autocast and action distribution for starVLA.

starVLA upstream targets CUDA. On other accelerators (Ascend NPU via
``torch_npu``, Intel XPU, ...) a ``"cuda"`` autocast silently does nothing and
some dtypes it leaves in place have no kernel, so the starVLA paths build their
autocast and policy distribution through these helpers.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch
from torch.distributions.normal import Normal

from rlinf.scheduler import Worker


def accelerator_autocast(dtype: torch.dtype) -> AbstractContextManager:
    """Autocast to ``dtype`` on the accelerator this worker resolved.

    Args:
        dtype: Autocast dtype, e.g. ``torch.bfloat16`` for the backbone or
            ``torch.float32`` to undo it around an action head.

    Returns:
        A ``torch.autocast`` on ``Worker.torch_device_type``, or a no-op context
        when the worker has no accelerator.
    """
    device_type = Worker.torch_device_type
    if device_type is None:
        return nullcontext()
    return torch.autocast(device_type, dtype=dtype)


def build_gaussian(mean: torch.Tensor, std: torch.Tensor) -> Normal:
    """Build the Gaussian action distribution in float32.

    ``mean`` inherits the backbone dtype (bfloat16 when autocast did not upcast
    it) and ``std`` inherits the policy parameter dtype. Sampling from a
    bfloat16 ``Normal`` calls ``torch.normal(mean, std)``, which has no
    bfloat16 kernel on Ascend ("tensor mean not implemented for DT_BFLOAT16"),
    so both are cast up front. Downstream log-probs, entropies and values are
    consumed as float32 anyway, so this costs nothing on CUDA.

    Args:
        mean: Distribution location, any floating dtype.
        std: Distribution scale, broadcastable against ``mean``.

    Returns:
        The float32 ``Normal`` over the (broadcast) action shape.
    """
    return Normal(mean.to(dtype=torch.float32), std.to(dtype=torch.float32))
