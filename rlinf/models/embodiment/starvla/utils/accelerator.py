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

"""Gaussian action-distribution helper for the starVLA RLinf wrapper.

starVLA upstream targets CUDA and relies on CUDA kernel coverage. On other
accelerators (Ascend NPU via ``torch_npu``, Intel XPU, ...) some of the dtypes
it leaves in place have no kernel, so the policy distribution is built in
float32 here.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.distributions.normal import Normal


def build_gaussian(
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = torch.float32,
) -> Normal:
    """Build a Gaussian action distribution with both arguments in ``dtype``.

    ``mean`` inherits the backbone dtype (bfloat16 when autocast did not upcast
    it) and ``std`` inherits the policy parameter dtype. Sampling from a
    bfloat16 ``Normal`` calls ``torch.normal(mean, std)``, which has no
    bfloat16 kernel on Ascend ("tensor mean not implemented for DT_BFLOAT16"),
    so both are cast up front. Downstream log-probs, entropies and values are
    consumed as float32 anyway, so this costs nothing on CUDA.

    Args:
        mean: Distribution location, any floating dtype.
        std: Distribution scale, broadcastable against ``mean``.
        dtype: Dtype to build the distribution in; ``None`` keeps the inputs
            unchanged.

    Returns:
        The ``Normal`` distribution over the (broadcast) action shape.
    """
    if dtype is not None:
        mean = mean.to(dtype=dtype)
        std = std.to(dtype=dtype)
    return Normal(mean, std)
