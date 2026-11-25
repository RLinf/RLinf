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

"""Preserve Qwen3.5's GDN norm through zero-centered actor weight sync."""

import torch

try:
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
except ModuleNotFoundError as error:
    if not error.name.startswith("sglang.kernels"):
        raise
    from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm


class Qwen35RMSNormGated(RMSNorm):
    """Keep the effective ``1 + actor_gamma`` scale in FP32 on rollout."""

    def __init__(self, hidden_size: int, **kwargs):
        kwargs["dtype"] = torch.float32
        super().__init__(hidden_size, **kwargs)
