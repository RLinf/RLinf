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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from einops import rearrange

def npu_rmsnorm_forward(self, hidden_states):
    return torch_npu.npu_rms_norm(
        hidden_states, self.weight, epsilon=self.variance_epsilon
    )[0]


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def get_radio_compatible_cuda_capability_on_npu(
    *_args, **_kwargs
) -> tuple[int, int]:
    """Bypass RADIO's import-time CUDA capability check on Ascend.

    Isaac-GR00T N1.5's radio_model calls
    torch.cuda.get_device_capability() without checking CUDA availability.
    Ascend has no CUDA capability, but RADIO uses a separate NPU attention
    path, so the CUDA-only check is irrelevant.

    Return RADIO's minimum accepted CUDA capability (Ampere 8.0). This value
    is only a compatibility sentinel and does not describe the Ascend device.
    """
    return (8, 0)

