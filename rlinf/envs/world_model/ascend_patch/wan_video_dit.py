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

"""NPU-capable flash_attention drop-in for the Wan video DiT.

Replaces ``diffsynth.models.wan_video_dit.flash_attention`` via the shared
``Patcher`` in ``WanEnv._build_pipeline``.  On Ascend NPU the mindiesd
optimised kernel is used; on GPU the implementation falls back to flash-attn
3/2, SageAttention, or torch SDPA in that order -- identical to upstream
diffsynth.  All optional backends are guarded by try/except so this module
imports cleanly without any of them present.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import torch_npu  # noqa: F401
    from mindiesd.layers.flash_attn.attention_forward import attention_forward
    from mindiesd import rotary_position_embedding
    MINDIESD_ENABLE = True
except Exception:
    MINDIESD_ENABLE = False

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode: bool = False,
) -> torch.Tensor:
    """Drop-in replacement for ``diffsynth.models.wan_video_dit.flash_attention``.

    ``q``/``k``/``v`` are packed as ``(batch, seq, num_heads * head_dim)``; the
    output matches that layout.  Backend priority: mindiesd (NPU) → flash-attn
    3 → flash-attn 2 → SageAttention → SDPA.
    """
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif MINDIESD_ENABLE:
        if q.shape[1] < 4000:
            q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
            x = attention_forward(
                q, k, v,
                opt_mode="manual",
                op_type="fused_attn_score",
                layout="BSND",
            )
        else:
            q = rearrange(q, "b s (n d) -> b s n d", n=num_heads).contiguous()
            k = rearrange(k, "b s (n d) -> b s n d", n=num_heads).contiguous()
            v = rearrange(v, "b s (n d) -> b s n d", n=num_heads).contiguous()
            x = attention_forward(
                q, k, v,
                opt_mode="manual",
                op_type="ascend_laser_attention",
                layout="BNSD", # only changes internal layout
            )
        x = rearrange(x, "b s n d -> b s (n d)")
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def rope_apply(x, freqs, num_heads):
    if MINDIESD_ENABLE:
        x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
        cos, sin = torch.chunk(
            torch.view_as_real(freqs.to(torch.complex64)),
            2,
            dim=-1
        )
        cos = cos.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)
        sin = sin.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)
        x_out = rotary_position_embedding(
            x,
            cos,
            sin,
            rotated_mode="rotated_interleaved",
            fused=True
        )
        return x_out.flatten(2)
    else:
        x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
        x_out = torch.view_as_complex(x.to(torch.float64).reshape(
            x.shape[0], x.shape[1], x.shape[2], -1, 2))
        x_out = torch.view_as_real(x_out * freqs).flatten(2)
        return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        if MINDIESD_ENABLE:
            return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight