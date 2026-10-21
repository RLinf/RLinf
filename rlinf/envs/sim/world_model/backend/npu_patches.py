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

"""Ascend kernels for the Wan video DiT built by :class:`WanBackend`."""

import torch
import torch.nn.functional as F
from einops import rearrange

from rlinf.scheduler import AcceleratorType, Worker
from rlinf.utils.logging import get_logger

try:
    import torch_npu
    from mindiesd import rotary_position_embedding
    from mindiesd.layers.flash_attn.attention_forward import attention_forward
except Exception as error:
    # Absent off Ascend. On Ascend the error (a missing MindIE-SD, a CANN
    # mismatch) is logged when the patches are requested.
    _KERNEL_IMPORT_ERROR = error
else:
    _KERNEL_IMPORT_ERROR = None

_LASER_ATTENTION_MIN_QUERY_TOKENS = 4000


def npu_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode: bool = False,
) -> torch.Tensor:
    """``wan_video_dit.flash_attention`` via MindIE-SD attention.

    ``q``, ``k`` and ``v`` are packed as ``(batch, seq, num_heads * head_dim)``
    and the output keeps that layout. ``compatibility_mode`` keeps diffsynth's
    SDPA path.
    """
    if compatibility_mode:
        q, k, v = (rearrange(t, "b s (n d) -> b n s d", n=num_heads) for t in (q, k, v))
        x = F.scaled_dot_product_attention(q, k, v)
        return rearrange(x, "b n s d -> b s (n d)")

    q, k, v = (rearrange(t, "b s (n d) -> b s n d", n=num_heads) for t in (q, k, v))
    if q.shape[1] < _LASER_ATTENTION_MIN_QUERY_TOKENS:
        op_type, layout = "fused_attn_score", "BSND"
    else:
        # The inputs stay BSND; ``layout`` only selects the kernel's internal one.
        op_type, layout = "ascend_laser_attention", "BNSD"
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    x = attention_forward(q, k, v, opt_mode="manual", op_type=op_type, layout=layout)
    return rearrange(x, "b s n d -> b s (n d)")


def npu_rope_apply(
    x: torch.Tensor, freqs: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """``wan_video_dit.rope_apply`` via MindIE-SD's fused rotary kernel.

    ``freqs`` holds complex rotations shaped ``(seq, 1, head_dim // 2)``. The
    kernel takes float32 cos/sin where diffsynth rotates in float64; the result
    returns in ``x``'s dtype, as diffsynth's does.
    """
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    cos, sin = torch.chunk(torch.view_as_real(freqs.to(torch.complex64)), 2, dim=-1)
    # Each angle rotates one interleaved (real, imag) pair, so it covers two lanes.
    cos = cos.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)
    sin = sin.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)
    x_out = rotary_position_embedding(
        x, cos, sin, rotated_mode="rotated_interleaved", fused=True
    )
    return x_out.flatten(2).to(x.dtype)


def npu_rmsnorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """``wan_video_dit.RMSNorm.forward`` via the fused ``npu_rms_norm`` kernel."""
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]


_WAN_DIT = "diffsynth.models.wan_video_dit"


def apply_npu_patches(patcher) -> None:
    """Register the Ascend kernels for building a Wan pipeline.

    No-op off NPU. Call before ``patcher.apply()`` and before the pipeline is
    constructed; the kernels stay for the process's lifetime. When MindIE-SD
    or ``torch_npu`` cannot be imported, logs the import error and leaves
    diffsynth's operators in place.
    """
    if Worker.accelerator_type != AcceleratorType.NPU:
        return
    if _KERNEL_IMPORT_ERROR is not None:
        get_logger().warning(
            "Wan keeps diffsynth's attention, RoPE and RMSNorm on Ascend because "
            f"the MindIE-SD kernels failed to import: {_KERNEL_IMPORT_ERROR!r}"
        )
        return

    patcher.add_patch(f"{_WAN_DIT}.flash_attention", f"{__name__}.npu_flash_attention")
    patcher.add_patch(f"{_WAN_DIT}.rope_apply", f"{__name__}.npu_rope_apply")
    patcher.add_patch(f"{_WAN_DIT}.RMSNorm.forward", f"{__name__}.npu_rmsnorm_forward")
