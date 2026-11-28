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

"""Real-valued encodings and fused operators for FastWAM on Ascend.

Call before constructing the experts. Frequency caches keep interleaved
cosine/sine values in their final dimension so upstream slicing, concatenation
and device transfers work without complex tensors.
"""

from collections.abc import Callable
from functools import wraps

import torch

_MODULE = "rlinf.models.embodiment.fastwam.npu_patches"
_WAN_DIT = "fastwam.models.wan22.wan_video_dit"


def precompute_freqs_real(
    dim: int, end: int = 1024, theta: float = 10000.0
) -> torch.Tensor:
    """Build an interleaved cosine/sine cache on CPU."""
    # Compute once at upstream precision, then transfer only FP32 values.
    frequencies = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2] / dim)
    )
    angles = torch.outer(torch.arange(end, dtype=torch.float64), frequencies)
    return torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(-2).float()


def rope_apply_real(
    x: torch.Tensor, freqs: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Rotate interleaved query/key pairs with an FP32 frequency cache."""
    batch, sequence, hidden = x.shape
    pairs = x.float().reshape(batch, sequence, num_heads, hidden // num_heads // 2, 2)
    rotations = freqs.reshape(sequence, 1, -1, 2).to(device=x.device)
    real, imag = pairs.unbind(-1)
    cosine, sine = rotations.unbind(-1)
    return (
        torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), dim=-1)
        .reshape_as(x)
        .to(x.dtype)
    )


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Compute time embeddings in FP32 for the NPU worker."""
    dtype = torch.float32
    frequencies = torch.pow(
        10000,
        -torch.arange(dim // 2, dtype=dtype, device=position.device) / (dim // 2),
    )
    angles = torch.outer(position.to(dtype), frequencies)
    return torch.cat((angles.cos(), angles.sin()), dim=1).to(position.dtype)


def rope_apply_fused(
    x: torch.Tensor, freqs: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Apply fused interleaved RoPE on supported NPU tensors.

    Keep the rotation in FP32, as in the unfused implementation. Learned
    frequencies, unsupported head sizes and JIT mode use the reference path.
    Frequency caches retain the same representation in both modes.
    """
    head_dim = x.shape[-1] // num_heads
    if (
        x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or freqs.requires_grad
        or head_dim % 2
        or not 0 < head_dim < 896
        or x.shape[0] * num_heads >= 1000
        or not torch.npu.is_jit_compile_false()
    ):
        return rope_apply_real(x, freqs, num_heads)

    import torch_npu

    batch, sequence, _ = x.shape
    cached = getattr(freqs, "_fastwam_fused_rotations", None)
    version = None if freqs.is_inference() else freqs._version
    if cached is not None and cached[0] == version and cached[1].device == x.device:
        _, cosine, sine = cached
    else:
        cosine, sine = _expand_rotations(freqs.to(x.device))
    output = torch_npu.npu_rotary_mul(
        x.float().reshape(batch, sequence, num_heads, head_dim),
        cosine,
        sine,
        rotary_mode="interleave",
    )
    return output.reshape_as(x).to(x.dtype)


def _expand_rotations(freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sequence, head_dim = freqs.shape[0], freqs.shape[-1]
    rotations = freqs.reshape(sequence, head_dim).float()
    cosine = rotations[:, 0::2].repeat_interleave(2, dim=-1)
    sine = rotations[:, 1::2].repeat_interleave(2, dim=-1)
    return (
        cosine.reshape(1, sequence, 1, head_dim),
        sine.reshape(1, sequence, 1, head_dim),
    )


def _wrap_pre_dit(original: Callable) -> Callable:
    if hasattr(original, "ORIG__fastwam"):
        return original

    @wraps(original)
    def pre_dit(self, *args, **kwargs):
        state = original(self, *args, **kwargs)
        freqs = state["freqs"]
        if not freqs.requires_grad:
            # Each pre_dit call creates a fresh, immutable frequency tensor.
            # Reuse its expanded trigonometry across Q/K and all expert blocks.
            # Attaching to the tensor bounds lifetime to this forward/recompute;
            # no global cache or storage-pointer identity is involved.
            version = None if freqs.is_inference() else freqs._version
            freqs._fastwam_fused_rotations = (version, *_expand_rotations(freqs))
        return state

    pre_dit.ORIG__fastwam = original
    return pre_dit


def npu_rmsnorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply fused RMSNorm to NPU inputs and weights with matching dtypes."""
    import torch_npu

    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]


def apply_npu_patches(patcher) -> None:
    """Register FastWAM's NPU encodings and fused operators with RLinf's Patcher.

    Call only in NPU workers, between ``patcher.clear()`` and ``patcher.apply()``,
    before building models. Registration leaves upstream functions unchanged
    until ``apply``.
    Repeating that sequence is idempotent. Existing models with complex caches
    must not share this process. Fusion is always enabled for supported NPU
    tensors in video/action DiT experts; T5 retains its upstream operators.
    """
    # Patcher updates all imported aliases by object identity, including the
    # ActionDiT encodings and MoT's rope_apply. Register each original only once.
    patcher.add_patch(
        f"{_WAN_DIT}.precompute_freqs_cis", f"{_MODULE}.precompute_freqs_real"
    )
    patcher.add_patch(
        f"{_WAN_DIT}.sinusoidal_embedding_1d", f"{_MODULE}.sinusoidal_embedding_1d"
    )
    patcher.add_patch(f"{_WAN_DIT}.rope_apply", f"{_MODULE}.rope_apply_fused")
    for expert in (
        f"{_WAN_DIT}.WanVideoDiT",
        "fastwam.models.wan22.action_dit.ActionDiT",
    ):
        patcher.add_wrapper(f"{expert}.pre_dit", _wrap_pre_dit)
    patcher.add_patch(f"{_WAN_DIT}.RMSNorm.forward", f"{_MODULE}.npu_rmsnorm_forward")
