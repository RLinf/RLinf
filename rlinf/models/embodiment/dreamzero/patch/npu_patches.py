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

"""DreamZero attention, rotary embedding and real-valued RoPE for Ascend SFT."""

import copy
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from transformers.feature_extraction_utils import BatchFeature


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Compute timestep embeddings in FP32 without NPU FP64 operations."""
    assert dim % 2 == 0, "Sinusoidal embedding dimension must be even."
    half = dim // 2
    frequencies = torch.pow(
        10000, -torch.arange(half, device=position.device, dtype=torch.float32) / half
    )
    angles = torch.outer(position.float(), frequencies)
    return torch.cat((angles.cos(), angles.sin()), dim=1)


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """Store adjacent cosine/sine pairs in FP32 instead of complex128.

    Tables are built on CPU with the upstream precision, then transferred by
    CausalWanModel._create_freqs. Its spatial concatenation preserves each pair.
    """
    assert dim % 2 == 0, "Rotary embedding dimension must be even."
    angles = torch.outer(
        torch.arange(max_seq_len, device="cpu"),
        1.0
        / torch.pow(
            theta, torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim
        ),
    )
    return torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(-2).float()


def _rotate(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    pairs = x.float().unflatten(-1, (-1, 2))
    frequencies = freqs.reshape(1, x.shape[1], 1, -1, 2)
    real, imag = pairs.unbind(-1)
    cosine, sine = frequencies.unbind(-1)
    return torch.stack(
        (real * cosine - imag * sine, real * sine + imag * cosine), dim=-1
    ).flatten(-2)


def rope_apply(
    x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Apply the precomputed video rotations with adjacent channel pairing."""
    return _rotate(x, freqs)


def rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int | None = None,
    num_state_per_block: int | None = None,
) -> torch.Tensor:
    """Rotate video, action and state tokens in the upstream token order."""
    if action_register_length is not None:
        assert num_action_per_block is not None and num_state_per_block is not None
        chunks = action_register_length // (num_action_per_block + num_state_per_block)
        action_freqs = freqs_action[: chunks * num_action_per_block].unsqueeze(1)
        state_freqs = freqs_state[: chunks * num_state_per_block].unsqueeze(1)
        freqs = torch.cat((freqs, action_freqs, state_freqs), dim=0)
    return _rotate(x, freqs)


def causal_rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
) -> torch.Tensor:
    """Rotate a cached block using its action/state frequency offsets."""
    if action_register_length is not None:
        assert action_register_length == num_action_per_block + num_state_per_block
        action_start = action_state_index * num_action_per_block
        state_start = action_state_index * num_state_per_block
        freqs = torch.cat(
            (
                freqs,
                freqs_action[
                    action_start : action_start + num_action_per_block
                ].unsqueeze(1),
                freqs_state[state_start : state_start + num_state_per_block].unsqueeze(
                    1
                ),
            ),
            dim=0,
        )
    return _rotate(x, freqs)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] | None = None,
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
) -> torch.Tensor:
    """Run BSHD attention through SDPA, preserving padding and FA2 masks.

    Causal and window masks align at the bottom right when Q and K lengths
    differ. The unmasked SFT path avoids allocating a quadratic mask.
    """
    output_dtype = q.dtype
    compute_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else dtype
    q, k, v = (t.transpose(1, 2).to(compute_dtype) for t in (q, k, v))
    if q_scale is not None:
        q = q * q_scale
    if q.shape[1] != k.shape[1]:
        assert q.shape[1] % k.shape[1] == 0, "Q heads must be divisible by KV heads."
        repeats = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    mask = None
    window_size = (-1, -1) if window_size is None else window_size
    if q_lens is not None or k_lens is not None or causal or window_size != (-1, -1):
        batch, _, query_length, _ = q.shape
        key_length = k.shape[2]
        q_lens = (
            torch.full((batch,), query_length, device=q.device)
            if q_lens is None
            else q_lens.to(q.device)
        )
        k_lens = (
            torch.full((batch,), key_length, device=q.device)
            if k_lens is None
            else k_lens.to(q.device)
        )
        qi = torch.arange(query_length, device=q.device)[None, :, None]
        ki = torch.arange(key_length, device=q.device)[None, None, :]
        mask = (qi < q_lens[:, None, None]) & (ki < k_lens[:, None, None])
        aligned_qi = qi + (k_lens - q_lens)[:, None, None]
        if causal:
            mask = mask & (ki <= aligned_qi)
        left, right = window_size
        if left >= 0:
            mask = mask & (ki >= aligned_qi - left)
        if right >= 0:
            mask = mask & (ki <= aligned_qi + right)
        mask = mask.unsqueeze(1)

    # Ascend SDPA needs a nonempty row; masked rows must have zero gradients.
    valid_rows = None
    if mask is not None:
        valid_rows = mask.any(dim=-1, keepdim=True)
        mask = mask | ~valid_rows

    output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=softmax_scale
    )
    if valid_rows is not None:
        output = output.masked_fill(~valid_rows, 0.0)
    return output.transpose(1, 2).contiguous().to(output_dtype)


def ensure_vae_on_device(self: torch.nn.Module, ref_tensor: torch.Tensor) -> None:
    """Let FSDP2 manage sharded VAE placement; lazily move an unsharded VAE."""
    if getattr(self, "_vae_device_ready", False):
        return
    # FSDP2 owns DTensor placement and mixed precision. Module.to() here can
    # mix local tensors and DTensors or invalidate the sharding state.
    if any(isinstance(param, DTensor) for param in self.vae.parameters()):
        return
    self.vae.to(device=ref_tensor.device, dtype=torch.bfloat16)
    self.vae.eval()
    self._vae_device_ready = True


def wrap_action_loss_mask(forward: Callable) -> Callable:
    """Adapt per-sample flags to the vendor's [B, T, D] action-loss broadcast."""

    @wraps(forward)
    def wrapped(
        self: torch.nn.Module,
        backbone_output: "BatchFeature",
        action_input: "BatchFeature",
    ) -> "BatchFeature":
        # The pinned vendor inserts one axis with has_real_action[:, None].
        # Supply [B, 1] so it produces [B, 1, 1], including when B == T.
        # Copy the mapping so repeated forwards never change caller inputs.
        if action_input.has_real_action.ndim == 1:
            action_input = copy.copy(action_input)
            action_input["has_real_action"] = action_input.has_real_action[:, None]
        return forward(self, backbone_output, action_input)

    return wrapped


def apply_npu_patches(patcher) -> None:
    """Enable CUDA API migration and register DreamZero patches on Ascend."""
    from rlinf.scheduler import AcceleratorType, Worker

    if Worker.accelerator_type != AcceleratorType.NPU:
        return

    # CUDA API migration for DreamZero worker process.
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    source = "groot.vla.model.dreamzero.modules"
    target = "rlinf.models.embodiment.dreamzero.patch.npu_patches"
    for name in (
        "sinusoidal_embedding_1d",
        "rope_params",
        "rope_apply",
        "rope_action_apply",
    ):
        patcher.add_patch(f"{source}.wan2_1_submodule.{name}", f"{target}.{name}")
    for module in ("attention", "wan2_1_attention"):
        patcher.add_patch(
            f"{source}.{module}.flash_attention", f"{target}.flash_attention"
        )
    patcher.add_patch(
        f"{source}.wan_video_dit_action_casual_chunk.causal_rope_action_apply",
        f"{target}.causal_rope_action_apply",
    )
    action_head = "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead"
    patcher.add_patch(
        f"{action_head}._ensure_vae_on_device", f"{target}.ensure_vae_on_device"
    )
    patcher.add_wrapper(f"{action_head}.forward", wrap_action_loss_mask)
