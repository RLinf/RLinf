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

"""Ascend NPU patches for building GR00T N1.6.

N1.6's backbone is Eagle3-VL: a Qwen3 language model and a SigLIP2 vision tower,
loaded from code vendored in Isaac-GR00T through ``trust_remote_code``. That code
requires flash attention rather than offering it as an option — the backbone
asserts it — and its SigLIP2 attention reads the ``flash_attn`` entry points from
its own module namespace. On Ascend the ``flash_attn`` package does not exist, so
those names are never bound and the vision forward pass fails. The fix routes
them to the Ascend kernels Transformers already ships. The Qwen3 language model
needs no such help: Transformers selects its own NPU flash-attention path.
"""

from __future__ import annotations

from rlinf.models.embodiment.gr00t.npu_common import (
    hide_unimportable_torchcodec,
    install_npu_flash_attention,
    is_npu,
)

# Patcher references replacement objects by string path.
_COMMON = "rlinf.models.embodiment.gr00t.npu_common"

# The vendored Eagle3-VL module whose attention reads flash-attention entry
# points from its own namespace. Named by final component because the package it
# lands in is created by trust_remote_code at load time.
_FLASH_ATTENTION_MODULES = ("modeling_siglip2",)


def apply_npu_patches(patcher) -> dict | None:
    """Register the Ascend patches for building GR00T N1.6; return restore state.

    No-op returning ``None`` off NPU. Call before ``patcher.apply()``; pass the
    result to :func:`restore_npu_patches` after model construction. Installs:

    * fused NPU kernels for Qwen3 RMSNorm / rotary embedding (kept for the
      model's lifetime, not restored);
    * the NPU flash-attention entry points in the vendored SigLIP2 module, bound
      as it is imported;
    * a meta-path finder that turns a CUDA-only torchcodec ``OSError`` into
      ``ImportError``, matching GR00T's optional-import handlers.
    """
    if not is_npu():
        return None

    hide_unimportable_torchcodec()

    patcher.add_patch(
        "transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb",
        f"{_COMMON}.npu_apply_rotary_pos_emb",
    )
    patcher.add_patch(
        "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward",
        f"{_COMMON}.npu_rmsnorm_forward",
    )
    patcher.add_patch(
        "gr00t.model.modules.dit._sdpa_context",
        f"{_COMMON}.npu_sdpa_context",
    )

    return {
        "stop_flash_attention_binding": install_npu_flash_attention(
            *_FLASH_ATTENTION_MODULES
        )
    }


def restore_npu_patches(patcher, state: dict | None) -> None:
    """Undo the process-global patches from :func:`apply_npu_patches`.

    ``state`` is that call's return value; ``None`` (off NPU) is a no-op. The
    fused RMSNorm / rotary patches and the bindings already made are left in
    place: the model built under them keeps calling those names.
    """
    if state is None:
        return

    state["stop_flash_attention_binding"]()
