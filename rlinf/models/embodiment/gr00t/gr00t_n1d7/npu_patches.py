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

"""Ascend NPU patches for building GR00T N1.7.

N1.7's backbone is Qwen3-VL, which Transformers implements itself, so unlike
N1.6 there is no vendored attention to repair. What it does do is decide between
flash attention and SDPA by importing ``flash_attn`` and falling back when the
import fails. That package cannot be installed on Ascend, so the backbone always
takes the SDPA path even though Transformers has Ascend flash-attention kernels
and accepts ``flash_attention_2`` on NPU. Registering a stub import lets the
backbone's probe succeed so the faster path is chosen; Transformers itself keeps
resolving availability from the installed distribution, which the stub is not, so
it still routes to its Ascend kernels rather than looking for the real package.
"""

from __future__ import annotations

from rlinf.models.embodiment.gr00t.npu_common import (
    hide_unimportable_torchcodec,
    is_npu,
)

# Patcher references replacement objects by string path.
_COMMON = "rlinf.models.embodiment.gr00t.npu_common"


def apply_npu_patches(patcher) -> dict | None:
    """Register the Ascend patches for building GR00T N1.7; return restore state.

    No-op returning ``None`` off NPU. Call before ``patcher.apply()``; pass the
    result to :func:`restore_npu_patches` after model construction. Installs:

    * fused NPU kernels for Qwen3-VL text RMSNorm / rotary embedding (kept for
      the model's lifetime, not restored);
    * a ``flash_attn`` stub so the backbone selects ``flash_attention_2``, which
      Transformers serves from its Ascend kernels;
    * a meta-path finder that turns a CUDA-only torchcodec ``OSError`` into
      ``ImportError``, matching GR00T's optional-import handlers.
    """
    if not is_npu():
        return None

    hide_unimportable_torchcodec()

    patcher.add_patch(
        "transformers.models.qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb",
        f"{_COMMON}.npu_apply_rotary_pos_emb",
    )
    patcher.add_patch(
        "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRMSNorm.forward",
        f"{_COMMON}.npu_rmsnorm_forward",
    )
    patcher.add_patch(
        "gr00t.model.modules.dit._sdpa_context",
        f"{_COMMON}.npu_sdpa_context",
    )
    patcher.skip_import("flash_attn")

    return {"flash_attn_stub": True}


def restore_npu_patches(patcher, state: dict | None) -> None:
    """Undo the process-global patches from :func:`apply_npu_patches`.

    ``state`` is that call's return value; ``None`` (off NPU) is a no-op. The
    fused RMSNorm / rotary patches are intentionally left in place.
    """
    if state is None:
        return

    patcher.clear_stub_import("flash_attn")
