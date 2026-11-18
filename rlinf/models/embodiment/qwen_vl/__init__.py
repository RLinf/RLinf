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

"""Accelerator-compatible helpers for Qwen-VL vision and attention modules."""

from .linear_patch_embed import patch_vision_patch_embed
from .npu_attention import enable_qwen_vl_npu_attention, npu_grouped_attention_forward
from .npu_patchify import (
    ensure_qwen_vl_npu_patchify_patch,
    qwen_vl_supports_device_images,
)

__all__ = [
    "enable_qwen_vl_npu_attention",
    "ensure_qwen_vl_npu_patchify_patch",
    "npu_grouped_attention_forward",
    "patch_vision_patch_embed",
    "qwen_vl_supports_device_images",
]
