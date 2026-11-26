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

"""SGLang external processors for span-aligned multi-turn multimodal rollouts."""

from .qwen_vl import (
    EXTERNAL_MM_PROCESSOR_PACKAGE,
    IMAGE_SPANS_PARAM,
    MIN_MULTIMODAL_SGLANG_VERSION,
    ensure_multimodal_processor_env,
    validate_image_spans,
)

__all__ = [
    "EXTERNAL_MM_PROCESSOR_PACKAGE",
    "IMAGE_SPANS_PARAM",
    "MIN_MULTIMODAL_SGLANG_VERSION",
    "ensure_multimodal_processor_env",
    "validate_image_spans",
]
