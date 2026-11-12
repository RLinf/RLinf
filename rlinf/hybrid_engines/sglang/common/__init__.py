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

try:
    from sglang.srt.constants import GPU_MEMORY_ALL_TYPES
except ImportError:
    # Older SGLang releases do not manage CUDA graphs as a separate memory type.
    GPU_MEMORY_ALL_TYPES = ["weights", "kv_cache"]

__all__ = ["GPU_MEMORY_ALL_TYPES"]
