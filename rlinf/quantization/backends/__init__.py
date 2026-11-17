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

"""Construct numerical backends independently of model projection adapters."""

from __future__ import annotations

from rlinf.quantization.backends.base import QuantizationBackend, QuantizedTensor

__all__ = ["QuantizationBackend", "QuantizedTensor", "create_quantization_backend"]


def create_quantization_backend(
    quantization_format: str, *, block_size: int
) -> QuantizationBackend:
    """Select one implementation without falling back to another format."""
    if quantization_format == "int8":
        from rlinf.quantization.backends.int8 import Int8Backend

        backend = Int8Backend
    elif quantization_format == "nvfp4":
        from rlinf.quantization.backends.nvfp4 import Nvfp4Backend

        backend = Nvfp4Backend
    else:
        raise ValueError(
            f"quantization format {quantization_format!r} is not supported"
        )
    return backend(block_size=block_size)
