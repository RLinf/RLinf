# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
import importlib.util

import torch
import torch.nn.functional as F

from rlinf.scheduler import AcceleratorType, Worker


@functools.cache
def inductor_available() -> bool:
    """Whether Inductor can compile these kernels in this interpreter.

    Inductor generates its device code through Triton, which ships only for
    CUDA and ROCm: Ascend and MUSA venvs have no ``triton``, and Isaac Sim's
    bundled torch cannot import the one in the venv. Compiling there fails at
    the first forward pass, so this decides the default for
    :func:`set_torch_compile`. A configuration that names ``torch_compile``
    explicitly still wins.
    """
    return importlib.util.find_spec("triton") is not None


# Kernels stay compiled wherever Inductor works, unless a caller turns this off
# before the first forward.
_torch_compile_enabled = inductor_available()
_sdpa_attention_enabled: bool | None = None


def set_torch_compile(enabled: bool) -> None:
    """Enable or disable OpenPI fused-kernel compilation for this process."""
    global _torch_compile_enabled
    _torch_compile_enabled = bool(enabled)


def torch_compile_enabled() -> bool:
    return _torch_compile_enabled


def set_sdpa_attention(enabled: bool | None) -> None:
    """Enable, disable, or reset OpenPI fused SDPA attention for this process."""
    global _sdpa_attention_enabled
    _sdpa_attention_enabled = enabled if enabled is None else bool(enabled)


def sdpa_attention_enabled() -> bool:
    """Whether OpenPI Gemma attention dispatches to fused SDPA.

    Defaults to the active accelerator via ``Worker.accelerator_type`` so
    rollout and training use the same attention backend on each platform.
    """
    if _sdpa_attention_enabled is not None:
        return _sdpa_attention_enabled
    return Worker.accelerator_type == AcceleratorType.NPU


def _gelu_glu_eager(
    gate_input: torch.Tensor, value_input: torch.Tensor
) -> torch.Tensor:
    return F.gelu(gate_input) * value_input


_gelu_glu_compiled = torch.compile(_gelu_glu_eager)


def gelu_glu(gate_input: torch.Tensor, value_input: torch.Tensor) -> torch.Tensor:
    """Fused GELU-GLU activation: ``gelu(gate_input) * value_input``."""
    fn = _gelu_glu_compiled if _torch_compile_enabled else _gelu_glu_eager
    return fn(gate_input, value_input)


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "mp_bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[dtype_str]
