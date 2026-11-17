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

"""Device and dependency compatibility for native CUDA quantization backends."""

from __future__ import annotations

import torch

from rlinf.quantization.config import BackendCapability


def _as_device(device: torch.device | str | int) -> torch.device:
    return (
        torch.device("cuda", device)
        if isinstance(device, int)
        else torch.device(device)
    )


def probe_cuda_int8_capability(device: torch.device | str | int) -> BackendCapability:
    """Check CUDA, BF16, and Triton support without running numerical operations."""
    device = _as_device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("quantization native runtime requires NVIDIA CUDA")
    with torch.cuda.device(device):
        major, minor = torch.cuda.get_device_capability(device)
        if major < 8 or not torch.cuda.is_bf16_supported():
            raise ValueError(
                "quantization requires BF16-capable CUDA architecture SM80 or newer"
            )
        import triton

    return BackendCapability(
        backend="triton_int8",
        version=f"torch={torch.__version__};cuda={torch.version.cuda};triton={triton.__version__};sm={major}{minor};rounding=nearest_even",
        supported_profiles=frozenset({"int8_block128"}),
    )


def probe_cuda_nvfp4_capability(device: torch.device | str | int) -> BackendCapability:
    """Check NVFP4 architecture and dependencies without packing or launching GEMM."""
    device = _as_device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("NVFP4 native runtime requires NVIDIA CUDA")
    with torch.cuda.device(device):
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) not in {(10, 0), (10, 3), (12, 0), (12, 1)}:
            raise ValueError("NVFP4 CUTLASS runtime requires SM100/SM120-family CUDA")
        from rlinf.quantization.kernels.nvfp4 import _flashinfer

        flashinfer = _flashinfer()
        if flashinfer.__version__ != "0.7.1":
            raise RuntimeError(
                f"NVFP4 requires FlashInfer 0.7.1, found {flashinfer.__version__}"
            )
        import triton

    return BackendCapability(
        backend="flashinfer_cutlass",
        version=f"torch={torch.__version__};cuda={torch.version.cuda};triton={triton.__version__};flashinfer={flashinfer.__version__};sm={major}{minor};pack=triton;gemm=cutlass;layout=128x4",
        supported_profiles=frozenset({"nvfp4"}),
    )
