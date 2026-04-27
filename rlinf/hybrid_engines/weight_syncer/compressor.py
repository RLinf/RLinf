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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from rlinf.utils.logging import get_logger

from .base import normalize_device

if TYPE_CHECKING:
    from .patch_syncer import WeightPatch, WeightPatchTransport

logger = get_logger()

_NVCOMP_COMPRESSION_ALGORITHMS = {
    "nvcomp_lz4": "LZ4",
}

_NVCOMP_DTYPE_TO_CODE = {
    torch.uint8: 0,
    torch.int16: 1,
    torch.int32: 2,
    torch.int64: 3,
    torch.float16: 4,
    torch.bfloat16: 5,
    torch.float32: 6,
    torch.float64: 7,
}

_NVCOMP_CODE_TO_DTYPE = {code: dtype for dtype, code in _NVCOMP_DTYPE_TO_CODE.items()}

_NVCOMP_CODE_TO_DECODE_TYPE = {
    _NVCOMP_DTYPE_TO_CODE[torch.uint8]: "|u1",
    _NVCOMP_DTYPE_TO_CODE[torch.int16]: "<i2",
    _NVCOMP_DTYPE_TO_CODE[torch.int32]: "<i4",
    _NVCOMP_DTYPE_TO_CODE[torch.int64]: "<i8",
    _NVCOMP_DTYPE_TO_CODE[torch.float16]: "<i2",
    _NVCOMP_DTYPE_TO_CODE[torch.bfloat16]: "<i2",
    _NVCOMP_DTYPE_TO_CODE[torch.float32]: "<i4",
    _NVCOMP_DTYPE_TO_CODE[torch.float64]: "<i8",
}

_NVCOMP_DTYPE_TO_VIEW_DTYPE = {
    torch.uint8: torch.uint8,
    torch.int16: torch.int16,
    torch.int32: torch.int32,
    torch.int64: torch.int64,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}


class PatchCompressor(ABC):
    def __init__(self, transport_device: torch.device | str):
        self.transport_device = normalize_device(transport_device)

    @abstractmethod
    def compress(self, patch: WeightPatch) -> WeightPatchTransport: ...

    @abstractmethod
    def decompress(self, payload: WeightPatchTransport) -> WeightPatch: ...

    @classmethod
    def create(
        cls,
        compression_algorithm: str,
        transport_device: torch.device | str,
    ) -> "PatchCompressor":
        if compression_algorithm == "none":
            return IdentityCompressor(transport_device=transport_device)
        if compression_algorithm in _NVCOMP_COMPRESSION_ALGORITHMS:
            return NVCompCompressor(
                compression_algorithm=compression_algorithm,
                transport_device=transport_device,
            )

        logger.warning(
            "PatchWeightSyncer uses flat tensor transport; "
            f"compression_algorithm={compression_algorithm} is ignored for now."
        )
        return IdentityCompressor(transport_device=transport_device)


class IdentityCompressor(PatchCompressor):
    def compress(self, patch: WeightPatch) -> WeightPatchTransport:
        return patch

    def decompress(self, payload: WeightPatchTransport) -> WeightPatch:
        from .patch_syncer import WeightPatch

        assert isinstance(payload, WeightPatch), (
            f"IdentityCompressor expected WeightPatch, got {type(payload)}"
        )
        return payload


class NVCompCompressor(PatchCompressor):
    def __init__(
        self,
        compression_algorithm: str,
        transport_device: torch.device | str,
    ):
        super().__init__(transport_device=transport_device)
        self.compression_algorithm = compression_algorithm
        if self.transport_device.type != "cuda":
            raise ValueError("nvcomp compression requires transport_device to be cuda")
        try:
            from nvidia import nvcomp  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "nvcomp compression requested but nvidia-nvcomp is not available"
            ) from error

    def _get_nvcomp_algorithm(self) -> str:
        algorithm = _NVCOMP_COMPRESSION_ALGORITHMS.get(self.compression_algorithm)
        assert algorithm is not None, (
            f"Unsupported nvcomp compression algorithm: {self.compression_algorithm}"
        )
        return algorithm

    def _make_nvcomp_codec(self, device: torch.device):
        from nvidia import nvcomp

        assert device.type == "cuda", "nvcomp only supports CUDA tensors"
        stream = torch.cuda.current_stream(device=device)
        device_id = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return nvcomp.Codec(
            algorithm=self._get_nvcomp_algorithm(),
            device_id=device_id,
            cuda_stream=int(stream.cuda_stream),
        )

    def _get_nvcomp_tensor_spec(self, tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        tensor = tensor.contiguous()
        dtype_code = _NVCOMP_DTYPE_TO_CODE.get(tensor.dtype)
        if dtype_code is None:
            raise TypeError(f"Unsupported nvcomp tensor dtype: {tensor.dtype}")
        view_dtype = _NVCOMP_DTYPE_TO_VIEW_DTYPE[tensor.dtype]
        if tensor.dtype == view_dtype:
            return tensor, dtype_code
        return tensor.view(view_dtype), dtype_code

    def _restore_nvcomp_tensor_dtype(
        self, tensor: torch.Tensor, dtype_code: int
    ) -> torch.Tensor:
        target_dtype = _NVCOMP_CODE_TO_DTYPE[dtype_code]
        if tensor.dtype == target_dtype:
            return tensor
        return tensor.view(target_dtype)

    def _compress_tensor(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_tensor, dtype_code = self._get_nvcomp_tensor_spec(tensor)
        dtype_code_tensor = torch.tensor(
            dtype_code, dtype=torch.int8, device=tensor.device
        )
        if view_tensor.numel() == 0:
            return (
                torch.empty(0, dtype=torch.int8, device=tensor.device),
                dtype_code_tensor,
            )

        from nvidia import nvcomp

        array = nvcomp.from_dlpack(torch.utils.dlpack.to_dlpack(view_tensor))
        with self._make_nvcomp_codec(tensor.device) as codec:
            compressed = codec.encode(array)
        compressed_tensor = torch.utils.dlpack.from_dlpack(compressed).clone()
        return compressed_tensor, dtype_code_tensor

    def _decompress_tensor(
        self, compressed_tensor: torch.Tensor, dtype_code_tensor: torch.Tensor
    ) -> torch.Tensor:
        dtype_code = int(dtype_code_tensor.item())
        target_dtype = _NVCOMP_CODE_TO_DTYPE[dtype_code]
        if compressed_tensor.numel() == 0:
            return torch.empty(0, dtype=target_dtype, device=compressed_tensor.device)

        decode_type = _NVCOMP_CODE_TO_DECODE_TYPE[dtype_code]

        from nvidia import nvcomp

        compressed_array = nvcomp.from_dlpack(
            torch.utils.dlpack.to_dlpack(compressed_tensor.contiguous())
        )
        with self._make_nvcomp_codec(compressed_tensor.device) as codec:
            restored_array = codec.decode(compressed_array, data_type=decode_type)
        restored_tensor = torch.utils.dlpack.from_dlpack(restored_array).clone()
        return self._restore_nvcomp_tensor_dtype(restored_tensor, dtype_code)

    def compress(self, patch: WeightPatch) -> WeightPatchTransport:
        from .patch_syncer import CompressedWeightPatch

        assert patch.rows.device.type == "cuda", (
            "nvcomp compression requires patch tensors on CUDA device"
        )
        rows_compressed, rows_dtype_code = self._compress_tensor(patch.rows)
        cols_compressed, cols_dtype_code = self._compress_tensor(patch.cols)
        values_compressed, values_dtype_code = self._compress_tensor(patch.values)
        return CompressedWeightPatch(
            version=patch.version,
            ordinals=patch.ordinals,
            nnz_per_tensor=patch.nnz_per_tensor,
            rows_compressed=rows_compressed,
            cols_compressed=cols_compressed,
            values_compressed=values_compressed,
            rows_dtype_code=rows_dtype_code,
            cols_dtype_code=cols_dtype_code,
            values_dtype_code=values_dtype_code,
        )

    def decompress(self, payload: WeightPatchTransport) -> WeightPatch:
        from .patch_syncer import CompressedWeightPatch, WeightPatch

        assert isinstance(payload, CompressedWeightPatch), (
            f"NVCompCompressor expected CompressedWeightPatch, got {type(payload)}"
        )
        rows = self._decompress_tensor(payload.rows_compressed, payload.rows_dtype_code)
        cols = self._decompress_tensor(payload.cols_compressed, payload.cols_dtype_code)
        values = self._decompress_tensor(
            payload.values_compressed, payload.values_dtype_code
        )
        return WeightPatch(
            version=payload.version,
            ordinals=payload.ordinals,
            nnz_per_tensor=payload.nnz_per_tensor,
            rows=rows,
            cols=cols,
            values=values,
        )
