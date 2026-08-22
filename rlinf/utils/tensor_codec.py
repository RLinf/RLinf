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


import ctypes
import ctypes.util
import weakref
from abc import ABC, abstractmethod

import torch


class TensorCodec(ABC):
    """Compress tensors into preallocated uint8 tensors without Python bytes."""

    @abstractmethod
    def compress_bound(self, source_bytes: int) -> int | None:
        """Return worst-case capacity, or ``None`` when the input is unsupported."""

    @abstractmethod
    def compress_into(self, source: torch.Tensor, destination: torch.Tensor) -> int:
        """Compress source into destination and return the encoded byte count."""

    @abstractmethod
    def decompress_into(
        self,
        source: torch.Tensor,
        compressed_bytes: int,
        destination: torch.Tensor,
    ) -> None:
        """Decompress source directly into destination."""


class LZ4TensorCodec(TensorCodec):
    """LZ4-fast tensor codec backed by the system liblz4 library."""

    _MAX_INPUT_SIZE = 0x7E000000

    def __init__(self, acceleration: int = 1) -> None:
        if acceleration <= 0:
            raise ValueError("LZ4 acceleration must be positive.")
        self.acceleration = acceleration
        self._library = _load_library("lz4")
        self._library.LZ4_compressBound.argtypes = [ctypes.c_int]
        self._library.LZ4_compressBound.restype = ctypes.c_int
        self._library.LZ4_compress_fast.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.LZ4_compress_fast.restype = ctypes.c_int
        self._library.LZ4_decompress_safe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.LZ4_decompress_safe.restype = ctypes.c_int

    def compress_bound(self, source_bytes: int) -> int | None:
        if not 0 <= source_bytes <= self._MAX_INPUT_SIZE:
            return None
        return int(self._library.LZ4_compressBound(source_bytes))

    def compress_into(self, source: torch.Tensor, destination: torch.Tensor) -> int:
        source_bytes = _tensor_bytes(source, "source")
        destination_bytes = _byte_tensor(destination, "destination")
        required_bytes = self.compress_bound(source_bytes)
        if required_bytes is None:
            raise ValueError(f"LZ4 does not support source size {source_bytes}.")
        if destination_bytes < required_bytes:
            raise ValueError(
                f"LZ4 destination requires {required_bytes} bytes, got "
                f"{destination_bytes}."
            )
        if source_bytes == 0:
            return 0
        compressed_bytes = self._library.LZ4_compress_fast(
            source.data_ptr(),
            destination.data_ptr(),
            source_bytes,
            destination_bytes,
            self.acceleration,
        )
        if compressed_bytes <= 0:
            raise RuntimeError("LZ4 compression failed.")
        return int(compressed_bytes)

    def decompress_into(
        self,
        source: torch.Tensor,
        compressed_bytes: int,
        destination: torch.Tensor,
    ) -> None:
        source_capacity = _byte_tensor(source, "source")
        destination_bytes = _tensor_bytes(destination, "destination")
        _validate_compressed_size(compressed_bytes, source_capacity)
        if compressed_bytes == 0 and destination_bytes == 0:
            return
        restored_bytes = self._library.LZ4_decompress_safe(
            source.data_ptr(),
            destination.data_ptr(),
            compressed_bytes,
            destination_bytes,
        )
        if restored_bytes < 0:
            raise RuntimeError("LZ4 decompression detected invalid compressed data.")
        if restored_bytes != destination_bytes:
            raise ValueError(
                f"LZ4 restored {restored_bytes} bytes, expected {destination_bytes}."
            )


class ZstdTensorCodec(TensorCodec):
    """Zstandard tensor codec with reusable native contexts.

    One instance must not be used concurrently by multiple threads.
    """

    def __init__(self, level: int = 1) -> None:
        self.level = level
        self._library = _load_library("zstd")
        self._library.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        self._library.ZSTD_compressBound.restype = ctypes.c_size_t
        self._library.ZSTD_createCCtx.restype = ctypes.c_void_p
        self._library.ZSTD_freeCCtx.argtypes = [ctypes.c_void_p]
        self._library.ZSTD_compressCCtx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._library.ZSTD_compressCCtx.restype = ctypes.c_size_t
        self._library.ZSTD_createDCtx.restype = ctypes.c_void_p
        self._library.ZSTD_freeDCtx.argtypes = [ctypes.c_void_p]
        self._library.ZSTD_decompressDCtx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._library.ZSTD_decompressDCtx.restype = ctypes.c_size_t
        self._library.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self._library.ZSTD_isError.restype = ctypes.c_uint
        self._library.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        self._library.ZSTD_getErrorName.restype = ctypes.c_char_p
        self._compression_context = self._library.ZSTD_createCCtx()
        self._decompression_context = self._library.ZSTD_createDCtx()
        if not self._compression_context or not self._decompression_context:
            raise RuntimeError("Zstd failed to allocate codec contexts.")
        self._compression_finalizer = weakref.finalize(
            self, self._library.ZSTD_freeCCtx, self._compression_context
        )
        self._decompression_finalizer = weakref.finalize(
            self, self._library.ZSTD_freeDCtx, self._decompression_context
        )

    def compress_bound(self, source_bytes: int) -> int | None:
        if source_bytes < 0:
            return None
        return int(self._library.ZSTD_compressBound(source_bytes))

    def compress_into(self, source: torch.Tensor, destination: torch.Tensor) -> int:
        source_bytes = _tensor_bytes(source, "source")
        destination_bytes = _byte_tensor(destination, "destination")
        required_bytes = self.compress_bound(source_bytes)
        if required_bytes is None:
            raise ValueError(f"Zstd does not support source size {source_bytes}.")
        if destination_bytes < required_bytes:
            raise ValueError(
                f"Zstd destination requires {required_bytes} bytes, got "
                f"{destination_bytes}."
            )
        compressed_bytes = self._library.ZSTD_compressCCtx(
            self._compression_context,
            destination.data_ptr(),
            destination_bytes,
            source.data_ptr(),
            source_bytes,
            self.level,
        )
        self._check_result(compressed_bytes, "compression")
        return int(compressed_bytes)

    def decompress_into(
        self,
        source: torch.Tensor,
        compressed_bytes: int,
        destination: torch.Tensor,
    ) -> None:
        source_capacity = _byte_tensor(source, "source")
        destination_bytes = _tensor_bytes(destination, "destination")
        _validate_compressed_size(compressed_bytes, source_capacity)
        restored_bytes = self._library.ZSTD_decompressDCtx(
            self._decompression_context,
            destination.data_ptr(),
            destination_bytes,
            source.data_ptr(),
            compressed_bytes,
        )
        self._check_result(restored_bytes, "decompression")
        if restored_bytes != destination_bytes:
            raise ValueError(
                f"Zstd restored {restored_bytes} bytes, expected {destination_bytes}."
            )

    def _check_result(self, result: int, operation: str) -> None:
        if self._library.ZSTD_isError(result):
            error = self._library.ZSTD_getErrorName(result).decode()
            raise RuntimeError(f"Zstd {operation} failed: {error}.")


def create_tensor_codec(name: str, *, level: int = 1) -> TensorCodec:
    """Construct a direct tensor codec by transport name."""
    if name == "lz4":
        return LZ4TensorCodec(acceleration=level)
    if name == "zstd":
        return ZstdTensorCodec(level=level)
    raise ValueError(f"Unsupported tensor codec: {name!r}.")


def _load_library(name: str) -> ctypes.CDLL:
    library_path = ctypes.util.find_library(name)
    if library_path is None:
        raise RuntimeError(f"System lib{name} is not available.")
    return ctypes.CDLL(library_path)


def _tensor_bytes(tensor: torch.Tensor, name: str) -> int:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")
    return tensor.numel() * tensor.element_size()


def _byte_tensor(tensor: torch.Tensor, name: str) -> int:
    size = _tensor_bytes(tensor, name)
    if tensor.dtype != torch.uint8:
        raise ValueError(f"{name} must have dtype torch.uint8.")
    return size


def _validate_compressed_size(compressed_bytes: int, capacity: int) -> None:
    if not 0 <= compressed_bytes <= capacity:
        raise ValueError(
            f"Compressed size must be in [0, {capacity}], got {compressed_bytes}."
        )
