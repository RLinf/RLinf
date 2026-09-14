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

"""Codecs available to collective tensor compression.

Each codec contributes three pieces, kept together below: the codec itself,
which drives a system compression library through ``ctypes``; a configuration
class registered under its ``CODEC_TYPE`` so that
``cluster.collective.tensor_compression.codec`` selects it; and a provider that
owns the codec's runtime resources and their concurrent acquisition policy.
"""

import ctypes
import ctypes.util
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Empty, LifoQueue
from typing import ClassVar, Literal, Optional

import torch

from ..cluster.config import TensorCompressionConfig, TensorCompressionManager


@dataclass(frozen=True)
class TensorCompressionWireMetadata:
    """Describe compressed CPU tensor payloads in one tensor-list transfer."""

    codec: Literal["lz4", "zstd"]
    compressed_numel: tuple[Optional[int], ...]


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


class TensorCodecProvider(ABC):
    """Own codec resources and their concurrent acquisition policy."""

    codec_name: str

    @abstractmethod
    def try_acquire_compressor(self) -> Optional[TensorCodec]:
        """Return an available compressor without blocking."""

    @abstractmethod
    def acquire_decompressor(self) -> TensorCodec:
        """Return a decompressor, waiting for one when required."""

    @abstractmethod
    def release(self, codec: TensorCodec) -> None:
        """Release a previously acquired codec."""


class LZ4TensorCodec(TensorCodec):
    """LZ4-fast tensor codec backed by the system liblz4 library."""

    _MAX_INPUT_SIZE = 0x7E000000

    def __init__(self, acceleration: int = 1) -> None:
        """Bind the system liblz4 entry points used by this codec."""
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
        """Return worst-case capacity, or ``None`` past LZ4's input-size limit."""
        if not 0 <= source_bytes <= self._MAX_INPUT_SIZE:
            return None
        return int(self._library.LZ4_compressBound(source_bytes))

    def compress_into(self, source: torch.Tensor, destination: torch.Tensor) -> int:
        """Compress source into destination and return the encoded byte count."""
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
        """Restore source into destination, which must be exactly the raw size."""
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


@TensorCompressionManager.register_codec_config
@dataclass
class LZ4CompressionConfig(TensorCompressionConfig):
    """LZ4 codec configuration.

    LZ4 favors codec speed over compression ratio and is stateless, so one codec
    is shared by all of a Worker's threads without an acquisition limit.
    """

    CODEC_TYPE: ClassVar[str] = "lz4"

    acceleration: int = 1
    """LZ4 fast-compression acceleration. Higher trades ratio for speed."""

    def __post_init__(self):
        """Validate the LZ4 parameters."""
        super().__post_init__()
        if type(self.acceleration) is not int:
            raise ValueError(
                "cluster.collective.tensor_compression.acceleration must be an "
                f"integer. But got {type(self.acceleration)}: {self.acceleration}"
            )
        if self.acceleration < 1:
            raise ValueError(
                "cluster.collective.tensor_compression.acceleration must be >= 1, "
                f"got {self.acceleration}."
            )

    def create_codec_provider(self) -> "LZ4CodecProvider":
        """Create the shared-codec provider for LZ4."""
        return LZ4CodecProvider(self)


class LZ4CodecProvider(TensorCodecProvider):
    """Share one stateless LZ4 codec across Worker threads."""

    codec_name = LZ4CompressionConfig.CODEC_TYPE

    def __init__(self, config: LZ4CompressionConfig) -> None:
        """Create the shared LZ4 codec."""
        self._codec = LZ4TensorCodec(acceleration=config.acceleration)

    def try_acquire_compressor(self) -> TensorCodec:
        """Return the shared codec without blocking."""
        return self._codec

    def acquire_decompressor(self) -> TensorCodec:
        """Return the shared codec without blocking."""
        return self._codec

    def release(self, codec: TensorCodec) -> None:
        """Leave the shared codec available to all callers."""
        pass


class ZstdTensorCodec(TensorCodec):
    """Zstandard tensor codec with reusable native contexts.

    One instance must not be used concurrently by multiple threads.
    """

    def __init__(self, level: int = 1) -> None:
        """Bind the system libzstd entry points and allocate native contexts."""
        if type(level) is not int or level < 1:
            raise ValueError("Zstd compression level must be a positive integer.")
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
        """Return worst-case capacity, or ``None`` for a negative size."""
        if source_bytes < 0:
            return None
        return int(self._library.ZSTD_compressBound(source_bytes))

    def compress_into(self, source: torch.Tensor, destination: torch.Tensor) -> int:
        """Compress source into destination and return the encoded byte count."""
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
        """Restore source into destination, which must be exactly the raw size."""
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


@TensorCompressionManager.register_codec_config
@dataclass
class ZstdCompressionConfig(TensorCompressionConfig):
    """Zstandard codec configuration.

    Zstd usually reduces wire bytes more than LZ4 at a higher codec cost, and
    its native contexts are stateful, so ``max_inflight`` of them are leased
    exclusively within each Worker.
    """

    CODEC_TYPE: ClassVar[str] = "zstd"

    level: int = 1
    """Zstd compression level. Higher spends more CPU for a better ratio."""

    max_inflight: int = 4
    """Reusable Zstd contexts per Worker, shared by compression and decompression."""

    def __post_init__(self):
        """Validate the Zstd parameters."""
        super().__post_init__()
        if type(self.level) is not int:
            raise ValueError(
                "cluster.collective.tensor_compression.level must be an integer. "
                f"But got {type(self.level)}: {self.level}"
            )
        if self.level < 1:
            raise ValueError(
                f"cluster.collective.tensor_compression.level must be >= 1, got {self.level}."
            )
        if type(self.max_inflight) is not int:
            raise ValueError(
                "cluster.collective.tensor_compression.max_inflight must be an "
                f"integer. But got {type(self.max_inflight)}: {self.max_inflight}"
            )
        if self.max_inflight < 1:
            raise ValueError(
                "cluster.collective.tensor_compression.max_inflight must be >= 1, "
                f"got {self.max_inflight}."
            )

    def create_codec_provider(self) -> "ZstdCodecProvider":
        """Create the bounded context-pool provider for Zstd."""
        return ZstdCodecProvider(self)


class ZstdCodecProvider(TensorCodecProvider):
    """Bound concurrent use of reusable Zstd codec contexts."""

    codec_name = ZstdCompressionConfig.CODEC_TYPE

    def __init__(self, config: ZstdCompressionConfig) -> None:
        """Create the bounded Zstd codec queue."""
        self._codecs: LifoQueue[TensorCodec] = LifoQueue(maxsize=config.max_inflight)
        for _ in range(config.max_inflight):
            self._codecs.put_nowait(ZstdTensorCodec(level=config.level))

    def try_acquire_compressor(self) -> Optional[TensorCodec]:
        """Return an available codec without blocking."""
        try:
            return self._codecs.get_nowait()
        except Empty:
            return None

    def acquire_decompressor(self) -> TensorCodec:
        """Wait for and return an available codec."""
        return self._codecs.get()

    def release(self, codec: TensorCodec) -> None:
        """Return the codec to the queue."""
        self._codecs.put_nowait(codec)


def probe_tensor_codec_library(codec: str) -> None:
    """Check that a configured codec's system library can be loaded.

    Args:
        codec (str): The registered codec name, whose system library shares it.

    Raises:
        ValueError: The codec is not registered.
    """
    if codec not in TensorCompressionManager.codec_config_register:
        raise ValueError(f"Unsupported tensor codec: {codec!r}.")
    _load_library(codec)


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
