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

"""Bounded reusable CPU tensor buffers for collective payload processing."""

import threading
from bisect import bisect_left, insort
from typing import Optional

import torch

from ..cluster.config import TensorBufferPoolConfig


class TensorBufferPool:
    """Share reusable CPU byte buffers within a fixed Worker memory budget."""

    def __init__(self, config: TensorBufferPoolConfig) -> None:
        """Create an empty buffer cache bounded by ``config.max_bytes``."""
        self.config = config
        self._allocated_bytes = 0
        self._cached_bytes = 0
        self._buffers_by_size: dict[int, list[torch.Tensor]] = {}
        self._available_sizes: list[int] = []
        self._lock = threading.Lock()

    @property
    def allocated_bytes(self) -> int:
        """Return bytes owned by active and cached buffers."""
        with self._lock:
            return self._allocated_bytes

    @property
    def cached_bytes(self) -> int:
        """Return bytes currently available for reuse."""
        with self._lock:
            return self._cached_bytes

    def _pop_cached_buffer(self, size_index: int) -> torch.Tensor:
        """Remove one cached buffer from the indexed size bucket."""
        size = self._available_sizes[size_index]
        buffers = self._buffers_by_size[size]
        buffer = buffers.pop()
        if not buffers:
            del self._buffers_by_size[size]
            self._available_sizes.pop(size_index)
        self._cached_bytes -= size
        return buffer

    def _evict_cached_buffers(self, bytes_to_free: int) -> None:
        """Discard large idle buffers until at least ``bytes_to_free`` are free."""
        while bytes_to_free > 0 and self._available_sizes:
            size = self._available_sizes[-1]
            buffers = self._buffers_by_size[size]
            count = min(len(buffers), (bytes_to_free + size - 1) // size)
            del buffers[-count:]
            freed_bytes = count * size
            self._allocated_bytes -= freed_bytes
            self._cached_bytes -= freed_bytes
            bytes_to_free -= freed_bytes
            if not buffers:
                del self._buffers_by_size[size]
                self._available_sizes.pop()

    def try_acquire(self, capacity: int) -> Optional["BufferLease"]:
        """Acquire a best-fit buffer without exceeding the memory budget."""
        with self._lock:
            if capacity > self.config.max_bytes:
                return None

            best_index = bisect_left(self._available_sizes, capacity)
            if best_index < len(self._available_sizes) and (
                self._available_sizes[best_index] <= capacity * 2
                or self._allocated_bytes + capacity > self.config.max_bytes
            ):
                return BufferLease(self, self._pop_cached_buffer(best_index))

            bytes_to_free = self._allocated_bytes + capacity - self.config.max_bytes
            if bytes_to_free > 0:
                self._evict_cached_buffers(bytes_to_free)
            if self._allocated_bytes + capacity > self.config.max_bytes:
                return None

            buffer = torch.empty(capacity, dtype=torch.uint8, device="cpu")
            self._allocated_bytes += buffer.numel()
            return BufferLease(self, buffer)

    def release(self, buffer: torch.Tensor, *, cache: bool) -> None:
        """Return a buffer to the cache or remove it from the budget."""
        with self._lock:
            size = buffer.numel()
            if cache:
                buffers = self._buffers_by_size.get(size)
                if buffers is None:
                    self._buffers_by_size[size] = [buffer]
                    insort(self._available_sizes, size)
                else:
                    buffers.append(buffer)
                self._cached_bytes += size
            else:
                self._allocated_bytes -= size


class BufferLease:
    """Own one tensor buffer until no payload references it."""

    def __init__(self, pool: TensorBufferPool, tensor: torch.Tensor) -> None:
        """Bind the tensor buffer to its pool."""
        self._pool = pool
        self._tensor: Optional[torch.Tensor] = tensor

    @property
    def tensor(self) -> torch.Tensor:
        """Return the owned tensor buffer."""
        if self._tensor is None:
            raise RuntimeError("BufferLease has already been released.")
        return self._tensor

    def release(self, *, cache: bool = True) -> None:
        """Release the buffer exactly once."""
        if self._tensor is None:
            return
        tensor = self._tensor
        self._tensor = None
        self._pool.release(tensor, cache=cache)
