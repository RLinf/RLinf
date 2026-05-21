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
"""Small numpy-aware msgpack helpers used by OpenPI websocket serving."""

from typing import Any

import msgpack
import numpy as np

_NDARRAY_EXT_CODE = 42


def _encode(obj: Any):
    if isinstance(obj, np.ndarray):
        payload = msgpack.packb(
            {
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tobytes(),
            },
            use_bin_type=True,
        )
        return msgpack.ExtType(_NDARRAY_EXT_CODE, payload)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Unsupported type for msgpack serialization: {type(obj)!r}")


def _decode(code: int, data: bytes):
    if code != _NDARRAY_EXT_CODE:
        return msgpack.ExtType(code, data)
    payload = msgpack.unpackb(data, raw=False)
    return np.frombuffer(payload["data"], dtype=np.dtype(payload["dtype"])).reshape(
        payload["shape"]
    )


def packb(obj: Any) -> bytes:
    """Pack an object, preserving numpy arrays."""
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    """Unpack an object, restoring numpy arrays."""
    return msgpack.unpackb(data, raw=False, ext_hook=_decode)


class Packer:
    """Compatibility wrapper matching the OpenPI client packer API."""

    def pack(self, obj: Any) -> bytes:
        """Pack an object, preserving numpy arrays."""
        return packb(obj)
