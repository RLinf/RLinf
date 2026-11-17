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

"""NumPy-aware MessagePack codec for the HTTP rollout workers.

The wire format matches the one the SGLang action protocol defines and the
PhyAI Gateway decodes: arrays travel as ``__ndarray__`` mappings carrying raw
bytes, the dtype string, and the shape, while NumPy scalars travel as
``__npgeneric__`` mappings. Keeping a local implementation lets
:class:`~rlinf.utils.http_client.InferenceHTTPClient` encode requests without
installing SGLang.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Structured, object, and complex dtypes have no portable byte representation.
_UNSUPPORTED_DTYPE_KINDS = ("V", "O", "c")


def pack_numpy_payload(obj: Any) -> Any:
    """Encode a NumPy array or scalar into its wire-format mapping.

    Args:
        obj: Value returned by the caller's ``default`` hook.

    Returns:
        A bytes-keyed mapping for NumPy values, or ``obj`` unchanged.

    Raises:
        ValueError: If the dtype has no portable byte representation.
    """
    if isinstance(obj, (np.ndarray, np.generic)) and (
        obj.dtype.kind in _UNSUPPORTED_DTYPE_KINDS
    ):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_numpy_payload(obj: Any) -> Any:
    """Decode a wire-format mapping back into a NumPy array or scalar.

    Accepts both bytes and string keys so payloads produced by either side of
    the wire decode identically.

    Args:
        obj: Mapping supplied by ``msgpack``'s ``object_hook``.

    Returns:
        The reconstructed NumPy value, or ``obj`` when it carries no marker.
    """
    if not isinstance(obj, dict):
        return obj
    data = obj.get("data", obj.get(b"data"))
    dtype = obj.get("dtype", obj.get(b"dtype"))
    if obj.get("__ndarray__") or obj.get(b"__ndarray__"):
        return np.ndarray(
            buffer=data,
            dtype=np.dtype(dtype),
            shape=obj.get("shape", obj.get(b"shape")),
        )
    if obj.get("__npgeneric__") or obj.get(b"__npgeneric__"):
        return np.dtype(dtype).type(data)
    return obj


def pack_msgpack(payload: Any) -> bytes:
    """Serialize ``payload`` to MessagePack with NumPy support."""
    import msgpack

    return msgpack.packb(payload, default=pack_numpy_payload, use_bin_type=True)


def unpack_msgpack(payload: bytes) -> Any:
    """Deserialize MessagePack bytes, rebuilding NumPy arrays and scalars."""
    import msgpack

    return msgpack.unpackb(payload, object_hook=unpack_numpy_payload, raw=False)
