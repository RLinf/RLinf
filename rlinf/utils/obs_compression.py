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

"""Optional lossless compression for Env -> Rollout observation transfer.

Image observations sent from :class:`EnvWorker` to the rollout workers are the
dominant payload on the Env->Rollout channel. In bandwidth-constrained or
multi-node setups this transfer can throttle rollout throughput. This module
provides an *optional*, *disabled-by-default*, and *fully lossless* codec that
compresses the ``uint8`` image tensors inside an observation payload while
leaving every other field untouched.

The codec is self-contained per message: each compressed tensor is replaced by
a small marker ``dict`` carrying everything required to reconstruct it
(``shape``, ``dtype``, ``device``, whether XOR-delta was applied, and the
compressed bytes). Decompression therefore needs no configuration and keeps no
cross-message or cross-worker state, which keeps it robust to reordering and
environment resets.

Two backends are supported and imported lazily so they are only required when
compression is actually enabled:

* ``lz4`` - fastest, good ratio.
* ``zstd`` - slightly slower, better ratio; ``level`` selects the effort.

An optional XOR-delta pre-pass (:func:`_xor_encode`) computes the byte-wise
difference between consecutive frames along the batch axis. Parallel embodied
environments (e.g. LIBERO, ManiSkill) render largely static backgrounds, so
consecutive frames share many identical bytes and the delta compresses far
better than the raw frames.
"""

from functools import lru_cache
from typing import Any, Callable, Optional

import numpy as np
import torch

# Reserved key marking a compressed-tensor payload inside an observation dict.
_CODEC_KEY = "__obs_codec__"

_INSTALL_HINT = {
    "lz4": "pip install lz4",
    "zstd": "pip install zstandard",
}


def _is_image_tensor(value: Any) -> bool:
    """Return whether ``value`` is a compressible image tensor.

    Image observations are ``uint8`` tensors with at least three dimensions
    (e.g. ``[B, H, W, C]``). Proprioceptive states (``float32``) and flag
    tensors (``bool``/``int``) are intentionally excluded.
    """
    return (
        isinstance(value, torch.Tensor)
        and value.dtype == torch.uint8
        and value.dim() >= 3
    )


def _xor_encode(array: np.ndarray) -> np.ndarray:
    """Replace each frame with its byte-wise XOR against the previous frame."""
    delta = array.copy()
    delta[1:] = array[1:] ^ array[:-1]
    return delta


def _xor_decode(delta: np.ndarray) -> np.ndarray:
    """Invert :func:`_xor_encode` via a cumulative XOR along the batch axis."""
    return np.bitwise_xor.accumulate(delta, axis=0)


@lru_cache(maxsize=None)
def _get_backend(codec: str) -> Callable[..., Any]:
    """Lazily import and return ``(compress, decompress)`` for ``codec``.

    Cached so the backend module is imported and the codec closures are built
    once per codec rather than for every image tensor on the rollout hot path.
    """
    if codec == "lz4":
        try:
            import lz4.frame as lz4_frame
        except ImportError as exc:  # pragma: no cover - exercised via message
            raise ImportError(
                f"Observation compression codec 'lz4' is not installed. "
                f"Install it with `{_INSTALL_HINT['lz4']}` or disable "
                f"`env.obs_compression`."
            ) from exc

        def _compress(raw: bytes, level: int) -> bytes:
            return lz4_frame.compress(raw, compression_level=level)

        return _compress, lz4_frame.decompress

    if codec == "zstd":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - exercised via message
            raise ImportError(
                f"Observation compression codec 'zstd' is not installed. "
                f"Install it with `{_INSTALL_HINT['zstd']}` or disable "
                f"`env.obs_compression`."
            ) from exc

        def _compress(raw: bytes, level: int) -> bytes:
            return zstandard.ZstdCompressor(level=level).compress(raw)

        def _decompress(raw: bytes) -> bytes:
            return zstandard.ZstdDecompressor().decompress(raw)

        return _compress, _decompress

    raise ValueError(
        f"Unknown observation compression codec {codec!r}; expected 'lz4' or 'zstd'."
    )


def _encode_image(
    tensor: torch.Tensor, codec: str, level: int, xor_delta: bool
) -> dict[str, Any]:
    """Compress a single ``uint8`` image tensor into a self-describing marker."""
    compress, _ = _get_backend(codec)
    array = tensor.detach().cpu().numpy()
    use_xor = bool(xor_delta) and array.shape[0] > 1
    payload = _xor_encode(array) if use_xor else array
    return {
        _CODEC_KEY: codec,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "xor": use_xor,
        "data": compress(np.ascontiguousarray(payload).tobytes(), level),
    }


def _decode_image(marker: dict[str, Any]) -> torch.Tensor:
    """Reconstruct the original tensor from a marker produced by ``_encode_image``."""
    _, decompress = _get_backend(marker[_CODEC_KEY])
    raw = decompress(marker["data"])
    np_dtype = np.dtype(marker["dtype"])
    # `np.frombuffer` returns a read-only view; XOR decoding produces a fresh
    # writable array, otherwise copy so the resulting tensor owns its memory.
    array = np.frombuffer(raw, dtype=np_dtype).reshape(marker["shape"])
    array = _xor_decode(array) if marker["xor"] else array.copy()
    tensor = torch.from_numpy(array)
    # Restore the tensor to the device it was on before compression, mirroring
    # how uncompressed tensors are placed. Fall back to CPU when that device is
    # not addressable from the receiving process (e.g. a CPU-only worker).
    device = marker["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return tensor.to(device=device, dtype=getattr(torch, marker["dtype"]))


def _map_encode(value: Any, codec: str, level: int, xor_delta: bool) -> Any:
    """Recursively compress every image tensor reachable from ``value``."""
    if isinstance(value, dict):
        return {k: _map_encode(v, codec, level, xor_delta) for k, v in value.items()}
    if _is_image_tensor(value):
        return _encode_image(value, codec, level, xor_delta)
    return value


def _map_decode(value: Any) -> Any:
    """Recursively reconstruct every compressed marker reachable from ``value``."""
    if isinstance(value, dict):
        if _CODEC_KEY in value:
            return _decode_image(value)
        return {k: _map_decode(v) for k, v in value.items()}
    return value


def is_compression_enabled(config: Optional[Any]) -> bool:
    """Return whether an ``env.obs_compression`` config block enables the codec."""
    return bool(config is not None and config.get("enable", False))


def compress_obs(data: dict[str, Any], config: Optional[Any]) -> dict[str, Any]:
    """Compress the image tensors in an Env->Rollout payload.

    Args:
        data: The payload dict built by ``EnvWorker._build_rollout_input_data``,
            containing ``obs`` / ``final_obs`` sub-dicts of tensors plus optional
            flag tensors.
        config: The ``env.obs_compression`` config block (or ``None``). When it
            is missing or ``enable`` is false, ``data`` is returned unchanged.

    Returns:
        A payload with every ``uint8`` image tensor replaced by a compressed
        marker dict. Non-image fields are passed through untouched.
    """
    if not is_compression_enabled(config):
        return data
    codec = config.get("codec", "lz4")
    level = int(config.get("level", 1))
    xor_delta = bool(config.get("xor_delta", True))
    return _map_encode(data, codec, level, xor_delta)


def decompress_obs(data: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct any compressed markers in a received Env->Rollout payload.

    This is the exact inverse of :func:`compress_obs`. It is safe to call on
    uncompressed payloads: dicts without compression markers pass through
    unchanged, so the rollout worker can always route received data through it.
    """
    if not isinstance(data, dict):
        return data
    return _map_decode(data)


def is_compressed_image(value: Any) -> bool:
    """Return whether ``value`` is a compressed image marker from ``compress_obs``."""
    return isinstance(value, dict) and _CODEC_KEY in value


def infer_obs_batch_size(obs_batch: dict[str, Any]) -> int:
    """Infer the batch size of an Env->Rollout observation payload.

    The rollout workers infer the batch size on the receive path, *before*
    decompression. A compressed image is neither a tensor nor a list, so its
    batch size is read from the marker's stored ``shape``. This makes inference
    work whether or not ``env.obs_compression`` is enabled, including payloads
    whose only batched field is an image (no ``states`` / ``task_descriptions``).

    Args:
        obs_batch: A received payload, either the observation dict itself or a
            wrapper dict containing it under an ``obs`` key.

    Returns:
        The batch size (size of the leading dimension).

    Raises:
        ValueError: If no batched field can be found.
    """
    obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
    for value in obs.values():
        if isinstance(value, torch.Tensor):
            return value.shape[0]
        if isinstance(value, list):
            return len(value)
        if is_compressed_image(value):
            return value["shape"][0]
    raise ValueError("Cannot infer batch size from env obs.")
