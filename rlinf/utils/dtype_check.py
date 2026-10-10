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

"""Weight dtype consistency checks for local HF model directories.

Serving engines such as vLLM, SGLang and HF Transformers default to the
``torch_dtype`` declared in a checkpoint's ``config.json``. When the weights
are stored in a different dtype (e.g. the RLinf/WideSeek-R1-4b checkpoint
ships fp16 weights while its config declares bfloat16, see issue #1662), the
default serving path silently upcasts the weights and can lose accuracy.

The helpers below read only JSON metadata (``config.json`` and safetensors
headers), never the weight tensors themselves, so they are cheap and safe to
call on CPU before engine initialization. They never raise: a failed check is
logged at debug level instead of interrupting engine startup.
"""

import json
import logging
import os
import struct
from pathlib import Path
from typing import Callable, Optional, Union

import torch

# Newer transformers versions write `dtype`, older ones `torch_dtype`.
_CONFIG_DTYPE_FIELDS = ("torch_dtype", "dtype")
_SAFETENSORS_HEADER_SIZE_BYTES = 8

# safetensors dtype strings for the floating point types that serving
# engines reason about. Non-floating entries (ints, quantized formats, ...)
# are intentionally ignored so that mixed checkpoints do not produce noise.
_SAFETENSORS_FLOAT_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


def _read_config_dtype(model_dir: Path) -> Optional[torch.dtype]:
    """Return the dtype declared by ``config.json``, or None if absent."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    for field in _CONFIG_DTYPE_FIELDS:
        value = config.get(field)
        if isinstance(value, str) and value != "auto":
            dtype = getattr(torch, value, None)
            if isinstance(dtype, torch.dtype):
                return dtype
    return None


def _read_safetensors_header_dtypes(path: Path) -> set:
    """Read only the header of a safetensors file and return its float dtypes."""
    with open(path, "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(_SAFETENSORS_HEADER_SIZE_BYTES))
        header = json.loads(f.read(header_size))
    return {
        _SAFETENSORS_FLOAT_DTYPES[tensor["dtype"]]
        for name, tensor in header.items()
        if name != "__metadata__" and tensor["dtype"] in _SAFETENSORS_FLOAT_DTYPES
    }


def _read_weight_dtypes(model_dir: Path) -> Optional[set]:
    """Return the float dtypes stored in the checkpoint, or None if unknown.

    Supports both single-file (``model.safetensors``) and sharded
    (``model.safetensors.index.json``) checkpoints. Only headers are read.
    """
    single_file = model_dir / "model.safetensors"
    if single_file.is_file():
        return _read_safetensors_header_dtypes(single_file)

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})
    dtypes: set = set()
    for shard_name in sorted(set(weight_map.values())):
        shard_path = model_dir / shard_name
        if shard_path.is_file():
            dtypes |= _read_safetensors_header_dtypes(shard_path)
    return dtypes or None


def check_weight_dtype_consistency(
    model_path: Union[str, os.PathLike],
    serving_dtype: Optional[torch.dtype] = None,
) -> list:
    """Check a local model directory for declared-vs-stored dtype mismatches.

    Args:
        model_path: Path of the model directory. Hugging Face hub ids and
            other non-local paths are skipped (returns no warnings).
        serving_dtype: The dtype the engine is about to run with, if known.

    Returns:
        A list of human-readable warning messages; empty when the checkpoint
        is consistent or cannot be inspected.
    """
    model_dir = Path(os.path.expanduser(str(model_path)))
    if not model_dir.is_dir():
        # Hub id or remote path: nothing to inspect locally.
        return []

    warnings = []
    config_dtype = _read_config_dtype(model_dir)
    weight_dtypes = _read_weight_dtypes(model_dir)

    # Only warn on checkpoints with a single float dtype to avoid false
    # positives on mixed-precision (e.g. fp8-quantized) checkpoints.
    if config_dtype is not None and weight_dtypes and len(weight_dtypes) == 1:
        (weight_dtype,) = weight_dtypes
        if config_dtype != weight_dtype:
            warnings.append(
                f"Model config at '{model_dir / 'config.json'}' declares dtype "
                f"'{config_dtype}' but the safetensors weights are stored as "
                f"'{weight_dtype}'. Serving engines that follow the config dtype "
                f'by default (vLLM/SGLang/HF Transformers with dtype="auto") will '
                f"upcast the weights, which can change results. Pass an explicit "
                f"--dtype matching the stored weights instead. "
                f"See https://github.com/RLinf/RLinf/issues/1662."
            )
        if serving_dtype is not None and serving_dtype != weight_dtype:
            warnings.append(
                f"Requested serving dtype '{serving_dtype}' does not match the "
                f"stored weight dtype '{weight_dtype}' of '{model_dir}'; the "
                f"weights will be cast at load time."
            )
    return warnings


def warn_on_weight_dtype_mismatch(
    model_path: Union[str, os.PathLike],
    serving_dtype: Optional[torch.dtype] = None,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Log any dtype mismatch found for ``model_path``; never raises.

    Intended to be called right before serving engine initialization so that
    users notice an upcast (and its accuracy cost) instead of hitting it in
    their benchmark numbers.

    Args:
        model_path: Path of the model directory to check.
        serving_dtype: The dtype the engine is about to run with, if known.
        log: Logging callable; defaults to ``logging.warning``.
    """
    log = log if log is not None else logging.warning
    try:
        for message in check_weight_dtype_consistency(model_path, serving_dtype):
            log(message)
    except Exception as e:  # noqa: BLE001 - diagnostics must not break startup
        logging.debug("Weight dtype consistency check skipped: %s", e)
