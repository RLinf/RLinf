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

"""Opt-in .pt debug dumps for comparing RLinf and reference pipelines."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

_COUNTS: dict[str, int] = {}


def _dump_dir() -> str | None:
    return os.environ.get("RLINF_DEBUG_DUMP_DIR") or os.environ.get("DEBUG_DUMP_DIR")


def enabled() -> bool:
    return bool(_dump_dir())


def _max_per_tag() -> int:
    raw_value = os.environ.get("RLINF_DEBUG_DUMP_MAX_PER_TAG") or os.environ.get(
        "DEBUG_DUMP_MAX_PER_TAG", "20"
    )
    try:
        return max(int(raw_value), 0)
    except ValueError:
        return 20


def _safe_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _to_cpu_leaf(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, np.generic):
        return value.item()
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if hasattr(value, "__jax_array__"):
        return np.ascontiguousarray(np.asarray(value))
    return value


def _to_dumpable(value: Any) -> Any:
    value = _to_cpu_leaf(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_dumpable(value.to_dict())
    if dataclasses.is_dataclass(value):
        return _to_dumpable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {key: _to_dumpable(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_to_dumpable(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_to_dumpable(item) for item in value)
    if isinstance(value, list):
        return [_to_dumpable(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_dumpable(item) for item in value]
    return value


def dump_pt(tag: str, payload: Any) -> pathlib.Path | None:
    """Save payload to a .pt file when debug dumping is enabled."""

    dump_dir = _dump_dir()
    if not dump_dir:
        return None

    max_per_tag = _max_per_tag()
    count = _COUNTS.get(tag, 0)
    if count >= max_per_tag:
        return None
    _COUNTS[tag] = count + 1

    root = pathlib.Path(dump_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{count:04d}_{_safe_tag(tag)}_pid{os.getpid()}_{int(time.time() * 1000)}.pt"

    record = {
        "tag": tag,
        "count": count,
        "pid": os.getpid(),
        "time": time.time(),
    }
    try:
        record["payload"] = _to_dumpable(payload)
        torch.save(record, path)
    except Exception as exc:  # pragma: no cover - debug helper must not break rollout.
        record["dump_error"] = repr(exc)
        record["payload_repr"] = repr(payload)
        torch.save(record, path)
    return path
