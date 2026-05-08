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

"""Lightweight latency profiling helpers."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator


def _get_runner_cfg(cfg: Any) -> Any:
    return cfg.get("runner", {}) if hasattr(cfg, "get") else {}


def get_run_id(cfg: Any) -> str:
    """Return the user-provided run id or a stable logger-derived fallback."""
    runner_cfg = _get_runner_cfg(cfg)
    run_id = runner_cfg.get("run_id", None) if hasattr(runner_cfg, "get") else None
    if run_id:
        return str(run_id)

    logger_cfg = runner_cfg.get("logger", {}) if hasattr(runner_cfg, "get") else {}
    experiment_name = (
        logger_cfg.get("experiment_name", None) if hasattr(logger_cfg, "get") else None
    )
    return str(experiment_name or "default")


def profile_cfg(cfg: Any) -> Any:
    runner_cfg = _get_runner_cfg(cfg)
    return runner_cfg.get("latency_profile", {}) if hasattr(runner_cfg, "get") else {}


def enabled(cfg: Any) -> bool:
    profiling_cfg = profile_cfg(cfg)
    return bool(
        profiling_cfg.get("enabled", False) if hasattr(profiling_cfg, "get") else False
    )


def sync_cuda_enabled(cfg: Any) -> bool:
    profiling_cfg = profile_cfg(cfg)
    return bool(
        profiling_cfg.get("sync_cuda", False)
        if hasattr(profiling_cfg, "get")
        else False
    )


def trace_jsonl_path(cfg: Any) -> str:
    profiling_cfg = profile_cfg(cfg)
    configured_path = (
        profiling_cfg.get("trace_jsonl_path", None)
        if hasattr(profiling_cfg, "get")
        else None
    )
    if configured_path:
        return str(configured_path)

    runner_cfg = _get_runner_cfg(cfg)
    logger_cfg = runner_cfg.get("logger", {}) if hasattr(runner_cfg, "get") else {}
    log_path = (
        logger_cfg.get("log_path", "logs") if hasattr(logger_cfg, "get") else "logs"
    )
    return os.path.join(str(log_path), f"{get_run_id(cfg)}_latency_trace.jsonl")


def _sync_cuda_if_requested(cfg: Any) -> None:
    if not sync_cuda_enabled(cfg):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
    except Exception:
        pass
    return str(value)


def emit_event(
    cfg: Any,
    *,
    component: str,
    event: str,
    begin_ns: int,
    end_ns: int,
    duration_ms: float,
    rank: int | None = None,
    **fields: Any,
) -> None:
    """Append one JSONL trace event if latency profiling is enabled."""
    if not enabled(cfg):
        return

    record = {
        "run_id": get_run_id(cfg),
        "component": component,
        "event": event,
        "rank": rank,
        "begin_ns": begin_ns,
        "end_ns": end_ns,
        "duration_ms": duration_ms,
    }
    record.update({key: _json_safe(value) for key, value in fields.items()})

    path = trace_jsonl_path(cfg)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


@contextmanager
def record_event(
    cfg: Any,
    *,
    component: str,
    event: str,
    rank: int | None = None,
    **fields: Any,
) -> Iterator[dict[str, Any]]:
    """Time a block and optionally write a JSONL event.

    The yielded dict lets callers add fields that are only known after the block
    completes, such as batch size or queue depth.
    """
    if not enabled(cfg):
        yield {}
        return

    dynamic_fields: dict[str, Any] = {}
    _sync_cuda_if_requested(cfg)
    begin_ns = time.time_ns()
    start = time.perf_counter()
    try:
        yield dynamic_fields
    finally:
        _sync_cuda_if_requested(cfg)
        end = time.perf_counter()
        end_ns = time.time_ns()
        merged_fields = dict(fields)
        merged_fields.update(dynamic_fields)
        emit_event(
            cfg,
            component=component,
            event=event,
            rank=rank,
            begin_ns=begin_ns,
            end_ns=end_ns,
            duration_ms=(end - start) * 1000.0,
            **merged_fields,
        )
