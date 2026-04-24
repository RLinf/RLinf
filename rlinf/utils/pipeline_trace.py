import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PipelineTraceConfig:
    enabled: bool
    trace_dir: str


class PipelineTracer:
    """Per-process JSONL tracer for pipeline overlap analysis."""

    def __init__(
        self,
        *,
        enabled: bool,
        trace_path: str | None,
        static_fields: dict[str, Any],
    ) -> None:
        self._enabled = bool(enabled and trace_path)
        self._static_fields = dict(static_fields or {})
        self._lock = threading.Lock()
        self._fp = None

        if not self._enabled:
            return

        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)

        # Line-buffered for lower overhead and better crash survivability.
        self._fp = open(trace_path, "a", buffering=1, encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def emit(self, event: str, **fields: Any) -> None:
        if not self._enabled or self._fp is None:
            return

        record = {
            "ts_ns": time.time_ns(),
            "event": event,
            **self._static_fields,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._fp.write(line + "\n")

    def close(self) -> None:
        if self._fp is None:
            return
        with self._lock:
            try:
                self._fp.flush()
            finally:
                self._fp.close()
                self._fp = None

    @staticmethod
    def _get_cfg(cfg_runner: Any) -> PipelineTraceConfig:
        enable_flag = bool(getattr(cfg_runner, "enable_pipeline_trace", False))
        trace_dir = getattr(cfg_runner, "trace_dir", None)
        enabled = enable_flag or bool(trace_dir)
        return PipelineTraceConfig(
            enabled=enabled, trace_dir=str(trace_dir) if trace_dir else ""
        )

    @classmethod
    def from_worker(
        cls,
        *,
        cfg_runner: Any,
        worker_type: str,
        group_name: str | None,
        rank: int | None,
        pid: int,
    ) -> "PipelineTracer":
        cfg = cls._get_cfg(cfg_runner)
        if not cfg.enabled:
            return cls(enabled=False, trace_path=None, static_fields={})

        if not cfg.trace_dir:
            # Explicitly enabled but no directory specified; disable silently to avoid
            # writing into unexpected locations.
            return cls(enabled=False, trace_path=None, static_fields={})

        hostname = socket.gethostname()
        safe_group = (group_name or "unknown").replace("/", "_")
        safe_rank = "unknown" if rank is None else str(rank)
        filename = (
            f"trace_{worker_type}_{safe_group}_r{safe_rank}_pid{pid}_{hostname}.jsonl"
        )
        trace_path = os.path.join(cfg.trace_dir, filename)

        static_fields = {
            "worker_type": worker_type,
            "group_name": group_name,
            "rank": rank,
            "pid": pid,
            "hostname": hostname,
        }
        return cls(enabled=True, trace_path=trace_path, static_fields=static_fields)
