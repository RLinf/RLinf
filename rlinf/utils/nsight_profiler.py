# Copyright 2025 The RLinf Authors.
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

"""Nsight Systems profiler integration for RLinf.

This module provides NVTX-based profiling that co-exists with Ray's nsight
runtime_env support. Workers launched under ``nsys profile`` with
``capture-range=cudaProfilerApi`` will only record data between
``torch.cuda.profiler.start()`` and ``stop()`` calls, which are controlled
per-step by the runner.

Usage:
    1. Configure ``nsight_profiler`` in your YAML config.
    2. Workers call ``NsightProfiler.from_config(cfg, rank)`` during init.
    3. Runner calls ``start_profile(step)`` / ``stop_profile()`` each step.
    4. Decorate worker methods with ``@NsightProfiler.annotate("label")``.
"""

from __future__ import annotations

import functools
import inspect
import logging
from contextlib import contextmanager
from typing import Any, Callable, Optional

try:
    import nvtx

    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False

import torch

logger = logging.getLogger(__name__)

# Module-level flag for Channel NVTX instrumentation. Channels don't hold a
# profiler instance, so they check this flag instead.
_channel_nvtx_enabled: bool = False


def is_channel_nvtx_enabled() -> bool:
    return _channel_nvtx_enabled


def set_channel_nvtx_enabled(enabled: bool) -> None:
    global _channel_nvtx_enabled
    _channel_nvtx_enabled = enabled


@contextmanager
def nvtx_range(name: str, color: Optional[str] = None, domain: Optional[str] = None):
    """Lightweight NVTX range context manager.

    No-op when the ``nvtx`` package is not installed.
    """
    if not _NVTX_AVAILABLE:
        yield
        return
    range_id = nvtx.start_range(message=name, color=color, domain=domain)
    try:
        yield
    finally:
        nvtx.end_range(range_id)


class NsightProfiler:
    """Per-worker Nsight Systems profiler controller.

    Manages ``torch.cuda.profiler.start/stop`` gating and provides the
    ``@annotate`` decorator for NVTX ranges on worker methods.

    Args:
        enable: Whether this profiler instance is active.
        rank: The rank of the current worker process.
        all_ranks: Profile all ranks.
        ranks: Specific ranks to profile (used when ``all_ranks`` is False).
        discrete: If True, each annotated function call gets its own nsys-rep
            segment via per-call ``cuda.profiler.start/stop``.  If False (default),
            a single segment per training step is controlled by the runner.
    """

    def __init__(
        self,
        enable: bool = False,
        rank: int = 0,
        all_ranks: bool = True,
        ranks: Optional[list[int]] = None,
        discrete: bool = False,
    ):
        self.enable = enable
        self.rank = rank
        self.all_ranks = all_ranks
        self.ranks = ranks or []
        self.discrete = discrete

        self._this_step: bool = False
        self._this_rank: bool = self._check_rank()

        if self.enable and not _NVTX_AVAILABLE:
            logger.warning(
                "Nsight profiler enabled but 'nvtx' package not installed. "
                "NVTX annotations will be skipped. Install with: pip install nvtx"
            )

    def _check_rank(self) -> bool:
        if not self.enable:
            return False
        if self.all_ranks:
            return True
        return self.rank in self.ranks

    @property
    def is_active(self) -> bool:
        """True when profiling should happen right now."""
        return self.enable and self._this_rank and self._this_step

    def start(self, step: Optional[int] = None) -> None:
        """Begin profiling for the current step.

        In non-discrete mode this calls ``torch.cuda.profiler.start()`` so
        that nsys (running with ``capture-range=cudaProfilerApi``) begins
        recording.
        """
        if not self.enable or not self._this_rank:
            return
        self._this_step = True
        set_channel_nvtx_enabled(True)
        if not self.discrete:
            torch.cuda.profiler.start()
        if step is not None:
            logger.info("Nsight profiler started for step %d on rank %d", step, self.rank)

    def stop(self) -> None:
        """End profiling for the current step."""
        if not self.enable or not self._this_rank:
            return
        if not self.discrete:
            torch.cuda.profiler.stop()
        self._this_step = False
        set_channel_nvtx_enabled(False)

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

    @staticmethod
    def annotate(
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Callable:
        """Decorator for worker methods that adds NVTX ranges.

        The decorated method's owning instance must have a ``profiler``
        attribute of type :class:`NsightProfiler`.  When profiling is not
        active the original function is called directly (zero overhead beyond
        attribute lookup).

        Supports both sync and async methods.

        Example::

            class MyWorker(Worker):
                @NsightProfiler.annotate("rollout/predict")
                def predict(self, obs):
                    ...
        """

        def decorator(func: Callable) -> Callable:
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(self_instance: Any, *args: Any, **kwargs: Any) -> Any:
                    profiler: Optional[NsightProfiler] = getattr(self_instance, "nsight_profiler", None)
                    if profiler is None or not profiler.is_active or not _NVTX_AVAILABLE:
                        return await func(self_instance, *args, **kwargs)

                    label = message or func.__name__
                    if profiler.discrete:
                        torch.cuda.profiler.start()
                    range_id = nvtx.start_range(message=label, color=color, domain=domain)
                    try:
                        result = await func(self_instance, *args, **kwargs)
                    finally:
                        nvtx.end_range(range_id)
                        if profiler.discrete:
                            torch.cuda.profiler.stop()
                    return result

                return async_wrapper

            @functools.wraps(func)
            def sync_wrapper(self_instance: Any, *args: Any, **kwargs: Any) -> Any:
                profiler: Optional[NsightProfiler] = getattr(self_instance, "nsight_profiler", None)
                if profiler is None or not profiler.is_active or not _NVTX_AVAILABLE:
                    return func(self_instance, *args, **kwargs)

                label = message or func.__name__
                if profiler.discrete:
                    torch.cuda.profiler.start()
                range_id = nvtx.start_range(message=label, color=color, domain=domain)
                try:
                    result = func(self_instance, *args, **kwargs)
                finally:
                    nvtx.end_range(range_id)
                    if profiler.discrete:
                        torch.cuda.profiler.stop()
                return result

            return sync_wrapper

        return decorator

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        nsight_cfg: Optional[dict[str, Any]],
        role: str,
        rank: int,
    ) -> "NsightProfiler":
        """Create a profiler from the ``nsight_profiler`` config section.

        Args:
            nsight_cfg: The full ``nsight_profiler`` dict from the top-level config.
                If None or ``steps`` is None, returns a disabled profiler.
            role: One of ``"actor"``, ``"rollout"``, ``"env"``.
            rank: The worker's rank.

        Returns:
            A configured NsightProfiler instance.
        """
        if nsight_cfg is None:
            return cls(enable=False, rank=rank)

        steps = nsight_cfg.get("steps", None)
        if steps is None:
            return cls(enable=False, rank=rank)

        role_cfg = nsight_cfg.get(role, {})
        return cls(
            enable=role_cfg.get("enable", False),
            rank=rank,
            all_ranks=role_cfg.get("all_ranks", True),
            ranks=role_cfg.get("ranks", []),
            discrete=nsight_cfg.get("discrete", False),
        )
