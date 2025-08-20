import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


class PyTorchProfiler:
    """
    PyTorch Profiler wrapper for RL training.

    This class provides a convenient interface to use PyTorch's profiler
    for performance analysis during training.
    """

    def __init__(
        self,
        enabled: bool = False,
        output_dir: str = "./profiler_output",
        activities: Optional[list] = None,
        record_shapes: bool = False,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        with_modules: bool = False,
        use_cuda: bool = True,
        use_cpu: bool = True,
        profile_steps: Optional[list] = None,
        filename_prefix: str = "chrome_trace",
        include_rank_in_filename: bool = True,
        synchronize_cuda_on_exit: bool = True,
        export_chrome_trace: bool = True,
        export_tensorboard: bool = True,
    ):
        """
        Initialize the PyTorch Profiler.

        Args:
            enabled: Whether to enable profiling
            output_dir: Directory to save profiler outputs
            activities: List of activities to profile (ProfilerActivity.CPU, ProfilerActivity.CUDA)
            record_shapes: Whether to record tensor shapes
            profile_memory: Whether to profile memory usage
            with_stack: Whether to include stack traces
            with_flops: Whether to compute FLOPs
            with_modules: Whether to include module information
            use_cuda: Whether to profile CUDA operations
            use_cpu: Whether to profile CPU operations
            export_chrome_trace: Whether to export Chrome trace
            export_tensorboard: Whether to export TensorBoard trace
        """
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        self.use_cuda = use_cuda
        self.use_cpu = use_cpu
        self._profile_steps_set = set(profile_steps) if profile_steps else None
        self.filename_prefix = filename_prefix
        self.include_rank_in_filename = include_rank_in_filename
        self.synchronize_cuda_on_exit = synchronize_cuda_on_exit
        self.export_chrome_trace = export_chrome_trace
        self.export_tensorboard = export_tensorboard

        # Set up activities
        if activities is None:
            activities = []
            if self.use_cpu:
                activities.append(ProfilerActivity.CPU)
            if self.use_cuda and torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        self.activities = activities

        # Compose exporters used when flushing a step
        self.on_trace_ready = self._compose_on_trace_ready()

        # Create output directory
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.profiler = None
        self._step_count = 0

    def _compose_on_trace_ready(self):
        """Compose a trace-ready callback from requested outputs."""
        handlers = []

        if self.export_tensorboard:
            handlers.append(
                tensorboard_trace_handler(str(self.output_dir / "tensorboard"))
            )

        if self.export_chrome_trace:

            def chrome_handler(prof):
                timestamp = int(time.time())
                chrome_trace_path = (
                    self.output_dir / f"{self.filename_prefix}_{timestamp}.json"
                )
                prof.export_chrome_trace(str(chrome_trace_path))
                print(f"Chrome trace exported to: {chrome_trace_path}")

            handlers.append(chrome_handler)

        def composed(prof):
            for handler in handlers:
                try:
                    handler(prof)
                except Exception as exc:
                    print(f"Profiler trace handler error: {exc}")

        return composed

    def is_selected_step(self, step_index: int) -> bool:
        """Return True iff this step should be profiled.

        If profile_steps provided, only those steps are profiled; else all.
        """
        if not self.enabled:
            return False
        if self._profile_steps_set is not None:
            return step_index in self._profile_steps_set
        return True

    @contextmanager
    def profile_step(self, step_name: Optional[str] = None):
        """
        Context manager for profiling a single step.

        Args:
            step_name: Name of the step being profiled
        """
        if not self.enabled:
            yield
            return

        # Always use a short-lived profiler per selected step and export immediately
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with profile(
            activities=self.activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules,
        ) as prof:
            yield

        # Ensure CUDA work is finished before export for accurate timings
        if self.synchronize_cuda_on_exit and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        # Export traces for this step
        if self.export_tensorboard:
            try:
                tb_handler = tensorboard_trace_handler(
                    str(self.output_dir / "tensorboard")
                )
                tb_handler(prof)
            except Exception as exc:
                print(f"TensorBoard export error: {exc}")

        if self.export_chrome_trace:
            try:
                step_tag = step_name or f"step_{self._step_count}"
                fname = f"{self.filename_prefix}_{step_tag}"
                # Enrich filename with rank/device to avoid collisions on multi-rank runs
                if self.include_rank_in_filename:
                    try:
                        rank = (
                            torch.distributed.get_rank()
                            if torch.distributed.is_initialized()
                            else None
                        )
                    except Exception:
                        rank = None
                    dev = None
                    try:
                        if torch.cuda.is_available():
                            dev = torch.cuda.current_device()
                    except Exception:
                        dev = None
                    if rank is not None:
                        fname += f"_rank{rank}"
                    if dev is not None:
                        fname += f"_dev{dev}"
                chrome_path = self.output_dir / f"{fname}.json"
                prof.export_chrome_trace(str(chrome_path))
            except Exception as exc:
                print(f"Chrome trace export error: {exc}")

        self._step_count += 1

    def start(self):
        """No-op (kept for API compatibility)."""
        return

    def step(self):
        """No-op (kept for API compatibility)."""
        return

    def stop(self):
        """No-op (kept for API compatibility)."""
        return

    def get_stats(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        if not self.enabled or self.profiler is None:
            return {}

        stats = {
            "step_count": self._step_count,
            "output_dir": str(self.output_dir),
        }

        return stats

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PyTorchProfiler":
        """
        Create profiler from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            PyTorchProfiler instance
        """
        return cls(**config)


def create_profiler_from_config(cfg) -> Optional[PyTorchProfiler]:
    """
    Create profiler from configuration.

    Args:
        cfg: Configuration object

    Returns:
        PyTorchProfiler instance or None if not enabled
    """
    if not hasattr(cfg, "profiler") or not cfg.profiler.get("enabled", False):
        return None

    profiler_config = cfg.profiler.copy()

    # Filter out parameters that are not supported by PyTorchProfiler
    # These parameters are used by megatron_worker.py for operator-level profiling
    supported_params = {
        "enabled",
        "output_dir",
        "activities",
        "record_shapes",
        "profile_memory",
        "with_stack",
        "with_flops",
        "with_modules",
        "use_cuda",
        "use_cpu",
        "profile_steps",
        "filename_prefix",
        "include_rank_in_filename",
        "synchronize_cuda_on_exit",
        "export_chrome_trace",
        "export_tensorboard",
    }

    # Remove unsupported parameters (like profile_interval)
    profiler_config = {
        k: v for k, v in profiler_config.items() if k in supported_params
    }

    # Set default output directory if not specified
    if "output_dir" not in profiler_config:
        profiler_config["output_dir"] = os.path.join(
            cfg.trainer.output_dir, cfg.trainer.experiment_name, "profiler_output"
        )

    return PyTorchProfiler.from_config(profiler_config)
