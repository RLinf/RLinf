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

# Override of Ray's AMDGPUAcceleratorManager
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py

import logging
import os
import shlex
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

from ray._private.accelerators.amd_gpu import AMDGPUAcceleratorManager

from .accelerator import AcceleratorManager, AcceleratorType, ProfileConfig

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level profiling state — toggled by the runner around gated steps.
# Workers read this via AMDGPUManager.is_profiling_active().
# ---------------------------------------------------------------------------

_amd_profiling_active: bool = False


# ---------------------------------------------------------------------------
# RocprofSysConfig — ROCm Systems Profiler profiling configuration
# ---------------------------------------------------------------------------

# Sensible defaults that mirror Ray's ROCPROFSYS_DEFAULT_CONFIG:
# https://github.com/ray-project/ray/blob/master/python/ray/_private/runtime_env/rocprof_sys.py
_ROCPROFSYS_DEFAULT_ENV: dict[str, str] = {
    "ROCPROFSYS_TIME_OUTPUT": "false",
}


@AcceleratorManager.register_profiling_config(AcceleratorType.AMD_GPU)
@dataclass
class RocprofSysConfig(ProfileConfig):
    """Configuration for profiling AMD GPU workers with ROCm Systems Profiler.

    Wraps each matching worker's Python interpreter with
    ``rocprof-sys-python [flags] -- <python_interpreter>``.  The scheduler
    automatically sets ``ROCPROFSYS_OUTPUT_PATH`` and
    ``ROCPROFSYS_OUTPUT_PREFIX`` from the configured profiling output
    directory; values provided in ``env`` override these defaults.

    Set ``cluster.profiling.backend: rocprof_sys`` to activate this backend.
    """

    BACKEND_TYPE: ClassVar[str] = "rocprof_sys"

    backend: str = "rocprof_sys"
    """Profiling backend identifier — always ``"rocprof_sys"`` for this class."""

    args: Optional[dict[str, str]] = None
    """CLI flags forwarded to ``rocprof-sys-python``.

    Single-character keys become ``-x val``; multi-character keys become
    ``--key=val``.  This mirrors the flag convention used by
    ``rocprof-sys-python`` and Ray's ``_rocprof_sys`` runtime-env plugin.
    """

    env: Optional[dict[str, str]] = None
    """Extra environment variables to inject into the profiled worker.

    Merged on top of the scheduler-derived output-path variables, so
    explicit values here take precedence over the automatic defaults.
    """

    def __post_init__(self) -> None:
        """Normalize args and env dicts after parsing from YAML."""
        super().__post_init__()

        if self.args is not None:
            assert hasattr(self.args, "keys"), (
                "RocprofSys args must be a dictionary in cluster profiling config. "
                f"But got {type(self.args)}: {self.args}"
            )
            self.args = {str(k): str(v) for k, v in self.args.items()}

        if self.env is not None:
            assert hasattr(self.env, "keys"), (
                "RocprofSys env must be a dictionary in cluster profiling config. "
                f"But got {type(self.env)}: {self.env}"
            )
            self.env = {str(k): str(v) for k, v in self.env.items()}

    def check(self) -> bool:
        """Return ``True`` if the ``rocprof-sys-python`` executable is available on PATH."""
        import shutil

        return shutil.which("rocprof-sys-python") is not None

    def to_cli_tokens(self) -> list[str]:
        """Render the ``rocprof-sys-python`` prefix tokens.

        Returns the token list up to and including the ``--`` separator;
        the caller appends the Python interpreter path as the final token.
        """
        tokens = ["rocprof-sys-python"]
        for key, val in (self.args or {}).items():
            if len(key) == 1:
                tokens.extend([f"-{key}", val])
            else:
                tokens.append(f"--{key}={val}")
        tokens.append("--")
        return tokens


@AcceleratorManager.register_manager(AcceleratorType.AMD_GPU)
class AMDGPUManager(AcceleratorManager):
    """Utility Class for AMD GPU."""

    @staticmethod
    def get_num_devices():
        """Get the number of AMD GPU devices on the node."""
        return AMDGPUAcceleratorManager.get_current_node_num_accelerators()

    @staticmethod
    def get_accelerator_type():
        """Get the type of the accelerator."""
        return AcceleratorType.AMD_GPU

    @staticmethod
    def get_accelerator_model():
        """Get the model of the AMD GPU."""
        return AMDGPUAcceleratorManager.get_current_node_accelerator_type()

    @staticmethod
    def get_accelerator_env_var(visible_accelerators: list[str]) -> dict[str, str]:
        """Get the environment variables related to the accelerator.

        Args:
            visible_accelerators (List[str]): A list of visible accelerator IDs.

        Returns:
            Dict[str, str]: A dictionary containing the accelerator environment variables.
        """
        env_vars = {}
        visible_accelerators_str = ",".join(visible_accelerators)

        env_vars["HIP_VISIBLE_DEVICES"] = visible_accelerators_str
        env_vars["RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"] = "1"
        # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L99
        env_vars["RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"] = "1"
        # https://github.com/ray-project/ray/blob/5d6c40aba865b2b80ee4dc549e603e89817ad285/python/ray/_private/accelerators/amd_gpu.py#L128

        return env_vars

    @staticmethod
    def get_visible_devices():
        """Get the visible device IDs."""
        visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", None)

        if visible_devices is None or visible_devices == "":
            return []
        else:
            try:
                visible_devices = [int(v.strip()) for v in visible_devices.split(",")]
            except ValueError:
                raise ValueError(
                    f"Invalid visible device IDs: {visible_devices}. "
                    "Please ensure they are integers separated by commas."
                )
            return visible_devices

    @staticmethod
    def get_ccl_backend():
        """Get the CCL backend."""
        return "nccl"

    @staticmethod
    def get_ccl_socket_ifname_env_var() -> str:
        """Get the network socket interface name environment variable.

        Returns:
            str: The network socket interface name environment variable.
        """
        return "NCCL_SOCKET_IFNAME"

    @staticmethod
    def get_torch_platform():
        """Get the PyTorch platform module."""
        import torch

        return torch.cuda

    @staticmethod
    def get_device_type() -> str:
        """Get the device type."""
        return "cuda"

    @staticmethod
    def get_accel_pg_options(options: Optional["CollectiveGroupOptions"]):
        """Get the accelerator CCL process group options."""
        from torch.distributed import ProcessGroupNCCL

        if options is None or options.is_empty_options():
            return None
        else:
            pg_options = ProcessGroupNCCL.Options()
            # Default values following https://github.com/NVIDIA/Megatron-LM/blob/98d8c56dbdc9cc91b8a473debcf400958bba4524/megatron/core/parallel_state.py#L160
            pg_options.config.cga_cluster_size = (
                options.accel_cluster_size or 4
            )  # Default 4
            pg_options.config.max_ctas = options.accel_max_ctas or 32  # Default 32
            pg_options.config.min_ctas = options.accel_min_ctas or 1  # Default 1
            pg_options.is_high_priority_stream = options.is_high_priority_stream

            config = pg_options.config
            assert 0 <= config.cga_cluster_size <= 8, (
                f"cga_cluster_size must be between 0 and 8, but got {config.cga_cluster_size}"
            )
            assert 1 <= config.max_ctas <= 32, (
                f"max_ctas must be between 1 and 32, but got {config.max_ctas}"
            )
            assert 1 <= config.min_ctas <= 32, (
                f"min_ctas must be between 1 and 32, but got {config.min_ctas}"
            )
            assert config.max_ctas >= config.min_ctas, (
                f"max_ctas must be greater than or equal to min_ctas, got {config.max_ctas} and {config.min_ctas}"
            )

            return pg_options

    # ------------------------------------------------------------------
    # Profiling API — ROCm Systems Profiler implementation
    # ------------------------------------------------------------------

    @staticmethod
    def modify_profiling_context(
        py_executable: str,
        profiling_cfg: "RocprofSysConfig",
        output_prefix: str,
    ) -> str:
        """Wrap ``py_executable`` with ``rocprof-sys-python`` using the given config."""
        tokens = profiling_cfg.to_cli_tokens()
        tokens.append(py_executable)
        return " ".join(shlex.quote(t) for t in tokens)

    @staticmethod
    def start_profiling(step_idx: Optional[int] = None) -> None:
        """Open a rocprof capture window via the CUDA profiler API."""
        global _amd_profiling_active
        if _amd_profiling_active:
            return
        import torch

        _amd_profiling_active = True
        torch.cuda.profiler.start()
        if step_idx is not None:
            logger.info("ROCm profiler window opened at step %d", step_idx)

    @staticmethod
    def stop_profiling() -> None:
        """Close the current rocprof capture window."""
        global _amd_profiling_active
        if not _amd_profiling_active:
            return
        import torch

        torch.cuda.profiler.stop()
        _amd_profiling_active = False

    @staticmethod
    def is_profiling_active() -> bool:
        """Check if the AMD GPU is profiling."""
        return _amd_profiling_active

    @staticmethod
    @contextmanager
    def profiling_range(
        label: str,
        color: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        """Emit an NVTX range around the enclosed block when profiling is active."""
        if not _amd_profiling_active:
            yield
            return
        import torch

        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    @staticmethod
    def get_profiling_env_vars(
        profiling_cfg: "RocprofSysConfig",
        output_prefix: str,
    ) -> dict[str, str]:
        """Return the env vars required to direct rocprof-sys output.

        Starts from ``_ROCPROFSYS_DEFAULT_ENV`` defaults, then derives
        ``ROCPROFSYS_OUTPUT_PATH`` / ``ROCPROFSYS_OUTPUT_PREFIX`` from
        ``output_prefix``, and finally overlays any explicit values from
        ``profiling_cfg.env`` (user values win).
        """
        env_vars: dict[str, str] = dict(_ROCPROFSYS_DEFAULT_ENV)

        if output_prefix:
            env_vars.setdefault(
                "ROCPROFSYS_OUTPUT_PATH", os.path.dirname(output_prefix)
            )
            env_vars.setdefault(
                "ROCPROFSYS_OUTPUT_PREFIX", os.path.basename(output_prefix)
            )

        env_vars.update(profiling_cfg.env or {})
        return env_vars
