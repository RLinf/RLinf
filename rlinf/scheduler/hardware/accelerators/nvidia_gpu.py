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

# Override Ray's NvidiaGPUAcceleratorManager
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py

import logging
import os
import shlex
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

from omegaconf import ListConfig
from ray._private.accelerators.nvidia_gpu import NvidiaGPUAcceleratorManager

from .accelerator import AcceleratorManager, AcceleratorType, ProfileConfig

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level profiling state — toggled by the runner around gated steps.
# Workers read this via NvidiaGPUManager.is_profiling_active().
# ---------------------------------------------------------------------------

_nv_profiling_active: bool = False


# ---------------------------------------------------------------------------
# NsightConfig — NVIDIA-specific profiling configuration
# ---------------------------------------------------------------------------


@AcceleratorManager.register_profiling_config(AcceleratorType.NV_GPU)
@dataclass
class NsightConfig(ProfileConfig):
    """Configuration for profiling worker processes with Nsight Systems."""

    BACKEND_TYPE: ClassVar[str] = "nsight"

    backend: str = "nsight"
    """Profiling backend identifier — always ``"nsight"`` for this class."""

    options: Optional[dict[str, str]] = None
    """Additional ``nsys profile`` options keyed by flag name."""

    flags: Optional[list[str] | str] = None
    """Additional bare ``nsys profile`` flags emitted without values."""

    @staticmethod
    def _stringify_option_value(option_value: object) -> str:
        if isinstance(option_value, bool):
            return str(option_value).lower()
        return str(option_value)

    @staticmethod
    def _normalize_capture_range_end_alias(option_value: object) -> str:
        assert option_value is not None, (
            "Nsight option 'stop-on-range-end' must have an explicit value."
        )
        normalized_value = str(option_value).strip().lower()
        if normalized_value in {"true", "1", "yes", "on"}:
            return "stop"
        if normalized_value in {"false", "0", "no", "off"}:
            return "none"
        return str(option_value)

    def __post_init__(self):
        """Normalize Nsight options and flags after parsing YAML."""
        super().__post_init__()

        if self.flags is not None:
            flags = self.flags
            if isinstance(flags, str):
                self.flags = [flags]
            else:
                assert isinstance(flags, (list, ListConfig)), (
                    "flags must be a list of strings or a single string "
                    "in cluster profiling config. "
                    f"But got {type(flags)}: {flags}"
                )
                self.flags = [str(flag_name) for flag_name in flags]
            assert all(flag_name != "" for flag_name in self.flags), (
                "Nsight flags must not contain empty names."
            )

        if self.options is not None:
            assert hasattr(self.options, "keys"), (
                "Nsight options must be a dictionary in cluster profiling config. "
                f"But got {type(self.options)}: {self.options}"
            )
            self.options = {
                str(option_name): self._stringify_option_value(option_value)
                for option_name, option_value in self.options.items()
            }
            if "stop-on-range-end" in self.options:
                assert "capture-range-end" not in self.options, (
                    "Nsight options must not specify both 'stop-on-range-end' "
                    "and 'capture-range-end'."
                )
                self.options["capture-range-end"] = (
                    self._normalize_capture_range_end_alias(
                        self.options.pop("stop-on-range-end")
                    )
                )
            assert not ("o" in self.options and "output" in self.options), (
                "Nsight options must not specify both 'o' and 'output'."
            )

        if self.steps is not None:
            # Auto-add the nsys flags that make step gating actually work.
            # cudaProfilerApi capture-range honors torch.cuda.profiler.start()/stop()
            # calls from the runner; without it, nsys ignores those API calls and
            # records the whole process, producing huge traces that mask the gating.
            # ``setdefault`` lets a user who knows what they want override these.
            self.options = self.options or {}
            self.options.setdefault("capture-range", "cudaProfilerApi")
            self.options.setdefault("capture-range-end", "stop")

        if self.flags is not None and self.options is not None:
            overlapping_names = sorted(set(self.flags).intersection(self.options))
            assert not overlapping_names, (
                "Nsight flags and options must not specify the same names. "
                f"Got duplicates: {overlapping_names}"
            )

    def check(self) -> bool:
        """Return ``True`` if the ``nsys`` executable is available on PATH."""
        import shutil

        return shutil.which("nsys") is not None

    def to_cli_tokens(self, default_output_prefix: Optional[str] = None) -> list[str]:
        """Render ``nsys profile`` options into CLI tokens."""
        flags = list(self.flags or [])
        options = dict(self.options or {})
        if default_output_prefix is not None:
            options.setdefault("o", default_output_prefix)

        option_tokens = []
        for flag_name in flags:
            if flag_name == "":
                raise ValueError("Nsight option names must not be empty.")
            option_tokens.append(
                f"--{flag_name}" if len(flag_name) > 1 else f"-{flag_name}"
            )
        for option_name, option_value in options.items():
            if option_name == "":
                raise ValueError("Nsight option names must not be empty.")
            if len(option_name) > 1:
                option_tokens.append(f"--{option_name}={option_value}")
            else:
                option_tokens.extend([f"-{option_name}", option_value])
        return option_tokens


# ---------------------------------------------------------------------------
# NvidiaGPUManager
# ---------------------------------------------------------------------------


@AcceleratorManager.register_manager(AcceleratorType.NV_GPU)
class NvidiaGPUManager(AcceleratorManager):
    """Utility Class for NVIDIA GPU."""

    @staticmethod
    def _parse_nvidia_gpu_model(model_str: str) -> str:
        """Parse the NVIDIA GPU model from the full name string.

        Args:
            model_str (str): The full name string of the NVIDIA GPU.

        Returns:
            str: The parsed model of the NVIDIA GPU.
        """
        # Example model_str: "NVIDIA GeForce RTX 3090, "NVIDIA A100-SXM4-40GB"
        UNRELATED_KEYWORDS = {"NVIDIA", "GeForce"}

        if model_str is None:
            return None

        parts = model_str.split()
        # Filter out unrelated keywords
        filtered_parts = [part for part in parts if part not in UNRELATED_KEYWORDS]
        if filtered_parts:
            return " ".join(filtered_parts)
        return "UNKNOWN"

    @staticmethod
    def get_num_devices():
        """Get the number of NVIDIA GPU devices on the node."""
        return NvidiaGPUAcceleratorManager.get_current_node_num_accelerators()

    @staticmethod
    def get_accelerator_type():
        """Get the type of the accelerator."""
        return AcceleratorType.NV_GPU

    @staticmethod
    def get_accelerator_model():
        """Get the model of the NVIDIA GPU."""
        import ray._private.thirdparty.pynvml as pynvml

        initialized = False
        try:
            pynvml.nvmlInit()
            initialized = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            model = pynvml.nvmlDeviceGetName(handle)
            if isinstance(model, bytes):
                model = model.decode("utf-8")
            model = NvidiaGPUManager._parse_nvidia_gpu_model(model)
            pynvml.nvmlShutdown()
            return model
        except pynvml.NVMLError as _:
            return "UNKNOWN"
        finally:
            if initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    # Ignore shutdown errors to avoid masking earlier exceptions.
                    pass

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

        # All the three types of GPU can be set together
        env_vars["CUDA_VISIBLE_DEVICES"] = visible_accelerators_str
        # Override Ray's control over GPU assignment
        env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96

        # Simulator env vars
        if len(visible_accelerators) > 0:
            env_vars["MUJOCO_EGL_DEVICE_ID"] = str(visible_accelerators[0])

        # NCCL env vars
        env_vars["NCCL_CUMEM_ENABLE"] = "0"
        env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        if os.environ.get("NCCL_CUMEM_ENABLE", "0") != "0":
            warnings.warn(
                f"NCCL_CUMEM_ENABLE is set to {os.environ['NCCL_CUMEM_ENABLE']}. However, "
                "This may increase memory overhead with cudagraph+allreduce: "
                "https://github.com/NVIDIA/nccl/issues/1234, and thus set to 0 by both vLLM and SGLang, see https://github.com/vllm-project/vllm/pull/24141.",
            )
            env_vars["NCCL_CUMEM_ENABLE"] = os.environ["NCCL_CUMEM_ENABLE"]

        return env_vars

    @staticmethod
    def get_visible_devices():
        """Get the visible device IDs."""
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

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
                f"max_ctas must be greater than or equal to min_ctas, but got {config.max_ctas} and {config.min_ctas}"
            )

            return pg_options

    # ------------------------------------------------------------------
    # Profiling API — NVIDIA-specific implementation
    # ------------------------------------------------------------------

    @staticmethod
    def modify_profiling_context(
        py_executable: str,
        profiling_cfg: "NsightConfig",
        output_prefix: str,
    ) -> str:
        """Wrap ``py_executable`` with ``nsys profile`` using the given config."""
        nsight_cmd = [
            "nsys",
            "profile",
            *profiling_cfg.to_cli_tokens(default_output_prefix=output_prefix),
            py_executable,
        ]
        return " ".join(shlex.quote(token) for token in nsight_cmd)

    @staticmethod
    def start_profiling(step_idx: Optional[int] = None) -> None:
        """Open an nsys capture window via the CUDA profiler API."""
        global _nv_profiling_active
        if _nv_profiling_active:
            return
        import torch

        _nv_profiling_active = True
        torch.cuda.profiler.start()
        if step_idx is not None:
            logger.info("Nsight profiler window opened at step %d", step_idx)

    @staticmethod
    def stop_profiling() -> None:
        """Close the current nsys capture window."""
        global _nv_profiling_active
        if not _nv_profiling_active:
            return
        import torch

        torch.cuda.profiler.stop()
        _nv_profiling_active = False

    @staticmethod
    def is_profiling_active() -> bool:
        """Check if the NVIDIA GPU is profiling."""
        return _nv_profiling_active

    @staticmethod
    @contextmanager
    def profiling_range(
        label: str,
        color: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        """Emit an NVTX range around the enclosed block when profiling is active."""
        if not _nv_profiling_active:
            yield
            return
        import torch

        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
