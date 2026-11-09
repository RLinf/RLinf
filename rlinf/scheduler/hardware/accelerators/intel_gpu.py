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

# Override of Ray's IntelGPUAcceleratorManager
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ray._private.accelerators.intel_gpu import IntelGPUAcceleratorManager

from .accelerator import AcceleratorManager, AcceleratorType, ProfileConfig

logger = logging.getLogger(__name__)

_ONEAPI_DEVICE_SELECTOR = "ONEAPI_DEVICE_SELECTOR"
_NOSET_ONEAPI_DEVICE_SELECTOR = "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR"

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions


def _parse_device_ids(value: str, env_var: str) -> list[int]:
    """Parse a comma-separated accelerator visibility environment variable."""
    try:
        return [int(device_id.strip()) for device_id in value.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"Invalid device IDs in {env_var}: {value!r}. "
            "Expected comma-separated integers."
        ) from exc


def _format_oneapi_selector(visible_accelerators: list[str]) -> str:
    """Format physical device IDs as a Level Zero selector."""
    device_ids = _parse_device_ids(",".join(visible_accelerators), "device list")
    return f"level_zero:{','.join(map(str, device_ids))}"


@AcceleratorManager.register_profiling_config(AcceleratorType.INTEL_GPU)
@dataclass
class IntelGPUProfileConfig(ProfileConfig):
    """Intel GPU profiling configuration."""


@AcceleratorManager.register_manager(AcceleratorType.INTEL_GPU)
class IntelGPUManager(AcceleratorManager):
    """Utility Class for Intel GPU."""

    @staticmethod
    def _torch_xpu_device_count() -> int:
        """Return the XPU count reported by PyTorch, or zero if unavailable."""
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.xpu.device_count()
        except Exception as exc:
            logger.debug("PyTorch XPU device discovery failed: %s", exc)
        return 0

    @staticmethod
    def get_num_devices() -> int:
        """Get the number of Intel GPU devices on the node."""
        try:
            num_devices = IntelGPUAcceleratorManager.get_current_node_num_accelerators()
            if num_devices > 0:
                return num_devices
        except Exception as exc:
            logger.debug("Ray Intel GPU device discovery failed: %s", exc)

        return IntelGPUManager._torch_xpu_device_count()

    @staticmethod
    def get_accelerator_type():
        """Get the type of the accelerator."""
        return AcceleratorType.INTEL_GPU

    @staticmethod
    def get_accelerator_model() -> str:
        """Get the model of the Intel GPU."""
        try:
            model = IntelGPUAcceleratorManager.get_current_node_accelerator_type()
            if model:
                return model
        except Exception as exc:
            logger.debug("Ray Intel GPU model discovery failed: %s", exc)

        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.xpu.get_device_name(0)
        except Exception as exc:
            logger.debug("PyTorch XPU model discovery failed: %s", exc)

        return "Intel-GPU"

    @staticmethod
    def get_accelerator_env_var(visible_accelerators: list[str]) -> dict[str, str]:
        """Get the environment variables related to the accelerator.

        Args:
            visible_accelerators (List[str]): A list of visible accelerator IDs.

        Returns:
            Dict[str, str]: A dictionary containing the accelerator environment variables.
        """
        env_vars = {_NOSET_ONEAPI_DEVICE_SELECTOR: "1"}
        if visible_accelerators:
            env_vars[_ONEAPI_DEVICE_SELECTOR] = _format_oneapi_selector(
                visible_accelerators
            )

        return env_vars

    @staticmethod
    def get_worker_local_env_var(visible_accelerators: list[int]) -> dict[str, str]:
        """Return renderer defaults without overriding explicit worker settings."""
        if visible_accelerators:
            return {"MUJOCO_EGL_DEVICE_ID": str(visible_accelerators[0])}
        return {}

    @staticmethod
    def get_visible_devices():
        """Get the visible device IDs."""
        oneapi_selector = os.environ.get(_ONEAPI_DEVICE_SELECTOR)
        if not oneapi_selector or oneapi_selector == "NoDevFiles":
            return []

        # Ray historically encoded the IDs as "level_zero:0,1". Bare IDs
        # remain supported for compatibility with existing RLinf deployments.
        prefix = "level_zero:"
        if oneapi_selector.startswith(prefix):
            oneapi_selector = oneapi_selector[len(prefix) :]
        if oneapi_selector in {"gpu", "*"}:
            return []
        return _parse_device_ids(oneapi_selector, _ONEAPI_DEVICE_SELECTOR)

    @staticmethod
    def get_ccl_backend():
        """Get the CCL backend."""
        return "ccl"

    @staticmethod
    def get_ccl_socket_ifname_env_var() -> str:
        """Get the network socket interface name environment variable.

        Returns:
            str: The network socket interface name environment variable.
        """
        return "CCL_MNIC_NAME"

    @staticmethod
    def get_torch_platform():
        """Get the PyTorch platform module."""
        import torch

        xpu_platform = torch.xpu

        # Some PyTorch/IPEX versions do not expose ipc_collect on torch.xpu.
        # Keep parity with CUDA call-sites by providing a no-op fallback.
        if not hasattr(xpu_platform, "ipc_collect"):

            def _ipc_collect_noop() -> None:
                return None

            setattr(xpu_platform, "ipc_collect", _ipc_collect_noop)

        return xpu_platform

    @staticmethod
    def get_device_type() -> str:
        """Get the device type."""
        return "xpu"

    @staticmethod
    def get_accel_pg_options(options: Optional["CollectiveGroupOptions"]):
        """Get the accelerator CCL process group options."""
        return None
