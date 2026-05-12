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

import os
from typing import TYPE_CHECKING, Optional

from ray._private.accelerators.intel_gpu import IntelGPUAcceleratorManager

from .accelerator import AcceleratorManager, AcceleratorType

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions


@AcceleratorManager.register_manager(AcceleratorType.INTEL_GPU)
class IntelGPUManager(AcceleratorManager):
    """Utility Class for Intel GPU."""
    @staticmethod
    def _format_oneapi_selector(visible_accelerators: list[str]) -> str:
        """Format selector values as '<index>:*' entries for oneAPI/SYCL."""
        # Different SYCL runtimes parse ONEAPI_DEVICE_SELECTOR differently.
        # A broadly compatible form is a single backend-qualified selector:
        #   "level_zero:0,1,2,3:*"
        backend = os.environ.get("ONEAPI_SELECTOR_BACKEND", "level_zero").strip()
        if backend == "":
            backend = "level_zero"

        ids: list[str] = []
        selectors: list[str] = []
        for acc in visible_accelerators:
            acc = acc.strip()
            if acc == "":
                continue
            if acc.isdigit():
                ids.append(acc)
                continue
            if acc.endswith(":*") and acc[:-2].isdigit():
                ids.append(acc[:-2])
                continue
            if ":" in acc:
                parts = [p.strip() for p in acc.split(":") if p.strip() != ""]
                if len(parts) >= 2 and parts[-1] == "*" and parts[-2].isdigit():
                    ids.append(parts[-2])
                    continue
                if parts and parts[-1].isdigit():
                    ids.append(parts[-1])
                    continue

            # Keep unknown/preformatted values unchanged as a fallback.
            selectors.append(acc)

        if ids:
            return f"{backend}:{','.join(ids)}"
        if selectors:
            return ",".join(selectors)
        return ""

    @staticmethod
    def _torch_xpu_device_count() -> int:
        """Get Intel XPU count from torch as a fallback when Ray probing is unavailable."""
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.xpu.device_count()
        except Exception:
            return 0
        return 0

    @staticmethod
    def get_num_devices():
        """Get the number of Intel GPU devices on the node."""
        #return IntelGPUAcceleratorManager.get_current_node_num_accelerators()
        if IntelGPUAcceleratorManager is not None:
            try:
                num_devices = (
                    IntelGPUAcceleratorManager.get_current_node_num_accelerators()
                )
                if num_devices > 0:
                    return num_devices
            except Exception:
                pass
        return IntelGPUManager._torch_xpu_device_count()

    @staticmethod
    def get_accelerator_type():
        """Get the type of the accelerator."""
        return AcceleratorType.INTEL_GPU

    @staticmethod
    def get_accelerator_model():
        """Get the model of the Intel GPU."""
        #return IntelGPUAcceleratorManager.get_current_node_accelerator_type()
        if IntelGPUAcceleratorManager is not None:
            try:
                model = IntelGPUAcceleratorManager.get_current_node_accelerator_type()
                if model is not None and model != "":
                    return model
            except Exception:
                pass

        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.xpu.get_device_name(0)
        except Exception:
            pass
        return "Intel_XPU"

    @staticmethod
    def get_accelerator_env_var(visible_accelerators: list[str]) -> dict[str, str]:
        """Get the environment variables related to the accelerator.

        Args:
            visible_accelerators (List[str]): A list of visible accelerator IDs.

        Returns:
            Dict[str, str]: A dictionary containing the accelerator environment variables.
        """
        env_vars = {}

        visible_accelerators_str = IntelGPUManager._format_oneapi_selector(
            visible_accelerators
        )
        if visible_accelerators_str != "":
            env_vars["ONEAPI_DEVICE_SELECTOR"] = visible_accelerators_str

        if len(visible_accelerators) > 0:
            env_vars["MUJOCO_EGL_DEVICE_ID"] = str(visible_accelerators[0])

        env_vars["RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR"] = "1"
        # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L94

        return env_vars

    @staticmethod
    def get_visible_devices():
        """Get the visible device IDs."""
        visible_devices = os.environ.get("ONEAPI_DEVICE_SELECTOR", None)

        if visible_devices is None or visible_devices == "":
            return []
        else:
            try:
               # parsed_visible_devices = []
              #  for selector in visible_devices.split(","):
               #     selector = selector.strip()
                #    if selector == "":
                 #       continue
                    # Accept both plain indexes ('0') and oneAPI selectors ('0:*').
                 #   if ":" in selector:
                  #      selector = selector.split(":", 1)[0].strip()
                  #  parsed_visible_devices.append(int(selector))
               # visible_devices = parsed_visible_devices
                parsed_visible_devices = []
                for selector in visible_devices.split(","):
                    selector = selector.strip()
                    if selector == "":
                        continue
                    # Accept selectors such as:
                    # - "3"
                    # - "3:*"
                    # - "level_zero:3"
                    # - "level_zero:3:*"
                    parts = [p.strip() for p in selector.split(":") if p.strip() != ""]
                    index_str = None

                    if len(parts) == 1 and parts[0].isdigit():
                        index_str = parts[0]
                    elif len(parts) >= 2 and parts[-1] == "*" and parts[-2].isdigit():
                        index_str = parts[-2]
                    elif parts[-1].isdigit():
                        index_str = parts[-1]

                    if index_str is None or not index_str.isdigit():
                        continue
                    parsed_visible_devices.append(int(index_str))
                visible_devices = parsed_visible_devices
                #visible_devices = [int(v.strip()) for v in visible_devices.split(",")]
            except ValueError:
                raise ValueError(
                    f"Invalid visible device IDs: {visible_devices}. "
                    "Please ensure they are integers separated by commas."
                )
            return visible_devices

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

        #return torch.xpu
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
