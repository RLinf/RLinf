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

import os
from typing import TYPE_CHECKING, Optional

from .accelerator import AcceleratorManager, AcceleratorType

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions


@AcceleratorManager.register_manager(AcceleratorType.KUNLUN_XPU)
class KunlunXPUManager(AcceleratorManager):
    """Utility Class for Kunlun XPU."""

    @staticmethod
    def get_num_devices():
        """Get the number of Kunlun XPU devices on the node."""
        initialized = False
        try:
            import torch_xmlir._XMLIRC as XMLIR_C

            XMLIR_C.xpumlInit()
            initialized = True
            device_count = XMLIR_C.xpumlDeviceGetCount()
            XMLIR_C.xpumlShutdown()
            return device_count
        except Exception:
            return 0
        finally:
            if initialized:
                try:
                    XMLIR_C.xpumlShutdown()
                except Exception:
                    # Ignore shutdown errors to avoid masking earlier exceptions.
                    pass

    @staticmethod
    def get_accelerator_type():
        """Get the type of the accelerator."""
        return AcceleratorType.KUNLUN_XPU

    @staticmethod
    def get_accelerator_model():
        """Get the model of the Kunlun XPU."""
        initialized = False
        try:
            import torch_xmlir._XMLIRC as XMLIR_C

            XMLIR_C.xpumlInit()
            initialized = True
            device_count = XMLIR_C.xpumlDeviceGetCount()
            if device_count > 0:
                device = XMLIR_C.xpumlDeviceGetHandleByIndex(0)
                model = XMLIR_C.xpumlDeviceGetName(device)
                return model
            else:
                return "UNKNOWN"
        except Exception:
            return "UNKNOWN"
        finally:
            if initialized:
                try:
                    XMLIR_C.xpumlShutdown()
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

        env_vars["CUDA_VISIBLE_DEVICES"] = visible_accelerators_str
        env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
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
        return "bkcl"

    @staticmethod
    def get_ccl_socket_ifname_env_var() -> str:
        """Get the network socket interface name environment variable.

        Returns:
            str: The network socket interface name environment variable.
        """
        return "BKCL_SOCKET_IFNAME"

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
        return None
