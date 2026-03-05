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

import os

from ray._private.accelerators.amd_gpu import AMDGPUAcceleratorManager

from .accelerator import AcceleratorManager, AcceleratorType


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

        env_vars["ROCR_VISIBLE_DEVICES"] = visible_accelerators_str
        env_vars["RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"] = "1"
        # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L99

        return env_vars

    @staticmethod
    def get_visible_devices():
        """Get the visible device IDs."""
        visible_devices = os.environ.get("ROCR_VISIBLE_DEVICES", None)

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
