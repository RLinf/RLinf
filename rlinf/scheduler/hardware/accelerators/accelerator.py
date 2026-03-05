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

from enum import Enum
from typing import Optional

from ..hardware import Hardware, HardwareConfig, HardwareInfo, HardwareResource


class AcceleratorType(str, Enum):
    """Enum representing different types of accelerators."""

    NV_GPU = "NV_GPU"
    AMD_GPU = "AMD_GPU"
    INTEL_GPU = "INTEL_GPU"
    NPU = "NPU"  # Huawei Ascend
    NO_ACCEL = "NO_ACCEL"


class AcceleratorManager:
    """Base Manager for accelerator-related operations."""

    manager_register: dict[AcceleratorType, type["AcceleratorManager"]] = {}

    @staticmethod
    def register_manager(accelerator_type: AcceleratorType):
        """Register an accelerator manager for a specific accelerator type."""

        def manager_decorator(manager):
            AcceleratorManager.manager_register[accelerator_type] = manager
            return manager

        return manager_decorator

    @staticmethod
    def get_num_devices():
        """Get the number of devices for the accelerator."""
        raise NotImplementedError

    @staticmethod
    def get_accelerator_type() -> AcceleratorType:
        """Get the type of the accelerator."""
        raise NotImplementedError

    @staticmethod
    def get_accelerator_model() -> str:
        """Get the model of the accelerator."""
        raise NotImplementedError

    @staticmethod
    def get_accelerator_env_var(visible_accelerators: list[str]) -> dict[str, str]:
        """Get the environment variables for a specific accelerator.

        Args:
            visible_accelerators (List[str]): A list of visible accelerator IDs.

        Returns:
            Dict[str, str]: A dictionary containing the accelerator environment variables.
        """
        raise NotImplementedError

    @staticmethod
    def get_visible_devices() -> list[int]:
        """Get the visible device IDs.

        Returns:
            List[int]: A list of visible device IDs.

        """
        raise NotImplementedError

    @staticmethod
    def get_ccl_backend() -> str:
        """Get the CCL backend.

        Returns:
            str: The CCL backend.
        """
        raise NotImplementedError

    @staticmethod
    def get_ccl_socket_ifname_env_var() -> str:
        """Get the network socket interface name environment variable.

        Returns:
            str: The network socket interface name environment variable.
        """
        raise NotImplementedError

    @staticmethod
    def get_torch_platform():
        """Get the PyTorch platform module."""
        raise NotImplementedError

    @staticmethod
    def get_device_type() -> str:
        """Get the device type."""
        raise NotImplementedError


@Hardware.register(is_default_hw=True)
class Accelerator(Hardware):
    """Enumeration policy for accelerators."""

    HW_TYPE = "Accelerator"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the hardware resources on a node.

        Args:
            node_rank (int): The rank of the node being enumerated.
            configs (Optional[list[HardwareConfig]]): The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: A list of HardwareInfo representing the hardware resources. None if no hardware is found.
        """
        for accel_type in AcceleratorManager.manager_register.keys():
            manager = AcceleratorManager.manager_register[accel_type]
            num_devices = manager.get_num_devices()
            if num_devices > 0:
                hardware_infos = [
                    HardwareInfo(
                        type=cls.HW_TYPE,
                        model=f"{accel_type.value}:{manager.get_accelerator_model()}",
                    )
                ] * num_devices
                return HardwareResource(type=cls.HW_TYPE, infos=hardware_infos)
        return None

    @classmethod
    def get_accelerator_type_from_model(cls, model: str) -> str:
        """Get the AcceleratorType from the model string.

        Args:
            model (str): The model string in the format "ACCELERATOR_TYPE:MODEL_NAME".

        Returns:
            str: The corresponding AcceleratorType.
        """
        accel_type_str = model.split(":")
        assert len(accel_type_str) == 2, (
            f"Invalid accelerator model format: {model}. Expected format: ACCELERATOR_TYPE:MODEL_NAME."
        )
        return accel_type_str[0]


class AcceleratorUtil:
    """Utility class representing an accelerator and abstracting device operations."""

    # To support an accelerator's CCL,
    # the `_new_process_group_helper` functions of `mult_channel_pg` need to be implemented
    CCL_SUPPORT_LIST = [AcceleratorType.NV_GPU, AcceleratorType.AMD_GPU]

    @staticmethod
    def get_accelerator_env_var(
        accelerator_type: AcceleratorType, visible_accelerators: list[str]
    ) -> dict[str, str]:
        """Get the environment variables related to the accelerator.

        Args:
            accelerator_type (AcceleratorType): The type of the accelerator.
            visible_accelerators (List[str]): A list of visible accelerator IDs.

        Returns:
            Dict[str, str]: A dictionary containing the accelerator environment variables.
        """
        env_vars = {}
        if accelerator_type in AcceleratorManager.manager_register:
            manager = AcceleratorManager.manager_register[accelerator_type]
            env_vars = manager.get_accelerator_env_var(visible_accelerators)
        return env_vars

    @staticmethod
    def get_visible_devices(accelerator_type: AcceleratorType) -> list[int]:
        """Get the visible device environment variable based on accelerator type.

        Args:
            accelerator_type (AcceleratorType): The type of the accelerator.

        Returns:
            List[int]: A list of visible device IDs.

        """
        visible_devices = []
        if accelerator_type in AcceleratorManager.manager_register:
            manager = AcceleratorManager.manager_register[accelerator_type]
            visible_devices = manager.get_visible_devices()
        return visible_devices

    @staticmethod
    def get_ccl_backend(accelerator_type: AcceleratorType):
        """Get the CCL backend based on the accelerator type.

        Args:
            accelerator_type (AcceleratorType): The type of the accelerator.

        Returns:
            str: The CCL backend.
        """
        if accelerator_type == AcceleratorType.NO_ACCEL:
            return None
        elif accelerator_type in AcceleratorManager.manager_register:
            manager = AcceleratorManager.manager_register[accelerator_type]
            return manager.get_ccl_backend()
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")

    @staticmethod
    def get_ccl_socket_ifname_env_var(accelerator_type: AcceleratorType):
        """Get the network socket interface name environment variable based on the accelerator type.

        Args:
            accelerator_type (AcceleratorType): The type of the accelerator.

        Returns:
            str: The network socket interface name environment variable.
        """
        if accelerator_type == AcceleratorType.NO_ACCEL:
            return "GLOO_SOCKET_IFNAME"
        elif accelerator_type in AcceleratorManager.manager_register:
            manager = AcceleratorManager.manager_register[accelerator_type]
            return manager.get_ccl_socket_ifname_env_var()
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")

    @staticmethod
    def get_torch_platform(accelerator_type: AcceleratorType):
        """Get the PyTorch platform module based on the accelerator type."""
        if accelerator_type == AcceleratorType.NO_ACCEL:
            return None
        elif accelerator_type in AcceleratorManager.manager_register:
            manager = AcceleratorManager.manager_register[accelerator_type]
            return manager.get_torch_platform()
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")

    @staticmethod
    def get_device_type(accelerator_type: AcceleratorType):
        """Get the device type based on the accelerator type."""
        if accelerator_type == AcceleratorType.NO_ACCEL:
            return None
        elif accelerator_type in AcceleratorManager.manager_register:
            manager = AcceleratorManager.manager_register[accelerator_type]
            return manager.get_device_type()
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")
