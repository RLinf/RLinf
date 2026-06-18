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

import contextlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Optional

import torch
from omegaconf import ListConfig

from ..hardware import Hardware, HardwareConfig, HardwareInfo, HardwareResource

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions


# ---------------------------------------------------------------------------
# ProfileConfig — generic base for per-accelerator profiling configurations
# ---------------------------------------------------------------------------


@dataclass
class ProfileConfig:
    """Base configuration for profiling worker processes.

    Concrete backends subclass this and set ``BACKEND_TYPE`` to their
    backend name string (e.g. ``"nsight"``).  Non-profiling accelerators
    register an empty subclass so that the registry is always complete.
    """

    BACKEND_TYPE: ClassVar[str] = ""
    """Backend identifier.  Subclasses override this to their backend name."""

    backend: str = ""
    """Profiling backend name matching the subclass's ``BACKEND_TYPE``."""

    enabled: bool = True
    """Whether to enable profiling for matching worker groups."""

    worker_groups: Optional[list[str] | str] = None
    """Worker group names to profile.  ``None`` means no groups are profiled."""

    steps: Optional[list[int]] = None
    """Function step indices to gate profiling around.

    When ``None`` (default), every training step is profiled if profiling is
    enabled.
    """

    output_dir: Optional[str] = None
    """Directory for profiling output files.

    When ``None``, ``validate_cfg`` derives a default path under the run's log directory.
    """

    def __post_init__(self) -> None:
        """Normalize common profiling fields."""
        if self.worker_groups is not None:
            worker_groups = self.worker_groups
            if isinstance(worker_groups, str):
                self.worker_groups = [worker_groups]
            else:
                assert isinstance(worker_groups, (list, ListConfig)), (
                    "worker_groups must be a list of strings or a single string "
                    "in profiling config. "
                    f"But got {type(worker_groups)}: {worker_groups}"
                )
                self.worker_groups = [str(g) for g in worker_groups]

        if self.steps is not None:
            assert isinstance(self.steps, (list, ListConfig)), (
                "steps must be a list of ints in profiling config. "
                f"But got {type(self.steps)}: {self.steps}"
            )
            self.steps = [int(s) for s in self.steps]
            assert all(s >= 0 for s in self.steps), (
                f"Profiling steps must be non-negative ints. But got: {self.steps}"
            )

    def profiles_worker_group(self, worker_group_name: str) -> bool:
        """Return whether this config should profile the given worker group."""
        if not self.enabled or not self.worker_groups:
            return False
        normalized = {g.lower() for g in self.worker_groups}
        return "all" in normalized or worker_group_name.lower() in normalized

    def should_profile_step(self, step_idx: int) -> bool:
        """Return whether the given step should be gated for profiling."""
        if not self.enabled:
            return False
        if self.steps is None:
            return True
        return step_idx in self.steps

    def check(self) -> bool:
        """Return ``True`` if all required profiling tools are available on this node.

        Subclasses override this to check backend-specific executables.
        """
        return True


class AcceleratorType(str, Enum):
    """Enum representing different types of accelerators."""

    NV_GPU = "NV_GPU"
    AMD_GPU = "AMD_GPU"
    INTEL_GPU = "INTEL_GPU"
    NPU = "NPU"  # Huawei Ascend
    NO_ACCEL = "NO_ACCEL"
    MUSA_GPU = "MUSA_GPU"


class AcceleratorManager:
    """Base Manager for accelerator-related operations."""

    manager_register: dict[AcceleratorType, type["AcceleratorManager"]] = {}
    profiling_config_register: dict[AcceleratorType, type["ProfileConfig"]] = {}
    profile_backend_register: dict[str, type["ProfileConfig"]] = {}

    @staticmethod
    def register_manager(accelerator_type: AcceleratorType):
        """Register an accelerator manager for a specific accelerator type."""

        def manager_decorator(manager):
            AcceleratorManager.manager_register[accelerator_type] = manager
            return manager

        return manager_decorator

    @staticmethod
    def register_profiling_config(accelerator_type: AcceleratorType):
        """Register a profiling config class for a specific accelerator type.

        Also registers the class in ``profile_backend_register`` keyed by
        ``cls.BACKEND_TYPE`` when that string is non-empty.
        """

        def decorator(cls):
            AcceleratorManager.profiling_config_register[accelerator_type] = cls
            if getattr(cls, "BACKEND_TYPE", ""):
                AcceleratorManager.profile_backend_register[cls.BACKEND_TYPE] = cls
            return cls

        return decorator

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

    @staticmethod
    def get_accel_pg_options(options: Optional["CollectiveGroupOptions"]):
        """Get the accelerator CCL process group options.

        Args:
            options (Optional[CollectiveGroupOptions]): The options for the collective group.

        Returns:
            Optional[dist.ProcessGroup.Options]: The accelerator CCL process group options.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Profiling API — no-op defaults, overridden per accelerator type.
    # ------------------------------------------------------------------

    @staticmethod
    def modify_profiling_context(
        py_executable: str,
        profiling_cfg,
        output_prefix: str,
    ) -> str:
        """Prepend a profiling wrapper command to the Python interpreter path.

        Returns the unmodified ``py_executable`` by default (no profiling).
        NVIDIA overrides this to prepend ``nsys profile ...``.
        """
        return py_executable

    @staticmethod
    def start_profiling(step_idx: Optional[int] = None) -> None:
        """Open a step-gated profiling capture window. No-op by default."""

    @staticmethod
    def stop_profiling() -> None:
        """Close the current profiling capture window. No-op by default."""

    @staticmethod
    def is_profiling_active() -> bool:
        """Return whether a profiling capture window is currently open."""
        return False

    @staticmethod
    def profiling_range(
        label: str,
        color: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        """Return a context manager that annotates a code range for profiling.

        The default implementation is a no-op. NVIDIA overrides this to emit
        an NVTX range when profiling is active.
        """
        return contextlib.nullcontext()

    @staticmethod
    def get_profiling_env_vars(
        profiling_cfg: "ProfileConfig",
        output_prefix: str,
    ) -> dict[str, str]:
        """Return additional environment variables required by the profiling backend.

        Called once per worker allocation when profiling is active.  The default
        implementation returns an empty dict; backends that need env-var-based
        configuration (e.g. ``ROCPROFSYS_OUTPUT_PATH``) override this.
        """
        return {}

    @staticmethod
    def check_profiler(profiling_cfg: "ProfileConfig") -> bool:
        """Return ``True`` if all tools required by the profiling backend are available on this node.

        Delegates to ``profiling_cfg.check()``.
        """
        return profiling_cfg.check()


@Hardware.register(is_default_hw=True)
class Accelerator(Hardware):
    """Enumeration policy for accelerators."""

    HW_TYPE = "Accelerator"

    @classmethod
    def enumerate(
        cls,
        node_rank: Optional[int] = None,
        configs: Optional[list[HardwareConfig]] = None,
    ) -> Optional[HardwareResource]:
        """Enumerate the hardware resources on a node.

        Args:
            node_rank (Optional[int]): The rank of the node being enumerated.
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
    CCL_SUPPORT_LIST = [
        AcceleratorType.NV_GPU,
        AcceleratorType.AMD_GPU,
        AcceleratorType.MUSA_GPU,
    ]

    @staticmethod
    def get_accelerator_type() -> AcceleratorType:
        """Get the current accelerator type even not in Worker environment."""
        hw_res = Accelerator.enumerate()
        if hw_res is not None and len(hw_res.infos) > 0:
            return Accelerator.get_accelerator_type_from_model(hw_res.infos[0].model)
        return AcceleratorType.NO_ACCEL

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
    def get_torch_platform(accelerator_type: AcceleratorType) -> torch.cuda:
        """Get the PyTorch platform module based on the accelerator type."""
        if accelerator_type == AcceleratorType.NO_ACCEL:
            return None
        elif accelerator_type in AcceleratorManager.manager_register:
            manager = AcceleratorManager.manager_register[accelerator_type]
            return manager.get_torch_platform()
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")

    @staticmethod
    def get_device_type(accelerator_type: AcceleratorType) -> str | None:
        """Get the device type based on the accelerator type."""
        if accelerator_type == AcceleratorType.NO_ACCEL:
            return None
        elif accelerator_type in AcceleratorManager.manager_register:
            manager = AcceleratorManager.manager_register[accelerator_type]
            return manager.get_device_type()
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")

    @staticmethod
    def get_accel_pg_options(
        accelerator_type: AcceleratorType, options: Optional["CollectiveGroupOptions"]
    ):
        """Get the accelerator CCL process group options based on the accelerator type."""
        if accelerator_type == AcceleratorType.NO_ACCEL:
            return None
        elif accelerator_type in AcceleratorManager.manager_register:
            manager = AcceleratorManager.manager_register[accelerator_type]
            return manager.get_accel_pg_options(options=options)
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")

    @staticmethod
    def start_profiling(
        accelerator_type: AcceleratorType, step_idx: Optional[int] = None
    ) -> None:
        """Open a step-gated profiling capture window for the given accelerator."""
        if accelerator_type in AcceleratorManager.manager_register:
            AcceleratorManager.manager_register[accelerator_type].start_profiling(
                step_idx
            )

    @staticmethod
    def stop_profiling(accelerator_type: AcceleratorType) -> None:
        """Close the current profiling capture window for the given accelerator."""
        if accelerator_type in AcceleratorManager.manager_register:
            AcceleratorManager.manager_register[accelerator_type].stop_profiling()

    @staticmethod
    def profiling_range(
        accelerator_type: AcceleratorType,
        label: str,
        color: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        """Return a context manager that annotates a code region for profiling.

        Dispatches to the registered manager for the given accelerator type.
        Falls back to a no-op context manager when no manager is registered.
        """
        if accelerator_type in AcceleratorManager.manager_register:
            return AcceleratorManager.manager_register[
                accelerator_type
            ].profiling_range(label, color=color, domain=domain)
        return contextlib.nullcontext()
