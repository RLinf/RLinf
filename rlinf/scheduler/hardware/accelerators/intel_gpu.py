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

import ctypes
import logging
import os
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Optional

from ray._private.accelerators.intel_gpu import IntelGPUAcceleratorManager

from .accelerator import AcceleratorManager, AcceleratorType, ProfileConfig

logger = logging.getLogger(__name__)

_ONEAPI_DEVICE_SELECTOR = "ONEAPI_DEVICE_SELECTOR"
_NOSET_ONEAPI_DEVICE_SELECTOR = "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR"

EGL_DEVICE_ID_ENV_VARS = ("MUJOCO_EGL_DEVICE_ID", "EGL_DEVICE_ID")

_EGL_DEVICE_UUID_EXT = 0x335C
_MAX_EGL_DEVICES = 64
_ZE_INIT_FLAG_GPU_ONLY = 1
_ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3
_ZE_MAX_DEVICE_UUID_SIZE = 16
_ZE_MAX_DEVICE_NAME = 256
_CPU_RENDERING_BACKENDS = frozenset({"osmesa", "glx"})

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


class _ZeDeviceUuid(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint8 * _ZE_MAX_DEVICE_UUID_SIZE)]


class _ZeDeviceProperties(ctypes.Structure):
    """ctypes representation of ``ze_device_properties_t``."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("vendorId", ctypes.c_uint32),
        ("deviceId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subdeviceId", ctypes.c_uint32),
        ("coreClockRate", ctypes.c_uint32),
        ("maxMemAllocSize", ctypes.c_uint64),
        ("maxHardwareContexts", ctypes.c_uint32),
        ("maxCommandQueuePriority", ctypes.c_uint32),
        ("numThreadsPerEU", ctypes.c_uint32),
        ("physicalEUSimdWidth", ctypes.c_uint32),
        ("numEUsPerSubslice", ctypes.c_uint32),
        ("numSubslicesPerSlice", ctypes.c_uint32),
        ("numSlices", ctypes.c_uint32),
        ("timerResolution", ctypes.c_uint64),
        ("timestampValidBits", ctypes.c_uint32),
        ("kernelTimestampValidBits", ctypes.c_uint32),
        ("uuid", _ZeDeviceUuid),
        ("name", ctypes.c_char * _ZE_MAX_DEVICE_NAME),
    ]


def _check_ze_result(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with Level Zero result 0x{result:x}")


def _query_level_zero_device_uuids() -> list[bytes]:
    """Return root-device UUIDs in Level Zero enumeration order."""
    try:
        libze = ctypes.CDLL("libze_loader.so.1")
    except OSError as exc:
        raise OSError(f"libze_loader.so.1 is not loadable: {exc}") from exc

    uint32_p = ctypes.POINTER(ctypes.c_uint32)
    handle_t = ctypes.c_void_p
    handle_p = ctypes.POINTER(handle_t)

    libze.zeInit.argtypes = [ctypes.c_uint32]
    libze.zeInit.restype = ctypes.c_uint32
    libze.zeDriverGet.argtypes = [uint32_p, handle_p]
    libze.zeDriverGet.restype = ctypes.c_uint32
    libze.zeDeviceGet.argtypes = [handle_t, uint32_p, handle_p]
    libze.zeDeviceGet.restype = ctypes.c_uint32
    libze.zeDeviceGetProperties.argtypes = [
        handle_t,
        ctypes.POINTER(_ZeDeviceProperties),
    ]
    libze.zeDeviceGetProperties.restype = ctypes.c_uint32

    _check_ze_result(libze.zeInit(_ZE_INIT_FLAG_GPU_ONLY), "zeInit")
    driver_count = ctypes.c_uint32()
    _check_ze_result(
        libze.zeDriverGet(ctypes.byref(driver_count), None), "zeDriverGet(count)"
    )
    drivers = (handle_t * driver_count.value)()
    _check_ze_result(
        libze.zeDriverGet(ctypes.byref(driver_count), drivers), "zeDriverGet"
    )

    device_uuids = []
    for driver in drivers:
        device_count = ctypes.c_uint32()
        _check_ze_result(
            libze.zeDeviceGet(driver, ctypes.byref(device_count), None),
            "zeDeviceGet(count)",
        )
        devices = (handle_t * device_count.value)()
        _check_ze_result(
            libze.zeDeviceGet(driver, ctypes.byref(device_count), devices),
            "zeDeviceGet",
        )
        for device in devices:
            properties = _ZeDeviceProperties(stype=_ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES)
            _check_ze_result(
                libze.zeDeviceGetProperties(device, ctypes.byref(properties)),
                "zeDeviceGetProperties",
            )
            device_uuids.append(bytes(properties.uuid.id))
    return device_uuids


def _query_egl_index_by_device_uuid() -> dict[bytes, int]:
    """Map EGL persistent device UUIDs to EGL enumeration indices."""
    try:
        libegl = ctypes.CDLL("libEGL.so.1")
    except OSError as exc:
        raise OSError(f"libEGL.so.1 is not loadable: {exc}") from exc

    libegl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    libegl.eglGetProcAddress.restype = ctypes.c_void_p
    query_devices_ptr = libegl.eglGetProcAddress(b"eglQueryDevicesEXT")
    query_binary_ptr = libegl.eglGetProcAddress(b"eglQueryDeviceBinaryEXT")
    if not query_devices_ptr or not query_binary_ptr:
        raise RuntimeError(
            "EGL_EXT_device_enumeration and EGL_EXT_device_persistent_id are required"
        )

    device_t = ctypes.c_void_p
    boolean_t = ctypes.c_uint
    int_t = ctypes.c_int
    query_devices = ctypes.CFUNCTYPE(
        boolean_t, int_t, ctypes.POINTER(device_t), ctypes.POINTER(int_t)
    )(query_devices_ptr)
    query_binary = ctypes.CFUNCTYPE(
        boolean_t,
        device_t,
        int_t,
        int_t,
        ctypes.c_void_p,
        ctypes.POINTER(int_t),
    )(query_binary_ptr)

    devices = (device_t * _MAX_EGL_DEVICES)()
    device_count = int_t()
    if not query_devices(_MAX_EGL_DEVICES, devices, ctypes.byref(device_count)):
        raise RuntimeError("eglQueryDevicesEXT failed")

    egl_index_by_uuid = {}
    for egl_index in range(device_count.value):
        uuid = (ctypes.c_uint8 * _ZE_MAX_DEVICE_UUID_SIZE)()
        uuid_size = int_t()
        queried = query_binary(
            devices[egl_index],
            _EGL_DEVICE_UUID_EXT,
            _ZE_MAX_DEVICE_UUID_SIZE,
            uuid,
            ctypes.byref(uuid_size),
        )
        if queried and uuid_size.value == _ZE_MAX_DEVICE_UUID_SIZE:
            egl_index_by_uuid[bytes(uuid)] = egl_index
    return egl_index_by_uuid


@cache
def _egl_index_by_level_zero_device() -> dict[int, int]:
    """Map Level Zero root-device ordinals to EGL enumeration indices."""
    try:
        level_zero_uuids = _query_level_zero_device_uuids()
        egl_index_by_uuid = _query_egl_index_by_device_uuid()
    except (OSError, RuntimeError) as exc:
        logger.debug("Cannot map Level Zero devices to EGL devices (%s).", exc)
        return {}

    return {
        device_id: egl_index_by_uuid[device_uuid]
        for device_id, device_uuid in enumerate(level_zero_uuids)
        if device_uuid in egl_index_by_uuid
    }


def _renders_with_egl() -> bool:
    """Report whether a renderer could go through EGL in this run."""
    return os.environ.get("MUJOCO_GL") not in _CPU_RENDERING_BACKENDS


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
        """Return worker-local oneCCL topology and renderer defaults."""
        env_vars = {}
        local_rank = os.environ.get("NODE_LOCAL_RANK")
        local_size = os.environ.get("NODE_LOCAL_WORLD_SIZE")
        if local_rank is not None and local_size is not None:
            # Ray is not an MPI launcher. Without this topology oneCCL can try
            # MPI discovery and hang while inferring local_idx/local_count.
            env_vars["CCL_LOCAL_RANK"] = local_rank
            env_vars["CCL_LOCAL_SIZE"] = local_size

        if not visible_accelerators or not _renders_with_egl():
            return env_vars

        level_zero_device_id = visible_accelerators[0]
        egl_device_id = IntelGPUManager.get_egl_device_id(level_zero_device_id)
        if egl_device_id is None:
            logger.debug(
                "No EGL device found for Level Zero device %s. Falling back to "
                "the Level Zero device id, which may render on the wrong GPU "
                "when the two namespaces disagree.",
                level_zero_device_id,
            )
            egl_device_id = level_zero_device_id
        env_vars.update(
            {env_var: str(egl_device_id) for env_var in EGL_DEVICE_ID_ENV_VARS}
        )
        return env_vars

    @staticmethod
    def get_egl_device_id(level_zero_device_id: int | str) -> Optional[int]:
        """Map a Level Zero device ordinal to its EGL enumeration index."""
        try:
            level_zero_device_id = int(level_zero_device_id)
        except (TypeError, ValueError):
            return None
        return _egl_index_by_level_zero_device().get(level_zero_device_id)

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
