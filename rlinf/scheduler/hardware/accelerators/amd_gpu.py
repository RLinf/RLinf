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

import ctypes
import logging
import os
import shlex
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Optional

from ray._private.accelerators.amd_gpu import AMDGPUAcceleratorManager

from .accelerator import AcceleratorManager, AcceleratorType, ProfileConfig

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Render devices
#
# HIP_VISIBLE_DEVICES restricts neither Vulkan nor EGL, so every simulator
# process renders on the same default GPU unless it is told otherwise.
#
# Vulkan (SAPIEN in RoboTwin and ManiSkill): Mesa's device-select layer makes
# the GPU named by DRI_PRIME=pci-DDDD_BB_DD_F the default.
#
# EGL (MuJoCo and robosuite in LIBERO): the renderer takes an index into
# eglQueryDevicesEXT, which Mesa orders independently of the HIP device ids.
#
# KFD supplies each HIP device's PCI identity and DRM render node; EGL supplies
# its enumeration index. Building the mappings does not initialize HIP.
# ---------------------------------------------------------------------------

DRI_PRIME_ENV = "DRI_PRIME"
#: EGL device index, read by MuJoCo and robosuite (``MUJOCO_EGL_DEVICE_ID``) and
#: by other EGL renderers such as pyrender (``EGL_DEVICE_ID``).
EGL_DEVICE_ID_ENV_VARS = ("MUJOCO_EGL_DEVICE_ID", "EGL_DEVICE_ID")
_EGL_DRM_RENDER_NODE_FILE_EXT = 0x3377
_MAX_EGL_DEVICES = 64
# robosuite renders with EGL for every MUJOCO_GL value other than these.
_CPU_RENDERING_BACKENDS = frozenset({"osmesa", "glx"})
_KFD_TOPOLOGY_NODES = "/sys/class/kfd/kfd/topology/nodes"
_DRI_DEVICE_DIR = "/dev/dri"
# gfx_target_version of the first RDNA ISA (gfx1010). CDNA parts (gfx9xx) have
# no rasterization hardware and render on the CPU, so they are not pinned.
_FIRST_RDNA_GFX_TARGET = 100100


def _read_kfd_node_properties(node_dir: str) -> dict[str, int]:
    """Parse the integer fields of a KFD topology ``properties`` file.

    Args:
        node_dir: Path of one ``.../topology/nodes/<id>`` directory.

    Returns:
        Property name to value. Non-integer fields are skipped.
    """
    props: dict[str, int] = {}
    with open(os.path.join(node_dir, "properties")) as handle:
        for line in handle:
            name, _, value = line.strip().partition(" ")
            try:
                props[name] = int(value)
            except ValueError:
                continue
    return props


@cache
def _rdna_gpus_by_hip_device() -> dict[int, tuple[str, str]]:
    """Map each HIP device id on this node to the render identity of its GPU.

    GPUs are numbered as ROCr numbers them: KFD nodes in order, skipping CPU
    nodes and GPUs whose render node this process cannot open, then filtered by
    ``ROCR_VISIBLE_DEVICES``. Only RDNA GPUs get an entry.

    Returns:
        HIP device id to ``(pci-DDDD_BB_DD_F, /dev/dri/renderD<minor>)``. Empty
        when the KFD topology cannot be read or ``ROCR_VISIBLE_DEVICES`` does
        not match it.
    """
    try:
        node_ids = sorted(
            int(name) for name in os.listdir(_KFD_TOPOLOGY_NODES) if name.isdigit()
        )
    except OSError as exc:
        logger.debug("Cannot read the KFD topology (%s).", exc)
        return {}

    gpus = []
    for node_id in node_ids:
        try:
            props = _read_kfd_node_properties(
                os.path.join(_KFD_TOPOLOGY_NODES, str(node_id))
            )
        except OSError:
            continue
        if not props.get("gfx_target_version"):
            continue
        render_node = os.path.join(
            _DRI_DEVICE_DIR, f"renderD{props.get('drm_render_minor', -1)}"
        )
        if not os.access(render_node, os.R_OK | os.W_OK):
            continue
        location = props.get("location_id", 0)
        gpus.append(
            (
                props["gfx_target_version"],
                f"pci-{props.get('domain', 0):04x}_{location >> 8:02x}_"
                f"{(location >> 3) & 0x1F:02x}_{location & 0x7:x}",
                render_node,
            )
        )

    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES", "")
    if rocr_visible:
        try:
            indices = [int(index) for index in rocr_visible.split(",")]
            # ROCr hides every GPU for an out-of-range index such as -1, while
            # Python would count a negative index from the end of the list.
            if any(index < 0 for index in indices):
                raise IndexError("negative device index")
            gpus = [gpus[index] for index in indices]
        except (ValueError, IndexError):
            logger.debug(
                "ROCR_VISIBLE_DEVICES=%s does not match the KFD topology.",
                rocr_visible,
            )
            return {}
    return {
        hip_id: (dri_prime, render_node)
        for hip_id, (gfx_target, dri_prime, render_node) in enumerate(gpus)
        if gfx_target >= _FIRST_RDNA_GFX_TARGET
    }


def _query_egl_index_by_render_node() -> dict[str, int]:
    """Map the DRM render node of each EGL device to its EGL enumeration index.

    The query goes through ``ctypes`` rather than PyOpenGL because PyOpenGL
    resolves extension entry points by calling ``eglQueryString`` on a display,
    and obtaining the right display is what the device index is needed for.

    Returns:
        Dict[str, int]: ``/dev/dri/renderD<minor>`` to EGL enumeration index.
        Devices without a render node, such as Mesa's software device, are
        left out.

    Raises:
        RuntimeError: If the EGL device extensions are unavailable or the device
            query fails.
        OSError: If ``libEGL.so.1`` cannot be loaded.
    """
    try:
        libegl = ctypes.CDLL("libEGL.so.1")
    except OSError as exc:
        raise OSError(f"libEGL.so.1 is not loadable: {exc}") from exc
    libegl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    libegl.eglGetProcAddress.restype = ctypes.c_void_p

    device_t = ctypes.c_void_p
    query_devices_ptr = libegl.eglGetProcAddress(b"eglQueryDevicesEXT")
    query_string_ptr = libegl.eglGetProcAddress(b"eglQueryDeviceStringEXT")
    if not query_devices_ptr or not query_string_ptr:
        raise RuntimeError(
            "EGL_EXT_device_enumeration and EGL_EXT_device_query are required"
        )
    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(device_t),
        ctypes.POINTER(ctypes.c_int),
    )(query_devices_ptr)
    query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, device_t, ctypes.c_int)(
        query_string_ptr
    )

    devices = (device_t * _MAX_EGL_DEVICES)()
    device_count = ctypes.c_int()
    if not query_devices(_MAX_EGL_DEVICES, devices, ctypes.byref(device_count)):
        raise RuntimeError("eglQueryDevicesEXT failed")

    egl_index_by_render_node = {}
    for egl_index in range(device_count.value):
        render_node = query_string(devices[egl_index], _EGL_DRM_RENDER_NODE_FILE_EXT)
        if render_node:
            egl_index_by_render_node[render_node.decode()] = egl_index
    return egl_index_by_render_node


@cache
def _egl_index_by_hip_device() -> dict[int, int]:
    """Map each HIP device id on this node to the EGL index of its GPU.

    Cached: neither enumeration can change under a running process.

    Returns:
        Dict[int, int]: HIP device id to EGL enumeration index, for RDNA GPUs
        whose render node EGL lists. Empty when EGL cannot be queried, e.g. on a
        node without Mesa.
    """
    render_node_by_hip_device = {
        hip_id: render_node
        for hip_id, (_, render_node) in _rdna_gpus_by_hip_device().items()
    }
    if not render_node_by_hip_device:
        return {}
    try:
        egl_index_by_render_node = _query_egl_index_by_render_node()
    except (OSError, RuntimeError) as exc:
        logger.debug("Cannot map HIP devices to EGL devices (%s).", exc)
        return {}
    return {
        hip_id: egl_index_by_render_node[render_node]
        for hip_id, render_node in render_node_by_hip_device.items()
        if render_node in egl_index_by_render_node
    }


def _renders_with_egl() -> bool:
    """Report whether a renderer could go through EGL in this run."""
    return os.environ.get("MUJOCO_GL") not in _CPU_RENDERING_BACKENDS


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
    def get_worker_local_env_var(visible_accelerators: list[int]) -> dict[str, str]:
        """Get the environment variables a worker resolves on its own node.

        Simulators render on the worker's first GPU: Vulkan through
        ``DRI_PRIME``, EGL through the device index. Both are resolved from the
        KFD topology and EGL enumeration of the node the worker runs on, which
        can differ from the driver's.

        Args:
            visible_accelerators (List[int]): The worker's HIP device IDs.

        Returns:
            Dict[str, str]: ``DRI_PRIME`` and the EGL device index for the first
            device, or nothing when it has no render device to pin.
        """
        env_vars = {}
        if len(visible_accelerators) > 0:
            dri_prime = AMDGPUManager.get_dri_prime(visible_accelerators[0])
            if dri_prime is not None:
                env_vars[DRI_PRIME_ENV] = dri_prime
            if _renders_with_egl():
                egl_device_id = AMDGPUManager.get_egl_device_id(visible_accelerators[0])
                if egl_device_id is not None:
                    for env_var in EGL_DEVICE_ID_ENV_VARS:
                        env_vars[env_var] = str(egl_device_id)
        return env_vars

    @staticmethod
    def get_egl_device_id(hip_device_id: int | str) -> Optional[int]:
        """Get the EGL enumeration index that renders on a HIP device.

        Args:
            hip_device_id (int | str): HIP device id, as in ``HIP_VISIBLE_DEVICES``.

        Returns:
            Optional[int]: The EGL device index, or ``None`` for a device that is
            not on this node, is addressed by UUID, is not an RDNA GPU, or that
            EGL does not list.
        """
        try:
            hip_device_id = int(hip_device_id)
        except ValueError:
            return None
        return _egl_index_by_hip_device().get(hip_device_id)

    @staticmethod
    def get_dri_prime(hip_device_id: int | str) -> Optional[str]:
        """Get the Mesa ``DRI_PRIME`` value that renders on a HIP device.

        Args:
            hip_device_id (int | str): HIP device id, as in ``HIP_VISIBLE_DEVICES``.

        Returns:
            Optional[str]: ``pci-DDDD_BB_DD_F``, or ``None`` for a device that is
            not on this node, is addressed by UUID, or is not an RDNA GPU.
        """
        try:
            hip_device_id = int(hip_device_id)
        except ValueError:
            return None
        gpu = _rdna_gpus_by_hip_device().get(hip_device_id)
        return gpu[0] if gpu is not None else None

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
