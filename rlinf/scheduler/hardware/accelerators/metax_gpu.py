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

"""MetaX GPU accelerator manager for RLinf."""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .accelerator import AcceleratorManager, AcceleratorType, ProfileConfig

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions


@AcceleratorManager.register_profiling_config(AcceleratorType.METAX_GPU)
@dataclass
class METAXGPUProfileConfig(ProfileConfig):
    """MetaX GPU profiling configuration."""


@AcceleratorManager.register_manager(AcceleratorType.METAX_GPU)
class METAXGPUManager(AcceleratorManager):
    """Utility Class for MetaX GPU."""

    @staticmethod
    def _get_pymxsml():
        """Import and initialize pymxsml on a best-effort basis.

        Do not run this at import time: some nodes may not have MetaX
        drivers installed.

        Returns:
            Optional[module]: The initialized ``pymxsml`` module, or None
            when the MetaX SML library/driver is not present on this node.
        """
        try:
            import pymxsml

            pymxsml.mxSmlInit()
            return pymxsml
        except Exception:
            # If MetaX library/driver isn't present on this node, leave it to
            # callers to handle by falling back to mx-smi / returning 0
            # devices / UNKNOWN model.
            return None

    @staticmethod
    def _run_mxsmi(args: list[str]) -> Optional[str]:
        """Run the mx-smi tool and return its standard output.

        Args:
            args (list[str]): Command-line arguments passed to mx-smi.

        Returns:
            Optional[str]: The standard output of mx-smi, or None when the
            tool is unavailable or fails to run (e.g. on nodes without
            MetaX GPUs installed).
        """
        try:
            result = subprocess.run(
                ["mx-smi", *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout

    @staticmethod
    def _query_mxsmi_json() -> Optional[dict]:
        """Query mx-smi in JSON mode and parse its output.

        Returns:
            Optional[dict]: The parsed JSON payload of ``mx-smi -j``, or None
            when mx-smi is unavailable or its output cannot be parsed.
        """
        output = METAXGPUManager._run_mxsmi(["-j"])
        if output is None:
            return None
        try:
            json_start = output.index("{")
            return json.loads(output[json_start:])
        except ValueError:
            return None

    @staticmethod
    def get_num_devices() -> int:
        """Get the number of MetaX GPU devices on the node.

        Returns:
            int: The number of available MetaX GPU devices. Returns 0 when
            neither pymxsml nor mx-smi can enumerate the devices, e.g. on
            nodes without MetaX GPUs.
        """
        pymxsml = METAXGPUManager._get_pymxsml()
        if pymxsml is not None:
            try:
                return int(pymxsml.mxSmlGetDeviceCount())
            except Exception:
                pass
        # pymxsml is the authoritative source here, but it is an optional
        # wheel shipped under /opt/maca/share/mxsml and may not be installed
        # in the current environment; fall back to parsing mx-smi.
        data = METAXGPUManager._query_mxsmi_json()
        if data is None:
            return 0
        num_devices = 0
        for device_info in data.values():
            if not isinstance(device_info, dict) or "device_id" not in device_info:
                continue
            # Skip devices that are present but unavailable to users.
            if not device_info.get("unavailable_reason"):
                num_devices += 1
        return num_devices

    @staticmethod
    def get_accelerator_type() -> AcceleratorType:
        """Get the type of the accelerator.

        Returns:
            AcceleratorType: The type of the accelerator.
        """
        return AcceleratorType.METAX_GPU

    @staticmethod
    def get_accelerator_model() -> str:
        """Get the model of the MetaX GPU, e.g. ``MetaX C500``.

        Note: pymxsml exposes ``mxSml*`` APIs (not NVML). Do not use
        ``nvml*`` names here; they do not exist and would break hardware
        enumeration.

        Returns:
            str: The model of the MetaX GPU, or ``"UNKNOWN"`` when the model
            cannot be determined.
        """
        pymxsml = METAXGPUManager._get_pymxsml()
        if pymxsml is not None:
            try:
                info = pymxsml.mxSmlGetDeviceInfo(0)
                # `c_mxsmlDeviceInfo_t` decodes bytes -> str via __getattribute__.
                model = getattr(info, "deviceName", None)
                if model:
                    return str(model).strip()
            except Exception:
                pass
        # Fall back to parsing the mx-smi table, e.g.
        # "| 0     MetaX C500 | ...". The leading device index and the table
        # borders are used to avoid matching the banner line.
        output = METAXGPUManager._run_mxsmi([])
        if output is not None:
            match = re.search(r"\|\s*\d+\s+(MetaX\s+\S+)\s*\|", output)
            if match:
                return match.group(1)
        return "UNKNOWN"

    @staticmethod
    def get_accelerator_env_var(visible_accelerators: list[str]) -> dict[str, str]:
        """Get the environment variables related to the accelerator.

        Args:
            visible_accelerators (list[str]): A list of visible accelerator IDs.

        Returns:
            dict[str, str]: A dictionary containing the accelerator environment variables.
        """
        env_vars = {}
        visible_accelerators_str = ",".join(visible_accelerators)

        # MetaX reads MACA_VISIBLE_DEVICES (MC_VISIBLE_DEVICES is ignored).
        env_vars["MACA_VISIBLE_DEVICES"] = visible_accelerators_str
        # Ray has no MetaX support, but MetaX torch is CUDA-API-compatible,
        # so Ray would otherwise rewrite CUDA_VISIBLE_DEVICES (to empty for
        # 0-GPU actors) and hide the devices.
        env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        return env_vars

    @staticmethod
    def get_visible_devices() -> list[int]:
        """Get the visible device IDs.

        Returns:
            list[int]: A list of visible device IDs.
        """
        visible_devices = os.environ.get("MACA_VISIBLE_DEVICES", None)

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
    def get_ccl_backend() -> str:
        """Get the CCL backend.

        Returns:
            str: The CCL backend.
        """
        return "mccl"

    @staticmethod
    def get_ccl_socket_ifname_env_var() -> str:
        """Get the network socket interface name environment variable.

        Returns:
            str: The network socket interface name environment variable.
        """
        return "MCCL_SOCKET_IFNAME"

    @staticmethod
    def get_torch_platform():
        """Get the PyTorch platform module.

        Returns:
            torch.cuda: The PyTorch CUDA-compatible platform module. MetaX's
            torch build is CUDA-API-compatible, so the standard ``torch.cuda``
            interface drives MACA devices directly.
        """
        import torch

        return torch.cuda

    @staticmethod
    def get_device_type() -> str:
        """Get the device type.

        Returns:
            str: The device type.
        """
        # MetaX torch is CUDA-API-compatible: tensors and modules use the
        # standard "cuda" device type.
        return "cuda"

    @staticmethod
    def get_accel_pg_options(options: Optional["CollectiveGroupOptions"]):
        """Get the accelerator CCL process group options.

        Args:
            options (Optional[CollectiveGroupOptions]): The options for the collective group.

        Returns:
            Optional[dist.ProcessGroup.Options]: The accelerator CCL process group options.
        """
        # MCCL process-group tuning options (CTA splitting etc.) depend on
        # the torch_macax build exposing a ProcessGroupMCCL-like Options
        # class; keep the defaults until that is verified on-device.
        return None
