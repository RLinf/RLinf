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

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class ADBHWInfo(HardwareInfo):
    """Hardware information for a phone system."""

    config: "ADBConfig"


@Hardware.register()
class ADBPhone(Hardware):
    """Hardware policy for phone systems."""

    HW_TYPE = "ADB"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["ADBConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the phone resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.
        """
        assert configs is not None, (
            "Phone hardware requires explicit configurations for device ID and ADB path."
        )
        adb_configs: list["ADBConfig"] = []
        for config in configs:
            if isinstance(config, ADBConfig) and config.node_rank == node_rank:
                adb_configs.append(config)

        if adb_configs:
            adb_infos = []
            for config in adb_configs:
                try:
                    version_cmd = [config.adb_path, "version"]
                    result = subprocess.run(
                        version_cmd, capture_output=True, text=True, timeout=10
                    )
                    if result.returncode != 0:
                        raise ConnectionError(
                            f"Failed to get ADB version with command '{version_cmd}': {result.stderr}"
                        )
                except subprocess.CalledProcessError as e:
                    raise ConnectionError(
                        f"Failed to get ADB version with command '{version_cmd}': {e}"
                    )
                except subprocess.TimeoutExpired:
                    raise ConnectionError(
                        f"Timeout when getting ADB version with command '{version_cmd}'."
                    )
                except Exception as e:
                    raise ConnectionError(
                        f"Failed to get ADB version with command '{version_cmd}': {e}"
                    )

                # check if the device is connected
                devices_cmd = [config.adb_path, "devices"]
                result = subprocess.run(
                    devices_cmd, capture_output=True, text=True, timeout=10
                )
                lines = result.stdout.strip().split("\n")
                # Filter out header and empty lines, look for 'device' status
                devices = [
                    line for line in lines[1:] if line.strip() and "\tdevice" in line
                ]
                if not devices:
                    raise ConnectionError(
                        f"No devices found with command '{devices_cmd}'. Please check if your device is connected."
                    )
                device_ids = [device.split("\t")[0] for device in devices]
                if config.device_id not in device_ids:
                    raise ConnectionError(
                        f"Device '{config.device_id}' not found in ADB devices: {device_ids}"
                    )
                adb_infos.append(
                    ADBHWInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )

            return HardwareResource(
                type=cls.HW_TYPE,
                infos=adb_infos,
            )
        return None


@NodeHardwareConfig.register_hardware_config(ADBPhone.HW_TYPE)
@dataclass
class ADBConfig(HardwareConfig):
    """Configuration for a phone system."""

    device_id: str
    """ID of the phone device."""

    adb_path: str = "adb"
    """Path to the ADB executable."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in ADB config must be an integer. But got {type(self.node_rank)}."
        )

        if not shutil.which(self.adb_path):
            raise FileNotFoundError(
                f"ADB executable not found at '{self.adb_path}'. Please check if ADB is installed and the path is correct."
            )
