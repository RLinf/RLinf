# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

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
class DOSW1HWInfo(HardwareInfo):
    """Hardware information for a DOS-W1 dual-arm robot."""

    config: "DOSW1HWConfig"


@Hardware.register()
class DOSW1Robot(Hardware):
    """Hardware policy for the DOS-W1 dual-arm robot.

    Connection parameters (gRPC URL / ports) and RealSense camera serials
    are placed here so that they are managed by the scheduler via
    ``cluster.node_groups.hardware`` and injected into the env worker
    through :class:`DOSW1HWInfo`.
    """

    HW_TYPE = "DOSW1"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["DOSW1HWConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the DOS-W1 robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Hardware resource descriptor, or ``None`` when the node has no
            matching DOS-W1 configuration.
        """
        assert configs is not None, (
            "DOSW1 hardware requires explicit configurations for robot URL, "
            "gRPC ports and camera serials."
        )
        robot_configs: list["DOSW1HWConfig"] = []
        for config in configs:
            if isinstance(config, DOSW1HWConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        if not robot_configs:
            return None

        infos: list[DOSW1HWInfo] = [
            DOSW1HWInfo(type=cls.HW_TYPE, model=cls.HW_TYPE, config=cfg)
            for cfg in robot_configs
        ]
        return HardwareResource(type=cls.HW_TYPE, infos=infos)


@NodeHardwareConfig.register_hardware_config(DOSW1Robot.HW_TYPE)
@dataclass
class DOSW1HWConfig(HardwareConfig):
    """Configuration for a DOS-W1 dual-arm robot.

    The env process runs on the node indicated by :attr:`node_rank`, and
    talks to the AirBot gRPC services over ``robot_url`` and the four
    follower / leader ports.
    """

    robot_url: str = "localhost"
    """Hostname or IP of the AirBot gRPC endpoint."""

    left_arm_port: int = 50051
    """gRPC port of the left follower arm."""

    right_arm_port: int = 50053
    """gRPC port of the right follower arm."""

    left_lead_port: int = 50050
    """gRPC port of the left leader arm."""

    right_lead_port: int = 50052
    """gRPC port of the right leader arm."""

    camera_serials: Optional[list[str]] = None
    """RealSense camera serial numbers used by the env."""

    def __post_init__(self):  # noqa: D105
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in DOSW1 config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.camera_serials is not None:
            self.camera_serials = [str(s) for s in self.camera_serials]
