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

import os
import warnings
from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)
from .auto_config import RobotAutoConfig


@dataclass
class SO101HWInfo(HardwareInfo):
    """Hardware information for an SO101 robotic arm."""

    config: "SO101Config"


@Hardware.register()
class SO101Robot(Hardware):
    """Hardware policy for SO101 arms (Feetech bus, 6-DOF).

    The SO101 is an open-source, 3D-printable 6-DOF manipulator using
    Feetech STS3215 motors. It is connected via USB serial and controlled
    through LeRobot's :class:`lerobot.robots.so_follower.SO101Follower`.
    """

    HW_TYPE = "SO101"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["SO101Config"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate SO101 robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware
                resources. None if no SO101 hardware is configured for this
                node.
        """
        assert configs is not None, "SO101 hardware requires explicit configurations."
        robot_configs: list["SO101Config"] = []
        for config in configs:
            if isinstance(config, SO101Config) and config.node_rank == node_rank:
                robot_configs.append(config)

        # Fill unset fields from env vars (e.g. ``SO101_PORT``), one value per
        # config when several robots share this node. With no configs given,
        # create one per comma-separated ``SO101_PORT``.
        robot_configs = RobotAutoConfig.resolve(
            robot_configs,
            config_cls=SO101Config,
            node_rank=node_rank,
            count_fields=("port",),
        )

        if robot_configs:
            so101_infos = []
            for config in robot_configs:
                if not config.disable_validate:
                    cls._validate_port(config.port, node_rank)

                so101_infos.append(
                    SO101HWInfo(
                        type=cls.HW_TYPE,
                        model=f"{cls.HW_TYPE}_{config.arm_variant}",
                        config=config,
                    )
                )

            return HardwareResource(type=cls.HW_TYPE, infos=so101_infos)
        return None

    @staticmethod
    def _validate_port(port: str, node_rank: int) -> None:
        """Warn if the serial port is not visible on this node."""
        if not os.path.exists(port):
            warnings.warn(
                f"Serial port '{port}' not found on node rank {node_rank}. "
                f"The SO101 controller may fail to start. "
                f"Check your USB connection and try 'ls /dev/tty.*'."
            )


@NodeHardwareConfig.register_hardware_config(SO101Robot.HW_TYPE)
@dataclass
class SO101Config(HardwareConfig):
    """Configuration for an SO101 robot arm.

    Most fields can be filled from environment variables by
    :class:`RobotAutoConfig` (e.g. ``SO101_PORT`` → ``port``).
    """

    port: str = "/dev/tty.usbmodem*"
    """Serial port for the Feetech motor bus (e.g. ``"/dev/ttyACM0"`` on Linux,
    ``"/dev/tty.usbmodem*"`` on macOS)."""

    leader_port: Optional[str] = None
    """Serial port for the leader (teleoperation) arm.
    Only needed for data collection with bilateral teleoperation."""

    arm_variant: str = "so101"
    """Arm variant: ``"so101"`` or ``"so100"``."""

    calibration_id: str = "default"
    """Calibration ID for the LeRobot calibration file.
    Stored at ``~/.cache/lerobot/calibration/robots/so_follower/{id}.json``."""

    camera_serials: Optional[list[str]] = None
    """Optional list of camera serial numbers or indices.
    Pass ``[]`` or leave ``None`` to run without cameras."""

    camera_type: str = "opencv"
    """Camera backend: ``"opencv"`` for USB cameras, ``"realsense"`` for Intel
    RealSense, or ``"zed"`` for ZED cameras."""

    use_degrees: bool = True
    """Whether joint angles use degrees (LeRobot default) or radians."""

    disable_validate: bool = False
    """Whether to skip serial port validation during enumeration."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in SO101 config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        assert self.arm_variant in ("so101", "so100"), (
            f"'arm_variant' must be 'so101' or 'so100'. "
            f"But got '{self.arm_variant}'."
        )
        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)
