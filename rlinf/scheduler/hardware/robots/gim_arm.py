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


@dataclass
class GimArmHWInfo(HardwareInfo):
    """Hardware information for a GimArm robotic system."""

    config: "GimArmConfig"


@Hardware.register()
class GimArmRobot(Hardware):
    """Hardware policy for GimArm robots (CAN bus, 6-DOF)."""

    HW_TYPE = "GimArm"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["GimArmConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate GimArm robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware
                resources. None if no GimArm hardware is configured for this
                node.
        """
        assert configs is not None, "GimArm hardware requires explicit configurations."
        robot_configs: list["GimArmConfig"] = []
        for config in configs:
            if isinstance(config, GimArmConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        if robot_configs:
            gim_arm_infos = []
            for config in robot_configs:
                if not config.disable_validate:
                    cls._validate_can_interface(config.can_interface, node_rank)

                gim_arm_infos.append(
                    GimArmHWInfo(
                        type=cls.HW_TYPE,
                        model=f"{cls.HW_TYPE}_{config.arm_variant}",
                        config=config,
                    )
                )

            return HardwareResource(type=cls.HW_TYPE, infos=gim_arm_infos)
        return None

    @staticmethod
    def _validate_can_interface(can_interface: str, node_rank: int) -> None:
        """Warn if the CAN interface is not visible on this node."""
        can_path = f"/sys/class/net/{can_interface}"
        if not os.path.exists(can_path):
            warnings.warn(
                f"CAN interface '{can_interface}' not found at {can_path} on node "
                f"rank {node_rank}. The GimArm controller may fail to start."
            )


@NodeHardwareConfig.register_hardware_config(GimArmRobot.HW_TYPE)
@dataclass
class GimArmConfig(HardwareConfig):
    """Configuration for a GimArm robot."""

    can_interface: str = "can0"
    """CAN socket interface name (e.g. ``"can0"``)."""

    arm_variant: str = "gim_arm_xl"
    """Arm variant: ``"gim_arm"`` or ``"gim_arm_xl"``."""

    camera_serials: Optional[list[str]] = None
    """Optional list of camera serial numbers.
    Pass ``[]`` or leave ``None`` to run without cameras.
    Camera auto-detection is not currently implemented for GimArm."""

    camera_type: str = "realsense"
    """Camera backend: ``"realsense"`` or ``"zed"``."""

    enable_gripper: bool = True
    """Whether the gripper is attached and should be controlled."""

    gripper_type: str = "parallel"
    """Gripper type: ``"parallel"`` or ``"single_side"``."""

    controller_node_rank: Optional[int] = None
    """Node rank where :class:`GimArmController` should run.
    When ``None`` (default), co-located with the env worker."""

    disable_validate: bool = False
    """Whether to skip CAN interface validation during enumeration."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in GimArm config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        assert self.arm_variant in ("gim_arm", "gim_arm_xl"), (
            f"'arm_variant' must be 'gim_arm' or 'gim_arm_xl'. "
            f"But got '{self.arm_variant}'."
        )
        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)
