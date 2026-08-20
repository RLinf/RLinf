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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from rlinf.scheduler.hardware import HardwareConfig, HardwareInfo, HardwareResource

from ..discovery import (
    RobotAutoConfig,
    RobotConfig,
    RobotDiscovery,
    RobotInfo,
)
from ..parts.cameras import declare_cameras
from ..robot import Robot


class GimArmRobot(Robot):
    """Composable GimArm robot."""

    ROBOT_TYPE = "GimArm"

    @classmethod
    def build_arms(
        cls,
        *,
        can_interface: str,
        arm_variant: str,
        enable_gripper: bool,
        gripper_type: str,
        control_mode: str,
        node_rank: int,
        worker_rank: int = 0,
        env_idx: int = 0,
    ) -> dict[str, Any]:
        """The one arm this robot carries, gripper included when fitted."""
        from ..parts.arms.gim_arm import GimArm

        connection = GimArm.at(
            can_interface,
            arm_variant,
            enable_gripper,
            gripper_type,
            control_mode,
            node_rank=node_rank,
            name=f"GimArm-{worker_rank}-{env_idx}",
        )
        parts = {"arm": connection.export("arm")}
        if enable_gripper:
            parts["end_effector"] = connection.export("end_effector")
        return parts

    @classmethod
    def build_cameras(
        cls,
        cameras: Optional[Mapping[str, Any]] = None,
        *,
        node_rank: Optional[int] = None,
    ) -> dict[str, Any]:
        """The cameras this robot carries."""
        return declare_cameras(cameras, node_rank=node_rank)

    @classmethod
    def build(
        cls,
        *,
        can_interface: str,
        arm_variant: str,
        enable_gripper: bool,
        gripper_type: str,
        control_mode: str,
        env_idx: int,
        node_rank: int,
        worker_rank: int,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
    ) -> "GimArmRobot":
        """Compose this robot from the parts it is made of."""
        return cls(
            **cls.build_arms(
                can_interface=can_interface,
                arm_variant=arm_variant,
                enable_gripper=enable_gripper,
                gripper_type=gripper_type,
                control_mode=control_mode,
                node_rank=node_rank,
                worker_rank=worker_rank,
                env_idx=env_idx,
            ),
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
        )


class GimArmDiscovery(RobotDiscovery):
    """Hardware policy for GimArm robots (CAN bus, 6-DOF)."""

    HW_TYPE = GimArmRobot.ROBOT_TYPE

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
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

        # Fill unset fields from env vars (e.g. ``CAN_INTERFACE``), one value per
        # config when several robots share this node. With no configs given,
        # create one per comma-separated ``CAN_INTERFACE``.
        robot_configs = RobotAutoConfig.resolve(
            robot_configs,
            config_cls=GimArmConfig,
            node_rank=node_rank,
            count_fields=("can_interface",),
        )

        if robot_configs:
            gim_arm_infos: list[HardwareInfo] = []
            for config in robot_configs:
                if not config.disable_validate:
                    cls._validate_can_interface(config.can_interface, node_rank)

                gim_arm_infos.append(
                    RobotInfo(
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


@dataclass
class GimArmConfig(RobotConfig):
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
    """Node rank where the arm part should run.
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


GimArmRobot.register(GimArmConfig, GimArmDiscovery)
