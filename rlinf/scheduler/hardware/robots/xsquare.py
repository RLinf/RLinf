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
class Turtle2HWInfo(HardwareInfo):
    """Hardware information for a robotic system."""

    config: "Turtle2Config"


@Hardware.register()
class Turtle2Robot(Hardware):
    """Hardware policy for robotic systems."""

    HW_TYPE = "Turtle2"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["Turtle2Config"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware resources. None if no hardware is found.
        """
        assert configs is not None, "Robot hardware requires explicit configurations"
        robot_configs: list["Turtle2Config"] = []
        for config in configs:
            if isinstance(config, Turtle2Config) and config.node_rank == node_rank:
                robot_configs.append(config)

        if robot_configs:
            turtle2_infos = []

            for config in robot_configs:
                turtle2_infos.append(
                    Turtle2HWInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )

            return HardwareResource(type=cls.HW_TYPE, infos=turtle2_infos)
        return None


@NodeHardwareConfig.register_hardware_config(Turtle2Robot.HW_TYPE)
@dataclass
class Turtle2Config(HardwareConfig):
    """Configuration for a robotic system."""

    # empty config

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in Turtle2 config must be an integer. But got {type(self.node_rank)}."
        )
