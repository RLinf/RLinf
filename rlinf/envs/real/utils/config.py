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

"""Hardware configuration accepted by real-world environments."""

from typing import TypeVar

from rlinf.robotics.discovery import RobotConfig, RobotInfo

ConfigT = TypeVar("ConfigT", bound=RobotConfig)


def get_hardware_config(
    config_cls: type[ConfigT],
    robot_info: RobotInfo[ConfigT] | None,
    *,
    is_dummy: bool,
) -> ConfigT:
    """Read the hardware config without modifying the scheduler's descriptor.

    Dummy environments may omit the descriptor only when hardware defaults
    define a valid observation layout. Franka requires a descriptor with cameras
    even in dummy mode. Real hardware requires enumerated info.
    """
    robot_config = None
    if robot_info is not None:
        if not isinstance(robot_info, RobotInfo):
            raise TypeError("robot_info must be a RobotInfo from hardware enumeration.")
        robot_config = robot_info.config
    if robot_config is None:
        if not is_dummy:
            raise ValueError(
                f"Supply robot_info containing {config_cls.__name__} from hardware "
                "enumeration or the scheduler."
            )
        robot_config = config_cls(node_rank=0)
    if not isinstance(robot_config, config_cls):
        raise TypeError(
            f"Expected {config_cls.__name__}, got {type(robot_config).__name__}."
        )
    return robot_config
