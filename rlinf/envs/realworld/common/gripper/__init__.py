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

from typing import Optional

from .base_gripper import BaseGripper

__all__ = [
    "BaseGripper",
    "create_gripper",
]


def create_gripper(
    gripper_type: str = "franka",
    ros=None,
    port: Optional[str] = None,
    robot_ip: Optional[str] = None,
    **kwargs,
) -> BaseGripper:
    """Factory that instantiates the right gripper backend.

    Args:
        gripper_type: ``"franka"`` or ``"robotiq"``.
        ros: :class:`ROSController` instance — used for the legacy ROS
            Franka Hand path when ``robot_ip`` is not provided.
        port: Serial device path (e.g. ``"/dev/ttyUSB0"``) — required for
            ``"robotiq"``.
        robot_ip: Arm/gripper IP — selects the libfranka ``franky.Gripper``
            backend for ``"franka"`` (preferred; no ROS).
        **kwargs: Forwarded to the gripper constructor (e.g. ``max_width``,
            ``baudrate``, ``slave_id``, ``grasp_force``).
    """
    gt = gripper_type.lower()
    if gt == "robotiq":
        if port is None:
            raise ValueError(
                "gripper_connection (serial port) must be specified "
                "for Robotiq grippers."
            )
        from .robotiq_gripper import RobotiqGripper

        return RobotiqGripper(port=port, **kwargs)
    if gt == "franka":
        if robot_ip is not None:
            from .franky_franka_gripper import FrankyFrankaGripper

            return FrankyFrankaGripper(robot_ip=robot_ip, **kwargs)
        if ros is None:
            raise ValueError(
                "Franka gripper requires either robot_ip (franky/libfranka) "
                "or a ROSController instance (legacy ROS path)."
            )
        from .franka_gripper import FrankaGripper

        return FrankaGripper(ros=ros, **kwargs)
    raise ValueError(
        f"Unsupported gripper_type={gripper_type!r}. "
        f"Supported types: 'franka', 'robotiq'."
    )
