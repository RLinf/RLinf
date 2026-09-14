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

"""Franka parallel-jaw gripper controlled through ROS topics."""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..base import EndEffector
from .base import BaseGripper

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlinf.robotics.parts.transports.ros import ROSController


@EndEffector.register("franka", "franka_gripper")
class FrankaGripper(BaseGripper):
    """Franka Emika parallel-jaw gripper (ROS-based).

    Communication uses three ROS topics:

    * ``/franka_gripper/move/goal`` for width commands
    * ``/franka_gripper/grasp/goal`` for force-controlled grasps
    * ``/franka_gripper/joint_states`` for finger-joint feedback

    The hand reaches ROS through this process's shared session, so it opens
    and closes on its own and can be composed beside any arm, whether or not
    that arm speaks ROS.

    Args:
        max_width: Stroke of the hand in metres. The default is the Franka
            Hand's own; ``open()`` travels to it, and it bounds the axis
            :meth:`move` and :pyattr:`position` share.
    """

    @classmethod
    def declare(
        cls,
        *,
        ros: Optional["ROSController"] = None,
        port: Optional[str] = None,
        robot_ip: Optional[str] = None,
        **settings: Any,
    ) -> "FrankaGripper":
        """Declare a hand that joins this process's ROS session."""
        return cls(**settings)

    def __init__(self, max_width: float = 0.08) -> None:
        self._ros: Optional["ROSController"] = None
        self._max_width = max_width
        self._GraspActionGoal = None
        self._MoveActionGoal = None

        self._position_value: float = 0.0
        self._is_open_flag: bool = True

        # ROS channels.
        self._move_channel = "/franka_gripper/move/goal"
        self._grasp_channel = "/franka_gripper/grasp/goal"
        self._state_channel = "/franka_gripper/joint_states"

    def _open(self) -> Any:
        """Join the ROS session and create the gripper's own channels."""
        from franka_gripper.msg import GraspActionGoal, MoveActionGoal
        from sensor_msgs.msg import JointState

        from rlinf.robotics.parts.transports.ros import ROSController

        self._GraspActionGoal = GraspActionGoal
        self._MoveActionGoal = MoveActionGoal
        # Topics are the hand's own, so joining a session an arm already opened
        # adds subscriptions rather than competing for anything.
        self._ros = ROSController.shared()

        self._ros.create_ros_channel(self._move_channel, MoveActionGoal, queue_size=1)
        self._ros.create_ros_channel(self._grasp_channel, GraspActionGoal, queue_size=1)
        self._ros.connect_ros_channel(
            self._state_channel, JointState, self._on_state_msg
        )
        return self._ros

    def _release(self, device: Any) -> None:
        """Let go of the session, which other parts keep using."""
        self._ros = None

    # BaseGripper interface

    def open(self, speed: float = 0.3) -> None:
        msg = self._MoveActionGoal()
        msg.goal.width = self._max_width
        msg.goal.speed = speed
        self._ros.put_channel(self._move_channel, msg)
        self._is_open_flag = True

    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        msg = self._GraspActionGoal()
        msg.goal.width = 0.01
        msg.goal.speed = speed
        msg.goal.epsilon.inner = 1
        msg.goal.epsilon.outer = 1
        msg.goal.force = force
        self._ros.put_channel(self._grasp_channel, msg)
        self._is_open_flag = False

    def move(self, width: float, speed: float = 0.3) -> None:
        """Move to an opening width in metres."""
        msg = self._MoveActionGoal()
        msg.goal.width = float(np.clip(width, 0.0, self._max_width))
        msg.goal.speed = speed
        self._ros.put_channel(self._move_channel, msg)

    @property
    def position(self) -> float:
        """Current opening width in metres: both finger joints together."""
        return self._position_value

    @property
    def max_width(self) -> float:
        """Stroke of the Franka Hand."""
        return self._max_width

    @property
    def is_open(self) -> bool:
        return self._is_open_flag

    def is_ready(self) -> bool:
        if self._ros is None:
            return False
        return self._ros.get_input_channel_status(self._state_channel)

    # ROS callback

    def _on_state_msg(self, msg: Any) -> None:
        self._position_value = np.sum(msg.position)
