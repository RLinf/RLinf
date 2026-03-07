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

"""Franka built-in two-finger gripper end-effector.

This implementation delegates to the ``ROSController`` that is already
set up by ``FrankaController``.  The ROS channel names are passed in at
construction so that ``FrankaGripper`` does **not** create its own
subscribers / publishers.
"""

import time

import numpy as np

from rlinf.envs.realworld.common.ros import ROSController
from rlinf.utils.logging import get_logger

from .base import EndEffector


class FrankaGripper(EndEffector):
    """Franka Emika Panda parallel-jaw gripper controlled via ROS topics.

    Args:
        ros_controller: A ``ROSController`` instance that is already
            initialised inside the ``FrankaController`` worker.
        binary_threshold: Threshold on the action value for binary
            open / close decisions.
        gripper_sleep: Time to wait after a gripper open / close command
            so that the physical motion finishes.
    """

    _NUM_DOFS = 1

    def __init__(
        self,
        ros_controller: ROSController,
        binary_threshold: float = 0.5,
        gripper_sleep: float = 0.6,
    ):
        self._ros = ros_controller
        self._binary_threshold = binary_threshold
        self._gripper_sleep = gripper_sleep
        self._logger = get_logger()

        # State
        self._position: float = 0.0
        self._is_open: bool = False

        # ROS channel names (initialised by ``_setup_channels``)
        self._move_channel = "/franka_gripper/move/goal"
        self._grasp_channel = "/franka_gripper/grasp/goal"
        self._state_channel = "/franka_gripper/joint_states"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._NUM_DOFS

    @property
    def state_dim(self) -> int:
        return self._NUM_DOFS

    @property
    def control_mode(self) -> str:
        return "binary"

    @property
    def is_open(self) -> bool:
        """Whether the gripper is currently open."""
        return self._is_open

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register ROS channels for the gripper."""
        from franka_gripper.msg import GraspActionGoal, MoveActionGoal
        from sensor_msgs.msg import JointState

        self._ros.create_ros_channel(
            self._move_channel, MoveActionGoal, queue_size=1
        )
        self._ros.create_ros_channel(
            self._grasp_channel, GraspActionGoal, queue_size=1
        )
        self._ros.connect_ros_channel(
            self._state_channel, JointState, self._on_state_msg
        )
        self._logger.debug("FrankaGripper ROS channels initialised.")

    def shutdown(self) -> None:
        """No special teardown required for the Franka gripper."""

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _on_state_msg(self, msg) -> None:
        """ROS callback for ``/franka_gripper/joint_states``."""
        self._position = float(np.sum(msg.position))

    def get_state(self) -> np.ndarray:
        return np.array([self._position], dtype=np.float64)

    def is_channel_active(self) -> bool:
        """Return ``True`` once the gripper state subscriber has received data."""
        return self._ros.get_input_channel_status(self._state_channel)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def open_gripper(self) -> None:
        """Open the gripper fully."""
        from franka_gripper.msg import MoveActionGoal

        msg = MoveActionGoal()
        msg.goal.width = 0.09
        msg.goal.speed = 0.3
        self._ros.put_channel(self._move_channel, msg)
        self._is_open = True
        self._logger.debug("FrankaGripper: open")

    def close_gripper(self) -> None:
        """Close the gripper with a firm grasp."""
        from franka_gripper.msg import GraspActionGoal

        msg = GraspActionGoal()
        msg.goal.width = 0.01
        msg.goal.speed = 0.3
        msg.goal.epsilon.inner = 1
        msg.goal.epsilon.outer = 1
        msg.goal.force = 130
        self._ros.put_channel(self._grasp_channel, msg)
        self._is_open = False
        self._logger.debug("FrankaGripper: close")

    def command(self, action: np.ndarray) -> bool:
        """Binary gripper control.

        Args:
            action: A 1-D array with a single element. Values
                ``<= -threshold`` trigger a close; ``>= threshold`` trigger
                an open.

        Returns:
            ``True`` if the gripper state changed.
        """
        value = float(action[0])
        if value <= -self._binary_threshold and self._is_open:
            self.close_gripper()
            time.sleep(self._gripper_sleep)
            return True
        elif value >= self._binary_threshold and not self._is_open:
            self.open_gripper()
            time.sleep(self._gripper_sleep)
            return True
        return False

    def reset(self, target_state: np.ndarray | None = None) -> None:
        """Reset gripper to open state."""
        self.open_gripper()
        time.sleep(self._gripper_sleep)
