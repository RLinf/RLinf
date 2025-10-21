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

import subprocess
import time
from dataclasses import dataclass
from typing import List

import geometry_msgs.msg as geom_msg
import numpy as np
import psutil
import rospy
from franka_gripper.msg import GraspActionGoal, MoveActionGoal
from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from serl_franka_controllers.msg import ZeroJacobian

from rlinf.envs.physical.common import ROSController
from rlinf.scheduler import Worker
from rlinf.utils.logging import get_logger


@dataclass
class FrankaRobotState:
    # https://docs.ros.org/en/kinetic/api/libfranka/html/structfranka_1_1RobotState.html
    arm_position: np.ndarray = np.zeros(7)  # FrankaState.O_T_EE
    arm_velocity: np.ndarray = np.zeros(6)
    arm_joint_position: np.ndarray = np.zeros(7)  # FrankaState.q
    arm_joint_velocity: np.ndarray = np.zeros(7)  # FrankaState.dq
    arm_force: np.ndarray = np.zeros(3)  # FrankaState.K_F_ext_hat_K[0:3]
    arm_torque: np.ndarray = np.zeros(3)  # FrankaState.K_F_ext_hat_K[3:6]
    arm_jacobian: np.ndarray = np.zeros((6, 7))  # ZeroJacobian.zero_jacobian

    gripper_position: int = 0  # Sum(JointState.position)
    gripper_open: bool = False


class FrankaController(Worker):
    """Franka robot arm controller."""

    def __init__(self, robot_ip: str, ros_pkg: str = "serl_franka_controllers"):
        """Initialize the Franka robot arm controller.

        Args:
            robot_ip (str): The IP address of the robot arm.
        """
        super().__init__()
        self._logger = get_logger()
        self._robot_ip = robot_ip
        self._ros_pkg = ros_pkg

        self._ros = ROSController()
        self._init_ros_channels()

        # roslaunch processes
        self._impedance: psutil.Process = None
        self._joint: psutil.Process = None

        # Franka state
        self._state = FrankaRobotState()

    def _init_ros_channels(self):
        """Initialize ROS channels for communication."""

        # ARM control channels
        self._arm_equilibrium_channel = (
            "/cartesian_impedance_controller/equilibrium_pose"
        )
        self._arm_reset_channel = "/franka_control/error_recovery/goal"
        self._arm_jacobian_channel = "/cartesian_impedance_controller/franka_jacobian"
        self._arm_state_channel = "franka_state_controller/franka_states"

        self._ros.create_ros_channel(
            self._arm_equilibrium_channel, geom_msg.PoseStamped, queue_size=10
        )
        self._ros.create_ros_channel(
            self._arm_reset_channel, ErrorRecoveryActionGoal, queue_size=1
        )
        self._ros.connect_ros_channel(
            self._arm_jacobian_channel, ZeroJacobian, self._on_arm_jacobian_msg
        )
        self._ros.connect_ros_channel(
            self._arm_state_channel, FrankaState, self._on_arm_state_msg
        )

        # Gripper control channels
        self._gripper_move_channel = "/franka_gripper/move/goal"
        self._gripper_grasp_channel = "/franka_gripper/grasp/goal"
        self._gripper_state_channel = "/franka_gripper/grasp/state"
        self._ros.create_ros_channel(
            self._gripper_move_channel, MoveActionGoal, queue_size=1
        )
        self._ros.create_ros_channel(
            self._gripper_grasp_channel, GraspActionGoal, queue_size=1
        )
        self._ros.connect_ros_channel(
            self._gripper_state_channel, JointState, self._on_gripper_state_msg
        )

    def _on_arm_jacobian_msg(self, msg: ZeroJacobian):
        """Callback for Jacobian messages.

        Args:
            msg (ZeroJacobian): The Jacobian message.
        """
        self._state.arm_jacobian = np.array(list(msg.zero_jacobian)).reshape(
            (6, 7), order="F"
        )

    def _on_arm_state_msg(self, msg: FrankaState):
        """Callback for Franka state messages.

        Args:
            msg (FrankaState): The Franka state message.
        """
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3])
        self._state.arm_position = np.concatenate([tmatrix[:3, -1], r.as_quat()])

        self._state.arm_joint_velocity = np.array(list(msg.dq)).reshape((7,))
        self._state.arm_joint_position = np.array(list(msg.q)).reshape((7,))
        self._state.arm_force = np.array(list(msg.K_F_ext_hat_K)[:3])
        self._state.arm_torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        try:
            self._state.arm_velocity = self._arm_jacobian @ self._arm_joint_velocity
        except Exception as e:
            self._state.arm_velocity = np.zeros(6)
            self._logger.warning(
                f"Jacobian not set, end-effector velocity temporarily not available with error {e}"
            )

    def _on_gripper_state_msg(self, msg: JointState):
        """Callback for gripper state messages.

        Args:
            msg (JointState): The gripper state message.
        """
        self._state.gripper_position = np.sum(msg.position)

    def _wait_robot(self, sleep_time: int = 1):
        """Wait for the robot to reach the desired state.

        Args:
            sleep_time (int): The time to wait in seconds.
        """
        time.sleep(sleep_time)

    def _wait_for_joint(self, target_pos: List[float], timeout: int = 30):
        """Wait for the robot joint to reach the desired position.

        Args:
            target_pos (List[float]): The target joint position.
            timeout (int): The maximum time to wait in seconds.
        """
        wait_time = 0.01
        waited_time = 0
        target_pos = np.array(target_pos)

        while (
            not np.allclose(
                target_pos, self._state.arm_joint_position, atol=1e-2, rtol=1e-2
            )
            and waited_time < timeout
        ):
            time.sleep(wait_time)
            waited_time += wait_time

        if waited_time >= timeout:
            self._logger.warning("Joint position wait timeout exceeded")
        else:
            self._logger.debug(
                f"Joint position reached {self._state.arm_joint_position}"
            )

    def get_state(self) -> FrankaRobotState:
        """Get the current state of the Franka robot.

        Returns:
            FrankaRobotState: The current state of the Franka robot.
        """
        return self._state

    def start_impedance(self):
        """Start the impedance controller."""
        self._impedance = psutil.Popen(
            [
                "roslaunch",
                self._ros_pkg,
                "impedance.launch",
                "robot_ip:=" + self._robot_ip,
                "load_gripper:='true'",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._wait_robot()

    def stop_impedance(self):
        """Stop the impedance controller."""
        if self._impedance:
            self._impedance.terminate()
            self._impedance = None
            self._wait_robot()

    def clear_errors(self):
        self._ros.put_channel(self._arm_reset_channel, ErrorRecoveryActionGoal())

    def reset_joint(self, reset_pos: List[float]):
        """
        Reset the joint positions of the robot arm.

        Args:
            reset_pos (List[float]): The desired joint positions. Must be a list of 7 floats, meaning [x, y, z, qx, qy, qz, qw]
        """
        # Stop impedance before reset
        self.stop_impedance()
        self.clear_errors()

        self._wait_robot()
        self.clear_errors()

        assert len(reset_pos) == 7, (
            f"Invalid reset position, expected 7 dimensions but got {len(reset_pos)}"
        )
        self._joint = psutil.Popen(
            [
                "roslaunch",
                self._ros_pkg,
                "joint.launch",
                "robot_ip:=" + self._robot_ip,
                "load_gripper:='true'",
            ],
            stdout=subprocess.PIPE,
        )
        self._wait_robot()
        self._logger.debug("Joint reset begins")
        self.clear_errors()

        self._wait_for_joint(reset_pos)

        self._joint.terminate()
        self._wait_robot()
        self.clear_errors()

        # Start impedance
        self.start_impedance()

    def move_arm(self, position: np.ndarray):
        """
        Move the robot arm to the desired position.

        Args:
            position (np.ndarray): The desired position. Must be a 1D array of 7 floats, meaning [x, y, z, qx, qy, qz, qw]
        """
        assert len(position) == 7, (
            f"Invalid position, expected 7 dimensions but got {len(position)}"
        )
        pose_msg = geom_msg.PoseStamped()
        pose_msg.header.frame_id = "0"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position = geom_msg.Point(position[0], position[1], position[2])
        pose_msg.pose.orientation = geom_msg.Quaternion(
            position[3], position[4], position[5], position[6]
        )

        self._ros.put_channel(self._arm_equilibrium_channel, pose_msg)

    def move_gripper(self, position: int, speed: float = 0.3):
        """
        Move the gripper to the desired position.

        Args:
            position (int): The desired position. Must be an integer between 0 and 255.
        """
        assert 0 <= position <= 255, (
            f"Invalid gripper position {position}, must be between 0 and 255"
        )
        move_msg = MoveActionGoal()
        move_msg.goal.width = float(position / (255 * 10))  # width in [0, 0.1]m
        move_msg.goal.speed = speed

        self._ros.put_channel(self._gripper_move_channel, move_msg)

    def open_gripper(self):
        """Open the gripper."""
        move_msg = MoveActionGoal()
        move_msg.goal.width = 0.09
        move_msg.goal.speed = 0.3

        self._ros.put_channel(self._gripper_move_channel, move_msg)

    def close_gripper(self):
        """Close the gripper."""
        grasp_msg = GraspActionGoal()
        grasp_msg.goal.width = 0.01
        grasp_msg.goal.speed = 0.3
        grasp_msg.goal.epsilon.inner = 1
        grasp_msg.goal.epsilon.outer = 1
        grasp_msg.goal.force = 130

        self._ros.put_channel(self._gripper_grasp_channel, grasp_msg)
