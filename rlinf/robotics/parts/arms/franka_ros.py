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

import sys
import time
from typing import Any

import numpy as np
import psutil
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.arms.base import Arm, BaseArm
from rlinf.robotics.parts.arms.franka import FrankaRobotState, validated_robot_ip
from rlinf.robotics.parts.base import Action, Features, Observation
from rlinf.utils.logging import get_logger


@Arm.register("franka_ros")
class FrankaROSArm(BaseArm):
    """Franka arm controlled through ROS."""

    @classmethod
    def declare(
        cls,
        address: str,
        *,
        load_gripper: bool = True,
        compliance: Any = None,
        **placement: Any,
    ) -> "FrankaROSArm":
        """Declare a ROS-backed Franka arm.

        The end effector is composed beside the arm and opens itself, so this
        backend takes no end-effector settings. It still needs *load_gripper*,
        which decides whether the ROS stack it launches brings up the Franka
        Hand driver the arm's own topics share a robot with.
        """
        # Its controller owns its gains; a task's request arrives through
        # reconfigure_compliance_params instead.
        del compliance
        return cls(address, load_gripper=load_gripper, **placement)

    def __init__(
        self,
        robot_ip: str,
        ros_pkg: str = "serl_franka_controllers",
        load_gripper: bool = True,
    ) -> None:
        self._logger = get_logger()
        self._robot_ip = validated_robot_ip(robot_ip, type(self).__name__)
        self._ros_pkg = ros_pkg
        self._load_gripper = load_gripper
        self._state = FrankaRobotState()
        self._impedance: psutil.Process | None = None
        self._joint: psutil.Process | None = None

    @property
    def action_features(self) -> Features:
        """Describe the Cartesian pose command."""
        return {"tcp_pose": {}}

    def _open(self) -> Any:
        """Connect ROS channels, controller processes, and the end effector."""
        import geometry_msgs.msg as geom_msg
        import rospy
        from dynamic_reconfigure.client import Client as ReconfClient
        from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
        from serl_franka_controllers.msg import ZeroJacobian

        from rlinf.robotics.parts.transports.ros import ROSController

        self._geom_msg = geom_msg
        self._rospy = rospy
        self._ErrorRecoveryActionGoal = ErrorRecoveryActionGoal
        self._FrankaState = FrankaState
        self._ZeroJacobian = ZeroJacobian
        self._ReconfClient = ReconfClient
        self._ros = ROSController.shared()
        self._init_ros_channels()
        self.start_impedance()
        self._reconf_client = self._ReconfClient(
            "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
        )
        return self._ros

    def reset(self) -> None:
        """Leave task-specific reset positions to the caller."""

    def send_action(self, action: Action) -> Observation:
        """Apply one Cartesian pose target."""
        if set(action) != {"tcp_pose"}:
            raise KeyError("Franka ROS action must contain only 'tcp_pose'.")
        self.move_arm(action["tcp_pose"])
        return action

    def _release(self, device: Any) -> None:
        """Stop impedance control, leaving the shared ROS session standing."""
        self.stop_impedance()
        self._ros = None

    def _init_ros_channels(self) -> None:
        """Initialize ROS channels for arm communication."""
        self._arm_equilibrium_channel = (
            "/cartesian_impedance_controller/equilibrium_pose"
        )
        self._arm_reset_channel = "/franka_control/error_recovery/goal"
        self._arm_jacobian_channel = "/cartesian_impedance_controller/franka_jacobian"
        self._arm_state_channel = "franka_state_controller/franka_states"

        self._ros.create_ros_channel(
            self._arm_equilibrium_channel,
            self._geom_msg.PoseStamped,
            queue_size=10,
        )
        self._ros.create_ros_channel(
            self._arm_reset_channel,
            self._ErrorRecoveryActionGoal,
            queue_size=1,
        )
        self._ros.connect_ros_channel(
            self._arm_jacobian_channel,
            self._ZeroJacobian,
            self._on_arm_jacobian_msg,
        )
        self._ros.connect_ros_channel(
            self._arm_state_channel,
            self._FrankaState,
            self._on_arm_state_msg,
        )

    def _on_arm_jacobian_msg(self, msg: Any) -> None:
        self._state.arm_jacobian = np.array(list(msg.zero_jacobian)).reshape(
            (6, 7), order="F"
        )

    def _on_arm_state_msg(self, msg: Any) -> None:
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3].copy())
        self._state.tcp_pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])

        self._state.arm_joint_velocity = np.array(list(msg.dq)).reshape((7,))
        self._state.arm_joint_position = np.array(list(msg.q)).reshape((7,))
        self._state.tcp_force = np.array(list(msg.K_F_ext_hat_K)[:3])
        self._state.tcp_torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        try:
            self._state.tcp_vel = (
                self._state.arm_jacobian @ self._state.arm_joint_velocity
            )
        except Exception as exc:
            self._state.tcp_vel = np.zeros(6)
            self._logger.warning(
                "Jacobian not set, end-effector velocity temporarily unavailable: %s",
                exc,
            )

    def reconfigure_compliance_params(self, params: dict[str, float]) -> None:
        self._reconf_client.update_configuration(params)
        self._logger.debug(f"Reconfigure compliance parameters: {params}")

    def is_robot_up(self) -> bool:
        """Whether the arm answers. An end effector reports its own readiness."""
        return self._ros.get_input_channel_status(self._arm_state_channel)

    def get_state(self) -> FrankaRobotState:
        """Return the current Franka state. The end effector reports its own."""
        return self._state

    def start_impedance(self) -> None:
        """Start the impedance controller."""
        load_gripper = "true" if self._load_gripper else "false"
        self._impedance = psutil.Popen(
            [
                "roslaunch",
                self._ros_pkg,
                "impedance.launch",
                "robot_ip:=" + self._robot_ip,
                f"load_gripper:={load_gripper}",
            ],
            stdout=sys.stdout,
            stderr=sys.stdout,
        )

        self._wait_robot()
        self._logger.debug(f"Start Impedance controller: {self._impedance.status()}")

    def stop_impedance(self) -> None:
        if self._impedance:
            self._impedance.terminate()
            self._impedance = None
            self._wait_robot()
        self._logger.debug("Stop Impedance controller")

    def clear_errors(self) -> None:
        self._ros.put_channel(self._arm_reset_channel, self._ErrorRecoveryActionGoal())

    def reset_joint(self, positions: list[float]) -> None:
        """Reset the joint positions of the robot arm."""
        self.stop_impedance()
        self.clear_errors()
        self._wait_robot()
        self.clear_errors()

        assert len(positions) == 7, (
            f"Invalid reset position, expected 7 dimensions but got {len(positions)}"
        )

        load_gripper = "true" if self._load_gripper else "false"
        self._rospy.set_param("/target_joint_positions", positions)
        self._joint = psutil.Popen(
            [
                "roslaunch",
                self._ros_pkg,
                "joint.launch",
                "robot_ip:=" + self._robot_ip,
                f"load_gripper:={load_gripper}",
            ],
            stdout=sys.stdout,
        )
        self._wait_robot()
        self._logger.debug("Joint reset begins")
        self.clear_errors()

        self._wait_for_joint(positions)

        self._joint.terminate()
        self._wait_robot()
        self.clear_errors()
        self.start_impedance()

    def move_arm(self, position: np.ndarray) -> None:
        """Move the robot arm to the desired position."""
        assert len(position) == 7, (
            f"Invalid position, expected 7 dimensions but got {len(position)}"
        )
        pose_msg = self._geom_msg.PoseStamped()
        pose_msg.header.frame_id = "0"
        pose_msg.header.stamp = self._rospy.Time.now()
        pose_msg.pose.position = self._geom_msg.Point(
            position[0], position[1], position[2]
        )
        pose_msg.pose.orientation = self._geom_msg.Quaternion(
            position[3], position[4], position[5], position[6]
        )

        self._ros.put_channel(self._arm_equilibrium_channel, pose_msg)
        self._logger.debug(f"Move arm to position: {position}")

    def _wait_robot(self, sleep_time: int = 1) -> None:
        time.sleep(sleep_time)

    def _wait_for_joint(self, target_pos: list[float], timeout: int = 30) -> None:
        wait_time = 0.01
        waited_time = 0.0
        target_pos = np.array(target_pos)

        while (
            not np.allclose(
                target_pos,
                self._state.arm_joint_position,
                atol=1e-2,
                rtol=1e-2,
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
