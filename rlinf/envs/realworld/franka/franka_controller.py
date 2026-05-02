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

import sys
import time
from typing import Optional

import numpy as np
import psutil
from scipy.spatial.transform import Rotation as R

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .end_effectors import (
    EndEffector,
    EndEffectorType,
    create_end_effector,
    normalize_end_effector_type,
)
from .franka_robot_state import FrankaRobotState


class FrankaController(Worker):
    """Franka robot arm controller."""

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        ros_pkg: str = "serl_franka_controllers",
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ):
        """Launch a FrankaController on the specified worker's node."""
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return FrankaController.create_group(
            robot_ip,
            ros_pkg,
            end_effector_type,
            end_effector_config or {},
            gripper_type,
            gripper_connection,
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"FrankaController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        ros_pkg: str = "serl_franka_controllers",
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ):
        super().__init__()
        self._logger = get_logger()
        self._robot_ip = robot_ip
        self._ros_pkg = ros_pkg
        self._end_effector_type = normalize_end_effector_type(
            end_effector_type,
            gripper_type,
        )

        # Lazy-import ROS packages so the module can be imported on non-ROS nodes.
        import geometry_msgs.msg as geom_msg
        import rospy
        from dynamic_reconfigure.client import Client as ReconfClient
        from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
        from serl_franka_controllers.msg import ZeroJacobian

        self._geom_msg = geom_msg
        self._rospy = rospy
        self._ErrorRecoveryActionGoal = ErrorRecoveryActionGoal
        self._FrankaState = FrankaState
        self._ZeroJacobian = ZeroJacobian
        self._ReconfClient = ReconfClient

        self._state = FrankaRobotState()
        self._end_effector: EndEffector | None = None
        self._gripper = None

        from rlinf.envs.realworld.common.ros import ROSController

        self._ros = ROSController()
        self._init_ros_channels()
        self._init_end_effector(end_effector_config or {}, gripper_connection)

        self._impedance: psutil.Process | None = None
        self._joint: psutil.Process | None = None

        self.start_impedance()
        self._reconf_client = self._ReconfClient(
            "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
        )

    def _init_end_effector(
        self,
        end_effector_config: dict,
        gripper_connection: Optional[str],
    ) -> None:
        if self._end_effector_type.is_gripper:
            from rlinf.envs.realworld.common.gripper import create_gripper

            self._gripper = create_gripper(
                gripper_type=self._end_effector_type.gripper_backend,
                ros=self._ros,
                port=gripper_connection,
                **end_effector_config,
            )
            self._logger.info(
                "Gripper initialised: end_effector=%s",
                self._end_effector_type.value,
            )
            return

        self._end_effector = create_end_effector(
            self._end_effector_type,
            **end_effector_config,
        )
        self._end_effector.initialize()
        self._logger.info(
            "End-effector initialised: %s",
            self._end_effector_type.value,
        )

    def _init_ros_channels(self):
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

    def _on_arm_jacobian_msg(self, msg):
        self._state.arm_jacobian = np.array(list(msg.zero_jacobian)).reshape(
            (6, 7), order="F"
        )

    def _on_arm_state_msg(self, msg):
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

    def reconfigure_compliance_params(self, params: dict[str, float]):
        self._reconf_client.update_configuration(params)
        self.log_debug(f"Reconfigure compliance parameters: {params}")

    def is_robot_up(self) -> bool:
        """Check whether the arm and active end-effector are ready."""
        arm_ok = self._ros.get_input_channel_status(self._arm_state_channel)
        if self._end_effector_type.is_gripper:
            return arm_ok and self._gripper.is_ready()
        return arm_ok

    def get_state(self) -> FrankaRobotState:
        """Get the current state of the Franka robot."""
        if self._end_effector_type.is_gripper:
            self._state.gripper_position = self._gripper.position
            self._state.gripper_open = self._gripper.is_open
            self._state.hand_position = None
        else:
            assert self._end_effector is not None
            self._state.hand_position = self._end_effector.get_state()
        return self._state

    def start_impedance(self):
        """Start the impedance controller."""
        load_gripper = (
            "true"
            if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER
            else "false"
        )
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
        self.log_debug(f"Start Impedance controller: {self._impedance.status()}")

    def stop_impedance(self):
        if self._impedance:
            self._impedance.terminate()
            self._impedance = None
            self._wait_robot()
        self.log_debug("Stop Impedance controller")

    def clear_errors(self):
        self._ros.put_channel(self._arm_reset_channel, self._ErrorRecoveryActionGoal())

    def reset_joint(self, reset_pos: list[float]):
        """Reset the joint positions of the robot arm."""
        self.stop_impedance()
        self.clear_errors()
        self._wait_robot()
        self.clear_errors()

        assert len(reset_pos) == 7, (
            f"Invalid reset position, expected 7 dimensions but got {len(reset_pos)}"
        )

        load_gripper = (
            "true"
            if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER
            else "false"
        )
        self._rospy.set_param("/target_joint_positions", reset_pos)
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

        self._wait_for_joint(reset_pos)

        self._joint.terminate()
        self._wait_robot()
        self.clear_errors()
        self.start_impedance()

    def move_arm(self, position: np.ndarray):
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
        self.log_debug(f"Move arm to position: {position}")

    def command_end_effector(self, action: np.ndarray) -> bool:
        """Send an action to the active end-effector."""
        if self._end_effector_type.is_gripper:
            value = float(np.asarray(action).reshape(-1)[0])
            if value <= -0.5 and self._state.gripper_open:
                self.close_gripper()
                return True
            if value >= 0.5 and not self._state.gripper_open:
                self.open_gripper()
                return True
            return False

        assert self._end_effector is not None
        return self._end_effector.command(action)

    def reset_end_effector(self, target_state: np.ndarray | None = None) -> None:
        """Reset the end-effector to a target or default state."""
        if self._end_effector_type.is_gripper:
            if target_state is not None:
                self.command_end_effector(np.asarray(target_state))
            return

        assert self._end_effector is not None
        self._end_effector.reset(target_state)

    def open_gripper(self):
        if self._end_effector_type.is_gripper:
            self._gripper.open()
            self._state.gripper_open = True
        self.log_debug("Open gripper")

    def close_gripper(self):
        if self._end_effector_type.is_gripper:
            self._gripper.close()
            self._state.gripper_open = False
        self.log_debug("Close gripper")

    def move_gripper(self, position: int, speed: float = 0.3):
        assert 0 <= position <= 255, (
            f"Invalid gripper position {position}, must be between 0 and 255"
        )
        if self._end_effector_type.is_gripper:
            self._gripper.move(position, speed)
        self.log_debug(f"Move gripper to position: {position}")

    def _wait_robot(self, sleep_time: int = 1):
        time.sleep(sleep_time)

    def _wait_for_joint(self, target_pos: list[float], timeout: int = 30):
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

    def get_hand_type(self) -> str:
        return self._end_effector_type.value

    def get_hand_state(self) -> np.ndarray | None:
        if self._end_effector_type.is_gripper:
            return None
        assert self._end_effector is not None
        return self._end_effector.get_state()

    def get_hand_detailed_state(self) -> dict:
        if self._end_effector_type.is_gripper:
            return {
                "gripper_position": self._gripper.position,
                "gripper_open": self._gripper.is_open,
            }
        assert self._end_effector is not None
        return self._end_effector.get_detailed_state()

    def get_hand_finger_names(self) -> list[str]:
        if self._end_effector_type.is_gripper:
            return ["gripper"]
        assert self._end_effector is not None
        return self._end_effector.finger_names
