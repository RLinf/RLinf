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

import geometry_msgs.msg as geom_msg
import numpy as np
import psutil
import rospy
from dynamic_reconfigure.client import Client as ReconfClient
from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from scipy.spatial.transform import Rotation as R
from serl_franka_controllers.msg import ZeroJacobian

from rlinf.envs.realworld.common.ros import ROSController
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .end_effectors import EndEffector, EndEffectorType, create_end_effector
from .end_effectors.franka_gripper import FrankaGripper
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
        node_group_label: str = None,
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
    ):
        """Launch a FrankaController on the specified worker's node.

        Args:
            robot_ip: The IP address of the robot arm.
            env_idx: The index of the environment.
            node_rank: The rank of the node to launch the controller on.
            worker_rank: The rank of the env worker.
            ros_pkg: The ROS package name for the Franka controllers.
            node_group_label: The label of the node group for env_configs.
            end_effector_type: One of ``"franka_gripper"``,
                ``"aoyi_hand"``, or ``"ruiyan_hand"``.
            end_effector_config: Extra keyword arguments forwarded to the
                end-effector constructor.

        Returns:
            FrankaController: The launched FrankaController instance.
        """
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank], node_group_label=node_group_label)
        return FrankaController.create_group(
            robot_ip, ros_pkg, end_effector_type, end_effector_config or {},
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
    ):
        """Initialize the Franka robot arm controller.

        Args:
            robot_ip: The IP address of the robot arm.
            ros_pkg: The ROS package name for the Franka controllers.
            end_effector_type: Type of end-effector to use.
            end_effector_config: Additional kwargs for the end-effector.
        """
        super().__init__()
        self._logger = get_logger()
        self._robot_ip = robot_ip
        self._ros_pkg = ros_pkg
        self._end_effector_type = EndEffectorType(end_effector_type)

        # Franka state
        self._state = FrankaRobotState()

        # ROS controller
        self._ros = ROSController()
        self._init_ros_channels()

        # roslaunch processes
        self._impedance: psutil.Process = None
        self._joint: psutil.Process = None

        # Start impedance control
        self.start_impedance()

        # Start reconfigure client
        self._reconf_client = ReconfClient(
            "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
        )

        # -- End-effector initialisation ----------------------------------
        ee_kwargs = dict(end_effector_config or {})
        if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER:
            ee_kwargs["ros_controller"] = self._ros
        self._end_effector: EndEffector = create_end_effector(
            self._end_effector_type, **ee_kwargs
        )
        self._end_effector.initialize()
        self._logger.info(
            f"End-effector initialised: {self._end_effector_type.value}"
        )

    def _init_ros_channels(self):
        """Initialize ROS channels for arm communication.

        Gripper-specific channels are set up by
        :class:`~.end_effectors.franka_gripper.FrankaGripper` during its
        own ``initialize()`` call.
        """

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
        """
        In exp, this func is about 30 Hz
        """
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
        except Exception as e:
            self._state.tcp_vel = np.zeros(6)
            self._logger.warning(
                f"Jacobian not set, end-effector velocity temporarily not available with error {e}"
            )

    def _wait_robot(self, sleep_time: int = 1):
        """Wait for the robot to reach the desired state.

        Args:
            sleep_time (int): The time to wait in seconds.
        """
        time.sleep(sleep_time)

    def _wait_for_joint(self, target_pos: list[float], timeout: int = 30):
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

    def reconfigure_compliance_params(self, params: dict[str, float]):
        """Reconfigure the compliance parameters.

        Args:
            params (dict[str, float]): The parameters to reconfigure.
        """
        self._reconf_client.update_configuration(params)
        self.log_debug(f"Reconfigure compliance parameters: {params}")

    @property
    def end_effector(self) -> EndEffector:
        """The active end-effector instance."""
        return self._end_effector

    def is_robot_up(self) -> bool:
        """Check if all ROS channels are connected.

        Returns:
            bool: True if all ROS channels are connected, False otherwise.
        """
        arm_state_status = self._ros.get_input_channel_status(self._arm_state_channel)
        if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER:
            assert isinstance(self._end_effector, FrankaGripper)
            return arm_state_status and self._end_effector.is_channel_active()
        return arm_state_status

    def get_state(self) -> FrankaRobotState:
        """Get the current state of the Franka robot.

        The end-effector state is refreshed before returning.

        Returns:
            FrankaRobotState: The current state of the Franka robot.
        """
        ee_state = self._end_effector.get_state()
        if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER:
            self._state.gripper_position = float(ee_state[0])
        else:
            self._state.hand_position = ee_state
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
        """Stop the impedance controller."""
        if self._impedance:
            self._impedance.terminate()
            self._impedance = None
            self._wait_robot()
        self.log_debug("Stop Impedance controller")

    def clear_errors(self):
        self._ros.put_channel(self._arm_reset_channel, ErrorRecoveryActionGoal())

    def reset_joint(self, reset_pos: list[float]):
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

        # Launch joint controller reset
        load_gripper = (
            "true"
            if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER
            else "false"
        )
        rospy.set_param("/target_joint_positions", reset_pos)
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
        self.log_debug(f"Move arm to position: {position}")

    # ------------------------------------------------------------------
    # End-effector convenience methods
    # ------------------------------------------------------------------

    def command_end_effector(self, action: np.ndarray) -> bool:
        """Send an action to the active end-effector.

        Args:
            action: Action vector whose shape matches the end-effector's
                ``action_dim``.

        Returns:
            ``True`` if the command caused a state change.
        """
        result = self._end_effector.command(action)
        # Sync state for Franka gripper
        if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER:
            assert isinstance(self._end_effector, FrankaGripper)
            self._state.gripper_open = self._end_effector.is_open
        return result

    def reset_end_effector(self, target_state: np.ndarray | None = None) -> None:
        """Reset the end-effector to a target or default state."""
        self._end_effector.reset(target_state)
        if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER:
            assert isinstance(self._end_effector, FrankaGripper)
            self._state.gripper_open = self._end_effector.is_open

    def open_gripper(self):
        """Open the gripper (convenience, works only with FrankaGripper)."""
        if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER:
            assert isinstance(self._end_effector, FrankaGripper)
            self._end_effector.open_gripper()
            self._state.gripper_open = True
        self.log_debug("Open gripper")

    def close_gripper(self):
        """Close the gripper (convenience, works only with FrankaGripper)."""
        if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER:
            assert isinstance(self._end_effector, FrankaGripper)
            self._end_effector.close_gripper()
            self._state.gripper_open = False
        self.log_debug("Close gripper")

    # ------------------------------------------------------------------
    # Dexterous hand diagnostic helpers
    # ------------------------------------------------------------------

    def get_hand_type(self) -> str:
        """Return the active end-effector type string."""
        return self._end_effector_type.value

    def get_hand_state(self) -> np.ndarray | None:
        """Return normalised hand positions (6-D), or ``None`` for grippers."""
        if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER:
            return None
        return self._end_effector.get_state()

    def get_hand_detailed_state(self) -> dict:
        """Return per-motor detailed diagnostics from the end-effector.

        For dexterous hands this includes position, velocity, current,
        and error status for every motor.  For the Franka gripper it
        returns basic gripper info.
        """
        return self._end_effector.get_detailed_state()

    def get_hand_finger_names(self) -> list[str]:
        """Return human-readable names for each DOF."""
        return self._end_effector.finger_names
