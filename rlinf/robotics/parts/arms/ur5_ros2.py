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

"""Universal Robots UR5 arm, driven through the ROS2 driver via ``rclpy``.

Registered as ``ur_ros2``, beside the ``ur_rtde`` backend, so the same arm can be
driven over the Universal Robots ROS2 driver (the stack used on the real robot)
instead of the RTDE SDK. It connects to an **already running**
``Universal_Robots_ROS2_Driver`` over the ROS graph -- it does not launch the
driver or the simulator. State comes from ``/joint_states``; motion goes to the
``scaled_joint_trajectory_controller`` as ``FollowJointTrajectory`` goals. In the
driver's headless mode the external-control program is (re)started through the
``resend_robot_program`` service before commanding motion.

Works in radians, like the arm contract. Carries no gripper, matching
:class:`~rlinf.robotics.parts.arms.ur5.UR5Arm`.
"""

import threading
import time
from typing import Any, ClassVar, Optional, Sequence

import numpy as np

from rlinf.robotics.parts.base import Action, Features, Observation
from rlinf.robotics.parts.claims import DeviceClaim
from rlinf.utils.logging import get_logger

from .base import Arm, BaseArm
from .ur5 import UR5RobotState

#: Canonical UR joint order used by the controller and reported in joint_states.
_UR_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@Arm.register("ur_ros2")
class UR5ROS2Arm(BaseArm):
    """Universal Robots UR5 over the ROS2 driver, via ``rclpy``.

    Args:
        address: Controller IP of the arm the ROS2 driver is attached to.
            Informational only -- this backend reaches the arm through the ROS
            graph, not by connecting to the IP itself.
        speed: Unused placeholder kept for a uniform arm signature; motion is
            paced by the trajectory ``time_from_start``.
        acceleration: Unused placeholder, as above.
        step_goal_time: ``time_from_start`` of a per-step trajectory goal, in
            seconds. Each ``send_action`` fires one short goal that preempts the
            previous, so the env's fixed-rate loop is not blocked.
    """

    SDK = "rclpy"

    DOF: ClassVar[int] = 6

    #: Conservative bench travel, matching UR5Arm; a task may narrow it further.
    JOINT_LIMITS_LOWER: ClassVar[np.ndarray] = np.full(6, -np.pi)
    JOINT_LIMITS_UPPER: ClassVar[np.ndarray] = np.full(6, np.pi)

    #: The driver reports joints and (optionally) tool pose; no force sensing.
    STATE_FIELDS = ("arm_joint_position", "tcp_pose")

    #: Controller, topics, and service of the Universal Robots ROS2 driver.
    TRAJECTORY_ACTION: ClassVar[str] = (
        "/scaled_joint_trajectory_controller/follow_joint_trajectory"
    )
    JOINT_STATES_TOPIC: ClassVar[str] = "/joint_states"
    RESEND_PROGRAM_SERVICE: ClassVar[str] = (
        "/io_and_status_controller/resend_robot_program"
    )

    def __init__(
        self,
        address: str,
        *,
        speed: float = 1.05,
        acceleration: float = 1.4,
        step_goal_time: float = 0.2,
    ) -> None:
        self._logger = get_logger()
        self._ip = address
        self._speed = float(speed)
        self._acceleration = float(acceleration)
        self._step_goal_time = float(step_goal_time)
        self._joint_names = _UR_JOINT_NAMES

        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: Optional[threading.Thread] = None
        self._action: Any = None
        self._resend: Any = None
        self._goal_cls: Any = None

        self._latest_lock = threading.Lock()
        self._latest: Optional[dict[str, float]] = None
        self._tcp_pose = np.zeros(7)
        self._rclpy_inited_here = False
        # One node commands one controller; a second driver would contend for it.
        self._claim = DeviceClaim(f"ur5_ros2:{address}", type(self).__name__)

    @classmethod
    def declare(
        cls,
        address: str,
        *,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        **placement: Any,
    ) -> "UR5ROS2Arm":
        """Declare a UR5 reached through the ROS2 driver.

        Carries no gripper, so the end-effector options are refused rather than
        silently dropped, matching :meth:`UR5Arm.declare`.
        """
        cls.refuse_unused(
            gripper_type=gripper_type,
            gripper_connection=gripper_connection,
            end_effector_type=end_effector_type,
            end_effector_config=end_effector_config,
        )
        settings = {
            key: placement.pop(key)
            for key in ("speed", "acceleration", "step_goal_time")
            if key in placement
        }
        return cls(address, **settings, **placement)

    @property
    def action_features(self) -> Features:
        """Describe the absolute joint target, in radians."""
        return {"joint_position": {}}

    # Lifecycle.

    def _open(self) -> Any:
        """Create the ROS2 node and connect to the running UR driver."""
        self._claim.acquire()
        try:
            import rclpy
            from control_msgs.action import FollowJointTrajectory
            from rclpy.action import ActionClient
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
            from std_srvs.srv import Trigger

            if not rclpy.ok():
                rclpy.init()
                self._rclpy_inited_here = True

            self._goal_cls = FollowJointTrajectory
            self._node = Node("rlinf_ur5_ros2")
            self._node.create_subscription(
                JointState, self.JOINT_STATES_TOPIC, self._on_joint_states, 10
            )
            self._action = ActionClient(
                self._node, FollowJointTrajectory, self.TRAJECTORY_ACTION
            )
            self._resend = self._node.create_client(
                Trigger, self.RESEND_PROGRAM_SERVICE
            )

            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin, daemon=True
            )
            self._spin_thread.start()

            # Headless external control stops when idle; restart it before use.
            self._call_resend_program()
            if not self._action.wait_for_server(timeout_sec=15.0):
                raise RuntimeError(
                    f"Trajectory action {self.TRAJECTORY_ACTION} is not available. "
                    "Start the UR ROS2 driver (e.g. ~/ur_stack_up.sh) first."
                )
            self._wait_for_state(timeout=10.0)
        except BaseException:
            self._release(None)
            raise
        self._logger.info("UR5 (ros2) connected via %s", self.TRAJECTORY_ACTION)
        return self._node

    def _release(self, device: Any) -> None:
        """Tear the node down; also unwinds a failed open, so fields may be None."""
        try:
            if self._executor is not None:
                self._executor.shutdown()
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=2.0)
            if self._node is not None:
                self._node.destroy_node()
        finally:
            self._node = None
            self._executor = None
            self._spin_thread = None
            self._action = None
            self._resend = None
            try:
                import rclpy

                if self._rclpy_inited_here and rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
            self._rclpy_inited_here = False
            self._claim.release()

    # State.

    def _on_joint_states(self, msg: Any) -> None:
        """Cache the latest joint reading, keyed by joint name."""
        reading = dict(zip(msg.name, msg.position))
        with self._latest_lock:
            self._latest = reading

    def _wait_for_state(self, timeout: float) -> None:
        """Block until a joint_states message with all arm joints has arrived."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._latest_lock:
                ready = self._latest is not None and all(
                    name in self._latest for name in self._joint_names
                )
            if ready:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"No {self.JOINT_STATES_TOPIC} received from the UR driver within "
            f"{timeout:.0f}s."
        )

    def get_state(self) -> UR5RobotState:
        """Read joints from the latest joint_states into canonical order."""
        with self._latest_lock:
            latest = self._latest
        if latest is None:
            joints = np.zeros(self.DOF)
        else:
            joints = np.array(
                [latest.get(name, 0.0) for name in self._joint_names], dtype=float
            )
        return UR5RobotState(arm_joint_position=joints, tcp_pose=self._tcp_pose.copy())

    # Commanding.

    def send_action(self, action: Action) -> Observation:
        """Move the arm joints to an absolute target in radians."""
        if set(action) != {"joint_position"}:
            raise KeyError(
                "A UR5 arm action holds only 'joint_position'; this driver "
                "carries no gripper."
            )
        sent = self.move_joints(action["joint_position"])
        return {"joint_position": sent}

    def move_joints(self, q_target: "Sequence[float]") -> np.ndarray:
        """Command absolute joint positions in radians (non-blocking).

        Fires one trajectory goal that preempts the previous one, so a
        fixed-rate control loop is not blocked waiting for each move to finish.
        The goal duration is scaled by the joint distance and :pyattr:`_speed`
        (floored at ``step_goal_time``), so a large jump gets a feasible time
        instead of an impossible one -- small per-step deltas still execute at
        the floor, giving servo-like motion.

        Returns:
            The target actually sent, in radians.
        """
        target = self._clip_target(q_target)
        current = np.asarray(self.get_state().arm_joint_position, dtype=float)
        distance = float(np.max(np.abs(target - current)))
        goal_time = max(distance / max(self._speed, 1e-3), self._step_goal_time)
        self._send_goal(target, goal_time, wait=False)
        return target

    def reset_joint(
        self,
        positions: "Sequence[float]",
        duration: float = 3.0,
        *,
        max_velocity: float = np.deg2rad(30.0),
    ) -> None:
        """Move to a rest pose in radians, returning once the arm has stopped.

        Sends a single blocking trajectory goal paced by ``duration``, then
        waits for the joints to settle. ``max_velocity`` is accepted for a
        uniform reset signature; the driver paces the move by ``duration``.
        """
        target = self._clip_target(positions)
        # A reset may follow an idle period; make sure control is live first.
        self._call_resend_program()
        self._send_goal(target, max(float(duration), 1.0), wait=True)
        self.wait_until_still(duration)

    def is_robot_up(self) -> bool:
        """Report whether the node is open and joint state is flowing."""
        if self._node is None:
            return False
        with self._latest_lock:
            return self._latest is not None

    # Helpers.

    def _clip_target(self, q_target: "Sequence[float]") -> np.ndarray:
        """Validate shape and clip to the configured travel."""
        target = np.asarray(q_target, dtype=float).reshape(-1)
        if target.shape != (self.DOF,):
            raise ValueError(
                f"Expected {self.DOF} joint targets for a UR5, got shape "
                f"{target.shape}."
            )
        return np.clip(target, self.JOINT_LIMITS_LOWER, self.JOINT_LIMITS_UPPER)

    def _send_goal(
        self, target: np.ndarray, time_from_start: float, wait: bool
    ) -> None:
        """Send a single-point FollowJointTrajectory goal to the controller."""
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = self._goal_cls.Goal()
        goal.trajectory.joint_names = list(self._joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in target]
        sec = int(time_from_start)
        point.time_from_start = Duration(
            sec=sec, nanosec=int((time_from_start - sec) * 1e9)
        )
        goal.trajectory.points = [point]

        send_future = self._action.send_goal_async(goal)
        if not wait:
            return
        if not self._spin_until(send_future, timeout=5.0):
            self._logger.warning("UR trajectory goal send timed out.")
            return
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._logger.warning("UR trajectory goal was rejected by the controller.")
            return
        result_future = goal_handle.get_result_async()
        self._spin_until(result_future, timeout=time_from_start + 10.0)

    def _call_resend_program(self) -> None:
        """Restart the driver's external-control program (headless mode)."""
        from std_srvs.srv import Trigger

        if not self._resend.wait_for_service(timeout_sec=5.0):
            self._logger.warning(
                "%s not available; assuming external control is already running.",
                self.RESEND_PROGRAM_SERVICE,
            )
            return
        future = self._resend.call_async(Trigger.Request())
        self._spin_until(future, timeout=5.0)

    @staticmethod
    def _spin_until(future: Any, timeout: float) -> bool:
        """Wait for a future the background executor is driving to complete."""
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        return future.done()
