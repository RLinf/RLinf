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

import os
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
        headless_mode: Whether the UR ROS2 driver was launched with
            ``headless_mode:=true``. When ``True`` (default), connecting and
            resetting restart its external-control program through
            ``resend_robot_program``, raising if that fails. Outside headless
            mode the driver's own ``sendRobotProgram()`` always refuses, so
            set this to ``False`` once an operator has started External
            Control manually, or the resend call would report a spurious
            failure.
    """

    SDK = "rclpy"

    DOF: ClassVar[int] = 6

    #: Conservative bench travel, matching UR5Arm; a task may narrow it further.
    JOINT_LIMITS_LOWER: ClassVar[np.ndarray] = np.full(6, -np.pi)
    JOINT_LIMITS_UPPER: ClassVar[np.ndarray] = np.full(6, np.pi)

    #: No tool-pose feedback yet: nothing here publishes TCP pose, so only
    #: joints are advertised. See UR5Arm (``ur_rtde``) for a backend that
    #: reports ``tcp_pose``.
    STATE_FIELDS = ("arm_joint_position",)

    #: A joint_states reading older than this is treated as lost connectivity.
    STATE_STALE_AFTER: ClassVar[float] = 1.0

    #: Controller, topics, and service of the Universal Robots ROS2 driver.
    TRAJECTORY_ACTION: ClassVar[str] = (
        "/scaled_joint_trajectory_controller/follow_joint_trajectory"
    )
    JOINT_STATES_TOPIC: ClassVar[str] = "/joint_states"
    RESEND_PROGRAM_SERVICE: ClassVar[str] = (
        "/io_and_status_controller/resend_robot_program"
    )

    #: How long teardown waits to confirm a goal was sent or canceled before
    #: giving up and logging the risk. Generous since this runs once, off
    #: the control loop, and the claims are released either way afterward.
    TEARDOWN_CONFIRM_TIMEOUT: ClassVar[float] = 10.0

    #: How long a blocking goal send waits for the controller's accept/reject
    #: decision before giving up on it. Also the deadline a non-blocking step
    #: send gets before it counts as overdue in _check_pending_step_sends.
    GOAL_ACK_TIMEOUT: ClassVar[float] = 5.0

    #: How many step-goal sends may be awaiting an accept/reject decision at
    #: once before a further send faults instead of adding to the pile-up --
    #: otherwise an unresponsive controller would never surface a fault,
    #: since /joint_states can keep publishing independently of it.
    MAX_PENDING_STEP_SENDS: ClassVar[int] = 4

    def __init__(
        self,
        address: str,
        *,
        speed: float = 1.05,
        acceleration: float = 1.4,
        step_goal_time: float = 0.2,
        headless_mode: bool = True,
    ) -> None:
        self._logger = get_logger()
        self._ip = address
        self._speed = float(speed)
        self._acceleration = float(acceleration)
        self._step_goal_time = float(step_goal_time)
        self._headless_mode = bool(headless_mode)
        self._joint_names = _UR_JOINT_NAMES

        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: Optional[threading.Thread] = None
        self._action: Any = None
        self._resend: Any = None
        self._goal_cls: Any = None

        self._latest_lock = threading.Lock()
        self._latest: Optional[dict[str, float]] = None
        self._latest_stamp: Optional[float] = None
        self._rclpy_inited_here = False

        self._goal_lock = threading.Lock()
        self._goal_generation = 0
        #: Highest generation confirmed accepted, tracked separately from
        #: _goal_generation (which bumps on every send) so a stale goal's
        #: failure is suppressed only once a later goal is actually known to
        #: have preempted it, not merely because a later one was sent.
        self._last_accepted_generation = 0
        self._active_goal_handle: Any = None
        #: Send futures not yet resolved into an accept/reject, so a goal
        #: accepted just after teardown starts can still be canceled.
        self._pending_sends: set[Any] = set()
        #: Deadline (monotonic) by which each entry in _pending_sends must
        #: resolve, so a step goal the controller never acknowledges (e.g.
        #: the action server has disappeared) is detected even though
        #: /joint_states keeps publishing and state freshness keeps passing.
        self._pending_send_deadlines: dict[Any, float] = {}
        #: Handles accepted after being superseded, whose cancellation is
        #: being confirmed through done-callbacks on the executor thread, so
        #: a concurrent teardown can still find and confirm them if those
        #: callbacks never run (e.g. the executor is shut down first). Keyed
        #: by ``id(handle)`` rather than the handle itself: a real rclpy
        #: ``ClientGoalHandle`` defines ``__eq__`` without ``__hash__`` and so
        #: is unhashable, which would make it unusable as a set element.
        self._pending_cancellations: dict[int, Any] = {}
        #: Generation and message of the last recorded failure, so a late
        #: accept confirmation can still discard a stale cancellation error.
        self._goal_error: Optional[tuple[int, str]] = None
        #: Set under _goal_lock once teardown starts, so a send racing with
        #: _cancel_pending_goal either lands in _pending_sends before the
        #: cancel sweep or is rejected outright -- never neither.
        self._shutting_down = False

        # Same key as UR5Arm: one physical controller, however it is reached,
        # so the two backends cannot both drive it at once.
        self._claim = DeviceClaim(f"ur5:{address}", type(self).__name__)
        # A ROS graph is scoped by its domain, and this backend reaches the
        # arm through one action endpoint on that graph. Two instances with
        # different (and merely informational) `address` values but the same
        # domain and action name would otherwise both believe they have
        # exclusive access while driving the same controller.
        ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
        self._ros_claim = DeviceClaim(
            f"ur5-ros2-domain:{ros_domain_id}:{self.TRAJECTORY_ACTION}",
            type(self).__name__,
        )

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
            for key in ("speed", "acceleration", "step_goal_time", "headless_mode")
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
        self._shutting_down = False
        self._claim.acquire()
        try:
            self._ros_claim.acquire()
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
        """Tear the node down; also unwinds a failed open, so fields may be None.

        Each step is nested in its own ``finally``, so a goal cancellation
        that raises does not skip the executor shutdown, thread join, or node
        destruction that follow it -- with a borrowed rclpy context, nothing
        else would ever stop that executor or destroy that node, even though
        the claims below are released as if teardown had fully succeeded.
        """
        try:
            # Cancel a step's in-flight trajectory before the executor that
            # would carry the cancel stops spinning.
            self._cancel_pending_goal()
        finally:
            try:
                if self._executor is not None:
                    self._executor.shutdown()
            finally:
                try:
                    if self._spin_thread is not None:
                        self._spin_thread.join(timeout=2.0)
                finally:
                    try:
                        if self._node is not None:
                            self._node.destroy_node()
                    finally:
                        self._node = None
                        self._executor = None
                        self._spin_thread = None
                        self._action = None
                        self._resend = None
                        with self._latest_lock:
                            self._latest = None
                            self._latest_stamp = None
                        try:
                            import rclpy

                            if self._rclpy_inited_here and rclpy.ok():
                                rclpy.shutdown()
                        except Exception:
                            pass
                        self._rclpy_inited_here = False
                        self._claim.release()
                        self._ros_claim.release()

    def _cancel_pending_goal(self) -> None:
        """Cancel the active trajectory goal, and any still being accepted.

        Sets ``_shutting_down`` first, so a send racing this method either
        completes its registration into ``_pending_sends`` before the
        snapshot below (and so is swept up here) or sees the flag and
        refuses to send at all -- never lands outside both. Bumps the goal
        generation for the same reason a response or result already in
        flight for this goal is recognized as superseded rather than
        reported as a failure once it arrives. Also sweeps up handles a
        superseded-but-accepted response is already canceling on the
        executor thread, in case that confirmation never finishes once the
        executor below is shut down. Handles already known -- the active
        goal and anything mid-cancellation -- are stopped first, so the arm
        stops moving without waiting on sends that may never acknowledge.
        One ``TEARDOWN_CONFIRM_TIMEOUT`` deadline, computed once here, bounds
        every wait below -- known handles, pending sends, and any handle a
        pending send turns out to have accepted -- rather than each getting
        its own: an outage affecting many of them would otherwise delay
        teardown by a multiple of the timeout. The claims are released
        regardless of whether a wait below times out, so a timeout is logged
        rather than swallowed -- it means the arm may still be moving after
        this method returns.
        """
        with self._goal_lock:
            self._shutting_down = True
            goal_handle = self._active_goal_handle
            self._active_goal_handle = None
            pending_sends = list(self._pending_sends)
            self._pending_sends.clear()
            self._pending_send_deadlines.clear()
            pending_cancellations = list(self._pending_cancellations.values())
            self._pending_cancellations.clear()
            self._goal_generation += 1
        if self._action is None:
            return
        deadline = time.monotonic() + self.TEARDOWN_CONFIRM_TIMEOUT
        handles_to_cancel = [goal_handle] if goal_handle is not None else []
        # Already being canceled by an executor-thread callback, but that
        # callback may never finish confirming it if the executor below is
        # shut down first; confirming it again here is redundant but safe.
        handles_to_cancel.extend(pending_cancellations)
        self._confirm_goals_stopped(handles_to_cancel, deadline)
        if not pending_sends:
            return
        self._spin_many_until(pending_sends, deadline)
        late_handles = []
        for send_future in pending_sends:
            if not send_future.done():
                self._logger.error(
                    "UR trajectory goal send was not acknowledged within "
                    "%.0fs while disconnecting; it may still be accepted "
                    "and run after the device claims are released.",
                    self.TEARDOWN_CONFIRM_TIMEOUT,
                )
                continue
            # A failure here must not skip the remaining sends -- one send
            # completing exceptionally is no reason to leave every other
            # send's eventual accept uncanceled.
            try:
                pending_handle = send_future.result()
            except Exception:
                self._logger.exception(
                    "UR trajectory goal send raised while disconnecting; "
                    "the arm may still be moving if it was actually accepted."
                )
                continue
            if pending_handle is not None and pending_handle.accepted:
                late_handles.append(pending_handle)
        self._confirm_goals_stopped(late_handles, deadline)

    def _confirm_goal_stopped(self, handle: Any) -> bool:
        """Cancel ``handle`` and confirm it actually stopped running.

        A ``cancel_goal_async()`` future completing only means the cancel
        service replied -- a rejected cancellation also completes it, and
        neither case means the trajectory stopped. Both the response and the
        goal's own terminal result (from :meth:`~.get_result_async`) are
        checked, so a rejection or a cancel that never takes effect is
        reported rather than treated as a safe stop. A terminal-result
        future that completes with an exception is treated the same as one
        that never completes -- both report an unconfirmed stop -- rather
        than letting ``done()`` alone stand in for a result that was never
        actually read. Blocks the caller, so this must only be called off
        the executor thread -- from teardown or from a goal-sending call
        that owns its own caller thread, never from a done-callback the
        executor itself is driving.

        Returns:
            Whether a terminal result was confirmed. ``False`` means the
            caller must keep tracking ``handle`` (rather than discard it) so
            a later disconnect can retry stopping a goal that may still be
            running.
        """
        cancel_future = handle.cancel_goal_async()
        if not self._spin_until(cancel_future, timeout=self.TEARDOWN_CONFIRM_TIMEOUT):
            self._logger.error(
                "UR trajectory goal cancellation was not confirmed within "
                "%.0fs; the arm may still be moving.",
                self.TEARDOWN_CONFIRM_TIMEOUT,
            )
            return False
        return_code = getattr(cancel_future.result(), "return_code", 0)
        if return_code != 0:
            self._logger.error(
                "UR trajectory goal cancellation was rejected by the "
                "controller (return_code=%s); the arm may still be moving.",
                return_code,
            )
        result_future = handle.get_result_async()
        if not self._spin_until(result_future, timeout=self.TEARDOWN_CONFIRM_TIMEOUT):
            self._logger.error(
                "UR trajectory goal did not reach a terminal state within "
                "%.0fs; the arm may still be moving.",
                self.TEARDOWN_CONFIRM_TIMEOUT,
            )
            return False
        try:
            result_future.result()
        except Exception:
            self._logger.exception(
                "UR trajectory goal result retrieval raised; the arm may "
                "still be moving."
            )
            return False
        return True

    def _confirm_goals_stopped(self, handles: list[Any], deadline: float) -> None:
        """Cancel every handle in ``handles`` and confirm each stopped.

        Used only during teardown, where every handle must share one
        deadline (see :meth:`_cancel_pending_goal`) rather than each getting
        its own fresh ``TEARDOWN_CONFIRM_TIMEOUT`` as :meth:`_confirm_goal_stopped`
        gives a single handle outside teardown. Submits every cancellation
        before waiting on any of them, so handles are not canceled one at a
        time with the deadline ticking down unused in between. A request,
        response, or result retrieval that raises is logged and skipped
        rather than aborting the whole sweep, so one bad handle does not
        leave the rest never asked to stop or never confirmed.
        """
        if not handles:
            return
        cancels = []
        for handle in handles:
            try:
                cancels.append((handle, handle.cancel_goal_async()))
            except Exception:
                self._logger.exception(
                    "UR trajectory goal cancellation request raised while "
                    "disconnecting; the arm may still be moving."
                )
        if not cancels:
            return
        self._spin_many_until([future for _, future in cancels], deadline)
        results = []
        for handle, cancel_future in cancels:
            if not cancel_future.done():
                self._logger.error(
                    "UR trajectory goal cancellation was not confirmed within "
                    "%.0fs while disconnecting; the arm may still be moving.",
                    self.TEARDOWN_CONFIRM_TIMEOUT,
                )
                continue
            try:
                return_code = getattr(cancel_future.result(), "return_code", 0)
            except Exception:
                self._logger.exception(
                    "UR trajectory goal cancellation response raised while "
                    "disconnecting; the arm may still be moving."
                )
                continue
            if return_code != 0:
                self._logger.error(
                    "UR trajectory goal cancellation was rejected by the "
                    "controller (return_code=%s); the arm may still be moving.",
                    return_code,
                )
            try:
                results.append((handle, handle.get_result_async()))
            except Exception:
                self._logger.exception(
                    "UR trajectory goal result retrieval raised while "
                    "disconnecting; the arm may still be moving."
                )
        if not results:
            return
        self._spin_many_until([future for _, future in results], deadline)
        for _handle, result_future in results:
            if not result_future.done():
                self._logger.error(
                    "UR trajectory goal did not reach a terminal state within "
                    "%.0fs while disconnecting; the arm may still be moving.",
                    self.TEARDOWN_CONFIRM_TIMEOUT,
                )
                continue
            try:
                result_future.result()
            except Exception:
                self._logger.exception(
                    "UR trajectory goal result retrieval raised while "
                    "disconnecting; the arm may still be moving."
                )

    @staticmethod
    def _spin_many_until(futures: list[Any], deadline: float) -> None:
        """Poll ``futures`` until every one is done or ``deadline`` passes."""
        while time.monotonic() < deadline and any(not f.done() for f in futures):
            time.sleep(0.02)

    # State.

    def _on_joint_states(self, msg: Any) -> None:
        """Cache the latest complete joint reading, keyed by joint name.

        A message missing one of the arm's joints (e.g. a gripper publishing
        on a merged topic) would otherwise overwrite a good reading with a
        partial one, or refresh the staleness clock without new arm data.
        """
        reading = dict(zip(msg.name, msg.position))
        if not all(name in reading for name in self._joint_names):
            return
        with self._latest_lock:
            self._latest = reading
            self._latest_stamp = time.monotonic()

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

    def _is_state_fresh(self) -> bool:
        """Report whether a joint_states reading has arrived recently."""
        with self._latest_lock:
            if self._latest is None or self._latest_stamp is None:
                return False
            age = time.monotonic() - self._latest_stamp
        return age <= self.STATE_STALE_AFTER

    def get_state(self) -> UR5RobotState:
        """Read joints from the latest joint_states into canonical order.

        Raises if the feed is missing or stale rather than returning a cached
        reading: :meth:`~.BaseArm.wait_until_still` and reward scoring both
        poll through this method, and a stopped feed would otherwise look
        like an arm that has settled at its last known pose instead of one
        whose connection was lost.
        """
        with self._latest_lock:
            latest = self._latest
            stamp = self._latest_stamp
        if latest is None or stamp is None:
            raise RuntimeError(
                f"UR5 (ros2) {self.JOINT_STATES_TOPIC} feed has not reported "
                "any state yet."
            )
        if time.monotonic() - stamp > self.STATE_STALE_AFTER:
            raise RuntimeError(
                f"UR5 (ros2) {self.JOINT_STATES_TOPIC} feed is stale (older "
                f"than {self.STATE_STALE_AFTER:.1f}s); refusing to report it "
                "as the current state."
            )
        joints = np.array(
            [latest.get(name, 0.0) for name in self._joint_names], dtype=float
        )
        return UR5RobotState(arm_joint_position=joints)

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

    def _require_fresh_state(self) -> None:
        """Raise if the joint-state feed is stale, so motion is never blind."""
        if not self._is_state_fresh():
            raise RuntimeError(
                f"UR5 (ros2) {self.JOINT_STATES_TOPIC} feed is stale or not "
                "connected; refusing to command motion."
            )

    def _check_goal_error(self) -> None:
        """Raise a failure a prior non-blocking goal recorded, then clear it.

        A rejected or aborted step goal is reported through a callback on the
        executor thread, not the caller's; surfacing it here, on the next
        command, is how the control loop learns about it at all. Re-checked
        against the latest accepted generation here, not only when recorded:
        a canceled goal's result can arrive before its replacement's
        acceptance is confirmed, so discarding only at write time would still
        raise here for a cancellation that normal preemption already
        explains. For that reason, a *cancellation* error is left in place
        rather than raised while a goal sent after it is still awaiting that
        acceptance decision -- raising here would abort the run on what may
        turn out to be ordinary supersession. It is resolved on the next
        call once that goal is accepted (discarded above), rejected, or
        found expired by :meth:`_on_goal_response` (which then overwrites it
        with that goal's own failure). Rejections and other failures are
        never deferred this way: unlike a canceled result, they are not an
        expected side effect of ordinary preemption, so they must still
        surface on the next command even while a later goal is in flight.
        """
        with self._goal_lock:
            error = self._goal_error
            if error is None:
                return
            generation, message, is_cancellation = error
            if self._last_accepted_generation > generation:
                self._goal_error = None
                return
            if (
                is_cancellation
                and self._pending_sends
                and self._goal_generation > generation
            ):
                return
            self._goal_error = None
        raise RuntimeError(message)

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
        self._check_goal_error()
        self._require_fresh_state()
        target = self._clip_target(q_target)
        current = np.asarray(self.get_state().arm_joint_position, dtype=float)
        distance = float(np.max(np.abs(target - current)))
        goal_time = max(distance / max(self._speed, 1e-3), self._step_goal_time)
        self._send_goal_async(target, goal_time)
        return target

    def reset_joint(
        self,
        positions: "Sequence[float]",
        duration: float = 3.0,
        *,
        max_velocity: float = np.deg2rad(30.0),
    ) -> None:
        """Move to a rest pose in radians, returning once the arm has stopped.

        Sends a single blocking trajectory goal, then waits for the joints to
        settle. The goal's ``time_from_start`` is stretched so the average
        joint speed never exceeds ``max_velocity``, and is never shorter than
        ``duration``; the controller's own trajectory interpolation then paces
        the move within that time.
        """
        self._check_goal_error()
        self._require_fresh_state()
        target = self._clip_target(positions)
        current = np.asarray(self.get_state().arm_joint_position, dtype=float)
        distance = float(np.max(np.abs(target - current)))
        max_velocity = max(float(max_velocity), 1e-3)
        time_from_start = max(float(duration), distance / max_velocity, 1.0)
        # A reset may follow an idle period; make sure control is live first.
        self._call_resend_program()
        self._send_goal(target, time_from_start)
        self.wait_until_still(duration)

    def is_robot_up(self) -> bool:
        """Report whether the node is open and joint state is flowing freshly."""
        if self._node is None:
            return False
        return self._is_state_fresh()

    # Helpers.

    def _clip_target(self, q_target: "Sequence[float]") -> np.ndarray:
        """Validate shape and clip to the configured travel.

        A non-finite target is rejected rather than clipped: ``nan`` survives
        ``np.clip`` unchanged, and ``inf``/``-inf`` would otherwise clip to an
        extreme-but-finite-looking joint target instead of surfacing the bad
        input.
        """
        target = np.asarray(q_target, dtype=float).reshape(-1)
        if target.shape != (self.DOF,):
            raise ValueError(
                f"Expected {self.DOF} joint targets for a UR5, got shape "
                f"{target.shape}."
            )
        if not np.all(np.isfinite(target)):
            raise ValueError(
                f"Expected finite joint targets for a UR5, got {target.tolist()}."
            )
        return np.clip(target, self.JOINT_LIMITS_LOWER, self.JOINT_LIMITS_UPPER)

    def _build_goal(self, target: np.ndarray, time_from_start: float) -> Any:
        """Build a single-point FollowJointTrajectory goal."""
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
        return goal

    def _send_goal(self, target: np.ndarray, time_from_start: float) -> None:
        """Send a single-point trajectory goal and block until it finishes.

        Raises instead of returning normally if the goal is rejected, is
        superseded before acceptance is confirmed, times out, or completes
        without success -- a caller such as :meth:`reset_joint` must not
        treat a failed motion as done. The still-active handle from a prior
        goal (e.g. a step trajectory a reset is replacing) is left in place
        until this one is confirmed accepted, so a concurrent teardown can
        still find and cancel whichever goal is actually running on the
        controller if this send fails, times out, or is rejected.
        """
        goal = self._build_goal(target, time_from_start)
        with self._goal_lock:
            if self._shutting_down:
                raise RuntimeError(
                    f"UR trajectory goal to {target.tolist()} was not sent: "
                    "the arm is disconnecting."
                )
            self._goal_generation += 1
            generation = self._goal_generation
            # Submitting and registering the send in the same critical
            # section as the shutdown check closes the window where a
            # concurrent _cancel_pending_goal could snapshot _pending_sends
            # before this send is added to it, and so never see a goal that
            # is about to be accepted and run after the claims are released.
            send_future = self._action.send_goal_async(goal)
            self._pending_sends.add(send_future)
        if not self._spin_until(send_future, timeout=self.GOAL_ACK_TIMEOUT):
            # Left in _pending_sends, so _cancel_pending_goal can still find
            # and cancel an eventual accept during teardown; this done
            # callback additionally cancels it immediately if it arrives
            # first, since this call is giving up ownership of the goal by
            # raising and no later command or disconnect may ever run.
            send_future.add_done_callback(
                lambda future: self._on_timed_out_send_response(future, target)
            )
            raise RuntimeError(
                f"UR trajectory goal to {target.tolist()} was not acknowledged "
                f"within {self.GOAL_ACK_TIMEOUT:.1f}s."
            )
        goal_handle = send_future.result()
        accepted = goal_handle is not None and goal_handle.accepted
        # Discarding the pending send and installing the accepted handle in
        # one critical section closes the window where a concurrent
        # _cancel_pending_goal (e.g. from a disconnect on another thread)
        # could see neither and let this goal run unmanaged. Superseded is
        # judged the same way the non-blocking path does -- against
        # _last_accepted_generation and _shutting_down, not send order --
        # so a step sent while this reset's accept is still pending does not
        # cancel it before that step's own acceptance (or rejection) is
        # known.
        with self._goal_lock:
            self._pending_sends.discard(send_future)
            installed = (
                accepted
                and generation >= self._last_accepted_generation
                and not self._shutting_down
            )
            if installed:
                self._active_goal_handle = goal_handle
                self._last_accepted_generation = generation
            elif accepted:
                # Superseded between send and accept, but already running on
                # the controller: tracked here so a concurrent teardown can
                # still find and confirm it if the blocking confirmation
                # below is cut short (e.g. the executor is shut down while
                # this call is waiting on it).
                self._pending_cancellations[id(goal_handle)] = goal_handle
        if not accepted:
            raise RuntimeError(
                f"UR trajectory goal to {target.tolist()} was rejected by the "
                "controller."
            )
        if not installed:
            # Blocking below is safe: this runs on the caller's own thread,
            # not the executor's. The handle stays in _pending_cancellations
            # unless a terminal result is actually confirmed, so a
            # subsequent disconnect can retry stopping a goal that may still
            # be running rather than losing track of it here.
            if self._confirm_goal_stopped(goal_handle):
                with self._goal_lock:
                    self._pending_cancellations.pop(id(goal_handle), None)
            raise RuntimeError(
                f"UR trajectory goal to {target.tolist()} was superseded "
                "before acceptance could be confirmed; canceled."
            )

        result_future = goal_handle.get_result_async()
        if not self._spin_until(result_future, timeout=time_from_start + 10.0):
            # Use the same confirmation helper as the superseded-before-accept
            # branch above, instead of only waiting for the cancel service's
            # reply: that reply means the request was received, not that the
            # trajectory actually stopped.
            if self._confirm_goal_stopped(goal_handle):
                raise RuntimeError(
                    f"UR trajectory goal to {target.tolist()} did not "
                    f"complete within {time_from_start + 10.0:.1f}s; "
                    "canceled and confirmed stopped."
                )
            raise RuntimeError(
                f"UR trajectory goal to {target.tolist()} did not complete "
                f"within {time_from_start + 10.0:.1f}s, and cancellation "
                "could not be confirmed; the arm may still be moving."
            )
        # Read before clearing _active_goal_handle: a retrieval failure must
        # leave the handle tracked so a subsequent disconnect can still find
        # and retry canceling it, instead of losing it to a result that was
        # never actually confirmed.
        result = result_future.result()
        with self._goal_lock:
            if generation == self._goal_generation:
                self._active_goal_handle = None
        message = self._goal_failure_message(result, target)
        if message is not None:
            raise RuntimeError(message)

    def _on_timed_out_send_response(self, future: Any, target: np.ndarray) -> None:
        """Cancel a goal that arrives accepted after its sender gave up.

        Runs on the executor thread, attached only when :meth:`_send_goal`
        has already raised on an acknowledgement timeout: no caller is left
        waiting for this goal's outcome, so an acceptance that arrives
        afterward must be canceled here instead of running until a later
        command or disconnect happens to find it in ``_pending_sends``.
        Resolving the future and transferring an accepted handle into
        ``_pending_cancellations`` in one critical section, as
        :meth:`_on_goal_response` does, closes the window where a concurrent
        teardown's snapshot would see this goal in neither set and release
        the claims without canceling it. A response that completes with an
        exception instead of a goal handle is logged and the future is
        still discarded from ``_pending_sends``, rather than left there
        forever with the caller long gone to ever revisit it.
        """
        try:
            goal_handle = future.result()
        except Exception:
            with self._goal_lock:
                self._pending_sends.discard(future)
            self._logger.exception(
                "UR trajectory goal to %s send raised after its "
                "acknowledgement timeout already made its caller give up; "
                "the arm may still be moving if it was actually accepted.",
                target.tolist(),
            )
            return
        accepted = goal_handle is not None and goal_handle.accepted
        with self._goal_lock:
            self._pending_sends.discard(future)
            if accepted:
                self._pending_cancellations[id(goal_handle)] = goal_handle
        if not accepted:
            return
        cancel_future = goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(
            lambda cancel_future: self._on_superseded_cancel_response(
                cancel_future, goal_handle, target
            )
        )

    def _check_pending_step_sends(self, target: np.ndarray) -> None:
        """Raise if earlier step-goal sends look like a stuck controller.

        Must be called with ``_goal_lock`` held, before registering a new
        send. A step send that is never acknowledged would otherwise sit in
        ``_pending_sends`` forever: nothing else revisits it, since
        :meth:`_on_goal_response` only runs once the controller actually
        responds. Faulting once enough sends have piled up, or once the
        oldest has outlived its ``GOAL_ACK_TIMEOUT`` deadline, turns that
        silent accumulation into a surfaced error on the next step.
        """
        if len(self._pending_sends) >= self.MAX_PENDING_STEP_SENDS:
            raise RuntimeError(
                f"UR trajectory goal to {target.tolist()} was not sent: "
                f"{len(self._pending_sends)} earlier step goal(s) are still "
                "unacknowledged; the controller may be unresponsive."
            )
        now = time.monotonic()
        if any(now >= deadline for deadline in self._pending_send_deadlines.values()):
            raise RuntimeError(
                f"UR trajectory goal to {target.tolist()} was not sent: an "
                f"earlier step goal was not acknowledged within "
                f"{self.GOAL_ACK_TIMEOUT:.1f}s; the controller may be "
                "unresponsive."
            )

    def _send_goal_async(self, target: np.ndarray, time_from_start: float) -> None:
        """Send a single-point trajectory goal without blocking.

        The server accepting a new goal preempts whatever the controller was
        already running, so nothing here needs to cancel the previous goal
        first. The outcome is tracked through done-callbacks on the executor
        thread instead of a caller-owned future: bumping the goal generation
        before sending marks any still-outstanding earlier goal as superseded,
        so its eventual response or result is dropped rather than reported as
        a failure. Raises instead of sending if teardown has already begun,
        or if earlier step goals look stuck (see
        :meth:`_check_pending_step_sends`) -- callers such as
        :meth:`move_joints` have no other signal that the controller has
        stopped responding, since a stalled action server does not affect
        ``/joint_states`` or state freshness.
        """
        goal = self._build_goal(target, time_from_start)
        with self._goal_lock:
            if self._shutting_down:
                raise RuntimeError(
                    f"UR trajectory goal to {target.tolist()} was not sent: "
                    "the arm is disconnecting."
                )
            self._check_pending_step_sends(target)
            self._goal_generation += 1
            generation = self._goal_generation
            # Submitting and registering the send in the same critical
            # section as the shutdown check closes the window where a
            # concurrent _cancel_pending_goal could snapshot _pending_sends
            # before this send is added to it, and so never see a goal that
            # is about to be accepted and run after the claims are released.
            send_future = self._action.send_goal_async(goal)
            self._pending_sends.add(send_future)
            self._pending_send_deadlines[send_future] = (
                time.monotonic() + self.GOAL_ACK_TIMEOUT
            )
        # Outside the lock: a fake or real future already done can invoke
        # this callback synchronously, and _on_goal_response acquires
        # _goal_lock itself -- calling it while still holding the lock would
        # deadlock, since threading.Lock is not reentrant.
        send_future.add_done_callback(
            lambda future: self._on_goal_response(future, generation, target)
        )

    def _on_goal_response(
        self, future: Any, generation: int, target: np.ndarray
    ) -> None:
        """Handle the controller's accept/reject decision for a step goal.

        Discards the pending send and installs the accepted handle in one
        critical section, so a concurrent _cancel_pending_goal (e.g. from a
        disconnect on another thread) cannot run between the two and see
        neither -- which would let this goal keep running, uncanceled, after
        the device claims are released. Superseded is judged against
        ``_last_accepted_generation``, not ``_goal_generation``: when
        acknowledgements run slower than the control period, later sends
        bump ``_goal_generation`` well before their own acceptance is known,
        and canceling this goal merely because a later one was *sent* would
        stop valid motion and leave nothing running if that later goal is
        then rejected. This goal is current, and stays uncanceled, as long
        as no strictly newer generation has actually been accepted yet,
        teardown has not already started, and this send has not already
        outlived the deadline :meth:`_check_pending_step_sends` judges it
        by: an accept that arrives after the teardown sweep has already
        timed out on this send, or after this send has already faulted a
        later ``move_joints`` call as unacknowledged, must still take the
        cancellation branch below rather than be installed as active and
        left running unmanaged. A response that completes with an exception
        instead of a goal handle is removed from both tracking containers
        here and reported through :meth:`_record_goal_error`, rather than
        left pending forever while the original failure goes unreported.
        """
        try:
            goal_handle = future.result()
        except Exception:
            with self._goal_lock:
                self._pending_sends.discard(future)
                self._pending_send_deadlines.pop(future, None)
            self._record_goal_error(
                generation,
                f"UR trajectory goal to {target.tolist()} acknowledgement "
                "retrieval raised; the controller's accept/reject response "
                "may have been lost.",
            )
            return
        accepted = goal_handle is not None and goal_handle.accepted
        with self._goal_lock:
            self._pending_sends.discard(future)
            deadline = self._pending_send_deadlines.pop(future, None)
            expired = deadline is not None and time.monotonic() >= deadline
            current = (
                accepted
                and generation >= self._last_accepted_generation
                and not self._shutting_down
                and not expired
            )
            if current:
                self._active_goal_handle = goal_handle
                self._last_accepted_generation = generation
            elif accepted:
                # Superseded between send and accept, but already running on
                # the controller: tracked here so a concurrent teardown can
                # still find and confirm it if the cancellation below never
                # finishes (e.g. the executor is shut down first).
                self._pending_cancellations[id(goal_handle)] = goal_handle
        if not accepted:
            self._record_goal_error(
                generation,
                f"UR trajectory goal to {target.tolist()} was rejected by "
                "the controller.",
            )
            return
        if expired:
            # Accepted after _check_pending_step_sends() already faulted this
            # send (or a teardown sweep timed out on it): the motion is about
            # to be canceled below, so record the lost acknowledgement now or
            # it is never reported. _record_goal_error() already suppresses
            # this once a strictly newer generation is confirmed accepted.
            self._record_goal_error(
                generation,
                f"UR trajectory goal to {target.tolist()} was accepted "
                f"after not being acknowledged within "
                f"{self.GOAL_ACK_TIMEOUT:.1f}s; canceled.",
            )
        if not current:
            # This callback runs on the executor thread, which must keep
            # spinning for any future (including this cancel's) to ever
            # resolve, so the cancellation response and terminal result are
            # checked through further done-callbacks rather than blocking.
            cancel_future = goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(
                lambda future: self._on_superseded_cancel_response(
                    future, goal_handle, target
                )
            )
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future: self._on_goal_result(future, generation, target)
        )

    def _on_superseded_cancel_response(
        self, future: Any, goal_handle: Any, target: np.ndarray
    ) -> None:
        """Check a superseded goal's cancellation, without blocking to do so.

        Runs on the executor thread: the response is already resolved on
        ``future``, and the next check is scheduled through another
        done-callback, so nothing here waits. Either the response retrieval
        or the terminal-result request can raise; both are caught and logged
        as an unconfirmed stop rather than left to propagate from an
        executor callback, and neither pops ``goal_handle`` from
        ``_pending_cancellations`` -- it stays tracked so a later teardown
        can still find and retry it.
        """
        try:
            return_code = getattr(future.result(), "return_code", 0)
        except Exception:
            self._logger.exception(
                "UR trajectory goal to %s cancellation response (after "
                "being superseded) raised; the arm may still be executing it.",
                target.tolist(),
            )
            return
        if return_code != 0:
            self._logger.error(
                "UR trajectory goal to %s cancellation (after being "
                "superseded) was rejected by the controller (return_code=%s); "
                "the arm may still be executing it.",
                target.tolist(),
                return_code,
            )
        try:
            result_future = goal_handle.get_result_async()
        except Exception:
            self._logger.exception(
                "UR trajectory goal to %s terminal-result request (after "
                "being superseded) raised; the arm may still be executing it.",
                target.tolist(),
            )
            return
        result_future.add_done_callback(
            lambda future: self._on_superseded_cancel_result(
                future, goal_handle, target
            )
        )

    def _on_superseded_cancel_result(
        self, future: Any, goal_handle: Any, target: np.ndarray
    ) -> None:
        """Stop tracking a superseded goal once it reaches a terminal state.

        The result is retrieved before the handle is dropped from
        ``_pending_cancellations``: if retrieval raises, the handle is left
        tracked so a later disconnect can still find it and retry
        cancellation, instead of the goal being forgotten uncanceled.
        """
        from action_msgs.msg import GoalStatus

        try:
            status = getattr(future.result(), "status", GoalStatus.STATUS_CANCELED)
        except Exception:
            self._logger.exception(
                "UR trajectory goal to %s result retrieval raised after "
                "being superseded; the arm may still be executing it.",
                target.tolist(),
            )
            return
        with self._goal_lock:
            self._pending_cancellations.pop(id(goal_handle), None)
        if status not in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_SUCCEEDED):
            self._logger.error(
                "UR trajectory goal to %s did not reach a canceled state "
                "after being superseded (status=%s); the arm may still have "
                "been executing it.",
                target.tolist(),
                status,
            )

    def _on_goal_result(self, future: Any, generation: int, target: np.ndarray) -> None:
        """Record a step goal's outcome, unless a later goal superseded it.

        The handle is cleared only after its result is retrieved and
        validated: a retrieval failure leaves ``_active_goal_handle`` set so
        a disconnect can still find and retry canceling it, and is reported
        through :meth:`_record_goal_error` so the next command also learns
        about it instead of seeing no failure at all. Compared against
        ``_last_accepted_generation`` rather than ``_goal_generation``: the
        latter bumps as soon as a later goal is sent, before it is known to
        be accepted, which would leave this now-terminal handle in place for
        a disconnect to needlessly re-cancel if that later send is rejected.
        """
        try:
            result = future.result()
        except Exception:
            self._record_goal_error(
                generation,
                f"UR trajectory goal to {target.tolist()} result retrieval "
                "raised; the arm may still be moving.",
            )
            return
        with self._goal_lock:
            if generation == self._last_accepted_generation:
                self._active_goal_handle = None
        message = self._goal_failure_message(result, target)
        if message is not None:
            from action_msgs.msg import GoalStatus

            status = getattr(result, "status", GoalStatus.STATUS_SUCCEEDED)
            self._record_goal_error(
                generation,
                message,
                is_cancellation=(status == GoalStatus.STATUS_CANCELED),
            )

    @staticmethod
    def _goal_failure_message(result: Any, target: np.ndarray) -> Optional[str]:
        """Return why a finished goal failed, or ``None`` if it succeeded.

        A canceled goal can still report ``error_code == 0`` in its trajectory
        payload, so success also requires the result wrapper's own ``status``
        to be ``STATUS_SUCCEEDED`` -- otherwise a cancellation (for instance
        from :meth:`_cancel_pending_goal`) would be read as a completed move.
        """
        from action_msgs.msg import GoalStatus

        status = getattr(result, "status", GoalStatus.STATUS_SUCCEEDED)
        error_code = getattr(result.result, "error_code", 0)
        if status == GoalStatus.STATUS_SUCCEEDED and error_code == 0:
            return None
        return (
            f"UR trajectory goal to {target.tolist()} finished unsuccessfully "
            f"(status={status}, error_code={error_code})."
        )

    def _record_goal_error(
        self, generation: int, message: str, is_cancellation: bool = False
    ) -> None:
        """Surface a step goal's failure to the next command.

        Stored with its generation so :meth:`_check_goal_error` can discard
        it once a newer goal is confirmed accepted, even if that
        confirmation arrives after this call -- a canceled goal's result and
        its replacement's acceptance can arrive in either order. A
        generation bump alone does not justify dropping it here: at a fast
        command rate a rejection or abort can arrive well after later sends
        have already bumped ``_goal_generation``, while none of those later
        goals has actually been accepted yet -- treating that as
        "superseded" would silently hide a controller that is rejecting
        every goal. Only replaces a stored error from an equal-or-newer
        generation: two goal-result callbacks can run out of order on the
        executor thread, and a stale goal's canceled-result callback arriving
        after a newer goal's genuine failure was already recorded must not
        overwrite it. ``is_cancellation`` marks a goal canceled as the
        expected side effect of being superseded, which
        :meth:`_check_goal_error` treats differently from a rejection or
        other genuine failure: it may still be waiting on its replacement's
        own acceptance decision when the next command checks.
        """
        with self._goal_lock:
            if self._shutting_down:
                return
            if self._goal_error is None or generation >= self._goal_error[0]:
                self._goal_error = (generation, message, is_cancellation)
        self._logger.error(message)

    def _call_resend_program(self) -> None:
        """Restart the driver's external-control program (headless mode).

        Outside headless mode the upstream driver's own ``sendRobotProgram()``
        always returns failure, so this is skipped entirely rather than
        raising on a service call that cannot succeed by design there.
        """
        if not self._headless_mode:
            return
        from std_srvs.srv import Trigger

        if not self._resend.wait_for_service(timeout_sec=5.0):
            self._logger.warning(
                "%s not available; assuming external control is already running.",
                self.RESEND_PROGRAM_SERVICE,
            )
            return
        future = self._resend.call_async(Trigger.Request())
        if not self._spin_until(future, timeout=5.0):
            raise RuntimeError(
                f"{self.RESEND_PROGRAM_SERVICE} did not respond to the "
                "restart request within 5.0s."
            )
        response = future.result()
        if not response.success:
            raise RuntimeError(
                f"{self.RESEND_PROGRAM_SERVICE} refused to restart external "
                f"control: {response.message}"
            )

    @staticmethod
    def _spin_until(future: Any, timeout: float) -> bool:
        """Wait for a future the background executor is driving to complete."""
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        return future.done()
