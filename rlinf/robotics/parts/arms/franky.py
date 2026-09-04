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

"""Franka control through the libfranka-based ``franky`` bindings.

A PREEMPT_RT kernel, ``rtprio>=80``, and unlimited memory locking are
recommended. The driver logs a warning if real-time hardening is unavailable.
"""

import ctypes
import ctypes.util
import os
import time
from collections.abc import Mapping
from typing import Any, ClassVar, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.arms.base import Arm, BaseArm, CartesianCompliance
from rlinf.robotics.parts.arms.franka import (
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    JOINT_VEL_LIMITS,
    FrankaRobotState,
    validated_robot_ip,
)
from rlinf.robotics.parts.base import Action, Features, Observation
from rlinf.robotics.parts.claims import DeviceClaim
from rlinf.utils.logging import get_logger


@Arm.register("franky")
class FrankyArm(BaseArm):
    """Franka arm controlled through libfranka by Franky."""

    #: Collision reflex trip points, in Nm and N.
    TORQUE_THRESHOLD: ClassVar[list[float]] = [80.0] * 4 + [11.0] * 3
    FORCE_THRESHOLD: ClassVar[list[float]] = [100.0] * 3 + [25.0] * 3
    #: Joint-space impedance gains, in Nm/rad and Nms/rad.
    JOINT_STIFFNESS: ClassVar[list[float]] = [
        103.75,
        265.734,
        227.273,
        221.445,
        13.5,
        12.818,
        5.134,
    ]
    JOINT_DAMPING: ClassVar[list[float]] = [16.7, 40.263, 25.0, 12.862, 1.5, 2.0, 1.331]
    #: Speed scale for position-controlled motions such as reset_joint.
    DYNAMICS_FACTOR: ClassVar[float] = 0.2
    #: SCHED_FIFO priority requested for the control thread.
    RT_PRIORITY: ClassVar[int] = 80
    #: Floor on the timestep used for velocity feedforward, in seconds.
    DQ_MIN_DT_S: ClassVar[float] = 1e-3
    #: mlockall(2) flags.
    MCL_CURRENT: ClassVar[int] = 1
    MCL_FUTURE: ClassVar[int] = 2
    MCL_ONFAULT: ClassVar[int] = 4

    @classmethod
    def declare(
        cls,
        address: str,
        *,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        compliance: Optional[CartesianCompliance] = None,
        **placement: Any,
    ) -> "FrankyArm":
        """Declare a libfranka arm with the impedance settings offered."""
        cls.refuse_unused(
            gripper_type=gripper_type,
            gripper_connection=gripper_connection,
            end_effector_type=end_effector_type,
            end_effector_config=end_effector_config,
        )
        return cls(address, compliance=compliance, **placement)

    def __init__(
        self, robot_ip: str, compliance: Optional[CartesianCompliance] = None
    ) -> None:
        self._logger = get_logger()
        self._robot_ip = validated_robot_ip(robot_ip, type(self).__name__)
        self._compliance = compliance or CartesianCompliance()
        self._cart_k_t = self._compliance.translational_stiffness
        self._cart_k_r = self._compliance.rotational_stiffness
        self._cart_k_ns = self._compliance.nullspace_stiffness
        self._cart_trans_clip = np.full(
            3, self._compliance.translational_clip, dtype=np.float64
        )
        self._cart_rot_clip = np.full(
            3, self._compliance.rotational_clip, dtype=np.float64
        )
        # libfranka gives out arm control once; a second session anywhere on
        # this machine reads as a UDP timeout in whichever holds it.
        self._claim = DeviceClaim(f"franky-arm:{self._robot_ip}", type(self).__name__)
        self._franky = None
        self._robot = None
        self._tracker = None
        self._prev_target_q: Optional[np.ndarray] = None
        self._prev_target_ts: Optional[float] = None
        self._cart_tracker = None
        self._prev_cart_target_xyz: Optional[np.ndarray] = None
        self._prev_cart_target_quat: Optional[np.ndarray] = None

    @property
    def action_features(self) -> Features:
        """Describe supported joint and Cartesian targets."""
        return {"joint_position": {}, "tcp_pose": {}}

    def _open(self) -> Any:
        """Connect the arm's libfranka session."""
        self._apply_rt_hardening()
        self._claim.acquire()

        import franky

        self._franky = franky
        self._robot = franky.Robot(self._robot_ip)
        self._robot.recover_from_errors()
        self._robot.relative_dynamics_factor = self.DYNAMICS_FACTOR
        self._robot.set_collision_behavior(self.TORQUE_THRESHOLD, self.FORCE_THRESHOLD)
        self._logger.info(f"FrankyArm connected to robot at {self._robot_ip}")
        return self._robot

    def reset(self) -> None:
        """Leave task-specific reset positions to the caller."""

    def send_action(self, action: Action) -> Observation:
        """Apply one or both canonical arm targets."""
        unknown = set(action) - {"joint_position", "tcp_pose"}
        if unknown:
            raise KeyError(f"Unknown Franky actions: {sorted(unknown)}")
        if "joint_position" in action:
            self.move_joints(action["joint_position"])
        if "tcp_pose" in action:
            self.move_tcp_pose(action["tcp_pose"])
        return action

    def _release(self, device: Any) -> None:
        """Stop active motion and drop the libfranka session."""
        try:
            self.cleanup()
        finally:
            self._robot = None
            self._claim.release()

    def _apply_rt_hardening(self) -> None:
        """Lock memory, raise priority, pin affinity. All best-effort."""
        try:
            libc = ctypes.CDLL(
                ctypes.util.find_library("c") or "libc.so.6", use_errno=True
            )
            # Locking on fault keeps the guarantee without reading every
            # mapped library off disk, which costs minutes on a cold cache
            # once the process has loaded torch.
            flags = self.MCL_CURRENT | self.MCL_FUTURE
            if libc.mlockall(flags | self.MCL_ONFAULT) != 0 and (
                libc.mlockall(flags) != 0
            ):
                self._logger.warning(f"mlockall: {os.strerror(ctypes.get_errno())}")
        except Exception as e:
            self._logger.warning(f"mlockall unavailable: {e}")
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(self.RT_PRIORITY))
        except PermissionError:
            self._logger.warning(
                f"SCHED_FIFO denied; user lacks rtprio>={self.RT_PRIORITY} "
                f"(check /etc/security/limits.d for `<user> - rtprio 99`)"
            )
        except Exception as e:
            self._logger.warning(f"SCHED_FIFO failed: {e}")
        ncpu = os.cpu_count() or 1
        if ncpu >= 6:
            try:
                os.sched_setaffinity(0, {0, 1} | set(range(4, ncpu)))
            except Exception as e:
                self._logger.warning(f"sched_setaffinity failed: {e}")

    def _safe_join(self) -> None:
        # Drain latched motion errors during setup and teardown.
        try:
            self._robot.join_motion()
        except Exception:
            pass

    def is_robot_up(self) -> bool:
        """Whether the arm answers. An end effector reports its own readiness."""
        try:
            _ = self._robot.state
            return True
        except Exception:
            return False

    def get_state(self) -> FrankaRobotState:
        raw = self._robot.state
        affine = raw.O_T_EE
        # Franky and SciPy both use xyzw quaternion order.
        tcp_pose = np.concatenate(
            [
                np.asarray(affine.translation, dtype=np.float64),
                np.asarray(affine.quaternion, dtype=np.float64),
            ]
        )
        joint_pos = np.asarray(raw.q, dtype=np.float64)
        joint_vel = np.asarray(raw.dq, dtype=np.float64)
        K_F_ext = np.asarray(raw.K_F_ext_hat_K, dtype=np.float64)
        jacobian = np.asarray(
            self._robot.model.zero_jacobian(self._franky.Frame.EndEffector, raw),
            dtype=np.float64,
        ).reshape(6, 7)

        s = FrankaRobotState()
        s.tcp_pose = tcp_pose
        s.arm_joint_position = joint_pos
        s.arm_joint_velocity = joint_vel
        s.tcp_force = K_F_ext[:3]
        s.tcp_torque = K_F_ext[3:]
        s.arm_jacobian = jacobian
        s.tcp_vel = jacobian @ joint_vel
        return s

    def reconfigure_compliance_params(self, params: "Mapping[str, float]") -> None:
        """Cap the request to what a client-side loop can hold, and apply it."""
        if not params:
            return

        def value(name: str, fallback: float) -> float:
            given = params.get(name, fallback)
            return float(fallback if given is None else given)

        limits = self._compliance
        k_t = min(
            value("translational_stiffness", self._cart_k_t), limits.stiffness_cap
        )
        k_r = min(
            value("rotational_stiffness", self._cart_k_r),
            limits.rotational_stiffness_cap,
        )
        k_ns = value("nullspace_stiffness", self._cart_k_ns)

        # franky clips symmetrically, so take the looser of each direction pair.
        def clip(prefix: str, current: np.ndarray, floor: float) -> np.ndarray:
            out = np.array(current, dtype=np.float64)
            for i, axis in enumerate("xyz"):
                named = [
                    params[key]
                    for key in (f"{prefix}_clip_{axis}", f"{prefix}_clip_neg_{axis}")
                    if params.get(key) is not None
                ]
                if named:
                    out[i] = max(max(float(v) for v in named), floor)
            return out

        trans_clip = clip("translational", self._cart_trans_clip, limits.clip_floor)
        rot_clip = clip("rotational", self._cart_rot_clip, limits.rotational_clip_floor)

        clips_changed = not (
            np.allclose(trans_clip, self._cart_trans_clip)
            and np.allclose(rot_clip, self._cart_rot_clip)
        )
        self._cart_k_t, self._cart_k_r, self._cart_k_ns = k_t, k_r, k_ns
        self._cart_trans_clip, self._cart_rot_clip = trans_clip, rot_clip

        self._logger.info(
            "Compliance: K_t=%.0f (cap %.0f), K_r=%.1f (cap %.1f), K_ns=%.1f, "
            "trans_clip=%s, rot_clip=%s",
            k_t,
            limits.stiffness_cap,
            k_r,
            limits.rotational_stiffness_cap,
            k_ns,
            np.array2string(trans_clip, precision=4),
            np.array2string(rot_clip, precision=4),
        )

        if self._cart_tracker is None:
            return
        if clips_changed:
            self._stop_cart_tracking_motion()
            return
        self._cart_tracker.set_gains(
            translational_stiffness=k_t,
            rotational_stiffness=k_r,
            nullspace_stiffness=k_ns,
        )

    def clear_errors(self) -> None:
        self._robot.recover_from_errors()

    def _ensure_tracking_motion(self) -> None:
        if self._tracker is not None:
            return
        self._stop_cart_tracking_motion()
        self._safe_join()
        self._robot.recover_from_errors()
        self._tracker = self._franky.JointImpedanceTracker(
            self._robot,
            stiffness=np.array(self.JOINT_STIFFNESS, dtype=np.float64),
            damping=np.array(self.JOINT_DAMPING, dtype=np.float64),
            compensate_coriolis=True,
        )
        self._logger.info("Joint impedance tracker started")

    def _stop_tracking_motion(self) -> None:
        if self._tracker is None:
            return
        # tracker.stop may re-raise a latched asynchronous reflex error.
        try:
            self._tracker.stop()
        except Exception as e:
            self._logger.warning(f"joint tracker.stop surfaced latched error: {e}")
        self._tracker = None
        self._prev_target_q = None
        self._prev_target_ts = None
        self._safe_join()
        self._robot.recover_from_errors()

    def move_joints(self, joint_positions: np.ndarray) -> None:
        # Velocity feedforward reduces PD lag and overshoot at 10 Hz.
        assert len(joint_positions) == 7
        q = np.clip(
            np.asarray(joint_positions, dtype=np.float64),
            JOINT_LIMITS_LOWER,
            JOINT_LIMITS_UPPER,
        )
        now = time.perf_counter()
        if self._prev_target_q is not None:
            dt = max(now - self._prev_target_ts, self.DQ_MIN_DT_S)
            dq_ff = np.clip(
                (q - self._prev_target_q) / dt, -JOINT_VEL_LIMITS, JOINT_VEL_LIMITS
            )
        else:
            dq_ff = None
        self._ensure_tracking_motion()
        self._tracker.set_target(q, dq=dq_ff)
        self._prev_target_q = q
        self._prev_target_ts = now

    def _ensure_cart_tracking_motion(self) -> None:
        if self._cart_tracker is not None:
            return
        self._stop_tracking_motion()
        self._safe_join()
        self._robot.recover_from_errors()
        nullspace_target = np.asarray(self._robot.state.q, dtype=np.float64).copy()

        self._cart_tracker = self._franky.CartesianImpedanceTracker(
            self._robot,
            translational_stiffness=self._cart_k_t,
            rotational_stiffness=self._cart_k_r,
            nullspace_target=nullspace_target,
            nullspace_stiffness=self._cart_k_ns,
            translational_error_clip=self._cart_trans_clip,
            rotational_error_clip=self._cart_rot_clip,
            max_delta_tau=self._compliance.max_delta_tau,
            gains_time_constant=self._compliance.gains_time_constant,
        )
        self._logger.info(
            f"Cartesian impedance tracker started "
            f"(K_t={self._cart_k_t:.0f} N/m, "
            f"K_r={self._cart_k_r:.1f} Nm/rad, "
            f"K_ns={self._cart_k_ns:.1f} Nm/rad, "
            f"trans_clip={np.array2string(self._cart_trans_clip, precision=4)}, "
            f"rot_clip={np.array2string(self._cart_rot_clip, precision=4)})"
        )

    def _stop_cart_tracking_motion(self) -> None:
        if self._cart_tracker is None:
            return
        try:
            self._cart_tracker.stop()
        except Exception as e:
            self._logger.warning(f"cart tracker.stop surfaced latched error: {e}")
        self._cart_tracker = None
        self._prev_cart_target_xyz = None
        self._prev_cart_target_quat = None
        self._safe_join()
        self._robot.recover_from_errors()

    def move_tcp_pose(self, pose: np.ndarray) -> None:
        # Twist feedforward from 10 Hz finite differences destabilizes joint 7.
        # Pose layout: xyz position followed by an xyzw quaternion.
        pose = np.asarray(pose, dtype=np.float64)
        assert pose.shape == (7,), (
            f"pose must be (7,) [xyz, quat_xyzw]; got {pose.shape}"
        )
        xyz_in = pose[:3]
        quat_in = pose[3:] / np.linalg.norm(pose[3:])

        self._ensure_cart_tracking_motion()

        if self._prev_cart_target_xyz is None:
            live = self._robot.state.O_T_EE
            self._prev_cart_target_xyz = np.asarray(live.translation, dtype=np.float64)
            seed_quat = np.asarray(live.quaternion, dtype=np.float64)
            self._prev_cart_target_quat = seed_quat / np.linalg.norm(seed_quat)

        prev_xyz = self._prev_cart_target_xyz
        prev_quat = self._prev_cart_target_quat

        if self._compliance.max_step > 0:
            dxyz = xyz_in - prev_xyz
            d = float(np.linalg.norm(dxyz))
            if d > self._compliance.max_step:
                xyz = prev_xyz + dxyz * (self._compliance.max_step / d)
            else:
                xyz = xyz_in
        else:
            xyz = xyz_in

        # Align quaternion hemispheres before interpolating along the short arc.
        if float(np.dot(quat_in, prev_quat)) < 0.0:
            quat_in = -quat_in
        if self._compliance.max_step_rad > 0:
            delta_R = R.from_quat(quat_in) * R.from_quat(prev_quat).inv()
            rotvec = delta_R.as_rotvec()
            ang = float(np.linalg.norm(rotvec))
            if ang > self._compliance.max_step_rad:
                rotvec = rotvec * (self._compliance.max_step_rad / ang)
                quat = (R.from_rotvec(rotvec) * R.from_quat(prev_quat)).as_quat()
            else:
                quat = quat_in
        else:
            quat = quat_in

        self._prev_cart_target_xyz = xyz
        self._prev_cart_target_quat = quat / np.linalg.norm(quat)

        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = xyz

        self._cart_tracker.set_target(self._franky.Affine(T))

    def reset_joint(self, positions: list[float]) -> None:
        assert len(positions) == 7
        self._stop_tracking_motion()
        self._stop_cart_tracking_motion()
        franky = self._franky
        motion = franky.JointMotion(
            franky.JointState(position=np.asarray(positions, dtype=np.float64)),
            reference_type=franky.ReferenceType.Absolute,
        )
        self._robot.move(motion)

    def cleanup(self) -> None:
        """Stop any motion in flight; the arm holds nothing else to release."""
        self._stop_tracking_motion()
        self._stop_cart_tracking_motion()
        self._safe_join()
