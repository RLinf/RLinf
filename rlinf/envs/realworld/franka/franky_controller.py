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

"""Franka controller backed by libfranka via the ``franky`` bindings.

Expects a PREEMPT_RT kernel and ``rtprio>=80`` / unlimited memlock for the
calling user; otherwise ``_apply_rt_hardening`` falls back to best-effort
and logs a warning.
"""

import ctypes
import ctypes.util
import os
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.gripper import create_gripper
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .franka_robot_state import FrankaRobotState

# Franka Panda joint position / velocity limits.
JOINT_LIMITS_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# Hard limits − 0.1 rad/s margin (same as polymetis).
JOINT_VEL_LIMITS = np.array([2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51])

_TORQUE_THRESHOLD = [80.0, 80.0, 80.0, 80.0, 11.0, 11.0, 11.0]
_FORCE_THRESHOLD = [100.0, 100.0, 100.0, 25.0, 25.0, 25.0]

_JOINT_STIFFNESS = [103.75, 265.734, 227.273, 221.445, 13.5, 12.818, 5.134]
_JOINT_DAMPING = [16.7, 40.263, 25.0, 12.862, 1.5, 2.0, 1.331]


def _envf(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


_CART_TRANS_STIFFNESS = _envf("RLINF_CART_K_T", 1000.0)  # N/m
_CART_ROT_STIFFNESS = _envf("RLINF_CART_K_R", 50.0)  # Nm/rad
_CART_MAX_DELTA_TAU = _envf("RLINF_CART_MAX_DTAU", 0.3)  # Nm / 1 kHz cycle
_CART_TRANS_ERROR_CLIP_M = _envf("RLINF_CART_ERR_CLIP_M", 0.05)  # m
_CART_ROT_ERROR_CLIP_RAD = _envf("RLINF_CART_ERR_CLIP_RAD", 0.3)  # rad
_CART_GAINS_TC = _envf("RLINF_CART_GAINS_TC", 0.1)  # s

# Per-call slew limit so a single-frame dataset jump becomes a ramp.
_CART_MAX_STEP_M = _envf("RLINF_CART_MAX_STEP_M", 0.03)  # m / call
_CART_MAX_STEP_RAD = _envf("RLINF_CART_MAX_STEP_RAD", 0.15)  # rad / call

_DYNAMICS_FACTOR = 0.2

_DQ_MIN_DT_S = 1e-3
_RT_PRIORITY = 80
_MCL_CURRENT, _MCL_FUTURE = 1, 2


class FrankyController(Worker):
    """One Franka arm. Spawned per-arm as a Ray actor by ``launch_controller``."""

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        gripper_type: str = "robotiq",
        gripper_connection: Optional[str] = None,
    ):
        return FrankyController.create_group(
            robot_ip, gripper_type, gripper_connection
        ).launch(
            cluster=Cluster(),
            placement_strategy=NodePlacementStrategy(node_ranks=[node_rank]),
            name=f"FrankyController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        gripper_type: str = "robotiq",
        gripper_connection: Optional[str] = None,
    ):
        super().__init__()
        self._logger = get_logger()

        # Must precede the franky import so mlockall catches its allocations.
        self._apply_rt_hardening()

        import franky

        self._franky = franky
        self._robot = franky.Robot(robot_ip)
        self._robot.recover_from_errors()
        self._robot.relative_dynamics_factor = _DYNAMICS_FACTOR
        self._robot.set_collision_behavior(_TORQUE_THRESHOLD, _FORCE_THRESHOLD)

        self._gripper = self._build_gripper(
            gripper_type=gripper_type,
            gripper_connection=gripper_connection,
            robot_ip=robot_ip,
        )

        # Joint and Cartesian trackers are mutually exclusive; each
        # _ensure_* stops the other before starting.
        self._tracker = None
        self._prev_target_q: Optional[np.ndarray] = None
        self._prev_target_ts: Optional[float] = None

        self._cart_tracker = None
        self._prev_cart_target_xyz: Optional[np.ndarray] = None
        self._prev_cart_target_quat: Optional[np.ndarray] = None

        self._logger.info(f"FrankyController connected to robot at {robot_ip}")

    def _build_gripper(
        self,
        gripper_type: str,
        gripper_connection: Optional[str],
        robot_ip: str,
    ):
        """Pick the right gripper for the franky/libfranka stack."""
        gt = (gripper_type or "robotiq").lower()
        if gt == "franka":
            raise NotImplementedError(
                "FrankyController: the libfranka backend for the original "
                "Franka Hand is not yet supported. Use gripper_type='robotiq' "
                "for now."
            )
        if gt == "robotiq":
            return create_gripper(gripper_type="robotiq", port=gripper_connection)
        raise ValueError(
            f"FrankyController: unsupported gripper_type={gripper_type!r}. "
            f"Supported: 'robotiq'."
        )

    def _apply_rt_hardening(self) -> None:
        """Lock memory, raise priority, pin affinity. All best-effort."""
        try:
            libc = ctypes.CDLL(
                ctypes.util.find_library("c") or "libc.so.6", use_errno=True
            )
            if libc.mlockall(_MCL_CURRENT | _MCL_FUTURE) != 0:
                self._logger.warning(f"mlockall: {os.strerror(ctypes.get_errno())}")
        except Exception as e:
            self._logger.warning(f"mlockall unavailable: {e}")
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(_RT_PRIORITY))
        except PermissionError:
            self._logger.warning(
                f"SCHED_FIFO denied; user lacks rtprio>={_RT_PRIORITY} "
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
        # join_motion re-raises latched errors from a prior motion; swallow
        # so setup/teardown can drain them and proceed.
        try:
            self._robot.join_motion()
        except Exception:
            pass

    def is_robot_up(self) -> bool:
        try:
            _ = self._robot.state
            return self._gripper.is_ready()
        except Exception:
            return False

    def get_state(self) -> FrankaRobotState:
        raw = self._robot.state
        affine = raw.O_T_EE
        # franky.Affine.quaternion is xyzw (Eigen coeffs) — same as scipy.
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
        s.gripper_position = self._gripper.position
        s.gripper_open = self._gripper.is_open
        return s

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
            stiffness=np.array(_JOINT_STIFFNESS, dtype=np.float64),
            damping=np.array(_JOINT_DAMPING, dtype=np.float64),
            compensate_coriolis=True,
        )
        self._logger.info("Joint impedance tracker started")

    def _stop_tracking_motion(self) -> None:
        if self._tracker is None:
            return
        # tracker.stop re-raises latched async reflexes (e.g. power_limit_violation).
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
        # dq feedforward is essential at 10 Hz — without it PD lags / overshoots.
        assert len(joint_positions) == 7
        q = np.clip(
            np.asarray(joint_positions, dtype=np.float64),
            JOINT_LIMITS_LOWER,
            JOINT_LIMITS_UPPER,
        )
        now = time.perf_counter()
        if self._prev_target_q is not None:
            dt = max(now - self._prev_target_ts, _DQ_MIN_DT_S)
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
        trans_clip = np.full(3, _CART_TRANS_ERROR_CLIP_M, dtype=np.float64)
        rot_clip = np.full(3, _CART_ROT_ERROR_CLIP_RAD, dtype=np.float64)
        self._cart_tracker = self._franky.CartesianImpedanceTracker(
            self._robot,
            translational_stiffness=_CART_TRANS_STIFFNESS,
            rotational_stiffness=_CART_ROT_STIFFNESS,
            translational_error_clip=trans_clip,
            rotational_error_clip=rot_clip,
            max_delta_tau=_CART_MAX_DELTA_TAU,
            gains_time_constant=_CART_GAINS_TC,
        )
        self._logger.info(
            f"Cartesian impedance tracker started "
            f"(K_t={_CART_TRANS_STIFFNESS:.0f} N/m, "
            f"K_r={_CART_ROT_STIFFNESS:.1f} Nm/rad)"
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
        # No twist feedforward: finite-diff'ing 10 Hz targets fed j7 oscillation.
        # Pose is (7,) [xyz, quat_xyzw].
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

        if _CART_MAX_STEP_M > 0:
            dxyz = xyz_in - prev_xyz
            d = float(np.linalg.norm(dxyz))
            if d > _CART_MAX_STEP_M:
                xyz = prev_xyz + dxyz * (_CART_MAX_STEP_M / d)
            else:
                xyz = xyz_in
        else:
            xyz = xyz_in

        # Hemisphere-align quat so we slerp the short arc.
        if float(np.dot(quat_in, prev_quat)) < 0.0:
            quat_in = -quat_in
        if _CART_MAX_STEP_RAD > 0:
            delta_R = R.from_quat(quat_in) * R.from_quat(prev_quat).inv()
            rotvec = delta_R.as_rotvec()
            ang = float(np.linalg.norm(rotvec))
            if ang > _CART_MAX_STEP_RAD:
                rotvec = rotvec * (_CART_MAX_STEP_RAD / ang)
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

    def reset_joint(self, reset_pos: list[float]) -> None:
        assert len(reset_pos) == 7
        self._stop_tracking_motion()
        self._stop_cart_tracking_motion()
        franky = self._franky
        motion = franky.JointMotion(
            franky.JointState(position=np.asarray(reset_pos, dtype=np.float64)),
            reference_type=franky.ReferenceType.Absolute,
        )
        self._robot.move(motion)

    def open_gripper(self) -> None:
        self._gripper.open(speed=1.0)

    def close_gripper(self) -> None:
        self._gripper.close(speed=1.0)

    def cleanup(self) -> None:
        self._stop_tracking_motion()
        self._stop_cart_tracking_motion()
        self._safe_join()
        try:
            self._gripper.cleanup()
        except Exception:
            pass
