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
and logs a warning. Without a realtime kernel, connection uses
``RealtimeConfig.Ignore`` (override with ``FRANKA_REALTIME_IGNORE=0/1``).
Used by both single-arm :class:`~rlinf.envs.realworld.franka.franka_env.FrankaEnv`
and dual-arm :class:`~rlinf.envs.realworld.franka.dual_franka_env.DualFrankaEnv`.
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

from .end_effectors import (
    EndEffector,
    EndEffectorType,
    create_end_effector,
    normalize_end_effector_type,
)
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


# Franky Cartesian impedance tuned vs stage2 ROS (serl) controller:
#   ROS uses K_t≈2000 + error clips ≈3 mm / 0.02 rad (no slew on pose pub).
#   Franky cannot take K_t=2000 on non-RT (shake); keep moderate K and
#   recover stage2 "feel" via tight error clips + slew ≈ env action_scale.
_CART_TRANS_STIFFNESS = float(os.environ.get("RLINF_CART_K_T", 1000.0))  # N/m
_CART_ROT_STIFFNESS = float(os.environ.get("RLINF_CART_K_R", 50.0))  # Nm/rad
_CART_NULLSPACE_STIFFNESS = float(os.environ.get("RLINF_CART_K_NS", 5.0))  # Nm/rad
# Cap when mapping ROS compliance (task often asks 2000/150).
_CART_K_T_CAP = float(os.environ.get("RLINF_CART_K_T_CAP", 1200.0))
_CART_K_R_CAP = float(os.environ.get("RLINF_CART_K_R_CAP", 80.0))
_CART_MAX_DELTA_TAU = float(
    os.environ.get("RLINF_CART_MAX_DTAU", 0.3)
)  # Nm / 1 kHz cycle
# Default clips ≈ stage2 peg compliance (slightly opened vs 3 mm for non-RT).
_CART_TRANS_ERROR_CLIP_M = float(os.environ.get("RLINF_CART_ERR_CLIP_M", 0.008))  # m
_CART_ROT_ERROR_CLIP_RAD = float(os.environ.get("RLINF_CART_ERR_CLIP_RAD", 0.04))  # rad
_CART_GAINS_TC = float(os.environ.get("RLINF_CART_GAINS_TC", 0.1))  # s
# Floors so ultra-tight ROS clips (3 mm) do not chatter under Ignore-RT.
_CART_TRANS_CLIP_FLOOR_M = float(os.environ.get("RLINF_CART_CLIP_FLOOR_M", 0.005))
_CART_ROT_CLIP_FLOOR_RAD = float(os.environ.get("RLINF_CART_CLIP_FLOOR_RAD", 0.02))

# Match BlockPegInsertion action_scale [0.03 m, 0.1 rad] — not PR #1439's 0.10/0.30
# (those made per-call target jumps feel like oversized pose corrections).
_CART_MAX_STEP_M = float(os.environ.get("RLINF_CART_MAX_STEP_M", 0.03))  # m / call
_CART_MAX_STEP_RAD = float(
    os.environ.get("RLINF_CART_MAX_STEP_RAD", 0.10)
)  # rad / call

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
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ):
        return FrankyController.create_group(
            robot_ip,
            end_effector_type,
            end_effector_config or {},
            gripper_type,
            gripper_connection,
        ).launch(
            cluster=Cluster(),
            placement_strategy=NodePlacementStrategy(node_ranks=[node_rank]),
            name=f"FrankyController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ):
        super().__init__()
        self._logger = get_logger()

        if not robot_ip:
            robot_ip = self._resolve_robot_ip_from_node()
        if not robot_ip:
            raise ValueError(
                "Franka 'robot_ip' is not set and could not be resolved from "
                f"node rank {self._cluster_node_rank}'s hardware infos. Provide "
                "it in the env config, the Franka hardware config, or set the "
                "'ROBOT_IP' environment variable on the controller's node."
            )
        self._robot_ip = robot_ip

        # Dual-arm callers historically pass gripper_type without
        # end_effector_type; normalization maps gripper_type over the default.
        self._end_effector_type = normalize_end_effector_type(
            end_effector_type,
            gripper_type,
        )

        # Must precede the franky import so mlockall catches its allocations.
        self._apply_rt_hardening()

        import franky

        self._franky = franky
        realtime_config = self._resolve_realtime_config(franky)
        self._robot = franky.Robot(robot_ip, realtime_config=realtime_config)
        self._robot.recover_from_errors()
        self._robot.relative_dynamics_factor = _DYNAMICS_FACTOR
        self._robot.set_collision_behavior(_TORQUE_THRESHOLD, _FORCE_THRESHOLD)

        self._gripper = None
        self._end_effector: EndEffector | None = None
        self._init_end_effector(
            end_effector_config or {},
            gripper_connection,
            robot_ip,
        )

        # Joint and Cartesian trackers are mutually exclusive; each
        # _ensure_* stops the other before starting.
        self._tracker = None
        self._prev_target_q: Optional[np.ndarray] = None
        self._prev_target_ts: Optional[float] = None

        self._cart_tracker = None
        self._prev_cart_target_xyz: Optional[np.ndarray] = None
        self._prev_cart_target_quat: Optional[np.ndarray] = None
        # Runtime gains/clips (defaults; updated by reconfigure_compliance_params).
        self._cart_k_t = _CART_TRANS_STIFFNESS
        self._cart_k_r = _CART_ROT_STIFFNESS
        self._cart_k_ns = _CART_NULLSPACE_STIFFNESS
        self._cart_trans_clip = np.full(3, _CART_TRANS_ERROR_CLIP_M, dtype=np.float64)
        self._cart_rot_clip = np.full(3, _CART_ROT_ERROR_CLIP_RAD, dtype=np.float64)

        self._logger.info(
            "FrankyController connected to robot at %s (end_effector=%s)",
            robot_ip,
            self._end_effector_type.value,
        )

    def _resolve_realtime_config(self, franky_mod):
        """Pick Enforce vs Ignore for libfranka realtime checks.

        ``FRANKA_REALTIME_IGNORE=1`` forces Ignore; ``=0`` forces Enforce.
        If unset, Ignore when ``/sys/kernel/realtime`` is missing or not ``1``.
        """
        env = os.environ.get("FRANKA_REALTIME_IGNORE")
        if env is not None:
            ignore = env.strip().lower() in ("1", "true", "yes", "on")
        else:
            try:
                with open("/sys/kernel/realtime", encoding="ascii") as f:
                    ignore = f.read().strip() != "1"
            except OSError:
                ignore = True

        if ignore:
            self._logger.warning(
                "Connecting with franky.RealtimeConfig.Ignore "
                "(non-RT kernel or FRANKA_REALTIME_IGNORE); "
                "1 kHz control may be unstable until PREEMPT_RT is installed"
            )
            return franky_mod.RealtimeConfig.Ignore
        return franky_mod.RealtimeConfig.Enforce

    def _resolve_robot_ip_from_node(self) -> Optional[str]:
        """Return the first ``robot_ip`` in this node's hardware infos, if any."""
        try:
            node_info = Cluster().get_node_info(self._cluster_node_rank)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.warning(
                "Could not access node info to resolve robot_ip: %s", exc
            )
            return None
        for resource in node_info.hardware_resources:
            for info in resource.infos:
                config = getattr(info, "config", None)
                if config is None:
                    continue
                robot_ip = getattr(config, "robot_ip", None)
                if robot_ip:
                    return robot_ip
                # Dual-arm hw configs expose left/right IPs.
                for attr in ("left_robot_ip", "right_robot_ip"):
                    robot_ip = getattr(config, attr, None)
                    if robot_ip:
                        return robot_ip
        return None

    def _init_end_effector(
        self,
        end_effector_config: dict,
        gripper_connection: Optional[str],
        robot_ip: str,
    ) -> None:
        if self._end_effector_type.is_gripper:
            self._gripper = create_gripper(
                gripper_type=self._end_effector_type.gripper_backend,
                port=gripper_connection,
                robot_ip=robot_ip,
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
            if self._end_effector_type.is_gripper:
                return self._gripper.is_ready()
            return True
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
        if self._end_effector_type.is_gripper:
            s.gripper_position = self._gripper.position
            s.gripper_open = self._gripper.is_open
            s.hand_position = None
        else:
            assert self._end_effector is not None
            s.hand_position = self._end_effector.get_state()
        return s

    def clear_errors(self) -> None:
        self._robot.recover_from_errors()

    def reconfigure_compliance_params(self, params: dict) -> None:
        """Map stage2 ROS compliance onto franky with safe caps.

        Stage2 ``franka_controller`` applies task ``compliance_param`` fully
        (K_t≈2000 + ~3 mm clips). On franky we:
          - **take error clips** (what limited ROS pose correction), floored
            for non-RT stability;
          - **cap stiffness** so high ROS gains do not shake under
            ``RealtimeConfig.Ignore``.
        """
        if not params:
            return

        def _f(key: str, default: float) -> float:
            val = params.get(key, default)
            return float(default if val is None else val)

        k_t = min(_f("translational_stiffness", self._cart_k_t), _CART_K_T_CAP)
        k_r = min(_f("rotational_stiffness", self._cart_k_r), _CART_K_R_CAP)
        k_ns = _f("nullspace_stiffness", self._cart_k_ns)

        trans_clip = np.maximum(
            np.array(
                [
                    max(
                        _f("translational_clip_x", self._cart_trans_clip[0]),
                        _f("translational_clip_neg_x", self._cart_trans_clip[0]),
                    ),
                    max(
                        _f("translational_clip_y", self._cart_trans_clip[1]),
                        _f("translational_clip_neg_y", self._cart_trans_clip[1]),
                    ),
                    max(
                        _f("translational_clip_z", self._cart_trans_clip[2]),
                        _f("translational_clip_neg_z", self._cart_trans_clip[2]),
                    ),
                ],
                dtype=np.float64,
            ),
            _CART_TRANS_CLIP_FLOOR_M,
        )
        rot_clip = np.maximum(
            np.array(
                [
                    max(
                        _f("rotational_clip_x", self._cart_rot_clip[0]),
                        _f("rotational_clip_neg_x", self._cart_rot_clip[0]),
                    ),
                    max(
                        _f("rotational_clip_y", self._cart_rot_clip[1]),
                        _f("rotational_clip_neg_y", self._cart_rot_clip[1]),
                    ),
                    max(
                        _f("rotational_clip_z", self._cart_rot_clip[2]),
                        _f("rotational_clip_neg_z", self._cart_rot_clip[2]),
                    ),
                ],
                dtype=np.float64,
            ),
            _CART_ROT_CLIP_FLOOR_RAD,
        )

        clips_changed = not (
            np.allclose(trans_clip, self._cart_trans_clip)
            and np.allclose(rot_clip, self._cart_rot_clip)
        )
        self._cart_k_t = k_t
        self._cart_k_r = k_r
        self._cart_k_ns = k_ns
        self._cart_trans_clip = trans_clip
        self._cart_rot_clip = rot_clip

        if self._cart_tracker is not None and clips_changed:
            self._stop_cart_tracking_motion()
            self._logger.info(
                "Cartesian clips updated from compliance_param "
                "(trans=%s, rot=%s); tracker will rebuild on next move",
                np.array2string(trans_clip, precision=4),
                np.array2string(rot_clip, precision=4),
            )
            return

        if self._cart_tracker is not None:
            self._cart_tracker.set_gains(
                translational_stiffness=k_t,
                rotational_stiffness=k_r,
                nullspace_stiffness=k_ns,
            )

        self._logger.info(
            "Franky compliance: K_t=%.0f (cap %.0f), K_r=%.1f (cap %.1f), "
            "K_ns=%.1f, trans_clip=%s, rot_clip=%s, step=%.2fm/%.2frad",
            k_t,
            _CART_K_T_CAP,
            k_r,
            _CART_K_R_CAP,
            k_ns,
            np.array2string(trans_clip, precision=4),
            np.array2string(rot_clip, precision=4),
            _CART_MAX_STEP_M,
            _CART_MAX_STEP_RAD,
        )

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
        nullspace_target = np.asarray(self._robot.state.q, dtype=np.float64).copy()
        self._cart_tracker = self._franky.CartesianImpedanceTracker(
            self._robot,
            translational_stiffness=self._cart_k_t,
            rotational_stiffness=self._cart_k_r,
            nullspace_target=nullspace_target,
            nullspace_stiffness=self._cart_k_ns,
            translational_error_clip=np.asarray(self._cart_trans_clip, dtype=np.float64),
            rotational_error_clip=np.asarray(self._cart_rot_clip, dtype=np.float64),
            max_delta_tau=_CART_MAX_DELTA_TAU,
            gains_time_constant=_CART_GAINS_TC,
        )
        self._logger.info(
            "Cartesian impedance tracker started "
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

    def move_arm(self, position: np.ndarray) -> None:
        """Compatibility alias for :meth:`move_tcp_pose` (FrankaEnv API)."""
        self.move_tcp_pose(position)

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

    def command_end_effector(self, action: np.ndarray) -> bool:
        """Send an action to the active end-effector."""
        if self._end_effector_type.is_gripper:
            value = float(np.asarray(action).reshape(-1)[0])
            # Read open state without a full get_state round-trip.
            gripper_open = bool(self._gripper.is_open)
            if value <= -0.5 and gripper_open:
                self.close_gripper()
                return True
            if value >= 0.5 and not gripper_open:
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

    def open_gripper(self) -> None:
        if self._end_effector_type.is_gripper:
            self._gripper.open(speed=1.0)
        self._logger.debug("Open gripper")

    def close_gripper(self) -> None:
        if self._end_effector_type.is_gripper:
            self._gripper.close(speed=1.0)
        self._logger.debug("Close gripper")

    def move_gripper(self, position: int, speed: float = 0.3) -> None:
        assert 0 <= position <= 255, (
            f"Invalid gripper position {position}, must be between 0 and 255"
        )
        if self._end_effector_type.is_gripper:
            self._gripper.move(position, speed)
        self._logger.debug(f"Move gripper to position: {position}")

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

    def cleanup(self) -> None:
        self._stop_tracking_motion()
        self._stop_cart_tracking_motion()
        self._safe_join()
        if self._end_effector_type.is_gripper:
            try:
                self._gripper.cleanup()
            except Exception:
                pass
        elif self._end_effector is not None:
            try:
                self._end_effector.shutdown()
            except Exception:
                pass
