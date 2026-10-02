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

"""Offline i2rt FK/IK for YAM with its actual flexible gripper model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import YamIKConfig


@dataclass(frozen=True)
class YamIKResult:
    """A checked six-joint solution; failed solutions must never be dispatched."""

    success: bool
    q_target: np.ndarray
    position_error: float
    rotation_error: float
    elapsed_s: float
    reason: str | None = None


class YamKinematicsAdapter:
    """Own one independent i2rt configuration without opening any hardware.

    V1 deliberately supports the verified YAM + flexible_4310 model only.
    Its grasp_site is attached before the two finger slides, whose normalized
    positions map linearly to their XML ranges and are frozen throughout IK.
    """

    def __init__(
        self,
        *,
        arm_type: str = "yam",
        gripper_type: str = "flexible_4310",
        config: YamIKConfig | None = None,
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
    ) -> None:
        if (arm_type, gripper_type) != ("yam", "flexible_4310"):
            raise ValueError(
                "YAM VR v1 requires yam + flexible_4310; other models need explicit TCP/joint validation"
            )
        import mink
        import mujoco
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

        self.config = config or YamIKConfig()
        xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.FLEXIBLE_4310)
        try:
            self._kin = Kinematics(xml_path, self.config.site_name)
        finally:
            # This is the exact temporary XML returned by our SDK invocation.
            Path(xml_path).unlink(missing_ok=True)
        # The pinned i2rt version exposes its MuJoCo model only through this
        # configuration. Keep that dependency confined to this adapter.
        self.model = self._kin._configuration.model
        names = [self.model.joint(i).name for i in range(self.model.njnt)]
        arm_names = [f"joint{i}" for i in range(1, 7)]
        finger_names = ["joint7", "joint8"]
        if (
            set(names) != set(arm_names + finger_names)
            or self.model.nq != 8
            or self.model.nv != 8
        ):
            raise ValueError(f"Unsupported YAM model joint layout: {names}")
        self._arm_ids = np.array([self.model.joint(n).id for n in arm_names])
        self._finger_ids = np.array([self.model.joint(n).id for n in finger_names])
        if np.any(
            self.model.jnt_type[self._arm_ids] != mujoco.mjtJoint.mjJNT_HINGE
        ) or np.any(
            self.model.jnt_type[self._finger_ids] != mujoco.mjtJoint.mjJNT_SLIDE
        ):
            raise ValueError("YAM model must have six hinges and two finger slides")
        self._arm_q = self.model.jnt_qposadr[self._arm_ids]
        self._finger_q = self.model.jnt_qposadr[self._finger_ids]
        self.lower = self.model.jnt_range[self._arm_ids, 0].copy()
        self.upper = self.model.jnt_range[self._arm_ids, 1].copy()
        if joint_lower is not None:
            self.lower = np.maximum(self.lower, self._vector(joint_lower))
        if joint_upper is not None:
            self.upper = np.minimum(self.upper, self._vector(joint_upper))
        if np.any(self.lower >= self.upper):
            raise ValueError("YAM configured and model joint limits do not overlap")
        self.model.jnt_range[self._arm_ids, 0] = self.lower
        self.model.jnt_range[self._arm_ids, 1] = self.upper
        self.model.site(self.config.site_name)  # Validate before any CAN startup.
        self._limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model, dict.fromkeys(finger_names, 0.0)),
        ]

    @staticmethod
    def _vector(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("YAM kinematics requires six finite joint positions")
        return q

    def _qpos(self, q: np.ndarray, gripper_position: float) -> np.ndarray:
        q = self._vector(q)
        if not np.isfinite(gripper_position) or not 0 <= gripper_position <= 1:
            raise ValueError("YAM gripper position must be finite and within [0, 1]")
        full = self.model.qpos0.copy()
        full[self._arm_q] = q
        ranges = self.model.jnt_range[self._finger_ids]
        full[self._finger_q] = ranges[:, 0] + gripper_position * (
            ranges[:, 1] - ranges[:, 0]
        )
        return full

    def fk(self, q: np.ndarray, gripper_position: float) -> np.ndarray:
        """Return a 4x4 TCP pose in the model's fixed robot-base frame."""
        return self._kin.fk(self._qpos(q, gripper_position)).copy()

    def solve(
        self, target: np.ndarray, q_seed: np.ndarray, gripper_position: float
    ) -> YamIKResult:
        """Solve and verify IK, including frozen fingers and joint limits.

        The control runtime limits command slew relative to measured joints.
        max_solve_s is a post-return rejection threshold, not a preemptive
        deadline. The underlying SDK call is bounded by max_iters.
        """
        started = time.monotonic()
        q_seed = self._vector(q_seed).copy()
        target = np.asarray(target, dtype=np.float64)
        reason = None
        pos_error = rot_error = float("inf")
        q_result = q_seed.copy()
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("IK target must be a finite 4x4 transform")
        if (
            not np.allclose(target[3], [0, 0, 0, 1])
            or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(target[:3, :3]), 1.0)
        ):
            raise ValueError("IK target must contain a valid rigid rotation")
        initial = self._qpos(q_seed, gripper_position)
        if np.any(q_seed < self.lower) or np.any(q_seed > self.upper):
            return YamIKResult(
                False,
                q_result,
                pos_error,
                rot_error,
                time.monotonic() - started,
                "seed_out_of_limits",
            )
        try:
            ok, solved = self._kin.ik(
                target,
                self.config.site_name,
                init_q=initial,
                limits=self._limits,
                solver=self.config.solver,
                dt=self.config.dt,
                max_iters=self.config.max_iters,
                pos_threshold=self.config.position_tolerance,
                ori_threshold=self.config.rotation_tolerance,
            )
            solved = np.asarray(solved).copy()
            if solved.shape != initial.shape or not np.all(np.isfinite(solved)):
                reason = "invalid_solution"
            elif not ok:
                reason = "not_converged"
            elif not np.allclose(
                solved[self._finger_q], initial[self._finger_q], atol=1e-8, rtol=0
            ):
                reason = "finger_motion"
            else:
                q_result = solved[self._arm_q].copy()
                actual = self.fk(q_result, gripper_position)
                pos_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                rot_error = float(
                    Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
                )
                if np.any(q_result < self.lower) or np.any(q_result > self.upper):
                    reason = "joint_limits"
                elif (
                    pos_error > self.config.position_tolerance
                    or rot_error > self.config.rotation_tolerance
                ):
                    reason = "residual"
        except Exception as exc:
            reason = f"solver_error:{type(exc).__name__}"
        elapsed = time.monotonic() - started
        if elapsed > self.config.max_solve_s:
            reason = "solve_timeout"
        return YamIKResult(
            reason is None, q_result, pos_error, rot_error, elapsed, reason
        )
