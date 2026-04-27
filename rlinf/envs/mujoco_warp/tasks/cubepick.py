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

"""Franka cube-pick task for MuJoCo-Warp GPU backend.

Mirrors the MuJoCo CubePickTask exactly so both physics backends can be
compared fairly:
  - Same Franka Panda robot model (from MuJoCo Menagerie)
  - Same home qpos, cube spawn range, and success height
  - Same reward function and coefficients
  - Same EEF offset and DLS IK parameters
  - Same control period: timestep=0.005s × n_substeps=8 → 0.04s/step

MuJoCo-Warp difference:
  - All num_envs worlds are stepped in parallel on GPU via mjw.step()
  - IK Jacobians are computed on GPU via mjw.jac() (batched across all worlds)
    and solved in a single batched solve_dls_ik_torch call on CUDA
  - The CUDA graph captures step(s) for zero kernel-launch overhead
  - Reset settle physics runs via a separate CUDA graph

State (32 dims):
  qpos(9) + qvel(9) + eef_pos(3) + eef_quat(4) + cube_pos(3) + cube_quat(4)

Action (7 dims):
  action[0:3] → TCP position delta in world frame
  action[3:6] → TCP rotation delta in axis-angle form
  action[6]   → binary gripper open / close command
"""

from __future__ import annotations

import os
from typing import Any, Optional

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from rlinf.envs.ee_dls_ik import (
    axis_angle_to_quat_torch,
    decode_delta_action_torch,
    quat_multiply_torch,
    solve_dls_ik_torch,
)
from rlinf.envs.mujoco_warp.mujoco_warp_env import MuJoCoWarpEnv

# ---------------------------------------------------------------------------
# Default paths and constants
# ---------------------------------------------------------------------------
_DEFAULT_PANDA_XML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "franka_emika_panda",
    "panda.xml",
)

_HOME_QPOS = np.array(
    [0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04], dtype=np.float64
)

_CUBE_HALF_SIZE = 0.02
_CUBE_DEFAULT_X = 0.65
_CUBE_DEFAULT_Y = 0.0
_CUBE_Z = _CUBE_HALF_SIZE

_DEFAULT_CUBE_X_RANGE = (0.45, 0.65)
_DEFAULT_CUBE_Y_RANGE = (-0.25, 0.25)
_SUCCESS_HEIGHT = 0.15
_SUCCESS_BONUS = 10.0
_LIFT_REWARD_WEIGHT = 4.0
_FINGER_PAD_OFFSET = np.array([0.0, 0.0055, 0.0445], dtype=np.float64)
_PRE_GRASP_PAD_SPAN_TARGET = 0.05
_PRE_GRASP_PAD_SPAN_STD = 0.02

_EEF_OFFSET = np.array([0.0, 0.0, 0.105], dtype=np.float64)

_DEFAULT_TIMESTEP = 0.005
_DEFAULT_N_SUBSTEPS = 8
_RESET_SETTLE_CONTROL_STEPS = 5

_TASK_DESCRIPTION = "Pick up the red cube and lift it."

# Body names for self-collision exclusion
_ARM_BODIES = [
    "link0",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger",
]

# Gripper and wrist actuator tuning
_GRIPPER_SCALE = 4.0
_WRIST_KP = 2500.0
_WRIST_KD = 250.0

# Grasp contact tuning
_GRASP_FRICTION = np.array([5.0, 0.2, 0.02])
_GRASP_SOLREF = np.array([0.02, 1.0])
_GRASP_SOLIMP = np.array([0.99, 0.999, 0.001, 0.5, 2.0])

# Ctrl smoothing
_CTRL_SMOOTHING_ALPHA = 0.75


class CubePickTask(MuJoCoWarpEnv):
    """Franka cube-pick task backed by MuJoCo-Warp (GPU-batched physics)."""

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    @property
    def _task_description(self) -> str:
        return _TASK_DESCRIPTION

    @property
    def _timestep(self) -> float:
        return self._init_params.get("timestep", _DEFAULT_TIMESTEP)

    @property
    def _n_substeps(self) -> int:
        return int(self._init_params.get("n_substeps", _DEFAULT_N_SUBSTEPS))

    @property
    def _mjcf_path(self) -> str:
        return str(self._init_params.get("panda_xml_path", _DEFAULT_PANDA_XML))

    @property
    def _cast_shadows(self) -> bool:
        return True

    @property
    def _render_enabled_geom_groups(self) -> list[int]:
        return [0, 2]

    @property
    def _use_reset_settle(self) -> bool:
        return True

    @property
    def _reset_settle_steps(self) -> int:
        reset_settle_control_steps = self._init_params.get("reset_settle_control_steps")
        if reset_settle_control_steps is not None:
            return int(reset_settle_control_steps) * self._n_substeps
        return int(
            self._init_params.get(
                "reset_settle_steps",
                _RESET_SETTLE_CONTROL_STEPS * self._n_substeps,
            )
        )

    @property
    def _track_reward_terms(self) -> bool:
        return True

    @property
    def _njmax(self) -> Optional[int]:
        return 2048

    # ------------------------------------------------------------------
    # Graph construction — capture step-only (jac computed fresh each step)
    # ------------------------------------------------------------------

    def _build_step_graph(self) -> Optional[wp.Graph]:
        if not wp.get_device().is_cuda:
            return None
        wp.copy(
            self._jac_body,
            wp.full(self.num_envs, self._hand_id, dtype=wp.int32),
        )
        with wp.ScopedCapture() as capture:
            for _ in range(self._n_substeps):
                mjw.step(self._mw_model, self._mw_data)
        return capture.graph

    def _build_reset_graph(self) -> Optional[wp.Graph]:
        if not wp.get_device().is_cuda:
            return None
        with wp.ScopedCapture() as capture:
            for _ in range(self._reset_settle_steps):
                mjw.step(self._mw_model, self._mw_data)
        return capture.graph

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self, spec: mujoco.MjSpec, wb) -> None:
        """Add floor, cube, self-collision excludes, remove finger equality."""
        floor = wb.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [2.0, 2.0, 0.1]
        floor.rgba = [0.3, 0.3, 0.4, 1.0]

        cube_body = wb.add_body()
        cube_body.name = "cube"
        cube_body.pos = [_CUBE_DEFAULT_X, _CUBE_DEFAULT_Y, _CUBE_Z]
        cube_jnt = cube_body.add_freejoint()
        cube_jnt.name = "cube_free"
        cube_geom = cube_body.add_geom()
        cube_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        cube_geom.size = [_CUBE_HALF_SIZE, _CUBE_HALF_SIZE, _CUBE_HALF_SIZE]
        cube_geom.rgba = [1.0, 0.3, 0.3, 1.0]
        cube_geom.mass = 0.05

        for i, b1 in enumerate(_ARM_BODIES):
            for b2 in _ARM_BODIES[i + 1 :]:
                ex = spec.add_exclude()
                ex.bodyname1 = b1
                ex.bodyname2 = b2

        for eq in list(spec.equalities):
            spec.delete(eq)

    def _configure_model_cpu(self) -> None:
        """Configure solver, actuator gains, contact params for grasping."""
        self._model_cpu.opt.impratio = 20.0

        # Boost gripper stiffness
        _gripper_act = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self._model_cpu.actuator_gainprm[_gripper_act, 0] *= _GRIPPER_SCALE
        self._model_cpu.actuator_biasprm[_gripper_act, 1] *= _GRIPPER_SCALE
        self._model_cpu.actuator_biasprm[_gripper_act, 2] *= _GRIPPER_SCALE
        self._model_cpu.actuator_forcerange[_gripper_act] = [
            -100.0 * _GRIPPER_SCALE,
            100.0 * _GRIPPER_SCALE,
        ]

        # Boost wrist joint PD stiffness
        for jidx, jname in enumerate(["joint5", "joint6", "joint7"]):
            aid = mujoco.mj_name2id(
                self._model_cpu, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{jidx + 5}"
            )
            self._model_cpu.actuator_gainprm[aid, 0] = _WRIST_KP
            self._model_cpu.actuator_biasprm[aid, 0] = 0.0
            self._model_cpu.actuator_biasprm[aid, 1] = -_WRIST_KP
            self._model_cpu.actuator_biasprm[aid, 2] = -_WRIST_KD

        # Contact tuning for reliable grasping
        for gid in range(self._model_cpu.ngeom):
            body = self._model_cpu.geom_bodyid[gid]
            if body == self._cube_id or body in (self._lf_id, self._rf_id):
                self._model_cpu.geom_friction[gid] = _GRASP_FRICTION
                self._model_cpu.geom_condim[gid] = 6
                self._model_cpu.geom_solref[gid] = _GRASP_SOLREF
                self._model_cpu.geom_solimp[gid] = _GRASP_SOLIMP

        floor_gid = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._model_cpu.geom_friction[floor_gid] = _GRASP_FRICTION
        self._model_cpu.opt.iterations = 200

    def _init_task_state(self) -> None:
        """Cache body/joint IDs, IK parameters, allocate Jacobian buffers."""
        self._hand_id = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        self._lf_id = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self._rf_id = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )
        self._cube_id = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )

        self._arm_joint_ids = np.array(
            [
                mujoco.mj_name2id(
                    self._model_cpu, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"
                )
                for i in range(1, 8)
            ],
            dtype=np.int32,
        )
        self._arm_qpos_idx = np.array(
            [self._model_cpu.jnt_qposadr[jid] for jid in self._arm_joint_ids],
            dtype=np.int32,
        )
        self._arm_dof_idx = np.array(
            [self._model_cpu.jnt_dofadr[jid] for jid in self._arm_joint_ids],
            dtype=np.int32,
        )
        self._arm_qpos_lower = self._model_cpu.jnt_range[self._arm_joint_ids, 0].copy()
        self._arm_qpos_upper = self._model_cpu.jnt_range[self._arm_joint_ids, 1].copy()

        _lf_jnt = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"
        )
        _rf_jnt = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"
        )
        self._lf_qpos_adr = int(self._model_cpu.jnt_qposadr[_lf_jnt])
        self._rf_qpos_adr = int(self._model_cpu.jnt_qposadr[_rf_jnt])

        # IK params
        self._lock_eef_orientation: bool = bool(
            self._init_params.get("lock_eef_orientation", True)
        )
        self._action_pos_scale: float = float(
            self._init_params.get("action_pos_scale", 0.04)
        )
        self._action_rot_scale: float = float(
            self._init_params.get("action_rot_scale", 0.35)
        )
        self._ik_damping: float = float(self._init_params.get("ik_damping", 0.1))
        self._ik_integration_dt: float = float(
            self._init_params.get("ik_integration_dt", 0.04)
        )
        self._ik_max_dq: float = float(self._init_params.get("ik_max_dq", 2.5))
        self._ik_pos_gain: float = float(self._init_params.get("ik_pos_gain", 0.95))
        self._ik_rot_gain: float = float(self._init_params.get("ik_rot_gain", 0.95))

        # Task params
        self._success_height: float = float(
            self._init_params.get("success_height", _SUCCESS_HEIGHT)
        )
        self._cube_x_range: tuple[float, float] = tuple(
            self._init_params.get("cube_x_range", _DEFAULT_CUBE_X_RANGE)
        )
        self._cube_y_range: tuple[float, float] = tuple(
            self._init_params.get("cube_y_range", _DEFAULT_CUBE_Y_RANGE)
        )

        # Ctrl smoothing state
        self._prev_ctrl = np.zeros(
            (self.num_envs, self._model_cpu.nu), dtype=np.float32
        )

        # Preferred EEF orientation  (lock_eef_orientation)
        self._preferred_eef_quat = np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (self.num_envs, 1)
        )
        self._home_eef_quat: Optional[np.ndarray] = None

        # Pre-allocated Warp arrays for batched Jacobian
        nv = self._model_cpu.nv
        self._jacp = wp.zeros((self.num_envs, 3, nv), dtype=wp.float32)
        self._jacr = wp.zeros((self.num_envs, 3, nv), dtype=wp.float32)
        self._jac_point = wp.zeros(self.num_envs, dtype=wp.vec3)
        self._jac_body = wp.zeros(self.num_envs, dtype=wp.int32)

        # Reward term tracking
        self._last_reward_terms: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _post_step_hook(self) -> None:
        """No-op: forward is not captured in step graph for CubePick;
        Jacobian is computed fresh each control step.  However, obs/reward
        reading will call numpy() which implicitly syncs and reads current
        state.  We still need forward to update xpos/xquat for reading."""
        mjw.forward(self._mw_model, self._mw_data)
        wp.synchronize()

    def _post_reset_hook(self) -> None:
        """Sync prev_ctrl with current GPU ctrl state after reset."""
        self._prev_ctrl = self._mw_data.ctrl.numpy().copy()

    # ------------------------------------------------------------------
    # Control — batched DLS IK
    # ------------------------------------------------------------------

    def _compute_ctrl(self, actions: np.ndarray) -> np.ndarray:
        """Compute ctrl for all envs via batched DLS IK.

        Computes a fresh Jacobian at the current post-step EEF position each
        call, then solves DLS IK via a single batched ``solve_dls_ik_torch``
        call on CUDA — no per-env loop.
        """
        nu = self._model_cpu.nu

        xpos_all = self._mw_data.xpos.numpy()
        xquat_all = self._mw_data.xquat.numpy()

        hand_xpos = xpos_all[:, self._hand_id, :]
        hand_xquat = xquat_all[:, self._hand_id, :]

        eef_pos = self._batch_eef_offset(hand_xpos, hand_xquat)

        wp.copy(self._jac_point, wp.array(eef_pos, dtype=wp.vec3))
        mjw.jac(
            self._mw_model,
            self._mw_data,
            self._jacp,
            self._jacr,
            self._jac_point,
            self._jac_body,
        )
        wp.synchronize()

        jacp_np = self._jacp.numpy()
        jacr_np = self._jacr.numpy()
        arm_jacp = jacp_np[:, :, self._arm_dof_idx]
        arm_jacr = jacr_np[:, :, self._arm_dof_idx]
        jacobian = np.concatenate([arm_jacp, arm_jacr], axis=1)

        actions_t = torch.from_numpy(actions).float().cuda()
        pos_delta, rot_delta, gripper = decode_delta_action_torch(
            actions_t,
            pos_scale=self._action_pos_scale,
            rot_scale=self._action_rot_scale,
        )

        current_pos = torch.from_numpy(eef_pos).float().cuda()
        current_quat = torch.from_numpy(hand_xquat).float().cuda()
        target_pos = current_pos + pos_delta
        if self._lock_eef_orientation:
            target_quat = torch.from_numpy(
                self._preferred_eef_quat.astype(np.float32)
            ).cuda()
        else:
            target_quat = quat_multiply_torch(
                axis_angle_to_quat_torch(rot_delta), current_quat
            )

        current_q = torch.from_numpy(
            self._mw_data.qpos.numpy()[:, self._arm_qpos_idx].astype(np.float32)
        ).cuda()
        jacobian_t = torch.from_numpy(jacobian.astype(np.float32)).cuda()

        arm_targets, _ = solve_dls_ik_torch(
            jacobian=jacobian_t,
            current_q=current_q,
            current_pos=current_pos,
            current_quat=current_quat,
            target_pos=target_pos,
            target_quat=target_quat,
            damping=self._ik_damping,
            integration_dt=self._ik_integration_dt,
            max_dq=self._ik_max_dq,
            joint_lower=torch.from_numpy(
                self._arm_qpos_lower.astype(np.float32)
            ).cuda(),
            joint_upper=torch.from_numpy(
                self._arm_qpos_upper.astype(np.float32)
            ).cuda(),
            pos_gain=self._ik_pos_gain,
            rot_gain=self._ik_rot_gain,
        )

        ctrl_out = np.zeros((self.num_envs, nu), dtype=np.float32)
        ctrl_out[:, :7] = arm_targets.cpu().numpy()
        ctrl_out[:, 7] = np.where(actions[:, 6] > 0.0, 255.0, 0.0).astype(np.float32)

        ctrl_out = (
            _CTRL_SMOOTHING_ALPHA * ctrl_out
            + (1.0 - _CTRL_SMOOTHING_ALPHA) * self._prev_ctrl
        )
        self._prev_ctrl = ctrl_out.copy()
        return ctrl_out

    def _batch_eef_offset(
        self, hand_xpos: np.ndarray, hand_xquat: np.ndarray
    ) -> np.ndarray:
        """Compute EEF positions with offset for all envs (batched)."""
        q_w = hand_xquat[:, 0]
        q_xyz = hand_xquat[:, 1:]
        t = _EEF_OFFSET
        cross = np.cross(q_xyz, t)
        eef_offset = t + 2.0 * q_w[:, None] * cross + 2.0 * np.cross(q_xyz, cross)
        return (hand_xpos + eef_offset).astype(np.float32)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs_dict(self) -> dict[str, Any]:
        """Build 32-D state observation from GPU batched arrays."""
        qpos_np = self._mw_data.qpos.numpy()
        qvel_np = self._mw_data.qvel.numpy()
        xpos_np = self._mw_data.xpos.numpy()
        xquat_np = self._mw_data.xquat.numpy()

        hand_xquat = xquat_np[:, self._hand_id, :]
        hand_xpos = xpos_np[:, self._hand_id, :]
        eef_pos = self._batch_eef_offset(hand_xpos, hand_xquat)
        eef_quat = hand_xquat.astype(np.float32)

        cube_pos = xpos_np[:, self._cube_id, :].astype(np.float32)
        cube_quat = xquat_np[:, self._cube_id, :].astype(np.float32)

        states_np = np.concatenate(
            [
                qpos_np[:, :9].astype(np.float32),
                qvel_np[:, :9].astype(np.float32),
                eef_pos,
                eef_quat,
                cube_pos,
                cube_quat,
            ],
            axis=1,
        )
        states = torch.from_numpy(states_np)

        obs: dict[str, Any] = {
            "states": states,
            "task_descriptions": [self._task_description] * self.num_envs,
        }

        if getattr(self.video_cfg, "save_video", False):
            imgs = self._render_images()
            if imgs is not None:
                obs["main_images"] = torch.from_numpy(imgs)

        return obs

    # ------------------------------------------------------------------
    # Reward and success
    # ------------------------------------------------------------------

    def _compute_reward_and_done(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute dense reward and success flag for all envs.

        Fully vectorised over the batch.  Contact detection uses real contact
        forces from ``mjw.contact_force``.  Lift-only variant: success is
        defined by cube height exceeding a threshold.
        """
        xpos_all = self._mw_data.xpos.numpy().astype(np.float64)
        xquat_all = self._mw_data.xquat.numpy().astype(np.float64)

        def _rotated_offset(body_id: int, offset: np.ndarray) -> np.ndarray:
            bpos = xpos_all[:, body_id, :]
            bquat = xquat_all[:, body_id, :]
            q_w = bquat[:, 0]
            q_xyz = bquat[:, 1:]
            cross = np.cross(q_xyz, offset)
            world_off = (
                offset + 2.0 * q_w[:, None] * cross + 2.0 * np.cross(q_xyz, cross)
            )
            return bpos + world_off

        lf_pad_pos = _rotated_offset(self._lf_id, _FINGER_PAD_OFFSET)
        rf_pad_pos = _rotated_offset(self._rf_id, _FINGER_PAD_OFFSET)
        cube_pos = xpos_all[:, self._cube_id, :]

        # Reaching reward
        pinch_mid = 0.5 * (lf_pad_pos + rf_pad_pos)
        reach_dist = np.linalg.norm(pinch_mid - cube_pos, axis=1)
        reaching_reward = 1.0 - np.tanh(5.0 * reach_dist)

        # Pinch geometry
        cube_to_lf = cube_pos - lf_pad_pos
        cube_to_rf = cube_pos - rf_pad_pos
        lf_dir = cube_to_lf / np.linalg.norm(cube_to_lf, axis=1, keepdims=True).clip(
            1e-6
        )
        rf_dir = cube_to_rf / np.linalg.norm(cube_to_rf, axis=1, keepdims=True).clip(
            1e-6
        )
        opposition_cos = np.sum(lf_dir * rf_dir, axis=1)
        opposition_score = np.clip((-opposition_cos + 1.0) * 0.5, 0.0, 1.0)
        midpoint_dist = np.linalg.norm(cube_pos - pinch_mid, axis=1)
        midpoint_score = 1.0 - np.tanh(12.0 * midpoint_dist)
        pad_span = np.linalg.norm(rf_pad_pos - lf_pad_pos, axis=1)
        span_score = np.exp(
            -(((pad_span - _PRE_GRASP_PAD_SPAN_TARGET) / _PRE_GRASP_PAD_SPAN_STD) ** 2)
        )
        pinch_score = opposition_score * midpoint_score * span_score

        # Contact forces
        lf_forces, rf_forces = self._read_finger_contact_forces()
        lf_contact_smooth = np.tanh(lf_forces / 0.05)
        rf_contact_smooth = np.tanh(rf_forces / 0.05)
        both_contact = lf_contact_smooth * rf_contact_smooth
        contact_score = np.tanh(np.minimum(lf_forces, rf_forces) / 5.0)

        grasp_quality = both_contact * contact_score * pinch_score

        # Dense rewards
        pinch_reward = 2.0 * pinch_score
        grasp_reward = 4.0 * grasp_quality

        cube_z = cube_pos[:, 2]
        lift_progress = np.clip(
            (cube_z - _CUBE_Z) / (self._success_height - _CUBE_Z), 0.0, 1.0
        )
        grasp_gate = both_contact * contact_score
        lift_reward = _LIFT_REWARD_WEIGHT * lift_progress * grasp_gate

        ep_success = cube_z > self._success_height
        success_bonus = _SUCCESS_BONUS * ep_success.astype(np.float64)

        rewards = (
            reaching_reward + pinch_reward + grasp_reward + lift_reward + success_bonus
        ).astype(np.float32)

        self._last_reward_terms = {
            "reward/reaching": reaching_reward.astype(np.float32),
            "reward/pinch": pinch_reward.astype(np.float32),
            "reward/grasp": grasp_reward.astype(np.float32),
            "reward/lift": lift_reward.astype(np.float32),
            "reward/success_bonus": success_bonus.astype(np.float32),
            "diagnostics/contact_score": contact_score.astype(np.float32),
            "diagnostics/grasp_evidence": np.maximum(
                contact_score, both_contact
            ).astype(np.float32),
            "diagnostics/grasp_quality": grasp_quality.astype(np.float32),
            "diagnostics/opposition_score": opposition_score.astype(np.float32),
            "diagnostics/midpoint_score": midpoint_score.astype(np.float32),
            "diagnostics/span_score": span_score.astype(np.float32),
            "diagnostics/both_contact": both_contact.astype(np.float32),
            "diagnostics/reach_dist": reach_dist.astype(np.float32),
            "diagnostics/midpoint_dist": midpoint_dist.astype(np.float32),
            "diagnostics/pad_span": pad_span.astype(np.float32),
            "diagnostics/cube_z": cube_z.astype(np.float32),
            "diagnostics/lift_progress": lift_progress.astype(np.float32),
        }

        return rewards, ep_success

    # ------------------------------------------------------------------
    # Contact force reading
    # ------------------------------------------------------------------

    def _read_finger_contact_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-env finger-cube contact force magnitudes."""
        nacon = int(min(self._mw_data.nacon.numpy()[0], self._mw_data.naconmax))
        lf_forces = np.zeros(self.num_envs, dtype=np.float64)
        rf_forces = np.zeros(self.num_envs, dtype=np.float64)
        if nacon == 0:
            return lf_forces, rf_forces

        contact_ids = wp.array(np.arange(nacon, dtype=np.int32), dtype=wp.int32)
        force_out = wp.zeros(nacon, dtype=wp.spatial_vector)
        mjw.contact_force(self._mw_model, self._mw_data, contact_ids, False, force_out)
        wp.synchronize()

        force_np = force_out.numpy()
        force_mag = np.linalg.norm(force_np[:, :3], axis=1)

        geom_np = self._mw_data.contact.geom.numpy()[:nacon]
        world_np = self._mw_data.contact.worldid.numpy()[:nacon]

        geom_bodyid = self._model_cpu.geom_bodyid
        valid = (geom_np[:, 0] >= 0) & (geom_np[:, 1] >= 0)
        g0 = np.clip(geom_np[:, 0], 0, len(geom_bodyid) - 1)
        g1 = np.clip(geom_np[:, 1], 0, len(geom_bodyid) - 1)
        body0 = geom_bodyid[g0]
        body1 = geom_bodyid[g1]

        lf_mask = valid & (
            ((body0 == self._lf_id) & (body1 == self._cube_id))
            | ((body1 == self._lf_id) & (body0 == self._cube_id))
        )
        rf_mask = valid & (
            ((body0 == self._rf_id) & (body1 == self._cube_id))
            | ((body1 == self._rf_id) & (body0 == self._cube_id))
        )

        np.add.at(lf_forces, world_np[lf_mask], force_mag[lf_mask])
        np.add.at(rf_forces, world_np[rf_mask], force_mag[rf_mask])
        return lf_forces, rf_forces

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _write_init_reset_states(
        self, indices: list[int], qpos_np: np.ndarray, qvel_np: np.ndarray
    ) -> None:
        """Write initial home + random cube state.

        Ctrl is also set (home arm qpos + gripper open).  Then
        ``_build_reset_graph`` runs settle physics.
        """
        ctrl_np = self._mw_data.ctrl.numpy().copy()
        for i in indices:
            cube_x = float(self._rng.uniform(*self._cube_x_range))
            cube_y = float(self._rng.uniform(*self._cube_y_range))
            qpos_np[i, :9] = _HOME_QPOS.astype(np.float32)
            qpos_np[i, 9:12] = np.array([cube_x, cube_y, _CUBE_Z], dtype=np.float32)
            qpos_np[i, 12:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            qvel_np[i] = 0.0
            ctrl_np[i, :7] = _HOME_QPOS[:7].astype(np.float32)
            ctrl_np[i, 7] = 255.0

        # Also write ctrl before settle physics
        wp.copy(self._mw_data.ctrl, wp.array(ctrl_np, dtype=wp.float32))

        if self._lock_eef_orientation:
            if self._home_eef_quat is None:
                _home_data = mujoco.MjData(self._model_cpu)
                _home_data.qpos[:9] = _HOME_QPOS
                mujoco.mj_forward(self._model_cpu, _home_data)
                self._home_eef_quat = _home_data.xquat[self._hand_id].copy()
            self._preferred_eef_quat[indices] = self._home_eef_quat

    # ------------------------------------------------------------------
    # Metrics — override to inject reward terms into infos
    # ------------------------------------------------------------------

    def _record_metrics(
        self,
        step_reward: torch.Tensor,
        success: torch.Tensor,
        infos: dict[str, Any],
    ) -> dict[str, Any]:
        reward_terms = {
            key: torch.from_numpy(value).float()
            for key, value in self._last_reward_terms.items()
        }
        infos["reward_terms"] = reward_terms
        return super()._record_metrics(step_reward, success, infos)

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def _get_task_extra_state(self) -> dict[str, Any]:
        wp.synchronize()
        return {
            "_prev_ctrl": self._mw_data.ctrl.numpy().copy(),
            "_preferred_eef_quat": self._preferred_eef_quat.copy(),
            "_last_reward_terms": {
                k: v.copy() for k, v in self._last_reward_terms.items()
            },
        }

    def _set_task_extra_state(self, state: dict[str, Any]) -> None:
        if "_prev_ctrl" in state:
            self._prev_ctrl = state["_prev_ctrl"]
        if "_preferred_eef_quat" in state:
            self._preferred_eef_quat = state["_preferred_eef_quat"]
