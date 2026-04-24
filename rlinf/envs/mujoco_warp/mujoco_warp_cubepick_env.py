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
  - State is read back as batched numpy arrays: shape (num_envs, ...)
  - IK Jacobians are computed on GPU via mjw.jac() (batched across all worlds)
    and solved in a single batched solve_dls_ik_torch call on CUDA
  - The CUDA graph captures forward + jac + step for zero kernel-launch overhead
  - Reset settle physics runs via a second CUDA graph (_reset_settle_steps steps)

State (32 dims):
  qpos       (9):  arm joints 1-7 + finger_joint1 + finger_joint2
  qvel       (9):  corresponding velocities
  eef_pos    (3):  hand position + 0.105m offset in hand z-axis direction
  eef_quat   (4):  hand quaternion (wxyz)
  cube_pos   (3):  cube centre position
  cube_quat  (4):  cube orientation (wxyz)

Action (7 dims):
  action[0:3] → TCP position delta in world frame
  action[3:6] → TCP rotation delta in axis-angle form
  action[6]   → binary gripper open / close command
"""

from __future__ import annotations

import copy
import os
from typing import Any, Optional, Union

import gymnasium as gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

from rlinf.envs.ee_dls_ik import (
    axis_angle_to_quat_torch,
    decode_delta_action_torch,
    quat_multiply_torch,
    solve_dls_ik_torch,
)

# ---------------------------------------------------------------------------
# Default paths and constants (identical to mujoco_cubepick_env.py)
# ---------------------------------------------------------------------------
_DEFAULT_PANDA_XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "franka_emika_panda", "panda.xml",
)

_HOME_QPOS = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04],
                      dtype=np.float64)

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

_RENDER_CAM_NAME = "env_cam"


def _lookat_quat(eye, lookat) -> list:
    """Compute MuJoCo camera quaternion (wxyz) from eye + lookat positions.

    MuJoCo camera convention: camera looks along its -Z axis, +Y is up.
    Returns wxyz quaternion suitable for ``MjSpec`` camera ``quat`` field.
    """
    eye    = np.asarray(eye,    dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    fwd = lookat - eye
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0., 0., 1.]) if abs(fwd[2]) < 0.99 else np.array([0., 1., 0.])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    z_cam = -fwd                                # camera looks in -Z
    y_cam = np.cross(z_cam, right)
    y_cam /= np.linalg.norm(y_cam)
    R = np.column_stack([right, y_cam, z_cam])  # columns = camera axes in world
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())      # mujoco expects row-major
    return quat.tolist()                        # wxyz


class MuJoCoWarpCubePickEnv(gym.Env):
    """Franka cube-pick task backed by MuJoCo-Warp (GPU-batched physics).

    Matches the MuJoCo CubePickEnv so both backends can be compared under
    identical RL settings.  Physics is stepped on GPU via mjw.step(); IK
    Jacobians are computed on GPU via mjw.jac() and solved in a single
    batched torch call on CUDA — no per-env CPU loop.

    The constructor signature follows the standard RLinf env contract.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics

        self.group_size: int = cfg.group_size
        self.num_group: int = num_envs // self.group_size
        self.auto_reset: bool = cfg.auto_reset
        self.use_rel_reward: bool = cfg.use_rel_reward
        self.ignore_terminations: bool = cfg.ignore_terminations
        self.max_episode_steps: int = cfg.max_episode_steps
        self.video_cfg = cfg.video_cfg
        self.reward_coef: float = float(cfg.get("reward_coef", 1.0))

        self.device = "cuda"  # MuJoCo-Warp runs on GPU
        self._is_start = True

        init_params = (
            OmegaConf.to_container(cfg.init_params, resolve=True)
            if hasattr(cfg, "init_params") and cfg.init_params is not None
            else {}
        )

        self._panda_xml_path: str = str(
            init_params.get("panda_xml_path", _DEFAULT_PANDA_XML)
        )
        self._timestep: float = float(init_params.get("timestep", _DEFAULT_TIMESTEP))
        self._n_substeps: int = int(init_params.get("n_substeps", _DEFAULT_N_SUBSTEPS))
        self._success_height: float = float(
            init_params.get("success_height", _SUCCESS_HEIGHT)
        )
        self._cube_x_range: tuple[float, float] = tuple(
            init_params.get("cube_x_range", _DEFAULT_CUBE_X_RANGE)
        )
        self._cube_y_range: tuple[float, float] = tuple(
            init_params.get("cube_y_range", _DEFAULT_CUBE_Y_RANGE)
        )
        self._cam_height: int = int(init_params.get("camera_height", 256))
        self._cam_width: int = int(init_params.get("camera_width", 256))
        self._cam_eye: list[float] = list(
            init_params.get("camera_eye", [1.5, -2.0, 1.5])
        )
        self._cam_lookat: list[float] = list(
            init_params.get("camera_lookat", [0.5, 0.0, 0.3])
        )
        self._cam_fov: float = float(init_params.get("camera_fov", 45.0))
        self._lock_eef_orientation: bool = bool(init_params.get("lock_eef_orientation", True))
        self._action_pos_scale: float = float(init_params.get("action_pos_scale", 0.04))
        self._action_rot_scale: float = float(init_params.get("action_rot_scale", 0.35))
        self._ik_damping: float = float(init_params.get("ik_damping", 0.1))
        self._ik_integration_dt: float = float(init_params.get("ik_integration_dt", 0.04))
        self._ik_max_dq: float = float(init_params.get("ik_max_dq", 2.5))
        self._ik_pos_gain: float = float(init_params.get("ik_pos_gain", 0.95))
        self._ik_rot_gain: float = float(init_params.get("ik_rot_gain", 0.95))
        reset_settle_control_steps = init_params.get("reset_settle_control_steps")
        if reset_settle_control_steps is None:
            self._reset_settle_steps = int(
                init_params.get(
                    "reset_settle_steps",
                    _RESET_SETTLE_CONTROL_STEPS * self._n_substeps,
                )
            )
        else:
            self._reset_settle_steps = int(reset_settle_control_steps) * self._n_substeps

        self._rng = np.random.default_rng(self.seed)

        self._build_sim()

        # Ctrl smoothing state (exponential moving average)
        self._prev_ctrl = np.zeros((num_envs, self._model_cpu.nu), dtype=np.float32)

        # Tracking tensors
        self.prev_step_reward = torch.zeros(num_envs, dtype=torch.float32)
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.int32)
        self._last_reward_terms: dict[str, np.ndarray] = {}
        self._preferred_eef_quat = np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (num_envs, 1)
        )

        if record_metrics:
            self._init_metrics()

    # ------------------------------------------------------------------
    # MuJoCo-Warp setup
    # ------------------------------------------------------------------

    def _build_sim(self) -> None:
        """Load Panda MJCF, add scene elements, build GPU model + data."""
        spec = mujoco.MjSpec.from_file(self._panda_xml_path)

        # Remove default keyframe (becomes invalid with cube free-joint)
        for k in list(spec.keys):
            spec.delete(k)

        wb = spec.worldbody

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

        # Disable self-collision for all arm body pairs.
        _ARM_BODIES = [
            "link0", "link1", "link2", "link3", "link4",
            "link5", "link6", "link7", "hand",
            "left_finger", "right_finger",
        ]
        for i, b1 in enumerate(_ARM_BODIES):
            for b2 in _ARM_BODIES[i + 1:]:
                ex = spec.add_exclude()
                ex.bodyname1 = b1
                ex.bodyname2 = b2

        l1 = wb.add_light()
        l1.name = "env_light1"
        l1.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        l1.pos = [0.0, 0.0, 3.0]; l1.dir = [0.0, 0.0, -1.0]
        l1.diffuse = [0.8, 0.8, 0.8]; l1.specular = [0.5, 0.5, 0.5]
        l1.castshadow = True

        l2 = wb.add_light()
        l2.name = "env_light2"
        l2.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        l2.pos = [-2.0, 2.0, 2.0]; l2.dir = [0.5, -0.5, -1.0]
        l2.diffuse = [0.4, 0.4, 0.4]; l2.specular = [0.0, 0.0, 0.0]
        l2.castshadow = True

        cam = wb.add_camera()
        cam.name = _RENDER_CAM_NAME
        cam.pos  = list(self._cam_eye)
        cam.quat = _lookat_quat(self._cam_eye, self._cam_lookat)
        cam.fovy = self._cam_fov

        # Remove finger equality constraint — the tendon actuator already couples
        # finger_joint1 and finger_joint2; the equality constraint adds solver load
        # and can cause contact oscillation when grasping.
        for eq in list(spec.equalities):
            spec.delete(eq)

        self._model_cpu = spec.compile()
        self._model_cpu.opt.timestep = self._timestep

        # Make contacts dominate the solver and stiffen contact response.
        # Default impratio=1.0 gives contacts equal weight with (removed)
        # equality constraints — far too soft for grasping.  impratio=20
        # ensures contacts override any residual soft constraint.
        self._model_cpu.opt.impratio = 20.0

        # Boost gripper stiffness so the Franka can grip a 4 cm cube reliably.
        # The Menagerie tendon uses coef=0.5 per finger, which halves the effective
        # per-finger force:  per_finger_kp = 0.5 * tendon_kp,  per_finger_kd = 0.5 * tendon_kd.
        # We need per-finger kp ≈ 200 and kd ≈ 20 (matching Genesis kp=200 per joint),
        # so the *tendon* gains must be 4× the original (not 2×).
        # The previous 2× scaling gave only per-finger kp=100, kd=5 — too weak for
        # stable lifting (2 N/finger grip ≈ 8 N friction vs 0.49 N cube weight, but
        # MuJoCo soft contacts lose effective friction during dynamic arm motion).
        # Also scale biasprm[2] (damping) which was previously missed.
        _GRIPPER_SCALE = 4.0
        _gripper_act = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self._model_cpu.actuator_gainprm[_gripper_act, 0] *= _GRIPPER_SCALE
        self._model_cpu.actuator_biasprm[_gripper_act, 1] *= _GRIPPER_SCALE
        self._model_cpu.actuator_biasprm[_gripper_act, 2] *= _GRIPPER_SCALE
        self._model_cpu.actuator_forcerange[_gripper_act] = [
            -100.0 * _GRIPPER_SCALE, 100.0 * _GRIPPER_SCALE
        ]

        # Boost wrist joint (5-7) PD stiffness for reliable trajectory tracking
        # during lifting.  The Menagerie XML sets kp=2000 for joints 5-7
        # vs kp=4500/3500 for joints 1-4, giving the wrist 2.25× lower
        # stiffness.  When the arm carries the cube, the wrist lags behind
        # the IK targets causing shaking and oscillation.  Raise joints 5-7
        # kp/kd to match joint 3-4 levels.
        _WRIST_KP = 2500.0
        _WRIST_KD = 250.0
        for jidx, jname in enumerate(["joint5", "joint6", "joint7"]):
            aid = mujoco.mj_name2id(
                self._model_cpu, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{jidx + 5}"
            )
            self._model_cpu.actuator_gainprm[aid, 0] = _WRIST_KP
            self._model_cpu.actuator_biasprm[aid, 0] = 0.0
            self._model_cpu.actuator_biasprm[aid, 1] = -_WRIST_KP
            self._model_cpu.actuator_biasprm[aid, 2] = -_WRIST_KD

        # Cache body IDs
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
                mujoco.mj_name2id(self._model_cpu, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
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

        # Stiffen contacts and boost friction for reliable grasping.
        # Root causes of cube slip during dynamic arm motion:
        #   1. solref timeconst (0.005) == timestep (0.005) → marginal solver
        #      stability; timeconst must be ≥ 2× timestep for safety.  Use 0.02
        #      (4× timestep) for robust contact even with finer physics steps.
        #   2. Missing solimp override → default dmin=0.9 loses 10% of grip force
        #      to compliance.  Set near-rigid: dmin=0.99, dmax=0.999.
        #   3. condim=6 → enables tangential + torsional + rolling friction
        #      for maximum grasp stability during dynamic arm motion.
        #   4. solver iterations too low for 16+ finger collision geoms.
        # Friction: element-wise min of two geom vectors, so both cube and
        # fingertip geoms must be raised to get the effective values at contact.
        _GRASP_FRICTION = np.array([5.0, 0.2, 0.02])
        _GRASP_SOLREF = np.array([0.02, 1.0])
        _GRASP_SOLIMP = np.array([0.99, 0.999, 0.001, 0.5, 2.0])
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

        # Cache finger joint qpos addresses for vectorised contact proxy
        _lf_jnt = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"
        )
        _rf_jnt = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"
        )
        self._lf_qpos_adr: int = int(self._model_cpu.jnt_qposadr[_lf_jnt])
        self._rf_qpos_adr: int = int(self._model_cpu.jnt_qposadr[_rf_jnt])

        # Cache camera index for GPU render calls
        self._render_cam_idx = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_CAMERA, _RENDER_CAM_NAME
        )

        # GPU model + batched data
        self._mw_model = mjw.put_model(self._model_cpu)
        # njmax is the constraint-Jacobian buffer size (nefc × nv per world).
        # Peak nefc for this scene is ~36 (nv=15 → nJ=540); 2048 gives 3.8× headroom.
        self._mw_data = mjw.make_data(self._model_cpu, nworld=self.num_envs, njmax=2048)

        # Pre-allocated Warp arrays for batched Jacobian via mjw.jac()
        nv = self._model_cpu.nv
        self._jacp = wp.zeros((self.num_envs, 3, nv), dtype=wp.float32)
        self._jacr = wp.zeros((self.num_envs, 3, nv), dtype=wp.float32)
        self._jac_point = wp.zeros(self.num_envs, dtype=wp.vec3)
        self._jac_body = wp.zeros(self.num_envs, dtype=wp.int32)

        # Capture n_substeps × step + forward as a CUDA graph.
        # jac is intentionally excluded from the graph: it is computed fresh
        # each control step in _compute_ctrl_all using the actual post-step
        # EEF position.  Including jac inside the graph would evaluate it at
        # the previous step's EEF point (one-step lag), causing the stiff PD
        # controllers (kp up to 4500) to amplify the resulting IK error into
        # visible arm oscillation.
        wp.copy(
            self._mw_data.ctrl,
            wp.array(np.zeros((self.num_envs, self._model_cpu.nu), dtype=np.float32), dtype=wp.float32),
        )
        wp.copy(self._jac_body, wp.full(self.num_envs, self._hand_id, dtype=wp.int32))
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                for _ in range(self._n_substeps):
                    mjw.step(self._mw_model, self._mw_data)
                mjw.forward(self._mw_model, self._mw_data)
            self._step_graph = capture.graph

            # Capture reset settle graph: _reset_settle_steps raw physics steps.
            # Called once per reset after writing initial states to GPU.  The
            # graph replays the same kernel launches against whatever is in
            # _mw_data at launch time, so pre-writing ctrl before
            # wp.capture_launch is the correct usage pattern (same as step).
            with wp.ScopedCapture() as capture:
                for _ in range(self._reset_settle_steps):
                    mjw.step(self._mw_model, self._mw_data)
                mjw.forward(self._mw_model, self._mw_data)
            self._reset_graph = capture.graph
        else:
            self._step_graph = None
            self._reset_graph = None

        # Precompute the EEF quaternion at home position (constant — home qpos never changes).
        _home_data = mujoco.MjData(self._model_cpu)
        _home_data.qpos[:9] = _HOME_QPOS
        mujoco.mj_forward(self._model_cpu, _home_data)
        self._home_eef_quat: np.ndarray = _home_data.xquat[self._hand_id].copy()

        # Lazy GPU render state (created on first _render_images call)
        self._render_ctx   = None
        self._rgb_buf      = None
        self._render_graph = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    def reset(
        self,
        env_idx: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environments (all or a subset).

        Initial states are written to the GPU batched Data arrays, then settle
        physics runs via the captured CUDA graph (_reset_settle_steps raw
        steps).  For partial resets, non-reset env states are saved before
        and restored after the graph launch.
        """
        if env_idx is None and options is not None and "env_idx" in options:
            env_idx = options["env_idx"]

        if env_idx is None:
            indices = list(range(self.num_envs))
            reset_tensor = None
        else:
            if isinstance(env_idx, torch.Tensor):
                indices = env_idx.tolist()
                reset_tensor = env_idx.to(dtype=torch.int64)
            elif isinstance(env_idx, np.ndarray):
                indices = env_idx.tolist()
                reset_tensor = torch.from_numpy(env_idx).to(dtype=torch.int64)
            else:
                indices = list(env_idx)
                reset_tensor = torch.tensor(indices, dtype=torch.int64)

        # Read current GPU state; we'll modify only reset-env rows before writing back.
        qpos_np       = self._mw_data.qpos.numpy().copy()        # (nworld, nq)
        qvel_np       = self._mw_data.qvel.numpy().copy()        # (nworld, nv)
        ctrl_np       = self._mw_data.ctrl.numpy().copy()        # (nworld, nu)

        # Save non-reset env states so the settle graph (which steps all worlds)
        # does not corrupt them.
        partial_reset = len(indices) < self.num_envs
        if partial_reset:
            non_reset_idx = [i for i in range(self.num_envs) if i not in set(indices)]
            saved_qpos       = qpos_np[non_reset_idx].copy()
            saved_qvel       = qvel_np[non_reset_idx].copy()
            saved_ctrl       = ctrl_np[non_reset_idx].copy()

        # Write initial unsettled states for each reset env (no CPU stepping).
        for i in indices:
            cube_x = float(self._rng.uniform(*self._cube_x_range))
            cube_y = float(self._rng.uniform(*self._cube_y_range))
            qpos_np[i, :9]   = _HOME_QPOS.astype(np.float32)
            qpos_np[i, 9:12] = np.array([cube_x, cube_y, _CUBE_Z],    dtype=np.float32)
            qpos_np[i, 12:]  = np.array([1.0, 0.0, 0.0, 0.0],         dtype=np.float32)
            qvel_np[i]       = 0.0
            ctrl_np[i, :7]   = _HOME_QPOS[:7].astype(np.float32)
            ctrl_np[i, 7]    = 255.0

        if self._lock_eef_orientation:
            self._preferred_eef_quat[indices] = self._home_eef_quat

        # Write initial states to GPU (in-place; preserves buffer addresses for CUDA graphs).
        wp.copy(self._mw_data.qpos,       wp.array(qpos_np,       dtype=wp.float32))
        wp.copy(self._mw_data.qvel,       wp.array(qvel_np,       dtype=wp.float32))
        wp.copy(self._mw_data.ctrl,       wp.array(ctrl_np,       dtype=wp.float32))

        # Settle on GPU via CUDA graph (_reset_settle_steps raw physics steps).
        if self._reset_graph is not None:
            wp.capture_launch(self._reset_graph)
            wp.synchronize()
        else:
            for _ in range(self._reset_settle_steps):
                mjw.step(self._mw_model, self._mw_data)
            mjw.forward(self._mw_model, self._mw_data)

        # Restore non-reset env states (the settle graph advanced all worlds).
        if partial_reset:
            settled_qpos       = self._mw_data.qpos.numpy().copy()
            settled_qvel       = self._mw_data.qvel.numpy().copy()
            settled_ctrl       = self._mw_data.ctrl.numpy().copy()
            settled_qpos[non_reset_idx]       = saved_qpos
            settled_qvel[non_reset_idx]       = saved_qvel
            settled_ctrl[non_reset_idx]       = saved_ctrl
            wp.copy(self._mw_data.qpos,       wp.array(settled_qpos,       dtype=wp.float32))
            wp.copy(self._mw_data.qvel,       wp.array(settled_qvel,       dtype=wp.float32))
            wp.copy(self._mw_data.ctrl,       wp.array(settled_ctrl,       dtype=wp.float32))
            mjw.forward(self._mw_model, self._mw_data)
            wp.synchronize()

        # Sync prev_ctrl with current GPU ctrl state for next step's smoothing
        self._prev_ctrl = self._mw_data.ctrl.numpy().copy()

        self._reset_metrics(reset_tensor)
        obs = self._wrap_obs()
        return obs, {}

    def step(
        self,
        actions: Union[torch.Tensor, np.ndarray],
        auto_reset: bool = True,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Execute one environment step.

        The CUDA graph captures ``n_substeps × step + forward``.  The
        Jacobian is computed fresh in ``_compute_ctrl_all`` at the actual
        post-step EEF position each control cycle, eliminating the one-step
        point lag that caused arm oscillation with stiff PD joints.

        Args:
            actions: Action tensor of shape ``(num_envs, 7)`` — TCP position
                delta, axis-angle rotation delta, and gripper command.
            auto_reset: Whether to auto-reset terminated/truncated envs.
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        actions_np = actions.cpu().numpy()  # (B, 7)
        if actions_np.ndim == 1:
            actions_np = actions_np[np.newaxis, :]

        self._elapsed_steps += 1

        # Compute ctrl for all envs via batched GPU IK
        ctrl_np = self._compute_ctrl_all(actions_np)  # (B, nu) float32

        # Write ctrl in-place (preserves GPU buffer address for CUDA graph)
        wp.copy(self._mw_data.ctrl, wp.array(ctrl_np, dtype=wp.float32))

        # Replay CUDA graph: n_substeps × step + forward
        if self._step_graph is not None:
            wp.capture_launch(self._step_graph)
            wp.synchronize()
        else:
            for _ in range(self._n_substeps):
                mjw.step(self._mw_model, self._mw_data)
            mjw.forward(self._mw_model, self._mw_data)

        # Read all post-step GPU arrays once; share across obs and reward.
        _xpos  = self._mw_data.xpos.numpy()
        _xquat = self._mw_data.xquat.numpy()
        _qpos  = self._mw_data.qpos.numpy()
        _qvel  = self._mw_data.qvel.numpy()
        _ctrl  = self._mw_data.ctrl.numpy()
        _lf_forces, _rf_forces = self._read_finger_contact_forces()

        obs = self._wrap_obs(_qpos, _qvel, _xpos, _xquat)
        rewards_np, success_np = self._calc_rewards_and_success(
            _xpos, _xquat, _qpos, _ctrl,
            lf_contact_force=_lf_forces,
            rf_contact_force=_rf_forces,
        )
        step_reward = torch.from_numpy(rewards_np).float()
        success = torch.from_numpy(success_np)
        reward_terms = {
            key: torch.from_numpy(value).float()
            for key, value in self._last_reward_terms.items()
        }

        step_reward = self.reward_coef * step_reward
        reward_diff = step_reward - self.prev_step_reward
        self.prev_step_reward = step_reward.clone()
        if self.use_rel_reward:
            step_reward = reward_diff

        terminations = success.clone()
        truncations = self._elapsed_steps >= self.max_episode_steps

        infos: dict[str, Any] = {
            "success": success,
            "fail": torch.zeros(self.num_envs, dtype=torch.bool),
        }
        if reward_terms:
            infos["reward_terms"] = reward_terms
        if self.record_metrics:
            infos = self._record_metrics(step_reward, success, infos)

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations.clone()
            terminations = torch.zeros_like(terminations)

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations, truncations, infos

    def chunk_step(
        self,
        chunk_actions: torch.Tensor,
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Execute an action chunk and aggregate results."""
        chunk_size = chunk_actions.shape[1]
        obs_list: list[dict[str, Any]] = []
        infos_list: list[dict[str, Any]] = []
        chunk_rewards: list[torch.Tensor] = []
        raw_chunk_terminations: list[torch.Tensor] = []
        raw_chunk_truncations: list[torch.Tensor] = []

        for i in range(chunk_size):
            obs, step_reward, terminations, truncations, infos = self.step(
                chunk_actions[:, i], auto_reset=False
            )
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards_t = torch.stack(chunk_rewards, dim=1)
        raw_term_t = torch.stack(raw_chunk_terminations, dim=1)
        raw_trunc_t = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_term_t.any(dim=1)
        past_truncations = raw_trunc_t.any(dim=1)
        past_dones = past_terminations | past_truncations

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_term_t)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_trunc_t)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_term_t.clone()
            chunk_truncations = raw_trunc_t.clone()

        return (obs_list, chunk_rewards_t, chunk_terminations, chunk_truncations, infos_list)

    # ------------------------------------------------------------------
    # Contact force reading
    # ------------------------------------------------------------------

    def _read_finger_contact_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-env finger-cube contact force magnitudes via mjw.contact_force.

        Mirrors ``mujoco.mj_contactForce``: reads EFC forces from the GPU
        constraint buffer and reconstructs the 6-D contact-frame force for
        every active contact.  Only contacts where one geom belongs to a
        finger body and the other to the cube body are accumulated.

        Returns:
            lf_forces: ``(num_envs,)`` float64 — summed linear contact-force
                magnitude on the left finger across all finger-cube contacts.
            rf_forces: ``(num_envs,)`` float64 — same for the right finger.
        """
        nacon = int(min(self._mw_data.nacon.numpy()[0], self._mw_data.naconmax))
        lf_forces = np.zeros(self.num_envs, dtype=np.float64)
        rf_forces = np.zeros(self.num_envs, dtype=np.float64)
        if nacon == 0:
            return lf_forces, rf_forces

        contact_ids = wp.array(np.arange(nacon, dtype=np.int32), dtype=wp.int32)
        force_out = wp.zeros(nacon, dtype=wp.spatial_vector)
        mjw.contact_force(
            self._mw_model, self._mw_data,
            contact_ids, False, force_out,
        )
        wp.synchronize()

        force_np = force_out.numpy()                           # (nacon, 6)
        force_mag = np.linalg.norm(force_np[:, :3], axis=1)   # (nacon,) linear force

        geom_np = self._mw_data.contact.geom.numpy()[:nacon]    # (nacon, 2) int32
        world_np = self._mw_data.contact.worldid.numpy()[:nacon] # (nacon,) int32

        geom_bodyid = self._model_cpu.geom_bodyid  # (ngeom,) int
        # Geom ID -1 means flex contact — guard by clipping before indexing
        valid = (geom_np[:, 0] >= 0) & (geom_np[:, 1] >= 0)
        g0 = np.clip(geom_np[:, 0], 0, len(geom_bodyid) - 1)
        g1 = np.clip(geom_np[:, 1], 0, len(geom_bodyid) - 1)
        body0 = geom_bodyid[g0]
        body1 = geom_bodyid[g1]

        # Accept only contacts where one body is finger AND other is cube
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
    # IK / action application
    # ------------------------------------------------------------------

    def _compute_ctrl_all(self, actions: np.ndarray) -> np.ndarray:
        """Compute ctrl for all envs via batched DLS IK.

        Computes a fresh Jacobian at the current post-step EEF position each
        call, then solves DLS IK via a single batched ``solve_dls_ik_torch``
        call on CUDA — no per-env loop.

        Args:
            actions: shape (num_envs, 7), float32/64

        Returns:
            ctrl: shape (num_envs, nu), float32
        """
        nu = self._model_cpu.nu

        xpos_all = self._mw_data.xpos.numpy()
        xquat_all = self._mw_data.xquat.numpy()

        hand_xpos = xpos_all[:, self._hand_id, :]
        hand_xquat = xquat_all[:, self._hand_id, :]

        eef_pos = self._batch_eef_offset(hand_xpos, hand_xquat)

        # Compute Jacobian fresh at the current post-step EEF position.
        # Computing it here (outside the CUDA graph) ensures the evaluation
        # point matches the actual current EEF, eliminating the one-step point
        # lag that was causing IK errors and arm oscillation.
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
            target_quat = torch.from_numpy(self._preferred_eef_quat.astype(np.float32)).cuda()
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
            joint_lower=torch.from_numpy(self._arm_qpos_lower.astype(np.float32)).cuda(),
            joint_upper=torch.from_numpy(self._arm_qpos_upper.astype(np.float32)).cuda(),
            pos_gain=self._ik_pos_gain,
            rot_gain=self._ik_rot_gain,
        )

        ctrl_out = np.zeros((self.num_envs, nu), dtype=np.float32)
        ctrl_out[:, :7] = arm_targets.cpu().numpy()
        ctrl_out[:, 7] = np.where(actions[:, 6] > 0.0, 255.0, 0.0).astype(np.float32)

        # Exponential smoothing on ctrl to suppress IK jitter from noisy actions.
        # Increased from 0.5 to 0.75 so the arm tracks targets faster — needed
        # for 4-second episode length.
        alpha = 0.75
        ctrl_out = alpha * ctrl_out + (1.0 - alpha) * self._prev_ctrl
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
    # Reward computation
    # ------------------------------------------------------------------

    def _calc_rewards_and_success(
        self,
        xpos_all: Optional[np.ndarray] = None,
        xquat_all: Optional[np.ndarray] = None,
        qpos_all: Optional[np.ndarray] = None,
        ctrl_all: Optional[np.ndarray] = None,
        lf_contact_force: Optional[np.ndarray] = None,
        rf_contact_force: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute dense reward and success flag for all envs.

        Fully vectorised over the batch.  Contact detection uses real contact
        forces from ``mjw.contact_force`` (same as ``mujoco.mj_contactForce``
        on the CPU backend).  Contact scoring mirrors Genesis CubePickTask:
        ``both_contact`` is binary (any finger-cube contact on both fingers),
        ``contact_score = tanh(min(lf_force, rf_force) / 5.0)``.

        Lift-only variant: success is defined by cube height exceeding a
        threshold, and the primary shaping reward is ``lift_reward`` which
        uses smooth grasp gating (no hard binary gate).

        Accepts pre-read arrays (from ``step()``) to avoid redundant GPU reads.
        """
        if xpos_all is None:
            xpos_all = self._mw_data.xpos.numpy()
        if xquat_all is None:
            xquat_all = self._mw_data.xquat.numpy()
        if qpos_all is None:
            qpos_all = self._mw_data.qpos.numpy()
        if ctrl_all is None:
            ctrl_all = self._mw_data.ctrl.numpy()

        xpos_all      = xpos_all.astype(np.float64)       # (B, nbody, 3)
        xquat_all     = xquat_all.astype(np.float64)      # (B, nbody, 4) wxyz
        qpos_all      = qpos_all.astype(np.float64)       # (B, nq)
        ctrl_all      = ctrl_all.astype(np.float64)       # (B, nu)

        # --- Batched quaternion rotation helper ---
        def _rotated_offset(body_id: int, offset: np.ndarray) -> np.ndarray:
            """Rotate a constant body-frame offset into world frame for all envs."""
            bpos  = xpos_all[:, body_id, :]   # (B, 3)
            bquat = xquat_all[:, body_id, :]  # (B, 4) wxyz
            q_w   = bquat[:, 0]               # (B,)
            q_xyz = bquat[:, 1:]              # (B, 3)
            cross = np.cross(q_xyz, offset)   # (B, 3)
            world_off = offset + 2.0 * q_w[:, None] * cross + 2.0 * np.cross(q_xyz, cross)
            return bpos + world_off           # (B, 3)

        # --- Finger pad and cube positions ---
        lf_pad_pos = _rotated_offset(self._lf_id, _FINGER_PAD_OFFSET)  # (B, 3)
        rf_pad_pos = _rotated_offset(self._rf_id, _FINGER_PAD_OFFSET)  # (B, 3)
        cube_pos   = xpos_all[:, self._cube_id, :]                      # (B, 3)

        # --- Reaching reward ---
        pinch_mid  = 0.5 * (lf_pad_pos + rf_pad_pos)                       # (B, 3)
        reach_dist = np.linalg.norm(pinch_mid - cube_pos, axis=1)           # (B,)
        reaching_reward = 1.0 - np.tanh(5.0 * reach_dist)

        # --- Pinch geometry ---
        cube_to_lf = cube_pos - lf_pad_pos                                  # (B, 3)
        cube_to_rf = cube_pos - rf_pad_pos                                  # (B, 3)
        lf_dir = cube_to_lf / np.linalg.norm(cube_to_lf, axis=1, keepdims=True).clip(1e-6)
        rf_dir = cube_to_rf / np.linalg.norm(cube_to_rf, axis=1, keepdims=True).clip(1e-6)
        opposition_cos   = np.sum(lf_dir * rf_dir, axis=1)                  # (B,)
        opposition_score = np.clip((-opposition_cos + 1.0) * 0.5, 0.0, 1.0)
        midpoint_dist  = np.linalg.norm(cube_pos - pinch_mid, axis=1)       # (B,)
        midpoint_score = 1.0 - np.tanh(12.0 * midpoint_dist)
        pad_span    = np.linalg.norm(rf_pad_pos - lf_pad_pos, axis=1)       # (B,)
        span_score  = np.exp(
            -((pad_span - _PRE_GRASP_PAD_SPAN_TARGET) / _PRE_GRASP_PAD_SPAN_STD) ** 2
        )
        pinch_score = opposition_score * midpoint_score * span_score

        # --- Real contact forces via mjw.contact_force (mirrors mj_contactForce) ---
        # Matches Genesis CubePickTask.compute_reward contact scoring:
        #   both_contact = any finger-cube contact on both fingers (binary)
        #   contact_score = tanh(min(lf_force, rf_force) / 5.0)
        # 5.0 N normalisation: a 5 N grip force maps to tanh(1) ≈ 0.76.
        lf_forces = lf_contact_force if lf_contact_force is not None \
            else np.zeros(self.num_envs, dtype=np.float64)
        rf_forces = rf_contact_force if rf_contact_force is not None \
            else np.zeros(self.num_envs, dtype=np.float64)
        lf_contact_smooth = np.tanh(lf_forces / 0.05)
        rf_contact_smooth = np.tanh(rf_forces / 0.05)
        both_contact = lf_contact_smooth * rf_contact_smooth
        contact_score = np.tanh(np.minimum(lf_forces, rf_forces) / 5.0)

        grasp_quality = both_contact * contact_score * pinch_score          # (B,)

        # --- Dense rewards ---
        pinch_reward = 2.0 * pinch_score
        grasp_reward = 4.0 * grasp_quality

        cube_z = cube_pos[:, 2]
        lift_progress = np.clip((cube_z - _CUBE_Z) / (self._success_height - _CUBE_Z), 0.0, 1.0)
        grasp_gate = both_contact * contact_score
        lift_reward = _LIFT_REWARD_WEIGHT * lift_progress * grasp_gate

        ep_success     = cube_z > self._success_height
        success_bonus  = _SUCCESS_BONUS * ep_success.astype(np.float64)

        rewards = (
            reaching_reward + pinch_reward + grasp_reward
            + lift_reward + success_bonus
        ).astype(np.float32)

        self._last_reward_terms = {
            "reward/reaching":              reaching_reward.astype(np.float32),
            "reward/pinch":                 pinch_reward.astype(np.float32),
            "reward/grasp":                 grasp_reward.astype(np.float32),
            "reward/lift":                  lift_reward.astype(np.float32),
            "reward/success_bonus":         success_bonus.astype(np.float32),
            "diagnostics/contact_score":    contact_score.astype(np.float32),
            "diagnostics/grasp_evidence":   np.maximum(contact_score, both_contact).astype(np.float32),
            "diagnostics/grasp_quality":    grasp_quality.astype(np.float32),
            "diagnostics/opposition_score": opposition_score.astype(np.float32),
            "diagnostics/midpoint_score":   midpoint_score.astype(np.float32),
            "diagnostics/span_score":       span_score.astype(np.float32),
            "diagnostics/both_contact":     both_contact.astype(np.float32),
            "diagnostics/reach_dist":       reach_dist.astype(np.float32),
            "diagnostics/midpoint_dist":    midpoint_dist.astype(np.float32),
            "diagnostics/pad_span":         pad_span.astype(np.float32),
            "diagnostics/cube_z":           cube_z.astype(np.float32),
            "diagnostics/lift_progress":    lift_progress.astype(np.float32),
        }
        return rewards, ep_success

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _wrap_obs(
        self,
        qpos_all: Optional[np.ndarray] = None,
        qvel_all: Optional[np.ndarray] = None,
        xpos_all: Optional[np.ndarray] = None,
        xquat_all: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Build 32-D state observation from GPU batched arrays.

        Layout:
          qpos(9) + qvel(9) + eef_pos(3) + eef_quat(4) + cube_pos(3)
          + cube_quat(4)

        Accepts pre-read arrays to avoid redundant GPU transfers when called
        from ``step()`` together with ``_calc_rewards_and_success``.
        """
        if qpos_all is None:
            qpos_all = self._mw_data.qpos.numpy()   # (nworld, nq)
        if qvel_all is None:
            qvel_all = self._mw_data.qvel.numpy()   # (nworld, nv)
        if xpos_all is None:
            xpos_all = self._mw_data.xpos.numpy()   # (nworld, nbody, 3)
        if xquat_all is None:
            xquat_all = self._mw_data.xquat.numpy()  # (nworld, nbody, 4)

        # EEF position: hand_pos + rotated EEF offset (batched, no loop)
        hand_xquat = xquat_all[:, self._hand_id, :]  # (B, 4)
        hand_xpos  = xpos_all[:, self._hand_id, :]   # (B, 3)
        eef_pos  = self._batch_eef_offset(hand_xpos, hand_xquat)  # (B, 3)
        eef_quat = hand_xquat.astype(np.float32)                  # (B, 4) wxyz

        cube_pos  = xpos_all[:, self._cube_id, :].astype(np.float32)   # (B, 3)
        cube_quat = xquat_all[:, self._cube_id, :].astype(np.float32)  # (B, 4)

        states_np = np.concatenate([
            qpos_all[:, :9].astype(np.float32),   # (B, 9)
            qvel_all[:, :9].astype(np.float32),   # (B, 9)
            eef_pos,                               # (B, 3)
            eef_quat,                              # (B, 4)
            cube_pos,                              # (B, 3)
            cube_quat,                             # (B, 4)
        ], axis=1)  # (B, 32)
        states = torch.from_numpy(states_np)

        obs: dict[str, Any] = {
            "states": states,
            "task_descriptions": [_TASK_DESCRIPTION] * self.num_envs,
        }

        if getattr(self.video_cfg, "save_video", False):
            imgs = self._render_images()
            if imgs is not None:
                obs["main_images"] = torch.from_numpy(imgs)

        return obs

    # ------------------------------------------------------------------
    # Rendering (GPU-side via mjw render pipeline)
    # ------------------------------------------------------------------

    def _render_images(self) -> Optional[np.ndarray]:
        """Render all parallel envs on GPU; return (B, H, W, 3) uint8 array."""
        if self._render_ctx is None:
            cam_active = [i == self._render_cam_idx
                          for i in range(self._model_cpu.ncam)]
            enabled_geom_groups = [0, 2]  # 0=floor/cube, 2=visual meshes; 3=collision meshes hidden
            self._render_ctx = mjw.create_render_context(
                self._model_cpu,
                nworld=self.num_envs,
                cam_res=(self._cam_width, self._cam_height),
                render_rgb=True,
                render_depth=False,
                use_textures=True,
                use_shadows=True,
                cam_active=cam_active,
                enabled_geom_groups=enabled_geom_groups,
            )
            self._rgb_buf = wp.zeros(
                (self.num_envs, self._cam_height, self._cam_width),
                dtype=wp.vec3,
            )
            if wp.get_device().is_cuda:
                with wp.ScopedCapture() as capture:
                    mjw.refit_bvh(self._mw_model, self._mw_data, self._render_ctx)
                    mjw.render(self._mw_model, self._mw_data, self._render_ctx)
                self._render_graph = capture.graph

        if self._render_graph is not None:
            wp.capture_launch(self._render_graph)
            wp.synchronize()
        else:
            mjw.refit_bvh(self._mw_model, self._mw_data, self._render_ctx)
            mjw.render(self._mw_model, self._mw_data, self._render_ctx)

        mjw.get_rgb(self._render_ctx, self._render_cam_idx, self._rgb_buf)
        wp.synchronize()

        rgb_np = self._rgb_buf.numpy()  # (B, H, W, 3) float32 in [0, 1]
        return (rgb_np * 255.0).clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.fail_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32)
        self.reward_term_returns: dict[str, torch.Tensor] = {}

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        if env_idx is not None:
            idx = env_idx
            self.prev_step_reward[idx] = 0.0
            if self.record_metrics:
                self.success_once[idx] = False
                self.fail_once[idx] = False
                self.returns[idx] = 0.0
                for value in self.reward_term_returns.values():
                    value[idx] = 0.0
            self._elapsed_steps[idx] = 0
        else:
            self.prev_step_reward[:] = 0.0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                for value in self.reward_term_returns.values():
                    value[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(
        self,
        step_reward: torch.Tensor,
        success: torch.Tensor,
        infos: dict[str, Any],
    ) -> dict[str, Any]:
        self.returns += step_reward
        self.success_once = self.success_once | success.bool()
        reward_terms = infos.get("reward_terms")
        episode_info: dict[str, Any] = {
            "success_once": self.success_once.clone(),
            "fail_once": self.fail_once.clone(),
            "return": self.returns.clone(),
            "episode_len": self._elapsed_steps.clone(),
            "reward": self.returns / self._elapsed_steps.clamp(min=1).float(),
        }
        if reward_terms:
            safe_len = self._elapsed_steps.clamp(min=1).float()
            for key, value in reward_terms.items():
                metric = value.detach().float()
                if key not in self.reward_term_returns:
                    self.reward_term_returns[key] = torch.zeros(
                        self.num_envs, dtype=torch.float32
                    )
                self.reward_term_returns[key] += metric
                episode_info[f"{key}_return"] = self.reward_term_returns[key].clone()
                episode_info[f"{key}_mean"] = self.reward_term_returns[key] / safe_len
        infos["episode"] = episode_info
        return infos

    # ------------------------------------------------------------------
    # Auto-reset
    # ------------------------------------------------------------------

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        final_obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset done environments, stashing the terminal observation."""
        final_obs = copy.deepcopy(final_obs)
        final_info = copy.deepcopy(infos)

        env_idx = torch.arange(self.num_envs)[dones]
        obs, infos = self.reset(env_idx=env_idx)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    # ------------------------------------------------------------------
    # RLinf worker interface
    # ------------------------------------------------------------------

    def update_reset_state_ids(self) -> None:
        """No-op: satisfies the RLinf worker interface."""
        pass

    def close(self) -> None:
        """Free resources."""
        self._render_ctx   = None
        self._rgb_buf      = None
        self._render_graph = None
