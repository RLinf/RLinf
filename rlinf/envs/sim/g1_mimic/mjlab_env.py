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

"""mjlab / MuJoCo Warp G1 backend.

Pinned to mjlab commit ``8bb2bc9e187797689ec82c78c35019afd2fd0376``.

Physics is 50 Hz: four 5 ms MuJoCo Warp steps per action. ``mjwarp.step``
integrates ``qpos`` and ``qvel`` but leaves ``xpos``, ``xquat``, and ``cvel``
at the previous kinematics pass. ``step`` calls ``sim.forward()`` after the
four substeps, so every readout is one physical time. ``step`` does not
reset; the caller resets only after that transition has been stored.

Domain randomization follows mjlab's tracking task and is drawn once, at
construction: foot friction shared by the foot geoms of an env, an additive
torso CoM offset, and joint encoder offsets. The sim context is read back
from the model, centered and scaled by each range.
"""

from __future__ import annotations

import re

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    DT,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    G1_JOINT_NAMES,
    JOINT_POS,
    JOINT_VEL,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
    STATE_DIM,
)
from rlinf.envs.sim.g1_mimic.features import (
    quat_from_euler_xyz,
    quat_mul,
    quat_to_rotmat,
    rot_apply,
    rot_apply_t,
)
from rlinf.envs.sim.g1_mimic.g1_fk import (
    TRACKED_BODY_NAMES,
    g1_action_scale,
    soft_joint_limits,
)
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.reward import failure_signals

PHYSICS_DT = 0.005
DECIMATION = 4
FOOT_GEOM = re.compile(r"^(left|right)_foot[1-7]_collision$")

FOOT_FRICTION_RANGE = (0.3, 1.2)
TORSO_COM_RANGE = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))
ENCODER_BIAS_RANGE = (-0.01, 0.01)
# Reset noise on the sampled reference frame.
RESET_POS_RANGE = ((-0.05, 0.05), (-0.05, 0.05), (-0.01, 0.01))
RESET_EULER_RANGE = ((-0.1, 0.1), (-0.1, 0.1), (-0.2, 0.2))
RESET_LIN_VEL_RANGE = ((-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2))
RESET_ANG_VEL_RANGE = ((-0.52, 0.52), (-0.52, 0.52), (-0.78, 0.78))
RESET_JOINT_POS_RANGE = (-0.1, 0.1)


def episode_end(
    failure: torch.Tensor, timeout: torch.Tensor, clip_end: torch.Tensor
) -> torch.Tensor:
    """One end kind per env: clip end, then timeout, then failure."""
    end = torch.full(
        failure.shape, EPISODE_CONT, dtype=torch.long, device=failure.device
    )
    end = torch.where(failure, torch.full_like(end, EPISODE_FAILURE), end)
    end = torch.where(timeout, torch.full_like(end, EPISODE_TIMEOUT), end)
    return torch.where(clip_end, torch.full_like(end, EPISODE_CLIP_END), end)


def context_normalization() -> tuple[torch.Tensor, torch.Tensor]:
    """Center and half-width of every sim context column."""
    ranges = [FOOT_FRICTION_RANGE, *TORSO_COM_RANGE] + [ENCODER_BIAS_RANGE] * len(
        G1_JOINT_NAMES
    )
    low = torch.tensor([lo for lo, _ in ranges])
    high = torch.tensor([hi for _, hi in ranges])
    return 0.5 * (low + high), 0.5 * (high - low)


def _uniform(ranges, n: int, device: torch.device) -> torch.Tensor:
    bounds = torch.tensor(ranges, device=device, dtype=torch.float32)
    if bounds.dim() == 1:
        bounds = bounds.unsqueeze(0)
    draw = torch.rand(n, bounds.shape[0], device=device)
    return bounds[:, 0] + draw * (bounds[:, 1] - bounds[:, 0])


def _expand_meaninertia(model: object, num_envs: int) -> None:
    """Give each world its own solver inertia scale when stored as an array."""
    stat = getattr(model, "stat", None)
    meaninertia = getattr(stat, "meaninertia", None)
    if num_envs <= 1 or getattr(meaninertia, "shape", None) != (1,):
        return
    import warp as wp

    stat.meaninertia = wp.array(
        meaninertia.numpy().repeat(num_envs),
        dtype=meaninertia.dtype,
        device=meaninertia.device,
    )


def check_default_pose(q0: torch.Tensor, pd_offset: torch.Tensor) -> None:
    """Refuse clips whose default pose is not the asset's PD offset."""
    q0 = q0.to(pd_offset)
    if not torch.allclose(q0, pd_offset, atol=1e-5):
        worst = int((q0 - pd_offset).abs().argmax())
        raise ValueError(
            "reference clip q0 differs from the mjlab G1 default joint pose "
            f"used as the PD offset; joint {G1_JOINT_NAMES[worst]}: clip "
            f"{float(q0[worst]):.4f}, asset {float(pd_offset[worst]):.4f}"
        )


class MjlabG1Env:
    """One vectorized G1 in MuJoCo Warp.

    ``randomize=False`` keeps the asset's friction, CoM and zero encoder
    offsets, and resets write the reference frame without noise.
    """

    def __init__(
        self,
        num_envs: int,
        motion: MotionLibrary,
        device: torch.device,
        max_episode_seconds: float | None = 10.0,
        randomize: bool = True,
    ):
        import mujoco
        import mujoco_warp as mjwarp
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
        from mjlab.managers.event_manager import RecomputeLevel
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import MujocoCfg, SimulationCfg
        from mjlab.sim.sim import Simulation
        from mjlab.terrains import TerrainEntityCfg

        if abs(DECIMATION * PHYSICS_DT - DT) > 1e-8:
            raise RuntimeError(
                f"control period must be {DT} s, got DECIMATION * PHYSICS_DT = "
                f"{DECIMATION * PHYSICS_DT} s"
            )
        self.num_envs = int(num_envs)
        self.motion = motion
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.max_steps = (
            None
            if max_episode_seconds is None
            else int(round(max_episode_seconds / DT))
        )
        self.randomize = bool(randomize)
        sim_device = (
            f"cuda:{self.device.index or 0}" if self.device.type == "cuda" else "cpu"
        )
        scene = Scene(
            SceneCfg(
                num_envs=self.num_envs,
                env_spacing=4.0,
                terrain=TerrainEntityCfg(terrain_type="plane"),
                entities={"robot": get_g1_robot_cfg()},
            ),
            device=sim_device,
        )
        compiled = scene.compile()
        self.sim = Simulation(
            self.num_envs,
            SimulationCfg(
                nconmax=35,
                njmax=250,
                mujoco=MujocoCfg(
                    timestep=PHYSICS_DT,
                    iterations=10,
                    ls_iterations=20,
                ),
            ),
            model=compiled,
            device=sim_device,
        )
        # One line per world when the solver stops at iterations=10 or
        # ls_iterations=20. The caps stay. Clear both before the step graph
        # is recaptured below.
        if hasattr(mjwarp, "OverflowType") and hasattr(
            self.sim.wp_model.opt, "warn_overflow"
        ):
            solver_warn = (
                mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS
            )
            self.sim.wp_model.opt.warn_overflow &= ~solver_warn
        scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot = scene["robot"]
        # Each world's randomized CoM needs its own set_const outputs.
        set_const_fields = (
            "body_subtreemass",
            "dof_invweight0",
            "body_invweight0",
            "tendon_invweight0",
            "tendon_length0",
            "actuator_acc0",
            "cam_pos0",
            "cam_poscom0",
            "cam_mat0",
            "light_pos0",
            "light_poscom0",
            "light_dir0",
        )
        # Expand nested statistics before expand_model_fields recaptures graphs.
        _expand_meaninertia(self.sim.wp_model, self.num_envs)
        self.sim.expand_model_fields(
            ("geom_friction", "body_ipos")
            + tuple(
                name
                for name in set_const_fields
                if getattr(self.sim.mj_model, name).size
            )
        )
        origins = scene.env_origins
        if origins is None:
            origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._origins = origins.to(device=self.device, dtype=self.dtype)
        if tuple(self.robot.joint_names) != G1_JOINT_NAMES:
            raise RuntimeError(
                "mjlab G1 joint order does not match G1_JOINT_NAMES. "
                f"Asset starts {tuple(self.robot.joint_names[:3])}."
            )
        self.action_scale = g1_action_scale().to(self.device)
        self.pd_offset = self.robot.data.default_joint_pos[0].to(self.dtype).clone()
        check_default_pose(motion.q0.to(self.device), self.pd_offset)
        self.soft_limits = soft_joint_limits().to(self.device)
        self._bodies = self.robot.find_bodies(TRACKED_BODY_NAMES, preserve_order=True)[
            0
        ]
        foot_ids = []
        for geom_id in range(self.sim.mj_model.ngeom):
            name = mujoco.mj_id2name(
                self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            leaf = name.split("/")[-1] if name else ""
            if FOOT_GEOM.fullmatch(leaf):
                foot_ids.append(geom_id)
        if not foot_ids:
            raise RuntimeError(
                f"expected G1 foot collision geoms named {FOOT_GEOM.pattern}, found none"
            )
        self._foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=self.device)
        torso = mujoco.mj_name2id(
            self.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link"
        )
        if torso < 0:
            torso = mujoco.mj_name2id(
                self.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
            )
        if torso < 0:
            raise RuntimeError("torso_link is missing from the G1 model")
        self._torso_id = int(torso)
        if self.randomize:
            self._randomize(RecomputeLevel.set_const)
        mean, scale = context_normalization()
        self.sim_context = (self._raw_context() - mean.to(self.device)) / scale.to(
            self.device
        )
        self.state = torch.zeros(
            self.num_envs, STATE_DIM, device=self.device, dtype=self.dtype
        )
        self.motion_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_time = torch.zeros_like(self.motion_id)
        self.time_in_episode = torch.zeros_like(self.motion_id)
        everyone = torch.arange(self.num_envs, device=self.device)
        self.reset(everyone, torch.zeros_like(everyone), torch.zeros_like(everyone))

    def _randomize(self, recompute) -> None:
        n, device = self.num_envs, self.device
        friction = _uniform([FOOT_FRICTION_RANGE], n, device)
        self.sim.model.geom_friction[:, self._foot_ids, 0] = friction
        self.sim.model.body_ipos[:, self._torso_id] += _uniform(
            TORSO_COM_RANGE, n, device
        )
        self.robot.data.encoder_bias[:] = _uniform(
            [ENCODER_BIAS_RANGE] * len(G1_JOINT_NAMES), n, device
        )
        self.sim.recompute_constants(recompute)

    def _raw_context(self) -> torch.Tensor:
        return torch.cat(
            (
                self.sim.model.geom_friction[:, self._foot_ids[0], 0:1],
                self.sim.model.body_ipos[:, self._torso_id],
                self.robot.data.encoder_bias,
            ),
            dim=-1,
        ).to(self.dtype)

    def _write_state(
        self,
        env_ids: torch.Tensor,
        state: torch.Tensor,
        ang_vel_w: torch.Tensor | None = None,
    ) -> None:
        pose = torch.cat(
            (state[:, ROOT_POS] + self._origins[env_ids], state[:, ROOT_QUAT]),
            dim=-1,
        )
        self.robot.write_root_link_pose_to_sim(pose, env_ids)
        if ang_vel_w is None:
            ang_vel_w = rot_apply(
                quat_to_rotmat(state[:, ROOT_QUAT]), state[:, ROOT_ANG_VEL]
            )
        velocity = torch.cat((state[:, ROOT_LIN_VEL], ang_vel_w), dim=-1)
        self.robot.write_root_link_velocity_to_sim(velocity, env_ids)
        self.robot.write_joint_state_to_sim(
            state[:, JOINT_POS], state[:, JOINT_VEL], env_ids=env_ids
        )
        self.sim.forward()

    def apply_horizontal_velocity_push(
        self,
        env_ids: torch.Tensor,
        delta_xy_world: torch.Tensor,
        mode: str = "add",
    ) -> torch.Tensor:
        """Write a horizontal root-velocity push into MuJoCo Warp.

        ``delta_xy_world`` is an xy velocity in the world frame, in m/s.
        ``add`` adds it to the current world linear velocity. ``set`` replaces
        the xy components and keeps the vertical component. Root position,
        world angular velocity, and joint state are not written. ``forward``
        refreshes derived velocities and does not integrate; the next
        ``step`` is what integrates the new ``qvel``.
        """
        if mode not in {"add", "set"}:
            raise ValueError(f"push mode must be add or set, got {mode!r}")
        env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        delta = delta_xy_world.to(device=self.device, dtype=self.dtype)
        if delta.shape != (int(env_ids.shape[0]), 2):
            raise ValueError(
                f"delta_xy_world must be [{int(env_ids.shape[0])}, 2], "
                f"got {tuple(delta.shape)}"
            )
        if not torch.isfinite(delta).all():
            raise ValueError("push velocity is not finite")
        linear = self.robot.data.root_link_lin_vel_w[env_ids].clone()
        angular = self.robot.data.root_link_ang_vel_w[env_ids].clone()
        if mode == "add":
            linear[:, 0] = linear[:, 0] + delta[:, 0]
            linear[:, 1] = linear[:, 1] + delta[:, 1]
        else:
            linear[:, 0] = delta[:, 0]
            linear[:, 1] = delta[:, 1]
        self.robot.write_root_link_velocity_to_sim(
            torch.cat((linear, angular), dim=-1), env_ids
        )
        self.sim.forward()
        self.state = self._read_state()
        return self.state

    def _read_state(self) -> torch.Tensor:
        quat = self.robot.data.root_link_quat_w
        ang_b = rot_apply_t(quat_to_rotmat(quat), self.robot.data.root_link_ang_vel_w)
        return torch.cat(
            (
                self.robot.data.root_link_pos_w - self._origins,
                quat,
                self.robot.data.joint_pos,
                self.robot.data.root_link_lin_vel_w,
                ang_b,
                self.robot.data.joint_vel,
            ),
            dim=-1,
        )

    def body_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World position, wxyz quaternion, linear and angular velocity of the
        tracked bodies, ``[N, 14, ...]`` in ``TRACKED_BODY_NAMES`` order."""
        data = self.robot.data
        return (
            data.body_link_pos_w[:, self._bodies] - self._origins.unsqueeze(1),
            data.body_link_quat_w[:, self._bodies],
            data.body_link_lin_vel_w[:, self._bodies],
            data.body_link_ang_vel_w[:, self._bodies],
        )

    def frame(self) -> torch.Tensor:
        """Library row of each env's current reference frame."""
        return self.motion.global_frame(self.motion_id, self.motion_time)

    def reference_state(self) -> torch.Tensor:
        """Reference state at each env's current motion frame."""
        return self.motion.state_at(self.motion_id, self.motion_time)

    def reset(
        self,
        env_ids: torch.Tensor,
        motion_id: torch.Tensor,
        motion_time: torch.Tensor,
    ) -> None:
        """Write the reference frame ``motion_time`` of ``motion_id`` into ``env_ids``.

        With ``randomize`` the root pose, root velocity and joint angles get
        mjlab's tracking reset noise; joint angles are then clipped to the
        soft limits.
        """
        n = int(env_ids.numel())
        if n == 0:
            return
        self.motion_id[env_ids] = motion_id
        self.motion_time[env_ids] = motion_time
        self.time_in_episode[env_ids] = 0
        start = self.motion.state_at(motion_id, motion_time).clone()
        ang_w = rot_apply(quat_to_rotmat(start[:, ROOT_QUAT]), start[:, ROOT_ANG_VEL])
        if self.randomize:
            device = self.device
            start[:, ROOT_POS] += _uniform(RESET_POS_RANGE, n, device)
            euler = _uniform(RESET_EULER_RANGE, n, device)
            start[:, ROOT_QUAT] = quat_mul(
                quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]),
                start[:, ROOT_QUAT],
            )
            start[:, ROOT_LIN_VEL] += _uniform(RESET_LIN_VEL_RANGE, n, device)
            ang_w = ang_w + _uniform(RESET_ANG_VEL_RANGE, n, device)
            joints = start[:, JOINT_POS] + _uniform(
                [RESET_JOINT_POS_RANGE] * len(G1_JOINT_NAMES), n, device
            )
            start[:, JOINT_POS] = torch.clamp(
                joints, self.soft_limits[:, 0], self.soft_limits[:, 1]
            )
        self._write_state(env_ids, start, ang_w)
        self.state = self._read_state()

    def step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply one 20 ms control step, scored against the frame it started from.

        The reference clock then advances by one frame. Returns ``end_kind``,
        ``failure``, ``timeout``, ``clip_end``, the scoring ``frame`` (library
        row) and its ``ref_state``.
        """
        target = (
            self.pd_offset + self.action_scale * action - self.robot.data.encoder_bias
        )
        self.robot.set_joint_position_target(target)
        for _ in range(DECIMATION):
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(PHYSICS_DT)
        # Refresh xpos/xquat/cvel from the qpos/qvel just integrated.
        # mjwarp.forward does not advance time.
        self.sim.forward()
        self.state = self._read_state()
        frame = self.frame()
        ref = self.motion.rows(frame)
        failed = failure_signals(self.state, ref)[2]
        if self.max_steps is None:
            timeout = torch.zeros_like(failed)
        else:
            timeout = self.time_in_episode + 1 >= self.max_steps
        clip_end = self.motion_time + 1 >= self.motion.clip_length(self.motion_id)
        self.motion_time = self.motion_time + 1
        self.time_in_episode = self.time_in_episode + 1
        return {
            "end_kind": episode_end(failed, timeout, clip_end),
            "failure": failed,
            "timeout": timeout,
            "clip_end": clip_end,
            "frame": frame,
            "ref_state": ref,
        }
