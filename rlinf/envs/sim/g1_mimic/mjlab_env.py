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
four substeps. That refreshes the derived state and does not integrate, so
the 71-d vector is one physical time. ``step`` does not reset. The caller
resets only after that transition has been stored.
"""

from __future__ import annotations

import re

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    DT,
    G1_JOINT_NAMES,
    JOINT_POS,
    JOINT_VEL,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
)
from rlinf.envs.sim.g1_mimic.env import compose_end_kind
from rlinf.envs.sim.g1_mimic.features import quat_to_rotmat
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.reward import failure_signals

PHYSICS_DT = 0.005
DECIMATION = 4
FOOT_GEOM = re.compile(r"^(left|right)_foot[1-7]_collision$")


def _rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", quat_to_rotmat(quat), vec)


def _rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ji,...j->...i", quat_to_rotmat(quat), vec)


def check_default_pose(q0: torch.Tensor, pd_offset: torch.Tensor) -> None:
    """Refuse clips whose default pose is not the asset's PD offset.

    The goal and observation subtract the clips' ``q0``; the PD target adds
    ``pd_offset`` (``q_sp = alpha a + q0``), so they must be the same pose.
    """
    q0 = q0.to(pd_offset)
    if not torch.allclose(q0, pd_offset, atol=1e-5):
        worst = int((q0 - pd_offset).abs().argmax())
        raise ValueError(
            "reference clip q0 differs from the mjlab G1 default joint pose "
            f"used as the PD offset; joint {G1_JOINT_NAMES[worst]}: clip "
            f"{float(q0[worst]):.4f}, asset {float(pd_offset[worst]):.4f}"
        )


class MjlabG1Env:
    """One vectorized G1 in MuJoCo Warp. Not the analytic integrator."""

    def __init__(
        self,
        num_envs: int,
        motion: MotionLibrary,
        device: torch.device,
        max_episode_seconds: float = 10.0,
        termination_body: str = "pelvis",
        eval_mode: bool = False,
    ):
        import mujoco
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
            G1_ACTION_SCALE,
            get_g1_robot_cfg,
        )
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
        self.max_steps = int(round(max_episode_seconds / DT))
        self.termination_body = termination_body
        self.eval_mode = eval_mode
        self.q0 = motion.q0.to(self.device)
        self._recompute = RecomputeLevel
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
        scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot = scene["robot"]
        self.sim.expand_model_fields(("geom_friction", "body_ipos"))
        origins = scene.env_origins
        if origins is None:
            origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._origins = origins.to(device=self.device, dtype=self.dtype)
        if tuple(self.robot.joint_names) != G1_JOINT_NAMES:
            raise RuntimeError(
                "mjlab G1 joint order does not match G1_JOINT_NAMES. "
                f"Asset starts {tuple(self.robot.joint_names[:3])}."
            )
        scale = torch.zeros(ACTION_DIM, device=self.device, dtype=self.dtype)
        hits = torch.zeros(ACTION_DIM, dtype=torch.long)
        for pattern, value in G1_ACTION_SCALE.items():
            for i, name in enumerate(self.robot.joint_names):
                if re.fullmatch(pattern, name):
                    scale[i] = float(value)
                    hits[i] += 1
        if not torch.equal(hits, torch.ones(ACTION_DIM, dtype=torch.long)):
            raise RuntimeError(
                f"each G1 joint needs one action scale, hits={hits.tolist()}"
            )
        self.action_scale = scale
        self.pd_offset = self.robot.data.default_joint_pos[0].to(self.dtype).clone()
        check_default_pose(self.q0, self.pd_offset)
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
        self._default_ipos = self.sim.model.body_ipos.detach().clone()
        self.state = torch.zeros(
            self.num_envs, 71, device=self.device, dtype=self.dtype
        )
        self.xi = torch.zeros(self.num_envs, 33, device=self.device, dtype=self.dtype)
        self.motion_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_time = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.time_in_episode = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reset(torch.arange(self.num_envs, device=self.device))

    def _apply_xi(self, env_ids: torch.Tensor) -> None:
        """Foot friction, encoder bias, and torso COM from ξ. Eval uses identity."""
        self.robot.data.encoder_bias[env_ids] = self.xi[env_ids, 1:30]
        friction = self.sim.model.geom_friction
        friction[env_ids[:, None], self._foot_ids, 0] = self.xi[env_ids, 0:1]
        self.sim.model.body_ipos[env_ids, self._torso_id] = (
            self._default_ipos[env_ids, self._torso_id] + self.xi[env_ids, 30:33]
        )
        self.sim.recompute_constants(self._recompute.set_const)

    def _write_state(self, env_ids: torch.Tensor, state: torch.Tensor) -> None:
        pose = torch.cat(
            (state[:, ROOT_POS] + self._origins[env_ids], state[:, ROOT_QUAT]),
            dim=-1,
        )
        self.robot.write_root_link_pose_to_sim(pose, env_ids)
        ang_w = _rotate(state[:, ROOT_QUAT], state[:, ROOT_ANG_VEL])
        velocity = torch.cat((state[:, ROOT_LIN_VEL], ang_w), dim=-1)
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
        ang_b = _rotate_inverse(quat, self.robot.data.root_link_ang_vel_w)
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

    def ref_state(self) -> torch.Tensor:
        """Reference state at each env's current motion frame."""
        return self.motion.state_at(self.motion_id, self.motion_time)

    def reset(
        self,
        env_ids: torch.Tensor,
        motion_id: torch.Tensor | None = None,
        motion_time: torch.Tensor | None = None,
    ) -> None:
        """Reset ``env_ids`` to reference frames and draw a new sim context.

        Without ``motion_id``/``motion_time`` the envs start at random clip frames.
        """
        from rlinf.envs.sim.g1_mimic.env import sample_xi

        n = int(env_ids.numel())
        if n == 0:
            return
        if motion_id is None:
            motion_id = torch.randint(
                0, self.motion.num_clips, (n,), device=self.device
            )
        if motion_time is None:
            lengths = self.motion.clip_length(motion_id).clamp_min(1)
            motion_time = (torch.rand(n, device=self.device) * lengths.float()).long()
        self.motion_id[env_ids] = motion_id
        self.motion_time[env_ids] = motion_time
        self.time_in_episode[env_ids] = 0
        self.xi[env_ids] = sample_xi(n, self.device, self.dtype)
        if self.eval_mode:
            self.xi[env_ids, 0] = 1.0
            self.xi[env_ids, 1:] = 0.0
        self._apply_xi(env_ids)
        start = self.motion.state_at(motion_id, motion_time).clone()
        start[:, JOINT_POS] = start[:, JOINT_POS] + 0.02 * torch.randn_like(
            start[:, JOINT_POS]
        ) * (0.0 if self.eval_mode else 1.0)
        self._write_state(env_ids, start)
        self.state = self._read_state()

    def reset_standing(self, env_ids: torch.Tensor) -> None:
        """Start from mjlab's standing keyframe, the Appendix G.2 initial state.

        The reference clock is set to clip 0, frame 0 only so that ``step``
        has a reference to read; velocity tests ignore it. Outside eval mode
        the sim context and the joint noise are drawn as in ``reset``.
        """
        from rlinf.envs.sim.g1_mimic.env import sample_xi
        from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state

        n = int(env_ids.numel())
        if n == 0:
            return
        self.motion_id[env_ids] = 0
        self.motion_time[env_ids] = 0
        self.time_in_episode[env_ids] = 0
        self.xi[env_ids] = sample_xi(n, self.device, self.dtype)
        if self.eval_mode:
            self.xi[env_ids, 0] = 1.0
            self.xi[env_ids, 1:] = 0.0
        self._apply_xi(env_ids)
        start = (
            asset_default_state()
            .to(device=self.device, dtype=self.dtype)
            .expand(n, -1)
            .clone()
        )
        if not self.eval_mode:
            start[:, JOINT_POS] = start[:, JOINT_POS] + 0.02 * torch.randn_like(
                start[:, JOINT_POS]
            )
        self._write_state(env_ids, start)
        self.state = self._read_state()

    def step(self, action_executed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply one 20 ms control step and check the new state against the next frame.

        Returns:
            ``next_state``, ``end_kind``, ``motion_failure``, ``timeout``,
            ``clip_end``, ``height_err``, ``tilt_err`` and the ``ref_state`` the
            new state is checked against.
        """
        target = (
            self.pd_offset
            + self.action_scale * action_executed
            - self.robot.data.encoder_bias
        )
        self.robot.set_joint_position_target(target)
        for _ in range(DECIMATION):
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(PHYSICS_DT)
        # Refresh xpos/xquat/cvel from the qpos/qvel just integrated.
        # mjwarp.forward does not advance time.
        self.sim.forward()
        next_state = self._read_state()
        next_time = self.motion_time + 1
        ref = self.motion.state_at(self.motion_id, next_time)
        height, tilt, failed = failure_signals(
            next_state, ref, termination_body=self.termination_body
        )
        timeout = (self.time_in_episode + 1 >= self.max_steps) & (not self.eval_mode)
        clip_end = self.motion.past_clip_end(self.motion_id, next_time)
        end_kind = compose_end_kind(failed, timeout, clip_end)
        self.state = next_state
        self.motion_time = next_time
        self.time_in_episode = self.time_in_episode + 1
        return {
            "next_state": next_state,
            "end_kind": end_kind,
            "motion_failure": failed,
            "timeout": timeout,
            "clip_end": clip_end,
            "height_err": height,
            "tilt_err": tilt,
            "ref_state": ref,
        }
