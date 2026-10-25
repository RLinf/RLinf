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

Physics is 50 Hz: four 5 ms MuJoCo Warp steps per action. ``step`` does not
reset. The returned state is the post-physics state, and the caller resets
only after that transition has been stored.
"""

from __future__ import annotations

import re

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
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
)
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
            raise RuntimeError("control period must stay 0.02 s")
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
        foot_ids = []
        for geom_id in range(self.sim.mj_model.ngeom):
            name = mujoco.mj_id2name(
                self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            leaf = name.split("/")[-1] if name else ""
            if FOOT_GEOM.fullmatch(leaf):
                foot_ids.append(geom_id)
        if not foot_ids:
            raise RuntimeError("G1 foot collision geoms were not found")
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
        self.xi = torch.zeros(
            self.num_envs, 33, device=self.device, dtype=self.dtype
        )
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
        return self.motion.state_at(self.motion_id, self.motion_time)

    def reset(
        self,
        env_ids: torch.Tensor,
        motion_id: torch.Tensor | None = None,
        motion_time: torch.Tensor | None = None,
    ) -> None:
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

    def step(self, action_executed: torch.Tensor) -> dict[str, torch.Tensor]:
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
        next_state = self._read_state()
        next_time = self.motion_time + 1
        ref = self.motion.state_at(self.motion_id, next_time)
        height, tilt, failed = failure_signals(
            next_state, ref, termination_body=self.termination_body
        )
        timeout = (not self.eval_mode) & (self.time_in_episode + 1 >= self.max_steps)
        clip_end = self.motion.past_clip_end(self.motion_id, next_time)
        end_kind = torch.full(
            (self.num_envs,), EPISODE_CONT, dtype=torch.long, device=self.device
        )
        end_kind = torch.where(
            clip_end, torch.full_like(end_kind, EPISODE_CLIP_END), end_kind
        )
        end_kind = torch.where(
            timeout, torch.full_like(end_kind, EPISODE_TIMEOUT), end_kind
        )
        end_kind = torch.where(
            failed, torch.full_like(end_kind, EPISODE_FAILURE), end_kind
        )
        self.state = next_state
        self.motion_time = next_time
        self.time_in_episode = self.time_in_episode + 1
        return {
            "next_state": next_state,
            "end_kind": end_kind,
            "height_err": height,
            "tilt_err": tilt,
            "ref_state": ref,
        }
