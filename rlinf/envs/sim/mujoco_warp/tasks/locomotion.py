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

"""Gymnasium MuJoCo v4 locomotion tasks on a MuJoCo Warp world."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from rlinf.envs.sim.mujoco_warp.task import TaskWrapper
from rlinf.envs.sim.mujoco_warp.world import MjWarpWorld

__all__ = ["TASK_SPECS", "LocomotionTask", "TaskSpec"]


class TaskSpec:
    """Static description of one locomotion task.

    Reward and termination follow the Gymnasium MuJoCo v4 task semantics:
    ``reward = healthy_reward + forward_velocity - ctrl_cost_weight * ||ctrl||^2``,
    and (for terminating tasks) an episode ends when the body leaves the
    healthy height/angle ranges or the observation becomes non-finite.
    """

    def __init__(
        self,
        xml_file: str,
        obs_qpos_skip: int,
        frame_skip: int,
        ctrl_cost_weight: float,
        healthy_reward: float,
        healthy_z_range: Optional[tuple[float, float]],
        healthy_angle_range: Optional[tuple[float, float]],
        reset_noise_scale: float,
        reset_noise_type: str,
    ) -> None:
        self.xml_file = xml_file
        self.obs_qpos_skip = obs_qpos_skip
        self.frame_skip = frame_skip
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_z_range = healthy_z_range
        self.healthy_angle_range = healthy_angle_range
        self.reset_noise_scale = reset_noise_scale
        self.reset_noise_type = reset_noise_type


TASK_SPECS: dict[str, TaskSpec] = {
    "half_cheetah": TaskSpec(
        xml_file="half_cheetah.xml",
        obs_qpos_skip=1,
        frame_skip=5,
        ctrl_cost_weight=0.1,
        healthy_reward=0.0,
        healthy_z_range=None,
        healthy_angle_range=None,
        reset_noise_scale=0.1,
        reset_noise_type="uniform",
    ),
    "hopper": TaskSpec(
        xml_file="hopper.xml",
        obs_qpos_skip=1,
        frame_skip=4,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        reset_noise_type="uniform",
    ),
    "walker2d": TaskSpec(
        xml_file="walker2d.xml",
        obs_qpos_skip=1,
        frame_skip=4,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        healthy_z_range=(0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        reset_noise_scale=5e-3,
        reset_noise_type="uniform",
    ),
    "ant": TaskSpec(
        xml_file="ant.xml",
        obs_qpos_skip=2,
        frame_skip=5,
        ctrl_cost_weight=5e-4,
        healthy_reward=1.0,
        healthy_z_range=(0.2, 1.0),
        healthy_angle_range=None,
        reset_noise_scale=0.1,
        reset_noise_type="uniform",
    ),
    "humanoid": TaskSpec(
        xml_file="humanoid.xml",
        obs_qpos_skip=2,
        frame_skip=5,
        ctrl_cost_weight=0.1,
        healthy_reward=1.25,
        healthy_z_range=(1.0, 2.0),
        healthy_angle_range=None,
        reset_noise_scale=1e-2,
        reset_noise_type="uniform",
    ),
}


class LocomotionTask(TaskWrapper):
    """Torque-controlled Gymnasium locomotion task.

    Args:
        spec: Static task description.
        seed: Seed mixed into the deterministic reset noise.
    """

    def __init__(self, spec: TaskSpec, seed: int) -> None:
        self.spec = spec
        self.seed = int(seed)

    def obs_dim(self, world: MjWarpWorld) -> int:
        """Return ``(nq - obs_qpos_skip) + nv``."""
        return (world.nq - self.spec.obs_qpos_skip) + world.nv

    def frame_skip(self) -> int:
        """Return the task's physics steps per control step."""
        return self.spec.frame_skip

    def observe(self, world: MjWarpWorld) -> torch.Tensor:
        """Return ``qpos`` without the skipped root coordinates, then ``qvel``."""
        skip = self.spec.obs_qpos_skip
        return torch.cat([world.qpos[:, skip:], world.qvel], dim=1)

    def reset(
        self,
        world: MjWarpWorld,
        world_ids: torch.Tensor,
        reset_state_ids: np.ndarray,
    ) -> torch.Tensor:
        """Reset the given worlds to ``qpos0`` plus task-specific noise."""
        reset_state_ids = np.asarray(reset_state_ids, dtype=np.int64)
        world_ids = world_ids.to(device=world.device, dtype=torch.long).reshape(-1)
        if reset_state_ids.shape != (int(world_ids.shape[0]),):
            raise ValueError(
                "reset_state_ids must have one ID per selected world; "
                f"expected {(int(world_ids.shape[0]),)}, got {reset_state_ids.shape}."
            )
        qpos_noise = torch.empty((world_ids.shape[0], world.nq), dtype=torch.float32)
        qvel_noise = torch.empty((world_ids.shape[0], world.nv), dtype=torch.float32)
        for row, state_id in enumerate(reset_state_ids):
            noise_gen = torch.Generator(device="cpu")
            noise_gen.manual_seed(self.seed + int(state_id))
            if self.spec.reset_noise_type == "normal":
                qpos_noise[row] = torch.randn(world.nq, generator=noise_gen)
                qvel_noise[row] = torch.randn(world.nv, generator=noise_gen)
            else:
                qpos_noise[row] = torch.rand(world.nq, generator=noise_gen) * 2 - 1
                qvel_noise[row] = torch.rand(world.nv, generator=noise_gen) * 2 - 1

        world.reset_worlds(world_ids)
        world.qpos[world_ids] = world.qpos0 + (
            qpos_noise * self.spec.reset_noise_scale
        ).to(world.device)
        world.qvel[world_ids] = (qvel_noise * self.spec.reset_noise_scale).to(
            world.device
        )
        world.ctrl[world_ids] = 0.0
        world.forward()
        return self.observe(world)[world_ids].clone()

    def step(
        self, world: MjWarpWorld, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply torques for ``frame_skip`` physics steps."""
        expected_shape = (world.num_envs, world.nu)
        if actions.shape != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}, got {tuple(actions.shape)}."
            )
        actions = actions.detach().to(device=world.device, dtype=torch.float32)
        ctrl = torch.clamp(actions, world.ctrl_low, world.ctrl_high)
        x_before = world.qpos[:, 0].clone()
        world.ctrl.copy_(ctrl)
        for _ in range(self.spec.frame_skip):
            world.physics_step()
        x_after = world.qpos[:, 0]

        frame_dt = world.timestep * self.spec.frame_skip
        forward_vel = (x_after - x_before) / frame_dt
        ctrl_cost = self.spec.ctrl_cost_weight * (world.ctrl**2).sum(dim=1)
        rewards = self.spec.healthy_reward + forward_vel - ctrl_cost

        obs = self.observe(world)
        terminations = torch.zeros(
            world.num_envs, dtype=torch.bool, device=world.device
        )
        if self.spec.healthy_z_range is not None:
            # qpos layout is [rootx, rootz, rooty(angle), ...] for planar tasks
            # (skip=1) and [rootx, rooty, rootz, quat, ...] for 3D tasks
            # (skip=2).
            z = world.qpos[:, 2] if self.spec.obs_qpos_skip == 2 else world.qpos[:, 1]
            min_z, max_z = self.spec.healthy_z_range
            terminations |= (z < min_z) | (z > max_z)
        if self.spec.healthy_angle_range is not None:
            angle = world.qpos[:, 2]
            min_a, max_a = self.spec.healthy_angle_range
            terminations |= (angle < min_a) | (angle > max_a)
        if self.spec.healthy_z_range is not None:
            terminations |= ~torch.isfinite(obs).all(dim=1)

        return obs, rewards, terminations
