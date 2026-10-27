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

"""Velocity-tracking reward, fall termination, and command sampling.

Table A.6 and Appendix G. Distillation is not implemented: this module is
the RL interface, whose reward is the velocity reward rather than the
motion-tracking reward.
"""

from __future__ import annotations

import torch

from rlinf.envs.sim.g1_mimic.constants import DT, JOINT_POS, ROOT_ANG_VEL, ROOT_LIN_VEL
from rlinf.envs.sim.g1_mimic.features import (
    heading_yaw,
    projected_gravity,
    quat_to_rotmat,
    yaw_rotation,
)
from rlinf.envs.sim.g1_mimic.reward import TORSO, differentiable_fk

# Positive extents of the Appendix G uniform command. The paper's Table A.4
# says the encoder sees a normalized command but does not print the divisor.
# Dividing by these extents maps the random box to about [-1, 1].
COMMAND_SCALE = (2.0, 1.32, 3.35)
COMMAND_LOW = (-1.5, -1.32, -3.35)
COMMAND_HIGH = (2.0, 1.32, 3.35)
RESAMPLE_STEPS = 250
ALIGN_WEIGHT = 0.1


def normalize_command(command: torch.Tensor) -> torch.Tensor:
    """Command scaled to roughly unit range for the velocity encoder."""
    scale = command.new_tensor(COMMAND_SCALE)
    return command / scale


def heading_velocity(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Planar velocity in the heading frame, and world-frame yaw rate."""
    quat = state[..., 3:7]
    heading_t = yaw_rotation(heading_yaw(quat)).transpose(-1, -2)
    planar = torch.matmul(heading_t, state[..., ROOT_LIN_VEL].unsqueeze(-1)).squeeze(-1)
    world_ang = torch.matmul(
        quat_to_rotmat(quat), state[..., ROOT_ANG_VEL].unsqueeze(-1)
    ).squeeze(-1)
    return planar[..., :2], world_ang[..., 2]


def sample_velocity_command(
    n: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """50% of the draw is marked as a reference source. The caller fills it.

    The other half is uniform in the Appendix G box, and 10% of that half
    is the standing command. The returned boolean is True for reference.
    """
    from_ref = torch.rand(n, device=device, generator=generator) < 0.5
    command = torch.empty(n, 3, device=device)
    for axis, low, high in zip(range(3), COMMAND_LOW, COMMAND_HIGH, strict=True):
        unit = torch.rand(n, device=device, generator=generator)
        command[:, axis] = low + (high - low) * unit
    stand = (~from_ref) & (torch.rand(n, device=device, generator=generator) < 0.1)
    command = torch.where(stand.unsqueeze(-1), torch.zeros_like(command), command)
    command = torch.where(from_ref.unsqueeze(-1), torch.zeros_like(command), command)
    return command, from_ref


STEP_TEST_PLANES = (("vx-vy", 0, 1), ("vx-wz", 0, 2), ("vy-wz", 1, 2))


def step_test_commands(points: int = 15) -> tuple[torch.Tensor, list[str], list[str]]:
    """Appendix G.2 commands: commands, names, and the plane of each.

    A ``points x points`` grid over the uniform command range on each of the
    planes (vx, vy), (vx, wz) and (vy, wz), with the standing command left
    out: 3 * (15 * 15 - 1) = 672 step responses. A command on an axis shared
    by two planes appears once per plane, as in the paper's count.
    """
    axes = [
        torch.linspace(low, high, points, dtype=torch.float64)
        for low, high in zip(COMMAND_LOW, COMMAND_HIGH, strict=True)
    ]
    # The middle of a symmetric range comes out near 1e-17; it is the zero command.
    axes = [
        torch.where(axis.abs() < 1e-9, torch.zeros_like(axis), axis) for axis in axes
    ]
    commands, names, planes = [], [], []
    for plane, first, second in STEP_TEST_PLANES:
        for u in axes[first]:
            for v in axes[second]:
                command = torch.zeros(3, dtype=torch.float64)
                command[first], command[second] = u, v
                if float(command.abs().max()) < 1e-6:
                    continue
                commands.append(command)
                names.append(
                    f"{plane}:{command[0]:+.3f},{command[1]:+.3f},{command[2]:+.3f}"
                )
                planes.append(plane)
    return torch.stack(commands).float(), names, planes


def velocity_fall(state: torch.Tensor) -> torch.Tensor:
    """Hard fall: root height below 0.35 m, or tilt past about 73 degrees."""
    height = state[..., 2]
    upright = 1.0 + projected_gravity(state[..., 3:7])[..., 2]
    return (height < 0.35) | (upright > 0.7)


def velocity_fall_soft(state: torch.Tensor) -> torch.Tensor:
    """Differentiable fall probability. Hard thresholds stay in the simulator."""
    height = state[..., 2]
    upright = 1.0 + projected_gravity(state[..., 3:7])[..., 2]
    p_height = torch.sigmoid((0.35 - height) / 0.02)
    p_tilt = torch.sigmoid((upright - 0.7) / 0.04)
    return 1.0 - (1.0 - p_height) * (1.0 - p_tilt)


def velocity_reward(
    state: torch.Tensor,
    command: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
) -> torch.Tensor:
    """Table A.6, multiplied by ``Δt = 0.02`` like Table A.2."""
    planar, yaw_rate = heading_velocity(state)
    lin = torch.exp(-((planar - command[..., :2]) ** 2).sum(dim=-1) / 0.25)
    yaw = torch.exp(-((yaw_rate - command[..., 2]) ** 2) / 0.5)
    smooth = (action - prev_action).pow(2).sum(dim=-1)
    q = state[..., JOINT_POS]
    over = (q - joint_high).clamp_min(0.0) + (joint_low - q).clamp_min(0.0)
    limit = over.sum(dim=-1)
    pos, rot, _, ang = differentiable_fk(state)
    del pos
    heading_t = yaw_rotation(heading_yaw(state[..., 3:7])).transpose(-1, -2)
    torso_z = torch.matmul(heading_t, rot[:, TORSO, :, 2].unsqueeze(-1)).squeeze(-1)
    lean = torso_z[..., 1].pow(2)
    torso_rate = torch.matmul(heading_t, ang[:, TORSO].unsqueeze(-1)).squeeze(-1)
    rate = torso_rate[..., 0].pow(2) + torso_rate[..., 1].pow(2)
    total = (
        2.0 * lin + 2.0 * yaw - 0.3 * smooth - 2.0 * limit - 4.0 * lean - 0.05 * rate
    )
    return total * DT
