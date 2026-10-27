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

"""Analytic 93-d observation head (paper Table A.1)."""

from __future__ import annotations

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    HISTORY_LEN,
    JOINT_POS,
    JOINT_VEL,
    ROOT_ANG_VEL,
    SENSOR_NOISE_STD,
    XI_ENCODER,
)
from rlinf.envs.sim.g1_mimic.features import (
    projected_gravity,
    quat_to_rotmat,
    rotmat_to_6d,
)


def relative_root_ori_6d(
    robot_quat: torch.Tensor, ref_quat: torch.Tensor
) -> torch.Tensor:
    """``Rot6D(R_robot^T R_reference)`` at the same time index.

    zita-ch appends this 6-d heading to the 93-d proprioception. It is a
    function of the robot root, so imagination can differentiate it.
    """
    robot = quat_to_rotmat(robot_quat)
    ref = quat_to_rotmat(ref_quat)
    return rotmat_to_6d(torch.matmul(robot.transpose(-1, -2), ref))


def analytic_obs(
    state: torch.Tensor,
    prev_action: torch.Tensor,
    xi: torch.Tensor,
    q0: torch.Tensor | None = None,
    ref_quat: torch.Tensor | None = None,
) -> torch.Tensor:
    """g(s, a_prev, ξ). ``prev_action`` is the clean policy output.

    Passing ``ref_quat`` appends the 6-d relative root orientation and makes
    a 99-d frame. The 93-d prefix is unchanged.
    """
    squeezed = state.dim() == 1
    if squeezed:
        state = state.unsqueeze(0)
        prev_action = prev_action.unsqueeze(0)
        xi = xi.unsqueeze(0)
        if ref_quat is not None and ref_quat.dim() == 1:
            ref_quat = ref_quat.unsqueeze(0)
    if q0 is None:
        q0 = torch.zeros_like(state[..., JOINT_POS])
    elif q0.dim() == 1:
        q0 = q0.expand_as(state[..., JOINT_POS])
    joint = state[..., JOINT_POS] - q0 + xi[..., XI_ENCODER]
    obs = torch.cat(
        (
            joint,
            state[..., JOINT_VEL],
            state[..., ROOT_ANG_VEL],
            projected_gravity(state[..., 3:7]),
            prev_action,
        ),
        dim=-1,
    )
    if ref_quat is not None:
        if ref_quat.dim() == 1 and not squeezed:
            ref_quat = ref_quat.unsqueeze(0)
        obs = torch.cat((obs, relative_root_ori_6d(state[..., 3:7], ref_quat)), dim=-1)
    return obs.squeeze(0) if squeezed else obs


def add_sensor_noise(obs: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Uniform sensor noise from Table A.1. Previous action is not noised.

    The draw is ``U[-s, s]`` on joint position, joint velocity, base angular
    velocity, and projected gravity. It is not Gaussian. Action noise applied
    to the PD target is a separate Gaussian and does not go through here.
    Noise is an exogenous sample: it is added to ``obs`` and does not detach
    the observation, so imagination can still backpropagate through the state.
    """
    if not enabled:
        return obs
    span = torch.zeros_like(obs)
    span[..., 0:29] = SENSOR_NOISE_STD["joint_pos"]
    span[..., 29:58] = SENSOR_NOISE_STD["joint_vel"]
    span[..., 58:61] = SENSOR_NOISE_STD["gyro"]
    span[..., 61:64] = SENSOR_NOISE_STD["gravity"]
    # Previous action is 64:93. A 99-d frame's relative orientation is 93:99
    # and is left at zero: Table A.1 does not define noise for it.
    unit = torch.rand_like(obs) * 2.0 - 1.0
    return obs + unit * span


def pad_history(frames: list[torch.Tensor]) -> torch.Tensor:
    """Repeat the oldest available frame until the stack has ``HISTORY_LEN``.

    Reset uses this with one freshly noised frame, so the repeated prefix
    shares that single noise sample. Later steps append a new noised frame
    and do not redraw noise on the frames already stored.
    """
    if not frames:
        raise ValueError("history needs at least one observation, got an empty list")
    first = frames[0]
    missing = HISTORY_LEN - len(frames)
    if missing > 0:
        frames = [first] * missing + list(frames)
    stacked = torch.stack(frames[-HISTORY_LEN:], dim=-2)
    width = int(stacked.shape[-1])
    return stacked.reshape(*stacked.shape[:-2], HISTORY_LEN * width)
