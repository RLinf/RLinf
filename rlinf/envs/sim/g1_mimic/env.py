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

"""Vectorized G1 mimic env. Analytic backend is the default for tests."""

from __future__ import annotations

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    DT,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    JOINT_POS,
    RESIDUAL_DIM,
    STATE_DIM,
    XI_DIM,
)
from rlinf.envs.sim.g1_mimic.features import default_state, integrate_residual
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.reward import analytic_reward, failure_signals


def sample_xi(num: int, device, dtype) -> torch.Tensor:
    xi = torch.zeros(num, XI_DIM, device=device, dtype=dtype)
    xi[:, 0] = torch.empty(num, device=device, dtype=dtype).uniform_(0.3, 1.2)
    xi[:, 1:30] = torch.empty(num, 29, device=device, dtype=dtype).uniform_(-0.01, 0.01)
    xi[:, 30] = torch.empty(num, device=device, dtype=dtype).uniform_(-0.025, 0.025)
    xi[:, 31:33] = torch.empty(num, 2, device=device, dtype=dtype).uniform_(-0.05, 0.05)
    return xi


class AnalyticG1Env:
    """Closed-form stand-in that keeps SuperMimic tensor contracts.

    It is not MuJoCo Warp. ``env.backend: mjlab`` is required for the
    paper-scale runs. Smoke and unit tests use this backend so they do not
    change Eq. 2 / Eq. 3, episode-end codes, or the two action streams.
    """

    def __init__(
        self,
        num_envs: int,
        motion: MotionLibrary,
        device: torch.device,
        max_episode_seconds: float = 10.0,
        termination_body: str = "pelvis",
        eval_mode: bool = False,
    ):
        self.num_envs = num_envs
        self.motion = motion
        self.device = device
        self.dtype = torch.float32
        self.max_steps = int(round(max_episode_seconds / DT))
        self.termination_body = termination_body
        self.eval_mode = eval_mode
        self.q0 = motion.q0.to(device)
        self.state = default_state(num_envs, device, self.dtype)
        self.xi = sample_xi(num_envs, device, self.dtype)
        self.motion_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.motion_time = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.time_in_episode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset(torch.arange(num_envs, device=device))

    def ref_state(self) -> torch.Tensor:
        return self.motion.state_at(self.motion_id, self.motion_time)

    def reset(
        self,
        env_ids: torch.Tensor,
        motion_id: torch.Tensor | None = None,
        motion_time: torch.Tensor | None = None,
    ) -> None:
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
        start = self.motion.state_at(motion_id, motion_time).clone()
        start[..., JOINT_POS] = start[..., JOINT_POS] + 0.02 * torch.randn_like(
            start[..., JOINT_POS]
        ) * (0.0 if self.eval_mode else 1.0)
        self.state[env_ids] = start

    def step(self, action_executed: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = torch.zeros(
            self.num_envs, RESIDUAL_DIM, device=self.device, dtype=self.dtype
        )
        residual[:, 6:35] = 0.15 * action_executed
        next_state = integrate_residual(self.state, residual)
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
        end_kind = torch.where(clip_end, torch.full_like(end_kind, EPISODE_CLIP_END), end_kind)
        end_kind = torch.where(timeout, torch.full_like(end_kind, EPISODE_TIMEOUT), end_kind)
        end_kind = torch.where(failed, torch.full_like(end_kind, EPISODE_FAILURE), end_kind)
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


def try_make_mjlab_env(**kwargs):
    """Load the mjlab backend when the optional extra is installed."""
    try:
        from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env
    except ImportError as exc:
        raise ImportError(
            "env.backend=mjlab requires the optional mjlab extra. "
            "Install from requirements/embodied/envs/supermimic.txt "
            "(commit 8bb2bc9)."
        ) from exc
    return MjlabG1Env(**kwargs)


def make_env(backend: str, **kwargs):
    if backend == "analytic":
        return AnalyticG1Env(**kwargs)
    if backend == "mjlab":
        return try_make_mjlab_env(**kwargs)
    raise ValueError(f"unknown SuperMimic env backend {backend!r}")


# Keep ACTION_DIM / STATE_DIM referenced for shape documentation.
_ = (ACTION_DIM, STATE_DIM, analytic_reward)
