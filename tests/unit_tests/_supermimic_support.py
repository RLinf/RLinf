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

"""Test doubles for the SuperMimic tests: a CPU G1 stand-in and synthetic clips.

Importing this module registers the ``analytic`` env backend in this process.
Production code registers only ``mjlab``, so a training config cannot select
the stand-in by accident.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    DT,
    G1_JOINT_NAMES,
    JOINT_POS,
    JOINT_VEL,
    RESIDUAL_DIM,
)
from rlinf.envs.sim.g1_mimic.env import (
    compose_end_kind,
    register_env_backend,
    sample_xi,
)
from rlinf.envs.sim.g1_mimic.features import default_state, integrate_residual
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.reward import failure_signals


class AnalyticG1Env:
    """Closed-form CPU stand-in with the simulator's tensor contract.

    It is not MuJoCo Warp: an action moves the joints directly. Tests use it
    for episode-end codes, replay, Eq. 2 / Eq. 3, and the two action streams.
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

    def reset_standing(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        if n == 0:
            return
        self.motion_id[env_ids] = 0
        self.motion_time[env_ids] = 0
        self.time_in_episode[env_ids] = 0
        self.xi[env_ids] = sample_xi(n, self.device, self.dtype)
        self.state[env_ids] = default_state(n, self.device, self.dtype)

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


register_env_backend("analytic", AnalyticG1Env)


def synthetic_clip_states(
    num_clips: int = 40, clip_len: int = 200
) -> list[torch.Tensor]:
    """Deterministic 50 Hz G1 states: forward drift and sinusoidal joints."""
    clips = []
    t = torch.arange(clip_len, dtype=torch.float32)
    for i in range(num_clips):
        state = default_state(clip_len, "cpu", torch.float32)
        freq = 0.04 + 0.01 * (i % 5)
        state[:, 0] = 0.02 * t
        state[:, JOINT_POS] = 0.25 * torch.sin(2 * torch.pi * freq * t).unsqueeze(-1)
        state[:, JOINT_VEL] = (
            0.25 * 2 * torch.pi * freq * torch.cos(2 * torch.pi * freq * t)
        ).unsqueeze(-1)
        clips.append(state)
    return clips


def synthetic_clips(
    num_clips: int = 40, clip_len: int = 200, device: torch.device | str = "cpu"
) -> MotionLibrary:
    return MotionLibrary(
        [c.to(device) for c in synthetic_clip_states(num_clips, clip_len)]
    )


_CLIP_DIRS: dict[tuple[int, int], str] = {}


def synthetic_clip_dir(num_clips: int = 40, clip_len: int = 200) -> str:
    """A directory of ``clip_*.pt`` payloads that the production loader reads.

    Written once per process and removed at exit.
    """
    key = (num_clips, clip_len)
    if key not in _CLIP_DIRS:
        directory = Path(tempfile.mkdtemp(prefix="supermimic_clips_"))
        atexit.register(shutil.rmtree, directory, True)
        for i, state in enumerate(synthetic_clip_states(num_clips, clip_len)):
            torch.save(
                {
                    "fps": 50.0,
                    "joint_names": list(G1_JOINT_NAMES),
                    "q0": torch.zeros(len(G1_JOINT_NAMES)),
                    "state": state,
                },
                directory / f"clip_{i:03d}.pt",
            )
        _CLIP_DIRS[key] = str(directory)
    return _CLIP_DIRS[key]
