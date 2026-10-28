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

"""Evaluation rollouts without training noise.

``run_tracking_eval`` plays reference clips from frame 0.
``run_velocity_eval`` holds fixed velocity commands until a fall.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    DT,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    HISTORY_LEN,
    MAX_TRACKING_REWARD,
)
from rlinf.envs.sim.g1_mimic.obs import analytic_obs
from rlinf.envs.sim.g1_mimic.reward import (
    TORSO,
    analytic_reward,
    differentiable_fk,
    xy_yaw_drift,
)
from rlinf.envs.sim.g1_mimic.velocity import (
    heading_velocity,
    normalize_command,
    velocity_fall,
)


class RollingHistory:
    """Last ``HISTORY_LEN`` observation frames per env, ``[N, 10, D]``.

    ``fill`` repeats one frame over the window, so a reset shares a single
    noise sample. ``push`` drops the oldest frame and keeps the stored ones
    as they were.
    """

    def __init__(self, first: torch.Tensor, length: int = HISTORY_LEN):
        self.frames = first.unsqueeze(1).expand(-1, length, -1).clone()

    @property
    def length(self) -> int:
        """Number of frames kept (10)."""
        return int(self.frames.shape[1])

    def flat(self) -> torch.Tensor:
        """History flattened to ``[num_envs, 10 * obs_dim]``."""
        return self.frames.reshape(self.frames.shape[0], -1)

    def push(self, frame: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Append one frame per env (or per ``env_ids``) and drop the oldest."""
        if env_ids is None:
            self.frames = torch.cat((self.frames[:, 1:], frame.unsqueeze(1)), dim=1)
        else:
            rows = self.frames[env_ids]
            self.frames[env_ids] = torch.cat((rows[:, 1:], frame.unsqueeze(1)), dim=1)
        if self.frames.shape[1] != HISTORY_LEN:
            raise RuntimeError(
                f"observation history must hold {HISTORY_LEN} frames, "
                f"has {self.frames.shape[1]}"
            )

    def fill(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        """Fill the whole history of ``env_ids`` with one frame, as after a reset."""
        self.frames[env_ids] = frame.unsqueeze(1).expand(-1, self.length, -1)


def _masked_add(
    total: torch.Tensor, value: torch.Tensor, running: torch.Tensor
) -> torch.Tensor:
    """Add ``value`` only on active rows. ``NaN * 0`` is not used."""
    return torch.where(running, total + value, total)


def _masked_max(
    total: torch.Tensor, value: torch.Tensor, running: torch.Tensor
) -> torch.Tensor:
    return torch.where(running, torch.maximum(total, value), total)


def _finite_inactive(next_state, ref_state, running):
    """Keep inactive rows finite so a later NaN cannot enter FK or the totals."""
    next_state = next_state.clone()
    ref_state = ref_state.clone()
    inactive = ~running
    if bool(inactive.any()):
        replacement = ref_state[inactive]
        bad = ~torch.isfinite(replacement).all(dim=-1)
        if bool(bad.any()):
            replacement = replacement.clone()
            replacement[bad] = 0
            replacement[bad, 3] = 1
        finite_next = next_state[inactive]
        use_ref = ~torch.isfinite(finite_next).all(dim=-1)
        if bool(use_ref.any()):
            finite_next = finite_next.clone()
            finite_next[use_ref] = replacement[use_ref]
        next_state[inactive] = finite_next
        ref_state[inactive] = replacement
    return next_state, ref_state


def _rot_angle(rot: torch.Tensor, ref_rot: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(ref_rot.transpose(-1, -2), rot)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos)


def run_tracking_eval(
    env: Any,
    policy: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q0: torch.Tensor,
    goal_fn: Callable,
    joint_low: torch.Tensor | None = None,
    joint_high: torch.Tensor | None = None,
    active: int | None = None,
    include_ref_ori: bool = False,
) -> dict[str, torch.Tensor]:
    """Roll each active env from frame 0 until failure or the clip end.

    No sensor noise and no action noise. The normalized-return denominator
    is ``0.1`` times the clip's full control-step count, not the number of
    steps that actually ran. An early failure therefore keeps the unplayed
    tail in the denominator.

    A clip of T stored frames is executed at reference times ``0 .. T-1``.
    The step that advances time from ``T-1`` to ``T`` is the last control,
    and ``past_clip_end`` ends the episode there. The scheduled length is
    therefore T control steps, not T-1.

    ``active`` lets a short final batch use the first environments of a
    larger vector. Bodies:

    * ``torso_*`` is the reward root (``torso_link``).
    * ``free_root_*`` is the floating base. On G1 that base is the pelvis,
      which is also the failure body, and it is not the torso.
    """
    n_env = env.num_envs
    n = n_env if active is None else int(active)
    if not bool(torch.all(env.motion_time[:n] == 0)):
        raise RuntimeError(
            "tracking eval must start at reference frame 0, got frames "
            f"{env.motion_time[:n].unique().tolist()}"
        )
    device = env.state.device
    prev = torch.zeros(n_env, q0.shape[-1], device=device)
    returns = torch.zeros(n_env, device=device)
    body = torch.zeros(n_env, device=device)
    root_pos_err = torch.zeros(n_env, device=device)
    root_ori_err = torch.zeros(n_env, device=device)
    free_pos_err = torch.zeros(n_env, device=device)
    free_ori_err = torch.zeros(n_env, device=device)
    drift_xy = torch.zeros(n_env, device=device)
    drift_yaw = torch.zeros(n_env, device=device)
    steps = torch.zeros(n_env, device=device)
    fail = torch.zeros(n_env, device=device)
    fail_t = torch.zeros(n_env, device=device)
    clip_end = torch.zeros(n_env, device=device)
    alive = torch.zeros(n_env, dtype=torch.bool, device=device)
    alive[:n] = True
    scheduled = env.motion.clip_length(env.motion_id[:n]).to(
        device=device, dtype=torch.float32
    )
    max_len = int(scheduled.max().item()) if n else 0
    obs0 = _eval_obs(env.state, prev, env.xi, q0, env.ref_state(), include_ref_ori)
    # Evaluation stores the clean frame. Reset repeats that one frame.
    # The window stays at HISTORY_LEN; finished clips do not grow it.
    history = RollingHistory(obs0)
    for t in range(max_len):
        if not bool(alive.any()):
            break
        goal = goal_fn(env.motion_id, env.motion_time, env.state)
        action = policy(history.flat(), goal)
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = torch.where(alive.unsqueeze(-1), action, torch.zeros_like(action))
        result = env.step(action)
        running = alive
        next_state, ref_state = _finite_inactive(
            result["next_state"], result["ref_state"], running
        )
        pack = analytic_reward(
            next_state,
            ref_state,
            action,
            prev,
            joint_low,
            joint_high,
        )
        xy, yaw = xy_yaw_drift(next_state, ref_state)
        free_pos = (next_state[..., :3] - ref_state[..., :3]).norm(dim=-1)
        free_ori = _quat_geodesic(next_state[..., 3:7], ref_state[..., 3:7])
        body_pos, body_rot, _, _ = differentiable_fk(next_state)
        ref_body_pos, ref_body_rot, _, _ = differentiable_fk(ref_state)
        torso_pos = (body_pos[:, TORSO] - ref_body_pos[:, TORSO]).norm(dim=-1)
        torso_ori = _rot_angle(body_rot[:, TORSO], ref_body_rot[:, TORSO])
        returns = _masked_add(returns, pack["reward"], running)
        body = _masked_add(body, pack["body_tracking"], running)
        root_pos_err = _masked_add(root_pos_err, torso_pos, running)
        root_ori_err = _masked_add(root_ori_err, torso_ori, running)
        free_pos_err = _masked_add(free_pos_err, free_pos, running)
        free_ori_err = _masked_add(free_ori_err, free_ori, running)
        drift_xy = _masked_max(drift_xy, xy, running)
        drift_yaw = _masked_max(drift_yaw, yaw, running)
        steps = torch.where(running, steps + 1, steps)
        new_fail = running & (result["end_kind"] == EPISODE_FAILURE)
        new_end = running & (result["end_kind"] == EPISODE_CLIP_END)
        fail = fail + new_fail.float()
        clip_end = clip_end + new_end.float()
        fail_t = torch.where(new_fail, torch.full_like(fail_t, float(t)), fail_t)
        alive = alive & (result["end_kind"] == EPISODE_CONT)
        prev = torch.where(running.unsqueeze(-1), action, prev)
        fresh = _eval_obs(
            env.state, prev, env.xi, q0, result["ref_state"], include_ref_ori
        )
        keep = history.frames[:, -1]
        use = running.unsqueeze(-1) & torch.isfinite(fresh)
        history.push(torch.where(use, fresh, keep))
    step_denom = steps[:n].clamp_min(1.0)
    scheduled = scheduled.clamp_min(1.0)
    return {
        "clip_id": env.motion_id[:n].detach().clone().to(dtype=torch.long),
        "normalized_return": (returns[:n] / (MAX_TRACKING_REWARD * scheduled)).detach(),
        "body_tracking": body[:n].detach(),
        "torso_pos_err": (root_pos_err[:n] / step_denom).detach(),
        "torso_ori_err": (root_ori_err[:n] / step_denom).detach(),
        "free_root_pos_err": (free_pos_err[:n] / step_denom).detach(),
        "free_root_ori_err": (free_ori_err[:n] / step_denom).detach(),
        "free_root_xy_drift": drift_xy[:n].detach(),
        "free_root_yaw_drift": drift_yaw[:n].detach(),
        "fail": fail[:n].detach().long(),
        "clip_end": clip_end[:n].detach().long(),
        "fail_step": fail_t[:n].detach().long(),
        "fail_time_s": (fail_t[:n] + 1.0) * DT * fail[:n],
        "episode_length_s": steps[:n].detach() * DT,
        "steps": steps[:n].detach().long(),
        "return_sum": returns[:n].detach(),
        "scheduled_steps": scheduled.detach().long(),
    }


def summarize_tracking_eval(rows: dict[str, torch.Tensor]) -> dict[str, float]:
    """Unweighted mean over clips. One row per clip, so batch size does not matter."""
    fail = rows["fail"].float()
    fail_count = fail.sum().clamp_min(1.0)
    return {
        "eval/normalized_return": float(rows["normalized_return"].mean()),
        "eval/body_tracking": float(rows["body_tracking"].mean()),
        "eval/torso_pos_err": float(rows["torso_pos_err"].mean()),
        "eval/torso_ori_err": float(rows["torso_ori_err"].mean()),
        "eval/free_root_pos_err": float(rows["free_root_pos_err"].mean()),
        "eval/free_root_ori_err": float(rows["free_root_ori_err"].mean()),
        "eval/free_root_xy_drift": float(rows["free_root_xy_drift"].mean()),
        "eval/free_root_yaw_drift": float(rows["free_root_yaw_drift"].mean()),
        "eval/fail_rate": float(fail.mean()),
        "eval/fail_step": float((rows["fail_step"].float() * fail).sum() / fail_count),
        "eval/fail_time_s": float(rows["fail_time_s"].sum() / fail_count),
        "eval/episode_length_s": float(rows["episode_length_s"].mean()),
        "eval/clip_end_rate": float(rows["clip_end"].float().mean()),
        "eval/steps": float(rows["steps"].float().mean()),
        "eval/num_clips": float(rows["clip_id"].shape[0]),
    }


def _eval_obs(state, prev, xi, q0, ref_state, include_ref_ori: bool) -> torch.Tensor:
    ref_quat = ref_state[..., 3:7] if include_ref_ori else None
    return analytic_obs(state, prev, xi, q0, ref_quat=ref_quat)


def _quat_geodesic(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    dot = (q * ref).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def run_velocity_eval(
    env: Any,
    velocity_policy: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q0: torch.Tensor,
    commands: torch.Tensor,
    steps: int,
) -> dict[str, torch.Tensor]:
    """Hold one fixed command per env for ``steps`` control steps.

    The env must already be reset; its reference clip only supplies the
    start pose. Motion-tracking failure and clip end are ignored, as in
    velocity training: only ``velocity_fall`` ends a rollout. Errors are
    the heading-frame planar speed error and the yaw-rate error, averaged
    over the steps before the fall. No sensor or action noise.
    """
    n = env.num_envs
    if commands.shape != (n, 3):
        raise ValueError(f"commands must be [{n}, 3], got {tuple(commands.shape)}")
    device = env.state.device
    commands = commands.to(device=device, dtype=torch.float32)
    norm = normalize_command(commands)
    prev = torch.zeros(n, q0.shape[-1], device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    lin_err = torch.zeros(n, device=device)
    yaw_err = torch.zeros(n, device=device)
    count = torch.zeros(n, device=device)
    fall = torch.zeros(n, device=device)
    fall_t = torch.zeros(n, device=device)
    history = RollingHistory(
        _eval_obs(env.state, prev, env.xi, q0, env.ref_state(), False)
    )
    for t in range(int(steps)):
        if not bool(alive.any()):
            break
        action = velocity_policy(history.flat(), norm)
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = torch.where(alive.unsqueeze(-1), action, torch.zeros_like(action))
        result = env.step(action)
        next_state, _ = _finite_inactive(
            result["next_state"], result["ref_state"], alive
        )
        planar, yaw_rate = heading_velocity(next_state)
        lin_err = _masked_add(lin_err, (planar - commands[:, :2]).norm(dim=-1), alive)
        yaw_err = _masked_add(yaw_err, (yaw_rate - commands[:, 2]).abs(), alive)
        count = torch.where(alive, count + 1, count)
        new_fall = alive & (
            velocity_fall(next_state) | ~torch.isfinite(next_state).all(dim=-1)
        )
        fall = fall + new_fall.float()
        fall_t = torch.where(new_fall, torch.full_like(fall_t, float(t)), fall_t)
        running = alive
        alive = alive & ~new_fall
        prev = torch.where(running.unsqueeze(-1), action, prev)
        fresh = _eval_obs(env.state, prev, env.xi, q0, result["ref_state"], False)
        keep = history.frames[:, -1]
        use = alive.unsqueeze(-1) & torch.isfinite(fresh)
        history.push(torch.where(use, fresh, keep))
    denom = count.clamp_min(1.0)
    return {
        "command": commands.detach().clone(),
        "lin_vel_err": (lin_err / denom).detach(),
        "yaw_rate_err": (yaw_err / denom).detach(),
        "fall": fall.detach().long(),
        "fall_time_s": ((fall_t + 1.0) * DT * fall).detach(),
        "survival_s": (count * DT).detach(),
        "scheduled_s": torch.full_like(count, float(steps) * DT),
    }


def summarize_velocity_eval(rows: dict[str, torch.Tensor]) -> dict[str, float]:
    """Unweighted mean over the smoke commands. Not the Appendix G.2 metric."""
    return {
        "eval_velocity_smoke/lin_vel_err": float(rows["lin_vel_err"].mean()),
        "eval_velocity_smoke/yaw_rate_err": float(rows["yaw_rate_err"].mean()),
        "eval_velocity_smoke/fall_rate": float(rows["fall"].float().mean()),
        "eval_velocity_smoke/survival_s": float(rows["survival_s"].mean()),
        "eval_velocity_smoke/num_commands": float(rows["command"].shape[0]),
    }


def run_velocity_step_test(
    env: Any,
    velocity_policy: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q0: torch.Tensor,
    commands: torch.Tensor,
    steps: int,
    steady_steps: int,
) -> dict[str, torch.Tensor]:
    """One Appendix G.2 trial per env: hold a command from the current state.

    The relative steady-state error is the norm of the mean velocity error
    ``(v_xy - c_xy, w_z - c_z)`` over the last ``steady_steps`` control steps,
    divided by the norm of the command. Planar velocity is in the heading
    frame and the yaw rate in the world frame. A rollout that falls is
    reported as fallen with a NaN error. No sensor or action noise.
    """
    n = env.num_envs
    if commands.shape != (n, 3):
        raise ValueError(f"commands must be [{n}, 3], got {tuple(commands.shape)}")
    if not 0 < steady_steps <= steps:
        raise ValueError(f"steady_steps must be in (0, {steps}], got {steady_steps}")
    device = env.state.device
    commands = commands.to(device=device, dtype=torch.float32)
    norm = normalize_command(commands)
    prev = torch.zeros(n, q0.shape[-1], device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    error_sum = torch.zeros(n, 3, device=device)
    history = RollingHistory(
        _eval_obs(env.state, prev, env.xi, q0, env.ref_state(), False)
    )
    for t in range(int(steps)):
        action = velocity_policy(history.flat(), norm)
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = torch.where(alive.unsqueeze(-1), action, torch.zeros_like(action))
        result = env.step(action)
        next_state, _ = _finite_inactive(
            result["next_state"], result["ref_state"], alive
        )
        planar, yaw_rate = heading_velocity(next_state)
        error = torch.cat(
            (planar - commands[:, :2], (yaw_rate - commands[:, 2]).unsqueeze(-1)),
            dim=-1,
        )
        if t >= steps - steady_steps:
            error_sum = torch.where(alive.unsqueeze(-1), error_sum + error, error_sum)
        fallen = velocity_fall(next_state) | ~torch.isfinite(next_state).all(dim=-1)
        running = alive
        alive = alive & ~fallen
        prev = torch.where(running.unsqueeze(-1), action, prev)
        fresh = _eval_obs(env.state, prev, env.xi, q0, result["ref_state"], False)
        keep = history.frames[:, -1]
        history.push(
            torch.where(alive.unsqueeze(-1) & torch.isfinite(fresh), fresh, keep)
        )
    relative = (error_sum / float(steady_steps)).norm(dim=-1) / commands.norm(dim=-1)
    fell = ~alive
    return {
        "command": commands.detach().clone(),
        "relative_error": torch.where(
            fell, torch.full_like(relative, float("nan")), relative
        ),
        "fell": fell.to(torch.long),
    }


def aggregate_step_test(
    relative_error: torch.Tensor, fell: torch.Tensor, planes: list[str]
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Per-command median over the trials that did not fall, then a summary.

    ``relative_error`` and ``fell`` are ``[trials, commands]``. A command
    whose every trial fell has a NaN value and is counted separately, as the
    paper draws it at the cap.
    """
    per_command = torch.nanmedian(relative_error, dim=0).values
    fell_trials = fell.sum(dim=0)
    all_fell = fell_trials == fell.shape[0]
    scored = per_command[~all_fell]
    summary = {
        "eval_velocity_g2/median_relative_error": float(scored.median())
        if scored.numel()
        else float("nan"),
        "eval_velocity_g2/commands_all_trials_fell": float(all_fell.sum()),
        "eval_velocity_g2/trial_fall_rate": float(fell.float().mean()),
        "eval_velocity_g2/num_commands": float(relative_error.shape[1]),
        "eval_velocity_g2/num_trials": float(relative_error.shape[0]),
    }
    plane_ids = sorted(set(planes))
    for plane in plane_ids:
        mask = (
            torch.tensor([p == plane for p in planes], device=per_command.device)
            & ~all_fell
        )
        values = per_command[mask]
        summary[f"eval_velocity_g2/median_relative_error_{plane}"] = (
            float(values.median()) if values.numel() else float("nan")
        )
    rows = {
        "relative_error": per_command,
        "fell_trials": fell_trials,
        **{
            f"trial_{i}_relative_error": relative_error[i]
            for i in range(relative_error.shape[0])
        },
    }
    return summary, rows
