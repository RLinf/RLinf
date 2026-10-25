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

"""One reference-tracking rollout from frame 0, without training noise."""

from __future__ import annotations

from collections.abc import Callable

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


def _rot_angle(rot: torch.Tensor, ref_rot: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(ref_rot.transpose(-1, -2), rot)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos)


def run_tracking_eval(
    env,
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
        raise RuntimeError("tracking eval must start at reference frame 0")
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
    scheduled = env.motion.clip_length(env.motion_id[:n]).to(device=device, dtype=torch.float32)
    max_len = int(scheduled.max().item()) if n else 0
    obs0 = _eval_obs(env.state, prev, env.xi, q0, env.ref_state(), include_ref_ori)
    # Evaluation stores the clean frame. Reset repeats that one frame.
    hist_frames = [obs0 for _ in range(HISTORY_LEN)]
    for t in range(max_len):
        if not bool(alive.any()):
            break
        history = torch.stack(hist_frames[-HISTORY_LEN:], dim=1).reshape(n_env, -1)
        goal = goal_fn(env.motion_id, env.motion_time, env.state)
        action = policy(history, goal)
        result = env.step(action)
        pack = analytic_reward(
            result["next_state"],
            result["ref_state"],
            action,
            prev,
            joint_low,
            joint_high,
        )
        xy, yaw = xy_yaw_drift(result["next_state"], result["ref_state"])
        free_pos = (result["next_state"][..., :3] - result["ref_state"][..., :3]).norm(dim=-1)
        free_ori = _quat_geodesic(
            result["next_state"][..., 3:7], result["ref_state"][..., 3:7]
        )
        body_pos, body_rot, _, _ = differentiable_fk(result["next_state"])
        ref_body_pos, ref_body_rot, _, _ = differentiable_fk(result["ref_state"])
        torso_pos = (body_pos[:, TORSO] - ref_body_pos[:, TORSO]).norm(dim=-1)
        torso_ori = _rot_angle(body_rot[:, TORSO], ref_body_rot[:, TORSO])
        running = alive
        returns = returns + pack["reward"] * running
        body = body + pack["body_tracking"] * running
        root_pos_err = root_pos_err + torso_pos * running
        root_ori_err = root_ori_err + torso_ori * running
        free_pos_err = free_pos_err + free_pos * running
        free_ori_err = free_ori_err + free_ori * running
        drift_xy = torch.maximum(drift_xy, xy * running)
        drift_yaw = torch.maximum(drift_yaw, yaw * running)
        steps = steps + running.float()
        new_fail = running & (result["end_kind"] == EPISODE_FAILURE)
        new_end = running & (result["end_kind"] == EPISODE_CLIP_END)
        fail = fail + new_fail.float()
        clip_end = clip_end + new_end.float()
        fail_t = torch.where(new_fail, torch.full_like(fail_t, float(t)), fail_t)
        alive = alive & (result["end_kind"] == EPISODE_CONT)
        prev = action
        hist_frames.append(
            _eval_obs(env.state, prev, env.xi, q0, result["ref_state"], include_ref_ori)
        )
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
        "fail": fail[:n].detach(),
        "clip_end": clip_end[:n].detach(),
        "fail_step": fail_t[:n].detach(),
        "fail_time_s": (fail_t[:n] + 1.0) * DT * fail[:n],
        "episode_length_s": steps[:n].detach() * DT,
        "steps": steps[:n].detach(),
        "return_sum": returns[:n].detach(),
        "scheduled_steps": scheduled.detach(),
    }


def summarize_tracking_eval(rows: dict[str, torch.Tensor]) -> dict[str, float]:
    """Unweighted mean over clips. One row per clip, so batch size does not matter."""
    fail = rows["fail"]
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
        "eval/fail_step": float((rows["fail_step"] * fail).sum() / fail_count),
        "eval/fail_time_s": float(rows["fail_time_s"].sum() / fail_count),
        "eval/episode_length_s": float(rows["episode_length_s"].mean()),
        "eval/clip_end_rate": float(rows["clip_end"].mean()),
        "eval/steps": float(rows["steps"].mean()),
        "eval/num_clips": float(rows["clip_id"].shape[0]),
    }


def _eval_obs(state, prev, xi, q0, ref_state, include_ref_ori: bool) -> torch.Tensor:
    ref_quat = ref_state[..., 3:7] if include_ref_ori else None
    return analytic_obs(state, prev, xi, q0, ref_quat=ref_quat)


def _quat_geodesic(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    dot = (q * ref).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)
