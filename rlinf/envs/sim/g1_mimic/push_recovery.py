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

"""Stationary standing-pose push recovery.

Paper §4.3 says a policy holds a static T-pose and receives velocity pushes
from 0.5 to 4 m/s. It reports that the critic policy settles in 0.5–2.4 s.
It does not publish the T-pose joint angles, the speed grid, the direction,
the trial count, the pre-push wait, or the numerical meaning of "settles".

This module does not implement that T-pose. It holds mjlab's knees-bent
keyframe, the same pose as ``q0`` and ``reset_standing``. Forward kinematics
puts each wrist about 0.31 m below its shoulder and only about 0.07 m to
the side. The asset has no T-pose keyframe, and the MJCF rest pose points
the forearms forward rather than out to the sides. Because the paper does
not publish the joint angles, this code does not invent a horizontal-arm
pose and call it the paper pose. The protocol name is
``stationary_standing_pose``.

The push adds (or sets) a horizontal world-frame root linear velocity.
Recovery, the stable window, and the pre-push wait below are this
implementation's criteria. A fall is the training ``failure_signals`` test
against that static reference.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    DT,
    EPISODE_CLIP_END,
    EPISODE_FAILURE,
    JOINT_POS,
    JOINT_VEL,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
    STATE_DIM,
)
from rlinf.envs.sim.g1_mimic.features import (
    heading_yaw,
    projected_gravity,
    yaw_rotation,
)
from rlinf.envs.sim.g1_mimic.motion import GOAL_FRAMES, MotionLibrary

# Paper §4.3 names the range. It does not name this grid or a trial count.
DEFAULT_SPEEDS = (0.5, 1.0, 2.0, 4.0)
REFERENCE_POSE = "mjlab_knees_bent_keyframe"
PROTOCOL_NAME = "stationary_standing_pose"
OUTCOMES = ("recovered", "fallen", "timeout", "initialization_failed")

PAPER_PROTOCOL = {
    "section": "4.3",
    "pose": "static T-pose",
    "implemented_protocol": PROTOCOL_NAME,
    "implemented_pose": REFERENCE_POSE,
    "push": "velocity push",
    "speed_range_m_s": [0.5, 4.0],
    "reported_settle_s": [0.5, 2.4],
    "unpublished": [
        "joint angles of the T-pose",
        "speed grid inside 0.5-4 m/s",
        "push direction and frame",
        "whether the push is added or assigned",
        "trials per speed",
        "pre-push wait",
        "numerical settle threshold and stable window",
    ],
}


@dataclass(frozen=True)
class RecoveryCriteria:
    """Thresholds this implementation uses. None of them are in §4.3."""

    speed_threshold: float = 0.15
    tilt_threshold: float = 0.35
    torso_ori_threshold: float = 0.5
    stable_seconds: float = 0.5
    max_xy_drift: float | None = None
    dt: float = DT

    def stable_frames(self) -> int:
        """Consecutive post-push steps that must all be stable.

        ``recovery_time_s`` is the time of the step that completes this
        window, measured from the push. The first step of the window is
        ``stable_onset_time_s``.
        """
        frames = int(round(float(self.stable_seconds) / float(self.dt)))
        return max(frames, 1)


def default_standing_state() -> torch.Tensor:
    """71-d static pose: mjlab's knees-bent keyframe, velocities zero.

    This is the standing pose the simulator, the PD offset, and ``q0`` share.
    It is not the paper's T-pose. Section 4.3 does not publish those joints,
    and this asset has no T-pose keyframe to reuse.
    """
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state

    state = asset_default_state().float().reshape(STATE_DIM).clone()
    state[ROOT_LIN_VEL] = 0
    state[ROOT_ANG_VEL] = 0
    state[JOINT_VEL] = 0
    return state


def hold_pose_library(state: torch.Tensor, steps: int) -> MotionLibrary:
    """Repeat one finite pose for ``steps`` control frames. Velocities are zero."""
    pose = state.detach().float().reshape(STATE_DIM).clone()
    if not torch.isfinite(pose).all():
        raise ValueError("static reference state is not finite")
    pose[ROOT_LIN_VEL] = 0
    pose[ROOT_ANG_VEL] = 0
    pose[JOINT_VEL] = 0
    count = int(steps)
    if count < 2:
        raise ValueError(f"a static reference needs at least 2 frames, got {count}")
    frames = pose.unsqueeze(0).expand(count, STATE_DIM).clone()
    library = MotionLibrary([frames], q0=pose[JOINT_POS].clone())
    library.clip_names = ["static_standing"]
    return library


def reference_steps(settle_steps: int, max_steps: int) -> int:
    """Frames so the goal window and the post-push budget stay inside the clip."""
    return int(settle_steps) + int(max_steps) + GOAL_FRAMES + 2


def control_steps(seconds: float) -> int:
    """Control steps in ``seconds``. Zero seconds is zero steps."""
    if seconds < 0 or not math.isfinite(seconds):
        raise ValueError(f"duration must be finite and >= 0, got {seconds}")
    if seconds == 0:
        return 0
    return int(float(seconds) / DT + 0.5)


def parse_direction(direction) -> torch.Tensor | None:
    """``None`` means sample a heading. A pair is a unitless xy direction."""
    if direction is None or (
        isinstance(direction, str) and direction.strip().lower() == "random"
    ):
        return None
    if isinstance(direction, str):
        raise ValueError(
            "eval.direction must be 'random' or [x, y], got " + repr(direction)
        )
    values = [float(item) for item in direction]
    if len(values) != 2 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"push direction must be two finite numbers, got {values}")
    vector = torch.tensor(values, dtype=torch.float32)
    if float(vector.norm()) < 1e-8:
        raise ValueError("push direction must be non-zero")
    return vector / vector.norm()


def check_push_protocol(
    speeds,
    trials: int,
    settle_seconds: float,
    max_seconds: float,
    criteria: RecoveryCriteria,
    frame: str,
    push_mode: str,
    direction,
) -> list[float]:
    """Reject a protocol the evaluator cannot run. Returns the speed list."""
    if frame not in {"world", "heading"}:
        raise ValueError(f"push frame must be world or heading, got {frame!r}")
    if push_mode not in {"add", "set"}:
        raise ValueError(f"push mode must be add or set, got {push_mode!r}")
    parse_direction(direction)
    parsed = [float(item) for item in speeds]
    if not parsed:
        raise ValueError("push speeds must list at least one speed")
    for speed in parsed:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError(f"push speeds must be finite and > 0, got {speed}")
    if isinstance(trials, bool) or int(trials) < 1:
        raise ValueError(f"trials per speed must be >= 1, got {trials}")
    for name, value in (
        ("settle_seconds", settle_seconds),
        ("max_seconds", max_seconds),
        ("speed_threshold", criteria.speed_threshold),
        ("tilt_threshold", criteria.tilt_threshold),
        ("torso_ori_threshold", criteria.torso_ori_threshold),
        ("stable_seconds", criteria.stable_seconds),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {value}")
    if control_steps(float(max_seconds)) < 1:
        raise ValueError(
            f"max_seconds must cover at least one control step, got {max_seconds}"
        )
    if criteria.stable_frames() * criteria.dt > float(max_seconds) + 1e-9:
        raise ValueError(
            "stable window "
            f"({criteria.stable_seconds} s) is longer than max_seconds "
            f"({max_seconds} s), so a trial cannot recover"
        )
    if criteria.max_xy_drift is not None and (
        not math.isfinite(float(criteria.max_xy_drift))
        or float(criteria.max_xy_drift) < 0
    ):
        raise ValueError(
            f"max_xy_drift must be >= 0 or null, got {criteria.max_xy_drift}"
        )
    return parsed


def sample_directions(count: int, seed: int) -> torch.Tensor:
    """Unit xy directions. ``seed`` fixes the whole list."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    yaw = torch.rand(int(count), generator=generator) * (2.0 * math.pi)
    return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)


def trial_plan(speeds, trials: int, direction, seed: int) -> list[dict]:
    """One record per trial, in speed-major order. Directions depend only on ``seed``."""
    fixed = parse_direction(direction)
    total = len(speeds) * int(trials)
    sampled = None if fixed is not None else sample_directions(total, seed)
    rows = []
    index = 0
    for speed in speeds:
        for trial in range(int(trials)):
            vector = fixed if fixed is not None else sampled[index]
            rows.append(
                {
                    "trial": index,
                    "speed_m_s": float(speed),
                    "direction_x": float(vector[0]),
                    "direction_y": float(vector[1]),
                    "seed": int(seed),
                }
            )
            index += 1
    return rows


def world_push_xy(
    direction_xy: torch.Tensor,
    speed: float,
    frame: str,
    quat: torch.Tensor,
) -> torch.Tensor:
    """Horizontal world-frame velocity of one push, shape ``[2]`` or ``[N, 2]``.

    ``direction_xy`` is a unit xy vector in ``frame``. ``heading`` rotates it
    by the root yaw into the world. ``world`` uses it as written. The speed
    is the magnitude in m/s.
    """
    direction = direction_xy.to(dtype=torch.float32)
    if direction.shape[-1] != 2:
        raise ValueError(f"direction must end in 2, got {tuple(direction.shape)}")
    squeeze = quat.dim() == 1 and direction.dim() == 1
    if direction.dim() == 1:
        direction = direction.unsqueeze(0)
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    if direction.shape[0] == 1 and quat.shape[0] != 1:
        direction = direction.expand(quat.shape[0], 2)
    norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    direction = direction / norm
    if frame == "heading":
        rotation = yaw_rotation(heading_yaw(quat))
        vector = direction.new_zeros(direction.shape[0], 3)
        vector[:, 0] = direction[:, 0]
        vector[:, 1] = direction[:, 1]
        world = torch.matmul(rotation, vector.unsqueeze(-1)).squeeze(-1)
        direction = world[:, :2]
    elif frame != "world":
        raise ValueError(f"push frame must be world or heading, got {frame!r}")
    pushed = direction * float(speed)
    return pushed.squeeze(0) if squeeze else pushed


def pushed_world_linear(
    current_lin: torch.Tensor, delta_xy: torch.Tensor, mode: str
) -> torch.Tensor:
    """World linear velocity after an add or a set. Z is unchanged."""
    updated = current_lin.clone()
    if mode == "add":
        updated[..., 0] = updated[..., 0] + delta_xy[..., 0]
        updated[..., 1] = updated[..., 1] + delta_xy[..., 1]
    elif mode == "set":
        updated[..., 0] = delta_xy[..., 0]
        updated[..., 1] = delta_xy[..., 1]
    else:
        raise ValueError(f"push mode must be add or set, got {mode!r}")
    return updated


def root_tilt(state: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Absolute difference of body-frame gravity z. Zero when the poses match."""
    gravity = projected_gravity(state[..., ROOT_QUAT])[..., 2]
    reference = projected_gravity(ref[..., ROOT_QUAT])[..., 2]
    return (gravity - reference).abs()


def is_stable(
    state: torch.Tensor,
    ref: torch.Tensor,
    origin_xy: torch.Tensor,
    criteria: RecoveryCriteria,
    torso_ori_err: torch.Tensor | None = None,
) -> torch.Tensor:
    """True when a post-push state meets this implementation's settle test.

    Horizontal root speed and root tilt are required. Torso orientation is
    required when ``torso_ori_err`` is passed. Horizontal drift from the
    pre-push position is required only when ``max_xy_drift`` is set. A small
    speed alone is not enough: the caller must also reject a fall.
    """
    speed = state[..., ROOT_LIN_VEL][..., :2].norm(dim=-1)
    stable = (speed <= criteria.speed_threshold) & (
        root_tilt(state, ref) <= criteria.tilt_threshold
    )
    if criteria.max_xy_drift is not None:
        drift = (state[..., ROOT_POS][..., :2] - origin_xy).norm(dim=-1)
        stable = stable & (drift <= float(criteria.max_xy_drift))
    if torso_ori_err is not None:
        stable = stable & (torso_ori_err <= criteria.torso_ori_threshold)
    return stable


def judge_post_push(
    stable: torch.Tensor,
    failed: torch.Tensor,
    criteria: RecoveryCriteria,
) -> dict:
    """Classify the steps after the push.

    ``stable[k]`` and ``failed[k]`` describe the state after control step
    ``k + 1``. A fall on that step wins over stability. Recovery time is the
    timestamp of the step that completes the stable window, not the first
    stable step and not zero when the trial fails.
    """
    required = criteria.stable_frames()
    run = 0
    onset = 0
    for index in range(int(stable.shape[0])):
        if bool(failed[index]):
            return {
                "outcome": "fallen",
                "reason": "fallen",
                "recovery_time_s": None,
                "stable_onset_time_s": None,
            }
        if bool(stable[index]):
            if run == 0:
                onset = index
            run += 1
            if run >= required:
                return {
                    "outcome": "recovered",
                    "reason": "recovered",
                    "recovery_time_s": (index + 1) * float(criteria.dt),
                    "stable_onset_time_s": (onset + 1) * float(criteria.dt),
                }
        else:
            run = 0
    return {
        "outcome": "timeout",
        "reason": "timeout",
        "recovery_time_s": None,
        "stable_onset_time_s": None,
    }


def initialization_failure(
    finite: bool, joints_inside: bool, failed: bool
) -> str | None:
    """Why the robot never received a push, or ``None`` when the push may run."""
    if not finite:
        return "nonfinite"
    if not joints_inside:
        return "joint_limit"
    if failed:
        return "fallen_before_push"
    return None


def _blank_trial(plan: dict, frame: str, push_mode: str) -> dict:
    return {
        "trial": int(plan["trial"]),
        "speed_m_s": float(plan["speed_m_s"]),
        "direction_x": float(plan["direction_x"]),
        "direction_y": float(plan["direction_y"]),
        "frame": frame,
        "push_mode": push_mode,
        "seed": int(plan["seed"]),
        "outcome": "initialization_failed",
        "reason": "",
        "recovered": 0,
        "fallen": 0,
        "timeout": 0,
        "initialization_failed": 1,
        "recovery_time_s": None,
        "stable_onset_time_s": None,
        "world_push_x": None,
        "world_push_y": None,
        "measured_dvx": None,
        "measured_dvy": None,
    }


def _mark(row: dict, outcome: str, reason: str) -> dict:
    row["outcome"] = outcome
    row["reason"] = reason
    row["recovered"] = int(outcome == "recovered")
    row["fallen"] = int(outcome == "fallen")
    row["timeout"] = int(outcome == "timeout")
    row["initialization_failed"] = int(outcome == "initialization_failed")
    return row


def torso_orientation_error(state: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between the robot torso and the reference torso."""
    from rlinf.envs.sim.g1_mimic.reward import TORSO, differentiable_fk

    _pos, rot, _lin, _ang = differentiable_fk(state)
    _ref_pos, ref_rot, _ref_lin, _ref_ang = differentiable_fk(ref)
    relative = torch.matmul(ref_rot[:, TORSO].transpose(-1, -2), rot[:, TORSO])
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _joints_inside(state: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> bool:
    joints = state[:, JOINT_POS]
    return bool(torch.all((joints >= low - 1e-3) & (joints <= high + 1e-3)))


def run_one_trial(
    env,
    policy: Callable,
    goal_fn: Callable,
    q0: torch.Tensor,
    plan: dict,
    criteria: RecoveryCriteria,
    frame: str,
    push_mode: str,
    settle_steps: int,
    max_steps: int,
    include_ref_ori: bool,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    record_trajectory: bool = False,
) -> tuple[dict, list[dict]]:
    """Settle, push once, then classify. One environment, already built."""
    if int(env.num_envs) != 1:
        raise ValueError(
            f"push recovery runs one environment at a time, got {env.num_envs}"
        )
    from rlinf.envs.sim.g1_mimic.obs import analytic_obs
    from rlinf.envs.sim.g1_mimic.reward import failure_signals
    from rlinf.workers.supermimic.eval_rollout import RollingHistory

    device = env.state.device
    env_ids = torch.zeros(1, dtype=torch.long, device=device)
    env.reset(env_ids, env_ids, env_ids)
    row = _blank_trial(plan, frame, push_mode)
    trajectory: list[dict] = []
    prev = torch.zeros(1, ACTION_DIM, device=device, dtype=torch.float32)
    q0 = q0.to(device=device, dtype=torch.float32)
    low = joint_low.to(device=device)
    high = joint_high.to(device=device)

    def observe(action):
        ref_quat = env.ref_state()[:, ROOT_QUAT] if include_ref_ori else None
        return analytic_obs(env.state, action, env.xi, q0, ref_quat=ref_quat)

    def failed_now() -> bool:
        _height, _tilt, failed = failure_signals(
            env.state, env.ref_state(), termination_body=env.termination_body
        )
        return bool(failed.reshape(-1)[0])

    history = RollingHistory(observe(prev))
    reason = initialization_failure(
        bool(torch.isfinite(env.state).all()),
        _joints_inside(env.state, low, high),
        failed_now(),
    )
    if reason is not None:
        return _mark(row, "initialization_failed", reason), trajectory

    for _step in range(int(settle_steps)):
        goal = goal_fn(env.motion_id, env.motion_time, env.state)
        action = torch.nan_to_num(policy(history.flat(), goal))
        result = env.step(action)
        if bool((result["end_kind"] == EPISODE_CLIP_END).any()):
            raise RuntimeError("static reference clip ended during the pre-push settle")
        prev = action
        fresh = observe(prev)
        history.push(torch.where(torch.isfinite(fresh), fresh, history.frames[:, -1]))
        if record_trajectory:
            trajectory.append(
                _trace(plan["trial"], "settle", env.state, env.ref_state())
            )
        step_reason = initialization_failure(
            bool(torch.isfinite(env.state).all()),
            _joints_inside(env.state, low, high),
            bool(result["motion_failure"].reshape(-1)[0])
            or int(result["end_kind"].reshape(-1)[0]) == EPISODE_FAILURE,
        )
        if step_reason is not None:
            return _mark(row, "initialization_failed", step_reason), trajectory

    before = env.state.clone()
    direction = torch.tensor(
        [plan["direction_x"], plan["direction_y"]], dtype=torch.float32, device=device
    )
    delta = world_push_xy(
        direction, float(plan["speed_m_s"]), frame, before[:, ROOT_QUAT]
    )
    env.apply_horizontal_velocity_push(env_ids, delta.reshape(1, 2), mode=push_mode)
    after = env.state
    measured = after[:, ROOT_LIN_VEL] - before[:, ROOT_LIN_VEL]
    row["world_push_x"] = float(delta.reshape(-1)[0])
    row["world_push_y"] = float(delta.reshape(-1)[1])
    row["measured_dvx"] = float(measured[0, 0])
    row["measured_dvy"] = float(measured[0, 1])
    if record_trajectory:
        trajectory.append(_trace(plan["trial"], "push", after, env.ref_state()))
    if not bool(torch.isfinite(after).all()):
        return _mark(row, "initialization_failed", "nonfinite"), trajectory

    origin = before[:, ROOT_POS][:, :2].clone()
    stable_flags = []
    failed_flags = []
    for _step in range(int(max_steps)):
        goal = goal_fn(env.motion_id, env.motion_time, env.state)
        action = torch.nan_to_num(policy(history.flat(), goal))
        result = env.step(action)
        if bool((result["end_kind"] == EPISODE_CLIP_END).any()):
            raise RuntimeError("static reference clip ended after the push")
        state = result["next_state"]
        prev = action
        fresh = observe(prev)
        history.push(torch.where(torch.isfinite(fresh), fresh, history.frames[:, -1]))
        nonfinite = not bool(torch.isfinite(state).all())
        fell = nonfinite or bool(result["motion_failure"].reshape(-1)[0])
        if record_trajectory:
            trajectory.append(
                _trace(plan["trial"], "after", state, result["ref_state"])
            )
        if nonfinite:
            failed_flags.append(True)
            stable_flags.append(False)
            break
        torso = torso_orientation_error(state[:1], result["ref_state"][:1])
        stable_flags.append(
            bool(
                is_stable(
                    state[:1],
                    result["ref_state"][:1],
                    origin,
                    criteria,
                    torso_ori_err=torso,
                )[0]
            )
        )
        failed_flags.append(fell)
        if fell:
            break
    if not stable_flags:
        return _mark(row, "timeout", "timeout"), trajectory
    verdict = judge_post_push(
        torch.tensor(stable_flags),
        torch.tensor(failed_flags),
        criteria,
    )
    row["recovery_time_s"] = verdict["recovery_time_s"]
    row["stable_onset_time_s"] = verdict["stable_onset_time_s"]
    if verdict["outcome"] == "fallen" and not bool(torch.isfinite(env.state).all()):
        verdict = {**verdict, "reason": "nonfinite"}
    return _mark(row, verdict["outcome"], verdict["reason"]), trajectory


def _trace(trial: int, phase: str, state: torch.Tensor, ref: torch.Tensor) -> dict:
    root = state[0].detach()
    return {
        "trial": int(trial),
        "phase": phase,
        "root_x": float(root[0]),
        "root_y": float(root[1]),
        "root_z": float(root[2]),
        "root_vx": float(root[ROOT_LIN_VEL][0]),
        "root_vy": float(root[ROOT_LIN_VEL][1]),
        "root_vz": float(root[ROOT_LIN_VEL][2]),
        "tilt": float(root_tilt(root.unsqueeze(0), ref[:1])[0]),
    }


def _rate(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return count / total


def _time_stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle]
    else:
        median = 0.5 * (ordered[middle - 1] + ordered[middle])
    return sum(ordered) / len(ordered), median


def summarize_trials(rows: list[dict], speeds: list[float]) -> dict:
    """Rates over trials that received a push. Empty recovery times stay null."""

    def pack(group: list[dict]) -> dict:
        post = [row for row in group if row["outcome"] != "initialization_failed"]
        recovered = [row for row in post if row["outcome"] == "recovered"]
        fallen = [row for row in post if row["outcome"] == "fallen"]
        timeout = [row for row in post if row["outcome"] == "timeout"]
        times = [
            float(row["recovery_time_s"])
            for row in recovered
            if row["recovery_time_s"] is not None
        ]
        mean, median = _time_stats(times)
        return {
            "trials": len(group),
            "pushed_trials": len(post),
            "recovered": len(recovered),
            "fallen": len(fallen),
            "timeout": len(timeout),
            "initialization_failed": len(group) - len(post),
            "recovery_rate": _rate(len(recovered), len(post)),
            "fall_rate": _rate(len(fallen), len(post)),
            "timeout_rate": _rate(len(timeout), len(post)),
            "initialization_failed_rate": _rate(len(group) - len(post), len(group)),
            "recovery_time_mean_s": mean,
            "recovery_time_median_s": median,
        }

    by_speed = []
    for speed in speeds:
        group = [row for row in rows if float(row["speed_m_s"]) == float(speed)]
        by_speed.append({"speed_m_s": float(speed), **pack(group)})
    return {"overall": pack(rows), "by_speed": by_speed}


def criteria_dict(criteria: RecoveryCriteria) -> dict:
    """JSON-ready copy of the implementation thresholds."""
    return {
        "speed_threshold_m_s": criteria.speed_threshold,
        "tilt_threshold": criteria.tilt_threshold,
        "torso_ori_threshold_rad": criteria.torso_ori_threshold,
        "stable_seconds": criteria.stable_seconds,
        "stable_frames": criteria.stable_frames(),
        "max_xy_drift_m": criteria.max_xy_drift,
        "recovery_time": (
            "time from the push until the stable window completes; "
            "stable_onset_time_s is the first frame of that window"
        ),
        "fall": "training failure_signals against the static reference",
        "reference_pose": REFERENCE_POSE,
    }


_TRIAL_FIELDS = (
    "trial",
    "speed_m_s",
    "direction_x",
    "direction_y",
    "frame",
    "push_mode",
    "seed",
    "outcome",
    "reason",
    "recovered",
    "fallen",
    "timeout",
    "initialization_failed",
    "recovery_time_s",
    "stable_onset_time_s",
    "world_push_x",
    "world_push_y",
    "measured_dvx",
    "measured_dvy",
)


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value)


def write_push_outputs(
    out_dir: Path,
    summary: dict,
    rows: list[dict],
    trajectory: list[dict] | None = None,
) -> None:
    """Write ``summary.json``, ``per_trial.csv``, and an optional trajectory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (out_dir / "per_trial.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TRIAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in _TRIAL_FIELDS})
    if trajectory is not None:
        fields = (
            "trial",
            "phase",
            "root_x",
            "root_y",
            "root_z",
            "root_vx",
            "root_vy",
            "root_vz",
            "tilt",
        )
        with (out_dir / "trajectory.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in trajectory:
                writer.writerow({key: _cell(row.get(key)) for key in fields})
