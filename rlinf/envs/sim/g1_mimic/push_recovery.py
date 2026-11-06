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

import math
from dataclasses import dataclass

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    DT,
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
    rot_apply,
    yaw_rotation,
)
from rlinf.envs.sim.g1_mimic.motion import GOAL_FRAMES, MotionLibrary

# Paper §4.3 names the range. It does not name this grid or a trial count.
DEFAULT_SPEEDS = (0.5, 1.0, 2.0, 4.0)
REFERENCE_POSE = "mjlab_knees_bent_keyframe"
PROTOCOL_NAME = "stationary_standing_pose"
OUTCOMES = ("recovered", "fallen", "timeout", "initialization_failed")
# Why a trial ended; ``initialization_failed`` carries one of the last three.
REASONS = (
    "recovered",
    "fallen",
    "timeout",
    "nonfinite",
    "joint_limit",
    "fallen_before_push",
)

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
        world = rot_apply(rotation, vector)
        direction = world[:, :2]
    elif frame != "world":
        raise ValueError(f"push frame must be world or heading, got {frame!r}")
    pushed = direction * float(speed)
    return pushed.squeeze(0) if squeeze else pushed


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


class RecoveryTracker:
    """Post-push verdicts of many trials, one control step at a time.

    Step ``k`` describes the state after post-push control step ``k + 1``.
    A fall on a step wins over stability on that step. A trial recovers on
    the step that completes ``stable_frames`` consecutive stable steps; its
    recovery time is that step's time from the push, and the first step of
    the window is its stable onset. A trial that does neither within the
    budget times out.
    """

    def __init__(self, num: int, criteria: RecoveryCriteria, device):
        self.required = criteria.stable_frames()
        self.dt = float(criteria.dt)
        self.run = torch.zeros(num, dtype=torch.long, device=device)
        self.onset = torch.zeros(num, dtype=torch.long, device=device)

    def reset(self, rows: torch.Tensor) -> None:
        self.run[rows] = 0
        self.onset[rows] = 0

    def step(
        self,
        index: int,
        rows: torch.Tensor,
        stable: torch.Tensor,
        failed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows that fell and rows that recovered on post-push step ``index``."""
        fallen = rows & failed
        judged = rows & ~failed
        stable_now = judged & stable
        self.onset = torch.where(
            stable_now & (self.run == 0), torch.full_like(self.onset, index), self.onset
        )
        self.run = torch.where(
            stable_now,
            self.run + 1,
            torch.where(judged, torch.zeros_like(self.run), self.run),
        )
        recovered = stable_now & (self.run >= self.required)
        return fallen, recovered

    def onset_time(self) -> torch.Tensor:
        """Stable-onset time of every row, in seconds from the push."""
        return (self.onset + 1).float() * self.dt


def initialization_reason(
    state: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    failed: torch.Tensor,
) -> torch.Tensor:
    """``REASONS`` index that stops each trial before its push, or -1.

    A non-finite state wins over a joint outside its limits, which wins over
    a fall.
    """
    finite = torch.isfinite(state).all(dim=-1)
    joints = state[:, JOINT_POS]
    inside = ((joints >= joint_low - 1e-3) & (joints <= joint_high + 1e-3)).all(dim=-1)
    code = torch.full_like(failed, -1, dtype=torch.long)
    code = torch.where(failed, REASONS.index("fallen_before_push"), code)
    code = torch.where(~inside, REASONS.index("joint_limit"), code)
    return torch.where(~finite, REASONS.index("nonfinite"), code)


def torso_orientation_error(state: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between the robot torso and the reference torso."""
    from rlinf.envs.sim.g1_mimic.reward import TORSO, differentiable_fk

    _pos, rot, _lin, _ang = differentiable_fk(state)
    _ref_pos, ref_rot, _ref_lin, _ref_ang = differentiable_fk(ref)
    trace = (ref_rot[:, TORSO] * rot[:, TORSO]).sum(dim=(-2, -1))
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine)


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


TRIAL_FIELDS = (
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
