# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Push-recovery protocol, including one real MuJoCo Warp velocity write."""

from __future__ import annotations

import csv
import importlib.util
import json
import math

import pytest
import torch

from evaluations.eval_supermimic import resolve_checkpoint
from evaluations.eval_supermimic_push import require_matching_checkpoint
from rlinf.envs.sim.g1_mimic.constants import (
    JOINT_POS,
    JOINT_VEL,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
    STATE_DIM,
)
from rlinf.envs.sim.g1_mimic.push_recovery import (
    DT,
    RecoveryCriteria,
    check_push_protocol,
    default_standing_state,
    hold_pose_library,
    initialization_failure,
    is_stable,
    judge_post_push,
    pushed_world_linear,
    root_tilt,
    summarize_trials,
    trial_plan,
    world_push_xy,
    write_push_outputs,
)
from rlinf.workers.supermimic.checkpoint import verify_checkpoint

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="mjlab is an optional SuperMimic extra",
)
needs_warp = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None or not torch.cuda.is_available(),
    reason="the velocity write is checked in MuJoCo Warp",
)


def _pose(z: float = 0.76, speed_xy=(0.0, 0.0), tilt_z: float = -1.0) -> torch.Tensor:
    """Upright identity root unless ``tilt_z`` asks for a pitched gravity."""
    state = torch.zeros(STATE_DIM)
    state[2] = z
    state[3] = 1.0
    state[ROOT_LIN_VEL.start] = speed_xy[0]
    state[ROOT_LIN_VEL.start + 1] = speed_xy[1]
    if tilt_z != -1.0:
        # Pitch so body-frame gravity z equals tilt_z. Identity gives -1.
        # g_body_z = -R_zz. For a pitch p, R_zz = cos(p), so cos(p) = -tilt_z.
        cosine = max(-1.0, min(1.0, -tilt_z))
        pitch = math.acos(cosine)
        half = 0.5 * pitch
        state[3] = math.cos(half)
        state[5] = math.sin(half)
    return state


def _criteria(**kwargs) -> RecoveryCriteria:
    values = {"stable_seconds": 0.04, "dt": DT}
    values.update(kwargs)
    return RecoveryCriteria(**values)


def test_heading_push_rotates_into_the_world_and_set_replaces_xy():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    forward = world_push_xy(torch.tensor([1.0, 0.0]), 2.0, "world", identity)
    assert torch.allclose(forward, torch.tensor([2.0, 0.0]), atol=1e-6)
    yaw = math.pi / 2
    half = 0.5 * yaw
    turned = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])
    # Robot x, after a +90 deg yaw, points along world y.
    pushed = world_push_xy(torch.tensor([1.0, 0.0]), 2.0, "heading", turned)
    assert torch.allclose(pushed, torch.tensor([0.0, 2.0]), atol=1e-5)
    # A world-frame request ignores that yaw.
    unchanged = world_push_xy(torch.tensor([1.0, 0.0]), 2.0, "world", turned)
    assert torch.allclose(unchanged, torch.tensor([2.0, 0.0]), atol=1e-5)

    current = torch.tensor([0.3, -0.1, 0.4])
    added = pushed_world_linear(current, torch.tensor([1.0, 0.0]), "add")
    assigned = pushed_world_linear(current, torch.tensor([1.0, 0.0]), "set")
    assert torch.allclose(added, torch.tensor([1.3, -0.1, 0.4]))
    assert torch.allclose(assigned, torch.tensor([1.0, 0.0, 0.4]))


def test_recovery_window_fall_and_timeout():
    ref = _pose()
    origin = torch.zeros(2)
    criteria = _criteria()
    required = criteria.stable_frames()
    assert required == 2
    still = _pose()
    moving = _pose(speed_xy=(1.0, 0.0))
    assert bool(is_stable(still.unsqueeze(0), ref.unsqueeze(0), origin, criteria)[0])
    assert not bool(
        is_stable(moving.unsqueeze(0), ref.unsqueeze(0), origin, criteria)[0]
    )
    tipped = _pose(tilt_z=-0.2)
    assert float(root_tilt(tipped.unsqueeze(0), ref.unsqueeze(0))[0]) > 0.35
    assert not bool(
        is_stable(tipped.unsqueeze(0), ref.unsqueeze(0), origin, criteria)[0]
    )

    # Two stable steps complete a 0.04 s window. Time is the completing step.
    recovered = judge_post_push(
        torch.tensor([False, True, True]),
        torch.tensor([False, False, False]),
        criteria,
    )
    assert recovered["outcome"] == "recovered"
    assert recovered["recovery_time_s"] == pytest.approx(3 * DT)
    assert recovered["stable_onset_time_s"] == pytest.approx(2 * DT)

    # A fall with zero speed is still a fall. The time is not zero.
    fallen = judge_post_push(
        torch.tensor([True, True]),
        torch.tensor([False, True]),
        criteria,
    )
    assert fallen["outcome"] == "fallen"
    assert fallen["recovery_time_s"] is None

    # One stable step does not finish a two-step window.
    waiting = judge_post_push(
        torch.tensor([False, True]),
        torch.tensor([False, False]),
        criteria,
    )
    assert waiting["outcome"] == "timeout"
    assert waiting["recovery_time_s"] is None

    # The window resets when a fast step breaks it, then completes later.
    restarted = judge_post_push(
        torch.tensor([True, False, True, True]),
        torch.zeros(4, dtype=torch.bool),
        criteria,
    )
    assert restarted["outcome"] == "recovered"
    assert restarted["stable_onset_time_s"] == pytest.approx(3 * DT)
    assert restarted["recovery_time_s"] == pytest.approx(4 * DT)


def test_pre_push_failure_is_not_a_post_push_fall():
    assert initialization_failure(True, True, False) is None
    assert initialization_failure(False, True, True) == "nonfinite"
    assert initialization_failure(True, False, False) == "joint_limit"
    assert initialization_failure(True, True, True) == "fallen_before_push"
    rows = [
        {
            "speed_m_s": 0.5,
            "outcome": "initialization_failed",
            "recovery_time_s": None,
        },
        {
            "speed_m_s": 0.5,
            "outcome": "fallen",
            "recovery_time_s": None,
        },
        {
            "speed_m_s": 4.0,
            "outcome": "recovered",
            "recovery_time_s": 1.2,
        },
    ]
    summary = summarize_trials(rows, [0.5, 4.0])
    slow = summary["by_speed"][0]
    assert slow["recovery_rate"] == 0.0
    assert slow["fall_rate"] == 1.0
    assert slow["pushed_trials"] == 1
    assert slow["initialization_failed"] == 1
    assert slow["recovery_time_mean_s"] is None
    fast = summary["by_speed"][1]
    assert fast["recovery_rate"] == 1.0
    assert fast["recovery_time_mean_s"] == pytest.approx(1.2)
    assert fast["recovery_time_median_s"] == pytest.approx(1.2)
    assert summary["overall"]["recovery_time_mean_s"] == pytest.approx(1.2)


def test_trial_plan_repeats_for_a_fixed_seed_and_rejects_bad_speeds():
    first = trial_plan([0.5, 4.0], 2, "random", 7)
    second = trial_plan([0.5, 4.0], 2, "random", 7)
    other = trial_plan([0.5, 4.0], 2, "random", 8)
    assert [(row["direction_x"], row["direction_y"]) for row in first] == [
        (row["direction_x"], row["direction_y"]) for row in second
    ]
    assert [(row["direction_x"], row["direction_y"]) for row in first] != [
        (row["direction_x"], row["direction_y"]) for row in other
    ]
    assert [row["speed_m_s"] for row in first] == [0.5, 0.5, 4.0, 4.0]
    fixed = trial_plan([1.0], 2, [0.0, 2.0], 1)
    assert fixed[0]["direction_x"] == pytest.approx(0.0)
    assert fixed[0]["direction_y"] == pytest.approx(1.0)
    criteria = _criteria(stable_seconds=0.5)
    with pytest.raises(ValueError, match="finite and > 0"):
        check_push_protocol(
            [0.5, float("nan")], 1, 0.0, 1.0, criteria, "world", "add", "random"
        )
    with pytest.raises(ValueError, match="longer than max_seconds"):
        check_push_protocol([0.5], 1, 0.0, 0.2, criteria, "heading", "set", "random")
    with pytest.raises(ValueError, match="non-zero"):
        check_push_protocol([0.5], 1, 0.0, 1.0, criteria, "world", "add", [0.0, 0.0])


def test_summary_files_leave_failed_recovery_times_blank(tmp_path):
    rows = [
        {
            "trial": 0,
            "speed_m_s": 0.5,
            "direction_x": 1.0,
            "direction_y": 0.0,
            "frame": "world",
            "push_mode": "add",
            "seed": 3,
            "outcome": "timeout",
            "reason": "timeout",
            "recovered": 0,
            "fallen": 0,
            "timeout": 1,
            "initialization_failed": 0,
            "recovery_time_s": None,
            "stable_onset_time_s": None,
            "world_push_x": 0.5,
            "world_push_y": 0.0,
            "measured_dvx": 0.5,
            "measured_dvy": 0.0,
        }
    ]
    summary = {
        "meta": {"seed": 3},
        **summarize_trials(rows, [0.5]),
    }
    write_push_outputs(
        tmp_path,
        summary,
        rows,
        trajectory=[
            {
                "trial": 0,
                "phase": "push",
                "root_x": 0.0,
                "root_y": 0.0,
                "root_z": 0.76,
                "root_vx": 0.5,
                "root_vy": 0.0,
                "root_vz": 0.0,
                "tilt": 0.0,
            }
        ],
    )
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded["by_speed"][0]["recovery_time_mean_s"] is None
    assert loaded["by_speed"][0]["timeout_rate"] == 1.0
    with (tmp_path / "per_trial.csv").open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert table[0]["outcome"] == "timeout"
    assert table[0]["recovery_time_s"] == ""
    assert table[0]["measured_dvx"] == "0.5"
    with (tmp_path / "trajectory.csv").open(encoding="utf-8") as handle:
        trace = list(csv.DictReader(handle))
    assert trace[0]["phase"] == "push"
    assert trace[0]["root_vx"] == "0.5"


def test_missing_and_mismatched_checkpoints_are_refused(tmp_path):
    empty = tmp_path / "global_step_1"
    empty.mkdir()
    with pytest.raises(ValueError, match="no checkpoint_meta.json"):
        verify_checkpoint(empty)
    with pytest.raises(FileNotFoundError, match="no complete"):
        resolve_checkpoint(str(tmp_path))
    with pytest.raises(ValueError, match="goal_dim"):
        require_matching_checkpoint(
            {
                "goal_dim": 730,
                "obs_dim": 93,
                "goal_encoding": "paper",
                "goal_window": [0, 9],
                "velocity_tracking": False,
            },
            {
                "goal_dim": 680,
                "obs_dim": 93,
                "goal_encoding": "paper",
                "goal_window": [0, 9],
                "velocity_tracking": False,
            },
        )


def test_static_clip_repeats_one_pose_with_zero_velocity():
    state = torch.zeros(STATE_DIM)
    state[2] = 0.76
    state[3] = 1.0
    state[7] = 0.2
    state[ROOT_LIN_VEL.start] = 3.0
    library = hold_pose_library(state, 8)
    clip = library.clip(0)
    assert clip.shape == (8, STATE_DIM)
    assert torch.allclose(clip[:, JOINT_POS], clip[:1, JOINT_POS])
    assert torch.count_nonzero(clip[:, ROOT_LIN_VEL]) == 0
    assert torch.count_nonzero(clip[:, JOINT_VEL]) == 0
    assert library.clip_names == ["static_standing"]
    assert torch.allclose(library.q0, clip[0, JOINT_POS])


@needs_mjlab
def test_standing_pose_is_inside_joint_limits_and_matches_fk():
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_joint_limits, g1_kinematics
    from rlinf.envs.sim.g1_mimic.reward import failure_signals

    state = default_standing_state()
    low, high = asset_joint_limits()
    joints = state[JOINT_POS]
    assert torch.all(joints >= low - 1e-4)
    assert torch.all(joints <= high + 1e-4)
    assert torch.allclose(state[ROOT_QUAT].norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.count_nonzero(state[ROOT_LIN_VEL]) == 0
    assert float(state[2]) == pytest.approx(0.76, abs=1e-3)
    body, _rot, _lin, _ang = g1_kinematics().forward(state.unsqueeze(0))
    assert torch.isfinite(body).all()
    free = g1_kinematics().free_tracked
    assert torch.allclose(body[0, free], state[ROOT_POS], atol=1e-4)
    ref = state.unsqueeze(0)
    _height, _tilt, failed = failure_signals(ref, ref)
    assert not bool(failed[0])
    dropped = state.clone()
    dropped[2] = float(state[2]) - 0.4
    _height, _tilt, failed = failure_signals(dropped.unsqueeze(0), ref)
    assert bool(failed[0])
    # A fallen pose can have zero speed. The fall still wins.
    verdict = judge_post_push(
        torch.tensor([True]),
        failed[:1].cpu(),
        _criteria(),
    )
    assert verdict["outcome"] == "fallen"
    assert verdict["recovery_time_s"] is None


@needs_mjlab
def test_standing_pose_is_not_the_paper_t_pose():
    """Knees-bent and the MJCF rest pose both fail an arms-out T-pose check.

    Section 4.3 does not publish joint angles. The asset has no T-pose
    keyframe, so the evaluator keeps the standing pose.
    """
    import mjlab.asset_zoo.robots.unitree_g1.g1_constants as g1_constants

    from rlinf.envs.sim.g1_mimic.constants import G1_JOINT_NAMES
    from rlinf.envs.sim.g1_mimic.g1_fk import TRACKED_BODY_NAMES, g1_kinematics
    from rlinf.envs.sim.g1_mimic.push_recovery import (
        PAPER_PROTOCOL,
        PROTOCOL_NAME,
        REFERENCE_POSE,
    )

    keyframes = [
        name
        for name in dir(g1_constants)
        if "KEYFRAME" in name or "TPOSE" in name or "T_POSE" in name
    ]
    assert keyframes == ["HOME_KEYFRAME", "KNEES_BENT_KEYFRAME"]
    assert PROTOCOL_NAME == "stationary_standing_pose"
    assert REFERENCE_POSE == "mjlab_knees_bent_keyframe"
    assert PAPER_PROTOCOL["pose"] == "static T-pose"
    assert PAPER_PROTOCOL["implemented_protocol"] == PROTOCOL_NAME
    assert PAPER_PROTOCOL["implemented_pose"] == REFERENCE_POSE

    names = list(TRACKED_BODY_NAMES)
    kin = g1_kinematics()

    def wrist_minus_shoulder(state):
        pos, _rot, _lin, _ang = kin.forward(state.unsqueeze(0))
        pos = pos[0]
        deltas = {}
        for side in ("left", "right"):
            shoulder = pos[names.index(f"{side}_shoulder_roll_link")]
            wrist = pos[names.index(f"{side}_wrist_yaw_link")]
            deltas[side] = wrist - shoulder
        return deltas

    stand = default_standing_state()
    for side, delta in wrist_minus_shoulder(stand).items():
        assert float(delta[2]) < -0.25, side
        assert abs(float(delta[1])) < 0.15, side
    for joint in ("left_elbow_joint", "right_elbow_joint"):
        value = float(stand[JOINT_POS][G1_JOINT_NAMES.index(joint)])
        assert value > 0.4
    for joint, sign in (
        ("left_shoulder_roll_joint", 1.0),
        ("right_shoulder_roll_joint", -1.0),
    ):
        value = float(stand[JOINT_POS][G1_JOINT_NAMES.index(joint)])
        assert abs(value) < 0.5
        assert value * sign > 0.0

    zero = torch.zeros_like(stand)
    zero[2] = 0.78
    zero[3] = 1.0
    for side, delta in wrist_minus_shoulder(zero).items():
        assert float(delta[0]) > 0.15, side
        assert float(delta[2]) < -0.15, side
        assert abs(float(delta[1])) < 0.05, side


@needs_warp
def test_push_changes_warp_root_velocity_before_the_next_integration():
    from rlinf.envs.sim.g1_mimic.env import make_env

    state = default_standing_state()
    motion = hold_pose_library(state, 30)
    device = torch.device("cuda:0")
    env = make_env(
        "mjlab",
        num_envs=1,
        motion=motion,
        device=device,
        max_episode_seconds=1.0e6,
        eval_mode=True,
    )
    env_ids = torch.zeros(1, dtype=torch.long, device=device)
    env.reset(env_ids, env_ids, env_ids)
    before = env.state.clone()
    env.apply_horizontal_velocity_push(
        env_ids, torch.tensor([[1.25, -0.5]], device=device), mode="add"
    )
    after = env.state
    delta = after[0, ROOT_LIN_VEL] - before[0, ROOT_LIN_VEL]
    assert torch.allclose(
        delta[:2], torch.tensor([1.25, -0.5], device=device), atol=1e-4
    )
    assert torch.allclose(delta[2], torch.zeros((), device=device), atol=1e-4)
    assert torch.allclose(after[0, ROOT_POS], before[0, ROOT_POS], atol=1e-5)
    assert torch.allclose(after[0, ROOT_QUAT], before[0, ROOT_QUAT], atol=1e-5)
    assert torch.allclose(after[0, ROOT_ANG_VEL], before[0, ROOT_ANG_VEL], atol=1e-4)
    assert torch.allclose(after[0, JOINT_POS], before[0, JOINT_POS], atol=1e-5)
    assert torch.allclose(after[0, JOINT_VEL], before[0, JOINT_VEL], atol=1e-4)
    stepped = env.step(torch.zeros(1, 29, device=device))
    assert torch.isfinite(stepped["next_state"]).all()
    moved = stepped["next_state"][0, 0] - before[0, 0]
    assert float(moved) > 1e-4
