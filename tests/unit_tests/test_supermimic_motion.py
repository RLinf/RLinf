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

"""SuperMimic reference motion: clip loading, resampling, goals and BONES-SEED."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.algorithms.supermimic.reset_sampler import BinResetSampler
from rlinf.envs.sim.g1_mimic.bones import (
    feasibility_reject,
    load_bones_library,
    name_rejected,
    upright_from_state,
)
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    G1_JOINT_NAMES,
    JOINT_POS,
    JOINT_VEL,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_QUAT,
    STATE_DIM,
)
from rlinf.envs.sim.g1_mimic.features import default_state, quat_mul, rotvec_to_quat
from rlinf.envs.sim.g1_mimic.motion import (
    MotionLibrary,
    control_time_grid,
    decode_motion_payload,
    load_motion_library,
    resample_states,
)

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="mjlab is an optional SuperMimic extra",
)


def _write_clip(path: Path, frames: int = 30, name: str | None = None) -> None:
    payload = {
        "fps": 50.0,
        "joint_names": list(G1_JOINT_NAMES),
        "q0": torch.zeros(29),
        "state": default_state(frames, "cpu", torch.float32),
    }
    if name is not None:
        payload["name"] = name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _reference_payload(state: torch.Tensor, fps: float = 50.0, names=None, q0=None):
    return {
        "fps": fps,
        "joint_names": list(G1_JOINT_NAMES if names is None else names),
        "q0": torch.zeros(ACTION_DIM) if q0 is None else q0,
        "state": state,
    }


def _constant_motion(
    frames: int, fps: float, speed: float, joint_rate: float, yaw_rate: float
):
    state = default_state(frames, "cpu", torch.float32)
    for i in range(frames):
        t = i / fps
        state[i, 0] = speed * t
        state[i, 7] = joint_rate * t
        state[i, ROOT_QUAT] = rotvec_to_quat(torch.tensor([0.0, 0.0, yaw_rate * t]))
    return state


class PaddedLibrary(MotionLibrary):
    """The phase 2 storage: one ``[N, max_T, 71]`` tensor, zero padded."""

    def __init__(self, clips, q0=None):
        super().__init__(clips, q0=q0)
        longest = int(self.lengths.max())
        self._packed = torch.zeros(len(clips), longest, STATE_DIM)
        for i, clip in enumerate(clips):
            self._packed[i, : clip.shape[0]] = clip.float()

    def clip_length(self, motion_id):
        return self.lengths.to(motion_id.device)[motion_id]

    def state_at(self, motion_id, time):
        squeezed = motion_id.dim() == 0
        if squeezed:
            motion_id = motion_id.unsqueeze(0)
            time = time.unsqueeze(0)
        mid = motion_id.long()
        index = time.long().clamp_min(0)
        index = torch.minimum(index, self.lengths.to(mid.device)[mid] - 1)
        out = self._packed.to(mid.device)[mid, index]
        return out.squeeze(0) if squeezed else out


def _random_clips(lengths, seed=0):
    gen = torch.Generator().manual_seed(seed)
    clips = []
    for length in lengths:
        state = default_state(length, "cpu", torch.float32)
        state[:, :3] += torch.randn(length, 3, generator=gen).cumsum(0) * 0.01
        quat = torch.randn(length, 4, generator=gen) * 0.1
        quat[:, 0] += 1.0
        state[:, 3:7] = quat / quat.norm(dim=-1, keepdim=True)
        state[:, 7:] += torch.randn(length, STATE_DIM - 7, generator=gen) * 0.2
        clips.append(state)
    return clips


def _standing(frames: int, dz: float = 0.0) -> torch.Tensor:
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state

    state = asset_default_state().float().unsqueeze(0).repeat(frames, 1)
    state[:, 2] += dz
    return state


def _save(directory, index, name, dz, frames=60):
    from rlinf.envs.sim.g1_mimic.constants import G1_JOINT_NAMES

    state = _standing(frames, dz)
    payload = {
        "fps": 50.0,
        "joint_names": list(G1_JOINT_NAMES),
        "q0": state[0, 7:36].clone(),
        "state": state,
    }
    if name is not None:
        payload["name"] = name
    torch.save(payload, directory / f"clip_{index:03d}.pt")


def _clip(directory, index, frames=20):
    state = torch.zeros(frames, STATE_DIM)
    state[:, 2] = 0.75
    state[:, 3] = 1.0
    state[:, 0] = torch.linspace(0, 1, frames)
    path = directory / f"clip_{index:03d}.pt"
    torch.save(
        {
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": torch.zeros(ACTION_DIM),
            "state": state,
        },
        path,
    )
    return path


def _manifest(directory, files):
    clips = [
        {
            "file": f.name,
            "source": f"src_{i}.csv",
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        }
        for i, f in enumerate(files)
    ]
    (directory / "manifest.json").write_text(json.dumps({"clips": clips}))


def _csv(path, frames=31):
    t = np.linspace(0, 1, frames)
    rows = np.zeros((frames, 7 + ACTION_DIM))
    rows[:, 0] = t
    rows[:, 2] = 0.78
    # xyzw: 90 degrees about z.
    rows[:, 5] = np.sqrt(0.5)
    rows[:, 6] = np.sqrt(0.5)
    rows[:, 7:] = 0.1 * np.sin(t)[:, None]
    np.savetxt(path, rows, delimiter=",")


def _run_prepare(monkeypatch, argv):
    from toolkits.supermimic import prepare_lafan1

    monkeypatch.setattr(sys, "argv", ["prepare_lafan1.py", *argv])
    return prepare_lafan1.main()


def test_resampled_velocities_are_central_differences_in_the_root_frame():
    omega, tilt, dt = 3.0, 0.4, 0.02
    t = torch.arange(60, dtype=torch.float64) * dt
    state = default_state(60, "cpu", torch.float64)
    state[:, 0] = torch.sin(omega * t)
    state[:, JOINT_POS] = 0.3 * torch.sin(omega * t).unsqueeze(-1)
    yaw = torch.sin(omega * t)
    zero = torch.zeros_like(t)
    q_yaw = torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), dim=-1)
    q_tilt = torch.tensor(
        [math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0], dtype=torch.float64
    )
    state[:, ROOT_QUAT] = quat_mul(q_yaw, q_tilt.expand(60, 4))
    out = resample_states(state, 50.0, 50.0)
    rate = omega * torch.cos(omega * t)
    inner = slice(1, -1)
    # Central differences are second order; a backward difference lags half
    # a step and is off by about omega^2 dt / 2 = 0.09 here.
    assert float((out[inner, ROOT_LIN_VEL][:, 0] - rate[inner]).abs().max()) < 5e-3
    assert float((out[inner, JOINT_VEL] - 0.3 * rate[inner, None]).abs().max()) < 2e-3
    body_axis = torch.tensor([0.0, math.sin(tilt), math.cos(tilt)], dtype=torch.float64)
    expected = rate[:, None] * body_axis
    assert float((out[inner, ROOT_ANG_VEL] - expected[inner]).abs().max()) < 5e-3
    # The two end frames use one-sided differences.
    assert float(out[0, 36]) == pytest.approx(float((state[1, 0] - state[0, 0]) / dt))
    assert float(out[-1, 36]) == pytest.approx(
        float((state[-1, 0] - state[-2, 0]) / dt)
    )


def test_null_clip_count_loads_every_bones_clip_past_the_filters(tmp_path):
    for i in range(45):
        _write_clip(tmp_path / f"clip_{i:03d}.pt", name=f"walk_{i}")
    _write_clip(tmp_path / "clip_045.pt", name="sit_on_chair")
    library = load_bones_library(str(tmp_path), None, "cpu")
    assert library.num_clips == 45
    assert library.filter_counts == {
        "files_read": 46,
        "files_in_dir": 46,
        "rejected_name": 1,
        "rejected_feasibility": 0,
        "kept": 45,
    }
    capped = load_bones_library(str(tmp_path), 3, "cpu")
    assert capped.num_clips == 3 and capped.clip_names == ["walk_0", "walk_1", "walk_2"]
    with pytest.raises(
        FileNotFoundError, match="kept 45 clips after filtering, need 50"
    ):
        load_bones_library(str(tmp_path), 50, "cpu")


def test_null_clip_count_loads_every_lafan1_file(tmp_path):
    for i in range(5):
        _write_clip(tmp_path / f"clip_{i:03d}.pt", frames=20 + i)
    assert load_motion_library(str(tmp_path), None, "cpu").num_clips == 5
    assert load_motion_library(str(tmp_path), 2, "cpu").lengths.tolist() == [20, 21]
    with pytest.raises(FileNotFoundError, match="has 5 clip_"):
        load_motion_library(str(tmp_path), 6, "cpu")


def test_lafan1_refuses_missing_clips(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_motion_library(str(tmp_path), 40, "cpu")
    with pytest.raises(FileNotFoundError, match="No verified G1 reference clips"):
        load_motion_library("", 3, "cpu")


def test_motion_payload_checks_fps_joints_and_default_pose(tmp_path):
    state = default_state(31, "cpu", torch.float32)
    state[:, 0] = torch.linspace(0.0, 1.0, 31)
    with pytest.raises(ValueError, match="bare state"):
        decode_motion_payload(state, "bare.pt")
    swapped = list(G1_JOINT_NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="joint order"):
        decode_motion_payload(_reference_payload(state, names=swapped), "order.pt")
    with pytest.raises(ValueError, match="fps"):
        decode_motion_payload(_reference_payload(state, fps=60.0), "fps.pt")

    decoded, q0 = decode_motion_payload(_reference_payload(state, fps=30.0), "30.pt")
    assert decoded.shape == (51, STATE_DIM)
    assert q0.shape == (ACTION_DIM,)
    assert float(decoded[-1, 0]) == pytest.approx(1.0, abs=1e-4)
    assert float(decoded[25, ROOT_LIN_VEL.start]) == pytest.approx(1.0, abs=1e-3)

    kept, _ = decode_motion_payload(_reference_payload(state[:8], fps=50.0), "50.pt")
    assert kept.shape[0] == 8

    q0_a = torch.arange(ACTION_DIM, dtype=torch.float32) * 0.01
    q0_b = q0_a + 0.2
    torch.save(_reference_payload(state[:8], q0=q0_a), tmp_path / "clip_000.pt")
    torch.save(_reference_payload(state[:8], q0=q0_b), tmp_path / "clip_001.pt")
    with pytest.raises(ValueError, match="default pose"):
        load_motion_library(str(tmp_path), 2, "cpu")
    (tmp_path / "clip_001.pt").unlink()
    torch.save(_reference_payload(state[:8], q0=q0_a), tmp_path / "clip_001.pt")
    library = load_motion_library(str(tmp_path), 2, "cpu")
    assert torch.allclose(library.q0, q0_a)


def test_resampled_yaw_rate_matches_the_pose_change():
    count = 31
    rate = 0.7
    state = default_state(count, "cpu", torch.float32)
    for i in range(count):
        state[i, ROOT_QUAT] = rotvec_to_quat(torch.tensor([0.0, 0.0, rate * i / 30.0]))
    out = resample_states(state, 30.0, 50.0)
    assert torch.allclose(
        out[2:, ROOT_ANG_VEL],
        torch.tensor([0.0, 0.0, rate]).expand(out.shape[0] - 2, 3),
        atol=0.02,
    )


def test_batched_goal_matches_one_time_at_a_time():
    clip = default_state(12, "cpu", torch.float32)
    for i in range(12):
        clip[i, 0] = float(i)
    library = MotionLibrary([clip])
    motion_id = torch.zeros(3, dtype=torch.long)
    time0 = torch.tensor([0, 2, 4])
    steps = torch.arange(5)
    times = time0.unsqueeze(1) + steps
    ids = motion_id.unsqueeze(1).expand_as(times)
    flat_id = ids.reshape(-1)
    flat_time = times.reshape(-1)
    goal = library.goal730(flat_id, flat_time).reshape(3, 5, 730)
    loop = torch.stack(
        [library.goal730(motion_id, time0 + t) for t in range(5)], dim=1
    )
    assert torch.allclose(goal, loop)
    ref = library.goal_ref(flat_id, flat_time).reshape(3, 5, -1)
    ref_loop = torch.stack(
        [library.goal_ref(motion_id, time0 + t) for t in range(5)], dim=1
    )
    assert torch.allclose(ref, ref_loop)


def test_goal730_uses_future_frames_and_holds_the_last():
    clip = default_state(4, "cpu", torch.float32)
    for i in range(4):
        clip[i, 0] = float(i)
        clip[i, JOINT_VEL] = 10.0
    goal = MotionLibrary([clip]).goal730(torch.tensor([0]), torch.tensor([0]))
    assert goal.shape == (1, 730)
    frames = goal.view(1, 10, 73)
    assert torch.allclose(
        frames[0, :, 0], torch.tensor([1, 2, 3, 3, 3, 3, 3, 3, 3, 3]).float()
    )
    assert torch.allclose(frames[0, 0, 44:73], torch.ones(29))


def test_bones_name_and_feasibility_filters(tmp_path):
    assert not name_rejected("walk_forward")
    assert not name_rejected("visit_friend")
    assert name_rejected("sit_down")
    assert name_rejected("climb_box")
    assert name_rejected("jump off 50cm")
    zmin = torch.zeros(50)
    feet = torch.zeros(50, 2)
    pelvis = torch.full((50,), 0.8)
    upright = torch.ones(50, dtype=torch.bool)
    speed = torch.zeros(50)
    assert feasibility_reject(zmin, feet, pelvis, upright, speed) is None
    high = zmin.clone()
    high[:40] = 0.3
    assert feasibility_reject(high, feet, pelvis, upright, speed) == "elevated support"
    deep = zmin.clone()
    deep[0] = -0.2
    assert (
        feasibility_reject(deep, feet, pelvis, upright, speed) == "ground penetration"
    )
    floating = feet.clone()
    floating[:30] = 0.05
    assert feasibility_reject(zmin, floating, pelvis, upright, speed) == "floating feet"
    with pytest.raises(FileNotFoundError):
        load_bones_library(str(tmp_path), 1, "cpu")
    state = default_state(8, "cpu", torch.float32)
    torch.save(
        {
            "name": "sit_on_chair",
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": torch.zeros(29),
            "state": state,
        },
        tmp_path / "clip_000.pt",
    )
    with pytest.raises(RuntimeError, match="removed every clip"):
        load_bones_library(str(tmp_path), 1, "cpu")
    torch.save(
        {
            "name": "walk_subject1",
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": torch.zeros(29),
            "state": state,
            "zmin": torch.zeros(8),
            "foot_clearance": torch.zeros(8, 2),
            "upright": torch.ones(8, dtype=torch.bool),
        },
        tmp_path / "clip_000.pt",
    )
    library = load_bones_library(str(tmp_path), 1, torch.device("cuda"))
    assert library.num_clips == 1
    assert library.clip_names == ["walk_subject1"]
    assert library.frames.device.type == "cpu"
    queried = library.state_at(torch.tensor([0]), torch.tensor([3]))
    assert torch.allclose(queried, state[3])
    bad = tmp_path / "human"
    bad.mkdir()
    human = torch.zeros(8, 20)
    torch.save(
        {
            "name": "human_walk",
            "fps": 50.0,
            "joint_names": ["a"],
            "q0": torch.zeros(20),
            "state": human,
        },
        bad / "clip_000.pt",
    )
    with pytest.raises(ValueError):
        load_bones_library(str(bad), 1, "cpu")


def test_resample_uses_the_control_grid_for_positions_and_velocity():
    # 30 Hz, two frames, 1 m/s. The old linspace kept the endpoint and
    # reported 0.833 m/s. The 50 Hz grid keeps 1 m/s and drops the short tail.
    state = _constant_motion(2, 30.0, 1.0, 0.5, 0.4)
    out = resample_states(state, 30.0, 50.0)
    _src, dst, dt = control_time_grid(2, 30.0, 50.0)
    assert dt == pytest.approx(0.02)
    assert out.shape[0] == 2
    assert float(dst[-1]) == pytest.approx(0.02)
    assert float(out[-1, 0]) == pytest.approx(0.02, abs=1e-5)
    assert float(out[:, ROOT_LIN_VEL][:, 0].mean()) == pytest.approx(1.0, abs=1e-4)
    assert float(out[:, ROOT_LIN_VEL][:, 0].mean()) != pytest.approx(0.8333, abs=1e-3)
    assert torch.allclose(
        out[:, ROOT_QUAT].norm(dim=-1), torch.ones(out.shape[0]), atol=1e-5
    )

    for frames in (2, 4, 31, 45):
        speed, joint_rate, yaw_rate = 1.3, -0.25, 0.4
        src = _constant_motion(frames, 30.0, speed, joint_rate, yaw_rate)
        sampled = resample_states(src, 30.0, 50.0)
        steps = int(math.floor((frames - 1) * 50.0 / 30.0 + 1e-8))
        assert sampled.shape[0] == steps + 1
        assert float(sampled[1, 0] - sampled[0, 0]) == pytest.approx(
            speed * 0.02, abs=1e-4
        )
        assert torch.allclose(
            sampled[:, ROOT_LIN_VEL][:, 0],
            torch.full((sampled.shape[0],), speed),
            atol=1e-4,
        )
        assert torch.allclose(
            sampled[:, JOINT_VEL][:, 0],
            torch.full((sampled.shape[0],), joint_rate),
            atol=1e-4,
        )
        assert torch.allclose(
            sampled[:, ROOT_ANG_VEL],
            torch.tensor([0.0, 0.0, yaw_rate]).expand(sampled.shape[0], 3),
            atol=1e-3,
        )
        dots = (sampled[:-1, ROOT_QUAT] * sampled[1:, ROOT_QUAT]).sum(-1)
        assert torch.all(dots > 0)

    native = _constant_motion(7, 50.0, 1.0, 0.2, 0.0)
    same = resample_states(native, 50.0, 50.0)
    assert same.shape[0] == 7
    assert torch.allclose(same[:, :3], native[:, :3], atol=1e-5)
    assert torch.allclose(same[:, 7:36], native[:, 7:36], atol=1e-5)


def test_bones_metadata_follows_the_state_clock_and_upright_is_derived(tmp_path):
    identity = default_state(4, "cpu", torch.float32)
    assert bool(upright_from_state(identity).all())
    tilted = identity.clone()
    tilted[:, ROOT_QUAT] = rotvec_to_quat(torch.tensor([math.pi / 2, 0.0, 0.0]))
    assert not bool(upright_from_state(tilted).any())

    zmin = torch.zeros(50)
    feet = torch.zeros(50, 2)
    pelvis = torch.full((50,), 0.8)
    upright = torch.ones(50, dtype=torch.bool)
    speed = torch.zeros(50)
    assert feasibility_reject(zmin, feet, pelvis, upright, speed) is None
    exact = zmin.clone()
    exact[:30] = 0.20
    assert feasibility_reject(exact, feet, pelvis, upright, speed) is None
    over = exact.clone()
    over[:30] = 0.2001
    assert feasibility_reject(over, feet, pelvis, upright, speed) == "elevated support"
    touch = zmin.clone()
    touch[0] = -0.12
    assert feasibility_reject(touch, feet, pelvis, upright, speed) is None
    deep = touch.clone()
    deep[0] = -0.12001
    assert (
        feasibility_reject(deep, feet, pelvis, upright, speed) == "ground penetration"
    )
    almost = zmin.clone()
    almost[:40] = -0.08
    assert feasibility_reject(almost, feet, pelvis, upright, speed) is None
    penetrated = almost.clone()
    penetrated[:41] = -0.0801
    assert (
        feasibility_reject(penetrated, feet, pelvis, upright, speed)
        == "ground penetration"
    )
    feet_edge = feet.clone()
    feet_edge[:25] = 0.02
    assert feasibility_reject(zmin, feet_edge, pelvis, upright, speed) is None
    floating = feet.clone()
    floating[:25] = 0.0201
    assert feasibility_reject(zmin, floating, pelvis, upright, speed) == "floating feet"
    low_pelvis = pelvis.clone()
    low_pelvis[:] = 0.6
    assert feasibility_reject(zmin, floating, low_pelvis, upright, speed) is None
    high_pelvis = torch.full((50,), 1.1)
    assert feasibility_reject(zmin, floating, high_pelvis, upright, speed) is None

    def save_clip(directory: Path, name: str, fps: float, state: torch.Tensor, **extra):
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "fps": fps,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": torch.zeros(29),
            "state": state,
        }
        payload.update(extra)
        torch.save(payload, directory / "clip_000.pt")

    frames = 31
    standing = default_state(frames, "cpu", torch.float32)
    standing[:, 2] = 0.8
    save_clip(
        tmp_path / "hz30",
        "walk_subject",
        30.0,
        standing,
        zmin=torch.zeros(frames),
        foot_clearance=torch.zeros(frames, 2),
    )
    library = load_bones_library(str(tmp_path / "hz30"), 1, "cpu")
    assert library.lengths[0] == 51

    floating_dir = tmp_path / "float"
    save_clip(
        floating_dir,
        "walk_subject",
        30.0,
        standing,
        zmin=torch.zeros(frames),
        foot_clearance=torch.full((frames, 2), 0.05),
    )
    with pytest.raises(RuntimeError, match="feasibility"):
        load_bones_library(str(floating_dir), 1, "cpu")
    given = upright_from_state(standing)
    save_clip(
        tmp_path / "float_marked",
        "walk_subject",
        30.0,
        standing,
        zmin=torch.zeros(frames),
        foot_clearance=torch.full((frames, 2), 0.05),
        upright=given,
    )
    with pytest.raises(RuntimeError, match="feasibility"):
        load_bones_library(str(tmp_path / "float_marked"), 1, "cpu")

    rolled = standing.clone()
    rolled[:, ROOT_QUAT] = rotvec_to_quat(torch.tensor([math.pi / 2, 0.0, 0.0]))
    save_clip(
        tmp_path / "rolled",
        "walk_subject",
        30.0,
        rolled,
        zmin=torch.zeros(frames),
        foot_clearance=torch.full((frames, 2), 0.05),
    )
    kept = load_bones_library(str(tmp_path / "rolled"), 1, "cpu")
    assert kept.num_clips == 1
    save_clip(
        tmp_path / "rolled_marked",
        "walk_subject",
        30.0,
        rolled,
        zmin=torch.zeros(frames),
        foot_clearance=torch.full((frames, 2), 0.05),
        upright=upright_from_state(rolled),
    )
    assert load_bones_library(str(tmp_path / "rolled_marked"), 1, "cpu").num_clips == 1
    save_clip(
        tmp_path / "lying_called_upright",
        "walk_subject",
        30.0,
        rolled,
        zmin=torch.zeros(frames),
        foot_clearance=torch.zeros(frames, 2),
        upright=torch.ones(frames, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="upright"):
        load_bones_library(str(tmp_path / "lying_called_upright"), 1, "cpu")

    short = default_state(8, "cpu", torch.float32)
    save_clip(
        tmp_path / "bad_len",
        "walk_subject",
        50.0,
        short,
        zmin=torch.zeros(31),
        foot_clearance=torch.zeros(31, 2),
    )
    with pytest.raises(ValueError, match="50 Hz|source time|does not match"):
        load_bones_library(str(tmp_path / "bad_len"), 1, "cpu")
    save_clip(
        tmp_path / "one_array",
        "walk_subject",
        50.0,
        short,
        zmin=torch.zeros(8),
    )
    with pytest.raises(ValueError, match="both"):
        load_bones_library(str(tmp_path / "one_array"), 1, "cpu")
    save_clip(
        tmp_path / "bad_feet",
        "walk_subject",
        50.0,
        short,
        zmin=torch.zeros(8),
        foot_clearance=torch.zeros(8, 3),
    )
    with pytest.raises(ValueError, match="foot_clearance"):
        load_bones_library(str(tmp_path / "bad_feet"), 1, "cpu")


def test_contiguous_library_matches_the_padded_one():
    lengths = [1, 7, 50, 233, 12]
    clips = _random_clips(lengths)
    new = MotionLibrary(clips)
    old = PaddedLibrary(clips)
    assert new.num_frames == sum(lengths)
    assert new.frames.shape == (sum(lengths), STATE_DIM)
    assert old._packed.numel() == len(lengths) * max(lengths) * STATE_DIM
    assert torch.equal(new.offsets, torch.tensor([0, 1, 8, 58, 291]))
    for i, clip in enumerate(clips):
        assert torch.equal(new.clip(i), clip)

    gen = torch.Generator().manual_seed(1)
    mid = torch.randint(0, len(lengths), (400,), generator=gen)
    time = torch.randint(-5, 260, (400,), generator=gen)
    assert torch.equal(new.state_at(mid, time), old.state_at(mid, time))
    assert torch.equal(new.clip_length(mid), old.clip_length(mid))
    assert torch.equal(new.past_clip_end(mid, time), old.past_clip_end(mid, time))
    assert torch.equal(new.goal_ref(mid, time), old.goal_ref(mid, time))
    assert torch.equal(new.goal730(mid, time), old.goal730(mid, time))
    assert torch.equal(
        new.reference_velocity_command(mid, time),
        old.reference_velocity_command(mid, time),
    )
    # Scalar queries keep their shape.
    one = new.state_at(torch.tensor(3), torch.tensor(500))
    assert one.shape == (STATE_DIM,)
    assert torch.equal(one, clips[3][-1])


def test_reset_bins_do_not_depend_on_the_storage():
    clips = _random_clips([60, 140, 9])
    new = BinResetSampler(MotionLibrary(clips).lengths, 4, 10.0, 0.99, 500, "cpu")
    old = BinResetSampler(PaddedLibrary(clips).lengths, 4, 10.0, 0.99, 500, "cpu")
    assert torch.equal(new.bin_clip, old.bin_clip)
    assert torch.equal(new.bin_start, old.bin_start)
    assert torch.equal(new.bin_end, old.bin_end)


def test_loaded_30hz_clips_keep_their_resampled_lengths(tmp_path):
    clips = _random_clips([31, 91])
    q0 = torch.zeros(ACTION_DIM)
    for i, clip in enumerate(clips):
        torch.save(
            {
                "fps": 30.0,
                "joint_names": list(G1_JOINT_NAMES),
                "q0": q0,
                "state": clip,
            },
            tmp_path / f"clip_{i:03d}.pt",
        )
    library = load_motion_library(str(tmp_path), 2, "cpu")
    # 30 frames span 1.0 s -> 51 control frames; 90 frames span 3.0 s -> 151.
    assert library.lengths.tolist() == [51, 151]
    assert library.num_frames == 202
    old = PaddedLibrary([library.clip(0), library.clip(1)])
    mid = torch.tensor([0, 1, 1, 0])
    time = torch.tensor([0, 150, 151, 60])
    assert torch.equal(library.state_at(mid, time), old.state_at(mid, time))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_library_answers_cuda_queries():
    clips = _random_clips([20, 40])
    library = MotionLibrary(clips)
    mid = torch.tensor([0, 1, 1], device="cuda")
    time = torch.tensor([3, 39, 80], device="cuda")
    out = library.state_at(mid, time)
    assert out.device.type == "cuda"
    assert library.frames.device.type == "cpu"
    assert torch.equal(out.cpu(), library.state_at(mid.cpu(), time.cpu()))


@needs_mjlab
def test_floor_distance_tracks_root_height_and_penetration():
    from rlinf.envs.sim.g1_mimic.bones_geometry import G1FloorDistance

    geometry = G1FloorDistance()
    base = geometry.frame(_standing(1)[0].double().numpy())
    assert 0.0 <= base[1] < 0.01 and 0.0 <= base[2] < 0.01
    assert base[0] <= min(base[1], base[2]) + 1e-9
    for dz in (0.3, -0.1):
        shifted = geometry.frame(_standing(1, dz)[0].double().numpy())
        assert shifted == pytest.approx(tuple(v + dz for v in base), abs=1e-4)
    zmin, feet = geometry.clip(_standing(5))
    assert zmin.shape == (5,) and feet.shape == (5, 2)


@needs_mjlab
def test_tool_annotates_clips_and_counts_each_gate(tmp_path, monkeypatch):
    from rlinf.envs.sim.g1_mimic.bones import load_bones_library
    from toolkits.supermimic import bones_feasibility

    clips = tmp_path / "clips"
    clips.mkdir()
    _save(clips, 0, "walk_forward", 0.0)
    _save(clips, 1, "dance_tiptoe", 0.30)
    _save(clips, 2, "crawl_low", -0.15)
    _save(clips, 3, "idle_hover", 0.05)
    _save(clips, 4, "sit_on_chair", 0.0)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["bones_feasibility.py", "--clips", str(clips), "--out", str(out)]
    )
    assert bones_feasibility.main() == 0

    stats = json.loads((out / "bones_stats.json").read_text())
    assert stats["label"].startswith("experimental")
    assert stats["num_files"] == 5
    assert stats["after_name"] == 4
    assert stats["after_feasibility"] == 1
    assert stats["rejected"] == {
        "elevated support": 1,
        "ground penetration": 1,
        "floating feet": 1,
        "name": 1,
    }
    payload = torch.load(out / "clip_003.pt", weights_only=True)
    assert payload["zmin"].shape == (60,) and payload["foot_clearance"].shape == (60, 2)
    library = load_bones_library(str(out), 1, "cpu")
    assert library.clip_names == ["walk_forward"]


@needs_mjlab
def test_tool_refuses_unnamed_clips_and_in_place_writes(tmp_path):
    from toolkits.supermimic import bones_feasibility

    _save(tmp_path, 0, None, 0.0, frames=10)
    with pytest.raises(SystemExit, match="allow-unnamed"):
        bones_feasibility.process(tmp_path, None)
    stats = bones_feasibility.process(tmp_path, None, allow_unnamed=True)
    assert stats["after_name"] == 1 and stats["after_feasibility"] == 1
    with pytest.raises(SystemExit, match="must differ"):
        bones_feasibility.process(tmp_path, tmp_path)


def test_manifest_names_clips_and_rejects_changed_files(tmp_path):
    files = [_clip(tmp_path, 0), _clip(tmp_path, 1)]
    _manifest(tmp_path, files)
    library = load_motion_library(str(tmp_path), 2, "cpu")
    assert library.clip_names == ["src_0.csv", "src_1.csv"]

    torch.save({"tampered": True}, tmp_path / "clip_001.pt")
    with pytest.raises(ValueError, match="SHA-256"):
        load_motion_library(str(tmp_path), 2, "cpu")


def test_manifest_rejects_an_unlisted_clip(tmp_path):
    files = [_clip(tmp_path, 0)]
    _manifest(tmp_path, files)
    _clip(tmp_path, 1)
    with pytest.raises(ValueError, match="not listed"):
        load_motion_library(str(tmp_path), 2, "cpu")


def test_clips_without_manifest_keep_file_names(tmp_path):
    _clip(tmp_path, 0)
    assert load_motion_library(str(tmp_path), 1, "cpu").clip_names == ["clip_000.pt"]


@needs_mjlab
def test_tools_run_from_the_repo_root_as_documented(tmp_path):
    """``python toolkits/supermimic/<tool>.py`` with no PYTHONPATH, as a user runs it."""
    import os
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    source = tmp_path / "csv"
    source.mkdir()
    _csv(source / "a_run.csv")
    clips = tmp_path / "bones"
    clips.mkdir()
    _save(clips, 0, "walk_forward", 0.0)
    for argv, made in (
        (
            [
                "prepare_lafan1.py",
                "--out",
                str(tmp_path / "lafan1"),
                "--source-dir",
                str(source),
                "--skip-remote-check",
                "--allow-partial",
                "--accept-license",
            ],
            tmp_path / "lafan1" / "manifest.json",
        ),
        (
            [
                "bones_feasibility.py",
                "--clips",
                str(clips),
                "--stats-only",
                "--stats",
                str(tmp_path / "bones_stats.json"),
            ],
            tmp_path / "bones_stats.json",
        ),
    ):
        proc = subprocess.run(
            [sys.executable, f"toolkits/supermimic/{argv[0]}", *argv[1:]],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert made.is_file(), argv[0]


@needs_mjlab
def test_prepare_converts_csv_and_writes_a_checked_manifest(tmp_path, monkeypatch):
    source = tmp_path / "csv"
    source.mkdir()
    _csv(source / "b_walk.csv")
    _csv(source / "a_run.csv", frames=61)
    out = tmp_path / "clips"
    assert (
        _run_prepare(
            monkeypatch,
            [
                "--out",
                str(out),
                "--source-dir",
                str(source),
                "--skip-remote-check",
                "--allow-partial",
                "--accept-license",
            ],
        )
        == 0
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["checked_against_revision"] is False
    assert [c["source"] for c in manifest["clips"]] == ["a_run.csv", "b_walk.csv"]
    assert [c["frames_50hz"] for c in manifest["clips"]] == [101, 51]
    assert "CC BY-NC-ND 4.0" in manifest["license_note"]

    payload = torch.load(out / "clip_001.pt", weights_only=True)
    assert payload["fps"] == 30.0
    assert torch.allclose(
        payload["state"][0, 3:7],
        torch.tensor([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]).float(),
    )
    library = load_motion_library(str(out), 2, "cpu")
    assert library.clip_names == ["a_run.csv", "b_walk.csv"]
    assert library.lengths.tolist() == [101, 51]


def test_prepare_requires_the_license_acknowledgement(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_prepare(
            monkeypatch,
            [
                "--out",
                str(tmp_path),
                "--source-dir",
                str(tmp_path),
                "--skip-remote-check",
            ],
        )


def test_prepare_rejects_a_partial_set_by_default(tmp_path, monkeypatch):
    source = tmp_path / "csv"
    source.mkdir()
    _csv(source / "only.csv")
    with pytest.raises(SystemExit, match="expected 40"):
        _run_prepare(
            monkeypatch,
            [
                "--out",
                str(tmp_path / "out"),
                "--source-dir",
                str(source),
                "--skip-remote-check",
                "--accept-license",
            ],
        )
