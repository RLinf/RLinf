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

"""Synthetic BONES-SEED G1 CSVs through the official-header converter.

No official CSV is in the repository. These tests check the published
header (centimeters, extrinsic XYZ Euler degrees, ``<joint>_dof`` degrees)
and that the ``clip_*.pt`` files load in MotionLibrary and the BONES filters.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from rlinf.envs.sim.g1_mimic.bones import load_bones_library, name_rejected
from rlinf.envs.sim.g1_mimic.constants import (
    G1_JOINT_NAMES,
    JOINT_POS,
    JOINT_VEL,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
    STATE_DIM,
)
from rlinf.envs.sim.g1_mimic.motion import load_motion_library
from toolkits.supermimic.convert_bones_seed import (
    CsvError,
    build_payload,
    clip_name_for,
    convert_paths,
    euler_xyz_extrinsic_to_wxyz,
    output_filename,
    parse_bones_csv,
)

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="mjlab is an optional SuperMimic extra",
)

ROOT_HEADER = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
]
Q0 = torch.zeros(29)


def _header(joint_order: list[str] | None = None) -> list[str]:
    names = list(G1_JOINT_NAMES if joint_order is None else joint_order)
    return ROOT_HEADER + [f"{name}_dof" for name in names]


def _write_csv(path: Path, rows: list[list[float]], joint_order=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _header(joint_order)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(
    frame: int,
    position_cm: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
    joints_deg: list[float] | None = None,
) -> list[float]:
    joints = [0.0] * len(G1_JOINT_NAMES) if joints_deg is None else joints_deg
    return [frame, *position_cm, *euler_deg, *joints]


def _constant_rows(frames: int, **kwargs) -> list[list[float]]:
    position = kwargs.get("position_cm", (0.0, 0.0, 90.0))
    euler = kwargs.get("euler_deg", (0.0, 0.0, 0.0))
    joints = kwargs.get("joints_deg")
    return [_row(index, position, euler, joints) for index in range(frames)]


def test_centimeters_become_meters_and_joints_follow_names(tmp_path):
    order = list(reversed(G1_JOINT_NAMES))
    degrees = [float(index + 1) for index in range(len(order))]
    # Column order is reversed, so the degrees travel with the joint name.
    rows = _constant_rows(8, position_cm=(100.0, -20.0, 80.0), joints_deg=degrees)
    path = tmp_path / "motion.csv"
    _write_csv(path, rows, joint_order=order)
    position_cm, _euler, joints_deg = parse_bones_csv(path)
    assert torch.allclose(
        position_cm[0],
        torch.tensor([100.0, -20.0, 80.0], dtype=position_cm.dtype),
    )
    by_name = {name: degrees[index] for index, name in enumerate(order)}
    expected = torch.tensor(
        [by_name[name] for name in G1_JOINT_NAMES], dtype=joints_deg.dtype
    )
    assert torch.allclose(joints_deg[0], expected)

    payload = build_payload(path, Path("motion.csv"), Q0, 120.0, 50.0)
    state = payload["state"]
    assert state.shape[1] == STATE_DIM
    assert state.dtype == torch.float32
    assert torch.isfinite(state).all()
    assert torch.allclose(state[0, ROOT_POS], torch.tensor([1.0, -0.2, 0.8]), atol=1e-5)
    assert torch.allclose(
        state[0, JOINT_POS],
        torch.deg2rad(expected).to(dtype=torch.float32),
        atol=1e-5,
    )


def test_extrinsic_xyz_euler_matches_the_axis_quaternions():
    z_only = euler_xyz_extrinsic_to_wxyz(torch.tensor([[0.0, 0.0, 90.0]]))[0]
    half = math.sqrt(0.5)
    assert torch.allclose(z_only, torch.tensor([half, 0.0, 0.0, half]), atol=1e-6)
    composed = euler_xyz_extrinsic_to_wxyz(torch.tensor([[90.0, 90.0, 0.0]]))[0]
    assert torch.allclose(composed, torch.tensor([0.5, 0.5, 0.5, -0.5]), atol=1e-6)


def test_euler_matches_scipy_from_euler_xyz():
    scipy_rotation = pytest.importorskip("scipy.spatial.transform")
    euler = torch.tensor([[20.0, -15.0, 40.0], [0.0, 90.0, -90.0]])
    got = euler_xyz_extrinsic_to_wxyz(euler)
    xyzw = scipy_rotation.Rotation.from_euler(
        "xyz", euler.numpy(), degrees=True
    ).as_quat()
    wxyz = torch.tensor(xyzw[:, [3, 0, 1, 2]], dtype=torch.float32)
    # q and -q are the same rotation.
    assert torch.allclose(got.abs(), wxyz.abs(), atol=1e-5)
    assert torch.allclose((got * wxyz).sum(dim=-1).abs(), torch.ones(2), atol=1e-5)


def test_constant_motion_has_zero_velocity_and_unit_quaternions(tmp_path):
    path = tmp_path / "stand.csv"
    _write_csv(path, _constant_rows(30, position_cm=(0.0, 0.0, 90.0)))
    state = build_payload(path, Path("stand.csv"), Q0, 120.0, 50.0)["state"]
    assert torch.allclose(
        state[:, ROOT_LIN_VEL], torch.zeros_like(state[:, ROOT_LIN_VEL])
    )
    assert torch.allclose(
        state[:, ROOT_ANG_VEL], torch.zeros_like(state[:, ROOT_ANG_VEL])
    )
    assert torch.allclose(state[:, JOINT_VEL], torch.zeros_like(state[:, JOINT_VEL]))
    norms = state[:, ROOT_QUAT].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    dots = (state[:-1, ROOT_QUAT] * state[1:, ROOT_QUAT]).sum(dim=-1)
    assert torch.all(dots >= -1e-5)


def test_constant_root_velocity_survives_120_to_50(tmp_path):
    velocity = torch.tensor([0.4, -0.2, 0.1])
    rows = []
    frames = 241  # 0 .. 2 s at 120 Hz
    for index in range(frames):
        time = index / 120.0
        position_m = velocity * time
        position_cm = tuple(float(value * 100.0) for value in position_m)
        rows.append(_row(index, position_cm, (0.0, 0.0, 0.0)))
    path = tmp_path / "slide.csv"
    _write_csv(path, rows)
    state = build_payload(path, Path("slide.csv"), Q0, 120.0, 50.0)["state"]
    # 2 s at 50 Hz is 101 frames, 0.00 .. 2.00 inclusive.
    assert state.shape[0] == 101
    assert torch.allclose(
        state[:, ROOT_LIN_VEL],
        velocity.to(dtype=torch.float32).expand_as(state[:, ROOT_LIN_VEL]),
        atol=1e-4,
    )


def test_constant_yaw_rate_is_root_frame_z(tmp_path):
    rate = 0.5  # rad/s about world Z, which stays the root Z axis
    rows = []
    frames = 241
    for index in range(frames):
        yaw_deg = math.degrees(rate * index / 120.0)
        rows.append(_row(index, (0.0, 0.0, 90.0), (0.0, 0.0, yaw_deg)))
    path = tmp_path / "spin.csv"
    _write_csv(path, rows)
    state = build_payload(path, Path("spin.csv"), Q0, 120.0, 50.0)["state"]
    expected = torch.tensor([0.0, 0.0, rate])
    assert torch.allclose(
        state[:, ROOT_ANG_VEL],
        expected.to(dtype=torch.float32).expand_as(state[:, ROOT_ANG_VEL]),
        atol=1e-4,
    )
    dots = (state[:-1, ROOT_QUAT] * state[1:, ROOT_QUAT]).sum(dim=-1)
    assert torch.all(dots > 0)


def test_joint_crossing_pi_interpolates_the_short_way(tmp_path):
    # Two source samples straddle +π. The 50 Hz grid samples the gap.
    # A long-way lerp would pass near 0; the short way stays near ±π.
    degrees = [0.0] * len(G1_JOINT_NAMES)
    knee = G1_JOINT_NAMES.index("left_knee_joint")
    rows = []
    for index in range(5):
        angle = 170.0 if index < 2 else -170.0
        joints = degrees.copy()
        joints[knee] = angle
        rows.append(_row(index, (0.0, 0.0, 90.0), (0.0, 0.0, 0.0), joints))
    path = tmp_path / "unwrap.csv"
    _write_csv(path, rows)
    state = build_payload(path, Path("unwrap.csv"), Q0, 120.0, 50.0)["state"]
    knee_angle = state[:, JOINT_POS][:, knee]
    assert torch.all(knee_angle.abs() > 2.5)


def test_bad_csv_rows_name_the_problem(tmp_path):
    good = _constant_rows(8)
    missing = tmp_path / "missing.csv"
    _write_csv(missing, good)
    text = missing.read_text(encoding="utf-8").replace(
        "left_knee_joint_dof", "not_a_joint_dof"
    )
    missing.write_text(text, encoding="utf-8")
    with pytest.raises(CsvError, match="unexpected columns"):
        parse_bones_csv(missing)

    dropped = tmp_path / "dropped.csv"
    header = _header()
    column = header.index("root_translateZ")
    header.pop(column)
    lines = [",".join(header)]
    for row in good:
        cells = [str(value) for value in row]
        cells.pop(column)
        lines.append(",".join(cells))
    dropped.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CsvError, match="missing columns"):
        parse_bones_csv(dropped)

    illegal_text = tmp_path / "text.csv"
    _write_csv(illegal_text, good)
    illegal_text.write_text(
        illegal_text.read_text(encoding="utf-8").replace("90.0", "bad", 1),
        encoding="utf-8",
    )
    with pytest.raises(CsvError, match="non-numeric"):
        parse_bones_csv(illegal_text)

    illegal = tmp_path / "nan.csv"
    _write_csv(illegal, good)
    illegal.write_text(
        illegal.read_text(encoding="utf-8").replace("90.0", "nan", 1), encoding="utf-8"
    )
    with pytest.raises(CsvError, match="non-finite"):
        parse_bones_csv(illegal)

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(CsvError, match="empty"):
        parse_bones_csv(empty)

    short = tmp_path / "short.csv"
    _write_csv(short, _constant_rows(2))
    with pytest.raises(CsvError, match="shorter than one"):
        build_payload(short, Path("short.csv"), Q0, 120.0, 50.0)


def test_same_sanitized_name_does_not_overwrite_the_first(tmp_path):
    root = tmp_path / "csv"
    _write_csv(root / "a" / "b.csv", _constant_rows(8, position_cm=(0.0, 0.0, 80.0)))
    _write_csv(root / "a__b.csv", _constant_rows(8, position_cm=(0.0, 0.0, 110.0)))
    assert output_filename(Path("a/b.csv")) == output_filename(Path("a__b.csv"))
    out = tmp_path / "clips"
    summary = convert_paths([root], out, Q0, jobs=2)
    assert summary.converted == 1
    assert summary.failed == 1
    assert any("also comes from" in reason for _label, reason in summary.failures)
    kept = torch.load(out / "clip_a__b.pt", weights_only=True)
    # ``a/b.csv`` sorts first and keeps its own root height.
    assert torch.allclose(kept["state"][0, ROOT_POS], torch.tensor([0.0, 0.0, 0.8]))
    assert kept["name"] == "a/b"


def test_existing_output_is_kept_unless_overwrite(tmp_path):
    root = tmp_path / "csv"
    _write_csv(root / "walk.csv", _constant_rows(8, position_cm=(0.0, 0.0, 80.0)))
    out = tmp_path / "clips"
    first = convert_paths([root], out, Q0)
    assert first.converted == 1
    dest = out / output_filename(Path("walk.csv"))
    original = dest.read_bytes()
    _write_csv(root / "walk.csv", _constant_rows(8, position_cm=(0.0, 0.0, 150.0)))
    second = convert_paths([root], out, Q0)
    assert second.converted == 0
    assert second.skipped == 1
    assert dest.read_bytes() == original
    third = convert_paths([root], out, Q0, overwrite=True)
    assert third.converted == 1
    replaced = torch.load(dest, weights_only=True)
    assert torch.allclose(replaced["state"][0, 2], torch.tensor(1.5), atol=1e-5)


def test_converted_clip_loads_and_passes_the_bones_filters(tmp_path):
    root = tmp_path / "g1" / "csv"
    _write_csv(
        root / "2024-01-01" / "walk_forward.csv",
        _constant_rows(16, position_cm=(0.0, 0.0, 90.0)),
    )
    _write_csv(
        root / "2024-01-01" / "sit_on_chair.csv",
        _constant_rows(16, position_cm=(0.0, 0.0, 50.0)),
    )
    _write_csv(
        root / "2024-01-02" / "jump_off_50cm.csv",
        _constant_rows(16, position_cm=(0.0, 0.0, 90.0)),
    )
    out = tmp_path / "clips"
    summary = convert_paths([root], out, Q0)
    assert summary.failed == 0
    assert summary.converted == 3
    library = load_motion_library(str(out), None, "cpu")
    assert library.num_clips == 3
    assert library.clip(0).shape[-1] == STATE_DIM

    bones = load_bones_library(str(out), None, "cpu")
    assert bones.clip_names == ["2024-01-01/walk_forward"]
    assert name_rejected("2024-01-01/sit_on_chair")
    assert name_rejected("2024-01-02/jump_off_50cm")
    assert bones.filter_counts["rejected_name"] == 2
    assert bones.filter_counts["rejected_feasibility"] == 0

    walk = torch.load(
        out / output_filename(Path("2024-01-01/walk_forward.csv")), weights_only=True
    )
    frames = walk["state"].shape[0]
    walk["zmin"] = torch.full((frames,), -0.2)
    walk["foot_clearance"] = torch.zeros(frames, 2)
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    torch.save(walk, filtered / "clip_walk.pt")
    with pytest.raises(RuntimeError, match="removed every clip"):
        load_bones_library(str(filtered), None, "cpu")

    walk["zmin"] = torch.zeros(frames)
    walk["foot_clearance"] = torch.zeros(frames, 2)
    torch.save(walk, filtered / "clip_walk.pt")
    kept = load_bones_library(str(filtered), None, "cpu")
    assert kept.num_clips == 1
    assert kept.filter_counts["rejected_feasibility"] == 0
    assert (
        clip_name_for(Path("2024-01-01/walk_forward.csv")) == "2024-01-01/walk_forward"
    )


@needs_mjlab
def test_cli_uses_the_mjlab_default_pose_and_fk_accepts_the_state(tmp_path):
    import os
    import subprocess
    import sys

    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state, g1_kinematics

    root = tmp_path / "csv"
    joints = torch.rad2deg(asset_default_state()[7:36]).tolist()
    _write_csv(
        root / "pose.csv",
        _constant_rows(12, position_cm=(10.0, 0.0, 90.0), joints_deg=joints),
    )
    _write_csv(
        root / "other.csv",
        _constant_rows(12, position_cm=(0.0, 0.0, 90.0), joints_deg=joints),
    )
    out = tmp_path / "clips"
    repo = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    proc = subprocess.run(
        [
            sys.executable,
            "toolkits/supermimic/convert_bones_seed.py",
            "--input",
            str(root),
            "--output",
            str(out),
            "--jobs",
            "2",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:] + proc.stdout[-2000:]
    payload = torch.load(next(out.glob("clip_*.pt")), weights_only=True)
    assert torch.allclose(payload["q0"], asset_default_state()[7:36], atol=1e-5)
    kinematics = g1_kinematics()
    body_pos, _rot, _lin, _ang = kinematics.forward(payload["state"])
    assert torch.isfinite(body_pos).all()
    # The free joint copies the converted root. A wrong 71-d layout would not.
    assert kinematics.free_tracked is not None
    assert torch.allclose(
        body_pos[:, kinematics.free_tracked],
        payload["state"][:, ROOT_POS],
        atol=1e-5,
    )
