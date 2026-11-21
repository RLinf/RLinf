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

"""Convert official BONES-SEED G1 CSVs into ``clip_*.pt`` clips.

The public G1 release (Hugging Face ``bones-studio/seed``, directory
``g1/csv/{date}/{motion}.csv``) is already retargeted. ProtoMotions and the
GR00T-WholeBodyControl converter both read the same header. This script
follows that header. It has not been checked against an official CSV; the
unit tests use synthetic files with that header.

One row is::

    Frame,
    root_translateX, root_translateY, root_translateZ,   # centimeters
    root_rotateX, root_rotateY, root_rotateZ,           # degrees
    <joint>_dof, ...                                    # degrees, 29 joints

``root_rotate*`` is an extrinsic XYZ Euler angle: the same convention as
``scipy.spatial.transform.Rotation.from_euler("xyz", degrees=True)``
(lowercase ``xyz`` is extrinsic). ProtoMotions documents this as the
convention confirmed for this CSV. The GR00T script calls the same
``from_euler("xyz")``. Columns are MuJoCo X, Y, Z with no extra axis swap.
Joint columns are matched by name to ``G1_JOINT_NAMES`` after stripping
``_dof``, so a permuted header is reordered and a missing joint is an error.

Output is the MotionLibrary payload at 50 Hz (or 30 Hz). Positions and
joint angles are resampled with ``resample_states``, which slerps the root
quaternion and recomputes linear and angular velocity on that grid. The
saved ``name`` is the CSV path relative to the input directory, so the
Appendix E name filter still sees the motion name and two dates do not
overwrite each other.

This script does not download data and does not apply the name or
feasibility filters. After conversion, run ``bones_feasibility.py``.

Convert one official directory::

    python toolkits/supermimic/convert_bones_seed.py \\
        --input /data/bones-seed/g1/csv \\
        --output /data/bones_g1
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# ``python toolkits/supermimic/convert_bones_seed.py`` puts only this folder
# on sys.path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROOT_POS_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_EULER_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")
DOF_SUFFIX = "_dof"
IGNORED_COLUMNS = frozenset({"frame"})
# MotionLibrary accepts only these rates. 50 Hz is stored as-is; 30 Hz is
# resampled again when the clip is loaded.
LOADER_FPS = (30.0, 50.0)
CSV_CONVENTION = (
    "BONES-SEED G1 CSV: root position in centimeters, extrinsic XYZ Euler "
    "degrees (scipy from_euler 'xyz'), joint degrees named <joint>_dof in "
    "any order. No axis swap. Not checked on an official file."
)


class CsvError(ValueError):
    """One CSV cannot be converted. Other files can still succeed."""


@dataclass
class ConvertSummary:
    """Counts for one conversion run."""

    converted: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


def default_q0():
    """mjlab's G1 default joint pose, the ``q0`` the simulator requires."""
    try:
        from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
    except ImportError as exc:
        raise SystemExit(
            "convert_bones_seed needs mjlab to read the G1 default pose. "
            "Use the SuperMimic environment from requirements/install.sh."
        ) from exc
    return asset_default_state()[7:36].clone().float()


def euler_xyz_extrinsic_to_wxyz(euler_deg):
    """Extrinsic XYZ Euler degrees to a wxyz quaternion.

    ``q = q_z * q_y * q_x`` with Hamilton product, matching a fixed-frame
    rotation about X, then Y, then Z. This is scipy ``from_euler("xyz")``.
    """
    import math

    import torch

    from rlinf.envs.sim.g1_mimic.features import quat_mul, quat_normalize

    half = euler_deg * (math.pi / 180.0) * 0.5
    cosine = torch.cos(half)
    sine = torch.sin(half)
    zero = torch.zeros_like(cosine[..., 0])
    qx = torch.stack((cosine[..., 0], sine[..., 0], zero, zero), dim=-1)
    qy = torch.stack((cosine[..., 1], zero, sine[..., 1], zero), dim=-1)
    qz = torch.stack((cosine[..., 2], zero, zero, sine[..., 2]), dim=-1)
    return quat_normalize(quat_mul(qz, quat_mul(qy, qx)))


def make_quat_continuous(quat):
    """Flip signs so each quaternion has a non-negative dot with the previous."""
    import torch

    out = quat.clone()
    for index in range(1, out.shape[0]):
        if torch.dot(out[index], out[index - 1]) < 0:
            out[index] = -out[index]
    return out


def unwrap_joint_radians(joints):
    """Remove 2π jumps so linear interpolation takes the short step."""
    import math

    import torch

    if joints.shape[0] < 2:
        return joints
    out = joints.clone()
    delta = out[1:] - out[:-1]
    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    out[1:] = out[:1] + torch.cumsum(delta, dim=0)
    return out


def joint_name_from_column(column: str) -> str | None:
    """G1 joint name for a CSV column, or None when it is not a joint."""
    text = column.strip()
    if text.lower() in IGNORED_COLUMNS:
        return None
    if text in ROOT_POS_COLUMNS or text in ROOT_EULER_COLUMNS:
        return None
    if text.endswith(DOF_SUFFIX):
        return text[: -len(DOF_SUFFIX)]
    return text


def read_csv_table(path: Path):
    """Return ``(header, rows)`` as strings. Blank lines are dropped."""
    import csv

    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise CsvError(f"cannot read {path.name}: {exc}") from exc
    with handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise CsvError(f"{path.name} is empty") from exc
        header = [cell.strip() for cell in header]
        if not any(header):
            raise CsvError(f"{path.name} has an empty header")
        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise CsvError(f"{path.name} repeats columns {duplicates}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise CsvError(
                    f"{path.name}:{line_number} has {len(row)} cells, "
                    f"header has {len(header)}"
                )
            rows.append([cell.strip() for cell in row])
    if not rows:
        raise CsvError(f"{path.name} has a header but no data rows")
    return header, rows


def parse_bones_csv(path: Path):
    """Root position (cm), extrinsic XYZ Euler (deg), joints (deg) in G1 order."""
    import numpy as np
    import torch

    from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM, G1_JOINT_NAMES

    header, rows = read_csv_table(path)
    missing = [
        name for name in ROOT_POS_COLUMNS + ROOT_EULER_COLUMNS if name not in header
    ]
    if missing:
        raise CsvError(f"{path.name} is missing columns {missing}")
    seen: dict[str, int] = {}
    unknown = []
    for index, column in enumerate(header):
        joint = joint_name_from_column(column)
        if joint is None and column not in ROOT_POS_COLUMNS + ROOT_EULER_COLUMNS:
            if column.lower() not in IGNORED_COLUMNS:
                unknown.append(column)
            continue
        if joint is None:
            continue
        if joint not in G1_JOINT_NAMES:
            unknown.append(column)
            continue
        if joint in seen:
            raise CsvError(f"{path.name} repeats joint {joint}")
        seen[joint] = index
    if unknown:
        raise CsvError(f"{path.name} has unexpected columns {unknown}")
    absent = [name for name in G1_JOINT_NAMES if name not in seen]
    if absent:
        raise CsvError(
            f"{path.name} is missing {len(absent)} G1 joints, e.g. {absent[:3]}"
        )
    if len(seen) != ACTION_DIM:
        raise CsvError(
            f"{path.name} has {len(seen)} joints, the G1 model has {ACTION_DIM}"
        )
    try:
        values = np.asarray(rows, dtype=np.float64)
    except ValueError as exc:
        raise CsvError(f"{path.name} has a non-numeric cell: {exc}") from exc
    if not np.isfinite(values).all():
        raise CsvError(f"{path.name} has a non-finite value")
    pos_index = [header.index(name) for name in ROOT_POS_COLUMNS]
    euler_index = [header.index(name) for name in ROOT_EULER_COLUMNS]
    joint_index = [seen[name] for name in G1_JOINT_NAMES]
    position_cm = torch.from_numpy(values[:, pos_index])
    euler_deg = torch.from_numpy(values[:, euler_index])
    joints_deg = torch.from_numpy(values[:, joint_index])
    return position_cm, euler_deg, joints_deg


def state_from_csv_columns(position_cm, euler_deg, joints_deg):
    """71-d state at the CSV rate. Velocities are filled by resampling."""
    import torch

    from rlinf.envs.sim.g1_mimic.constants import (
        JOINT_POS,
        JOINT_VEL,
        ROOT_ANG_VEL,
        ROOT_LIN_VEL,
        ROOT_POS,
        ROOT_QUAT,
        STATE_DIM,
    )

    frames = int(position_cm.shape[0])
    state = torch.zeros(frames, STATE_DIM, dtype=torch.float32)
    state[:, ROOT_POS] = (position_cm / 100.0).to(dtype=torch.float32)
    quat = euler_xyz_extrinsic_to_wxyz(euler_deg.to(dtype=torch.float32))
    state[:, ROOT_QUAT] = make_quat_continuous(quat)
    joints = unwrap_joint_radians(torch.deg2rad(joints_deg).to(dtype=torch.float32))
    state[:, JOINT_POS] = joints
    state[:, ROOT_LIN_VEL] = 0
    state[:, ROOT_ANG_VEL] = 0
    state[:, JOINT_VEL] = 0
    return state


def resample_csv_state(state, input_fps: float, output_fps: float):
    """Resample a CSV-rate state onto ``output_fps`` and keep quaternions continuous."""
    from rlinf.envs.sim.g1_mimic.constants import ROOT_QUAT
    from rlinf.envs.sim.g1_mimic.motion import resample_states

    converted = resample_states(state, float(input_fps), float(output_fps))
    converted = converted.clone()
    converted[:, ROOT_QUAT] = make_quat_continuous(converted[:, ROOT_QUAT])
    return converted


def clip_name_for(relative: Path) -> str:
    """Filter name: the CSV path relative to the input root, without the suffix."""
    return relative.with_suffix("").as_posix()


def output_filename(relative: Path) -> str:
    """``clip_*.pt`` name. Slashes become ``__`` so dates do not collide."""
    parts = []
    for part in relative.with_suffix("").parts:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("._")
        parts.append(cleaned or "_")
    return "clip_" + "__".join(parts) + ".pt"


def build_payload(
    csv_path: Path, relative: Path, q0, input_fps: float, output_fps: float
):
    """One MotionLibrary payload. Raises ``CsvError`` or ``ValueError``."""
    import torch

    from rlinf.envs.sim.g1_mimic.constants import G1_JOINT_NAMES

    if output_fps not in LOADER_FPS:
        raise CsvError(
            f"output fps {output_fps} is not loadable. "
            f"MotionLibrary accepts {LOADER_FPS[0]:.0f} or {LOADER_FPS[1]:.0f} Hz."
        )
    if input_fps <= 0 or output_fps <= 0:
        raise CsvError(
            f"frame rates must be positive, got input {input_fps} output {output_fps}"
        )
    position_cm, euler_deg, joints_deg = parse_bones_csv(csv_path)
    source = state_from_csv_columns(position_cm, euler_deg, joints_deg)
    try:
        state = resample_csv_state(source, input_fps, output_fps)
    except ValueError as exc:
        raise CsvError(f"{csv_path.name}: {exc}") from exc
    if not torch.isfinite(state).all():
        raise CsvError(f"{csv_path.name} produced a non-finite state")
    return {
        "fps": float(output_fps),
        "joint_names": list(G1_JOINT_NAMES),
        "q0": q0.detach().to(dtype=torch.float32).reshape(-1).cpu(),
        "state": state.cpu(),
        "name": clip_name_for(relative),
        "source": relative.as_posix(),
        "source_fps": float(input_fps),
        "csv_convention": CSV_CONVENTION,
    }


def collect_csvs(inputs: list[Path]) -> list[tuple[Path, Path]]:
    """``(absolute csv, path relative to the directory the user passed)``."""
    found: list[tuple[Path, Path]] = []
    for raw in inputs:
        path = raw.expanduser()
        if not path.exists():
            raise SystemExit(f"input does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() != ".csv":
                raise SystemExit(f"input file is not a CSV: {path}")
            found.append((path.resolve(), Path(path.name)))
            continue
        if not path.is_dir():
            raise SystemExit(f"input is not a file or directory: {path}")
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() == ".csv"
        )
        found.extend((item.resolve(), item.relative_to(path)) for item in files)
    found.sort(key=lambda item: item[1].as_posix())
    return found


def _write_payload(payload: dict, dest: Path) -> None:
    import torch

    dest.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=dest.parent, prefix=dest.name + ".", suffix=".tmp", delete=False
    )
    handle.close()
    temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, dest)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _importable_convert_task():
    """The worker as a package function, including when this file is ``__main__``.

    A process pool started from ``python toolkits/supermimic/convert_bones_seed.py``
    cannot unpickle a function whose module is ``__main__``.
    """
    import importlib

    return importlib.import_module(
        "toolkits.supermimic.convert_bones_seed"
    )._convert_task


def _convert_task(task: dict) -> dict:
    """Convert one CSV. Used directly and inside a process pool."""
    import torch

    csv_path = Path(task["csv"])
    relative = Path(task["relative"])
    dest = Path(task["dest"])
    label = relative.as_posix()
    try:
        if dest.is_file() and not task["overwrite"]:
            return {"status": "skipped", "label": label, "output": dest.name}
        payload = build_payload(
            csv_path,
            relative,
            torch.tensor(task["q0"], dtype=torch.float32),
            float(task["input_fps"]),
            float(task["output_fps"]),
        )
        _write_payload(payload, dest)
    except (CsvError, ValueError, OSError) as exc:
        return {"status": "failed", "label": label, "reason": str(exc)}
    frames = int(payload["state"].shape[0])
    return {
        "status": "ok",
        "label": label,
        "output": dest.name,
        "frames": frames,
        "name": payload["name"],
    }


def convert_paths(
    inputs: list[Path],
    output: Path,
    q0,
    input_fps: float = 120.0,
    output_fps: float = 50.0,
    overwrite: bool = False,
    jobs: int = 1,
) -> ConvertSummary:
    """Convert every CSV under ``inputs``. One bad file does not stop the rest."""
    pairs = collect_csvs(inputs)
    summary = ConvertSummary()
    if not pairs:
        summary.failed = 1
        summary.failures.append(("", "no CSV files found"))
        print("[bones-seed] fail: no CSV files found", flush=True)
        return summary
    output.mkdir(parents=True, exist_ok=True)
    claimed: dict[str, str] = {}
    tasks = []
    for csv_path, relative in pairs:
        filename = output_filename(relative)
        label = relative.as_posix()
        previous = claimed.get(filename)
        if previous is not None:
            reason = (
                f"output {filename} also comes from {previous}; "
                "refusing to overwrite it in this run"
            )
            summary.failed += 1
            summary.failures.append((label, reason))
            print(f"[bones-seed] fail {label}: {reason}", flush=True)
            continue
        claimed[filename] = label
        tasks.append(
            {
                "csv": str(csv_path),
                "relative": label,
                "dest": str(output / filename),
                "overwrite": overwrite,
                "input_fps": float(input_fps),
                "output_fps": float(output_fps),
                "q0": [float(value) for value in q0.reshape(-1).tolist()],
            }
        )
    workers = max(int(jobs), 1)
    if workers == 1 or len(tasks) <= 1:
        results = [_convert_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_importable_convert_task(), tasks))
    total = len(pairs)
    done = 0
    for result in results:
        done += 1
        status = result["status"]
        label = result["label"]
        if status == "ok":
            summary.converted += 1
            summary.outputs.append(result["output"])
            print(
                f"[bones-seed] {done}/{total} ok {result['output']} <- {label} "
                f"({result['frames']} frames at {output_fps:g} Hz, "
                f"name={result['name']})",
                flush=True,
            )
        elif status == "skipped":
            summary.skipped += 1
            print(
                f"[bones-seed] {done}/{total} skip {label}: "
                f"{result['output']} exists (pass --overwrite to replace it)",
                flush=True,
            )
        else:
            summary.failed += 1
            summary.failures.append((label, result["reason"]))
            print(
                f"[bones-seed] {done}/{total} fail {label}: {result['reason']}",
                flush=True,
            )
    print(
        "[bones-seed] done: "
        f"{summary.converted} converted, {summary.skipped} skipped, "
        f"{summary.failed} failed",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        type=Path,
        help="a G1 CSV file, or a directory scanned recursively (g1/csv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="directory for clip_*.pt files",
    )
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument(
        "--output-fps",
        type=float,
        default=50.0,
        help="50 (default) or 30. Other rates cannot be loaded.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing clip_*.pt. The default is to skip it.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="worker processes. Each worker converts one file.",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    q0 = default_q0()
    summary = convert_paths(
        args.input,
        args.output,
        q0,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        overwrite=args.overwrite,
        jobs=args.jobs,
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
