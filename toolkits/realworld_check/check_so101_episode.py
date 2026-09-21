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

"""Validate an SO-101 LeRobot episode without accessing robot hardware.

The checker accepts the canonical field names from the validation schema and the
aliases emitted by the current RLinf LeRobot writer. It only reads dataset and
configuration files; it never opens a camera or serial port.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = REPO_ROOT / "examples/embodiment/config/so101/validation_schema.yaml"


@dataclass
class CheckReport:
    """Machine-readable result of an SO-101 dataset validation check."""

    dataset_path: str
    baseline_path: str
    passed: bool = True
    summary: dict[str, Any] = field(default_factory=dict)
    field_mapping: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        """Record a validation failure."""
        self.passed = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        """Record a non-blocking future-contract gap."""
        self.warnings.append(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def _resolve_dataset_path(raw_path: str, baseline_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if baseline_path.resolve() == DEFAULT_BASELINE.resolve():
        return REPO_ROOT / path
    return baseline_path.parent / path


def _resolve_data_files(dataset_path: Path) -> list[Path]:
    data_dir = dataset_path / "data"
    return sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []


def _find_alias(columns: set[str], aliases: list[str]) -> str | None:
    return next((alias for alias in aliases if alias in columns), None)


def _scalar_bool(value: Any) -> bool:
    array = np.asarray(value, dtype=bool).reshape(-1)
    if array.size != 1:
        raise ValueError(f"expected one bool value, got shape {array.shape}")
    return bool(array[0])


def _validate_vectors(
    values: list[Any],
    *,
    field_name: str,
    expected_dim: int,
    normalized_range: tuple[float, float],
    report: CheckReport,
    last_component_range: tuple[float, float] | None = None,
) -> None:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        report.error(f"{field_name} could not be converted to float32: {exc}")
        return
    if array.ndim != 2 or array.shape[1] != expected_dim:
        report.error(
            f"{field_name} must have shape [frames, {expected_dim}], got {array.shape}."
        )
        return
    if not np.isfinite(array).all():
        report.error(f"{field_name} contains non-finite values.")
    low, high = normalized_range
    if np.any(array < low - 1e-6) or np.any(array > high + 1e-6):
        report.error(f"{field_name} contains values outside [{low}, {high}].")
    if last_component_range is not None:
        grip_low, grip_high = last_component_range
        gripper = array[:, -1]
        if np.any(gripper < grip_low - 1e-6) or np.any(gripper > grip_high + 1e-6):
            report.error(
                f"{field_name} gripper component contains values outside "
                f"[{grip_low}, {grip_high}]."
            )


def _validate_so101_vectors(
    values: list[Any],
    *,
    field_name: str,
    expected_dim: int,
    control: dict[str, Any],
    report: CheckReport,
) -> None:
    """Accept both RLinf-normalized and canonical LeRobot SO-101 units.

    Local collection may write radians with either a ``[-1, 1]`` control
    gripper or a ``[0, 1]`` environment gripper.  The LeRobot v3 recorder
    writes degree-like joint values and a ``[0, 100]`` gripper.  Select the
    joint envelope first, then accept both normalized gripper conventions.
    """
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        report.error(f"{field_name} could not be converted to float32: {exc}")
        return
    if array.ndim != 2 or array.shape[1] != expected_dim:
        report.error(
            f"{field_name} must have shape [frames, {expected_dim}], got {array.shape}."
        )
        return
    local_joint_range = tuple(float(value) for value in control["local_joint_range"])
    local_gripper_range = tuple(
        float(value) for value in control["local_gripper_range"]
    )
    canonical_joint_range = tuple(
        float(value) for value in control["canonical_joint_range"]
    )
    canonical_gripper_range = tuple(
        float(value) for value in control["canonical_gripper_range"]
    )
    canonical_units = np.any(array[:, :-1] < local_joint_range[0] - 1e-6) or np.any(
        array[:, :-1] > local_joint_range[1] + 1e-6
    )
    canonical_units = canonical_units or np.any(
        array[:, -1] < local_gripper_range[0] - 1e-6
    )
    canonical_units = canonical_units or np.any(
        array[:, -1] > local_gripper_range[1] + 1e-6
    )
    if canonical_units:
        joint_range, gripper_range = canonical_joint_range, canonical_gripper_range
    else:
        joint_range, gripper_range = local_joint_range, local_gripper_range
    _validate_vectors(
        values,
        field_name=field_name,
        expected_dim=expected_dim,
        normalized_range=joint_range,
        last_component_range=gripper_range,
        report=report,
    )


def _validate_images(
    values: list[Any],
    *,
    field_name: str,
    expected_shape: list[int] | None,
    dataset_path: Path,
    report: CheckReport,
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("SO101 image checking requires Pillow.") from exc

    invalid_frames: list[str] = []
    for frame_index, value in enumerate(values):
        try:
            if not isinstance(value, dict):
                raise ValueError("image cell is not a LeRobot image struct")
            raw_bytes = value.get("bytes")
            raw_path = value.get("path")
            if raw_bytes:
                source: Any = io.BytesIO(raw_bytes)
            elif raw_path:
                source = dataset_path / raw_path
                if not source.is_file():
                    raise ValueError(f"image path does not exist: {raw_path}")
            else:
                raise ValueError("image has neither embedded bytes nor a path")
            with Image.open(source) as image:
                image.load()
                shape = [image.height, image.width, len(image.getbands())]
            if expected_shape is not None and shape != expected_shape:
                raise ValueError(f"decoded shape={shape}, expected={expected_shape}")
        except (OSError, TypeError, ValueError) as exc:
            invalid_frames.append(f"frame {frame_index}: {exc}")
    if invalid_frames:
        preview = "; ".join(invalid_frames[:3])
        report.error(
            f"{field_name} has {len(invalid_frames)} invalid frames ({preview})."
        )


def _validate_episode_sequences(
    table: Any,
    mapping: dict[str, str],
    fps: float,
    report: CheckReport,
) -> None:
    episode_values = np.asarray(table[mapping["episode_index"]].to_pylist())
    frame_values = np.asarray(table["frame_index"].to_pylist())
    timestamps = np.asarray(table[mapping["timestamp"]].to_pylist(), dtype=float)
    done_values = table[mapping["done"]].to_pylist() if "done" in mapping else None
    success_values = (
        table[mapping["success"]].to_pylist() if "success" in mapping else None
    )

    for episode_index in np.unique(episode_values):
        mask = episode_values == episode_index
        frames = frame_values[mask]
        times = timestamps[mask]
        dones = (
            [_scalar_bool(v) for v, selected in zip(done_values, mask) if selected]
            if done_values is not None
            else None
        )
        successes = (
            [_scalar_bool(v) for v, selected in zip(success_values, mask) if selected]
            if success_values is not None
            else None
        )
        expected_frames = np.arange(len(frames), dtype=frames.dtype)
        if not np.array_equal(frames, expected_frames):
            report.error(
                f"episode {episode_index} frame_index is not contiguous from zero."
            )
        if len(times) and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            report.error(
                f"episode {episode_index} timestamps are not finite/increasing."
            )
        if len(times) > 1:
            expected_period = 1.0 / fps
            max_period_error = float(np.max(np.abs(np.diff(times) - expected_period)))
            if max_period_error > max(1e-3, expected_period * 0.05):
                report.error(
                    f"episode {episode_index} timestamp spacing exceeds 5% tolerance "
                    f"(max error {max_period_error:.6f}s)."
                )
        if dones is not None and (
            not dones or dones[-1] is not True or any(dones[:-1])
        ):
            report.error(
                f"episode {episode_index} must have done=true only on its final frame."
            )
        if successes is not None and (not successes or not successes[-1]):
            report.error(f"episode {episode_index} is not marked successful.")
        elif successes is not None and len(successes) > 1 and all(successes):
            report.warn(
                f"episode {episode_index} has success=true on every frame; treat this "
                "field as an episode-level outcome, not an instantaneous RL signal."
            )


def _video_files_exist(dataset_path: Path, video_key: str) -> bool:
    """Return whether a LeRobot video feature has at least one media file."""
    return any((dataset_path / "videos" / video_key).rglob("*.mp4"))


def _validate_video_feature(
    dataset_path: Path,
    feature_name: str,
    expected_shape: list[int] | None,
    info: dict[str, Any],
    report: CheckReport,
) -> None:
    """Validate a video feature stored outside parquet by LeRobot v3."""
    feature = info.get("features", {}).get(feature_name, {})
    if feature.get("dtype") != "video":
        report.error(f"Missing video feature metadata for {feature_name}.")
        return
    files = list((dataset_path / "videos" / feature_name).rglob("*.mp4"))
    if not files:
        report.error(f"No video files found for {feature_name}.")
        return
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3:
        report.error(f"{feature_name} metadata must describe an RGB video: {shape!r}.")
    if expected_shape is not None and shape != expected_shape:
        report.warn(
            f"{feature_name} metadata shape={shape!r} differs from configured "
            f"shape={expected_shape!r}; camera resolution is deployment-specific."
        )


def check_dataset(dataset_path: Path, baseline_path: Path) -> CheckReport:
    """Check one finalized SO-101 LeRobot dataset against the validation schema."""
    report = CheckReport(str(dataset_path), str(baseline_path))
    baseline = _load_yaml(baseline_path)
    info_path = dataset_path / "meta/info.json"
    if not info_path.is_file():
        report.error(f"Missing LeRobot metadata: {info_path}")
        return report
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.error(f"Invalid JSON in {info_path}: {exc}")
        return report

    data_files = _resolve_data_files(dataset_path)
    if not data_files:
        report.error(f"No data parquet files found under {dataset_path / 'data'}.")
        return report

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("SO101 episode checking requires pyarrow.") from exc

    tables = [pq.read_table(path) for path in data_files]
    table = pa.concat_tables(tables)
    columns = set(table.column_names)
    contract = baseline["data_contract"]
    for canonical, spec in contract["required_basic"].items():
        alias = _find_alias(columns, list(spec["aliases"]))
        if alias is None:
            feature = info.get("features", {}).get(canonical, {})
            if feature.get("dtype") == "video" and _video_files_exist(
                dataset_path, canonical
            ):
                report.field_mapping[canonical] = canonical
                report.warn(
                    f"{canonical} is stored as a LeRobot video feature, not a parquet column."
                )
            elif spec.get("optional"):
                report.warn(
                    f"Optional field {canonical} is not present in the parquet schema."
                )
            else:
                report.error(
                    f"Missing required field {canonical}; accepted aliases={spec['aliases']}."
                )
        else:
            report.field_mapping[canonical] = alias

    for canonical, spec in contract.get("required_dagger", {}).items():
        if canonical not in columns:
            report.warn(
                f"{canonical} is planned for {spec['target_stage']} and is not present."
            )

    if report.errors:
        return report

    sample = baseline["sample"]
    total_frames = table.num_rows
    unique_episodes = set(table[report.field_mapping["episode_index"]].to_pylist())
    report.summary = {
        "robot_type": info.get("robot_type"),
        "episodes": len(unique_episodes),
        "frames": total_frames,
        "fps": info.get("fps"),
        "data_files": len(data_files),
    }
    expected_robot_types = sample.get("expected_robot_types")
    if expected_robot_types is None:
        expected_robot_types = [sample.get("expected_robot_type")]
    if info.get("robot_type") not in expected_robot_types:
        report.error(
            f"robot_type={info.get('robot_type')!r}, expected one of {expected_robot_types!r}."
        )
    for key, actual in (
        ("expected_episodes", len(unique_episodes)),
        ("expected_frames", total_frames),
    ):
        expected = sample.get(key)
        if expected is not None and actual != expected:
            report.error(f"{key}={expected}, dataset contains {actual}.")
    if info.get("total_frames") != total_frames:
        report.error(
            f"info.total_frames={info.get('total_frames')} but parquet rows={total_frames}."
        )
    if info.get("total_episodes") != len(unique_episodes):
        report.error("info.total_episodes does not match the parquet episode indices.")

    control = baseline["control"]
    _validate_so101_vectors(
        table[report.field_mapping["observation.state"]].to_pylist(),
        field_name="observation.state",
        expected_dim=int(control["state_dim"]),
        control=control,
        report=report,
    )
    _validate_so101_vectors(
        table[report.field_mapping["action"]].to_pylist(),
        field_name="action",
        expected_dim=int(control["action_dim"]),
        control=control,
        report=report,
    )
    image_spec = contract["required_basic"]["observation.images.wrist"]
    image_column = report.field_mapping["observation.images.wrist"]
    if image_column in columns:
        _validate_images(
            table[image_column].to_pylist(),
            field_name="observation.images.wrist",
            expected_shape=image_spec.get("shape"),
            dataset_path=dataset_path,
            report=report,
        )
    else:
        _validate_video_feature(
            dataset_path,
            "observation.images.wrist",
            image_spec.get("shape"),
            info,
            report,
        )
    if "frame_index" not in columns:
        report.error("Missing required LeRobot frame_index field.")
    else:
        fps = float(info.get("fps", 0))
        if not math.isfinite(fps) or fps <= 0:
            report.error(f"Invalid dataset fps: {info.get('fps')!r}.")
        else:
            _validate_episode_sequences(table, report.field_mapping, fps, report)

    task_alias = report.field_mapping["task"]
    if (
        task_alias == "task_index"
        and not (dataset_path / "meta/tasks.parquet").is_file()
    ):
        report.error("task_index is present but meta/tasks.parquet is missing.")
    return report


def _format_report(report: CheckReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [f"SO101 dataset check: {status}", f"dataset: {report.dataset_path}"]
    if report.summary:
        lines.append(
            "summary: "
            + ", ".join(f"{key}={value}" for key, value in report.summary.items())
        )
    lines.append("field mapping:")
    lines.extend(
        f"  {canonical} <- {source}"
        for canonical, source in sorted(report.field_mapping.items())
    )
    lines.extend(f"ERROR: {message}" for message in report.errors)
    lines.extend(f"PLANNED: {message}" for message in report.warnings)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_path", nargs="?", help="Finalized LeRobot dataset root"
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--json", action="store_true", help="Print JSON instead of text"
    )
    args = parser.parse_args()

    baseline_path = args.baseline.expanduser().resolve()
    baseline = _load_yaml(baseline_path)
    raw_dataset_path = args.dataset_path or baseline["sample"]["dataset_path"]
    dataset_path = _resolve_dataset_path(raw_dataset_path, baseline_path).resolve()
    report = check_dataset(dataset_path, baseline_path)
    if args.json:
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    else:
        print(_format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
