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

"""Configuration and episode selection without importing Isaac Sim."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from rlinf.data.schema.embodied_types import JointPositionTargets

from .evaluation import normalize_profile


@dataclass(frozen=True)
class Bench2DexSpec:
    """Task and embodiment contract shared with the simulator child."""

    source_path: Path
    assets_path: Path
    task_path: Path
    robot_key: str
    action_dim: int
    full_dof: int
    collect_config: Path
    main_camera: str
    extra_cameras: tuple[str, ...]
    wrist_cameras: tuple[str, ...]
    physics_dt: float
    control_decimation: int
    max_control_steps: int
    active_joint_names: tuple[str, ...]
    full_joint_names: tuple[str, ...]
    active_indices: tuple[int, ...]

    @property
    def camera_names(self) -> tuple[str, ...]:
        return (self.main_camera, *self.extra_cameras, *self.wrist_cameras)

    @classmethod
    def from_config(cls, config: dict) -> Bench2DexSpec:
        """Resolve a task and read DOF metadata from the selected checkout."""
        source = Path(config["source_path"]).expanduser().resolve()
        assets = Path(config["assets_path"]).expanduser().resolve()
        task = Path(config["task"])
        if not task.is_absolute():
            task = source / "scenes" / task.with_suffix(".yaml")
        with task.open() as stream:
            task_data = yaml.safe_load(stream)
        with (source / "robots/active_dof_maps.yml").open() as stream:
            robots = yaml.safe_load(stream)["robots"]
        robot_key = config["embodiment"]
        if robot_key not in robots:
            raise ValueError(
                f"Unknown Bench2Dex embodiment {robot_key!r}; choose {sorted(robots)}"
            )
        dof = robots[robot_key]
        full_names = tuple(dof["full_joint_names"])
        active_indices = tuple(int(index) for index in dof["active_indices"])
        if (
            len(full_names) != int(dof["full_dof"])
            or len(set(full_names)) != len(full_names)
            or len(active_indices) != int(dof["active_dof"])
            or len(set(active_indices)) != len(active_indices)
            or any(index < 0 or index >= len(full_names) for index in active_indices)
        ):
            raise ValueError(f"Invalid named joint layout for {robot_key!r}")
        dt = float(config["physics_dt"])
        decimation = int(config["control_decimation"])
        if dt <= 0 or decimation <= 0:
            raise ValueError("physics_dt and control_decimation must be positive")
        budget = config.get("max_control_steps")
        if budget is None:
            budget = math.ceil(
                1.5 * task_data["metrics"]["expert_time_step"] / decimation
            )
        if int(budget) <= 0:
            raise ValueError(f"Expected a positive control-step budget, got {budget}")
        cameras = config["cameras"]
        collect = Path(config["collect_config"])
        if not collect.is_absolute():
            collect = source / collect
        spec = cls(
            source,
            assets,
            task.resolve(),
            robot_key,
            int(dof["active_dof"]),
            int(dof["full_dof"]),
            collect,
            cameras["main"],
            tuple(cameras.get("extra", [])),
            tuple(cameras.get("wrist", [])),
            dt,
            decimation,
            int(budget),
            tuple(full_names[index] for index in active_indices),
            full_names,
            active_indices,
        )
        if len(set(spec.camera_names)) != len(spec.camera_names):
            raise ValueError(f"Camera roles must be unique, got {spec.camera_names}")
        return spec


def joint_targets_in_runtime_order(
    actions,
    *,
    active_joint_names: tuple[str, ...],
    full_joint_names: tuple[str, ...],
) -> tuple[np.ndarray, str]:
    """Validate absolute joint targets and reorder named commands for execution."""
    layout = "active"
    names = active_joint_names
    if isinstance(actions, JointPositionTargets):
        if len(actions.joint_names) != 1:
            raise ValueError("Expected named targets for one simulator environment")
        layout = actions.joint_layout
        names = tuple(actions.joint_names[0])
        values = actions.joint_positions
        if values.ndim == 3:
            values = values[0]
    else:
        values = actions
    expected = active_joint_names if layout == "active" else full_joint_names
    values = np.asarray(values, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape != (values.shape[0], len(names))
        or not len(values)
    ):
        raise ValueError(
            f"Expected nonempty [C,{len(names)}] joint targets, got {values.shape}"
        )
    if len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError(f"Expected joint names {expected}, got {names}")
    if not np.isfinite(values).all():
        raise ValueError("Bench2Dex joint targets must be finite")
    order = [names.index(name) for name in expected]
    return values[:, order], layout


@dataclass(frozen=True)
class Bench2DexEpisode:
    """An optional scene anchor and an explicit seed supplied by a manifest."""

    anchor_hdf5: Path | None
    seed: int | None = None


def _anchor_pool(anchor_dir: str) -> list[Bench2DexEpisode]:
    path = Path(anchor_dir).expanduser().resolve()
    main, variants = [], []
    for anchor in path.glob("episode_*.hdf5"):
        match = re.fullmatch(r"episode_(\d+)(_1)?\.hdf5", anchor.name)
        if match:
            (variants if match[2] else main).append((int(match[1]), anchor))
    main.sort()
    variants.sort()
    selected = [p for i, p in main if i < 25] + [p for i, p in variants if i >= 25]
    selected = selected or [p for _, p in main]
    if not selected:
        raise ValueError(f"Bench2Dex anchor directory has no episode files: {path}")
    return [Bench2DexEpisode(p) for p in selected]


def load_episodes(config: dict) -> list[Bench2DexEpisode]:
    """Resolve the official anchor pool or an explicitly seeded episode manifest.

    Directory anchors follow the native main-0-24 and variant-25+ ordering.
    The fully resampled channel uses no anchor. All returned paths are absolute.
    """
    profile = normalize_profile(config.get("generalization_profile", "none"))
    if profile == "inv_cov":
        return [Bench2DexEpisode(None)]
    if config.get("anchor_dir"):
        return _anchor_pool(config["anchor_dir"])
    manifest = config.get("episode_manifest")
    if manifest:
        path = Path(manifest).expanduser().resolve()
        with path.open() as stream:
            entries = json.load(stream)
        base = path.parent
    else:
        if not config.get("anchor_hdf5"):
            raise ValueError(
                f"Bench2Dex {profile} requires anchor_dir, anchor_hdf5 or episode_manifest"
            )
        entries = [{"anchor_hdf5": config["anchor_hdf5"]}]
        base = Path.cwd()
    if not isinstance(entries, list) or not entries:
        raise ValueError("episode_manifest must contain a nonempty JSON list")
    episodes = []
    for entry in entries:
        anchor = Path(entry["anchor_hdf5"]).expanduser()
        anchor = anchor if anchor.is_absolute() else base / anchor
        seed = int(entry["seed"]) if "seed" in entry else None
        if seed is not None and not 0 <= seed < 2**32:
            raise ValueError(f"Expected seed in [0, 2**32), got {seed}")
        if not anchor.is_file():
            raise FileNotFoundError(f"Bench2Dex anchor not found: {anchor}")
        episodes.append(Bench2DexEpisode(anchor.resolve(), seed))
    return episodes
