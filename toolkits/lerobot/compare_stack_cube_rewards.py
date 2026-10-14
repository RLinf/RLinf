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

"""Compare frozen Stack Cube rewards with the pinned IsaacLab task source.

The numeric comparison uses unchanged source definitions and lightweight scene
objects. It does not launch Isaac Sim or validate cross-simulator physics.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import logging
import subprocess
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from prettytable import PrettyTable

ISAACLAB_COMMIT = "4246b6b4f4a3e74ee20e002ed7536b1c788d39f4"


@dataclass
class _RewardTermCfg:
    """Configuration fields consumed by the pinned reward manager."""

    func: object
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class _SceneEntityCfg:
    """A named asset in the detached reference scene."""

    name: str


def _load_definitions(path: Path, names: tuple[str, ...], namespace: dict) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if len(selected) != len(names):
        raise ValueError(f"Expected definitions {names} in {path}.")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *selected], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)


def compare_trace(trace_path: Path, root: Path) -> dict:
    """Compare a frozen trace against source from the qualified IsaacLab checkout."""
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != ISAACLAB_COMMIT:
        raise ValueError(
            f"IsaacLab must be checked out at {ISAACLAB_COMMIT}; got {commit}."
        )
    base = root / "source/isaaclab/isaaclab/managers/manager_base.py"
    manager_source = root / "source/isaaclab/isaaclab/managers/reward_manager.py"
    predicate_source = (
        root
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/mdp/terminations.py"
    )
    reward_cfg_source = (
        root
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_visumotor_rewarded_env_cfg.py"
    )
    termination_source = root / "source/isaaclab/isaaclab/envs/mdp/terminations.py"
    task_cfg_source = (
        root
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/stack_env_cfg.py"
    )
    source_paths = (
        base,
        manager_source,
        predicate_source,
        reward_cfg_source,
        termination_source,
        task_cfg_source,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            ISAACLAB_COMMIT,
            "--",
            *(str(path.relative_to(root)) for path in source_paths),
        ],
        check=True,
    )
    namespace = {
        "__name__": __name__,
        "torch": torch,
        "copy": copy,
        "inspect": inspect,
        "weakref": weakref,
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "PrettyTable": PrettyTable,
        "ManagerTermBaseCfg": _RewardTermCfg,
        "RewardTermCfg": _RewardTermCfg,
        "SceneEntityCfg": _SceneEntityCfg,
        "configclass": lambda cls: cls,
        "RewTerm": _RewardTermCfg,
        "logger": logging.getLogger("isaaclab_source_comparison"),
    }
    _load_definitions(base, ("ManagerTermBase", "ManagerBase"), namespace)
    _load_definitions(manager_source, ("RewardManager",), namespace)
    _load_definitions(predicate_source, ("cubes_stacked",), namespace)
    _load_definitions(termination_source, ("root_height_below_minimum",), namespace)
    namespace["mdp"] = SimpleNamespace(cubes_stacked=namespace["cubes_stacked"])
    _load_definitions(reward_cfg_source, ("RewardsCfg",), namespace)

    task_tree = ast.parse(task_cfg_source.read_text())
    settings = {}
    for node in ast.walk(task_tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if ast.unparse(target) in {"self.decimation", "self.sim.dt"}:
                    settings[ast.unparse(target)] = node.value.value
    dt = settings["self.decimation"] * settings["self.sim.dt"]
    with np.load(trace_path) as source:
        trace = {name: source[name] for name in source.files}
    positions = torch.from_numpy(trace["positions"].reshape(-1, 3, 3))
    fingers = torch.from_numpy(trace["fingers"].reshape(-1, 2))
    count = len(positions)
    scene = {
        "robot": SimpleNamespace(
            data=SimpleNamespace(joint_pos=fingers),
            find_joints=lambda names: ([0, 1], names),
        ),
        **{
            f"cube_{i + 1}": SimpleNamespace(
                data=SimpleNamespace(root_pos_w=positions[:, i])
            )
            for i in range(3)
        },
    }
    env = SimpleNamespace(
        num_envs=count,
        device="cpu",
        cfg=SimpleNamespace(
            gripper_joint_names=["left", "right"], gripper_open_val=0.04
        ),
        sim=SimpleNamespace(is_playing=lambda: True),
        scene=scene,
        max_episode_length_s=22.5,
    )
    manager = namespace["RewardManager"](
        {"success": namespace["RewardsCfg"].success}, env
    )
    expected = manager.compute(dt=dt).clone()
    actual = torch.from_numpy(trace["rewards"].reshape(-1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    failure = torch.stack(
        [
            namespace["root_height_below_minimum"](
                env, minimum_height=-0.05, asset_cfg=_SceneEntityCfg(f"cube_{i + 1}")
            )
            for i in range(3)
        ],
        dim=1,
    ).any(dim=1)
    if "failure" in trace:
        torch.testing.assert_close(
            failure, torch.from_numpy(trace["failure"].reshape(-1))
        )
    else:
        torch.testing.assert_close(failure, (positions[..., 2] < -0.05).any(dim=1))
    manager.reset()
    torch.testing.assert_close(manager.compute(dt=dt), actual, rtol=0, atol=0)
    report = {
        "scope": "Pinned IsaacLab RewardManager/ManagerBase class bodies, reward config and task predicate compiled unchanged from source; lightweight config and measured-state scene objects replace Omniverse runtime dependencies. This is source-level reward/termination validation, not an IsaacLab physics rollout.",
        "isaaclab_commit": "4246b6b4f4a3e74ee20e002ed7536b1c788d39f4",
        "scalar_rewards_compared": count,
        "nonzero_rewards": int((actual > 0).sum()),
        "exact_match": True,
        "maximum_absolute_error": float((actual - expected).abs().max()),
        "drop_predicate_match": True,
        "isaaclab_weight": manager.get_term_cfg("success").weight,
        "isaaclab_step_dt": dt,
        "manager_reset_verified": True,
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "reference_source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                base,
                manager_source,
                predicate_source,
                reward_cfg_source,
                termination_source,
                task_cfg_source,
            )
        },
    }
    return report


def main() -> None:
    """Compare one or more recorded traces and write a portable result report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [compare_trace(path, args.isaaclab_root) for path in args.trace]
    report = {
        "scope": "Pinned-source numerical comparison; no IsaacLab physics rollout.",
        "traces": results,
        "scalar_rewards_compared": sum(
            item["scalar_rewards_compared"] for item in results
        ),
        "maximum_absolute_error": max(
            item["maximum_absolute_error"] for item in results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
