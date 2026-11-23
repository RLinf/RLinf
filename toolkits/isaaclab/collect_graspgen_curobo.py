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

"""Collect validated Panda demonstrations with GraspGen, cuRobo, and Isaac Lab."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

COLLECTOR_REVISION = "9f1feae9aa0a66a31053b4e237b29cf388a45637"
OBJECTS = ("block55", "slender95", "cylinder36h52", "disk56h24")


def verify_runtime(root: Path, revision: str = COLLECTOR_REVISION) -> None:
    """Require the validated collector revision and unchanged tracked files."""
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != revision:
        raise ValueError(f"Collector must be at {revision}; found {head} in {root}")
    changed = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if changed:
        raise ValueError(f"Collector has modified tracked files: {changed}")


class CollectionWorkflow:
    """Own collection configuration and the external collector's lifecycle.

    The external runtime owns simulation/planning processes and fixed-attempt
    recovery. This driver runs each stage synchronously and propagates failures.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.runtime.root).expanduser().resolve()
        self.output = Path(cfg.output).expanduser().resolve()
        self.candidates = Path(cfg.candidates.root).expanduser().resolve()
        if not cfg.collection.objects or any(
            name not in OBJECTS for name in cfg.collection.objects
        ):
            raise ValueError(f"collection.objects must select from {OBJECTS}")
        if len(set(cfg.collection.objects)) != len(cfg.collection.objects):
            raise ValueError("collection.objects must not contain duplicates")
        if cfg.collection.episodes_per_object < 1 or cfg.collection.workers < 1:
            raise ValueError("Episode and worker counts must be positive")
        if cfg.candidates.num_grasps not in (200, 500, 1000):
            raise ValueError("candidates.num_grasps must be 200, 500, or 1000")
        self.objects = ",".join(cfg.collection.objects)

    def environment(self) -> dict[str, str]:
        """Keep incompatible simulator, planner, and model Python stacks apart."""
        env = os.environ.copy()
        env.update(
            SIM_PYTHON=str(self.cfg.runtime.sim_python),
            PLANNER_PYTHON=str(self.cfg.runtime.planner_python),
            GRASPGEN_PYTHON=str(self.cfg.runtime.graspgen_python),
            GRASPGEN_CONFIG=str(self.cfg.runtime.graspgen_config),
            PYTHONDONTWRITEBYTECODE="1",
        )
        return env

    def run_runtime(self, arguments: list[str]) -> None:
        """Run the runtime supervisor; its children inherit the same terminal."""
        subprocess.run(
            [sys.executable, "-B", str(self.root / "cli.py"), *arguments],
            env=self.environment(),
            check=True,
        )

    def collect(self, resume: bool = False) -> None:
        """Record fixed attempts, including failures, with raw data validation."""
        self.output.mkdir(parents=True, exist_ok=True)
        config_path = self.output / "collection.yaml"
        saved_config = OmegaConf.to_yaml(self.cfg, resolve=True)
        if resume:
            if not config_path.is_file() or config_path.read_text() != saved_config:
                raise ValueError(
                    "Resume requires the original collection configuration"
                )
        else:
            if config_path.exists() or (self.output / "raw").exists():
                raise FileExistsError(f"Use a new output or --resume: {self.output}")
            config_path.write_text(saved_config)
        arguments = [
            "collect",
            "--output",
            str(self.output / "raw"),
            "--candidate-root",
            str(self.candidates),
            "--objects",
            self.objects,
            "--gpus",
            str(self.cfg.collection.gpus),
            "--workers",
            str(self.cfg.collection.workers),
            "--episodes-per-object",
            str(self.cfg.collection.episodes_per_object),
            "--seed",
            str(self.cfg.collection.seed),
            "--execution-profile",
            str(self.cfg.collection.execution_profile),
            "--arm-speed-profile",
            str(self.cfg.collection.arm_speed_profile),
            "--obstacle",
            str(self.cfg.collection.obstacle),
            "--startup-timeout",
            str(self.cfg.collection.startup_timeout),
            "--episode-timeout",
            str(self.cfg.collection.episode_timeout),
        ]
        if resume:
            arguments.append("--resume")
        self.run_runtime(arguments)

    def export(self) -> None:
        """Select strict successes and export them through RLinf's LeRobot writer."""
        from toolkits.isaaclab.export_graspgen_curobo import export_selection

        selection = self.output / "SELECTION.json"
        if selection.exists():
            raise FileExistsError(f"Selection already exists; preserve it: {selection}")
        subprocess.run(
            [
                str(self.cfg.runtime.planner_python),
                "-B",
                str(self.root / "tools/lerobot_export/select_sources.py"),
                "--run",
                str(self.output / "raw"),
                "--runtime",
                str(self.root / "runtime"),
                "--output",
                str(selection),
            ],
            env=self.environment(),
            check=True,
        )
        export_selection(selection, self.output / "export", self.cfg.dataset.repo_id)

    def run(self, stage: str, resume: bool = False) -> None:
        """Run a preparation, collection, export, or complete pipeline stage."""
        verify_runtime(self.root)
        if resume and stage != "collect":
            raise ValueError("--resume applies to --stage collect")
        if stage == "check":
            self.run_runtime(
                [
                    "check",
                    "--objects",
                    self.objects,
                    "--gpus",
                    str(self.cfg.collection.gpus),
                ]
            )
        if stage in ("prepare", "pipeline"):
            self.run_runtime(
                [
                    "prepare",
                    "--objects",
                    self.objects,
                    "--output-root",
                    str(self.candidates),
                    "--num-grasps",
                    str(self.cfg.candidates.num_grasps),
                    "--gpu",
                    str(self.cfg.collection.gpus).split(",")[0],
                ]
            )
        if stage in ("collect", "pipeline"):
            self.collect(resume)
        if stage in ("export", "pipeline"):
            self.export()


def main() -> None:
    """Load a YAML collection configuration and execute the selected stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("check", "prepare", "collect", "export", "pipeline"),
        default="pipeline",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    CollectionWorkflow(cfg).run(args.stage, args.resume)


if __name__ == "__main__":
    main()
