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

"""Evaluate a RoboLab task list with the official shell entry and explicit resume."""

import argparse
import csv
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "evaluations/robolab/pi05_droid_jointpos.yaml"


def write_json(path: Path, value: dict) -> None:
    """Atomically replace a JSON artifact so interruption cannot truncate it."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_metrics(directory: Path, expected_episodes: int) -> dict:
    """Read a completed runner evaluation and validate its episode count and scores."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = list((directory / "tensorboard").glob("events.out.tfevents.*"))
    if len(files) != 1:
        raise ValueError(f"Expected one TensorBoard event file in {directory}, got {files}")
    events = EventAccumulator(str(files[0]))
    events.Reload()
    metrics = {}
    for key in ("num_trajectories", "success_once", "score"):
        tag = f"eval/{key}"
        if tag not in events.Tags()["scalars"]:
            raise ValueError(f"Expected {tag} in {directory}")
        metrics[key] = events.Scalars(tag)[-1].value
    if metrics["num_trajectories"] != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, got {metrics}")
    for key in ("success_once", "score"):
        if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
            raise ValueError(f"Expected {key} in [0, 1], got {metrics[key]}")
    successes = round(metrics["success_once"] * expected_episodes)
    if not math.isclose(successes, metrics["success_once"] * expected_episodes, abs_tol=1e-4):
        raise ValueError(f"Expected an integral success count, got {metrics}")
    return dict(episodes=expected_episodes, successes=successes,
                success_rate=metrics["success_once"], mean_score=metrics["score"])


def stop_evaluation(process: subprocess.Popen) -> None:
    """Stop the launched shell and its children without stopping a shared Ray cluster."""
    import psutil

    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=10)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    psutil.wait_procs(alive, timeout=5)


def load_config(path: Path) -> dict:
    """Compose an evaluation YAML using RLinf's embodied configuration search path."""
    path = path.resolve()
    os.environ["EMBODIED_PATH"] = str(REPO_ROOT / "examples/embodiment")
    with initialize_config_dir(version_base="1.1", config_dir=str(path.parent)):
        return OmegaConf.to_container(compose(config_name=path.stem), resolve=True)


def load_tasks(path: Path) -> list[dict]:
    """Read unique task identifiers and positive control-step limits from a TSV."""
    with path.open() as stream:
        tasks = [
            dict(task=row["task"], max_steps=int(row["max_steps"]))
            for row in csv.DictReader(stream, delimiter="\t")
        ]
    if not tasks or len({task["task"] for task in tasks}) != len(tasks):
        raise ValueError("Expected a nonempty TSV with unique task names")
    if any(not task["task"].isidentifier() or task["max_steps"] <= 0 for task in tasks):
        raise ValueError("Expected identifier task names and positive max_steps in the TSV")
    return tasks


class Benchmark:
    """Run tasks sequentially and persist completion only after validating runner metrics."""

    def __init__(self, output: Path, manifest: dict, interval: float) -> None:
        self.output = output
        self.manifest = manifest
        self.config = manifest["config"]
        self.tasks = manifest["tasks"]
        self.interval = interval
        self.logger = get_logger()
        self.current = None
        self.results = {}
        for task in self.tasks:
            path = output / task["task"] / "result.json"
            if path.exists():
                self.results[task["task"]] = json.loads(path.read_text())
        OmegaConf.save(OmegaConf.create(self.config), output / "config.yaml")

    def progress(self, status: str, error: str | None = None) -> None:
        """Atomically record and display task progress and episode-weighted metrics."""
        episodes = sum(result["episodes"] for result in self.results.values())
        successes = sum(result["successes"] for result in self.results.values())
        score_sum = sum(result["mean_score"] * result["episodes"] for result in self.results.values())
        summary = dict(
            status=status, updated_at=time.time(),
            wall_elapsed_seconds=time.time() - self.manifest["started_at"],
            completed_tasks=len(self.results), total_tasks=len(self.tasks),
            current_task=self.current, error=error,
            episodes=episodes, successes=successes,
            success_rate=successes / episodes if episodes else None,
            mean_score=score_sum / episodes if episodes else None,
        )
        write_json(self.output / "progress.json", summary)
        self.logger.info(
            "[%s/%s] %s; task=%s, attempt=%s; completed=%s; episodes=%s; elapsed=%.1f min; success=%s, score=%s",
            self.current["index"] if self.current else len(self.results), len(self.tasks),
            status, self.current["task"] if self.current else "-",
            self.current["attempt"] if self.current else "-", len(self.results),
            episodes, summary["wall_elapsed_seconds"] / 60,
            summary["success_rate"], summary["mean_score"],
        )

    def evaluate(self, task: dict, directory: Path) -> dict:
        """Invoke run_eval.sh with this task's limit and artifact directories."""
        chunk = int(self.config["rollout"]["model"]["num_action_chunks"])
        rollout_steps = math.ceil(task["max_steps"] / chunk) * chunk
        command = [
            "bash", str(REPO_ROOT / "evaluations/run_eval.sh"), "robolab", "pi05_droid_jointpos",
            f"env.eval.init_params.task={task['task']}",
            f"env.eval.max_episode_steps={task['max_steps']}",
            f"env.eval.max_steps_per_rollout_epoch={rollout_steps}",
            f"env.eval.init_params.output_dir={json.dumps(str(directory / 'native'))}",
            f"runner.logger.log_path={json.dumps(str(directory))}",
            f"env.eval.video_cfg.video_base_dir={json.dumps(str(directory / 'video'))}",
            f"hydra.run.dir={json.dumps(str(REPO_ROOT))}", "hydra.output_subdir=null",
            "hydra/job_logging=stdout",
            "--config-path", str(self.output), "--config-name", "config",
        ]
        write_json(directory / "command.json", dict(argv=command))
        child_env = os.environ.copy()
        child_env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + child_env.get("PATH", "")
        with (directory / "console.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=child_env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                self.current["pid"] = process.pid
                self.progress("running")
                while True:
                    try:
                        returncode = process.wait(timeout=self.interval)
                        break
                    except subprocess.TimeoutExpired:
                        self.progress("running")
            finally:
                stop_evaluation(process)
        if returncode:
            raise RuntimeError(f"Evaluation exited with code {returncode}; see {directory / 'console.log'}")
        env = self.config["env"]["eval"]
        return read_metrics(directory, env["total_num_envs"] * env["rollout_epoch"])

    def run(self) -> int:
        """Skip completed tasks and stop on failure or interruption for explicit resume."""
        self.progress("running")
        for index, task in enumerate(self.tasks, start=1):
            name = task["task"]
            if name in self.results:
                self.logger.info("[%s/%s] Skipping completed task %s", index, len(self.tasks), name)
                continue
            task_dir = self.output / name
            task_dir.mkdir(exist_ok=True)
            attempts = [int(path.name.split("-")[1]) for path in task_dir.glob("attempt-*")]
            attempt = max(attempts, default=0) + 1
            directory = task_dir / f"attempt-{attempt:03d}"
            directory.mkdir()
            started = time.time()
            self.current = dict(task=name, index=index, attempt=attempt,
                                started_at=started, log=str(directory / "console.log"))
            try:
                self.progress("starting")
                result = self.evaluate(task, directory)
                result.update(task=name, max_episode_steps=task["max_steps"],
                              attempt=attempt, elapsed_seconds=time.time() - started)
                write_json(task_dir / "result.json", result)
                self.results[name] = result
                self.progress("task_completed")
            except KeyboardInterrupt:
                write_json(directory / "status.json", dict(status="interrupted"))
                self.progress("interrupted")
                return 130
            except Exception as exc:
                write_json(directory / "status.json", dict(status="failed", error=str(exc)))
                self.progress("failed", str(exc))
                self.logger.error("%s. Continue with --resume-dir %s after fixing the error.", exc, self.output)
                return 1
        self.current = None
        self.progress("completed")
        return 0


def main() -> None:
    """Start a new benchmark or resume its saved configuration and task list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Evaluation YAML; defaults to pi05_droid_jointpos.yaml for new runs.")
    parser.add_argument("--tasks", type=Path, help="Task TSV; defaults to ./robolab_tasks.tsv for new runs.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path, help="Empty or new directory for a new experiment.")
    mode.add_argument("--resume-dir", type=Path, help="Previous output directory; reuse its saved configuration and tasks.")
    parser.add_argument("--progress-interval", type=float, default=30, help="Seconds between progress updates.")
    args = parser.parse_args()
    if not math.isfinite(args.progress_interval) or args.progress_interval <= 0:
        parser.error("Expected a finite --progress-interval > 0")
    output = (args.resume_dir or args.output_dir).resolve()
    if args.resume_dir and not (output / "manifest.json").is_file():
        parser.error(f"Expected manifest.json in --resume-dir {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Expected exclusive access: another launcher owns {output}")
        if args.resume_dir:
            manifest = json.loads((output / "manifest.json").read_text())
            if args.config and load_config(args.config) != manifest["config"]:
                parser.error("Expected the saved configuration to match --config; use a new --output-dir")
            if args.tasks and load_tasks(args.tasks) != manifest["tasks"]:
                parser.error("Expected the saved task list to match --tasks; use a new --output-dir")
        else:
            if any(path.name != ".lock" for path in output.iterdir()):
                parser.error("Expected an empty --output-dir; use --resume-dir to continue an existing experiment")
            manifest = dict(
                config=load_config(args.config or DEFAULT_CONFIG),
                tasks=load_tasks(args.tasks or Path("robolab_tasks.tsv")), started_at=time.time(),
            )
        config = manifest["config"]
        env = config["env"]["eval"]
        if env["env_type"] != "robolab" or not env["init_params"]["enable_subtask"]:
            parser.error("Expected env.eval.env_type=robolab and enable_subtask=true")
        if env["total_num_envs"] <= 0 or env["rollout_epoch"] <= 0:
            parser.error("Expected positive total_num_envs and rollout_epoch")
        if config["rollout"]["model"]["num_action_chunks"] <= 0:
            parser.error("Expected rollout.model.num_action_chunks > 0")
        if "tensorboard" not in config["runner"]["logger"]["logger_backends"]:
            parser.error("Expected tensorboard in runner.logger.logger_backends for result collection")
        model_path = Path(config["rollout"]["model"]["model_path"])
        if not (REPO_ROOT / model_path).is_dir():
            parser.error(f"Expected a local checkpoint directory, found {model_path}")
        if not args.resume_dir:
            write_json(output / "manifest.json", manifest)

        def interrupt(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        raise SystemExit(Benchmark(output, manifest, args.progress_interval).run())


if __name__ == "__main__":
    main()
