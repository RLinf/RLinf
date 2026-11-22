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

"""Command assembly, environment preparation, and execution for ``rlinf``."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from .catalog import RecipeMode, RecipeSpec

_KEY_ENV_KEYS: tuple[str, ...] = (
    "REPO_PATH",
    "EMBODIED_PATH",
    "EVALUATIONS_PATH",
    "DIFFUSION_PATH",
    "REWARD_PATH",
    "VLM_PATH",
    "PYTHONPATH",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "ROBOT_PLATFORM",
    "LIBERO_TYPE",
    "RLINF_NODE_RANK",
    "RLINF_COMM_NET_DEVICES",
    "RLINF_ROBOT_MOCKS",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "TOKENIZERS_PARALLELISM",
    "PYTORCH_CUDA_ALLOC_CONF",
    "NUMBA_DISABLE_JIT",
)


@dataclass(frozen=True)
class LaunchPlan:
    """Fully resolved subprocess invocation for an RLinf run."""

    recipe: RecipeSpec
    command: list[str]
    env: dict[str, str]
    log_dir: Path | None
    log_file: Path | None


def _prepend_pythonpath(env: dict[str, str], paths: Sequence[str]) -> None:
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    ordered: list[str] = []
    for item in [*paths, *existing]:
        if item and item not in ordered:
            ordered.append(item)
    env["PYTHONPATH"] = os.pathsep.join(ordered)


def _apply_sim_env(env: dict[str, str], repo_root: Path) -> None:
    """Apply the same non-destructive defaults as ``examples/embodiment/sim_env.sh``."""
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "1")
    env.setdefault("OMNIGIBSON_DEBUG", "0")
    env.setdefault("OMNIGIBSON_HEADLESS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    og_data = env.get("OMNIGIBSON_DATA_PATH")
    if og_data:
        env.setdefault("OMNIGIBSON_DATASET_PATH", f"{og_data}/behavior-1k-assets/")
        env.setdefault("OMNIGIBSON_KEY_PATH", f"{og_data}/omnigibson.key")
        env.setdefault("OMNIGIBSON_ASSET_PATH", f"{og_data}/omnigibson-robot-assets/")

    isaac_path = env.get("ISAAC_PATH")
    if isaac_path:
        env.setdefault("EXP_PATH", f"{isaac_path}/apps")
        env.setdefault("CARB_APP_PATH", f"{isaac_path}/kit")

    extra_pythonpaths: list[str] = []
    if env.get("DREAMZERO_PATH"):
        extra_pythonpaths.append(env["DREAMZERO_PATH"])
    if env.get("ROBOTWIN_PATH"):
        extra_pythonpaths.extend([str(repo_root), env["ROBOTWIN_PATH"]])
    if extra_pythonpaths:
        _prepend_pythonpath(env, extra_pythonpaths)


def _build_env(
    recipe: RecipeSpec,
    repo_root: Path,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    env = dict(base_env)
    env["REPO_PATH"] = str(repo_root)
    _prepend_pythonpath(env, [str(repo_root)])

    profile = recipe.env_profile
    if profile == "embodied_sim":
        env.setdefault("EMBODIED_PATH", str(repo_root / "examples/embodiment"))
        env.setdefault("EVALUATIONS_PATH", str(repo_root / "evaluations"))
        _apply_sim_env(env, repo_root)
        if recipe.subgroup == "robocasa365":
            env.setdefault("NUMBA_DISABLE_JIT", "1")
        if recipe.subgroup == "libero" or "libero" in recipe.name.lower():
            env.setdefault("ROBOT_PLATFORM", "LIBERO")
            libero_type = env.setdefault("LIBERO_TYPE", "standard")
            if libero_type == "pro":
                env.setdefault("LIBERO_PERTURBATION", "all")
            elif libero_type == "plus":
                env.setdefault("LIBERO_SUFFIX", "all")
        else:
            env.setdefault("ROBOT_PLATFORM", "LIBERO")
    elif profile == "embodied_real":
        env.setdefault("EMBODIED_PATH", str(repo_root / "examples/embodiment"))
        env.setdefault("EVALUATIONS_PATH", str(repo_root / "evaluations"))
    elif profile == "diffusion":
        env.setdefault("DIFFUSION_PATH", str(repo_root / "examples/diffusion"))
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    elif profile == "sft_vla":
        env.setdefault("EMBODIED_PATH", str(repo_root / "examples/sft"))
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        if env.get("DREAMZERO_PATH"):
            _prepend_pythonpath(env, [env["DREAMZERO_PATH"]])
    elif profile == "sft_vlm":
        env.setdefault("VLM_PATH", str(repo_root / "examples/sft"))
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    elif profile == "agent_reason":
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("RAY_DEDUP_LOGS", "0")
        extra_paths = [str(repo_root / "examples")]
        megatron_path = Path("/opt/Megatron-LM")
        if megatron_path.is_dir():
            extra_paths.append(str(megatron_path))
        _prepend_pythonpath(env, extra_paths)
    elif profile == "reward":
        env.setdefault("REWARD_PATH", str(repo_root / "examples/reward"))
        env.setdefault("EMBODIED_PATH", str(repo_root / "examples/reward"))

    return env


def _has_override_key(overrides: Sequence[str], key: str) -> bool:
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    return any(item.startswith(prefixes) for item in overrides)


def build_launch_plan(
    recipe: RecipeSpec,
    repo_root: Path,
    overrides: Sequence[str] = (),
    *,
    log_dir: Path | None = None,
    base_env: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> LaunchPlan:
    """Build the resolved subprocess command and environment for ``recipe``.

    Args:
        recipe: Resolved :class:`RecipeSpec`.
        repo_root: RLinf repository root.
        overrides: Raw Hydra overrides forwarded verbatim.
        log_dir: Optional explicit log directory (overrides default timestamped dir).
        base_env: Base environment mapping (defaults to ``os.environ``).
        python_executable: Python interpreter path (defaults to ``sys.executable``).

    Returns:
        A :class:`LaunchPlan` ready for execution or dry-run inspection.
    """
    env = _build_env(recipe, repo_root, os.environ if base_env is None else base_env)
    py = python_executable or sys.executable

    command: list[str] = [
        py,
        str(recipe.entry_script),
        "--config-path",
        str(recipe.config_dir),
        "--config-name",
        recipe.name,
    ]

    for path in recipe.searchpath:
        command.extend(["--config-dir", str(path)])

    resolved_log_dir: Path | None = None
    resolved_log_file: Path | None = None
    if log_dir is not None:
        resolved_log_dir = log_dir.expanduser().resolve()
    elif recipe.inject_log_path and not _has_override_key(
        overrides, "runner.logger.log_path"
    ):
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        resolved_log_dir = repo_root / "logs" / f"{timestamp}-{recipe.name}"

    if resolved_log_dir is not None:
        log_stem = (
            "eval_embodiment.log"
            if recipe.mode == RecipeMode.EVAL
            else f"run_{recipe.family.value}.log"
        )
        resolved_log_file = resolved_log_dir / log_stem
        if not _has_override_key(overrides, "runner.logger.log_path"):
            command.append(f"runner.logger.log_path={resolved_log_dir}")

    command.extend(overrides)
    return LaunchPlan(
        recipe=recipe,
        command=command,
        env=env,
        log_dir=resolved_log_dir,
        log_file=resolved_log_file,
    )


def format_dry_run(plan: LaunchPlan) -> str:
    """Format a human-readable shell preview of ``plan``."""
    lines = [
        f"# recipe: {plan.recipe.family.value}/{plan.recipe.mode.value} ({plan.recipe.qualified_name})",
        f"# config: {plan.recipe.config_file}",
    ]
    for key in _KEY_ENV_KEYS:
        if key in plan.env:
            lines.append(f"export {key}={shlex.quote(plan.env[key])}")
    lines.append(shlex.join(plan.command))
    return "\n".join(lines) + "\n"


def execute_launch_plan(
    plan: LaunchPlan,
    *,
    dry_run: bool = False,
    stream: TextIO | None = None,
) -> int:
    """Execute ``plan`` or print its resolved command when ``dry_run`` is set.

    Forwards ``SIGINT`` and ``SIGTERM`` to the child process so Ray workers and
    accelerator resources run their shutdown handlers before the CLI exits.

    Args:
        plan: Resolved :class:`LaunchPlan`.
        dry_run: When ``True``, print the resolved command and exit ``0``.
        stream: Output stream (defaults to ``sys.stdout``).

    Returns:
        Subprocess exit code (``0`` on dry run).
    """
    out = sys.stdout if stream is None else stream
    if dry_run:
        out.write(format_dry_run(plan))
        out.flush()
        return 0

    log_handle = None
    if plan.log_dir is not None and plan.log_file is not None:
        plan.log_dir.mkdir(parents=True, exist_ok=True)
        log_handle = plan.log_file.open("a", encoding="utf-8")
        log_handle.write(shlex.join(plan.command) + "\n")
        log_handle.flush()

    proc = subprocess.Popen(
        plan.command,
        env=plan.env,
        stdout=subprocess.PIPE if log_handle is not None else None,
        stderr=subprocess.STDOUT if log_handle is not None else None,
        text=True,
        bufsize=1,
    )

    prev_sigterm = None
    if threading.current_thread() is threading.main_thread():

        def _forward_sigterm(_signum, _frame) -> None:
            if proc.poll() is None:
                proc.terminate()

        prev_sigterm = signal.signal(signal.SIGTERM, _forward_sigterm)

    try:
        if log_handle is not None and proc.stdout is not None:
            for line in proc.stdout:
                out.write(line)
                out.flush()
                log_handle.write(line)
                log_handle.flush()
        return proc.wait()
    except KeyboardInterrupt:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        return proc.wait()
    finally:
        if prev_sigterm is not None:
            signal.signal(signal.SIGTERM, prev_sigterm)
        if log_handle is not None:
            log_handle.close()
