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

import importlib.util
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from .catalog import RecipeMode, RecipeSpec, extract_required_env_vars

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
    missing_searchpath_env: tuple[str, ...] = ()
    missing_body_env: tuple[str, ...] = ()


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


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _prepend_path_env(env: dict[str, str], entry: str) -> None:
    existing = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    if existing and existing[0] == entry:
        return
    ordered = [entry, *(p for p in existing if p != entry)]
    env["PATH"] = os.pathsep.join(ordered)


def resolve_venv_python(
    repo_root: Path,
    *,
    venv: Path | None = None,
    base_env: Mapping[str, str] | None = None,
) -> tuple[str, Path | None]:
    """Resolve the target Python executable and optional venv directory.

    Resolution order:
    1. Explicit ``--venv`` path (fails with :class:`FileNotFoundError` if missing).
    2. Active ``VIRTUAL_ENV`` when it points to a valid Python interpreter.
    3. Current ``sys.executable`` when it already has RLinf's runtime stack.
    4. Default ``<repo_root>/.venv`` when running from a bare/global CLI launcher.
    5. Current ``sys.executable`` as final fallback.

    Args:
        repo_root: RLinf repository root.
        venv: Optional explicit virtual environment directory.
        base_env: Environment mapping to inspect for ``VIRTUAL_ENV``.

    Returns:
        Tuple of ``(python_executable, resolved_venv_dir_or_none)``.

    Raises:
        FileNotFoundError: If an explicit ``venv`` does not contain a Python binary.
    """
    current_py_abs = os.path.abspath(sys.executable)

    if venv is not None:
        expanded = venv.expanduser()
        if expanded.is_absolute():
            candidate = expanded
        else:
            cwd_candidate = Path.cwd() / expanded
            repo_candidate = repo_root / expanded
            candidate = (
                cwd_candidate
                if _venv_python_path(cwd_candidate).is_file()
                or not _venv_python_path(repo_candidate).is_file()
                else repo_candidate
            )
        py_path = _venv_python_path(candidate)
        if not py_path.is_file():
            raise FileNotFoundError(
                f"Virtual environment not found at '{venv}' (expected '{py_path}'). "
                f"Run 'rlinf install ... --venv {venv}' first."
            )
        resolved_venv = candidate.resolve()
        py_abs = os.path.abspath(str(py_path))
        if current_py_abs == py_abs:
            return sys.executable, resolved_venv
        return py_abs, resolved_venv

    env_map = os.environ if base_env is None else base_env
    active_venv_raw = env_map.get("VIRTUAL_ENV")
    if active_venv_raw:
        active_venv = Path(active_venv_raw).expanduser()
        active_py = _venv_python_path(active_venv)
        if active_py.is_file():
            resolved_active = active_venv.resolve()
            active_py_abs = os.path.abspath(str(active_py))
            if current_py_abs == active_py_abs:
                return sys.executable, resolved_active
            return active_py_abs, resolved_active

    if importlib.util.find_spec("hydra") is not None:
        return sys.executable, None

    default_venv = repo_root / ".venv"
    default_py = _venv_python_path(default_venv)
    if default_py.is_file():
        return os.path.abspath(str(default_py)), default_venv.resolve()

    return sys.executable, None


def build_launch_plan(
    recipe: RecipeSpec,
    repo_root: Path,
    overrides: Sequence[str] = (),
    *,
    log_dir: Path | None = None,
    venv: Path | None = None,
    base_env: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> LaunchPlan:
    """Build the resolved subprocess command and environment for ``recipe``.

    Args:
        recipe: Resolved :class:`RecipeSpec`.
        repo_root: RLinf repository root.
        overrides: Raw Hydra overrides forwarded verbatim.
        log_dir: Optional explicit log directory (overrides default timestamped dir).
        venv: Optional virtual environment directory whose Python should launch the run.
        base_env: Base environment mapping (defaults to ``os.environ``).
        python_executable: Explicit Python interpreter path (takes precedence over ``venv``).

    Returns:
        A :class:`LaunchPlan` ready for execution or dry-run inspection.
    """
    env = _build_env(recipe, repo_root, os.environ if base_env is None else base_env)
    if python_executable is not None:
        py = python_executable
        resolved_venv: Path | None = None
    else:
        py, resolved_venv = resolve_venv_python(repo_root, venv=venv, base_env=env)
        if resolved_venv is not None:
            env["VIRTUAL_ENV"] = str(resolved_venv)
            _prepend_path_env(
                env, str(resolved_venv / ("Scripts" if os.name == "nt" else "bin"))
            )

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
    env_search_dirs = [
        recipe.config_dir,
        *recipe.searchpath,
        repo_root / "examples/embodiment/config",
    ]
    searchpath_vars, body_vars = extract_required_env_vars(
        recipe.config_file, search_dirs=env_search_dirs
    )
    missing_searchpath = tuple(v for v in searchpath_vars if not env.get(v))
    missing_body = tuple(v for v in body_vars if not env.get(v))
    return LaunchPlan(
        recipe=recipe,
        command=command,
        env=env,
        log_dir=resolved_log_dir,
        log_file=resolved_log_file,
        missing_searchpath_env=missing_searchpath,
        missing_body_env=missing_body,
    )


def format_dry_run(plan: LaunchPlan) -> str:
    """Format a human-readable shell preview of ``plan``."""
    lines = [
        f"# recipe: {plan.recipe.family.value}/{plan.recipe.mode.value} ({plan.recipe.qualified_name})",
        f"# config: {plan.recipe.config_file}",
    ]
    if plan.missing_searchpath_env:
        lines.append(
            f"# error: missing required searchpath env var(s): {', '.join(plan.missing_searchpath_env)}"
        )
    if plan.missing_body_env:
        lines.append(
            f"# warning: unset env var(s) referenced in YAML: {', '.join(plan.missing_body_env)}"
        )
    for key in _KEY_ENV_KEYS:
        if key in plan.env:
            lines.append(f"export {key}={shlex.quote(plan.env[key])}")
    lines.append(shlex.join(plan.command))
    return "\n".join(lines) + "\n"


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h{mins:02d}m{secs:02d}s"
    if mins:
        return f"{mins}m{secs:02d}s"
    return f"{seconds:.1f}s"


def check_launch_plan(
    plan: LaunchPlan,
    *,
    stream: TextIO | None = None,
    err_stream: TextIO | None = None,
) -> int:
    """Validate Hydra config composition and overrides via ``--cfg job`` without launching Ray.

    Args:
        plan: Resolved :class:`LaunchPlan`.
        stream: Output stream for confirmation messages.
        err_stream: Error stream for validation failures.

    Returns:
        ``0`` if Hydra composes the configuration cleanly, otherwise non-zero.
    """
    out = sys.stdout if stream is None else stream
    err = sys.stderr if err_stream is None else err_stream

    if plan.missing_searchpath_env:
        missing = ", ".join(plan.missing_searchpath_env)
        err.write(
            f"error: recipe '{plan.recipe.qualified_name}' requires environment "
            f"variable(s) in hydra.searchpath: {missing}.\n"
        )
        err.flush()
        return 2

    check_cmd = [*plan.command, "--cfg", "job"]
    proc = subprocess.run(
        check_cmd,
        env=plan.env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if proc.stderr:
            err.write(proc.stderr)
        elif proc.stdout:
            err.write(proc.stdout)
        err.flush()
        return proc.returncode

    out.write(
        f"[rlinf] Config OK: {plan.recipe.family.value}/{plan.recipe.mode.value} "
        f"({plan.recipe.qualified_name})\n"
    )
    if plan.missing_body_env:
        out.write(
            f"[rlinf] Note: YAML references unset env var(s): "
            f"{', '.join(plan.missing_body_env)}\n"
        )
    out.flush()
    return 0


def execute_launch_plan(
    plan: LaunchPlan,
    *,
    dry_run: bool = False,
    check_only: bool = False,
    stream: TextIO | None = None,
    err_stream: TextIO | None = None,
) -> int:
    """Execute ``plan``, validate config when ``check_only`` is set, or preview on ``dry_run``.

    Forwards ``SIGINT`` and ``SIGTERM`` to the child process so Ray workers and
    accelerator resources run their shutdown handlers before the CLI exits.

    Args:
        plan: Resolved :class:`LaunchPlan`.
        dry_run: When ``True``, print the resolved command and exit ``0``.
        check_only: When ``True``, run Hydra ``--cfg job`` validation without starting Ray.
        stream: Output stream (defaults to ``sys.stdout``).
        err_stream: Error stream (defaults to ``sys.stderr``).

    Returns:
        Subprocess exit code (``0`` on dry run or successful validation).
    """
    out = sys.stdout if stream is None else stream
    err = sys.stderr if err_stream is None else err_stream
    if dry_run:
        out.write(format_dry_run(plan))
        out.flush()
        return 0
    if check_only:
        return check_launch_plan(plan, stream=out, err_stream=err)

    if plan.missing_searchpath_env:
        missing = ", ".join(plan.missing_searchpath_env)
        err.write(
            f"error: recipe '{plan.recipe.qualified_name}' requires environment "
            f"variable(s) in hydra.searchpath: {missing}. Export them before running.\n"
        )
        err.flush()
        return 2

    if plan.missing_body_env:
        err.write(
            f"[rlinf] warning: YAML references unset env var(s): "
            f"{', '.join(plan.missing_body_env)}\n"
        )
        err.flush()

    log_note = f" | Logs: {plan.log_file}" if plan.log_file is not None else ""
    out.write(
        f"[rlinf] Launching {plan.recipe.family.value}/{plan.recipe.mode.value}: "
        f"{plan.recipe.qualified_name}{log_note}\n"
    )
    out.flush()

    log_handle = None
    if plan.log_dir is not None and plan.log_file is not None:
        plan.log_dir.mkdir(parents=True, exist_ok=True)
        log_handle = plan.log_file.open("a", encoding="utf-8")
        log_handle.write(shlex.join(plan.command) + "\n")
        log_handle.flush()

    start_time = time.monotonic()
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
        rc = proc.wait()
    except KeyboardInterrupt:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        rc = proc.wait()
    finally:
        if prev_sigterm is not None:
            signal.signal(signal.SIGTERM, prev_sigterm)
        if log_handle is not None:
            log_handle.close()

    elapsed = _format_duration(time.monotonic() - start_time)
    status_word = "Finished" if rc == 0 else f"Failed (exit {rc})"
    out.write(f"[rlinf] {status_word} in {elapsed}{log_note}\n")
    out.flush()
    return rc
