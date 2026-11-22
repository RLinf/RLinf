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

"""Thin ``rlinf install`` wrapper backed by ``requirements/install.sh``."""

from __future__ import annotations

import difflib
import json
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO

_ARRAY_RE = re.compile(
    r"^(SUPPORTED_(?:TARGETS|ENGINES|MODELS|ENVS|PLATFORMS))=\(([^)]*)\)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class InstallOptions:
    """Supported install targets and option values parsed from ``requirements/install.sh``."""

    targets: tuple[str, ...]
    models: tuple[str, ...]
    envs: tuple[str, ...]
    engines: tuple[str, ...]
    platforms: tuple[str, ...]


@dataclass(frozen=True)
class InstallPlan:
    """Resolved ``requirements/install.sh`` subprocess invocation."""

    target: str
    venv_dir: str
    script_path: Path
    repo_root: Path
    command: list[str]


def load_install_options(repo_root: Path) -> InstallOptions:
    """Parse ``SUPPORTED_*`` arrays directly from ``requirements/install.sh``.

    Args:
        repo_root: RLinf repository root containing ``requirements/install.sh``.

    Returns:
        :class:`InstallOptions` parsed from the installer script.

    Raises:
        FileNotFoundError: If ``requirements/install.sh`` does not exist.
    """
    script = repo_root / "requirements/install.sh"
    if not script.is_file():
        raise FileNotFoundError(
            f"Installer script not found at '{script}'. Expected an RLinf checkout."
        )

    text = script.read_text(encoding="utf-8")
    parsed: dict[str, tuple[str, ...]] = {}
    for match in _ARRAY_RE.finditer(text):
        key = match.group(1)
        items = tuple(re.findall(r'"([^"]+)"', match.group(2)))
        parsed[key] = items

    return InstallOptions(
        targets=parsed.get("SUPPORTED_TARGETS", ("embodied", "agentic", "docs")),
        models=parsed.get("SUPPORTED_MODELS", ()),
        envs=parsed.get("SUPPORTED_ENVS", ()),
        engines=parsed.get("SUPPORTED_ENGINES", ()),
        platforms=parsed.get("SUPPORTED_PLATFORMS", ("nvidia",)),
    )


def _validate_choice(flag: str, value: str | None, allowed: Sequence[str]) -> None:
    if value is None or not allowed or value in allowed:
        return
    close = difflib.get_close_matches(value, allowed, n=4, cutoff=0.45)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(
        f"Unsupported {flag} '{value}'.{hint} "
        "Run 'rlinf install --list' to see all supported values."
    )


def build_install_plan(
    repo_root: Path,
    *,
    target: str = "embodied",
    model: str | None = None,
    env: str | None = None,
    engine: str | None = None,
    venv: str | None = None,
    platform: str | None = None,
    python_version: str | None = None,
    torch_version: str | None = None,
    rocm_version: str | None = None,
    sglang_version: str | None = None,
    vllm_version: str | None = None,
    transformers_version: str | None = None,
    use_mirror: bool = False,
    paddle_gpu: bool = False,
    no_root: bool = False,
    no_flash_attn: bool = False,
    no_apex: bool = False,
    no_natten: bool = False,
    install_rlinf: bool = False,
    no_install_rlinf: bool = False,
    extra_args: Sequence[str] = (),
) -> InstallPlan:
    """Validate install options and build the ``requirements/install.sh`` command.

    Args:
        repo_root: RLinf repository root.
        target: Install target (``embodied``, ``agentic``, or ``docs``).
        model: Optional embodied model name.
        env: Optional embodied environment name.
        engine: Optional rollout engine name.
        venv: Optional virtual environment path (defaults to ``.venv``).
        platform: Optional hardware platform (``nvidia``, ``amd``, ``ascend``, etc.).
        python_version: Optional Python version override.
        torch_version: Optional PyTorch version override.
        rocm_version: Optional ROCm version override.
        sglang_version: Optional SGLang version override.
        vllm_version: Optional vLLM version override.
        transformers_version: Optional Transformers version override.
        use_mirror: Enable mainland China mirrors.
        paddle_gpu: Install CUDA PaddlePaddle for diffusion OCR rewards.
        no_root: Skip system dependency installation.
        no_flash_attn: Skip flash-attn installation.
        no_apex: Skip apex installation.
        no_natten: Skip natten installation.
        install_rlinf: Explicitly install RLinf editable.
        no_install_rlinf: Skip installing RLinf editable.
        extra_args: Raw extra flags forwarded verbatim after ``--``.

    Returns:
        Resolved :class:`InstallPlan`.

    Raises:
        ValueError: If any target, model, env, engine, or platform is invalid.
    """
    options = load_install_options(repo_root)
    _validate_choice("target", target, options.targets)
    _validate_choice("--model", model, options.models)
    _validate_choice("--env", env, options.envs)
    _validate_choice("--engine", engine, options.engines)
    _validate_choice("--platform", platform, options.platforms)

    if target != "embodied" and (model is not None or env is not None):
        raise ValueError(
            f"'--model' and '--env' apply only to target 'embodied', not '{target}'."
        )
    if install_rlinf and no_install_rlinf:
        raise ValueError("Cannot pass both '--install-rlinf' and '--no-install-rlinf'.")

    script_path = repo_root / "requirements/install.sh"
    cmd: list[str] = ["bash", str(script_path), target]

    kv_flags: tuple[tuple[str, str | None], ...] = (
        ("--model", model),
        ("--env", env),
        ("--engine", engine),
        ("--venv", venv),
        ("--platform", platform),
        ("--python", python_version),
        ("--torch", torch_version),
        ("--rocm", rocm_version),
        ("--sglang", sglang_version),
        ("--vllm", vllm_version),
        ("--transformers", transformers_version),
    )
    for flag, val in kv_flags:
        if val is not None:
            cmd.extend([flag, val])

    bool_flags: tuple[tuple[str, bool], ...] = (
        ("--use-mirror", use_mirror),
        ("--paddle-gpu", paddle_gpu),
        ("--no-root", no_root),
        ("--no-flash-attn", no_flash_attn),
        ("--no-apex", no_apex),
        ("--no-natten", no_natten),
        ("--install-rlinf", install_rlinf),
        ("--no-install-rlinf", no_install_rlinf),
    )
    for flag, enabled in bool_flags:
        if enabled:
            cmd.append(flag)

    cmd.extend(extra_args)
    return InstallPlan(
        target=target,
        venv_dir=venv or ".venv",
        script_path=script_path,
        repo_root=repo_root,
        command=cmd,
    )


def format_install_options(options: InstallOptions, *, as_json: bool = False) -> str:
    """Format supported install targets/models/envs/engines/platforms."""
    if as_json:
        payload = {
            "targets": list(options.targets),
            "models": list(options.models),
            "envs": list(options.envs),
            "engines": list(options.engines),
            "platforms": list(options.platforms),
        }
        return json.dumps(payload, indent=2) + "\n"

    lines = [
        f"Targets ({len(options.targets)}):    {', '.join(options.targets)}",
        f"Platforms ({len(options.platforms)}):  {', '.join(options.platforms)}",
        f"Engines ({len(options.engines)}):    {', '.join(options.engines)}",
        f"Models ({len(options.models)}):   {', '.join(options.models)}",
        f"Envs ({len(options.envs)}):     {', '.join(options.envs)}",
        "",
        "Examples:",
        "  rlinf install embodied --env maniskill_libero",
        "  rlinf install embodied --model openvla --env maniskill_libero",
        "  rlinf install agentic --engine sglang",
        "  rlinf install docs --venv .docs-venv",
    ]
    return "\n".join(lines) + "\n"


def execute_install_plan(
    plan: InstallPlan,
    *,
    dry_run: bool = False,
    stream: TextIO | None = None,
) -> int:
    """Execute ``plan`` in ``plan.repo_root`` or print the shell command on ``dry_run``.

    Forwards ``SIGINT`` and ``SIGTERM`` so ``requirements/install.sh`` runs its
    exit trap and restores ``pyproject.toml`` cleanly if interrupted.

    Args:
        plan: Resolved :class:`InstallPlan`.
        dry_run: When ``True``, print the command and return ``0``.
        stream: Output stream (defaults to ``sys.stdout``).

    Returns:
        Subprocess exit code (``0`` on dry run).
    """
    out = sys.stdout if stream is None else stream
    if dry_run:
        out.write(f"# cwd: {plan.repo_root}\n")
        out.write(shlex.join(plan.command) + "\n")
        out.flush()
        return 0

    out.write(f"[rlinf] Installing target '{plan.target}' into '{plan.venv_dir}'\n")
    out.flush()

    start_time = time.monotonic()
    proc = subprocess.Popen(plan.command, cwd=plan.repo_root)

    prev_sigterm = None
    if threading.current_thread() is threading.main_thread():

        def _forward_sigterm(_signum, _frame) -> None:
            if proc.poll() is None:
                proc.terminate()

        prev_sigterm = signal.signal(signal.SIGTERM, _forward_sigterm)

    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        rc = proc.wait()
    finally:
        if prev_sigterm is not None:
            signal.signal(signal.SIGTERM, prev_sigterm)

    elapsed = time.monotonic() - start_time
    if rc == 0:
        out.write(
            f"[rlinf] Install complete in {elapsed:.1f}s. Activate with:\n"
            f"  source {plan.venv_dir}/bin/activate\n"
            "  rlinf doctor\n"
        )
    else:
        out.write(f"[rlinf] Install failed (exit {rc}) after {elapsed:.1f}s.\n")
    out.flush()
    return rc
