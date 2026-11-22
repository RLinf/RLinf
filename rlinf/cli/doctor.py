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

"""Environment diagnostics behind ``rlinf doctor``."""

from __future__ import annotations

import ctypes.util
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TextIO

from .catalog import discover_recipes, find_repo_root


class CheckStatus(str, Enum):
    """Outcome level for a single diagnostic check."""

    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    """Single diagnostic line rendered by ``rlinf doctor``."""

    name: str
    status: CheckStatus
    detail: str


def _check_python() -> CheckResult:
    venv = os.environ.get("VIRTUAL_ENV")
    venv_note = f", venv={venv}" if venv else ""
    return CheckResult(
        name="Python",
        status=CheckStatus.OK,
        detail=f"{platform.python_version()} ({sys.executable}{venv_note})",
    )


def _check_repo(repo_root: Path | None) -> CheckResult:
    try:
        root = find_repo_root(repo_root)
    except FileNotFoundError as exc:
        return CheckResult(name="Repository", status=CheckStatus.FAIL, detail=str(exc))
    recipes = discover_recipes(root)
    return CheckResult(
        name="Repository",
        status=CheckStatus.OK,
        detail=f"{root} ({len(recipes)} recipes discovered)",
    )


def _visible_device_env_summary() -> str:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "MTHREADS_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
    )
    parts = [f"{k}={os.environ[k]}" for k in keys if os.environ.get(k)]
    return f" ({', '.join(parts)})" if parts else ""


def _check_accelerators() -> CheckResult:
    try:
        import torch
    except ImportError as exc:
        return CheckResult(
            name="Accelerators",
            status=CheckStatus.FAIL,
            detail=f"PyTorch not importable: {exc}",
        )

    torch_ver = torch.__version__
    vis = _visible_device_env_summary()
    try:
        from rlinf.scheduler.hardware.accelerators import Accelerator

        hw_resource = Accelerator.enumerate()
    except (ImportError, RuntimeError, OSError) as exc:
        return CheckResult(
            name="Accelerators",
            status=CheckStatus.WARN,
            detail=f"torch {torch_ver}, accelerator probe failed: {exc}{vis}",
        )

    if hw_resource is None or not hw_resource.infos:
        return CheckResult(
            name="Accelerators",
            status=CheckStatus.WARN,
            detail=f"torch {torch_ver}, no accelerators detected (CPU-only){vis}",
        )

    model = hw_resource.infos[0].model
    count = len(hw_resource.infos)
    return CheckResult(
        name="Accelerators",
        status=CheckStatus.OK,
        detail=f"torch {torch_ver}, {count}x {model}{vis}",
    )


def _check_rendering() -> CheckResult:
    found = [
        label
        for label, lib in (
            ("EGL", "EGL"),
            ("Vulkan", "vulkan"),
            ("OSMesa", "OSMesa"),
        )
        if ctypes.util.find_library(lib) is not None
    ]
    mujoco_gl = os.environ.get("MUJOCO_GL", "unset (defaults to egl)")
    if found:
        return CheckResult(
            name="Rendering",
            status=CheckStatus.OK,
            detail=f"libraries={', '.join(found)}; MUJOCO_GL={mujoco_gl}",
        )
    return CheckResult(
        name="Rendering",
        status=CheckStatus.WARN,
        detail=(
            f"none of EGL/Vulkan/OSMesa found via ldconfig; MUJOCO_GL={mujoco_gl} "
            "(needed for embodied simulators)"
        ),
    )


def _check_ray() -> CheckResult:
    try:
        import ray
    except ImportError as exc:
        return CheckResult(
            name="Ray",
            status=CheckStatus.FAIL,
            detail=f"Ray not importable: {exc}",
        )

    ray_bin = shutil.which("ray")
    cluster_note = "no active cluster"
    if ray_bin is not None:
        try:
            proc = subprocess.run(
                [ray_bin, "status"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0:
                cluster_note = "active cluster detected"
        except subprocess.TimeoutExpired:
            cluster_note = "status timed out"

    node_rank = os.environ.get("RLINF_NODE_RANK")
    net_dev = os.environ.get("RLINF_COMM_NET_DEVICES")
    extra_parts = [cluster_note]
    if node_rank is not None:
        extra_parts.append(f"RLINF_NODE_RANK={node_rank}")
    if net_dev:
        extra_parts.append(f"RLINF_COMM_NET_DEVICES={net_dev}")

    return CheckResult(
        name="Ray",
        status=CheckStatus.OK,
        detail=f"{ray.__version__} ({', '.join(extra_parts)})",
    )


def _check_huggingface(*, check_network: bool) -> CheckResult:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    try:
        import huggingface_hub

        version = huggingface_hub.__version__
    except ImportError:
        return CheckResult(
            name="Hugging Face",
            status=CheckStatus.WARN,
            detail=f"huggingface_hub not installed; endpoint={endpoint}",
        )

    if not check_network:
        return CheckResult(
            name="Hugging Face",
            status=CheckStatus.OK,
            detail=(
                f"huggingface_hub {version}, endpoint={endpoint} "
                "(pass --network to test reachability)"
            ),
        )

    req = urllib.request.Request(endpoint, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return CheckResult(
                name="Hugging Face",
                status=CheckStatus.OK,
                detail=f"huggingface_hub {version}, {endpoint} reachable (HTTP {resp.status})",
            )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return CheckResult(
            name="Hugging Face",
            status=CheckStatus.WARN,
            detail=(
                f"huggingface_hub {version}, {endpoint} unreachable ({exc}); "
                "set HF_ENDPOINT or use local model paths"
            ),
        )


def run_doctor(
    *,
    repo_root: Path | None = None,
    check_network: bool = False,
) -> list[CheckResult]:
    """Run all local environment diagnostics and return structured results.

    Args:
        repo_root: Optional explicit repository root.
        check_network: When ``True``, issue a 5-second HEAD request to ``HF_ENDPOINT``.

    Returns:
        Ordered list of :class:`CheckResult` objects.
    """
    return [
        _check_python(),
        _check_repo(repo_root),
        _check_accelerators(),
        _check_rendering(),
        _check_ray(),
        _check_huggingface(check_network=check_network),
    ]


def format_doctor_report(results: list[CheckResult]) -> str:
    """Format diagnostic checks into an aligned plain-text report."""
    name_width = max((len(r.name) for r in results), default=12)
    lines = [
        f"[{r.status.value:4s}] {r.name:<{name_width}}  {r.detail}" for r in results
    ]
    return "\n".join(lines) + "\n"


def render_doctor(
    *,
    repo_root: Path | None = None,
    check_network: bool = False,
    stream: TextIO | None = None,
) -> int:
    """Execute ``rlinf doctor``, write the report to ``stream``, and return exit code."""
    out = sys.stdout if stream is None else stream
    results = run_doctor(repo_root=repo_root, check_network=check_network)
    out.write(format_doctor_report(results))
    out.flush()
    return 1 if any(r.status == CheckStatus.FAIL for r in results) else 0
