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

"""Installer: steps of ``requirements/install.sh`` that need no GPU, network, or venv."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_amd_installer(checkout: Path, rocm: str, *target: str):
    """Run the CLI and capture its resolver inputs when uv first sees them."""
    (checkout / "requirements").mkdir()
    shutil.copy(_REPO_ROOT / "requirements" / "install.sh", checkout / "requirements")
    project = checkout / "pyproject.toml"
    shutil.copy(_REPO_ROOT / "pyproject.toml", project)
    caller_constraints = checkout / "caller-constraints.txt"
    caller_constraints.write_text("numpy<2\n")

    uv = checkout / "uv"
    uv.write_text(
        """#!/bin/sh
set -eu
cp pyproject.toml uv-project.toml
for file in ${UV_CONSTRAINT:-}; do cat "$file"; done > uv-constraints.txt
exit 42
"""
    )
    uv.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "requirements/install.sh",
            "--platform",
            "amd",
            "--rocm",
            rocm,
            "embodied",
            *target,
        ],
        cwd=checkout,
        env={
            **os.environ,
            "PATH": f"{checkout}{os.pathsep}{os.environ['PATH']}",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "UV_PYTHON_PREFERENCE": "only-system",
            "UV_CONSTRAINT": str(caller_constraints),
            "AMD_THEROCK_INDEX": "https://therock.example.test/simple",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert project.read_bytes() == (_REPO_ROOT / "pyproject.toml").read_bytes()
    return result


def test_rocm10_install_adds_torch_pins_to_the_callers_uv_constraints(tmp_path):
    result = _run_amd_installer(
        tmp_path, "10.0", "--model", "openpi", "--env", "libero"
    )
    assert result.returncode == 42, result.stderr

    project = (tmp_path / "uv-project.toml").read_text()
    assert 'url = "https://therock.example.test/simple"' in project
    assert 'rocm-sdk-core = { index = "pytorch-platform" }' in project
    constraints = (tmp_path / "uv-constraints.txt").read_text().splitlines()
    assert "numpy<2" in constraints
    rocm10_pins = [pin for pin in constraints if pin.endswith("+rocm10.0.0")]
    assert sorted(pin.split("==")[0] for pin in rocm10_pins) == [
        "torch",
        "torchaudio",
        "torchvision",
    ]


def test_supermimic_rocm72_uses_its_radeon_wheels(tmp_path):
    result = _run_amd_installer(tmp_path, "7.2", "--env", "supermimic")
    assert result.returncode == 42, result.stderr

    project = (tmp_path / "uv-project.toml").read_text()
    assert 'url = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/"' in project
    assert 'format = "flat"' in project
    assert '"torch==2.9.1"' in project
    assert '"torchvision==0.24.0"' in project
    assert (tmp_path / "uv-constraints.txt").read_text() == "numpy<2\n"


def test_supermimic_rejects_therock_before_install(tmp_path):
    result = _run_amd_installer(tmp_path, "10.0", "--env", "supermimic")
    assert result.returncode == 1
    assert "SuperMimic" in result.stderr and "--rocm 7.2" in result.stderr
    assert not (tmp_path / "uv-project.toml").exists()
