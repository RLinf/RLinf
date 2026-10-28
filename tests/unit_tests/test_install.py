# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Installer CLI contracts without downloading packages or requiring hardware."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def ascend_install(tmp_path):
    root = Path(__file__).resolve().parents[2]
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    script = requirements / "install.sh"
    shutil.copyfile(root / "requirements/install.sh", script)
    pyproject = tmp_path / "pyproject.toml"
    original = (root / "pyproject.toml").read_text()
    pyproject.write_text(original)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text("")

    # The package manager records the real installer's requirements. The fake
    # torch distribution exposes the version that those requirements select.
    (bin_dir / "uv").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['INSTALL_LOG'], 'a') as log:\n"
        "    log.write(json.dumps({'args': sys.argv[1:], "
        "'pyproject': Path('pyproject.toml').read_text()}) + '\\n')\n"
    )
    (bin_dir / "uv").chmod(0o755)
    (bin_dir / "torch.py").write_text(
        "import re\n"
        "from pathlib import Path\n"
        '__version__ = re.search(r\'"torch==([^"]+)"\', '
        "Path('pyproject.toml').read_text()).group(1)\n"
    )

    def run(chip_output="910B4", probe_exit=0, arch="x86_64", chip=None, torch=None):
        probe = bin_dir / "npu-smi"
        probe.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$PROBE_OUTPUT"\nexit "$PROBE_EXIT"\n'
        )
        probe.chmod(0o755)
        uname = bin_dir / "uname"
        uname.write_text('#!/bin/bash\nprintf "%s\\n" "$TEST_ARCH"\n')
        uname.chmod(0o755)
        log = tmp_path / "install.jsonl"
        env = {
            **os.environ,
            "HOME": str(tmp_path),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(bin_dir),
            "INSTALL_LOG": str(log),
            "PROBE_OUTPUT": chip_output,
            "PROBE_EXIT": str(probe_exit),
            "TEST_ARCH": arch,
        }
        env.pop("ASCEND_CHIP", None)
        env.pop("UV_TORCH_BACKEND", None)
        if chip is not None:
            env["ASCEND_CHIP"] = chip
        command = [
            "bash",
            str(script),
            "--platform",
            "ascend",
            "--no-root",
            "--python",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "--venv",
            str(venv),
            "embodied",
            "--env",
            "dummy",
        ]
        if torch is not None:
            command.extend(["--torch", torch])
        result = subprocess.run(
            command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert pyproject.read_text() == original
        return [json.loads(line) for line in log.read_text().splitlines()]

    return run


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize(
    ("chip_name", "torch_version", "npu_requirement"),
    [
        ("910B4", "2.6.0", "torch-npu==2.6.0"),
        ("Ascend910B4", "2.6.0", "torch-npu==2.6.0"),
        ("950PR", "2.10.0", "torch-npu==2.10.0.post4"),
        ("Ascend950PR", "2.10.0", "torch-npu==2.10.0.post4"),
    ],
)
def test_ascend_installer_selects_compatible_stack(
    ascend_install, arch, chip_name, torch_version, npu_requirement
):
    calls = ascend_install(chip_output=f"| 0 | {chip_name} | OK |", arch=arch)
    assert any(npu_requirement in call["args"] for call in calls)
    project = next(call["pyproject"] for call in calls if call["args"][0] == "sync")
    suffix = "+cpu" if arch == "x86_64" else ""
    assert f'"torch=={torch_version}{suffix}"' in project
    if arch == "x86_64":
        assert "https://download.pytorch.org/whl/cpu" in project


@pytest.mark.parametrize(("output", "exit_code"), [("", 1), ("Device unavailable", 0)])
def test_ascend_installer_handles_failed_chip_probe(ascend_install, output, exit_code):
    calls = ascend_install(chip_output=output, probe_exit=exit_code)
    assert any("torch-npu==2.6.0" in call["args"] for call in calls)


def test_ascend_installer_honors_torch_override_without_chip_probe(ascend_install):
    calls = ascend_install(chip_output="", probe_exit=1, torch="2.9.0")
    assert any("torch-npu==2.9.0" in call["args"] for call in calls)


def test_ascend_installer_honors_chip_override(ascend_install):
    calls = ascend_install(chip_output="", probe_exit=1, chip="Ascend950PR")
    assert any("torch-npu==2.10.0.post4" in call["args"] for call in calls)


def test_ascend_installer_keeps_950_post_build_with_matching_torch_override(
    ascend_install,
):
    calls = ascend_install(chip="Ascend950PR", torch="2.10.0")
    assert any("torch-npu==2.10.0.post4" in call["args"] for call in calls)
