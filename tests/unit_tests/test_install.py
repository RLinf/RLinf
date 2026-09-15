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

"""Installer dispatch and additive environment dependency installation."""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def run_install(tmp_path, monkeypatch):
    """Run the CLI with package downloads and host setup replaced by recorders."""
    for variable in ("FRANKY_WHEEL", "LIBFRANKA_VERSION", "GITHUB_PREFIX"):
        monkeypatch.delenv(variable, raising=False)
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/bash\nprintf "UV %s\\n" "$*"\n')
    uv.chmod(0o755)
    (tmp_path / "python").symlink_to(sys.executable)
    script = (REPO_ROOT / "requirements/install.sh").read_text()
    definitions, entrypoint = script.rsplit('\nmain "$@"', 1)
    assert not entrypoint.strip()
    setup = f"""
SCRIPT_DIR={shlex.quote(str(REPO_ROOT / "requirements"))}
configure_platform() {{ :; }}
setup_mirror() {{ :; }}
apply_torch_override() {{ :; }}
install_platform_extras() {{ :; }}
create_and_sync_venv() {{ uv sync --active; }}
install_common_embodied_deps() {{ uv sync --extra embodied --active; }}
clone_or_reuse_repo() {{ printf '%s/model\\n' "$VENV_DIR"; }}
install_flash_attn() {{ printf 'FLASH_ATTN\\n'; }}
maybe_build_decord_from_source() {{ :; }}
cp() {{ :; }}
bash() {{ printf 'BASH %s\\n' "$*"; }}
main "$@"
"""
    installer = tmp_path / "install.sh"
    installer.write_text(definitions + setup)

    def run(*args):
        result = subprocess.run(
            ["bash", str(installer), "embodied", "--no-root", *args],
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.splitlines()

    return run


@pytest.mark.parametrize(
    ("env_name", "requirements"),
    [
        ("franka", "franka"),
        ("franka-dexhand", "franka"),
        ("franka-ros", "franka"),
        ("xsquare_turtle2", "xsquare_turtle2"),
        ("gim_arm", "gim_arm"),
        ("so101", "so101"),
        ("piper", "piper"),
        ("dosw1", "dosw1"),
    ],
)
def test_robot_install_adds_requirements_without_resync(
    run_install, monkeypatch, env_name, requirements
):
    monkeypatch.setenv("SKIP_ROS", "1")
    lines = run_install("--env", env_name)
    requirement_file = REPO_ROOT / f"requirements/embodied/envs/{requirements}.txt"
    assert requirement_file.is_file()
    assert f"UV pip install -r {requirement_file}" in lines
    assert [line for line in lines if line.startswith("UV sync")] == [
        "UV sync --active"
    ]


@pytest.mark.parametrize("model", ["openvla", "openvla-oft", "openpi", "gr00t"])
def test_franka_model_install_keeps_model_and_robot_dependencies(run_install, model):
    lines = run_install("--model", model, "--env", "franka")
    requirement_file = REPO_ROOT / "requirements/embodied/envs/franka.txt"
    robot_install = lines.index(f"UV pip install -r {requirement_file}")
    assert [line for line in lines[:robot_install] if line.startswith("UV sync")] == [
        "UV sync --active",
        "UV sync --extra embodied --active",
    ]
    assert not any(line.startswith("UV sync") for line in lines[robot_install:])
    assert any("wheels-libfranka-0.19.0/franky_control-" in line for line in lines)
    assert any(model in line for line in lines if line.startswith("UV pip install"))
    assert "FLASH_ATTN" in lines


@pytest.mark.parametrize("platform", ["nvidia", "amd", "ascend", "musa"])
def test_franka_docker_uses_selected_platform(platform):
    dockerfile = (REPO_ROOT / "docker/Dockerfile").read_text()
    bases = {
        stage: base
        for base, stage in re.findall(r"^FROM (\S+) AS (\S+)$", dockerfile, re.M)
    }
    assert bases["base-image-embodied-franka"] == "base-image-platform-${PLATFORM}"
    assert "ARG FRANKA_BASE_IMAGE=" not in dockerfile

    stage = dockerfile.split(
        "FROM embodied-common-image AS embodied-franka-image\n", 1
    )[1].split("\nFROM ", 1)[0]
    installs = [
        line.removeprefix("RUN ")
        for line in stage.replace("\\\n", "").splitlines()
        if line.startswith("RUN ") and "requirements/install.sh" in line
    ]
    assert installs
    result = subprocess.run(
        ["bash", "-ec", 'bash() { printf "%s\\n" "$*"; }\n' + "\n".join(installs)],
        text=True,
        capture_output=True,
        env={**os.environ, "RLINF_PLATFORM": platform, "INSTALL_MIRROR_OPTION": ""},
        timeout=10,
        check=True,
    )
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 6
    for args in commands:
        assert args[args.index("--platform") + 1] == platform
