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

"""Installer synchronization against an offline uv environment."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("model", ["diffusion", "openpi"])
def test_installer_sync_preserves_external_paddle_only_for_diffusion(tmp_path, model):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("requires uv")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="sync-test"\nversion="0.0.1"\n'
        "[project.optional-dependencies]\nembodied=[]\n[tool.uv]\npackage=false\n"
    )
    requirements = tmp_path / "requirements"
    common = requirements / "embodied/envs/common.txt"
    common.parent.mkdir(parents=True)
    common.touch()
    wheel = tmp_path / "paddlepaddle_gpu-2.6.2.post120-py3-none-any.whl"
    dist = "paddlepaddle_gpu-2.6.2.post120.dist-info"
    files = {
        f"{dist}/METADATA": "Metadata-Version: 2.1\nName: paddlepaddle-gpu\nVersion: 2.6.2.post120\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files[f"{dist}/RECORD"] = (
        "".join(f"{name},,\n" for name in files) + f"{dist}/RECORD,,\n"
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    env = os.environ | {
        "PATH": f"{Path(uv).parent}{os.pathsep}{os.environ['PATH']}",
        "UV_OFFLINE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
    }
    venv = tmp_path / ".venv"
    subprocess.run(
        [uv, "venv", str(venv), "--python", sys.executable], env=env, check=True
    )
    subprocess.run(
        [uv, "pip", "install", "--python", str(venv / "bin/python"), str(wheel)],
        env=env,
        check=True,
    )
    script = r"""
source "$1"
MODEL="$2"
SCRIPT_DIR="$3"
VENV_DIR="$4"
PYTHON_VERSION="$5"
NO_ROOT=1
check_paddle() {
    python -c 'from importlib.metadata import distributions; print("PADDLE_PRESENT=" + str(any(d.metadata["Name"] == "paddlepaddle-gpu" for d in distributions())))'
}
for attempt in 1 2; do
    create_and_sync_venv
    check_paddle
    install_common_embodied_deps
    check_paddle
done
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "bash",
            str(Path(__file__).resolve().parents[2] / "requirements/install.sh"),
            model,
            str(requirements),
            str(venv),
            f"{sys.version_info.major}.{sys.version_info.minor}",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    present = "True" if model == "diffusion" else "False"
    assert result.stdout.count(f"PADDLE_PRESENT={present}") == 4


@pytest.mark.parametrize(
    ("cuda_version", "paddle_version"),
    [
        ("11.8", "2.6.2"),
        ("12.0", "2.6.2.post120"),
        ("12.8", "2.6.2.post120"),
        ("13.0", None),
        ("14.0", None),
        ("10.2", None),
        ("", None),
    ],
)
def test_paddle_gpu_accepts_only_supported_cuda_versions(
    tmp_path, cuda_version, paddle_version
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "script = sys.stdin.read()\n"
        "if 'torch.version.cuda' in script:\n"
        "    version = os.environ['TEST_CUDA_VERSION']\n"
        "    if not version: raise SystemExit(1)\n"
        "    print(version.replace('.', ' '))\n"
        "elif 'PADDLE_GPU_VERSION' in os.environ:\n"
        "    print('CHECKED_PADDLE=' + os.environ['PADDLE_GPU_VERSION'])\n"
        "else:\n"
        "    raise SystemExit('Unexpected Python invocation')\n"
    )
    python.chmod(0o755)
    nvcc = bin_dir / "nvcc"
    nvcc.write_text("#!/bin/sh\nexit 0\n")
    nvcc.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\necho 'Unexpected package modification' >&2\nexit 99\n")
    uv.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; install_paddle_gpu',
            "bash",
            str(Path(__file__).resolve().parents[2] / "requirements/install.sh"),
        ],
        env=os.environ
        | {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "TEST_CUDA_VERSION": cuda_version,
        },
        text=True,
        capture_output=True,
        timeout=15,
    )
    output = result.stdout + result.stderr
    assert "Unexpected package modification" not in output
    if paddle_version is None:
        assert result.returncode == 1, output
        assert "supports only CUDA 11.x and 12.x" in output
        assert "CHECKED_PADDLE=" not in output
    else:
        assert result.returncode == 0, output
        assert f"CHECKED_PADDLE={paddle_version}" in output
        assert "already installed; skipping" in output
