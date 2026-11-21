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

# Read the constraints before the shell exits; the installer's EXIT trap
# removes its temporary constraint file.
_ROCM10_UV_CONSTRAINTS = """
source requirements/install.sh
ROCM_VERSION=10.0
configure_amd > /dev/null
apply_torch_override > /dev/null
cat $UV_CONSTRAINT
"""


def _rocm10_uv_constraints(checkout: Path, **env: str) -> list[str]:
    """Run the ROCm 10 torch setup in a scratch checkout; return its uv constraints."""
    (checkout / "requirements").mkdir(parents=True)
    shutil.copy(_REPO_ROOT / "requirements" / "install.sh", checkout / "requirements")
    shutil.copy(_REPO_ROOT / "pyproject.toml", checkout)
    inherited = {
        name: value for name, value in os.environ.items() if name != "UV_CONSTRAINT"
    }
    result = subprocess.run(
        ["bash", "-c", _ROCM10_UV_CONSTRAINTS],
        cwd=checkout,
        env={**inherited, **env},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def test_rocm10_install_adds_torch_pins_to_the_callers_uv_constraints(tmp_path):
    caller_constraints = tmp_path / "caller-constraints.txt"
    caller_constraints.write_text("numpy<2\n")

    constraints = _rocm10_uv_constraints(
        tmp_path / "checkout", UV_CONSTRAINT=str(caller_constraints)
    )

    assert "numpy<2" in constraints
    rocm10_pins = [pin for pin in constraints if pin.endswith("+rocm10.0.0")]
    assert sorted(pin.split("==")[0] for pin in rocm10_pins) == [
        "torch",
        "torchaudio",
        "torchvision",
    ]
