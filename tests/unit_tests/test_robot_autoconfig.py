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

"""Cluster-level tests for environment-driven robot auto-configuration.

Each scenario runs in a fresh process because ``Cluster`` is process-wide and
Ray must inherit the environment before startup.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCENARIO = os.path.join(os.path.dirname(__file__), "_robot_autoconfig_cluster.py")
_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _run_scenario(mode: str) -> subprocess.CompletedProcess:
    """Run a cluster scenario in a subprocess and capture its output."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (_REPO_ROOT, env.get("PYTHONPATH")) if path
    )
    return subprocess.run(
        [sys.executable, _SCENARIO, mode],
        capture_output=True,
        text=True,
        timeout=420,
        env=env,
    )


@pytest.mark.parametrize(
    "mode",
    [
        # Two robots from comma-separated environment values.
        "create_multi",
        # A single robot keeps the whole comma-separated camera list.
        "create_single",
        # Explicit YAML values take precedence; omitted values are resolved.
        "explicit_fill",
        # GimArm resolves through its own identifier variable.
        "gim_create",
        # Shared fields alone do not create a robot.
        "gating",
        # The identifier env var has too few values for the configs.
        "mismatch_low",
        # The identifier env var has too many values for the configs.
        "mismatch_high",
        # The identifier count is fine but a secondary field disagrees.
        "mismatch_secondary",
        # Fully specified YAML remains supported for one or more robots.
        "yaml_single",
        "yaml_multi",
        "yaml_dosw1",
    ],
)
def test_real_cluster_robot_autoconfig(mode):
    result = _run_scenario(mode)
    assert f"{mode}:OK" in result.stdout, (
        f"scenario {mode} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0
