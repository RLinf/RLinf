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

"""Real cluster + worker tests for env-var-driven robot auto-config.

Every scenario boots a real single-node cluster from a cluster config and
launches real workers (see ``_robot_autoconfig_cluster.py``), rather than
calling ``RobotAutoConfig`` with a patched environment. Scenarios run in
isolated subprocesses because ``Cluster`` is a process-wide singleton and the
enumeration actor must inherit the env vars set before Ray starts.
"""

import os
import subprocess
import sys

import pytest

_SCENARIO = os.path.join(os.path.dirname(__file__), "_robot_autoconfig_cluster.py")


def _run_scenario(mode: str) -> subprocess.CompletedProcess:
    """Run one cluster scenario in a fresh process and capture its output."""
    return subprocess.run(
        [sys.executable, _SCENARIO, mode],
        capture_output=True,
        text=True,
        timeout=420,
    )


@pytest.mark.parametrize(
    "mode",
    [
        # Two robots created from comma-separated env vars (one value each),
        # read back through both one-worker-per-robot and one-worker-owns-both
        # placements.
        "create_multi",
        # A single robot keeps the whole comma-separated camera list.
        "create_single",
        # Explicit configs: a YAML robot_ip is kept, an omitted one is filled
        # from the env value in its slot, YAML cameras preserved.
        "explicit_fill",
        # A second robot type (GimArm) is created from its own identifier env.
        "gim_create",
        # A shared field (camera serials) without the identifier (ROBOT_IP)
        # must not create any robot.
        "gating",
        # The identifier env var has too few values for the configs.
        "mismatch_low",
        # The identifier env var has too many values for the configs.
        "mismatch_high",
        # The identifier count is fine but a secondary field disagrees.
        "mismatch_secondary",
        # Regression: the legacy path (full YAML config, no env vars) must keep
        # working unchanged, for a single robot, several robots, and another
        # robot type (DOSW1).
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
