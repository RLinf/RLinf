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

"""Exercise the embodied bash entry point without launching model workers."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("exit_code", [0, 7])
def test_embodied_launcher_forwards_overrides_and_propagates_failure(
    tmp_path, exit_code
):
    repo = Path(__file__).resolve().parents[2]
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    python = binary_dir / "python"
    python.write_text(
        "#!/bin/bash\n"
        'printf "%s\\0" "$ROBOT_PLATFORM" "$@" > "$CAPTURE_FILE"\n'
        'echo "worker output"\n'
        'exit "$WORKER_EXIT_CODE"\n'
    )
    python.chmod(0o755)
    log_dir = tmp_path / "20261029_object0_branch8"
    log_file = log_dir / "20261029_object0_branch8.log"
    capture = tmp_path / "arguments"
    env = {
        **os.environ,
        "PATH": f"{binary_dir}{os.pathsep}{os.environ['PATH']}",
        "LOG_DIR": str(log_dir),
        "MEGA_LOG_FILE": str(log_file),
        "CAPTURE_FILE": str(capture),
        "WORKER_EXIT_CODE": str(exit_code),
        "STEPS": "-2",
        "SAVE_INTER": "-1",
        "NODES": "1",
    }
    override = "actor.model.model_path=/path/to/model with spaces"
    result = subprocess.run(
        [
            "bash",
            str(repo / "examples/embodiment/run_embodiment.sh"),
            "libero_object_0_art_openpi",
            "LIBERO",
            override,
            "runner.max_steps=100",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    arguments = capture.read_bytes().decode().rstrip("\0").split("\0")
    assert result.returncode == exit_code
    assert arguments[:3] == [
        "LIBERO",
        "-u",
        str(repo / "examples/embodiment/train_embodied_agent.py"),
    ]
    assert (
        arguments[arguments.index("--config-name") + 1] == "libero_object_0_art_openpi"
    )
    assert override in arguments
    assert f"runner.logger.log_path={log_dir}" in arguments
    assert "runner.save_interval=-1" in arguments and "cluster.num_nodes=1" in arguments
    assert "runner.max_steps=-2" not in arguments
    assert arguments[-1] == "runner.max_steps=100"
    assert "worker output" in result.stdout and "worker output" in log_file.read_text()


def test_embodied_launcher_preserves_config_only_invocation(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    python = binary_dir / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$CAPTURE_FILE"\n')
    python.chmod(0o755)
    capture = tmp_path / "arguments"
    log_dir = tmp_path / "legacy"
    env = {
        **os.environ,
        "PATH": f"{binary_dir}{os.pathsep}{os.environ['PATH']}",
        "LOG_DIR": str(log_dir),
        "CAPTURE_FILE": str(capture),
        "STEPS": "-2",
        "SAVE_INTER": "-2",
        "NODES": "-2",
    }
    env.pop("MEGA_LOG_FILE", None)
    result = subprocess.run(
        [
            "bash",
            str(repo / "examples/embodiment/run_embodiment.sh"),
            "libero_spatial_ppo_openpi",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    arguments = capture.read_bytes().decode().rstrip("\0").split("\0")
    assert result.returncode == 0
    assert (
        arguments[arguments.index("--config-name") + 1] == "libero_spatial_ppo_openpi"
    )
    assert not any(value.startswith("runner.max_steps=") for value in arguments)
    assert (log_dir / "run_embodiment.log").exists()
