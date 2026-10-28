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

"""The Slurm launcher's commands, with scontrol, srun and ray stubbed.

No Slurm, Ray cluster or GPU is involved. This checks the node ranks, the
head address and the placement the script hands to run_supermimic.sh.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "examples" / "embodiment" / "slurm_supermimic.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _stub(path: Path, body: str) -> None:
    path.write_text("#! /bin/bash\n" + body)
    path.chmod(0o755)


def _setup(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    _stub(bin_dir / "scontrol", "printf 'node-a\\nnode-b\\nnode-c\\n'\n")
    _stub(
        bin_dir / "srun",
        f'echo "$*" >> {log}\n'
        'case "$*" in *hostname*) echo "192.0.2.10 172.17.0.1" ;; esac\n',
    )
    _stub(bin_dir / "ray", "exit 0\n")
    repo = tmp_path / "repo"
    embodied = repo / "examples" / "embodiment"
    embodied.mkdir(parents=True)
    (embodied / "train_supermimic.py").write_text("")
    args = tmp_path / "driver_args.txt"
    _stub(
        embodied / "run_supermimic.sh", f'echo "RAY_ADDRESS=$RAY_ADDRESS $*" > {args}\n'
    )
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_JOB_NODELIST": "node-[a-c]",
        "SLURM_SUBMIT_DIR": str(repo),
        "RAY_HEAD_WAIT": "0",
    }
    return env, log, args


def test_slurm_script_starts_one_ray_node_per_host_and_spans_all_gpus(tmp_path):
    env, log, args = _setup(tmp_path)
    env["SLURM_GPUS_ON_NODE"] = "4"
    result = subprocess.run(
        ["bash", str(SCRIPT), "supermimic_lafan1", "env.motion_path=/shared/lafan1"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert args.read_text().split() == [
        "RAY_ADDRESS=192.0.2.10:6379",
        "supermimic_lafan1",
        "cluster.num_nodes=3",
        "cluster.component_placement.supermimic=0-11",
        "env.motion_path=/shared/lafan1",
    ]
    calls = log.read_text()
    assert (
        "-w node-a env RLINF_NODE_RANK=0 ray start --head "
        "--node-ip-address=192.0.2.10 --port=6379 --num-gpus=4 --block"
    ) in calls
    for rank, node in ((1, "node-b"), (2, "node-c")):
        assert (
            f"-w {node} env RLINF_NODE_RANK={rank} ray start "
            "--address=192.0.2.10:6379 --num-gpus=4 --block"
        ) in calls
    # The stop step must overlap the blocking start steps that hold every slot.
    assert "--overlap --nodes=3 --ntasks=3 ray stop" in calls


def test_slurm_script_needs_a_gpu_count(tmp_path):
    env, _log, args = _setup(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), "supermimic_lafan1"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "GPUS_PER_NODE" in result.stderr
    assert not args.exists()
