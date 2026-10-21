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

"""Run the four controlled comparisons sequentially on the same eight GPUs."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="searchr1-ppo-bsz1024-200")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--critic-updates", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["sync-shared", "async-shared", "sync-split", "async-split"],
        default=["sync-shared", "async-shared", "sync-split", "async-split"],
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    output = args.output_dir.resolve()
    supervisor = output / args.name
    supervisor.mkdir(parents=True, exist_ok=True)
    configs = {
        "sync-shared": "train_qwen2.5_ppo_shared",
        "async-shared": "train_qwen2.5_sao",
        "sync-split": "train_qwen2.5_ppo_split",
        "async-split": "train_qwen2.5_sao_split",
    }
    environment = os.environ.copy()
    megatron = environment.get("MEGATRON_PATH", str(Path(sys.prefix) / "Megatron-LM"))
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [
                    str(repo),
                    str(repo / "examples"),
                    megatron,
                    environment.get("PYTHONPATH", ""),
                ]
            ),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "RAY_DEDUP_LOGS": "0",
        }
    )
    common = [
        f"runner.max_steps={args.steps}",
        # The step budget can span multiple dataset epochs.
        f"runner.max_epochs={args.steps}",
        "runner.save_interval=50",
        "runner.val_check_interval=-1",
        f"runner.output_dir={output}",
        f"data.rollout_batch_size={args.batch_size}",
        "algorithm.group_size=1",
        "algorithm.n_minibatches=1",
        "agentloop.truncate_response=false",
        "algorithm.recompute_logprobs=false",
        "actor.optim.lr=1e-6",
        "actor.lr_sched.max_lr=1e-6",
        "actor.lr_sched.lr_warmup_fraction=null",
        "actor.lr_sched.lr_warmup_iters=0",
        "critic.optim.lr=5e-6",
        "critic.lr_sched.max_lr=5e-6",
        "critic.lr_sched.lr_warmup_fraction=null",
        "critic.lr_sched.lr_warmup_iters=10",
        "data.seed=1234",
        "actor.seed=1234",
        "critic.seed=1234",
        "algorithm.sampling_params.temperature=1.0",
        "algorithm.sampling_params.top_p=1.0",
        "algorithm.sampling_params.top_k=-1",
        "algorithm.sampling_params.repetition_penalty=1.0",
    ]
    common.extend(
        f"{role}.{setting}=true"
        for role in ("actor", "critic")
        for setting in ("offload_weight", "offload_grad", "offload_optimizer")
    )
    patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
    untracked = (
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo
        )
        .decode()
        .split("\0")
    )
    for name in filter(None, untracked):
        patch += subprocess.run(
            ["git", "diff", "--no-index", "--", "/dev/null", name],
            cwd=repo,
            stdout=subprocess.PIPE,
            check=False,
        ).stdout
    (supervisor / "source.patch").write_bytes(patch)
    (supervisor / "base_commit.txt").write_bytes(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
    )
    runs = []
    for mode in args.modes:
        command = [
            sys.executable,
            str(Path(__file__).with_name("benchmark.py")),
            "--config-path",
            str(repo / "examples/agent/searchr1/config"),
            "--config-name",
            configs[mode],
            f"runner.experiment_name={args.name}-{mode}",
            *common,
            f"critic.update_epochs={args.critic_updates if mode.startswith('async') else 1}",
        ]
        record = {"mode": mode, "command": command, "started": time.time()}
        runs.append(record)
        (supervisor / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        with (supervisor / f"{mode}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        record.update({"finished": time.time(), "returncode": result.returncode})
        (supervisor / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(
                f"{mode} failed with status {result.returncode}; see {supervisor / (mode + '.log')}"
            )


if __name__ == "__main__":
    main()
