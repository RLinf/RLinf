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

"""Prepare paired Search-R1 configurations without launching workers."""

import argparse
import copy
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import yaml


def main() -> None:
    """Freeze input rows and write configurations differing only by test mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--search-server", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    assert args.steps > 0 and args.batch_size > 0, (
        "Steps and batch size must be positive"
    )
    root = Path(__file__).resolve().parents[4]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = root / "examples/agent/searchr1/config/train_glm4.7_flash.yaml"
    config = yaml.safe_load(source.read_text())
    count = args.steps * args.batch_size
    with args.data.open() as stream:
        rows = list(itertools.islice(stream, count))
    assert len(rows) == count, f"Expected {count} input rows, found {len(rows)}"
    assert all(isinstance(json.loads(row).get("messages"), list) for row in rows), (
        "Expected GLM message-list samples under the messages key"
    )
    dataset = output / "questions.jsonl"
    dataset.write_text("".join(rows))
    config["runner"].update(
        max_steps=args.steps,
        max_epochs=1,
        save_interval=-1,
        val_check_interval=-1,
        experiment_name="train",
        per_worker_log=True,
    )
    config["runner"]["logger"]["logger_backends"] = ["tensorboard"]
    config["data"].update(
        train_data_paths=[str(dataset)],
        prompt_key="messages",
        rollout_batch_size=args.batch_size,
        shuffle=False,
        validation_shuffle=False,
        seed=42,
    )
    config["actor"]["seed"] = 42
    config["tools"]["search"]["server_addr"] = args.search_server
    config["rollout"]["model"]["model_path"] = str(args.model.resolve())
    hashes = {}
    for case, enabled in (("baseline", False), ("ragged", True)):
        current = copy.deepcopy(config)
        current["actor"]["enable_ragged_batch"] = enabled
        current["runner"]["output_dir"] = str(output / case / "training")
        directory = output / case / "config"
        directory.mkdir(parents=True)
        target = directory / "run.yaml"
        target.write_text(yaml.safe_dump(current, sort_keys=False, allow_unicode=True))
        hashes[case] = hashlib.sha256(target.read_bytes()).hexdigest()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    (output / "code.diff").write_bytes(diff)
    manifest = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "steps": args.steps,
        "questions": count,
        "batch_size": args.batch_size,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "config_sha256": hashes,
        "source_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "code_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "comparison": "identical input rows; independent sampled model outputs",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
