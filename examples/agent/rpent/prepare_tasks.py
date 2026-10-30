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

"""Write the LIBERO-PRO task files of the RPent PPO example.

A task is used when it has all 50 initial states and its three frozen memory
notes. One task per suite is held out. The other tasks train on seeds 1-32;
seeds 33-39 validate them and seeds 40-49 are left for a final test. Seed 0
produced the memory notes and is never used.

Output files, one JSON row per episode:

- ``train.jsonl``: training tasks on the training seeds.
- ``panel_train.jsonl``: four of those seeds per training task.
- ``dev.jsonl`` and ``dev_quick.jsonl``: training tasks on validation seeds;
  the quick file keeps two of them per task.
- ``holdout.jsonl``: held-out tasks on seeds 1-16.
- ``test.jsonl``: every task on the test seeds.

``manifest.json`` lists the held-out and excluded tasks and each file's hash.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

SUITES = [
    f"{kind}_{variant}"
    for kind in ("goal", "object", "spatial", "10")
    for variant in ("task", "swap")
]
INITIAL_STATES = 50
TRAIN_SEEDS = range(1, 33)
PANEL_SEEDS = (1, 9, 17, 25)
DEV_SEEDS = range(33, 40)
TEST_SEEDS = range(40, 50)
HOLDOUT_SEEDS = range(1, 17)
# Pi0.5 action sampling seed, fixed so that seeds alone define an episode.
VLA_SEED = 20261023


def memory_files(suite: str, task: int) -> list[str]:
    """Return a task's frozen memory notes, relative to the memory directory."""
    # libero_10 family notes are named task-family_libero10_<variant>_t<k>.md.
    family = f"libero{'' if suite.startswith('10_') else '_'}{suite}"
    return [
        f"task-specific/{suite}_t{task}_s0.json",
        f"task-specific/{suite}_t{task}_s0_recipe.jsonl",
        f"task-family/task-family_{family}_t{task}.md",
    ]


def load_tasks(memory_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return the usable tasks and the excluded ones with what they lack."""
    from liberopro.liberopro import benchmark

    tasks, excluded = [], []
    for suite in SUITES:
        bench = benchmark.get_benchmark(f"libero_{suite}")()
        for task in range(bench.n_tasks):
            files = memory_files(suite, task)
            missing = [name for name in files if not (memory_dir / name).is_file()]
            states = len(bench.get_task_init_states(task))
            if states != INITIAL_STATES or missing:
                excluded.append(
                    {
                        "suite": f"libero_{suite}",
                        "task": task,
                        "initial_states": states,
                        "missing_memory": missing,
                    }
                )
                continue
            entries = []
            for name in files:
                payload = (memory_dir / name).read_bytes()
                entries.append(
                    {
                        "bytes": len(payload),
                        "id": name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            tasks.append(
                {
                    "suite": f"libero_{suite}",
                    "task": task,
                    "language": bench.get_task(task).language,
                    "memory_entries": entries,
                }
            )
    return tasks, excluded


def row(task: dict, seed: int, split: str) -> dict:
    """Return the dataset row of one episode."""
    answer = {
        "memory_entries": task["memory_entries"],
        "seed": seed,
        "suite": task["suite"],
        "task": task["task"],
        "vla_seed": VLA_SEED,
    }
    return {
        "prompt": task["language"],
        "answer": json.dumps(answer, sort_keys=True),
        "query_id": f"{split}-{task['suite']}-{task['task']}-{seed}",
    }


def split_rows(tasks: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Return the rows of every output file and the held-out tasks."""
    pick = random.Random(20261028)
    holdout = [
        pick.choice([task for task in tasks if task["suite"] == f"libero_{suite}"])
        for suite in SUITES
    ]
    train = [task for task in tasks if task not in holdout]
    quick = random.Random(20261029)
    rows = {
        "train": [row(t, s, "train") for t in train for s in TRAIN_SEEDS],
        "panel_train": [row(t, s, "panel") for t in train for s in PANEL_SEEDS],
        "dev_quick": [
            row(t, s, "dev-quick")
            for t in train
            for s in sorted(quick.sample(list(DEV_SEEDS), 2))
        ],
        "dev": [row(t, s, "dev") for t in train for s in DEV_SEEDS],
        "holdout": [row(t, s, "holdout") for t in holdout for s in HOLDOUT_SEEDS],
        "test": [row(t, s, "test") for t in tasks for s in TEST_SEEDS],
    }
    return rows, holdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-dir",
        type=Path,
        required=True,
        help="Frozen LIBERO memory corpus, the same as tools.rpent.frozen_memory_dir.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tasks, excluded = load_tasks(args.memory_dir)
    rows, holdout = split_rows(tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tasks": len(tasks),
        "holdout_tasks": [f"{t['suite']}/t{t['task']}" for t in holdout],
        "excluded": excluded,
        "files": {},
    }
    for name, split in rows.items():
        payload = "".join(json.dumps(r) + "\n" for r in split).encode()
        (args.output_dir / f"{name}.jsonl").write_bytes(payload)
        manifest["files"][name] = {
            "rows": len(split),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
