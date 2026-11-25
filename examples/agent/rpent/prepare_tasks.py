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

"""Write a LIBERO episode task pool or fixed evaluation manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path

EPISODE_SUITES = ("libero_goal", "libero_spatial", "libero_object")


def state_digest(state) -> str:
    """Hash a native state's dtype, shape and exact values."""
    import numpy as np

    array = np.ascontiguousarray(state)
    return hashlib.sha256(
        str((array.dtype.str, array.shape)).encode() + array.tobytes()
    ).hexdigest()


def unique_states(states) -> list[dict]:
    """Keep original native state IDs, deduplicated within their task."""
    seen, result = set(), []
    for state_id, state in enumerate(states):
        digest = state_digest(state)
        if digest not in seen:
            result.append({"init_state_id": state_id, "state_sha256": digest})
            seen.add(digest)
    return result


def write_episode_tasks(args: argparse.Namespace) -> None:
    """Write the complete native training pool or a fixed regression panel."""
    if args.libero_config_path is not None:
        os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config_path)
    os.environ["LIBERO_TYPE"] = args.libero_type
    if args.libero_type == "pro":
        from liberopro.liberopro import benchmark
    else:
        from libero.libero import benchmark
    rows, tasks = [], []
    for suite in EPISODE_SUITES:
        bench = benchmark.get_benchmark(suite)()
        for task in range(bench.n_tasks):
            states = bench.get_task_init_states(task)
            available = unique_states(states)
            selected = available
            if args.seeds != "all":
                seeds = parse_seeds(args.seeds)
                if any(seed >= len(states) for seed in seeds):
                    raise ValueError(
                        f"{suite}/t{task} has only {len(states)} native states."
                    )
                selected = [
                    {"init_state_id": seed, "state_sha256": state_digest(states[seed])}
                    for seed in seeds
                ]
            tasks.append(
                {
                    "suite": suite,
                    "task": task,
                    "native_count": len(states),
                    "unique_states": available,
                }
            )
            for entry in selected:
                answer = {
                    "environment": "libero",
                    "suite": suite,
                    "task": task,
                    "seed": entry["init_state_id"],
                    "init_state_id": entry["init_state_id"],
                }
                rows.append(
                    {
                        "prompt": bench.get_task(task).language,
                        "answer": json.dumps(answer, sort_keys=True),
                        "query_id": f"{suite}-t{task}-state{entry['init_state_id']}",
                    }
                )
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.split}.jsonl").write_bytes(payload)
    manifest = {
        "libero_type": args.libero_type,
        "suites": EPISODE_SUITES,
        "tasks": tasks,
        "seeds": args.seeds,
        "eval_is_training_pool_regression": True,
        "files": {
            args.split: {
                "rows": len(rows),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    (args.output_dir / f"{args.split}-manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


def parse_seeds(value: str) -> list[int]:
    """Parse nonnegative seed IDs and inclusive ranges without duplicates."""
    seeds = set()
    for part in value.split(","):
        bounds = part.strip().split("-")
        if len(bounds) not in (1, 2) or not all(item.isdecimal() for item in bounds):
            raise ValueError(f"Expected nonnegative seeds/ranges, got {part!r}.")
        first, last = int(bounds[0]), int(bounds[-1])
        if last < first:
            raise ValueError(f"Seed range ends before it starts: {part!r}.")
        seeds.update(range(first, last + 1))
    return sorted(seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-type", choices=("standard", "pro"), default="pro")
    parser.add_argument("--libero-config-path", type=Path)
    parser.add_argument(
        "--seeds", default="all", help="Native seeds or ranges, e.g. 0-1"
    )
    parser.add_argument(
        "--split", choices=("train", "dev", "test", "eval"), default="train"
    )
    write_episode_tasks(parser.parse_args())


if __name__ == "__main__":
    main()
