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

"""Write episode tasks from native LIBERO states or a RoboTwin seed table."""

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


def write_libero_tasks(args: argparse.Namespace) -> None:
    """Write the complete native training pool or a fixed regression panel."""
    if args.libero_config_path is not None:
        os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config_path)
    os.environ["LIBERO_TYPE"] = args.libero_type
    if args.libero_type == "pro":
        from liberopro.liberopro import benchmark
    else:
        from libero.libero import benchmark
    rows, tasks = [], []
    for suite in args.suites:
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
    write_task_pool(
        args.output_dir,
        args.split,
        rows,
        {
            "libero_type": args.libero_type,
            "suites": args.suites,
            "tasks": tasks,
            "seeds": args.seeds,
            "eval_is_training_pool_regression": True,
        },
    )


def write_robotwin_tasks(args: argparse.Namespace) -> None:
    """Convert the RPent task/seed table to one episode task per scene."""
    seed_table = json.loads(args.robotwin_seeds.read_text(encoding="utf-8"))
    task_config = seed_table["task_config"]
    rows = []
    for task, entries in seed_table["tasks"].items():
        for entry in entries:
            seed = entry["seed"]
            answer = {
                "environment": "robotwin",
                "task_name": task,
                "task_config": task_config,
                "seed": seed,
            }
            rows.append(
                {
                    "query_id": f"{task_config}-{task}-{seed}",
                    "prompt": entry["task_language"],
                    "answer": json.dumps(answer, sort_keys=True),
                }
            )
    write_task_pool(
        args.output_dir,
        args.split,
        rows,
        {"task_config": task_config, "source": args.robotwin_seeds.name},
    )


def write_metaworld_tasks(args: argparse.Namespace) -> None:
    """Replay native MT1 indices once per task and partition them by ML45 split."""
    from metaworld import MT1
    from robots.rl.envs.metaworld import initial_scene

    seeds = list(range(50)) if args.seeds == "all" else parse_seeds(args.seeds)
    if any(seed >= 50 for seed in seeds):
        raise ValueError("MetaWorld task indices must be in [0, 49]")
    config = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "rlinf/envs/sim/metaworld/metaworld_config.json"
        ).read_text(encoding="utf-8")
    )
    descriptions = config["TASK_DESCRIPTIONS"]
    pools = {
        "mt50": sorted(descriptions),
        "ml45-train": sorted(config["ML45"]["train"]),
        "ml45-ood": sorted(config["ML45"]["test"]),
    }
    rows = {}
    for name in pools["mt50"]:
        benchmark = MT1(name, seed=0)
        rows[name] = []
        for seed in seeds:
            env, identity = initial_scene(name, seed, benchmark)
            env.close()
            task = {
                "environment": "metaworld",
                "task_name": name,
                "seed": seed,
                "init_state_id": seed,
                "task_index": seed,
            }
            rows[name].append(
                {
                    "query_id": f"metaworld-{name}-s{seed}",
                    "prompt": descriptions[name],
                    "answer": json.dumps(task, sort_keys=True),
                    "initial_state": identity,
                }
            )
    for split, names in pools.items():
        write_task_pool(
            args.output_dir,
            split,
            [row for name in names for row in rows[name]],
            {"environment": "metaworld", "seeds": seeds},
        )


def write_task_pool(
    output_dir: Path, split: str, rows: list[dict], manifest: dict
) -> None:
    """Write the task pool and its content digest."""
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split}.jsonl").write_bytes(payload)
    manifest["files"] = {
        split: {"rows": len(rows), "sha256": hashlib.sha256(payload).hexdigest()}
    }
    (output_dir / f"{split}-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
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
        "--suites", nargs="+", default=EPISODE_SUITES, help="LIBERO suite names"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--metaworld", action="store_true", help="Prepare MT50 and ML45 pools"
    )
    mode.add_argument(
        "--robotwin-seeds", type=Path, help="RPent RoboTwin task/seed table"
    )
    parser.add_argument(
        "--seeds", default="all", help="Native seeds or ranges, e.g. 0-1"
    )
    parser.add_argument(
        "--split", choices=("train", "dev", "test", "eval"), default="train"
    )
    args = parser.parse_args()
    if args.metaworld:
        write_metaworld_tasks(args)
    elif args.robotwin_seeds is not None:
        write_robotwin_tasks(args)
    else:
        write_libero_tasks(args)


if __name__ == "__main__":
    main()
