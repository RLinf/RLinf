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

"""Verify a released ``norm_stats.json`` against a LeRobot episode on CPU.

RLinf OpenPI SFT configs such as ``pi0_libero`` set
``extra_delta_transform=True`` (see
``rlinf/models/embodiment/openpi/dataconfig/__init__.py``), which pushes an
additional :class:`openpi.transforms.DeltaActions` transform onto the input
pipeline. ``toolkits/lerobot/calculate_norm_stats.py`` computes statistics
*after* ``data_config.data_transforms.inputs``, so for those configs the
stored ``actions`` statistics describe **(actions - state)** on the first
``num_delta_dims`` dimensions (gripper kept absolute), not the raw dataset
actions. This is why the ``actions`` stats shipped with RLinf LIBERO
π₀ SFT checkpoints differ substantially from the raw
``physical-intelligence/libero`` dataset stats (``state`` stats are
unaffected because no transform touches the state).

This tool recomputes mean/std/q01/q99 from one or more episode ``.parquet``
files and diffs them against a reference ``norm_stats.json`` so the two can
be compared apples-to-apples::

    # Raw stats (compare against the HF dataset meta/stats.json):
    python toolkits/lerobot/verify_norm_stats.py \
        --parquet "~/.cache/huggingface/lerobot/data/chunk-000/episode_000000.parquet" \
        --norm-stats /path/to/norm_stats.json

    # Pipeline stats (compare against the checkpoint norm_stats.json):
    python toolkits/lerobot/verify_norm_stats.py \
        --parquet ... --norm-stats /path/to/norm_stats.json --extra-delta

Stats converge only with enough episodes; the tool reports per-key diffs so
the *direction* of the agreement is still readable from a single episode.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

STAT_NAMES = ("mean", "std", "q01", "q99")


def apply_delta_actions(
    actions: np.ndarray,
    state: np.ndarray,
    horizon: int = 50,
    num_delta_dims: int = 6,
) -> np.ndarray:
    """Apply openpi's ``DeltaActions`` transform to an episode.

    Mirrors ``openpi.transforms.DeltaActions`` as applied through the chunked
    dataset: for each chunk starting at frame ``s``, the first
    ``num_delta_dims`` action dims get the state at frame ``s`` subtracted
    (broadcast over the chunk). The remaining dims (e.g. the gripper) stay
    absolute. ``actions`` has shape ``(T, A)`` and ``state`` has shape
    ``(T, S)``.
    """
    actions = actions.copy()
    total = actions.shape[0]
    for start in range(0, total, horizon):
        actions[start : start + horizon, :num_delta_dims] -= state[start][
            None, :num_delta_dims
        ]
    return actions


def compute_episode_stats(
    state: np.ndarray, actions: np.ndarray
) -> dict[str, dict[str, list[float]]]:
    """Compute mean/std/q01/q99 for ``state`` and ``actions``."""
    stats: dict[str, dict[str, list[float]]] = {}
    for key, values in (("state", state), ("actions", actions)):
        stats[key] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    return stats


def extract_reference_stats(reference: dict, key: str) -> dict[str, np.ndarray]:
    """Pull one key's stats out of a norm-stats JSON.

    Supports both the LeRobot layout (``{key: {stat: [...]}}``) and the
    openpi/RLinf checkpoint layout (``{"norm_stats": {key: {stat: [...]}}}``),
    where arrays may be padded (e.g. to 32 dims) with zeros.
    """
    norm_stats = reference.get("norm_stats", reference)
    if key not in norm_stats:
        raise KeyError(f"key {key!r} not found in reference stats")
    return {
        stat: np.asarray(values, dtype=np.float64)
        for stat, values in norm_stats[key].items()
        if stat in STAT_NAMES
    }


def compare_stats(
    computed: dict[str, dict[str, list[float]]],
    reference: dict,
    keys: tuple[str, ...] = ("state", "actions"),
) -> dict[str, dict[str, dict]]:
    """Compare computed stats against a reference on the overlapping dims.

    Returns ``{key: {stat: {"max_abs_diff", "n_dims", "computed", "reference"}}}``.
    """
    report: dict[str, dict[str, dict]] = {}
    for key in keys:
        ref = extract_reference_stats(reference, key)
        report[key] = {}
        for stat, values in computed[key].items():
            if stat not in ref:
                continue
            comp = np.asarray(values, dtype=np.float64)
            n = min(comp.shape[-1], ref[stat].shape[-1])
            diff = np.abs(comp[:n] - ref[stat][:n]).max()
            report[key][stat] = {
                "max_abs_diff": float(diff),
                "n_dims": int(n),
                "computed": comp[:n].tolist(),
                "reference": ref[stat][:n].tolist(),
            }
    return report


def print_report(report: dict[str, dict[str, dict]]) -> None:
    for key, stats in report.items():
        print(f"=== {key} ===")
        for stat, info in stats.items():
            print(
                f"  {stat}: max_abs_diff={info['max_abs_diff']:.6g} "
                f"(over {info['n_dims']} dims)"
            )


def load_episode(parquet_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(state, actions)`` arrays from one LeRobot episode parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["state", "actions"])
    state = np.asarray(table.column("state").to_pylist(), dtype=np.float64).reshape(
        len(table), -1
    )
    actions = np.asarray(table.column("actions").to_pylist(), dtype=np.float64).reshape(
        len(table), -1
    )
    return state, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        required=True,
        nargs="+",
        help="One or more episode .parquet files (globs allowed).",
    )
    parser.add_argument(
        "--norm-stats",
        required=True,
        help="Reference norm_stats.json (LeRobot meta/stats.json or a "
        "checkpoint norm_stats.json).",
    )
    parser.add_argument(
        "--extra-delta",
        action="store_true",
        help="Apply openpi's DeltaActions transform before computing stats "
        "(use when comparing against stats generated with "
        "extra_delta_transform=True, e.g. pi0_libero).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="Action chunk length used by the DeltaActions transform.",
    )
    parser.add_argument(
        "--delta-dims",
        type=int,
        default=6,
        help="Number of leading action dims converted to delta actions.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.parquet:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not matches:
            raise FileNotFoundError(f"no parquet files match {pattern!r}")
        paths.extend(Path(p) for p in matches)

    states, actions = [], []
    for path in paths:
        state, action = load_episode(path)
        states.append(state)
        actions.append(action)
        print(f"loaded {path}: state={state.shape} actions={action.shape}")
    state = np.concatenate(states)
    actions = np.concatenate(actions)

    if args.extra_delta:
        actions = apply_delta_actions(
            actions, state, horizon=args.horizon, num_delta_dims=args.delta_dims
        )
        print(
            f"applied DeltaActions on first {args.delta_dims} dims "
            f"(horizon={args.horizon})"
        )

    computed = compute_episode_stats(state, actions)
    with open(args.norm_stats) as f:
        reference = json.load(f)
    report = compare_stats(computed, reference)
    print_report(report)


if __name__ == "__main__":
    main()
