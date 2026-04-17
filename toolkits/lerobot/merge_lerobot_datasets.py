#!/usr/bin/env python3
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

"""Merge all LeRobot datasets found under a directory into a single dataset.

Recursively discovers every sub-directory that contains a valid LeRobot layout
(``meta/info.json`` + ``meta/episodes.jsonl`` + ``data/``), re-indexes all
episodes and frames globally, and writes the unified dataset to ``--output-dir``.

Typical usage
-------------
Single run directory
    python toolkits/replay_buffer/merge_lerobot_datasets.py \\
        --source-dir logs/20260402-16:27:36-maniskill_ppo_cnn/maniskill \\
        --output-dir merged_data

Multiple independent run directories into one dataset
    python toolkits/replay_buffer/merge_lerobot_datasets.py \\
        --source-dir logs/run_a/maniskill logs/run_b/maniskill \\
        --output-dir merged_data

Dry-run (just print what would be merged, no files written)
    python toolkits/replay_buffer/merge_lerobot_datasets.py \\
        --source-dir logs/20260402-16:27:36-maniskill_ppo_cnn/maniskill \\
        --output-dir merged_data --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _is_lerobot_dataset(path: Path) -> bool:
    """Return True if *path* looks like a LeRobot dataset root."""
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def _discover_datasets(roots: list[Path]) -> list[Path]:
    """Walk each root and return every LeRobot dataset directory found.

    A directory is considered a dataset iff it contains ``meta/info.json``
    and a ``data/`` sub-directory.  The search is recursive so it handles
    layouts like ``rank_0/id_0/``, ``rank_0/id_64/``, etc.

    The root itself is included if it is a valid dataset.
    """
    found: list[Path] = []
    seen: set[Path] = set()

    def _walk(p: Path) -> None:
        if p in seen:
            return
        seen.add(p)
        if _is_lerobot_dataset(p):
            found.append(p)
            # Don't recurse into sub-datasets of a dataset (data/ / meta/ are
            # leaf dirs, not nested datasets).
            return
        try:
            for child in sorted(p.iterdir()):
                if child.is_dir():
                    _walk(child)
        except PermissionError:
            pass

    for root in roots:
        _walk(root)

    return found


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _reindex_episode_stats(
    stats: dict,
    *,
    new_ep_idx: int,
    new_frame_start: int,
    old_frame_start: int,
) -> dict:
    """Return a copy of *stats* with episode_index and index fields updated.

    The per-feature statistics (min/max/mean/std) for ``episode_index`` and
    ``index`` (global frame index) need to reflect the new global positions
    after re-indexing.  All other feature stats are carried over unchanged.

    Args:
        stats: Original stats dict from the source ``episodes_stats.jsonl`` record.
        new_ep_idx: New global episode index assigned to this episode.
        new_frame_start: First global frame index in the merged dataset.
        old_frame_start: First global frame index in the source dataset.

    Returns:
        Updated stats dict.
    """
    import copy

    out = copy.deepcopy(stats)
    frame_offset = new_frame_start - old_frame_start

    # episode_index: all frames share the same constant new_ep_idx
    if "episode_index" in out:
        ep_s = out["episode_index"]
        count = ep_s.get("count", [1])
        out["episode_index"] = {
            "min": [new_ep_idx],
            "max": [new_ep_idx],
            "mean": [float(new_ep_idx)],
            "std": [0.0],
            "count": count,
        }

    # index (global frame index): shift by offset; std is unchanged (contiguous range)
    if "index" in out:
        idx_s = out["index"]
        out["index"] = {
            "min": [v + frame_offset for v in _ensure_list(idx_s.get("min", [0]))],
            "max": [v + frame_offset for v in _ensure_list(idx_s.get("max", [0]))],
            "mean": [v + frame_offset for v in _ensure_list(idx_s.get("mean", [0.0]))],
            "std": idx_s.get("std", [0.0]),
            "count": idx_s.get("count", [1]),
        }

    return out


def _ensure_list(value: object) -> list:
    """Wrap a scalar in a list if it is not already a list."""
    return value if isinstance(value, list) else [value]


def merge_lerobot_datasets(
    source_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    dry_run: bool = False,
) -> int:
    """Merge all LeRobot datasets discovered under *source_dirs* into *output_dir*.

    Args:
        source_dirs: One or more root directories to search recursively.
        output_dir: Destination for the merged dataset.
        dry_run: If True, only print what would be done without writing any files.

    Returns:
        Total number of merged episodes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    roots = [Path(d) for d in source_dirs]
    output_path = Path(output_dir)

    # ------------------------------------------------------------------
    # 1. Discover all sub-datasets
    # ------------------------------------------------------------------
    datasets = _discover_datasets(roots)
    if not datasets:
        print(
            f"[merge] No LeRobot datasets found under: {[str(r) for r in roots]}",
            file=sys.stderr,
        )
        return 0

    print(f"[merge] Found {len(datasets)} dataset(s):")
    for d in datasets:
        print(f"  {d}")

    # ------------------------------------------------------------------
    # 2. Collect all episodes across datasets
    # ------------------------------------------------------------------
    # Each entry: (dataset_path, episode_meta, parquet_path)
    all_episodes: list[tuple[Path, dict, Path]] = []
    # Global task name → global task index
    global_tasks: dict[str, int] = {}
    reference_info: dict[str, Any] | None = None

    # episode_index (within dataset) → stats record, keyed by (ds_path, ep_idx)
    source_episode_stats: dict[tuple[Path, int], dict] = {}

    for ds_path in datasets:
        info_path = ds_path / "meta" / "info.json"
        episodes_path = ds_path / "meta" / "episodes.jsonl"

        if not episodes_path.is_file():
            print(f"[merge] WARNING: missing episodes.jsonl in {ds_path}, skipping")
            continue

        with open(info_path) as f:
            info = json.load(f)
        if reference_info is None:
            reference_info = info

        episodes = _read_jsonl(episodes_path)
        chunks_size: int = info.get("chunks_size", 1000)

        # Load per-episode stats if present
        ep_stats_path = ds_path / "meta" / "episodes_stats.jsonl"
        if ep_stats_path.is_file():
            for rec in _read_jsonl(ep_stats_path):
                source_episode_stats[(ds_path, rec["episode_index"])] = rec

        for ep_meta in episodes:
            ep_idx: int = ep_meta["episode_index"]
            chunk_idx = ep_idx // chunks_size
            parquet_path = (
                ds_path
                / "data"
                / f"chunk-{chunk_idx:03d}"
                / f"episode_{ep_idx:06d}.parquet"
            )
            if not parquet_path.is_file():
                print(
                    f"[merge] WARNING: parquet not found: {parquet_path}, skipping episode"
                )
                continue

            all_episodes.append((ds_path, ep_meta, parquet_path))

            for task in ep_meta.get("tasks", []):
                if task not in global_tasks:
                    global_tasks[task] = len(global_tasks)

    total_episodes = len(all_episodes)
    print(f"[merge] Total episodes to merge: {total_episodes}")
    print(
        f"[merge] Unique tasks: {len(global_tasks)} → {list(global_tasks.keys())[:5]}"
    )

    if total_episodes == 0:
        print("[merge] Nothing to merge.", file=sys.stderr)
        return 0

    if dry_run:
        print("[merge] Dry-run mode — no files written.")
        return total_episodes

    # ------------------------------------------------------------------
    # 3. Prepare output directories
    # ------------------------------------------------------------------
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Re-index and write parquet files
    # ------------------------------------------------------------------
    global_frame_index = 0
    merged_episode_metas: list[dict] = []
    merged_episode_stats: list[dict] = []

    for new_ep_idx, (ds_path, ep_meta, parquet_path) in enumerate(all_episodes):
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        n_frames = len(df)

        old_ep_idx: int = ep_meta["episode_index"]
        old_frame_start = int(df["index"].min()) if "index" in df.columns else 0

        # Update index columns
        df["episode_index"] = new_ep_idx
        df["index"] = range(global_frame_index, global_frame_index + n_frames)

        task = ep_meta.get("tasks", ["unknown task"])[0]
        df["task_index"] = global_tasks.get(task, 0)

        # Determine output chunk
        output_chunks_size = 1000
        chunk_idx = new_ep_idx // output_chunks_size
        chunk_dir = output_path / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        out_parquet = chunk_dir / f"episode_{new_ep_idx:06d}.parquet"

        # Rebuild table preserving original schema metadata
        new_table = pa.Table.from_pandas(df, preserve_index=False)
        # Carry over any existing schema-level metadata (e.g. HuggingFace tags)
        if table.schema.metadata:
            new_schema = new_table.schema.with_metadata(table.schema.metadata)
            new_table = new_table.cast(new_schema)
        pq.write_table(new_table, out_parquet)

        merged_episode_metas.append(
            {
                "episode_index": new_ep_idx,
                "tasks": ep_meta.get("tasks", ["unknown task"]),
                "length": n_frames,
                **{
                    k: ep_meta[k]
                    for k in ep_meta
                    if k not in {"episode_index", "tasks", "length"}
                },
            }
        )

        # Re-index episode stats if available
        src_stats_rec = source_episode_stats.get((ds_path, old_ep_idx))
        if src_stats_rec is not None:
            new_stats = _reindex_episode_stats(
                src_stats_rec["stats"],
                new_ep_idx=new_ep_idx,
                new_frame_start=global_frame_index,
                old_frame_start=old_frame_start,
            )
            merged_episode_stats.append(
                {"episode_index": new_ep_idx, "stats": new_stats}
            )

        global_frame_index += n_frames

        if (new_ep_idx + 1) % 50 == 0 or (new_ep_idx + 1) == total_episodes:
            print(f"[merge] Processed {new_ep_idx + 1}/{total_episodes} episodes …")

    # ------------------------------------------------------------------
    # 5. Write meta files
    # ------------------------------------------------------------------
    total_chunks = (total_episodes + output_chunks_size - 1) // output_chunks_size

    # Build info.json from reference, patching totals/splits
    info_out: dict[str, Any] = dict(reference_info) if reference_info else {}
    info_out.update(
        {
            "total_episodes": total_episodes,
            "total_frames": global_frame_index,
            "total_tasks": len(global_tasks),
            "total_videos": 0,
            "total_chunks": max(1, total_chunks),
            "chunks_size": output_chunks_size,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        }
    )
    # Remove video_path if no videos were written
    info_out.pop("video_path", None)

    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info_out, f, indent=4)

    _write_jsonl(output_path / "meta" / "episodes.jsonl", merged_episode_metas)

    sorted_tasks = sorted(global_tasks.items(), key=lambda x: x[1])
    _write_jsonl(
        output_path / "meta" / "tasks.jsonl",
        [{"task_index": idx, "task": task} for task, idx in sorted_tasks],
    )

    if merged_episode_stats:
        _write_jsonl(
            output_path / "meta" / "episodes_stats.jsonl", merged_episode_stats
        )
        print(
            f"[merge] Written episodes_stats.jsonl ({len(merged_episode_stats)} records)"
        )
    else:
        print("[merge] No episodes_stats.jsonl found in source datasets; skipping.")

    print(
        f"[merge] Done: {total_episodes} episodes, {global_frame_index} frames "
        f"→ {output_path}"
    )
    return total_episodes


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge all LeRobot datasets under one or more directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source-dir",
        nargs="+",
        required=True,
        metavar="DIR",
        help=(
            "Root directory (or directories) to search for LeRobot datasets. "
            "All sub-directories with a valid meta/info.json are included."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Output directory for the merged dataset.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered datasets and episode counts without writing any files.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    n = merge_lerobot_datasets(
        source_dirs=args.source_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
