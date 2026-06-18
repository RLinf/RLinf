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
(``meta/info.json`` + ``data/``), re-indexes all episodes and frames globally,
and writes the unified dataset to ``--output-dir``.

Supports both LeRobot **v2.x** and **v3.0** dataset layouts:

* v2.x — ``meta/episodes.jsonl`` + ``meta/tasks.jsonl`` (jsonlines), one parquet
  per episode under ``data/chunk-{C:03d}/episode_{E:06d}.parquet``.
* v3.0 — ``meta/episodes/chunk-{C:03d}/file-{F:03d}.parquet`` +
  ``meta/tasks.parquet``, with multiple episodes packed into
  ``data/chunk-{C:03d}/file-{F:03d}.parquet`` and per-episode row ranges
  recorded in ``dataset_from_index`` / ``dataset_to_index``.

The output dataset uses the same codebase_version as the first discovered
source dataset.  Mixing v2.x and v3.0 inputs is not supported.

Typical usage
-------------
Single run directory
    python toolkits/lerobot/merge_lerobot_datasets.py \\
        --source-dir logs/20260402-16:27:36-maniskill_ppo_cnn/maniskill \\
        --output-dir merged_data

Multiple independent run directories into one dataset
    python toolkits/lerobot/merge_lerobot_datasets.py \\
        --source-dir logs/run_a/maniskill logs/run_b/maniskill \\
        --output-dir merged_data

Dry-run (just print what would be merged, no files written)
    python toolkits/lerobot/merge_lerobot_datasets.py \\
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


def _is_v3_dataset(ds_path: Path) -> bool:
    """Return True iff *ds_path* uses the LeRobot v3.0 layout.

    v3.0 keeps episode metadata under ``meta/episodes/chunk-XXX/file-YYY.parquet``
    and tasks under ``meta/tasks.parquet`` (with multiple episodes packed into
    one data parquet).  v2.x uses jsonlines + one parquet per episode.
    """
    info_path = ds_path / "meta" / "info.json"
    if not info_path.is_file():
        return False
    try:
        with open(info_path) as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    version = str(info.get("codebase_version", ""))
    # info.json itself tells us; double-check the parquet path template too.
    if version.startswith("v3"):
        return True
    data_path_template = str(info.get("data_path", ""))
    return "file-" in data_path_template and "episode_" not in data_path_template


def _merge_v3(
    datasets: list[Path],
    output_path: Path,
    *,
    dry_run: bool,
) -> int:
    """Merge LeRobot v3.0 datasets.

    Each source has:

    * ``meta/info.json`` (codebase_version=v3.0, features, fps, …)
    * ``meta/tasks.parquet`` (columns: ``task_index``, ``task``)
    * ``meta/episodes/chunk-XXX/file-YYY.parquet`` with per-episode metadata
      (``episode_index``, ``tasks``, ``length``, ``data/chunk_index``,
      ``data/file_index``, ``dataset_from_index``, ``dataset_to_index``, plus
      flattened per-feature stats columns).
    * ``data/chunk-XXX/file-YYY.parquet`` containing all frames for the
      episodes recorded in that file (rows are sliced by
      ``dataset_from_index``..``dataset_to_index``).

    The merged output collapses everything into single files
    (``data/chunk-000/file-000.parquet``,
    ``meta/episodes/chunk-000/file-000.parquet``) — fine for small datasets,
    and dwarfed by the upstream chunks_size of 1000 for typical RLinf runs.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    # ---- 1. Walk sources, build global task map and episode list ---------
    reference_info: dict[str, Any] | None = None
    global_tasks: dict[str, int] = {}
    # Each entry: (source_dataset_path, episode_meta_row dict, source_data_parquet_path)
    all_eps: list[tuple[Path, dict, Path]] = []

    for ds_path in datasets:
        info_path = ds_path / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        if reference_info is None:
            reference_info = info

        episodes_dir = ds_path / "meta" / "episodes"
        if not episodes_dir.is_dir():
            print(
                f"[merge] WARNING: missing meta/episodes/ in {ds_path}, skipping",
                file=sys.stderr,
            )
            continue

        ep_files = sorted(episodes_dir.rglob("file-*.parquet"))
        if not ep_files:
            print(
                f"[merge] WARNING: no episode parquet files in {ds_path}, skipping",
                file=sys.stderr,
            )
            continue

        # tasks: align to global mapping by name
        tasks_path = ds_path / "meta" / "tasks.parquet"
        if tasks_path.is_file():
            tasks_df = pq.read_table(tasks_path).to_pandas()
            # task name and integer index may live as columns or as the index
            if "task" in tasks_df.columns and "task_index" in tasks_df.columns:
                for _, row in tasks_df.iterrows():
                    name = str(row["task"])
                    if name not in global_tasks:
                        global_tasks[name] = len(global_tasks)
            else:
                # task as index, task_index as column
                for task_name in tasks_df.index.tolist():
                    name = str(task_name)
                    if name not in global_tasks:
                        global_tasks[name] = len(global_tasks)

        for ep_file in ep_files:
            ep_table = pq.read_table(ep_file)
            ep_df = ep_table.to_pandas()
            data_template = info.get(
                "data_path",
                "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            )
            for _, ep_row in ep_df.iterrows():
                ep_meta = ep_row.to_dict()
                chunk_idx = int(ep_meta.get("data/chunk_index", 0))
                file_idx = int(ep_meta.get("data/file_index", 0))
                data_parquet = ds_path / data_template.format(
                    chunk_index=chunk_idx, file_index=file_idx
                )
                if not data_parquet.is_file():
                    print(
                        f"[merge] WARNING: data file not found {data_parquet}, "
                        f"skipping episode {ep_meta.get('episode_index')}",
                        file=sys.stderr,
                    )
                    continue
                all_eps.append((ds_path, ep_meta, data_parquet))

                # Register every task name this episode lists, in case the
                # tasks.parquet table was missing or incomplete.
                for task_name in ep_meta.get("tasks", []) or []:
                    name = str(task_name)
                    if name not in global_tasks:
                        global_tasks[name] = len(global_tasks)

    total_eps = len(all_eps)
    print(f"[merge:v3] Total episodes to merge: {total_eps}")
    print(
        f"[merge:v3] Unique tasks: {len(global_tasks)} → "
        f"{list(global_tasks.keys())[:5]}"
    )

    if total_eps == 0:
        print("[merge:v3] Nothing to merge.", file=sys.stderr)
        return 0

    if dry_run:
        print("[merge:v3] Dry-run mode — no files written.")
        return total_eps

    # ---- 2. Prepare output dirs ----------------------------------------
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # Cache decoded data parquets so we don't re-read the same source twice.
    data_cache: dict[Path, "pd.DataFrame"] = {}

    def _load_data(p: Path) -> "pd.DataFrame":
        if p not in data_cache:
            data_cache[p] = pq.read_table(p).to_pandas()
        return data_cache[p]

    # ---- 3. Re-index frames and episodes -------------------------------
    out_frames: list["pd.DataFrame"] = []
    out_ep_rows: list[dict] = []
    global_frame_index = 0
    out_data_chunk_index = 0
    out_data_file_index = 0
    out_ep_chunk_index = 0
    out_ep_file_index = 0
    reference_schema_metadata = None

    for new_ep_idx, (ds_path, ep_meta, data_parquet) in enumerate(all_eps):
        src_df = _load_data(data_parquet)
        # Cache the first source's pyarrow schema metadata (HF feature dtypes,
        # image encoding hints, etc.) and reuse it for the merged output.
        if reference_schema_metadata is None:
            reference_schema_metadata = pq.read_table(
                data_parquet, columns=[src_df.columns[0]]
            ).schema.metadata

        from_idx = int(ep_meta.get("dataset_from_index", 0))
        to_idx = int(ep_meta.get("dataset_to_index", from_idx + int(ep_meta.get("length", 0))))
        slice_df = src_df.iloc[from_idx:to_idx].copy()
        n_frames = len(slice_df)
        if n_frames == 0:
            print(
                f"[merge:v3] WARNING: empty slice for episode "
                f"{ep_meta.get('episode_index')} in {data_parquet}, skipping"
            )
            continue

        new_from = global_frame_index
        new_to = global_frame_index + n_frames

        # Per-row reindexing.
        slice_df["episode_index"] = new_ep_idx
        slice_df["index"] = range(new_from, new_to)
        slice_df["frame_index"] = range(0, n_frames)
        # Tasks: map each episode's task list to global indices.  Most lerobot
        # writers only ever attach one task per episode, but the schema allows
        # several — take the first one for the per-frame ``task_index``.
        ep_tasks = ep_meta.get("tasks", []) or []
        first_task = str(ep_tasks[0]) if len(ep_tasks) else ""
        new_task_index = global_tasks.get(first_task, 0) if first_task else 0
        slice_df["task_index"] = new_task_index

        out_frames.append(slice_df)

        # Build the merged episodes row from the source row.
        new_ep_row = dict(ep_meta)
        new_ep_row["episode_index"] = new_ep_idx
        new_ep_row["tasks"] = list(ep_tasks)
        new_ep_row["length"] = int(n_frames)
        new_ep_row["data/chunk_index"] = out_data_chunk_index
        new_ep_row["data/file_index"] = out_data_file_index
        new_ep_row["dataset_from_index"] = new_from
        new_ep_row["dataset_to_index"] = new_to
        new_ep_row["meta/episodes/chunk_index"] = out_ep_chunk_index
        new_ep_row["meta/episodes/file_index"] = out_ep_file_index

        # Adjust the global-index-style stats so they match the new layout.
        frame_offset = new_from - from_idx
        for key_min, key_max, key_mean in (
            ("stats/episode_index/min", "stats/episode_index/max", "stats/episode_index/mean"),
        ):
            if key_min in new_ep_row:
                new_ep_row[key_min] = [float(new_ep_idx)]
                new_ep_row[key_max] = [float(new_ep_idx)]
                new_ep_row[key_mean] = [float(new_ep_idx)]
            if "stats/episode_index/std" in new_ep_row:
                new_ep_row["stats/episode_index/std"] = [0.0]
        # ``index`` stats: shift by frame_offset.  std is preserved (contiguous range).
        for stat_key in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
            full_key = f"stats/index/{stat_key}"
            if full_key in new_ep_row:
                # ``_ensure_list`` may yield numpy arrays from a parquet read;
                # ``.item()`` collapses 0-d / 1-d arrays cleanly without the
                # numpy-1.25 "ndim > 0 → scalar" DeprecationWarning.
                shifted = []
                for v in _ensure_list(new_ep_row[full_key]):
                    if hasattr(v, "item"):
                        v = v.item()
                    shifted.append(float(v) + float(frame_offset))
                new_ep_row[full_key] = shifted
        # ``task_index`` stats: every frame in this episode now points to
        # the new global task id.
        if "stats/task_index/min" in new_ep_row:
            new_ep_row["stats/task_index/min"] = [float(new_task_index)]
            new_ep_row["stats/task_index/max"] = [float(new_task_index)]
            new_ep_row["stats/task_index/mean"] = [float(new_task_index)]
            if "stats/task_index/std" in new_ep_row:
                new_ep_row["stats/task_index/std"] = [0.0]

        out_ep_rows.append(new_ep_row)
        global_frame_index = new_to

        if (new_ep_idx + 1) % 50 == 0 or (new_ep_idx + 1) == total_eps:
            print(
                f"[merge:v3] Processed {new_ep_idx + 1}/{total_eps} episodes "
                f"({global_frame_index} frames)"
            )

    # ---- 4. Write merged data parquet ----------------------------------
    import pandas as pd  # noqa: F811 — re-import so type checkers see the alias here

    merged_data_df = pd.concat(out_frames, ignore_index=True)
    merged_data_table = pa.Table.from_pandas(merged_data_df, preserve_index=False)
    if reference_schema_metadata:
        # Preserve HuggingFace ``datasets`` feature metadata so consumers can
        # decode image columns the same way the source did.
        merged_data_table = merged_data_table.replace_schema_metadata(
            reference_schema_metadata
        )
    pq.write_table(
        merged_data_table,
        output_path / "data" / "chunk-000" / "file-000.parquet",
    )

    # ---- 5. Write merged episodes parquet ------------------------------
    merged_ep_df = pd.DataFrame(out_ep_rows)
    pq.write_table(
        pa.Table.from_pandas(merged_ep_df, preserve_index=False),
        output_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    # ---- 6. Write merged tasks parquet ---------------------------------
    sorted_tasks = sorted(global_tasks.items(), key=lambda x: x[1])
    tasks_df = pd.DataFrame(
        {
            "task": [t for t, _ in sorted_tasks],
            "task_index": [i for _, i in sorted_tasks],
        }
    )
    # The original LeRobot writer keeps task name as the index; mirror that
    # for downstream loaders that read it back via ``set_index``.
    tasks_df = tasks_df.set_index("task")
    pq.write_table(
        pa.Table.from_pandas(tasks_df, preserve_index=True),
        output_path / "meta" / "tasks.parquet",
    )

    # ---- 7. Write merged info.json -------------------------------------
    info_out: dict[str, Any] = dict(reference_info) if reference_info else {}
    info_out["total_episodes"] = total_eps
    info_out["total_frames"] = int(global_frame_index)
    info_out["total_tasks"] = len(global_tasks)
    info_out["splits"] = {"train": f"0:{total_eps}"}
    # We emit a single output data file and a single output episodes file.
    info_out["chunks_size"] = max(int(info_out.get("chunks_size", 1000)), total_eps)
    info_out.setdefault(
        "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    # No videos are produced by the merge — drop video_path if present.
    info_out.pop("video_path", None)

    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info_out, f, indent=4)

    # Carry forward stats.json if all sources agree (or copy the first one's
    # for completeness).  Downstream openpi only needs episode-level stats,
    # which we already wrote above.
    first_stats = datasets[0] / "meta" / "stats.json"
    if first_stats.is_file():
        with open(first_stats) as f:
            stats_json = f.read()
        with open(output_path / "meta" / "stats.json", "w") as f:
            f.write(stats_json)
        print("[merge:v3] Copied stats.json from first source dataset.")

    print(
        f"[merge:v3] Done: {total_eps} episodes, {global_frame_index} frames → "
        f"{output_path}"
    )
    return total_eps


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
    # 1a. Dispatch to the v3.0 path when the (first) source uses that layout
    # ------------------------------------------------------------------
    v3_flags = [_is_v3_dataset(d) for d in datasets]
    if all(v3_flags):
        return _merge_v3(datasets, output_path, dry_run=dry_run)
    if any(v3_flags):
        raise ValueError(
            "Cannot mix LeRobot v2.x and v3.0 datasets in the same merge. "
            f"v3.0 sources: {[str(d) for d, v in zip(datasets, v3_flags) if v]}; "
            f"v2.x sources: {[str(d) for d, v in zip(datasets, v3_flags) if not v]}"
        )

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
