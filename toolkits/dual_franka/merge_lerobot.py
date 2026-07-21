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
"""Merge a multi-id LeRobot dataset into a single-id dataset.

Example:
    python merge_lerobot.py \
        --rank-dir /path/to/dataset \
        --out-dir  /path/to/dataset \
        --log-file /path/to/log \
        --dry-run

Input layout::

    rank-dir/
      ├── id_0/data/chunk-XXX/episode_*.parquet
      ├── id_0/meta/{info,episodes,episodes_stats,tasks}.jsonl|json
      ├── id_1/data/...
      └── id_1/meta/...

Output layout::

    out-dir/
      ├── data/chunk-XXX/episode_*.parquet (renumbered + reindexed globally)
      └── meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl}

The merge proceeds in seven steps; each step prints "before -> after" lines so
the operator can audit the result:

  STEP 0  Pre-flight + cross-id consistency check
          (fps / robot_type / codebase_version / chunks_size / features)
  STEP 1  Globally rename parquet files into a continuous 0..N-1 sequence.
  STEP 2  Rewrite ``episode_index`` and ``index`` columns inside each parquet.
          Huggingface schema metadata is preserved; writes go through ``.tmp``
          followed by ``replace`` for atomicity.
  STEP 3  Update each id's ``episodes.jsonl`` and ``episodes_stats.jsonl``
          (top-level ``episode_index`` plus nested ``stats.episode_index`` and
          ``stats.index``). Other fields (frame_index/state/actions/...) are
          left untouched.
  STEP 4  Build the merged ``info.json`` and ``tasks.jsonl``.
          ``info.json``: only ``total_episodes`` / ``total_frames`` /
          ``total_tasks`` / ``total_videos`` / ``total_chunks`` /
          ``splits.train`` are overwritten; the rest is taken from id_0.
          ``tasks.jsonl``: deduplicated by string, renumbered consecutively.
          A remapping that would require rewriting ``task_index`` columns in
          parquet files is detected and rejected (not supported by this tool).
  STEP 5  Concatenate per-id ``episodes.jsonl`` / ``episodes_stats.jsonl``
          into ``out-dir/meta/``.
  STEP 6  Move parquet files into ``out-dir/data/chunk-{episode_chunk:03d}/``,
          bucketed by ``chunks_size``.
  STEP 7  End-to-end verification of ep/frame/index columns inside parquet,
          jsonl line counts, and seamless ``index`` continuity.

Flags:
  --dry-run     Plan and print everything but do not write any file.
  --log-file    Mirror logs to the given file.
  --skip-verify Skip STEP 7.
"""

import argparse
import json
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from toolkits.dual_franka.utils_lerobot import (
    EP_RE,
    add_log_file,
    find_id_dirs,
    fmt_stats,
    get_toolkit_logger,
    header,
    load_info,
    read_jsonl,
    write_jsonl_atomic,
)

logger = get_toolkit_logger("rlinf.dual_franka_merge_lerobot")

CONSISTENT_TOP_FIELDS: tuple[str, ...] = (
    "codebase_version",
    "robot_type",
    "fps",
    "chunks_size",
    "data_path",
    "video_path",
)


def load_tasks(id_dir: Path) -> list[tuple[int, str]]:
    """Load ``meta/tasks.jsonl`` as a list of ``(task_index, task)`` tuples.

    Args:
        id_dir: A ``id_*`` directory.

    Returns:
        ``[]`` if the file does not exist, otherwise a list of
        ``(task_index, task_string)``.
    """
    p = id_dir / "meta" / "tasks.jsonl"
    if not p.exists():
        return []
    out: list[tuple[int, str]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            out.append((int(o["task_index"]), str(o["task"])))
    return out


def step0_precheck(
    rank_dir: Path,
) -> tuple[list[Path], list[tuple[str, dict[str, Any]]]]:
    """Validate that every per-id directory is well-formed and consistent.

    Args:
        rank_dir: Directory containing one ``id_*`` subdirectory per rank.

    Returns:
        A tuple ``(id_dirs, infos)`` where ``id_dirs`` is the list of
        per-id directories and ``infos`` is a list of
        ``(id_name, info_json_dict)``.

    Raises:
        RuntimeError: If a required directory is missing or any of the
            ``CONSISTENT_TOP_FIELDS`` / ``features`` differ across ids.
    """
    header(logger, "STEP 0  pre-flight + cross-id consistency")
    id_dirs = find_id_dirs(rank_dir)
    if not id_dirs:
        raise RuntimeError(f"No id_* subdirectory found under {rank_dir}")

    infos: list[tuple[str, dict[str, Any]]] = []
    for d in id_dirs:
        if not (d / "data").exists():
            raise RuntimeError(f"{d} is missing data/")
        if not (d / "meta").exists():
            raise RuntimeError(f"{d} is missing meta/")
        info = load_info(d)
        infos.append((d.name, info))
        n_ep = info.get("total_episodes")
        n_fr = info.get("total_frames")
        n_pq = sum(1 for _ in (d / "data").rglob("episode_*.parquet"))
        logger.info(
            "  %s: parquet=%d  info.total_episodes=%s  info.total_frames=%s",
            d.name,
            n_pq,
            n_ep,
            n_fr,
        )

    base_name, base = infos[0]
    for name, info in infos[1:]:
        for f in CONSISTENT_TOP_FIELDS:
            if info.get(f) != base.get(f):
                raise RuntimeError(
                    f"Field {f} differs across ids: "
                    f"{base_name}={base.get(f)!r} vs {name}={info.get(f)!r}"
                )
        bf, tf = base.get("features", {}), info.get("features", {})
        for k in set(bf.keys()) | set(tf.keys()):
            if bf.get(k) != tf.get(k):
                raise RuntimeError(
                    f"features[{k}] differs across ids: {base_name} vs {name}"
                )

    logger.info(
        "  cross-id consistency OK: fps=%s, robot_type=%s, "
        "codebase_version=%s, chunks_size=%s",
        base["fps"],
        base["robot_type"],
        base["codebase_version"],
        base["chunks_size"],
    )
    return id_dirs, infos


def step1_renumber(id_dirs: list[Path], dry_run: bool) -> list[dict[str, Any]]:
    """Plan and execute a global rename of every ``episode_*.parquet`` file.

    The new index is assigned by sorting on ``(id_name, old_episode_index)``,
    yielding a contiguous ``0..N-1`` range across all ids.

    Args:
        id_dirs: Per-id directories returned by :func:`step0_precheck`.
        dry_run: If True, only print the planned moves.

    Returns:
        A list of plan entries, one per parquet, each containing ``new_idx``,
        ``old_idx``, ``id_name``, ``id_dir``, ``old_path``, ``new_path`` and
        ``current_path`` (after rename).
    """
    header(logger, "STEP 1  globally renumber parquet files")
    records: list[tuple[str, int, Path, Path]] = []
    for id_dir in id_dirs:
        for p in (id_dir / "data").rglob("episode_*.parquet"):
            m = EP_RE.match(p.name)
            if m:
                records.append((id_dir.name, int(m.group(1)), p, id_dir))
            else:
                logger.warning(
                    "  Skipping parquet with unexpected name "
                    "(expected episode_<digits>.parquet): %s",
                    p,
                )
    records.sort(key=lambda x: (x[0], x[1]))

    plan: list[dict[str, Any]] = []
    for new_idx, (id_name, old_idx, old_path, id_dir) in enumerate(records):
        new_name = f"episode_{new_idx:06d}.parquet"
        new_path = old_path.parent / new_name
        plan.append(
            {
                "new_idx": new_idx,
                "old_idx": old_idx,
                "id_name": id_name,
                "id_dir": id_dir,
                "old_path": old_path,
                "new_path": new_path,
            }
        )

    n_total = len(plan)
    n_to_rename = sum(1 for e in plan if e["old_path"] != e["new_path"])
    logger.info("  total parquet=%d, to rename=%d", n_total, n_to_rename)
    for e in plan:
        if e["old_path"] != e["new_path"]:
            tag = "[DRY-RUN]" if dry_run else "[RENAME]"
            logger.info(
                "  %s %s/%s -> %s",
                tag,
                e["id_name"],
                e["old_path"].name,
                e["new_path"].name,
            )

    if not dry_run:
        # Use a __tmp__ stop-over so old/new names do not collide.
        tmps: list[tuple[Path, Path]] = []
        for i, e in enumerate(plan):
            if e["old_path"] != e["new_path"]:
                tmp = e["old_path"].with_name(e["old_path"].name + f".__tmp_{i}__")
                e["old_path"].rename(tmp)
                tmps.append((tmp, e["new_path"]))
        for tmp, new in tmps:
            tmp.rename(new)
        for e in plan:
            e["current_path"] = e["new_path"]
    else:
        for e in plan:
            e["current_path"] = e["old_path"]

    return plan


def step2_change_index(
    plan: list[dict[str, Any]], dry_run: bool
) -> tuple[list[dict[str, Any]], int]:
    """Rewrite ``episode_index`` and ``index`` columns of every parquet.

    ``index`` is made globally contiguous by accumulating ``num_rows`` across
    parquet files in the planned order; ``episode_index`` is set to the new
    per-file index. Huggingface schema metadata is verified to survive the
    rewrite.

    Args:
        plan: Output of :func:`step1_renumber`.
        dry_run: If True, plan only — no parquet is rewritten.

    Returns:
        A tuple ``(mapping, total_frames)`` where ``mapping`` records, for
        each parquet, ``new_idx``, ``old_idx``, ``id_dir``, ``id_name``,
        ``current_path``, ``new_path``, ``start``, ``end`` and ``n``.

    Raises:
        RuntimeError: If ``new_idx`` is not contiguous, or a required column
            is missing from a parquet.
    """
    header(logger, "STEP 2  rewrite episode_index + index inside parquet")

    new_idxs = [e["new_idx"] for e in plan]
    if new_idxs != list(range(len(new_idxs))):
        raise RuntimeError(f"new_idx is not contiguous 0..N-1, got {new_idxs[:15]}...")

    running = 0
    mapping: list[dict[str, Any]] = []
    for e in plan:
        cur: Path = e["current_path"]
        t = pq.read_table(cur)
        n = t.num_rows
        for col in ("episode_index", "index"):
            if col not in t.column_names:
                raise RuntimeError(f"{cur} is missing column {col}")

        old_ep = sorted(set(t.column("episode_index").combine_chunks().to_pylist()))
        old_ix = t.column("index").combine_chunks().to_pylist()
        old_ix_first, old_ix_last = old_ix[0], old_ix[-1]

        new_idx = e["new_idx"]
        ep_arr = pa.array(np.full(n, new_idx, dtype=np.int64), type=pa.int64())
        idx_arr = pa.array(
            np.arange(running, running + n, dtype=np.int64), type=pa.int64()
        )

        ci_ep = t.column_names.index("episode_index")
        t = t.set_column(ci_ep, t.schema.field(ci_ep), ep_arr)
        ci_ix = t.column_names.index("index")
        t = t.set_column(ci_ix, t.schema.field(ci_ix), idx_arr)

        if not dry_run:
            tmp = cur.with_suffix(".parquet.tmp")
            pq.write_table(t, tmp, compression="snappy")
            tmp.replace(cur)
            chk = pq.read_table(cur, columns=["episode_index", "index"])
            assert set(chk.column("episode_index").to_pylist()) == {new_idx}, (
                f"{cur}: episode_index write-back mismatch"
            )
            assert chk.column("index").to_pylist() == list(
                range(running, running + n)
            ), f"{cur}: index write-back mismatch"
            md = chk.schema.metadata or {}
            assert b"huggingface" in md, f"{cur}: huggingface metadata lost"

        tag = "[DRY-RUN]" if dry_run else "[WRITE]"
        logger.info(
            "  %s ep_idx: %s -> %d   index: [%d..%d] -> [%d..%d]   rows=%d   %s/%s",
            tag,
            old_ep,
            new_idx,
            old_ix_first,
            old_ix_last,
            running,
            running + n - 1,
            n,
            e["id_name"],
            cur.name,
        )

        mapping.append(
            {
                "new_idx": new_idx,
                "old_idx": e["old_idx"],
                "id_dir": e["id_dir"],
                "id_name": e["id_name"],
                "current_path": cur,
                "new_path": e["new_path"],
                "start": running,
                "end": running + n - 1,
                "n": n,
            }
        )
        running += n

    logger.info("\n  cumulative total_frames = %d", running)
    return mapping, running


def step3_update_stats(
    id_dirs: list[Path],
    mapping: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    """Rewrite per-id ``episodes.jsonl`` and ``episodes_stats.jsonl``.

    Top-level ``episode_index`` and the nested ``stats.episode_index`` /
    ``stats.index`` are updated. ``frame_index``, sensor stats and timestamps
    are left untouched.

    Args:
        id_dirs: Per-id directories.
        mapping: Output of :func:`step2_change_index`.
        dry_run: If True, plan only — no file is written.

    Raises:
        RuntimeError: If line counts do not match the mapping, recorded
            ``length`` disagrees with rewritten ``n``, or required nested
            stats fields are missing.
    """
    header(logger, "STEP 3  per-id episodes.jsonl + episodes_stats.jsonl")
    groups: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for m in mapping:
        groups[m["id_dir"]].append(m)
    for k in groups:
        groups[k].sort(key=lambda x: x["new_idx"])

    for id_dir in id_dirs:
        entries = groups.get(id_dir, [])
        if not entries:
            logger.info("  -- %s -- (not in mapping, skipped)", id_dir.name)
            continue
        logger.info("  -- %s --", id_dir.name)

        ep_path = id_dir / "meta" / "episodes.jsonl"
        if not ep_path.exists():
            logger.info("    [SKIP] %s does not exist", ep_path)
        else:
            lines = read_jsonl(ep_path)
            if len(lines) != len(entries):
                raise RuntimeError(
                    f"{ep_path} line count={len(lines)} != mapping {len(entries)}"
                )
            ep_by_idx = {obj["episode_index"]: obj for obj in lines}
            new_lines: list[dict[str, Any]] = []
            for m in entries:
                old_idx = m["old_idx"]
                obj = ep_by_idx.get(old_idx)
                if obj is None:
                    raise RuntimeError(
                        f"episode_index={old_idx} not found in {ep_path}"
                    )
                old_top = obj.get("episode_index")
                length = obj.get("length")
                if length is not None and int(length) != int(m["n"]):
                    raise RuntimeError(
                        f"{ep_path} ep={old_top} length={length} != mapping n={m['n']}"
                    )
                obj = dict(obj)
                obj["episode_index"] = int(m["new_idx"])
                new_lines.append(obj)
                logger.info(
                    "    episodes.jsonl       ep_idx: %s -> %d   length=%s",
                    old_top,
                    m["new_idx"],
                    length,
                )
            if not dry_run:
                write_jsonl_atomic(ep_path, new_lines)
                logger.info("    [WRITE] %s  (%d rows)", ep_path, len(new_lines))

        st_path = id_dir / "meta" / "episodes_stats.jsonl"
        if not st_path.exists():
            logger.info("    [SKIP] %s does not exist", st_path)
        else:
            lines = read_jsonl(st_path)
            if len(lines) != len(entries):
                raise RuntimeError(
                    f"{st_path} line count={len(lines)} != mapping {len(entries)}"
                )
            st_by_idx = {obj["episode_index"]: obj for obj in lines}
            new_lines = []
            for m in entries:
                old_idx = m["old_idx"]
                obj = st_by_idx.get(old_idx)
                if obj is None:
                    raise RuntimeError(
                        f"episode_index={old_idx} not found in {st_path}"
                    )
                old_top = obj.get("episode_index")
                stats = dict(obj.get("stats", {}))
                if "episode_index" not in stats or "index" not in stats:
                    raise RuntimeError(
                        f"{st_path}: row missing stats.episode_index or stats.index"
                    )
                old_ep_s = dict(stats["episode_index"])
                old_ix_s = dict(stats["index"])
                new_idx = m["new_idx"]
                start = m["start"]
                end = m["end"]
                n = m["n"]
                stats["episode_index"] = {
                    "min": [int(new_idx)],
                    "max": [int(new_idx)],
                    "mean": [float(new_idx)],
                    "std": [0.0],
                    "count": [int(n)],
                }
                stats["index"] = {
                    "min": [int(start)],
                    "max": [int(end)],
                    "mean": [float(start + end) / 2.0],
                    # Translation does not change std.
                    "std": old_ix_s.get("std", [0.0]),
                    "count": [int(n)],
                }
                obj = dict(obj)
                obj["stats"] = stats
                obj["episode_index"] = int(new_idx)
                new_lines.append(obj)
                logger.info(
                    "    episodes_stats.jsonl ep_idx: %s -> %d", old_top, new_idx
                )
                logger.info(
                    "      stats.episode_index: %s -> %s",
                    fmt_stats(old_ep_s),
                    fmt_stats(stats["episode_index"]),
                )
                logger.info(
                    "      stats.index        : %s -> %s",
                    fmt_stats(old_ix_s),
                    fmt_stats(stats["index"]),
                )
            if not dry_run:
                write_jsonl_atomic(st_path, new_lines)
                logger.info("    [WRITE] %s  (%d rows)", st_path, len(new_lines))


def step4_info_and_tasks(
    id_dirs: list[Path],
    infos: list[tuple[str, dict[str, Any]]],
    mapping: list[dict[str, Any]],
    total_frames: int,
    out_meta: Path,
    dry_run: bool,
) -> None:
    """Build the merged ``info.json`` and ``tasks.jsonl``.

    ``info.json`` reuses every field from id_0 except ``total_episodes``,
    ``total_frames``, ``total_tasks``, ``total_videos``, ``total_chunks`` and
    ``splits.train``, which are recomputed.

    ``tasks.jsonl`` is the deduplicated union of every per-id ``tasks.jsonl``,
    renumbered ``0..M-1``. If deduplication would change any per-id
    ``task_index`` (i.e. parquet rows would need their ``task_index`` column
    rewritten), this function refuses to proceed.

    Args:
        id_dirs: Per-id directories.
        infos: Output of :func:`step0_precheck`.
        mapping: Output of :func:`step2_change_index`.
        total_frames: Total number of frames across all ids.
        out_meta: Destination ``meta/`` directory.
        dry_run: If True, plan only — no file is written.

    Raises:
        RuntimeError: If task remapping would require rewriting parquet
            ``task_index`` columns.
    """
    header(logger, "STEP 4  build merged info.json + tasks.jsonl")
    all_tasks = {d.name: load_tasks(d) for d in id_dirs}
    seen: dict[str, int] = {}
    order: list[str] = []
    for id_name in sorted(all_tasks):
        for _, ts in all_tasks[id_name]:
            if ts not in seen:
                seen[ts] = len(order)
                order.append(ts)
    merged_tasks = list(enumerate(order))
    total_tasks = len(merged_tasks)

    total_episodes = len(mapping)
    chunks_size = int(infos[0][1].get("chunks_size", 1000))
    total_chunks = math.ceil(total_episodes / chunks_size) if total_episodes else 0
    total_videos = sum(int(i.get("total_videos", 0)) for _, i in infos)

    template_name, template_info = infos[0]
    merged_info = dict(template_info)
    overrides: dict[str, Any] = {
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "splits": {"train": f"0:{total_episodes}"},
    }
    merged_info.update(overrides)

    logger.info("  template from: %s/meta/info.json", template_name)
    for key in ("total_episodes", "total_frames", "total_videos"):
        per = [int(i.get(key, 0)) for _, i in infos]
        logger.info("  %-16s: %s = %s", key, " + ".join(map(str, per)), overrides[key])
    logger.info("  %-16s: deduped = %s", "total_tasks", overrides["total_tasks"])
    logger.info(
        "  %-16s: ceil(%d/%d) = %s",
        "total_chunks",
        total_episodes,
        chunks_size,
        overrides["total_chunks"],
    )
    logger.info("  %-16s: -> %s", "splits", overrides["splits"])

    needs_remap = False
    for id_name, tasks in all_tasks.items():
        for old_ti, ts in tasks:
            if seen[ts] != old_ti:
                needs_remap = True
                logger.warning(
                    "  WARN %s: old task_index=%d -> new=%d  task=%r",
                    id_name,
                    old_ti,
                    seen[ts],
                    ts[:40] + "...",
                )
    if needs_remap:
        raise RuntimeError(
            "task_index remapping is required after merging, which would mean "
            "rewriting the task_index column inside parquet files; this script "
            "does not support that path. Please pre-align task_index manually "
            "or extend the script."
        )
    logger.info("  merged tasks (%d):", total_tasks)
    for ti, t in merged_tasks:
        preview = t if len(t) <= 70 else (t[:67] + "...")
        logger.info('    task_index=%d  "%s"', ti, preview)

    if not dry_run:
        out_meta.mkdir(parents=True, exist_ok=True)
        info_out = out_meta / "info.json"
        tmp = info_out.with_suffix(info_out.suffix + ".tmp")
        tmp.write_text(
            json.dumps(merged_info, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(info_out)
        logger.info("  [WRITE] %s", info_out)

        tasks_out = out_meta / "tasks.jsonl"
        write_jsonl_atomic(
            tasks_out,
            [{"task_index": i, "task": t} for i, t in merged_tasks],
        )
        logger.info("  [WRITE] %s", tasks_out)


def step5_merge_jsonl(id_dirs: list[Path], out_meta: Path, dry_run: bool) -> None:
    """Concatenate per-id ``episodes.jsonl`` / ``episodes_stats.jsonl``.

    Rows are sorted by ``episode_index`` to keep the output stable.

    Args:
        id_dirs: Per-id directories.
        out_meta: Destination ``meta/`` directory.
        dry_run: If True, plan only — no file is written.
    """
    header(logger, "STEP 5  merge per-id episodes.jsonl / episodes_stats.jsonl")
    for fname in ("episodes.jsonl", "episodes_stats.jsonl"):
        merged: list[dict[str, Any]] = []
        for id_dir in id_dirs:
            p = id_dir / "meta" / fname
            if not p.exists():
                logger.info("  [MISS] %s/%s", id_dir.name, fname)
                continue
            rows = read_jsonl(p)
            logger.info("  [OK]   %s/%s: %d", id_dir.name, fname, len(rows))
            merged.extend(rows)
        if merged and isinstance(merged[0], dict) and "episode_index" in merged[0]:
            merged.sort(key=lambda x: x.get("episode_index", 1 << 60))
        out = out_meta / fname
        if not dry_run:
            write_jsonl_atomic(out, merged)
            logger.info("  [WRITE] %s  (%d rows)", out, len(merged))
        else:
            logger.info("  [DRY-RUN] would write %s  (%d rows)", out, len(merged))


def step6_move_parquets(
    mapping: list[dict[str, Any]],
    out_data_root: Path,
    chunks_size: int,
    dry_run: bool,
) -> None:
    """Move every parquet into its destination ``chunk-XXX`` directory.

    Args:
        mapping: Output of :func:`step2_change_index`.
        out_data_root: Destination ``data/`` directory.
        chunks_size: Chunk capacity from ``info.json``.
        dry_run: If True, plan only — no file is moved.
    """
    header(
        logger, f"STEP 6  move parquet to {out_data_root} (chunks_size={chunks_size})"
    )
    moved = 0
    skipped = 0
    for m in mapping:
        new_idx: int = m["new_idx"]
        chunk = new_idx // chunks_size
        dst_dir = out_data_root / f"chunk-{chunk:03d}"
        # During dry-run ``current_path`` is still the old path; otherwise
        # it equals ``new_path`` after STEP 1.
        src: Path = m["current_path"]
        dst = dst_dir / f"episode_{new_idx:06d}.parquet"

        if dst.exists():
            logger.info("  [SKIP] %s already exists", dst)
            skipped += 1
            continue
        tag = "[DRY-RUN]" if dry_run else "[MOVE]"
        logger.info("  %s %s -> %s", tag, src, dst)
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        moved += 1
    logger.info("  done: moved=%d, skipped=%d", moved, skipped)


def step7_verify(out_dir: Path) -> None:
    """End-to-end sanity check on the merged output directory.

    Args:
        out_dir: Merged dataset root (parent of ``data/`` and ``meta/``).

    Raises:
        RuntimeError: If any of the structural invariants is violated, any
            parquet has wrong ``episode_index`` / ``frame_index`` / ``index``
            values, or ``stats.index`` is not contiguous across episodes.
    """
    header(logger, "STEP 7  end-to-end verification of merged directory")
    info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    total_frames: int = info["total_frames"]
    total_episodes: int = info["total_episodes"]

    parquet_files: list[Path] = []
    for p in (out_dir / "data").rglob("episode_*.parquet"):
        m = EP_RE.match(p.name)
        if m:
            parquet_files.append(p)
        else:
            logger.warning(
                "  Skipping parquet with unexpected name "
                "(expected episode_<digits>.parquet): %s",
                p,
            )
    parquet_files.sort(key=lambda p: int(EP_RE.match(p.name).group(1)))
    logger.info(
        "  parquet files=%d, info.total_episodes=%d",
        len(parquet_files),
        total_episodes,
    )
    assert len(parquet_files) == total_episodes, (
        f"parquet count {len(parquet_files)} != total_episodes {total_episodes}"
    )

    running = 0
    bad = False
    for p in parquet_files:
        t = pq.read_table(p, columns=["episode_index", "frame_index", "index"])
        n = t.num_rows
        ei = t.column("episode_index").to_pylist()
        fi = t.column("frame_index").to_pylist()
        ix = t.column("index").to_pylist()
        m = EP_RE.match(p.name)
        assert m is not None, f"{p.name} does not match episode_<digits>.parquet"
        expect_ei = int(m.group(1))
        ok_ei = set(ei) == {expect_ei}
        ok_fi = fi == list(range(n))
        ok_ix = ix == list(range(running, running + n))
        if not (ok_ei and ok_fi and ok_ix):
            logger.error(
                "    FAIL %s  ep=%s (expected %d)  index=[%d..%d] (expected [%d..%d])",
                p.name,
                set(ei),
                expect_ei,
                ix[0],
                ix[-1],
                running,
                running + n - 1,
            )
            bad = True
        running += n
    if bad:
        raise RuntimeError("parquet verification failed")
    assert running == total_frames, (
        f"cumulative frames {running} != total_frames {total_frames}"
    )
    logger.info(
        "  parquet: episode_index/frame_index/index all correct (sum %d = total_frames)",
        running,
    )

    ep = read_jsonl(out_dir / "meta" / "episodes.jsonl")
    es = read_jsonl(out_dir / "meta" / "episodes_stats.jsonl")
    assert len(ep) == total_episodes, (
        f"episodes.jsonl line count {len(ep)} != total_episodes {total_episodes}"
    )
    assert len(es) == total_episodes, (
        f"episodes_stats.jsonl line count {len(es)} != total_episodes {total_episodes}"
    )
    logger.info("  episodes.jsonl       rows=%d OK", len(ep))
    logger.info("  episodes_stats.jsonl rows=%d OK", len(es))

    prev_max = -1
    for r in sorted(es, key=lambda x: x["episode_index"]):
        mn = r["stats"]["index"]["min"][0]
        mx = r["stats"]["index"]["max"][0]
        if mn != prev_max + 1:
            raise RuntimeError(
                f"stats.index discontinuous at ep {r['episode_index']}: "
                f"min={mn}, expected {prev_max + 1}"
            )
        prev_max = mx
    assert prev_max + 1 == total_frames, (
        f"stats.index final {prev_max + 1} != total_frames {total_frames}"
    )
    logger.info("  stats.index continuous 0..%d OK", prev_max)
    expected_split = f"0:{total_episodes}"
    assert info["splits"]["train"] == expected_split, (
        f"splits.train={info['splits']['train']!r} != expected {expected_split!r}"
    )
    logger.info("  splits.train=%s OK", info["splits"]["train"])
    logger.info("\n  >>> merged directory passed full end-to-end verification <<<")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="One-shot LeRobot multi-id dataset merger."
    )
    ap.add_argument(
        "--rank-dir", required=True, help="Source directory containing id_X subdirs."
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory; data/ and meta/ will be created.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Plan only; do not write files."
    )
    ap.add_argument("--log-file", default=None, help="Mirror logs to the given file.")
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the STEP 7 end-to-end verification.",
    )
    return ap.parse_args()


def main() -> None:
    """Drive the seven merge steps end-to-end."""
    args = parse_args()
    file_handler: Optional[logging.FileHandler] = None
    if args.log_file:
        file_handler = add_log_file(logger, args.log_file)

    try:
        rank_dir = Path(args.rank_dir).resolve()
        out_dir = Path(args.out_dir).resolve()

        logger.info("rank-dir = %s", rank_dir)
        logger.info("out-dir  = %s", out_dir)
        logger.info("dry-run  = %s", args.dry_run)
        if args.log_file:
            logger.info("log-file = %s", Path(args.log_file).resolve())

        id_dirs, infos = step0_precheck(rank_dir)
        plan = step1_renumber(id_dirs, args.dry_run)
        mapping, total_frames = step2_change_index(plan, args.dry_run)
        step3_update_stats(id_dirs, mapping, args.dry_run)

        out_meta = out_dir / "meta"
        out_data = out_dir / "data"
        chunks_size = int(infos[0][1].get("chunks_size", 1000))

        step4_info_and_tasks(
            id_dirs, infos, mapping, total_frames, out_meta, args.dry_run
        )
        step5_merge_jsonl(id_dirs, out_meta, args.dry_run)
        step6_move_parquets(mapping, out_data, chunks_size, args.dry_run)

        if not args.dry_run and not args.skip_verify:
            step7_verify(out_dir)

        header(
            logger,
            "all done" if not args.dry_run else "dry-run finished (nothing written)",
        )
    finally:
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()


if __name__ == "__main__":
    main()
