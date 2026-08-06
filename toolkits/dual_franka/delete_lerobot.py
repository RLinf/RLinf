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
"""Delete specified episodes and renumber remaining data.

This script deletes specified episodes from LeRobot datasets and renumbers
the remaining episodes to ensure continuity across all indices.

Example usage:
    # Preview what would be deleted (no files are written)
    python delete_lerobot.py \
        --data-dir /path/to/dataset/rank_0 \
        --delete "id_0:3,5" \
        --dry-run

    # Actually delete episodes 3 and 5 from id_0
    python delete_lerobot.py \
        --data-dir /path/to/dataset/rank_0 \
        --delete "id_0:3,5"

    # Delete episodes from multiple ids
    python delete_lerobot.py \
        --data-dir /path/to/dataset \
        --delete "id_0:3,5" "id_34:0,2" \
        --log-file /tmp/delete.log

    # Skip final verification
    python delete_lerobot.py \
        --data-dir /path/to/dataset \
        --delete "id_0:3" \
        --skip-verify

Deletion format:
    id_name:episode_index[,episode_index,...] - Episode indices are the original
    numbering before deletion (filename episode_XXXXXX.parquet numbers).

Execution steps:
    STEP 0: Precheck directories, parse deletion list, confirm target files exist
    STEP 1: Delete parquet files
    STEP 2: Rename remaining parquets (continuous numbering) and rewrite
            episode_index / index columns
    STEP 3: Rewrite episodes.jsonl and episodes_stats.jsonl
    STEP 4: Update info.json (total_episodes, total_frames, total_chunks, splits)
    STEP 5: End-to-end verification

Options:
    --dry-run: Plan and print everything but do not write any file.
    --log-file: Mirror logs to the given file.
    --skip-verify: Skip STEP 5 verification.
"""

import argparse
import json
import logging
import math
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
    read_jsonl,
    write_jsonl_atomic,
)

logger = get_toolkit_logger("rlinf.dual_franka_delete_lerobot")


def parse_delete_args(delete_args: list[str]) -> dict[str, set[int]]:
    """Parse --delete command-line arguments.

    Args:
        delete_args: List of deletion specifications in format "id_name:ep1,ep2,..."

    Returns:
        Dictionary mapping id_name to set of episode indices to delete.

    Raises:
        ValueError: If deletion specification format is invalid.
    """
    result: dict[str, set[int]] = {}
    for item in delete_args:
        if ":" not in item:
            raise ValueError(f"Invalid --delete format (missing colon): {item!r}")
        id_name, nums = item.split(":", 1)
        id_name = id_name.strip()
        indices: set[int] = set()
        for n in nums.split(","):
            n = n.strip()
            if n:
                try:
                    indices.add(int(n))
                except ValueError as e:
                    raise ValueError(f"Invalid episode number in {item}: {n}") from e
        if id_name in result:
            result[id_name] |= indices
        else:
            result[id_name] = indices
    return result


def step0_precheck(
    data_dir: Path, delete_map: dict[str, set[int]]
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    """Precheck directory structure and validate deletion list.

    Args:
        data_dir: Root data directory.
        delete_map: Dictionary mapping id_name to episode indices to delete.

    Returns:
        Tuple of (id_dirs, affected) where affected is a dict with deletion metadata.

    Raises:
        ValueError: If required directories or files are missing or validation fails.
    """
    header(logger, "STEP 0  pre-check: directory structure + deletion list validation")
    id_dirs = find_id_dirs(data_dir)
    if not id_dirs:
        raise ValueError(f"No id_* directories found in {data_dir}")

    id_dir_map = {d.name: d for d in id_dirs}

    for id_name in delete_map:
        if id_name not in id_dir_map:
            raise ValueError(
                f"--delete specified {id_name} does not exist in {data_dir}"
            )

    affected: dict[str, dict[str, Any]] = {}
    for id_name, to_del in delete_map.items():
        id_dir = id_dir_map[id_name]
        data_path = id_dir / "data"
        if not data_path.exists():
            raise ValueError(f"Missing data directory: {data_path}")

        all_parquets: dict[int, Path] = {}
        for p in data_path.rglob("episode_*.parquet"):
            m = EP_RE.match(p.name)
            if m:
                all_parquets[int(m.group(1))] = p
            else:
                logger.warning(
                    "  Skipping parquet with unexpected name "
                    "(expected episode_<digits>.parquet): %s",
                    p,
                )

        missing = to_del - set(all_parquets.keys())
        if missing:
            raise ValueError(
                f"{id_name}: episodes to delete {sorted(missing)} do not exist"
            )

        keep = sorted(set(all_parquets.keys()) - to_del)
        logger.info(
            "  %s: %d episodes total, delete %s, keep %d",
            id_name,
            len(all_parquets),
            sorted(to_del),
            len(keep),
        )
        affected[id_name] = {
            "id_dir": id_dir,
            "all_parquets": all_parquets,
            "to_del": to_del,
            "keep": keep,
        }

    for id_name, id_dir in id_dir_map.items():
        if id_name not in affected:
            all_parquets = {}
            for p in (id_dir / "data").rglob("episode_*.parquet"):
                m = EP_RE.match(p.name)
                if m:
                    all_parquets[int(m.group(1))] = p
                else:
                    logger.warning(
                        "  Skipping parquet with unexpected name "
                        "(expected episode_<digits>.parquet): %s",
                        p,
                    )
            keep = sorted(all_parquets.keys())
            logger.info(
                "  %s: %d episodes total, no deletion",
                id_name,
                len(all_parquets),
            )
            affected[id_name] = {
                "id_dir": id_dir,
                "all_parquets": all_parquets,
                "to_del": set(),
                "keep": keep,
            }

    return id_dirs, affected


def step1_delete_parquets(affected: dict[str, dict[str, Any]], dry_run: bool) -> None:
    """Delete specified parquet files.

    Args:
        affected: Dictionary with deletion information for each id.
        dry_run: If True, log actions without executing them.
    """
    header(logger, "STEP 1  delete low-quality parquet files")
    for id_name, info in affected.items():
        to_del = info["to_del"]
        if not to_del:
            logger.info("  %s: nothing to delete", id_name)
            continue
        for old_idx in sorted(to_del):
            p = info["all_parquets"][old_idx]
            tag = "[DRY-RUN]" if dry_run else "[DELETE]"
            logger.info("  %s %s/episode_%06d.parquet", tag, id_name, old_idx)
            if not dry_run:
                p.unlink()


def step2_renumber_and_reindex(
    affected: dict[str, dict[str, Any]], dry_run: bool
) -> dict[str, dict[str, Any]]:
    """Rename remaining parquets and rewrite episode_index and index columns.

    Args:
        affected: Dictionary with deletion information for each id.
        dry_run: If True, log actions without executing them.

    Returns:
        Dictionary mapping id_name to mapping information for later use.

    Raises:
        ValueError: If parquet schema is invalid or metadata is corrupted.
    """
    header(
        logger,
        "STEP 2  rename remaining parquets + rewrite episode_index / index columns",
    )

    all_mappings: dict[str, dict[str, Any]] = {}

    for id_name, info in affected.items():
        keep = info["keep"]
        all_parquets = info["all_parquets"]

        logger.info("\n  -- %s --", id_name)
        mapping: list[dict[str, Any]] = []
        running = 0

        for new_idx, old_idx in enumerate(keep):
            old_path = all_parquets[old_idx]
            new_name = f"episode_{new_idx:06d}.parquet"
            new_path = old_path.parent / new_name

            t = pq.read_table(old_path)
            n = t.num_rows

            for col in ("episode_index", "index"):
                if col not in t.column_names:
                    raise ValueError(f"Missing required column {col} in {old_path}")

            old_ep = sorted(set(t.column("episode_index").combine_chunks().to_pylist()))
            old_ix = t.column("index").combine_chunks().to_pylist()

            ep_arr = pa.array(np.full(n, new_idx, dtype=np.int64), type=pa.int64())
            idx_arr = pa.array(
                np.arange(running, running + n, dtype=np.int64), type=pa.int64()
            )

            ci_ep = t.column_names.index("episode_index")
            t = t.set_column(ci_ep, t.schema.field(ci_ep), ep_arr)
            ci_ix = t.column_names.index("index")
            t = t.set_column(ci_ix, t.schema.field(ci_ix), idx_arr)

            tag = "[DRY-RUN]" if dry_run else "[WRITE]"
            logger.info(
                "  %s ep_idx: %s -> %d   index: [%d..%d] -> [%d..%d]   rows=%d   %s -> %s",
                tag,
                old_ep,
                new_idx,
                old_ix[0],
                old_ix[-1],
                running,
                running + n - 1,
                n,
                old_path.name,
                new_name,
            )

            if not dry_run:
                tmp = old_path.with_suffix(".parquet.tmp")
                pq.write_table(t, tmp, compression="snappy")

                if old_path != new_path:
                    tmp.replace(new_path)
                    if old_path.exists():
                        old_path.unlink()
                else:
                    tmp.replace(new_path)

                # Self-check
                chk = pq.read_table(new_path, columns=["episode_index", "index"])
                chk_ep = set(chk.column("episode_index").to_pylist())
                chk_idx = chk.column("index").to_pylist()
                if chk_ep != {new_idx}:
                    raise ValueError(
                        f"episode_index verification failed in {new_path}: got {chk_ep}, "
                        f"expected {{{new_idx}}}"
                    )
                if chk_idx != list(range(running, running + n)):
                    raise ValueError(f"index column verification failed in {new_path}")
                md = chk.schema.metadata or {}
                if b"huggingface" not in md:
                    raise ValueError(f"huggingface metadata lost in {new_path}")

            mapping.append(
                {
                    "old_idx": old_idx,
                    "new_idx": new_idx,
                    "new_path": old_path.parent / new_name,
                    "start": running,
                    "end": running + n - 1,
                    "n": n,
                }
            )
            running += n

        logger.info(
            "  %s: kept %d episodes, %d frames total", id_name, len(keep), running
        )
        all_mappings[id_name] = {"mapping": mapping, "total_frames": running}

    return all_mappings


def step3_update_meta_jsonl(
    affected: dict[str, dict[str, Any]],
    all_mappings: dict[str, dict[str, Any]],
    dry_run: bool,
) -> None:
    """Rewrite episodes.jsonl and episodes_stats.jsonl with new indices.

    Args:
        affected: Dictionary with deletion information for each id.
        all_mappings: Dictionary with mapping information from step2.
        dry_run: If True, log actions without executing them.

    Raises:
        ValueError: If JSONL structure is invalid.
    """
    header(logger, "STEP 3  rewrite episodes.jsonl + episodes_stats.jsonl")

    for id_name, info in affected.items():
        id_dir = info["id_dir"]
        mapping = all_mappings[id_name]["mapping"]
        logger.info("\n  -- %s --", id_name)

        ep_path = id_dir / "meta" / "episodes.jsonl"
        if not ep_path.exists():
            logger.info("    [SKIP] %s does not exist", ep_path)
        else:
            lines = read_jsonl(ep_path)
            ep_by_idx = {obj["episode_index"]: obj for obj in lines}

            new_lines = []
            for m in mapping:
                obj = ep_by_idx.get(m["old_idx"])
                if obj is None:
                    raise ValueError(
                        f"episode_index={m['old_idx']} not found in {ep_path}"
                    )
                old_top = obj.get("episode_index")
                length = obj.get("length")
                if length is not None and int(length) != int(m["n"]):
                    raise ValueError(
                        f"{ep_path} ep={old_top} length={length} != parquet rows={m['n']}"
                    )
                obj = dict(obj)
                obj["episode_index"] = int(m["new_idx"])
                new_lines.append(obj)
                logger.info(
                    "    episodes.jsonl  ep_idx: %s -> %d  length=%s",
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
            st_by_idx = {obj["episode_index"]: obj for obj in lines}

            new_lines = []
            for m in mapping:
                obj = st_by_idx.get(m["old_idx"])
                if obj is None:
                    raise ValueError(
                        f"episode_index={m['old_idx']} not found in {st_path}"
                    )
                stats = dict(obj.get("stats", {}))
                if "episode_index" not in stats or "index" not in stats:
                    raise ValueError(f"{st_path} missing stats.episode_index/index")

                old_ep_s = dict(stats["episode_index"])
                old_ix_s = dict(stats["index"])

                stats["episode_index"] = {
                    "min": [int(m["new_idx"])],
                    "max": [int(m["new_idx"])],
                    "mean": [float(m["new_idx"])],
                    "std": [0.0],
                    "count": [int(m["n"])],
                }
                stats["index"] = {
                    "min": [int(m["start"])],
                    "max": [int(m["end"])],
                    "mean": [float(m["start"] + m["end"]) / 2.0],
                    "std": old_ix_s.get("std", [0.0]),
                    "count": [int(m["n"])],
                }
                obj = dict(obj)
                obj["stats"] = stats
                obj["episode_index"] = int(m["new_idx"])
                new_lines.append(obj)

                logger.info(
                    "    episodes_stats  ep_idx: %d -> %d",
                    m["old_idx"],
                    m["new_idx"],
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


def step4_update_info(
    affected: dict[str, dict[str, Any]],
    all_mappings: dict[str, dict[str, Any]],
    dry_run: bool,
) -> None:
    """Update info.json with new episode/frame/chunk counts.

    Args:
        affected: Dictionary with deletion information for each id.
        all_mappings: Dictionary with mapping information from step2.
        dry_run: If True, log actions without executing them.
    """
    header(logger, "STEP 4  update info.json")

    for id_name, info in affected.items():
        id_dir = info["id_dir"]
        mapping = all_mappings[id_name]["mapping"]
        total_frames = all_mappings[id_name]["total_frames"]
        total_episodes = len(mapping)

        info_path = id_dir / "meta" / "info.json"
        old_info = json.loads(info_path.read_text(encoding="utf-8"))
        chunks_size = int(old_info.get("chunks_size", 1000))
        total_chunks = math.ceil(total_episodes / chunks_size) if total_episodes else 0

        logger.info("\n  -- %s --", id_name)
        logger.info(
            "    total_episodes : %s -> %d",
            old_info.get("total_episodes"),
            total_episodes,
        )
        logger.info(
            "    total_frames   : %s -> %d",
            old_info.get("total_frames"),
            total_frames,
        )
        logger.info(
            "    total_chunks   : %s -> %d",
            old_info.get("total_chunks"),
            total_chunks,
        )
        logger.info(
            "    splits.train   : %s -> 0:%d",
            old_info.get("splits", {}).get("train"),
            total_episodes,
        )

        new_info = dict(old_info)
        new_info["total_episodes"] = total_episodes
        new_info["total_frames"] = total_frames
        new_info["total_chunks"] = total_chunks
        new_info["splits"] = {"train": f"0:{total_episodes}"}

        if not dry_run:
            tmp = info_path.with_suffix(info_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(new_info, indent=4, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp.replace(info_path)
            logger.info("    [WRITE] %s", info_path)


def step5_verify(
    affected: dict[str, dict[str, Any]], all_mappings: dict[str, dict[str, Any]]
) -> None:
    """End-to-end verification of data consistency.

    Args:
        affected: Dictionary with deletion information for each id.
        all_mappings: Dictionary with mapping information from step2.

    Raises:
        ValueError: If verification fails.
    """
    header(logger, "STEP 5  end-to-end verification")
    all_ok = True

    for id_name, info in affected.items():
        id_dir = info["id_dir"]
        mapping = all_mappings[id_name]["mapping"]
        total_frames_expected = all_mappings[id_name]["total_frames"]
        total_episodes_expected = len(mapping)

        logger.info("\n  -- %s --", id_name)

        info_data = json.loads(
            (id_dir / "meta" / "info.json").read_text(encoding="utf-8")
        )
        if info_data["total_episodes"] != total_episodes_expected:
            raise ValueError(
                f"info.total_episodes={info_data['total_episodes']} != {total_episodes_expected}"
            )
        if info_data["total_frames"] != total_frames_expected:
            raise ValueError(
                f"info.total_frames={info_data['total_frames']} != {total_frames_expected}"
            )

        parquet_files: list[Path] = []
        for p in (id_dir / "data").rglob("episode_*.parquet"):
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
        if len(parquet_files) != total_episodes_expected:
            raise ValueError(
                f"parquet count {len(parquet_files)} != {total_episodes_expected}"
            )

        running = 0
        for p in parquet_files:
            t = pq.read_table(p, columns=["episode_index", "frame_index", "index"])
            n = t.num_rows
            ei = t.column("episode_index").to_pylist()
            fi = t.column("frame_index").to_pylist()
            ix = t.column("index").to_pylist()
            expect_ei = int(EP_RE.match(p.name).group(1))
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
                all_ok = False
            running += n

        if running != total_frames_expected:
            raise ValueError(f"cumulative frames {running} != {total_frames_expected}")
        logger.info(
            "    parquet: episode_index/frame_index/index correct ✓ (%d frames total)",
            running,
        )

        ep_path = id_dir / "meta" / "episodes.jsonl"
        if ep_path.exists():
            ep = read_jsonl(ep_path)
            if len(ep) != total_episodes_expected:
                raise ValueError(
                    f"episodes.jsonl {len(ep)} != {total_episodes_expected}"
                )
            logger.info("    episodes.jsonl       rows=%d ✓", len(ep))

        st_path = id_dir / "meta" / "episodes_stats.jsonl"
        if st_path.exists():
            es = read_jsonl(st_path)
            if len(es) != total_episodes_expected:
                raise ValueError(
                    f"episodes_stats.jsonl {len(es)} != {total_episodes_expected}"
                )
            prev_max = -1
            for r in sorted(es, key=lambda x: x["episode_index"]):
                mn = r["stats"]["index"]["min"][0]
                mx = r["stats"]["index"]["max"][0]
                if mn != prev_max + 1:
                    raise ValueError(
                        f"stats.index discontinuous: ep {r['episode_index']} min={mn} expected {prev_max + 1}"
                    )
                prev_max = mx
            if prev_max + 1 != total_frames_expected:
                raise ValueError(
                    f"stats.index final value {prev_max + 1} != {total_frames_expected}"
                )
            logger.info(
                "    episodes_stats.jsonl rows=%d ✓  stats.index continuous ✓",
                len(es),
            )

    if not all_ok:
        raise ValueError("verification failed, see FAIL lines above")
    logger.info("\n  >>> end-to-end verification passed <<<")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Delete low-quality episodes and renumber the rest"
    )
    ap.add_argument(
        "--data-dir",
        required=True,
        help="Root data directory containing id_X subdirectories",
    )
    ap.add_argument(
        "--delete",
        required=True,
        nargs="+",
        metavar="ID:EP[,EP...]",
        help="Episodes to delete, format: id_0:3,5. May specify different ids multiple times",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print only; do not write files",
    )
    ap.add_argument(
        "--log-file",
        default=None,
        help="Also write logs to this file",
    )
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip STEP 5 end-to-end verification",
    )
    return ap.parse_args()


def main() -> None:
    """Main entry point for the delete episodes script."""
    args = parse_args()
    file_handler: Optional[logging.FileHandler] = None
    if args.log_file:
        file_handler = add_log_file(logger, args.log_file)

    try:
        data_dir = Path(args.data_dir).resolve()
        delete_map = parse_delete_args(args.delete)

        logger.info("data-dir = %s", data_dir)
        logger.info("dry-run  = %s", args.dry_run)
        logger.info("delete list = %s", delete_map)
        if args.log_file:
            logger.info("log-file = %s", Path(args.log_file).resolve())

        _, affected = step0_precheck(data_dir, delete_map)
        step1_delete_parquets(affected, args.dry_run)
        all_mappings = step2_renumber_and_reindex(affected, args.dry_run)
        step3_update_meta_jsonl(affected, all_mappings, args.dry_run)
        step4_update_info(affected, all_mappings, args.dry_run)

        if not args.dry_run and not args.skip_verify:
            step5_verify(affected, all_mappings)

        header(
            logger,
            "All done ✓" if not args.dry_run else "Dry-run finished (nothing written)",
        )
    finally:
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()


if __name__ == "__main__":
    main()
