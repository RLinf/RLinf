# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Out-of-process tqdm monitor for ``examples/embodiment/collect_real_data.py``.

Runs in its own terminal, tails the collector's tee'd log, and renders a
live bar with the latest pedal / success / discard event. The collector
itself can't host the bar because Ray batches worker stdout (~500 ms),
which shreds tqdm's ``\\r`` refresh.

Usage (two terminals on the collector node)::

    # terminal 1 — launch (stdout tee'd to a log file)
    bash collect_data.sh 2>&1 | tee run_embodiment.log

    # terminal 2 — live bar
    python toolkits/realworld_check/collect_monitor.py run_embodiment.log

Pass ``--no-replay`` to tail from EOF instead of replaying the existing
file at startup.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

# Matches both "Total: N/M" (success line) and "Total success: N/M" (discard line).
_TOTAL_RE = re.compile(r"Total(?:\s+success)?:\s*(\d+)\s*/\s*(\d+)")
_KB_RE = re.compile(r"\[keyboard\]\s+(\S+)")
_SUCCESS_RE = re.compile(r"Success\s*\(reward=([-\d.]+)")
# Anchors on the closing ``|`` of the tqdm gauge so percentages aren't read as the numerator.
_TQDM_RE = re.compile(r"Collecting Data Episodes:.*?\|\s*(\d+)\s*/\s*(\d+)")


_ENVGROUP_PID_RE = re.compile(r"EnvGroup\(rank=0\)\s+pid=(\d+)")


def _probe_target(paths: list[Path]) -> Optional[int]:
    """Return the collector's target episode count by scanning candidate files
    for the first ``Collecting Data Episodes:.*N/M`` tqdm line.

    Looks at paths in order (use more reliable sources first). tqdm writes
    stderr using ``\\r`` to overwrite itself, so we split on ``\\r`` as well
    as lines before matching. Returns None if no match in any file."""
    for p in paths:
        try:
            if not p or not p.exists():
                continue
            txt = p.read_text(errors="replace").replace("\r", "\n")
            m = _TQDM_RE.search(txt)
            if m:
                return int(m.group(2))
        except OSError:
            continue
    return None


def _find_worker_file(tee_log: Path) -> Optional[Path]:
    """Locate the Ray per-worker stdout file for EnvGroup(rank=0).

    Ray forwards worker stdout through log_monitor to the driver, which can
    introduce multi-second (observed: 1-2 min under load) batching delays.
    Tailing the worker file directly bypasses that path — latency then comes
    only from Python's line-buffered stdout.

    We get the worker pid from the tee'd log (which prints ``(EnvGroup(rank=0)
    pid=NNNNN)`` in front of forwarded lines) and glob the session logs dir.
    Returns None if the tee log doesn't have the pid yet or the worker file
    isn't on this host (e.g. EnvGroup is on a different Ray node).
    """
    if not tee_log.exists():
        return None
    try:
        head = tee_log.read_text(errors="replace")
    except OSError:
        return None
    m = _ENVGROUP_PID_RE.search(head)
    if not m:
        return None
    pid = m.group(1)
    candidates = sorted(
        glob.glob(f"/tmp/ray/session_latest/logs/worker-*-{pid}.out"),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    return Path(candidates[0]) if candidates else None


def _follow(path: Path, poll_s: float, replay: bool) -> Iterator[str]:
    """Yield lines from ``path``. If ``replay`` is true, read existing content
    from the start before switching to tail mode; otherwise jump straight to
    EOF. Waits for the file to appear either way."""
    while not path.exists():
        time.sleep(poll_s)
    with path.open("r", errors="replace") as f:
        if not replay:
            f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_s)
                continue
            yield line.rstrip("\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live progress monitor for collect_real_data.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "log_path",
        type=Path,
        help="Path to the collector's tee'd log file (stdout of launch script).",
    )
    ap.add_argument(
        "--poll",
        type=float,
        default=0.1,
        help="Tail poll interval in seconds (default: 0.1).",
    )
    ap.add_argument(
        "--no-replay",
        action="store_true",
        help="Skip replaying existing file content on startup (legacy behavior).",
    )
    ap.add_argument(
        "--source",
        choices=("auto", "worker", "tee"),
        default="auto",
        help=(
            "Which file to tail. 'worker' tails the Ray worker stdout file "
            "directly (bypasses log_monitor batching, ~1-2 min faster). "
            "'tee' tails the driver's tee'd log (works across nodes but slow). "
            "'auto' tries worker first, falls back to tee."
        ),
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print each matched event line to stderr.",
    )
    args = ap.parse_args()

    source_path = args.log_path
    if args.source in ("auto", "worker"):
        worker_file = _find_worker_file(args.log_path)
        if worker_file is not None:
            source_path = worker_file
            print(
                f"[monitor] tailing Ray worker file: {worker_file}",
                file=sys.stderr,
                flush=True,
            )
        elif args.source == "worker":
            print(
                "[monitor] --source=worker requested but no EnvGroup worker "
                "file found under /tmp/ray/session_latest/logs/. Is EnvGroup "
                "on a different Ray node?",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(2)
        else:
            print(
                f"[monitor] no worker file yet, falling back to tee log: "
                f"{args.log_path}",
                file=sys.stderr,
                flush=True,
            )

    pbar: Optional[tqdm] = None
    last_saved = 0
    last_event = ""
    last_reward: Optional[str] = None

    def _dbg(tag: str, line: str) -> None:
        if args.debug:
            print(f"[monitor {tag}] {line}", file=sys.stderr, flush=True)

    # Pre-create the bar from the worker's stderr (.err sibling) before any episode ends.
    probe_candidates: list[Path] = []
    if source_path.suffix == ".out":
        probe_candidates.append(source_path.with_suffix(".err"))
    probe_candidates.append(source_path)
    if args.log_path != source_path:
        probe_candidates.append(args.log_path)
    target_from_probe = _probe_target(probe_candidates)
    if target_from_probe is not None:
        pbar = tqdm(
            total=target_from_probe,
            dynamic_ncols=True,
            desc="collect",
            unit="ep",
            leave=True,
        )
        _dbg("init", f"target={target_from_probe} (from tqdm probe)")

    try:
        for line in _follow(source_path, args.poll, replay=not args.no_replay):
            m_tot = _TOTAL_RE.search(line)
            if m_tot:
                saved = int(m_tot.group(1))
                target = int(m_tot.group(2))
                _dbg("total", f"saved={saved} target={target}")
                if pbar is None:
                    pbar = tqdm(
                        total=target,
                        initial=saved,
                        dynamic_ncols=True,
                        desc="collect",
                        unit="ep",
                        leave=True,
                    )
                    last_saved = saved
                elif saved > last_saved:
                    pbar.update(saved - last_saved)
                    last_saved = saved
                if saved >= target:
                    pbar.refresh()
                    break

            m_kb = _KB_RE.search(line)
            if m_kb:
                last_event = m_kb.group(1)
                _dbg("kb", last_event)

            m_succ = _SUCCESS_RE.search(line)
            if m_succ:
                last_reward = m_succ.group(1)
                _dbg("reward", last_reward)

            if pbar is not None:
                post: dict[str, str] = {}
                if last_event:
                    post["last_event"] = last_event
                if last_reward is not None:
                    post["last_reward"] = last_reward
                if post:
                    pbar.set_postfix(post, refresh=False)
                    pbar.refresh()
    except KeyboardInterrupt:
        pass
    finally:
        if pbar is not None:
            pbar.close()


if __name__ == "__main__":
    main()
