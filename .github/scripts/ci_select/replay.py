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

"""Replay merged PRs through the selector and the current path filters.

Usage::

    python .github/scripts/ci_select/replay.py --rev origin/main -n 80 --out replay.json

Each first-parent commit is one squash-merged PR. For every PR it reports the
jobs each policy runs, and sorts the jobs the filters run but the selector
drops into two kinds: jobs that do not depend on any changed file even with
no config pruning, and jobs that drop out only because their config selects
a different branch than the one that changed.
"""

import argparse
import json
import statistics
import sys
import time

from jobs import build_jobs
from select_jobs import current_filter_jobs, diff, select, touches
from snapshot import GitRepo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--rev", default="origin/main")
    ap.add_argument("-n", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = GitRepo(args.repo)
    log = repo.git(
        "log", "--first-parent", "--format=%H %s", "-n", str(args.n), args.rev
    ).splitlines()
    cache: dict[str, dict] = {}

    def jobs_at(rev: str):
        if rev not in cache:
            t = time.time()
            cache[rev] = build_jobs(repo.snapshot(rev))
            print(
                f"  mapped {rev[:8]}: {len(cache[rev])} jobs in {time.time() - t:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        return cache[rev]

    rows = []
    for line in log:
        sha, subject = line.split(" ", 1)
        base = repo.git("rev-parse", f"{sha}^").strip()
        post, pre = repo.snapshot(sha), repo.snapshot(base)
        post_jobs, pre_jobs = jobs_at(sha), jobs_at(base)
        changes = diff(repo, base, sha)
        sel = select(changes, post_jobs, pre_jobs, post, pre)
        cur = current_filter_jobs(changes, post, post_jobs)
        new = set(sel.jobs)
        # Rollout policy: a file no job depends on falls back to the old filters.
        unc = [c for c in changes if c.path in sel.uncovered]
        fallback = current_filter_jobs(unc, post, post_jobs) if unc else set()
        safe = new | fallback
        dropped = {}
        for name in sorted(cur - new):
            job = post_jobs[name]
            # Would this job depend on a changed path if no branch were pruned?
            via = [
                c.path
                for c in changes
                if touches(c, job, pre_jobs.get(name), unpruned=True)
            ]
            dropped[name] = via
        added = {name: sel.jobs[name] for name in sorted(new - cur)}
        rows.append(
            {
                "sha": sha[:8],
                "subject": subject,
                "files": [c.path for c in changes],
                "current": sorted(cur),
                "selected": sorted(new),
                "selected_fallback": sorted(safe),
                "selected_fallback_gpu": sum(post_jobs[n].gpu for n in safe),
                "current_gpu": sum(post_jobs[n].gpu for n in cur),
                "selected_gpu": sum(post_jobs[n].gpu for n in new),
                "full": sel.full,
                "uncovered": sel.uncovered,
                "no_test": sel.no_test,
                "dropped_pruned": {n: v for n, v in dropped.items() if v},
                "dropped_unrelated": sorted(n for n, v in dropped.items() if not v),
                "added": added,
                "reasons": {n: sorted(set(v))[:5] for n, v in sel.jobs.items()},
            }
        )
        r = rows[-1]
        print(
            f"{r['sha']} cur={len(cur):3d} new={len(new):3d} safe={len(safe):3d} gpu {r['current_gpu']:3d}->{r['selected_gpu']:3d}"
            f" pruned-drop={len(r['dropped_pruned']):3d} added={len(added):2d} unc={len(sel.uncovered)}"
            f"{' FULL' if sel.full else ''} | {subject[:60]}",
            flush=True,
        )

    n = len(rows)
    total = len(next(iter(cache.values())))

    def stat(key: str) -> str:
        vals = [len(r[key]) if isinstance(r[key], list) else r[key] for r in rows]
        return f"mean {statistics.mean(vals):.1f}, median {statistics.median(vals):.0f}, sum {sum(vals)}"

    summary = {
        "prs": n,
        "jobs_in_ci": total,
        "current_jobs": stat("current"),
        "selected_jobs": stat("selected"),
        "current_gpu_jobs": stat("current_gpu"),
        "selected_gpu_jobs": stat("selected_gpu"),
        "selected_fallback_jobs": stat("selected_fallback"),
        "selected_fallback_gpu_jobs": stat("selected_fallback_gpu"),
        "prs_full_fallback": sum(1 for r in rows if r["full"]),
        "prs_with_pruned_drops": sum(1 for r in rows if r["dropped_pruned"]),
        "pruned_drop_pairs": sum(len(r["dropped_pruned"]) for r in rows),
        "prs_with_added": sum(1 for r in rows if r["added"]),
        "added_pairs": sum(len(r["added"]) for r in rows),
        "prs_with_uncovered": sum(1 for r in rows if r["uncovered"]),
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
