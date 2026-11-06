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

"""Compare a shadow selection with the jobs that actually ran in this CI run.

Runs as the last job of ``ci-tests.yml`` while the selector is in shadow mode:
CI still runs every job its path filters pick, and this report says which of
them the selector would have skipped, and whether any skipped job failed.

Usage (inside a workflow, with ``actions: read``)::

    python .github/scripts/ci_select/shadow_report.py --selection ci-select.json --out ci-select-report.json
"""

import argparse
import json
import os
import posixpath
import sys
import urllib.request

import yaml
from jobs import CALLER
from snapshot import FsSnapshot


def run_jobs(repo: str, run_id: str, attempt: str, token: str) -> list[dict]:
    """All jobs of one workflow run attempt, from the Actions API."""
    out, page = [], 1
    while True:
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100&page={page}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        out += data.get("jobs", [])
        if len(out) >= data.get("total_count", 0) or not data.get("jobs"):
            return out
        page += 1


def display_names(snap: FsSnapshot) -> dict[str, str]:
    """``caller-job / called-job-name`` as the Actions UI shows it -> selector name."""
    caller = yaml.safe_load(snap.read(CALLER) or "") or {}
    out = {}
    for caller_id, spec in (caller.get("jobs") or {}).items():
        uses = (spec or {}).get("uses") or ""
        if not uses.startswith("./"):
            continue
        wf = uses[2:]
        stem = posixpath.splitext(posixpath.basename(wf))[0]
        called = yaml.safe_load(snap.read(wf) or "") or {}
        for job_id, job in (called.get("jobs") or {}).items():
            name = (job or {}).get("name", job_id)
            out[f"{caller_id} / {name}"] = f"{stem}/{job_id}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--selection", required=True, help="output of select_jobs.py --json"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    selection = json.load(open(args.selection))
    selected = set(selection["jobs"])
    names = display_names(FsSnapshot("."))
    jobs = run_jobs(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_RUN_ID"],
        os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        os.environ["GITHUB_TOKEN"],
    )

    rows = []
    for job in jobs:
        name = names.get(job["name"])
        if name is None:
            continue
        rows.append(
            {
                "job": name,
                "conclusion": job.get("conclusion"),
                "selected": name in selected,
            }
        )
    ran = [r for r in rows if r["conclusion"] not in (None, "skipped")]
    missed = [r for r in ran if r["conclusion"] == "failure" and not r["selected"]]
    skipped_by_selector = [r for r in ran if not r["selected"]]
    not_run = sorted(selected - {r["job"] for r in ran})

    report = {
        "selection_full": selection.get("full"),
        "ran": len(ran),
        "would_skip": len(skipped_by_selector),
        "selected_total": len(selected),
        "failures": [r["job"] for r in ran if r["conclusion"] == "failure"],
        "missed_failures": [r["job"] for r in missed],
        "selected_not_run": not_run,
        "uncovered": selection.get("uncovered", []),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    lines = [
        "## CI job selector (shadow mode)",
        "",
        f"- Jobs that ran: **{len(ran)}**; the selector would have run **{len(ran) - len(skipped_by_selector)}** of them "
        f"and skipped **{len(skipped_by_selector)}**.",
        f"- Failed jobs: {len(report['failures'])}; failed jobs the selector would have skipped: "
        f"**{len(missed)}**" + (" :warning:" if missed else ""),
    ]
    if selection.get("full"):
        lines.append(f"- The selector chose every job: {selection['full']}")
    for r in missed:
        lines.append(f"  - missed: `{r['job']}`")
    if not_run:
        lines.append(f"- Selected but not run by the path filters: {len(not_run)}")
        lines += [f"  - `{n}`" for n in not_run[:30]]
    if report["uncovered"]:
        lines.append(f"- Changed files no job depends on: {len(report['uncovered'])}")
        lines += [f"  - `{p}`" for p in report["uncovered"][:30]]
    text = "\n".join(lines) + "\n"
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
