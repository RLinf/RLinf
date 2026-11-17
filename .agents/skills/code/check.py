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

"""Check changed RLinf code: lint, package layout, headers, CI filters, test layout.

Usage::

    python3 .agents/skills/code/check.py            # files changed against main
    python3 .agents/skills/code/check.py FILE ...

``pre-commit`` runs Ruff with ``--fix``, so it may rewrite files in place; run
the check again after it does.
"""

from __future__ import annotations

import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import route  # noqa: E402

CI_WORKFLOW = ".github/workflows/ci-tests.yml"
HEADER = re.compile(r"# Copyright (\d{4}) The RLinf Authors\.")


def check_pre_commit(paths: list[str]) -> list[route.Finding]:
    if not shutil.which("pre-commit"):
        print("skipped: pre-commit is not installed (pip install pre-commit)")
        return []
    result = subprocess.run(
        ["pre-commit", "run", "--files", *paths],
        cwd=route.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return []
    print(result.stdout + result.stderr)
    return [
        route.Finding(
            "major",
            ".pre-commit-config.yaml",
            None,
            "code/pre-commit",
            "pre-commit failed on the changed files (output above); fix what it reports",
        )
    ]


def check_init() -> list[route.Finding]:
    result = subprocess.run(
        [sys.executable, "tests/unit_tests/check_missing_init.py"],
        cwd=route.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return [
        route.Finding(
            "major",
            line.strip()[2:],
            None,
            "code/init",
            "has Python files but no __init__.py",
        )
        for line in result.stdout.splitlines()
        if line.strip().startswith("- ")
    ]


def check_header(path: str) -> list[route.Finding]:
    head = (route.REPO_ROOT / path).read_text(encoding="utf-8").splitlines()[:5]
    years = [m[1] for line in head if (m := HEADER.match(line))]
    if not years:
        return [
            route.Finding(
                "major",
                path,
                1,
                "code/license",
                "new file without '# Copyright <year> The RLinf Authors.'; "
                "a vendored file keeps its upstream header",
            )
        ]
    year = str(datetime.date.today().year)
    if years[0] != year:
        return [
            route.Finding(
                "minor",
                path,
                1,
                "code/license",
                f"new file carries {years[0]}; use {year} unless it moved from another path",
            )
        ]
    return []


def flatten(items: object) -> list[str]:
    """Patterns from a filter, with YAML aliases (nested lists) expanded."""
    if isinstance(items, str):
        return [items]
    if isinstance(items, dict):
        return [p for value in items.values() for p in flatten(value)]
    return [p for item in items for p in flatten(item)]


def check_ci_filters(changes: list[route.Change]) -> list[route.Finding]:
    """Replay CI's filter-coverage-test: every changed file it tracks is covered."""
    try:
        import yaml
    except ImportError:
        print("skipped: CI filter coverage needs PyYAML (pip install pyyaml)")
        return []
    workflow = yaml.safe_load(
        (route.REPO_ROOT / CI_WORKFLOW).read_text(encoding="utf-8")
    )
    filters = next(
        yaml.safe_load(step["with"]["filters"])
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("dorny/paths-filter")
    )
    tracked = [route.glob_regex(p) for p in flatten(filters["all_changes_filter"])]
    covered = [route.glob_regex(p) for p in flatten(filters["checked_files_filter"])]
    return [
        route.Finding(
            "major",
            c.path,
            None,
            "code/ci-filter",
            f"matches no path filter in {CI_WORKFLOW}; add it to the filter whose "
            "tests exercise it",
        )
        for c in changes
        if c.status in ("A", "M")
        and any(r.match(c.path) for r in tracked)
        and not any(r.match(c.path) for r in covered)
    ]


def main() -> int:
    args = route.check_parser(__doc__.splitlines()[0]).parse_args()
    changes = route.select("code", args).changes
    present = [c for c in changes if c.status != "D"]
    findings = check_pre_commit([c.path for c in present]) if present else []
    if any(c.path.startswith("rlinf/") for c in changes):
        findings += check_init()
    for change in present:
        if change.status == "A" and change.path.endswith(".py"):
            findings += check_header(change.path)
        if change.status == "A" and re.fullmatch(
            r"tests/unit_tests/test_\w+\.py", change.path
        ):
            findings.append(
                route.Finding(
                    "minor",
                    change.path,
                    None,
                    "code/test-layout",
                    "new test file; tests/unit_tests holds one file per component, so add "
                    "these cases to the component's file unless this is a new component",
                )
            )
    findings += check_ci_filters(changes)
    return route.report("code", findings)


if __name__ == "__main__":
    raise SystemExit(main())
