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

"""Map the files a change touches to the area skills that own them.

Each area skill lists the files it owns under ``paths:`` in its ``SKILL.md``
frontmatter and ships a ``check.py``. A change is the difference between the
merge base of ``--base`` and the working tree, so uncommitted and untracked
files count. The contract is described in ``README.md`` next to this file;
area checks import this module for the change and for the finding format.

The skills come from the checkout that holds this script, and the repository
under check is the one the current directory is in. A review can therefore run
the current skills inside a worktree of a PR that predates them.

Usage::

    python3 .agents/skills/route.py list    # touched areas and their files
    python3 .agents/skills/route.py check   # run each touched area's check

Both take ``--base REF``; the default is ``main`` of the RLinf remote.
``check`` exits 1 when any area's check reports a critical or major finding.
"""

from __future__ import annotations

import argparse
import functools
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
RLINF_REMOTE = re.compile(r"github\.com[:/]RLinf/RLinf(?:\.git)?\s+\(fetch\)")
SEVERITIES = ("critical", "major", "minor")
HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def _toplevel() -> Path:
    found = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if found.returncode:
        raise SystemExit(
            f"error: run this inside an RLinf checkout ({found.stderr.strip()})"
        )
    return Path(found.stdout.strip())


REPO_ROOT = _toplevel()


def shown(path: Path) -> str:
    """A skills path as the agent reads it: ``.agents/skills/...``."""
    return path.relative_to(SKILLS_DIR.parents[1]).as_posix()


def git(*args: str) -> str:
    """Run git in the repository under check and return stdout."""
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a path glob: ``*`` stays in one directory, ``**`` spans them."""
    parts, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(parts) + r"\Z")


@dataclass(frozen=True)
class Area:
    """An area skill: its name, the globs it owns, and its check script."""

    name: str
    patterns: tuple[str, ...]

    @property
    def skill(self) -> Path:
        return SKILLS_DIR / self.name / "SKILL.md"

    @property
    def check(self) -> Path:
        return SKILLS_DIR / self.name / "check.py"

    def owns(self, path: str) -> bool:
        return any(glob_regex(p).match(path) for p in self.patterns)


def read_paths(skill: Path) -> tuple[str, ...]:
    """Return the ``paths:`` block list from a SKILL.md frontmatter."""
    lines = skill.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return ()
    patterns: list[str] = []
    in_paths = False
    for line in lines[1:]:
        if line == "---":
            break
        if line.startswith("paths:"):
            in_paths = True
        elif in_paths and line.lstrip().startswith("- "):
            patterns.append(line.lstrip()[2:].strip().strip("\"'"))
        elif in_paths and line.strip():
            in_paths = False
    return tuple(patterns)


def load_areas() -> list[Area]:
    """Every skill that declares ``paths:`` is an area and must have a check."""
    areas = []
    for skill in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        patterns = read_paths(skill)
        if not patterns:
            continue
        area = Area(skill.parent.name, patterns)
        if not area.check.is_file():
            raise SystemExit(
                f"error: {shown(area.skill)} declares paths: "
                f"but {shown(area.check)} does not exist"
            )
        areas.append(area)
    return areas


def default_base() -> str:
    """Return ``<remote>/main`` for the newest RLinf remote, else ``origin/main``."""
    remotes = {
        line.split()[0]
        for line in git("remote", "-v").splitlines()
        if RLINF_REMOTE.search(line)
    }
    candidates = []
    for remote in remotes:
        ref = f"{remote}/main"
        found = subprocess.run(
            ["git", "log", "-1", "--format=%ct", ref, "--"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if found.returncode == 0 and found.stdout.strip():
            candidates.append((int(found.stdout), ref))
    return max(candidates)[1] if candidates else "origin/main"


def merge_base(base: str) -> str:
    return git("merge-base", base, "HEAD").strip()


@dataclass(frozen=True)
class Change:
    """One changed file: git status letter (A, C, D, M, R, T) and path."""

    status: str
    path: str


def changes(base: str) -> list[Change]:
    """Files that differ between the merge base of ``base`` and the working tree."""
    found: dict[str, Change] = {}
    for line in git("diff", "--name-status", "-M", merge_base(base)).splitlines():
        status, *paths = line.split("\t")
        found[paths[-1]] = Change(status[0], paths[-1])
    for path in git("ls-files", "--others", "--exclude-standard").splitlines():
        found[path] = Change("A", path)
    return sorted(found.values(), key=lambda c: c.path)


def area_changes(name: str, base: str) -> list[Change]:
    """The changed files that the area called ``name`` owns."""
    area = next((a for a in load_areas() if a.name == name), None)
    if area is None:
        raise SystemExit(f"error: no area skill called {name!r} declares paths:")
    return [c for c in changes(base) if area.owns(c.path)]


@dataclass(frozen=True)
class Finding:
    """One problem a check reports, in the format every check prints."""

    severity: str
    path: str
    line: int | None
    rule: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"severity must be one of {SEVERITIES}, got {self.severity!r}"
            )

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{self.severity} {where} {self.rule}: {self.message}"


def report(area: str, findings: list[Finding]) -> int:
    """Print findings most severe first; return 1 if any is critical or major."""
    findings = sorted(
        set(findings),
        key=lambda f: (SEVERITIES.index(f.severity), f.path, f.line or 0, f.rule),
    )
    for finding in findings:
        print(finding)
    counts = {s: sum(f.severity == s for f in findings) for s in SEVERITIES}
    summary = ", ".join(f"{n} {s}" for s, n in counts.items() if n)
    print(f"{area}: {summary or 'no findings'}")
    return 1 if counts["critical"] or counts["major"] else 0


@functools.cache
def added_lines(base: str, path: str) -> frozenset[int]:
    """1-based line numbers of ``path`` that are new since the merge base."""
    diff = git("diff", "-U0", merge_base(base), "--", path)
    lines: set[int] = set()
    for hunk in HUNK.finditer(diff):
        start, count = int(hunk[1]), int(hunk[2] or 1)
        lines.update(range(start, start + count))
    return frozenset(lines)


@dataclass(frozen=True)
class Selection:
    """What an area check looks at: the base, the files, and the lines that count.

    Line findings count only on lines the change adds, so a check reports what
    the change introduces rather than what ``main`` already has. ``all_lines``
    counts every line of the selected files.
    """

    base: str
    changes: tuple[Change, ...]
    all_lines: bool

    def lines(self, change: Change) -> frozenset[int] | None:
        """The lines of a file whose findings count, or None for all of them."""
        if self.all_lines or change.status == "A":
            return None
        return added_lines(self.base, change.path)

    def relevant(self, findings: list[Finding]) -> list[Finding]:
        """Findings that have no line, or sit on a line that counts."""
        by_path = {c.path: c for c in self.changes}
        kept = []
        for finding in findings:
            change = by_path.get(finding.path)
            lines = self.lines(change) if change and finding.line else None
            if lines is None or finding.line in lines:
                kept.append(finding)
        return kept


def check_parser(description: str) -> argparse.ArgumentParser:
    """The command line every area check accepts; a check may add options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base", help="compare against this ref (default: RLinf main)")
    parser.add_argument(
        "--all-lines",
        action="store_true",
        help="report findings on every line of the files, not only on new lines",
    )
    parser.add_argument("files", nargs="*", help="files to check (default: changed)")
    return parser


def select(area: str, args: argparse.Namespace) -> Selection:
    """The given files, or the area's changed files when none are given.

    A given file keeps its status from the change, and is ``M`` otherwise.
    """
    base = args.base or default_base()
    if not args.files:
        return Selection(base, tuple(area_changes(area, base)), args.all_lines)
    status = {c.path: c.status for c in changes(base)}
    paths = [Path(f).resolve().relative_to(REPO_ROOT).as_posix() for f in args.files]
    selected = tuple(Change(status.get(p, "M"), p) for p in paths)
    return Selection(base, selected, args.all_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("list", "check"))
    parser.add_argument("--base", help="compare against this ref (default: RLinf main)")
    args = parser.parse_args()
    base = args.base or default_base()

    areas = load_areas()
    touched = changes(base)
    print(f"Base: {base} (merge base {merge_base(base)[:9]})\n")
    routed = {a.name: [c for c in touched if a.owns(c.path)] for a in areas}
    unrouted = [c for c in touched if not any(a.owns(c.path) for a in areas)]

    if args.command == "list":
        for area in areas:
            if routed[area.name]:
                print(f"{area.name}: read {shown(area.skill)}")
                for change in routed[area.name]:
                    print(f"  {change.status} {change.path}")
        if unrouted:
            print("unrouted (no area owns these):")
            for change in unrouted:
                print(f"  {change.status} {change.path}")
        if not touched:
            print("No changes against the base.")
        return 0

    failed = []
    for area in areas:
        files = [c.path for c in routed[area.name]]
        if not files:
            continue
        print(f"== {area.name} ({len(files)} file(s))", flush=True)
        result = subprocess.run(
            [sys.executable, str(area.check), "--base", base, *files], cwd=REPO_ROOT
        )
        if result.returncode:
            failed.append(area.name)
        print(flush=True)
    if unrouted:
        print("unrouted (no area check covers these):")
        for change in unrouted:
            print(f"  {change.status} {change.path}")
    if failed:
        print(f"Checks with critical or major findings: {', '.join(failed)}")
        return 1
    print(
        "No critical or major findings." if touched else "No changes against the base."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
