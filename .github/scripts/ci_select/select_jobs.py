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

"""Select the CI jobs a change needs.

Usage (inside a checkout, comparing against the PR base)::

    python .github/scripts/ci_select/select_jobs.py --base origin/main --head HEAD

A job is selected when a changed line or file is in its dependency map (see
``jobs.py``). A changed file no job depends on is reported as uncovered; a
change to the selector itself selects everything.
"""

import argparse
import ast
import fnmatch
import json
import re
import shlex
import sys
from dataclasses import dataclass, field

import hydra_cfg
import yaml
from jobs import CALLER, Job, build_jobs, caller_groups
from pyscan import PyIndex
from snapshot import GitRepo, Snapshot

# Changes no CI job can check; lint and the PR title check run on every PR.
NO_TEST = [
    "docs/**",
    "**/*.md",
    "**/*.rst",
    "**/*.png",
    "**/*.jpg",
    "**/*.svg",
    "**/*.gif",
    "LICENSE",
    "DCO",
    ".gitignore",
    ".github/CODEOWNERS",
    ".github/ISSUE_TEMPLATE/**",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/lint.yml",
    ".github/workflows/commit_check.yml",
    ".github/workflows/pr-title-check.yml",
    ".pre-commit-config.yaml",
    ".agents/**",
    ".claude/**",
    "AGENTS.md",
    "CLAUDE.md",
]
# Changing how jobs are chosen re-runs all of them. In ci-tests.yml only the
# lines outside the path-filter block count: the selector replaces those lists.
SELECTOR = [".github/scripts/ci_select/**"]
# pyproject.toml tables that only configure tools CI jobs do not run.
TOOL_ONLY_TABLES = re.compile(r"^tool\.(ruff|codespell|isort|black|mypy|pyright)\b")


def matches(path: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatchcase(path, pat) or (
            pat.startswith("**/") and fnmatch.fnmatchcase(path, pat[3:])
        ):
            return True
    return False


@dataclass
class Change:
    path: str
    old_path: str | None
    status: str
    added: set[int] = field(default_factory=set)  # new-side lines
    removed: set[int] = field(default_factory=set)  # old-side lines
    anchors: set[int] = field(
        default_factory=set
    )  # new-side neighbours of pure deletions

    @property
    def comment_only(self) -> bool:
        """A content edit that touched only comments and blank lines."""
        return self.status == "M" and not (self.added or self.removed or self.anchors)


# Line comments in the files CI reads (bash, Python, YAML, TOML, Dockerfile).
_COMMENT_SUFFIXES = (".sh", ".py", ".yaml", ".yml", ".toml", ".txt", "Dockerfile")


def _is_comment(path: str, text: str) -> bool:
    t = text.strip()
    return path.endswith(_COMMENT_SUFFIXES) and (
        not t or (t.startswith("#") and not t.startswith("#!"))
    )


def diff(repo: GitRepo, base: str, head: str) -> list[Change]:
    changes: dict[str, Change] = {}
    for line in repo.git(
        "diff", "--name-status", "-M", "--no-color", base, head
    ).splitlines():
        parts = line.split("\t")
        status = parts[0][0]
        if status == "R":
            changes[parts[2]] = Change(parts[2], parts[1], "R")
        else:
            changes[parts[1]] = Change(parts[1], None, status)
    current = None
    for line in repo.git(
        "diff", "-U0", "-M", "--no-color", "--no-ext-diff", base, head
    ).splitlines():
        if line.startswith("--- "):
            old = line[4:]
            current = (
                None
                if old == "/dev/null"
                else next(
                    (c for c in changes.values() if (c.old_path or c.path) == old[2:]),
                    None,
                )
            )
        elif line.startswith("+++ "):
            new = line[4:]
            if new != "/dev/null":
                current = changes.get(new[2:], current)
        elif line.startswith("@@") and current is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            old_no, new_no = int(m.group(1)), int(m.group(3))
            if int(m.group(4) or 1) == 0:
                hunk_anchor = {new_no, new_no + 1}
            else:
                hunk_anchor = None
        elif current is not None and line.startswith("-"):
            if not _is_comment(current.path, line[1:]):
                current.removed.add(old_no)
                if hunk_anchor:
                    current.anchors.update(hunk_anchor)
            old_no += 1
        elif current is not None and line.startswith("+"):
            if not _is_comment(current.path, line[1:]):
                current.added.add(new_no)
            new_no += 1
    return list(changes.values())


def _toml_tables(text: str) -> list[str]:
    table, out = "", []
    for line in text.split("\n"):
        m = re.match(r"^\s*\[+([^\]]+)\]+", line)
        if m:
            table = m.group(1).strip()
        out.append(table)
    return out


def _tool_only(ch: Change, post: Snapshot, pre: Snapshot) -> bool:
    if ch.path != "pyproject.toml":
        return False
    new_t = _toml_tables(post.read(ch.path) or "")
    old_t = _toml_tables(pre.read(ch.path) or "")
    tables = [new_t[i - 1] for i in ch.added | ch.anchors if 0 < i <= len(new_t)]
    tables += [old_t[i - 1] for i in ch.removed if 0 < i <= len(old_t)]
    return bool(tables) and all(TOOL_ONLY_TABLES.match(t) for t in tables)


_ARRAY = re.compile(r"^\s*(?:readonly\s+|declare\s+-a\s+)?([A-Za-z_]\w*)=\((.*)\)\s*$")


def array_additions(
    ch: Change, post: Snapshot, pre: Snapshot
) -> dict[int, tuple[int, set[str]]]:
    """Bash array literals the change only appended to.

    ``SUPPORTED_ENVS=(... "robodojo")`` runs in every job, but a job is
    affected only if it passes one of the new elements. Returns
    new line -> (old line, added elements).
    """
    if not ch.path.endswith(".sh") or ch.status != "M":
        return {}
    new_lines = (post.read(ch.path) or "").split("\n")
    old_lines = (pre.read(ch.path) or "").split("\n")
    old_arrays = {}
    for i in ch.removed:
        m = _ARRAY.match(old_lines[i - 1]) if 0 < i <= len(old_lines) else None
        if m:
            old_arrays[m.group(1)] = (i, m.group(2))
    out = {}
    for i in ch.added:
        m = _ARRAY.match(new_lines[i - 1]) if 0 < i <= len(new_lines) else None
        if not m or m.group(1) not in old_arrays:
            continue
        old_i, old_body = old_arrays[m.group(1)]
        try:
            new_items, old_items = shlex.split(m.group(2)), shlex.split(old_body)
        except ValueError:
            continue
        if (
            set(old_items) <= set(new_items)
            and [x for x in new_items if x in old_items] == old_items
        ):
            out[i] = (old_i, set(new_items) - set(old_items))
    return out


@dataclass
class PyLines:
    """Line-level Python checks for one change: post and pre snapshots."""

    post: PyIndex
    pre: PyIndex
    vocab_post: frozenset[str]
    vocab_pre: frozenset[str]


def _py_live(
    ch: Change, job: Job | None, lines: set[int], index: PyIndex, vocab, path: str
) -> set[int]:
    """The changed lines of ``path`` some Python run of ``job`` can execute."""
    if (
        job is None
        or not path.endswith(".py")
        or not job.py_values
        or None in job.py_values
    ):
        return lines
    return {
        i for i in lines if any(index.line_ok(path, i, v, vocab) for v in job.py_values)
    }


_MEMBER = re.compile(r"^\s+([A-Z][A-Z0-9_]*)\s*=\s*[\"']([^\"']+)[\"']\s*,?\s*$")


def member_additions(ch: Change, post: Snapshot, pre: Snapshot) -> dict[int, str]:
    """Added class constants with a name and value the class had nowhere else.

    ``ROBODOJO = "robodojo"`` in ``SupportedEnvType`` matters only to a job
    whose config uses ``robodojo``. Returns new line -> value.
    """
    if not ch.path.endswith(".py") or ch.status != "M":
        return {}
    new_text, old_text = post.read(ch.path) or "", pre.read(ch.path) or ""
    new_lines = new_text.split("\n")
    out = {}
    for i in ch.added:
        line = new_lines[i - 1] if 0 < i <= len(new_lines) else ""
        m = _MEMBER.match(line)
        if not m:
            continue
        name, value = m.groups()
        if (
            re.search(rf"\b{name}\b", old_text)
            or new_text.count(f'"{value}"') + new_text.count(f"'{value}'") > 1
        ):
            continue
        out[i] = value.lower()
    return out


def new_definitions(ch: Change, post: Snapshot, pre: Snapshot) -> dict[str, set[int]]:
    """Top-level functions and classes the change added: name -> new lines.

    Defining a name runs nothing, so these lines matter only to a job whose
    code refers to the name.
    """
    name = ch.path.rsplit("/", 1)[-1]
    if not ch.path.endswith(".py") or ch.status != "M":
        return {}
    # pytest collects test files by name, so nothing in them is unused.
    if (
        ch.path.startswith("tests/")
        or name.startswith("test_")
        or name == "conftest.py"
    ):
        return {}
    try:
        new_tree = ast.parse(post.read(ch.path) or "")
        old_tree = ast.parse(pre.read(ch.path) or "")
    except SyntaxError:
        return {}
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    old_names = {n.name for n in old_tree.body if isinstance(n, kinds)}
    out = {}
    for node in new_tree.body:
        # A decorator can register the definition by key; keep those.
        if (
            isinstance(node, kinds)
            and node.name not in old_names
            and not node.decorator_list
        ):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            span = set(range(start, node.end_lineno + 1))
            # Blank lines just above the definition belong to it as well.
            out[node.name] = span | {start - 1, start - 2}
    return out


class NameIndex:
    """Which Python files of a snapshot mention a name."""

    def __init__(self, snap: Snapshot):
        self.snap = snap
        self._cache: dict[str, frozenset[str]] = {}

    def files(self, name: str) -> frozenset[str]:
        if name not in self._cache:
            pat = re.compile(rf"\b{re.escape(name)}\b")
            self._cache[name] = frozenset(
                f
                for f in self.snap.files
                if f.endswith(".py") and pat.search(self.snap.read(f) or "")
            )
        return self._cache[name]


def _drop_unused_definitions(
    ch: Change, job: Job, defs: dict[str, set[int]], names: NameIndex, post: Snapshot
) -> Change:
    unused: set[int] = set()
    for name, span in defs.items():
        users = names.files(name) & (job.files | {ch.path})
        if users == {ch.path}:
            # Only this file mentions it: used if referenced outside its own body.
            text = (post.read(ch.path) or "").split("\n")
            outside = [
                t
                for i, t in enumerate(text, 1)
                if i not in span and re.search(rf"\b{re.escape(name)}\b", t)
            ]
            if outside:
                continue
        elif users:
            continue
        unused |= span
    if not unused:
        return ch
    return Change(
        ch.path,
        ch.old_path,
        ch.status,
        ch.added - unused,
        ch.removed,
        ch.anchors - unused,
    )


def touches(
    ch: Change,
    post: Job,
    pre: Job | None,
    unpruned: bool = False,
    appended: dict[int, tuple[int, set[str]]] | None = None,
    py: PyLines | None = None,
    members: dict[int, str] | None = None,
    defs: tuple[dict[str, set[int]], NameIndex, Snapshot] | None = None,
) -> bool:
    if defs and defs[0]:
        ch = _drop_unused_definitions(ch, post, *defs)
        if ch.status == "M" and not (ch.added or ch.removed or ch.anchors):
            return False
    if members:
        values = (
            set().union(*(v for v in post.py_values if v is not None))
            if post.py_values
            else set()
        )
        if None in post.py_values:
            keep = set(members)
        else:
            keep = {i for i, v in members.items() if v in values}
        ch = Change(
            ch.path,
            ch.old_path,
            ch.status,
            (ch.added - members.keys()) | keep,
            ch.removed,
            ch.anchors,
        )
        if ch.status == "M" and not (ch.added or ch.removed or ch.anchors):
            return False
    if appended:
        values = post.bash_values.get(ch.path, set())
        hits = {i for i, (_, added) in appended.items() if added & values}
        ch = Change(
            ch.path,
            ch.old_path,
            ch.status,
            (ch.added - appended.keys()) | hits,
            ch.removed - {old for i, (old, _) in appended.items() if i not in hits},
            ch.anchors,
        )
    if ch.status in "ADR":
        for d, name_rx in post.listings.items():
            for p in (ch.path, ch.old_path or ""):
                if p.startswith(d + "/") and (
                    name_rx is None or name_rx.search(p[len(d) + 1 :])
                ):
                    return True
    if ch.path in post.lines:
        if unpruned or (ch.added | ch.anchors) & post.lines[ch.path]:
            return True
    elif ch.path in (post.files_unpruned if unpruned else post.files):
        # A modified Python file counts only through lines the job can run.
        if unpruned or py is None or ch.status != "M" or not ch.path.endswith(".py"):
            return True
        if _py_live(ch, post, ch.added | ch.anchors, py.post, py.vocab_post, ch.path):
            return True
        if _py_live(ch, pre, ch.removed, py.pre, py.vocab_pre, ch.path):
            return True
        return False
    if pre is not None:
        old = ch.old_path or ch.path
        if old in pre.lines:
            return unpruned or bool(ch.removed & pre.lines[old])
        return old in (pre.files_unpruned if unpruned else pre.files)
    return False


@dataclass
class Selection:
    jobs: dict[str, list[str]] = field(default_factory=dict)  # job -> changed paths
    full: str | None = None  # why everything runs
    no_test: list[str] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)


def _filter_block(text: str) -> set[int]:
    """1-based lines of the ``filters: |`` block in ci-tests.yml."""
    lines = text.split("\n")
    out: set[int] = set()
    for i, line in enumerate(lines):
        if re.match(r"^\s*filters:\s*\|\s*$", line):
            indent = len(line) - len(line.lstrip())
            out.add(i + 1)
            for j in range(i + 1, len(lines)):
                nxt = lines[j]
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                out.add(j + 1)
    return out


def _caller_rule_change(ch: Change, post: Snapshot, pre: Snapshot) -> bool:
    if ch.path != CALLER:
        return False
    new_f = _filter_block(post.read(CALLER) or "")
    old_f = _filter_block(pre.read(CALLER) or "")
    return bool(((ch.added | ch.anchors) - new_f) or (ch.removed - old_f))


def select(
    changes: list[Change],
    post_jobs: dict[str, Job],
    pre_jobs: dict[str, Job],
    post: Snapshot,
    pre: Snapshot,
) -> Selection:
    sel = Selection()
    py = PyLines(
        PyIndex(post),
        PyIndex(pre),
        hydra_cfg.vocabulary(post),
        hydra_cfg.vocabulary(pre),
    )
    names = NameIndex(post)
    for ch in changes:
        if (
            matches(ch.path, SELECTOR)
            or (ch.old_path and matches(ch.old_path, SELECTOR))
            or _caller_rule_change(ch, post, pre)
        ):
            sel.full = sel.full or f"selector changed: {ch.path}"
            continue
        if ch.path == CALLER:
            sel.no_test.append(ch.path)
            continue
        if matches(ch.path, NO_TEST) or ch.comment_only or _tool_only(ch, post, pre):
            sel.no_test.append(ch.path)
            continue
        hit = False
        appended = array_additions(ch, post, pre)
        members = member_additions(ch, post, pre)
        defs = (new_definitions(ch, post, pre), names, post)
        for name, job in post_jobs.items():
            if touches(
                ch,
                job,
                pre_jobs.get(name),
                appended=appended,
                py=py,
                members=members,
                defs=defs,
            ):
                sel.jobs.setdefault(name, []).append(ch.path)
                hit = True
        if not hit:
            sel.uncovered.append(ch.path)
    if sel.full:
        sel.jobs = {name: ["(all)"] for name in post_jobs}
    return sel


# -- the current path filters, for comparison --------------------------------


def _glob_re(g: str) -> re.Pattern:
    out, i = "", 0
    while i < len(g):
        if g[i : i + 3] == "**/":
            out += "(?:.*/)?"
            i += 3
        elif g[i : i + 2] == "**":
            out += ".*"
            i += 2
        elif g[i] == "*":
            out += "[^/]*"
            i += 1
        else:
            out += re.escape(g[i])
            i += 1
    return re.compile(out + "$")


def _flatten(items) -> list[str]:
    out = []
    for x in items or []:
        if isinstance(x, list):
            out += _flatten(x)
        elif isinstance(x, str):
            out.append(x)
    return out


def current_filter_jobs(
    changes: list[Change], post: Snapshot, post_jobs: dict[str, Job]
) -> set[str]:
    """Jobs the path filters in ci-tests.yml run for this change."""
    data = yaml.safe_load(post.read(CALLER) or "") or {}
    step = next(
        s for s in data["jobs"]["check-changes"]["steps"] if s.get("id") == "filter"
    )
    filters = yaml.safe_load(step["with"]["filters"])
    paths = [c.path for c in changes] + [c.old_path for c in changes if c.old_path]
    hit = set()
    for name, globs in filters.items():
        pats = [_glob_re(g) for g in _flatten(globs)]
        if any(p.match(f) for f in paths for p in pats):
            hit.add(name)
    out = set()
    for wf, flt in caller_groups(post).items():
        if flt in hit:
            out |= {n for n, j in post_jobs.items() if j.workflow == wf}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--json", action="store_true", help="print the selection as JSON")
    args = ap.parse_args()
    repo = GitRepo(args.repo)
    base = repo.git("merge-base", args.base, args.head).strip()
    post, pre = repo.snapshot(args.head), repo.snapshot(base)
    post_jobs, pre_jobs = build_jobs(post), build_jobs(pre)
    sel = select(diff(repo, base, args.head), post_jobs, pre_jobs, post, pre)
    if args.json:
        json.dump(
            {
                "total": len(post_jobs),
                "jobs": sorted(sel.jobs),
                "reasons": {n: sorted(set(p))[:5] for n, p in sel.jobs.items()},
                "full": sel.full,
                "uncovered": sel.uncovered,
            },
            sys.stdout,
            indent=1,
        )
        return 0
    print(
        f"{len(sel.jobs)} of {len(post_jobs)} jobs"
        + (f" ({sel.full})" if sel.full else "")
    )
    for name in sorted(sel.jobs):
        print(f"  {name}  <- {', '.join(sorted(set(sel.jobs[name]))[:3])}")
    if sel.uncovered:
        print("Changed files no CI job depends on:")
        for p in sel.uncovered:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
