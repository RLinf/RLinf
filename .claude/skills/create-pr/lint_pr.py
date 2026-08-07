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

"""Lint an RLinf pull-request title and body against CONTRIBUTING.md.

Checks the Conventional Commits title format and the
`.github/PULL_REQUEST_TEMPLATE.md` section structure, and catches the
rendering defects that show up repeatedly in real RLinf PRs: bodies pasted
with a uniform leading indent, tables whose rows and header disagree on
indentation, template sections that were dropped or left empty, and
checklists with nothing ticked.

Subcommands::

    lint      --pr N | --title T [--body-file F]   check a PR or a draft
    fix       --pr N | --body-file F               emit a de-indented body
    template                                       print a clean skeleton

Exit code is 1 when any ERROR is reported, 0 otherwise (warnings pass).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Iterable

# The title rules below mirror `openGemini/pr-title-checker`, the action run by
# the required `pr-title-check` job in .github/workflows/pr-title-check.yml. It
# runs with `strict: true` and `max_description_length: 50`, so everything it
# rejects is an ERROR here — a WARN would let a title through that CI blocks.

# Conventional Commits types listed in CONTRIBUTING.md, plus the extra ones
# already used on main (`perf`) and the standard remainder. Same set as the
# action's ALLOWED_TYPES.
ALLOWED_TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "test",
    "chore",
    "perf",
    "build",
    "ci",
    "revert",
)

TITLE_RE = re.compile(
    r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!?): (?P<desc>.+)$"
)

# Scopes that carry no information about *what* changed.
VAGUE_SCOPES = {
    "new",
    "update",
    "updates",
    "change",
    "changes",
    "misc",
    "various",
    "code",
    "main",
    "all",
    "general",
    "tmp",
    "temp",
    "wip",
    "pr",
}

# Section headings of .github/PULL_REQUEST_TEMPLATE.md, in template order.
# `required_filled` follows CONTRIBUTING.md: "The PR description should fill in
# at least the `Description` and `Checklist` sections".
TEMPLATE_SECTIONS = (
    ("Description", True),
    ("Motivation and Context", False),
    ("How has this been tested?", False),
    ("Additional information", False),
    ("Types of changes", True),
    ("Checklist", True),
)

# The action counts only the part after `: `, not the whole title.
TITLE_DESC_MAX = 50
SCOPE_RE = re.compile(r"^[a-z0-9_-]+$")
NON_ASCII_RE = re.compile(r"[^\x20-\x7e]")

DESCRIPTION_MAX_WORDS = 250

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<hashes>#{1,6})\s+(?P<text>.*?)\s*:?\s*$"
)
FENCE_RE = re.compile(r"^\s*(```|~~~)")
CHECKED_RE = re.compile(r"^\s*[-*]\s*\[[xX]\]")
UNCHECKED_RE = re.compile(r"^\s*[-*]\s*\[\s*\]")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
ISSUE_SUFFIX_RE = re.compile(r"\(#\d+\)\s*$")

# The exact set the pr-title-check action rejects (its nonImperativePatterns).
CI_NON_IMPERATIVE = {
    "added",
    "adds",
    "adding",
    "updated",
    "updates",
    "updating",
    "fixed",
    "fixes",
    "fixing",
    "removed",
    "removes",
    "removing",
    "deleted",
    "deletes",
    "deleting",
}

# Non-imperative starts CI misses but reviewers still flag.
PAST_TENSE_STARTS = (
    "added",
    "fixed",
    "updated",
    "removed",
    "changed",
    "refactored",
    "implemented",
    "supported",
    "improved",
)


@dataclass
class Finding:
    level: str  # "ERROR" | "WARN"
    where: str  # "title" | "body"
    code: str
    message: str
    fix: str = ""


@dataclass
class Section:
    name: str
    start: int  # index of the heading line
    lines: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        stripped = COMMENT_RE.sub("", "\n".join(self.lines)).strip()
        return not stripped

    @property
    def text(self) -> str:
        return COMMENT_RE.sub("", "\n".join(self.lines)).strip()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def fence_mask(lines: Iterable[str]) -> list[bool]:
    """Return, per line, whether it sits inside a fenced code block."""
    mask, inside = [], False
    for line in lines:
        if FENCE_RE.match(line):
            mask.append(True)  # the fence itself counts as code
            inside = not inside
            continue
        mask.append(inside)
    return mask


def split_sections(body: str) -> list[Section]:
    """Split a PR body into `###`-delimited sections (code fences respected)."""
    lines = body.splitlines()
    in_code = fence_mask(lines)
    sections: list[Section] = []
    for i, line in enumerate(lines):
        if in_code[i]:
            if sections:
                sections[-1].lines.append(line)
            continue
        m = HEADING_RE.match(line)
        if m:
            sections.append(Section(name=m.group("text").strip(), start=i))
        elif sections:
            sections[-1].lines.append(line)
    return sections


def find_section(sections: list[Section], name: str) -> Section | None:
    key = name.lower().rstrip(":?").strip()
    for sec in sections:
        if sec.name.lower().rstrip(":?").strip().startswith(key):
            return sec
    return None


def fetch_pr(number: str) -> tuple[str, str]:
    """Return (title, body) for a PR via the `gh` CLI."""
    out = subprocess.run(
        ["gh", "pr", "view", str(number), "--json", "title,body"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.exit(f"error: gh pr view {number} failed:\n{out.stderr.strip()}")
    data = json.loads(out.stdout)
    return data["title"], (data.get("body") or "").replace("\r\n", "\n")


# --------------------------------------------------------------------------- #
# title checks
# --------------------------------------------------------------------------- #


def check_title(title: str) -> list[Finding]:
    # Not stripped: CI validates the title verbatim, and a trailing space in
    # the description is one of the things it rejects.
    out: list[Finding] = []
    if not title.strip():
        return [Finding("ERROR", "title", "title-empty", "PR title is empty.")]

    bad = NON_ASCII_RE.findall(title)
    if bad:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-non-ascii",
                f"Title contains non-ASCII characters: {''.join(sorted(set(bad)))!r}",
                "Use plain ASCII (32-126) — no smart quotes, em dashes, "
                "subscripts, or CJK.",
            )
        )

    m = TITLE_RE.match(title)
    if not m:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-format",
                f"Title is not Conventional Commits: {title!r}",
                "Rewrite as `<type>(<scope>): <description>`, e.g. "
                "`fix(libero): reset action queues between episodes`.",
            )
        )
        return out

    typ, scope, desc = m.group("type"), m.group("scope"), m.group("desc")

    if typ.lower() not in ALLOWED_TYPES:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-type",
                f"Unknown type {typ!r}.",
                "Use one of: " + ", ".join(ALLOWED_TYPES) + ".",
            )
        )
    elif typ != typ.lower():
        out.append(
            Finding(
                "ERROR", "title", "title-type-case", f"Type {typ!r} must be lowercase."
            )
        )

    if scope is not None:
        if not scope.strip():
            out.append(
                Finding(
                    "ERROR",
                    "title",
                    "title-scope-empty",
                    "Empty scope `()`.",
                    "Drop the parentheses or name the touched component.",
                )
            )
        elif scope.strip().lower() in VAGUE_SCOPES:
            out.append(
                Finding(
                    "WARN",
                    "title",
                    "title-scope-vague",
                    f"Scope {scope!r} says nothing about what changed.",
                    "Name the component the diff touches (module, model, env, "
                    "or subsystem), e.g. `embodiment`, `libero`, `docker`, "
                    "`readme` — or drop the scope entirely.",
                )
            )
        elif not SCOPE_RE.match(scope):
            out.append(
                Finding(
                    "ERROR",
                    "title",
                    "title-scope-format",
                    f"Scope {scope!r} is not lowercase [a-z0-9_-].",
                    "Scopes take lowercase letters, digits, hyphens and "
                    "underscores only — e.g. `embodiment`, `source-en`, "
                    "`libero`. No spaces, slashes, dots, or capitals.",
                )
            )

    if len(desc) > TITLE_DESC_MAX:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-desc-long",
                f"Description is {len(desc)} chars; CI caps it at "
                f"{TITLE_DESC_MAX} (the part after `: `, not the whole title).",
                "Say the single most important thing and move the rest into "
                "the Description section.",
            )
        )

    if desc != desc.rstrip():
        out.append(
            Finding(
                "ERROR", "title", "title-trailing-space", "Drop the trailing space."
            )
        )
    if re.search(r":\s{2,}", title):
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-colon-spacing",
                "Use exactly one space after the colon.",
            )
        )
    if "!" in desc:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-bang-position",
                "`!` may only appear right before the colon, as the breaking-"
                "change marker — never inside the description.",
            )
        )

    desc = desc.strip()
    if desc.endswith("."):
        out.append(
            Finding("ERROR", "title", "title-period", "Drop the trailing period.")
        )

    first = desc.split()[0] if desc.split() else ""
    if not re.match(r"^[a-z]", desc):
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-capital",
                f"Description must start with a lowercase a-z, not {first!r}.",
                "A leading digit or backtick fails too — reword so the first "
                "character is a lowercase letter (`support 5D parallelism`, "
                "not `5D parallelism support`).",
            )
        )
    if first.lower() in CI_NON_IMPERATIVE:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-mood",
                f"Use the imperative mood, not {first!r} "
                f"(e.g. {imperative_of(first)}).",
            )
        )
    elif first.lower() in PAST_TENSE_STARTS:
        out.append(
            Finding(
                "WARN",
                "title",
                "title-mood",
                f"Use the imperative mood, not {first!r} "
                f"(e.g. {imperative_of(first)}).",
            )
        )
    if desc.lower().startswith(("this pr", "pr ", "a pr")):
        out.append(
            Finding(
                "WARN", "title", "title-throat-clearing", "Drop the 'this PR' preamble."
            )
        )
    if ISSUE_SUFFIX_RE.search(desc):
        out.append(
            Finding(
                "WARN",
                "title",
                "title-pr-number",
                "Drop the trailing `(#N)`; GitHub appends it on squash-merge.",
            )
        )
    return out


PAST_TENSE_STARTS_IMPERATIVE = {
    "added": "add",
    "fixed": "fix",
    "updated": "update",
    "removed": "remove",
    "changed": "change",
    "refactored": "refactor",
    "implemented": "implement",
    "supported": "support",
    "improved": "improve",
}


def imperative_of(word: str) -> str:
    """Best-effort imperative form of a past-tense/third-person verb."""
    stem = word.lower()
    if stem in PAST_TENSE_STARTS_IMPERATIVE:
        return PAST_TENSE_STARTS_IMPERATIVE[stem]
    for suffix in ("ing", "ed", "es", "s"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            return base + "e" if suffix in ("ing", "ed") and base in DROPS_E else base
    return stem


# Verbs whose imperative keeps the `e` that `-ed`/`-ing` swallowed.
DROPS_E = {"updat", "remov", "delet", "creat", "mov", "renam", "merg"}


# --------------------------------------------------------------------------- #
# body checks
# --------------------------------------------------------------------------- #


def check_indentation(body: str) -> list[Finding]:
    """Catch bodies pasted with a uniform leading indent (the #1421 defect)."""
    out: list[Finding] = []
    lines = body.splitlines()
    in_code = fence_mask(lines)

    indented_headings = [
        i
        for i, line in enumerate(lines)
        if not in_code[i] and (m := HEADING_RE.match(line)) and m.group("indent")
    ]
    if indented_headings:
        out.append(
            Finding(
                "ERROR",
                "body",
                "body-indented-heading",
                f"{len(indented_headings)} template heading(s) have leading "
                f"whitespace (first at line {indented_headings[0] + 1}).",
                "Strip the indent — run `lint_pr.py fix` and repost the body. "
                "Indented headings nest sections inside each other and break "
                "the template structure reviewers look for.",
            )
        )

    prose = [
        i
        for i, line in enumerate(lines)
        if not in_code[i] and line.strip() and not HEADING_RE.match(line)
    ]
    if prose:
        indented = [
            i
            for i in prose
            if line_indent(lines[i]) and not continues_list(lines, i, in_code)
        ]
        if len(indented) >= max(4, 0.3 * len(prose)):
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "body-indented",
                    f"{len(indented)}/{len(prose)} prose lines start with "
                    "whitespace — the body was pasted with a uniform indent.",
                    "Run `lint_pr.py fix` to de-indent, then repost. Four or "
                    "more leading spaces turn a paragraph into a code block.",
                )
            )

    out.extend(check_tables(lines, in_code))
    return out


def line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def continues_list(lines: list[str], i: int, in_code: list[bool]) -> bool:
    """True when line `i` is an indented continuation of a markdown list."""
    if LIST_RE.match(lines[i]):
        # A nested list item is legitimate only under a shallower item.
        pass
    for j in range(i - 1, -1, -1):
        if in_code[j]:
            return False
        if not lines[j].strip():
            continue
        if LIST_RE.match(lines[j]):
            return line_indent(lines[i]) > line_indent(lines[j])
        return False
    return False


def check_tables(lines: list[str], in_code: list[bool]) -> list[Finding]:
    """Flag markdown tables whose rows disagree on leading indentation."""
    out: list[Finding] = []
    block: list[int] = []

    def flush(block: list[int]) -> None:
        if len(block) < 2:
            return
        indents = {line_indent(lines[i]) for i in block}
        if len(indents) > 1:
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "body-table-indent",
                    f"Table at line {block[0] + 1} mixes row indentation "
                    f"{sorted(indents)} — GitHub will not render it as a table.",
                    "Give every row of the table the same (preferably zero) "
                    "indentation.",
                )
            )
        elif indents and min(indents) >= 4:
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "body-table-indent",
                    f"Table at line {block[0] + 1} is indented "
                    f"{min(indents)} spaces — it renders as a code block.",
                    "Un-indent the table.",
                )
            )

    for i, line in enumerate(lines):
        if not in_code[i] and line.strip().count("|") >= 2:
            block.append(i)
        else:
            flush(block)
            block = []
    flush(block)
    return out


def check_body(body: str, *, docs_only: bool = False) -> list[Finding]:
    out: list[Finding] = []
    if not body.strip():
        return [
            Finding(
                "ERROR",
                "body",
                "body-empty",
                "PR body is empty.",
                "Start from `lint_pr.py template`.",
            )
        ]

    out.extend(check_indentation(body))

    lines = body.splitlines()
    if body.count("```") % 2:
        out.append(
            Finding(
                "ERROR",
                "body",
                "body-fence",
                "Unbalanced ``` fence — later text is swallowed.",
            )
        )

    sections = split_sections(body)
    names = [s.name for s in sections]

    for name, must_fill in TEMPLATE_SECTIONS:
        sec = find_section(sections, name)
        if sec is None:
            out.append(
                Finding(
                    "ERROR" if must_fill else "WARN",
                    "body",
                    "section-missing",
                    f"Template section `### {name}` is missing.",
                    "Keep every heading from .github/PULL_REQUEST_TEMPLATE.md, "
                    "even when a section is short.",
                )
            )
            continue
        if must_fill and sec.is_empty:
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "section-empty",
                    f"Required section `### {name}` is empty.",
                    "CONTRIBUTING.md: PRs that leave Description or Checklist "
                    "empty are marked draft and not reviewed.",
                )
            )

    # Order: the template's headings should stay in template order.
    present = [n for n, _ in TEMPLATE_SECTIONS if find_section(sections, n)]
    order = [
        i
        for i, n in enumerate(names)
        if any(
            n.lower().rstrip(":?").startswith(t.lower().rstrip(":?"))
            for t, _ in TEMPLATE_SECTIONS
        )
    ]
    if len(present) > 1 and order != sorted(order):
        out.append(
            Finding(
                "WARN",
                "body",
                "section-order",
                "Template sections are out of template order.",
            )
        )

    # Prose before the first heading.
    if sections:
        head = COMMENT_RE.sub("", "\n".join(lines[: sections[0].start])).strip()
        if head:
            out.append(
                Finding(
                    "WARN",
                    "body",
                    "body-preamble",
                    f"{len(head.split())} word(s) sit above the first `###` heading.",
                    "Move that text into `### Description`.",
                )
            )

    # Checkbox sections.
    for name in ("Types of changes", "Checklist"):
        sec = find_section(sections, name)
        if sec is None:
            continue
        text = "\n".join(sec.lines)
        checked = [ln for ln in sec.lines if CHECKED_RE.match(ln)]
        boxes = checked + [ln for ln in sec.lines if UNCHECKED_RE.match(ln)]
        if not boxes:
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "checkbox-missing",
                    f"`### {name}` has no checkboxes.",
                    "Paste the list back from .github/PULL_REQUEST_TEMPLATE.md.",
                )
            )
        elif not checked:
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "checkbox-unticked",
                    f"Nothing is ticked in `### {name}`.",
                    "Tick every box that applies; an all-empty list reads as "
                    "'not self-reviewed'.",
                )
            )
        elif name == "Types of changes" and len(checked) > 2:
            out.append(
                Finding(
                    "WARN",
                    "body",
                    "checkbox-overticked",
                    f"{len(checked)} change types ticked — pick the ones that "
                    "genuinely apply.",
                )
            )
        del text

    # Description quality.
    desc = find_section(sections, "Description")
    if desc is not None and not desc.is_empty:
        words = len(desc.text.split())
        if words > DESCRIPTION_MAX_WORDS:
            out.append(
                Finding(
                    "WARN",
                    "body",
                    "description-bloated",
                    f"`### Description` is {words} words.",
                    f"Trim to <= {DESCRIPTION_MAX_WORDS}: one sentence on what "
                    "changed, then a bullet per user-visible change. Push "
                    "detail into the code or the docs.",
                )
            )
        bullets = [ln for ln in desc.lines if LIST_RE.match(ln)]
        if len(bullets) > 12:
            out.append(
                Finding(
                    "WARN",
                    "body",
                    "description-bullets",
                    f"{len(bullets)} bullets in `### Description`.",
                    "Group them; a reviewer scans, not reads.",
                )
            )

    tested = find_section(sections, "How has this been tested?")
    if tested is not None and tested.is_empty and not docs_only:
        out.append(
            Finding(
                "WARN",
                "body",
                "tested-empty",
                "`### How has this been tested?` is empty.",
                "Name the command you ran and its outcome (e2e config, eval "
                "numbers, or `pre-commit run --all-files` for docs-only PRs).",
            )
        )

    # Leftover template scaffolding.
    comments = COMMENT_RE.findall(body)
    if len(comments) >= 4:
        out.append(
            Finding(
                "WARN",
                "body",
                "body-comments",
                f"{len(comments)} template `<!--- ... -->` hints are still in "
                "the body.",
                "Delete the hints once each section is written.",
            )
        )

    # Hard-wrapped prose.
    in_code = fence_mask(lines)
    wrapped = 0
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        if in_code[i] or in_code[i + 1]:
            continue
        if HEADING_RE.match(a) or HEADING_RE.match(b) or LIST_RE.match(b):
            continue
        if (
            a.strip()
            and b.strip()
            and 40 <= len(a.rstrip()) <= 90
            and not a.rstrip().endswith((".", ":", "|"))
        ):
            wrapped += 1
    if wrapped >= 6:
        out.append(
            Finding(
                "WARN",
                "body",
                "body-hardwrap",
                f"~{wrapped} hard-wrapped prose lines.",
                "GitHub reflows paragraphs; one line per paragraph survives "
                "edits and diffs better.",
            )
        )
    return out


# --------------------------------------------------------------------------- #
# fix
# --------------------------------------------------------------------------- #


def fix_body(body: str, *, unwrap: bool = False) -> str:
    """De-indent a uniformly-indented body and drop the template hints.

    The de-indent only runs when :func:`check_indentation` actually reports an
    error, so a body with legitimate nested lists is left alone. Nesting inside
    a mangled body is preserved: the smallest *positive* indent is treated as
    the paste offset and subtracted, leaving deeper levels relative to it.
    """
    lines = body.replace("\r\n", "\n").splitlines()
    in_code = fence_mask(lines)

    mangled = any(f.level == "ERROR" for f in check_indentation(body))
    if mangled:
        positives = [
            line_indent(line)
            for i, line in enumerate(lines)
            if not in_code[i] and line.strip() and line_indent(line) > 0
        ]
        offset = min(positives) if positives else 0
    else:
        offset = 0

    out = []
    for i, line in enumerate(lines):
        if in_code[i]:
            out.append(line)
            continue
        fixed = line[offset:] if offset and line_indent(line) >= offset else line
        # Headings and tables must sit flush left to render.
        if HEADING_RE.match(fixed) or fixed.strip().count("|") >= 2:
            fixed = fixed.lstrip()
        out.append(fixed.rstrip())

    text = COMMENT_RE.sub("", "\n".join(out))
    if unwrap:
        text = unwrap_paragraphs(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def unwrap_paragraphs(text: str) -> str:
    """Join hard-wrapped prose into one line per paragraph.

    Headings, list items, tables, and fenced code are left untouched.
    """
    lines = text.splitlines()
    in_code = fence_mask(lines)
    out: list[str] = []
    for i, line in enumerate(lines):
        joinable = (
            out
            and not in_code[i]
            and line.strip()
            and out[-1].strip()
            and not HEADING_RE.match(line)
            and not LIST_RE.match(line)
            and "|" not in line
            and not HEADING_RE.match(out[-1])
            and not LIST_RE.match(out[-1])
            and "|" not in out[-1]
            and not (i and in_code[i - 1])
        )
        if joinable:
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


TEMPLATE = """\
### Description

<one sentence: what this PR changes and where>

- <user-visible change 1>
- <user-visible change 2>

### Motivation and Context

<why the change is needed; link the issue as `Fixes #NNN` if there is one>

### How has this been tested?

<the exact command you ran and its outcome — e2e config name, eval numbers,
or `pre-commit run --all-files` for docs-only PRs>

### Additional information (optional, e.g., figures and logs):

### Types of changes

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Documentation update (Document-only update)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)

### Checklist:

- [ ] My code follows the code style of this project.
- [ ] My change requires a change to the documentation.
- [ ] I have updated the documentation accordingly.
- [ ] I have added tests to cover my changes.
- [ ] All new and existing tests passed.
"""


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def report(findings: list[Finding], *, title: str, quiet: bool, scope: str) -> int:
    errors = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]

    print(f"PR title: {title}")
    print(f"{len(errors)} error(s), {len(warns)} warning(s)\n")
    for f in errors + warns:
        mark = "E" if f.level == "ERROR" else "W"
        print(f"[{mark}] {f.where}/{f.code}: {f.message}")
        if f.fix and not quiet:
            for line in textwrap.wrap(f.fix, 74):
                print(f"      -> {line}")
    if not findings:
        print(
            f"clean — {scope} checked, nothing to fix against CONTRIBUTING.md\nand .github/PULL_REQUEST_TEMPLATE.md."
        )
    return 1 if errors else 0


def load(args: argparse.Namespace) -> tuple[str, str]:
    if args.pr:
        return fetch_pr(args.pr)
    title = args.title or ""
    body = ""
    if args.body_file:
        with open(args.body_file, encoding="utf-8") as fh:
            body = fh.read().replace("\r\n", "\n")
    return title, body


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    lint = sub.add_parser("lint", help="check a PR or a draft")
    lint.add_argument("--pr", help="PR number (fetched with `gh pr view`)")
    lint.add_argument("--title", help="draft title")
    lint.add_argument("--body-file", help="file holding the draft body")
    lint.add_argument(
        "--docs-only", action="store_true", help="skip the testing-section warning"
    )
    lint.add_argument("--quiet", action="store_true", help="omit the suggested fixes")

    fix = sub.add_parser("fix", help="print a de-indented, hint-free body")
    fix.add_argument("--pr", help="PR number")
    fix.add_argument("--body-file", help="file holding the body")
    fix.add_argument(
        "--unwrap",
        action="store_true",
        help="also join hard-wrapped prose into one line per paragraph",
    )

    sub.add_parser("template", help="print the filled-in skeleton")

    args = p.parse_args(argv)

    if args.cmd == "template":
        print(TEMPLATE, end="")
        return 0

    if args.cmd == "fix":
        if args.pr:
            _, body = fetch_pr(args.pr)
        elif args.body_file:
            with open(args.body_file, encoding="utf-8") as fh:
                body = fh.read()
        else:
            sys.exit("error: fix needs --pr or --body-file")
        print(fix_body(body, unwrap=args.unwrap), end="")
        return 0

    if not args.pr and not args.title and not args.body_file:
        sys.exit("error: lint needs --pr, or --title and/or --body-file")

    title, body = load(args)
    findings: list[Finding] = []
    if title or args.pr:
        findings += check_title(title)
    if body or args.pr:
        findings += check_body(body, docs_only=args.docs_only)
    scope = " and ".join(
        part
        for part, on in (
            ("title", bool(title or args.pr)),
            ("body", bool(body or args.pr)),
        )
        if on
    )
    return report(
        findings, title=title or "(none given)", quiet=args.quiet, scope=scope
    )


if __name__ == "__main__":
    raise SystemExit(main())
