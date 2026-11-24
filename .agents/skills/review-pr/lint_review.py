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

"""Lint and post an RLinf PR review written as a GitHub review payload.

The payload is the JSON body of GitHub's create-review API
(``POST /repos/{owner}/{repo}/pulls/{number}/reviews``)::

    {
        "commit_id": "<PR head SHA>",
        "event": "COMMENT",
        "body": "<the review: verdict, findings, reproduction>",
        "comments": [
            {"path": "rlinf/x.py", "line": 42, "side": "RIGHT", "body": "..."},
            {"path": "rlinf/y.py", "start_line": 10, "line": 12, "body": "..."},
        ],
    }

Subcommands::

    template                         print the main-comment skeleton
    lint [--pr N] review.json        check the payload (and its lines, with --pr)
    post --pr N review.json          lint, then post it as one review
    update --pr N --review ID review.json
                                     lint the body against the review's posted
                                     inline comments, then replace its body

Exit code is 1 when any ERROR is reported, 0 otherwise (warnings pass).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import cached_property

RECOMMENDATIONS = ("Ready for senior review", "Changes requested", "Blocked")
SEVERITIES = ("critical", "major", "minor")
CATEGORIES = ("correctness", "design", "reuse", "tests", "docs", "docs-style", "style")
STATUSES = ("confirmed", "unresolved")
POSTED = ("inline", "here")
REPRODUCTION, FINDINGS = "### Reproduction", "### Findings"
SECTIONS = (REPRODUCTION, FINDINGS)
REPRO_LABELS = ("- Result:", "- Setup:")
REPRO_STATUSES = ("**Passed**", "**Failed**", "**Partial**", "**Not run**")
NO_FINDINGS = "No findings."
TABLE_COLUMNS = 7  # ID, Severity, Category, Location, Summary, Status, Posted
MAX_BODY = 65536  # GitHub rejects a longer review body
EVENTS = ("COMMENT", "REQUEST_CHANGES", "APPROVE")
FIX_LEAD = "Suggested fix:"
EVIDENCE_LEAD = "Evidence:"
WALKTHROUGH_LEAD = "Code walkthrough:"
TEMPLATE = """\
**<Recommendation>** · <n critical, n major, n minor, or no findings> · head `<sha7>` · base `main@<sha7>`

<One short sentence: what blocks merge, or why it is ready.>

### Reproduction

- Result: **<Passed | Failed | Partial | Not run>** — <one short sentence: functional status and key metric vs. baseline, or first blocker / what is missing>
- Setup: <hardware count/type and config or command run; or what a run needs if not run>

### Findings

| ID | Severity | Category | Location | Summary | Status | Posted |
|---|---|---|---|---|---|---|
| F1 | critical | correctness | `rlinf/x.py:42` | <one line> | confirmed | inline |
| F2 | major | docs | `docs/source-zh/...` | <one line> | confirmed | here |

#### F2 · major · docs: <plain title>

<what it tries to do, what happens instead, and why it matters, in 1-3 plain sentences>

Suggested fix: <one sentence, then a code block when code changes>

Evidence: <one line: command and result, or trace; for unresolved, what settles it>
"""
PAYLOAD_KEYS = {"commit_id", "event", "body", "comments"}
COMMENT_KEYS = {"path", "line", "side", "start_line", "start_side", "body"}
MAX_COMMENT_PROSE = 1000
MAX_MINOR_FINDINGS = 5
MAX_REPRO_BULLET = 300


@dataclass(frozen=True)
class Finding:
    level: str  # "ERROR" | "WARN"
    where: str
    message: str


class Markdown:
    """Prose checks for text GitHub renders as a comment.

    GitHub turns every newline in a comment into a line break, so a paragraph
    must stay on one line.
    """

    FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
    BLOCK = re.compile(r"^\s*([#|>]|[-*+] |\d+[.)] |<)")
    HEADING = re.compile(r"^(#{1,6})\s")
    # A tag, an HTML comment, or a <placeholder> left from the template.
    HTML = re.compile(r"<!--|</?[A-Za-z][\w-]*(\s[^<>]*)?/?>")
    CODE_SPAN = re.compile(r"(`+).+?\1")
    HEDGING = re.compile(
        r"\b(you might want to|you may want to|perhaps consider|"
        r"consider (?:using|adding|changing|updating|removing)|"
        r"it might be worth|it is worth noting|just a small)\b",
        re.IGNORECASE,
    )

    @classmethod
    def unclosed_fence(cls, text: str) -> int | None:
        """Returns the 1-based opening line of an unclosed code fence, if any."""
        fence, opened_at = "", None
        for n, line in enumerate(text.splitlines(), 1):
            m = cls.FENCE.match(line)
            if not fence:
                if m:
                    fence, opened_at = m[1], n
            elif (
                m
                and m[1][0] == fence[0]
                and len(m[1]) >= len(fence)
                and not m[2].strip()
            ):
                fence, opened_at = "", None
        return opened_at

    @classmethod
    def unfenced(cls, text: str) -> list[tuple[int, str]]:
        """Returns (line number, line) for every line outside code fences.

        As in CommonMark, only a bare fence of the opening character, at least
        as long as the opening one, closes a block, so fences can nest.
        """
        out, fence = [], ""
        for n, line in enumerate(text.splitlines(), 1):
            m = cls.FENCE.match(line)
            if not fence:
                if m:
                    fence = m[1]
                else:
                    out.append((n, line))
            elif (
                m
                and m[1][0] == fence[0]
                and len(m[1]) >= len(fence)
                and not m[2].strip()
            ):
                fence = ""
        return out

    @classmethod
    def wrapped(cls, text: str) -> list[int]:
        """Returns the lines that continue the paragraph above them."""
        hits, prev_n, prev_prose = [], -1, False
        for n, line in cls.unfenced(text):
            stripped = line.strip()
            if (
                prev_prose
                and n == prev_n + 1
                and stripped
                and not cls.BLOCK.match(line)
            ):
                hits.append(n)
            prev_n = n
            prev_prose = bool(stripped) and not stripped.startswith(("#", "|", "<"))
        return hits

    @classmethod
    def prose_length(cls, text: str) -> int:
        return sum(len(line.strip()) for _, line in cls.unfenced(text))


class PullRequest:
    """The PR state a review is checked against, read through the `gh` CLI."""

    HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    def __init__(self, repo: str, number: int) -> None:
        self.repo = repo
        self.number = number

    @staticmethod
    def gh(*args: str, stdin: str | None = None) -> str:
        proc = subprocess.run(
            ["gh", *args], input=stdin, capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            sys.exit(f"error: gh {' '.join(args[:2])} failed: {proc.stderr.strip()}")
        return proc.stdout

    def api_items(self, endpoint: str, jq: str) -> list[dict]:
        out = self.gh("api", endpoint, "--paginate", "--jq", jq)
        return [json.loads(line) for line in out.splitlines() if line.strip()]

    @cached_property
    def head_sha(self) -> str:
        out = self.gh(
            "pr", "view", str(self.number), "--repo", self.repo, "--json", "headRefOid"
        )
        return json.loads(out)["headRefOid"]

    @cached_property
    def diff_lines(self) -> dict[tuple[str, str], dict[int, int]]:
        """Maps (path, side) to {line number: hunk index} for commentable lines."""
        files = self.api_items(
            f"repos/{self.repo}/pulls/{self.number}/files",
            ".[] | {filename, patch}",
        )
        lines: dict[tuple[str, str], dict[int, int]] = {}
        for f in files:
            right = lines.setdefault((f["filename"], "RIGHT"), {})
            left = lines.setdefault((f["filename"], "LEFT"), {})
            self.parse_patch(f.get("patch") or "", left, right)
        return lines

    @classmethod
    def parse_patch(cls, patch: str, left: dict, right: dict) -> None:
        hunk, old, new = -1, 0, 0
        for line in patch.splitlines():
            m = cls.HUNK.match(line)
            if m:
                hunk, old, new = hunk + 1, int(m[1]), int(m[2])
            elif line.startswith("+"):
                right[new], new = hunk, new + 1
            elif line.startswith("-"):
                left[old], old = hunk, old + 1
            elif not line.startswith("\\"):
                left[old], right[new] = hunk, hunk
                old, new = old + 1, new + 1

    @cached_property
    def commented_lines(self) -> dict[tuple[str, str, int], str]:
        """Maps (path, side, line) of existing review threads to their author."""
        items = self.api_items(
            f"repos/{self.repo}/pulls/{self.number}/comments",
            ".[] | {path, side, line, user: .user.login, in_reply_to_id}",
        )
        return {
            (c["path"], c["side"], c["line"]): c["user"]
            for c in items
            if c["line"] is not None and c["in_reply_to_id"] is None
        }

    def review_comments(self, review_id: int) -> list[dict]:
        """The inline comments a posted review holds, at the lines it posted them on.

        The per-review endpoint omits line numbers, so this filters the PR's
        comments by review instead.
        """
        return self.api_items(
            f"repos/{self.repo}/pulls/{self.number}/comments",
            f".[] | select(.pull_request_review_id == {review_id}"
            " and .in_reply_to_id == null) | {path, "
            "line: (.original_line // .line), "
            "start_line: (.original_start_line // .start_line), "
            'side: (.side // "RIGHT"), body}',
        )

    def update(self, review_id: int, body: str) -> str:
        endpoint = f"repos/{self.repo}/pulls/{self.number}/reviews/{review_id}"
        out = self.gh(
            "api",
            "-X",
            "PUT",
            endpoint,
            "--input",
            "-",
            stdin=json.dumps({"body": body}),
        )
        return json.loads(out)["html_url"]

    def post(self, payload: dict) -> str:
        endpoint = f"repos/{self.repo}/pulls/{self.number}/reviews"
        out = self.gh(
            "api", "-X", "POST", endpoint, "--input", "-", stdin=json.dumps(payload)
        )
        return json.loads(out)["html_url"]


@dataclass(frozen=True)
class Row:
    """One row of the findings table."""

    n: int
    id: str
    severity: str
    category: str
    location: str
    posted: str


@dataclass
class Section:
    """One finding's #### section, after its heading."""

    n: int
    severity: str
    category: str
    lines: list[str]


class ReviewLinter:
    """Checks a review payload against the review-pr skill's output format."""

    VERDICT = re.compile(
        r"^\*\*(" + "|".join(map(re.escape, RECOMMENDATIONS)) + r")\*\*"
    )
    ROW = re.compile(r"^\|\s*F\d+\s*\|")
    FINDING = re.compile(r"^#### (F\d+) · ([\w-]+) · ([\w-]+): \S")
    INLINE_HEADING = re.compile(r"^\*\*(F\d+) · ([\w-]+) · ([\w-]+): \S.*\*\*$")
    LOCATION = re.compile(r"`([^`:\s]+):(\d+)(?:-(\d+))?`")

    def __init__(self, payload: dict, pr: PullRequest | None) -> None:
        self.payload = payload
        self.pr = pr
        self.findings: list[Finding] = []

    def error(self, where: str, message: str) -> None:
        self.findings.append(Finding("ERROR", where, message))

    def warn(self, where: str, message: str) -> None:
        self.findings.append(Finding("WARN", where, message))

    def run(self) -> list[Finding]:
        extra = set(self.payload) - PAYLOAD_KEYS
        if extra:
            self.error(
                "payload",
                f"unknown keys {sorted(extra)}; expected {sorted(PAYLOAD_KEYS)}",
            )
        if self.payload.get("event", "COMMENT") not in EVENTS:
            self.error(
                "payload",
                f"event must be one of {EVENTS}, found {self.payload['event']!r}",
            )
        if self.pr and self.payload.get("commit_id") not in (None, self.pr.head_sha):
            self.error(
                "payload",
                f"commit_id {self.payload['commit_id'][:7]} is not the PR head "
                f"{self.pr.head_sha[:7]}; re-check the findings on the new head",
            )
        self.check_body(self.payload.get("body") or "")
        seen: set[tuple[str, str, int]] = set()
        for i, comment in enumerate(self.payload.get("comments") or [], 1):
            self.check_comment(i, comment, seen)
        return self.findings

    def check_prose(self, where: str, text: str) -> None:
        opened_at = Markdown.unclosed_fence(text)
        if opened_at is not None:
            self.error(
                where,
                f"line {opened_at} opens a code fence that is never closed",
            )
        for n in Markdown.wrapped(text):
            self.error(
                where,
                f"line {n} is hard-wrapped; GitHub renders it as a line break, so keep each paragraph on one line",
            )
        for n, line in Markdown.unfenced(text):
            stripped = Markdown.CODE_SPAN.sub("", line)
            m = Markdown.HTML.search(stripped)
            if m:
                self.error(
                    where,
                    f"line {n} holds HTML or an unfilled template placeholder ({m[0][:40]!r}); write plain Markdown",
                )
            h = Markdown.HEDGING.search(stripped)
            if h:
                self.warn(
                    where,
                    f"line {n} uses hedging or filler ({h[0]!r}); state the issue and fix directly",
                )

    def check_body(self, body: str) -> None:
        self.check_prose("body", body)
        if len(body) > MAX_BODY:
            self.error(
                "body",
                f"body is {len(body)} characters and GitHub accepts at most {MAX_BODY}; shorten the walkthroughs",
            )
        lines = Markdown.unfenced(body)
        self.check_verdict(lines)
        sections: list[tuple[str, list[tuple[int, str]]]] = []
        for n, line in lines:
            m = Markdown.HEADING.match(line)
            if m and len(m[1]) <= 3:
                sections.append((line.strip(), []))
            elif sections:
                sections[-1][1].append((n, line))
        found = [heading for heading, _ in sections]
        if found != list(SECTIONS):
            self.error(
                "body",
                f"sections must be {list(SECTIONS)} in this order, found {found}; use #### inside a section",
            )
        content = dict(reversed(sections))
        rows = self.check_findings(content.get(FINDINGS, []), body.splitlines())
        self.check_inline_rows(rows)
        if REPRODUCTION in content:
            self.check_reproduction(content[REPRODUCTION])

    def check_verdict(self, lines: list[tuple[int, str]]) -> None:
        first = next((line.strip() for _, line in lines if line.strip()), "")
        if not self.VERDICT.match(first):
            choices = ", ".join(f"**{r}**" for r in RECOMMENDATIONS)
            self.error(
                "body",
                f"the first line must start with one bold recommendation ({choices}), found {first[:60]!r}",
            )
        elif sum(r in first for r in RECOMMENDATIONS) > 1:
            self.error(
                "body", "the verdict line names several recommendations; state one"
            )

    def check_findings(
        self, lines: list[tuple[int, str]], raw_lines: list[str]
    ) -> list[Row]:
        """Checks the findings table against the finding sections below it."""
        rows: list[Row] = []
        sections: dict[str, Section] = {}
        current: Section | None = None
        for n, line in lines:
            if line.startswith("#### "):
                m = self.FINDING.match(line)
                current = None
                if not m:
                    self.error(
                        "body",
                        f"line {n}: a #### heading in Findings must read 'F<n> · <severity> · <category>: <title>'",
                    )
                elif m[1] in sections:
                    self.error("body", f"line {n}: {m[1]} has a second section")
                else:
                    current = sections[m[1]] = Section(n, m[2], m[3], [])
            elif current is not None:
                current.lines.append(line)
            elif not sections and self.ROW.match(line):
                rows.append(self.parse_row(n, line))
        if not rows and not sections:
            if NO_FINDINGS not in "\n".join(line for _, line in lines):
                self.error(
                    "body",
                    f"Findings lists no finding; write '{NO_FINDINGS}' and what was checked",
                )
            return rows
        ranks = [SEVERITIES.index(r.severity) for r in rows if r.severity in SEVERITIES]
        if ranks != sorted(ranks):
            self.warn("body", "list the findings by severity, most severe first")
        minor_count = sum(1 for r in rows if r.severity == "minor")
        if minor_count > MAX_MINOR_FINDINGS:
            self.warn(
                "body",
                f"review lists {minor_count} minor findings; merge or drop low-value nitpicks so major issues stay in focus",
            )
        starts = sorted(s.n for s in sections.values())
        next_start = {
            start: starts[i + 1] - 1 if i + 1 < len(starts) else len(raw_lines)
            for i, start in enumerate(starts)
        }
        seen: set[str] = set()
        for row in rows:
            if row.id in seen:
                self.error("body", f"line {row.n}: {row.id} has a second table row")
            seen.add(row.id)
            section = sections.get(row.id)
            if row.posted == "inline":
                if section is not None:
                    self.error(
                        "body",
                        f"line {section.n}: {row.id} is posted inline; do not repeat it as a section in the main comment",
                    )
                continue
            if section is None:
                self.error(
                    "body",
                    f"{row.id} is posted here but has no '#### {row.id} · ...' section",
                )
                continue
            if (section.severity, section.category) != (row.severity, row.category):
                self.error(
                    "body",
                    f"line {section.n}: the {row.id} heading says {section.severity} {section.category}, "
                    f"the table says {row.severity} {row.category}",
                )
            self.check_section(
                row, section, raw_lines[section.n : next_start[section.n]]
            )
        for fid, section in sections.items():
            if fid not in seen:
                self.error(
                    "body", f"line {section.n}: {fid} has a section but no table row"
                )
        return rows

    def parse_row(self, n: int, line: str) -> Row:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != TABLE_COLUMNS:
            self.error(
                "body",
                f"line {n}: a findings row needs {TABLE_COLUMNS} cells "
                "(ID, Severity, Category, Location, Summary, Status, Posted), "
                f"found {len(cells)}",
            )
            cells += [""] * TABLE_COLUMNS
        fid, severity, category, location, _, status, posted = cells[:TABLE_COLUMNS]
        for value, allowed, column in (
            (severity, SEVERITIES, "Severity"),
            (category, CATEGORIES, "Category"),
            (status, STATUSES, "Status"),
            (posted, POSTED, "Posted"),
        ):
            if value not in allowed:
                self.error(
                    "body",
                    f"line {n}: {fid} {column} must be one of {allowed}, found {value!r}",
                )
        return Row(n, fid, severity, category, location, posted)

    def check_section(self, row: Row, section: Section, raw_slice: list[str]) -> None:
        text = "\n".join(line.strip() for line in section.lines)
        where = f"body {row.id}"
        if not any(
            line.startswith(EVIDENCE_LEAD) and line[len(EVIDENCE_LEAD) :].strip()
            for line in text.splitlines()
        ):
            self.error(where, f"the section needs an '{EVIDENCE_LEAD}' paragraph")
        if FIX_LEAD not in text:
            self.error(where, f"a finding posted here needs a '{FIX_LEAD}' paragraph")
        if any(line.strip() == WALKTHROUGH_LEAD for line in section.lines) and not any(
            Markdown.FENCE.match(line) for line in raw_slice
        ):
            self.warn(
                where,
                f"'{WALKTHROUGH_LEAD}' has no fenced code block; quote the relevant lines under path:line",
            )

    def check_inline_rows(self, rows: list[Row]) -> None:
        """Pairs every inline row with the inline comment at its location."""
        comments = self.payload.get("comments") or []
        matched: set[int] = set()
        for row in rows:
            if row.posted != "inline":
                continue
            m = self.LOCATION.search(row.location)
            if not m:
                self.error(
                    f"body {row.id}",
                    "an inline finding's Location must be `path:line` or `path:start-end`",
                )
                continue
            path, start, end = m[1], int(m[2]), int(m[3] or m[2])
            hits = [
                i
                for i, c in enumerate(comments)
                if c.get("path") == path
                and c.get("line") == end
                and (c.get("start_line") or end) == start
            ]
            if not hits:
                self.error(
                    f"body {row.id}",
                    f"posted inline at {m[0]}, but no inline comment sits there",
                )
            for i in hits:
                first = next(
                    (
                        line.strip()
                        for line in (comments[i].get("body") or "").splitlines()
                        if line.strip()
                    ),
                    "",
                )
                hm = self.INLINE_HEADING.match(first)
                if hm and (hm[1], hm[2], hm[3]) != (
                    row.id,
                    row.severity,
                    row.category,
                ):
                    self.error(
                        f"comment {i + 1} ({path}:{end})",
                        f"heading says {hm[1]} {hm[2]} {hm[3]}, "
                        f"the table says {row.id} {row.severity} {row.category}",
                    )
            matched.update(hits)
        for i, c in enumerate(comments):
            if i not in matched:
                self.error(
                    f"comment {i + 1} ({c.get('path')}:{c.get('line')})",
                    "has no inline row in the findings table; every inline comment needs an inline row in the findings table",
                )

    def check_reproduction(self, lines: list[tuple[int, str]]) -> None:
        text = [line.strip() for _, line in lines]
        for label in REPRO_LABELS:
            matches = [
                t for t in text if t.startswith(label) and t[len(label) :].strip()
            ]
            if not matches:
                self.error("body", f"Reproduction needs a '{label}' bullet")
                continue
            bullet = matches[0]
            content = bullet[len(label) :].strip()
            if label == "- Result:" and not content.startswith(REPRO_STATUSES):
                self.warn(
                    "body",
                    f"start '- Result:' with one of {', '.join(REPRO_STATUSES)} so the outcome is clear at a glance",
                )
            if len(bullet) > MAX_REPRO_BULLET:
                self.warn(
                    "body",
                    f"Reproduction '{label}' bullet is {len(bullet)} characters (limit {MAX_REPRO_BULLET}); "
                    "keep only outcome, key metrics, and essential hardware/command",
                )

    def check_comment(self, i: int, c: dict, seen: set) -> None:
        path, line, body = c.get("path"), c.get("line"), c.get("body") or ""
        side = c.get("side", "RIGHT")
        where = f"comment {i} ({path}:{line})"
        extra = set(c) - COMMENT_KEYS
        if extra:
            self.error(
                where, f"unknown keys {sorted(extra)}; expected {sorted(COMMENT_KEYS)}"
            )
        if not isinstance(path, str) or not isinstance(line, int) or not body.strip():
            self.error(
                where, "needs a string path, an integer line, and a non-empty body"
            )
            return
        if side not in ("RIGHT", "LEFT") or c.get("start_side", side) != side:
            self.error(
                where, f"side and start_side must both be RIGHT or LEFT, found {side!r}"
            )
        start = c.get("start_line")
        if start is not None and (not isinstance(start, int) or start >= line):
            self.error(
                where,
                f"start_line must be an integer below line {line}, found {start!r}",
            )
        first = next((line.strip() for line in body.splitlines() if line.strip()), "")
        if not self.INLINE_HEADING.match(first):
            self.error(
                where,
                "start the inline comment with '**F<n> · <severity> · <category>: <title>**' matching its table row",
            )
        if FIX_LEAD not in body:
            self.error(where, f"every inline comment needs a '{FIX_LEAD}' paragraph")
        if not any(
            line.strip().startswith(EVIDENCE_LEAD)
            and line.strip()[len(EVIDENCE_LEAD) :].strip()
            for _, line in Markdown.unfenced(body)
        ):
            self.error(where, f"every inline comment needs an '{EVIDENCE_LEAD}' line")
        if "```suggestion" in body and side == "LEFT":
            self.error(
                where, "a suggestion block only applies to RIGHT-side (new) lines"
            )
        self.check_prose(where, body)
        if Markdown.prose_length(body) > MAX_COMMENT_PROSE:
            self.warn(
                where,
                f"prose is over {MAX_COMMENT_PROSE} characters; keep the explanation to 1-3 sentences, a short fix, and a one-line Evidence",
            )
        key = (path, side, line)
        if key in seen:
            self.warn(
                where,
                "another comment in this review sits on the same line; merge them",
            )
        seen.add(key)
        if self.pr:
            self.check_location(where, path, side, start, line)

    def check_location(
        self, where: str, path: str, side: str, start: int | None, line: int
    ) -> None:
        lines = self.pr.diff_lines.get((path, side))
        if lines is None:
            self.error(
                where,
                "the PR does not change this file; post the finding in the main comment",
            )
            return
        for n in (start, line):
            if n is not None and n not in lines:
                self.error(
                    where,
                    f"{side} line {n} is outside the diff hunks; GitHub rejects the whole review",
                )
        if start in lines and line in lines and lines[start] != lines[line]:
            self.error(where, f"lines {start}-{line} span two diff hunks")
        author = self.pr.commented_lines.get((path, side, line))
        if author:
            self.warn(
                where,
                f"@{author} already has a thread on this line; reply there instead",
            )


def report(findings: list[Finding], payload: dict, checked_lines: bool) -> int:
    errors = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]
    n = len(payload.get("comments") or [])
    print(f"review: {payload.get('event', 'COMMENT')}, body + {n} inline comment(s)")
    print(f"{len(errors)} error(s), {len(warns)} warning(s)")
    for f in errors + warns:
        print(f"[{f.level[0]}] {f.where}: {f.message}")
    if not checked_lines:
        print("note: comment lines and existing threads were not checked; pass --pr")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default="RLinf/RLinf", help="owner/name of the PR's repo")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template", help="print the main-comment skeleton")
    for name, help_text in (
        ("lint", "check a review payload"),
        ("post", "lint, then post"),
        ("update", "lint, then replace the body of a posted review"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--pr", type=int, required=name != "lint", help="PR number")
        if name == "update":
            cmd.add_argument("--review", type=int, required=True, help="review id")
        cmd.add_argument("payload", help="review payload JSON file")
    args = p.parse_args(argv)

    if args.cmd == "template":
        print(TEMPLATE, end="")
        return 0

    with open(args.payload, encoding="utf-8") as fh:
        payload = json.load(fh)
    pr = PullRequest(args.repo, args.pr) if args.pr else None
    if args.cmd == "update":
        # Only the body changes; its findings table must match the inline
        # comments the review already holds, which sit on checked lines.
        payload = {"body": payload.get("body") or ""}
        payload["comments"] = pr.review_comments(args.review)
        status = report(ReviewLinter(payload, None).run(), payload, True)
        if not status:
            print(f"updated: {pr.update(args.review, payload['body'])}")
        return status
    status = report(ReviewLinter(payload, pr).run(), payload, pr is not None)
    if args.cmd == "lint" or status:
        return status
    payload.setdefault("commit_id", pr.head_sha)
    payload.setdefault("event", "COMMENT")
    print(f"posted: {pr.post(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
