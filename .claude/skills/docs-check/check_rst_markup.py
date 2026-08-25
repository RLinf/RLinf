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

"""Check RST inline markup against the docutils recognition rules.

docutils only recognises inline markup when the start-string is preceded by
whitespace, an *opener* or a *delimiter*, and the end-string is followed by
whitespace, a *closer* or a *delimiter*.  Full-width CJK punctuation splits
across those classes in ways that are easy to get wrong:

    （《        openers  -> fine before markup, NOT allowed after it
    ）》”’      closers  -> fine after markup
    ，、：；。—  delimiters -> fine on either side

So a Chinese page that writes ``` ``KEY``（说明） ``` produces

    WARNING: Inline literal start-string without end-string. [docutils]

which fails the build because both Read the Docs projects set
``fail_on_warning: true`` — and the English page with ASCII parentheses is
unaffected, so the failure shows up as "only the CN docs are broken".

The nastier variant is silent: when a *later* end-string candidate in the same
paragraph does satisfy the suffix rule, docutils closes the literal there and
swallows the intervening prose into the code span.  No warning, wrong page.

This script reports both, plus start-strings that CJK text renders inert
(``通常**并不是**`` renders as literal asterisks).  The fix in every case is an
escaped space — ``\\ `` — between the markup and the adjacent character, the
convention already used across ``docs/source-zh``.

Requires ``docutils`` (``pip install docutils``), whose own punctuation tables
drive the check so it cannot drift from the parser.

Usage::

    python3 .claude/skills/docs-check/check_rst_markup.py            # all docs
    python3 .claude/skills/docs-check/check_rst_markup.py FILE...    # subset
    python3 .claude/skills/docs-check/check_rst_markup.py --errors-only

Exit code is 1 when any ERROR is reported, 0 otherwise (warnings pass).
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOTS = ("docs/source-en", "docs/source-zh")

# Directives whose body is verbatim text, not RST.
LITERAL_DIRECTIVES = frozenset(
    {
        "code",
        "code-block",
        "highlight",
        "literalinclude",
        "math",
        "parsed-literal",
        "raw",
        "sourcecode",
    }
)


def _character_classes() -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Build the character classes docutils uses around inline markup.

    ``Inliner.init_customizations`` interpolates the ``punctuation_chars``
    tables straight into a regex character class, which is not the same as
    membership in those tables — the tables contain ``-`` and so form ranges.
    Reusing the exact expressions keeps this check faithful to the parser.
    """
    try:
        from docutils.utils import punctuation_chars as pc
    except ImportError:  # pragma: no cover - environment problem, not a finding
        sys.exit("error: this check needs docutils; run 'pip install docutils'")

    prefix = re.compile("[\\s%s%s]" % (pc.openers, pc.delimiters))
    suffix = re.compile(
        "[\\s\x00%s%s%s]" % (pc.closing_delimiters, pc.delimiters, pc.closers)
    )
    return prefix, suffix


START_PREFIX_RE, END_SUFFIX_RE = _character_classes()


@dataclass
class Finding:
    path: Path
    line: int
    level: str  # "ERROR" or "WARNING"
    marker: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.level}: {self.message}"


@dataclass
class Block:
    """A paragraph of prose, flattened to one string with a line map."""

    text: str
    starts: list[int]  # index in `text` where each source line begins
    lines: list[int]  # 1-based source line number of each source line

    def line_of(self, index: int) -> int:
        line = self.lines[0]
        for start, number in zip(self.starts, self.lines):
            if start > index:
                break
            line = number
        return line


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def prose_blocks(source: str) -> list[Block]:
    """Split a document into paragraphs, dropping verbatim blocks.

    Skipped: comments, the bodies of literal directives, literal blocks
    introduced by a trailing ``::``, and doctest blocks.
    """
    lines = source.splitlines()
    blocks: list[Block] = []
    buffer: list[tuple[int, str]] = []
    skip_above: int | None = None  # skip lines indented deeper than this
    pending_literal = False

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text_parts, starts, numbers, cursor = [], [], [], 0
        for number, line in buffer:
            starts.append(cursor)
            numbers.append(number)
            text_parts.append(line)
            cursor += len(line) + 1
        blocks.append(Block("\n".join(text_parts), starts, numbers))
        buffer = []

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()

        if skip_above is not None:
            if not stripped or _indent(line) > skip_above:
                continue
            skip_above = None

        if not stripped:
            flush()
            if pending_literal:
                skip_above = _indent(lines[number - 2]) if number >= 2 else 0
                pending_literal = False
            continue

        if stripped.startswith(".."):
            flush()
            name = stripped[2:].strip().split("::", 1)[0].strip()
            is_comment = "::" not in stripped[2:]
            if is_comment or name in LITERAL_DIRECTIVES:
                skip_above = _indent(line)
            continue

        if stripped.startswith(">>>"):
            flush()
            skip_above = _indent(line)
            continue

        buffer.append((number, line))
        # A trailing "::" introduces a literal block after the blank line.
        pending_literal = stripped.endswith("::")

    flush()
    return blocks


def _escaped(text: str, index: int) -> bool:
    backslashes = 0
    while index - backslashes - 1 >= 0 and text[index - backslashes - 1] == "\\":
        backslashes += 1
    return backslashes % 2 == 1


def _occurrences(text: str, marker: str) -> list[int]:
    found, index = [], text.find(marker)
    while index != -1:
        if not _escaped(text, index):
            found.append(index)
        index = text.find(marker, index + len(marker))
    return found


def _valid_start(text: str, index: int, marker: str) -> bool:
    after = index + len(marker)
    if after >= len(text) or text[after].isspace():
        return False
    if index == 0:
        return True
    return START_PREFIX_RE.match(text[index - 1]) is not None


def _valid_end(text: str, index: int, marker: str) -> bool:
    if index == 0 or text[index - 1].isspace():
        return False
    after = index + len(marker)
    if after >= len(text):
        return True
    return END_SUFFIX_RE.match(text[after]) is not None


def _describe(char: str) -> str:
    name = unicodedata.name(char, "unnamed")
    return f"{char!r} (U+{ord(char):04X} {name})"


MARKERS = {"``": "Inline literal", "**": "Strong emphasis"}


def _markers_in(text: str) -> list[tuple[int, str]]:
    """All unescaped marker occurrences, in document order."""
    found: list[tuple[int, str]] = []
    for marker in MARKERS:
        for index in _occurrences(text, marker):
            if marker == "**" and text[index : index + 3] == "***":
                continue  # docutils never reads "***" as strong
            found.append((index, marker))
    return sorted(found)


def check_block(path: Path, block: Block) -> list[Finding]:
    """Emulate the docutils inline scan over one paragraph.

    RST inline markup does not nest: whatever sits between a start-string and
    its end-string is the construct's text, so the scan jumps over a matched
    span instead of looking inside it.  That is what makes
    ``**stuck in ``reach``**`` legal — the backticks are literal characters
    inside the strong, not a literal of their own.
    """
    text = block.text
    findings: list[Finding] = []
    occurrences = _markers_in(text)
    cursor = 0

    for order, (index, marker) in enumerate(occurrences):
        if index < cursor:
            continue
        later = [p for p, m in occurrences[order + 1 :] if m == marker]

        if not _valid_start(text, index, marker):
            # Worth reporting only when the author clearly meant markup, and
            # only for a non-ASCII neighbour: "2**3" is prose, "个**粗体**"
            # is a Chinese page whose bold silently became asterisks.
            partner = next((p for p in later if not text[p - 1].isspace()), None)
            if partner is not None and index > 0 and ord(text[index - 1]) > 0x7F:
                findings.append(
                    Finding(
                        path,
                        block.line_of(index),
                        "WARNING",
                        marker,
                        f"{label(marker)} start-string is preceded by "
                        f"{_describe(text[index - 1])}, so docutils does not see "
                        f"markup here and renders {marker!r} literally; "
                        f"insert an escaped space (backslash-space) before it",
                    )
                )
                cursor = partner + len(marker)
            continue

        candidates = [p for p in later if not text[p - 1].isspace()]
        if not candidates:
            continue
        valid = [p for p in candidates if _valid_end(text, p, marker)]

        if not valid:
            offender = candidates[0] + len(marker)
            suffix = (
                _describe(text[offender]) if offender < len(text) else "end of block"
            )
            findings.append(
                Finding(
                    path,
                    block.line_of(index),
                    "ERROR",
                    marker,
                    f"{label(marker)} start-string without end-string: the closing "
                    f"{marker!r} is followed by {suffix}, which docutils does not "
                    f"accept; insert an escaped space (backslash-space) after it",
                )
            )
            cursor = candidates[0] + len(marker)
            continue

        if valid[0] != candidates[0]:
            offender = candidates[0] + len(marker)
            swallowed = text[candidates[0] + len(marker) : valid[0]].replace("\n", " ")
            findings.append(
                Finding(
                    path,
                    block.line_of(index),
                    "WARNING",
                    marker,
                    f"{label(marker)} closes late and silently swallows "
                    f"{swallowed.strip()!r}: the intended closing {marker!r} is "
                    f"followed by {_describe(text[offender])}; insert an escaped "
                    f"space (backslash-space) after it",
                )
            )
        cursor = valid[0] + len(marker)

    return findings


def label(marker: str) -> str:
    return MARKERS[marker]


def check_file(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    findings: list[Finding] = []
    for block in prose_blocks(source):
        findings += check_block(path, block)
    return sorted(findings, key=lambda f: (f.line, f.level))


def collect(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        candidate = Path(target)
        if candidate.is_dir():
            paths += sorted(candidate.rglob("*.rst"))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            print(f"warning: {target} does not exist", file=sys.stderr)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="RST files or directories (default: docs/source-en docs/source-zh)",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="report only build-breaking findings, not silent mis-renders",
    )
    args = parser.parse_args()

    findings: list[Finding] = []
    for path in collect(args.targets or list(DEFAULT_ROOTS)):
        findings += check_file(path)
    if args.errors_only:
        findings = [f for f in findings if f.level == "ERROR"]

    for finding in findings:
        print(finding.format())

    errors = sum(1 for f in findings if f.level == "ERROR")
    warnings = len(findings) - errors
    if findings:
        print(
            f"\n{errors} error(s) that fail the Sphinx build, "
            f"{warnings} warning(s) that silently mis-render.",
            file=sys.stderr,
        )
    else:
        print("No inline-markup problems found.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
