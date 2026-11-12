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

"""Writing rules that a script can see, applied by page type.

Each page's type comes from where it lives (``rules/base.md``). These rules
catch the mechanical forms of the mistakes in ``rules/mistakes.md``: internal
code names and equations on pages for users, missing recipe sections, code in
headings, and English and Chinese phrasing that the language rules ban. They
are heuristics; ``docs/check.py`` runs them on the lines a change adds.
"""

from __future__ import annotations

import re
from pathlib import Path

import check_rst_markup
import route

TYPE_PREFIXES = (
    ("evaluations/get_started/", "recipe"),
    ("evaluations/guides/", "guide"),
    ("evaluations/reference/", "reference"),
    ("examples/", "recipe"),
    ("start/", "recipe"),
    ("guides/", "guide"),
    ("concepts/", "concept"),
    ("extending/", "extending"),
    ("reference/", "reference"),
    ("resources/publications/", "publication"),
)
GALLERY_PAGE = re.compile(
    r"docs/source-(en|zh)/rst_source/examples/embodied/(?!_)[^/]+\.rst"
)
RECIPE_SECTIONS = {"en": ("Installation", "Run It"), "zh": ("安装", "运行")}
GENERIC_SECTIONS = {"Algorithm", "Environment", "算法", "环境"}

#: Severity of an internal code name, by page type; other types are not checked.
CODE_NAME_SEVERITY = {"recipe": "major", "guide": "major", "concept": "minor"}

LITERAL = re.compile(r"``([^`\n]+)``")
#: A name that reads as code rather than as a word or a product: CamelCase with
#: an inner capital, or snake_case, at least four characters long.
CODE_LIKE = re.compile(
    r"^(?=.{4,})(?:[A-Za-z0-9]*[a-z][A-Z][A-Za-z0-9]*|[a-z0-9]+(?:_[a-z0-9]+)+)$"
)
FILE_NAME = re.compile(r"\.[A-Za-z]{1,12}$")
DEFINITION = re.compile(r"^\s*(?:class|def)\s+([A-Za-z_]\w*)", re.MULTILINE)
YAML_KEY = re.compile(r"^\s*([A-Za-z_]\w*)\s*:", re.MULTILINE)
UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
MATH = re.compile(r"^\s*\.\. math::|:math:`")
MARKUP = re.compile(r"``[^`]*``|:\w+:`[^`]*`|`[^`]*`_{0,2}|\*\*|\|")

EN_PHRASES = re.compile(
    r"\b(simply|easily|obviously|basically|seamless(?:ly)?|powerful|robust|"
    r"comprehensive|cutting-edge|state-of-the-art|leverag(?:e|es|ed|ing)|crucial|"
    r"pivotal|delve|showcases?|empowers?|it is worth noting|please note|note that|"
    r"in order to|allows you to|enables you to|a wide range of|currently)\b",
    re.IGNORECASE,
)
HAN = r"一-鿿"
ZH_PATTERNS = (
    (re.compile("您"), "address the reader as 你, or omit the subject"),
    (re.compile("[「」]"), "use “” for quotes"),
    (
        re.compile("值得注意的是|需要注意的是"),
        "state the fact, or move it into a note or warning",
    ),
    (re.compile("进行"), "use the verb directly: 进行训练 → 训练"),
    (
        re.compile("当[^，。；：！？]{1,30}时"),
        "drop 当……时 where ……后 or ……时 alone reads naturally",
    ),
    (
        re.compile(f"(?<=[{HAN}\u3001-\u301f\uff01-\uff5e])[,;:!?](?=\\s|$|[{HAN}])"),
        "use full-width punctuation in a Chinese sentence",
    ),
    (
        re.compile(f"[{HAN}][A-Za-z0-9]|[A-Za-z0-9][{HAN}]"),
        "put a space between Chinese and English or digits",
    ),
)


def page_type(rel: str) -> str | None:
    """The page type of a docs source file, from the axis it lives in."""
    match = re.match(r"docs/source-(?:en|zh)/rst_source/(.+\.rst)$", rel)
    if not match:
        return None
    name = match[1].rsplit("/", 1)[-1]
    if name == "index.rst" or name.endswith("_index.rst"):
        return "index"
    return next(
        (kind for prefix, kind in TYPE_PREFIXES if match[1].startswith(prefix)), "other"
    )


def code_names(root: Path) -> frozenset[str]:
    """Classes and functions defined under ``rlinf/``, minus names used as config keys."""
    defined, keys = set(), set()
    for path in (root / "rlinf").rglob("*.py"):
        defined.update(
            DEFINITION.findall(path.read_text(encoding="utf-8", errors="ignore"))
        )
    for folder in ("examples", "rlinf", "tests/e2e_tests"):
        for path in (root / folder).rglob("*.yaml"):
            keys.update(
                YAML_KEY.findall(path.read_text(encoding="utf-8", errors="ignore"))
            )
    return frozenset(n for n in defined - keys if CODE_LIKE.match(n))


def headings(lines: list[str]) -> list[tuple[int, str]]:
    """``(line, title)`` of every section title: a line underlined by punctuation."""
    found = []
    for i, (title, under) in enumerate(zip(lines, lines[1:]), 1):
        text = title.strip()
        if (
            text
            and not text.startswith("..")
            and UNDERLINE.match(under)
            and len(under.strip()) >= len(text) // 2
        ):
            found.append((i, text))
    return found


def prose_rows(source: str) -> list[tuple[int, str]]:
    """Every prose line with its number, leaving out code and literal blocks."""
    rows = []
    for block in check_rst_markup.prose_blocks(source):
        rows += list(zip(block.lines, block.text.split("\n")))
    return rows


def check_page(
    path: Path, rel: str, names: frozenset[str], new: bool
) -> list[route.Finding]:
    """The writing rules for one page, given its type and whether it is new."""
    kind = page_type(rel)
    lang = "zh" if rel.startswith("docs/source-zh/") else "en"
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    rows = prose_rows(source)
    findings: list[route.Finding] = []

    def add(severity: str, line: int | None, rule: str, message: str) -> None:
        findings.append(route.Finding(severity, rel, line, rule, message))

    if kind in CODE_NAME_SEVERITY:
        for number, text in rows:
            for token in LITERAL.findall(text):
                token = token.strip().removesuffix("()")
                internal = (token.startswith("rlinf/") and ":" not in token) or (
                    "/" not in token
                    and not FILE_NAME.search(token)
                    and any(part in names for part in token.split("."))
                )
                if internal:
                    add(
                        CODE_NAME_SEVERITY[kind],
                        number,
                        "docs/code-name",
                        f"``{token}`` is RLinf's internal code, which the reader never "
                        "touches on this page; describe what it does instead",
                    )
    if kind in ("recipe", "guide"):
        for number, line in enumerate(lines, 1):
            if MATH.search(line):
                add(
                    "major",
                    number,
                    "docs/math",
                    "equations belong in Reference; link to it",
                )
    if kind not in ("reference", "extending"):
        for number, title in headings(lines):
            if "``" in title:
                add(
                    "minor",
                    number,
                    "docs/heading",
                    "name the task or idea, not a code identifier",
                )
    if new and GALLERY_PAGE.match(rel):
        titles = {title for _, title in headings(lines)}
        missing = [s for s in RECIPE_SECTIONS[lang] if s not in titles]
        if missing:
            add(
                "major",
                None,
                "docs/structure",
                f"a gallery recipe needs the sections {', '.join(missing)}",
            )
        for generic in sorted(titles & GENERIC_SECTIONS):
            add(
                "major",
                None,
                "docs/structure",
                f"remove the generic {generic!r} section; link to Reference",
            )

    for number, text in rows:
        if lang == "en":
            for match in EN_PHRASES.finditer(text):
                add(
                    "minor",
                    number,
                    "docs/en-style",
                    f"cut or replace {match[0]!r} (rules/english.md)",
                )
        else:
            plain = MARKUP.sub(" ", text)
            for pattern, advice in ZH_PATTERNS:
                if pattern.search(plain):
                    add(
                        "minor", number, "docs/zh-style", f"{advice} (rules/chinese.md)"
                    )
    return findings
