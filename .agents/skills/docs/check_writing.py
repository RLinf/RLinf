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
RECIPE_PAGE = re.compile(
    r"docs/source-(en|zh)/rst_source/"
    r"(?:examples/(?:embodied|agentic)/|evaluations/guides/)(?!_).+\.rst$"
)
AGENTIC_PROJECT_INDEX = re.compile(
    r"docs/source-(?:en|zh)/rst_source/examples/agentic/"
    r"(?!agents/|math_reasoning/)[^/]+/index\.rst$"
)
RECIPE_NON_RUNNABLE = ("retriever.rst", "tools.rst", "video_models.rst")
RECIPE_SECTIONS = {"en": ("Setup and Run",), "zh": ("安装与运行",)}
LEGACY_SPLIT_SECTIONS = {
    "en": {
        "Run It": "fold launch commands into the unified 'Setup and Run' tab-set instead of a separate 'Run It' section",
        "Run on Different Hardware Backends": "place non-CUDA backends as flat tabs (:sync: rocm/ascend/musa) inside 'Setup and Run' with .. _<recipe>-hardware: above it",
    },
    "zh": {
        "运行": "将启动命令整合到统一的“安装与运行”tab-set 中，不再单独拆出“运行”大节",
        "在不同硬件后端上运行": "将非 CUDA 后端以平铺标签页（:sync: rocm/ascend/musa）并入“安装与运行”，并在其上方放置 .. _<recipe>-hardware: 锚点",
    },
}
AGENTIC_OVERVIEW_SECTIONS = {"en": "Overview", "zh": "概览"}
GENERIC_SECTIONS = {
    "Algorithm",
    "Environment",
    "Model Architecture",
    "Technical Details",
    "GRPO Principle",
    "How GRPO Works",
    "How PPO Works",
    "算法",
    "环境",
    "模型架构",
    "技术细节",
    "GRPO 原理",
    "GRPO 工作方式",
    "PPO 工作方式",
}
ZH_TERM_HEADING = (
    (
        re.compile(r"优势函数|策略损失|损失函数|规则奖励|奖励模型|新增环境"),
        "keep Advantage / Policy Loss / Reward / Env in Chinese headings (rules/glossary.md)",
    ),
)

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
    r"in order to|allows you to|enables you to|a wide range of|currently|not just)\b",
    re.IGNORECASE,
)
JARGON_PATTERNS = (
    (
        re.compile(
            r"冒烟测试|冒烟验证|冒烟检查|\bsmoke[\s-]*(?:test(?:ed|ing)?|check|run)\b",
            re.IGNORECASE,
        ),
        "avoid testing jargon (smoke test / 冒烟测试); say quick check / 快速验证 or 自检",
    ),
    (
        re.compile(r"采数|数采"),
        "use the formal term 数据采集 instead of 采数 / 数采",
    ),
    (
        re.compile(r"遥操(?!作)"),
        "use the formal term 遥操作 instead of 遥操",
    ),
)
TODO_PATTERN = re.compile(r"\b(?:TODO|TBD|FIXME)\b")
SKIP_TODO_PATHS = (
    "/resources/release_v0",
    "/resources/blog/",
    "/resources/publications/",
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
        re.compile("(?:不是|并非)[^。！？\n]{1,40}而是"),
        "state directly what it is instead of framing it as 不是……而是……",
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
    source = path.read_text(encoding="utf-8")
    is_agentic_leaf_index = bool(
        AGENTIC_PROJECT_INDEX.match(rel) and ".. toctree::" not in source
    )
    kind = "recipe" if is_agentic_leaf_index else page_type(rel)
    lang = "zh" if rel.startswith("docs/source-zh/") else "en"
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
    skip_heading_len = any(
        s in rel
        for s in (
            "/faq.rst",
            "/publications/",
            "/blog/",
            "/releases",
            "/reference/algorithms/",
        )
    )
    if not skip_heading_len:
        for number, title in headings(lines):
            clean_title = MARKUP.sub("", title).strip()
            if lang == "zh":
                # Count CJK as 1 and ASCII as 0.5 so mixed ZH/EN titles are judged fairly.
                eff_len = sum(0.5 if ord(ch) < 128 else 1.0 for ch in clean_title)
                if eff_len > 20:
                    add(
                        "minor",
                        number,
                        "docs/heading-length",
                        f"heading is too long ({int(eff_len)} CJK-equiv chars); "
                        "keep titles concise and move secondary details into the body",
                    )
            elif len(clean_title) > 48:
                add(
                    "minor",
                    number,
                    "docs/heading-length",
                    f"heading is too long ({len(clean_title)} chars); "
                    "keep titles concise (<= 48 chars) and move details into the body",
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
    if lang == "zh":
        for number, title in headings(lines):
            if title == "一键上手":
                add(
                    "major",
                    number,
                    "docs/structure",
                    "use '安装与运行' instead of '一键上手' to align with English 'Setup and Run'",
                )
            for pattern, advice in ZH_TERM_HEADING:
                if pattern.search(title):
                    add("minor", number, "docs/zh-term", advice)

    is_index = rel.endswith(("/index.rst", "_index.rst")) and not is_agentic_leaf_index
    is_agentic = "/rst_source/examples/agentic/" in rel
    is_runnable_recipe = (
        not is_index
        and ":orphan:" not in source[:240]
        and bool(RECIPE_PAGE.match(rel))
        and not rel.endswith(RECIPE_NON_RUNNABLE)
    )
    if is_runnable_recipe:
        page_headings = headings(lines)
        titles = {title for _, title in page_headings}
        required = RECIPE_SECTIONS[lang]
        legacy_map = LEGACY_SPLIT_SECTIONS[lang]
        for number, title in page_headings:
            if title in legacy_map:
                if title in ("Run It", "运行") and required[0] not in titles:
                    continue
                add("major", number, "docs/structure", legacy_map[title])
    if (new or is_agentic) and is_runnable_recipe:
        titles = {title for _, title in headings(lines)}
        required = RECIPE_SECTIONS[lang]
        missing = [s for s in required if s not in titles]
        if missing:
            add(
                "major",
                None,
                "docs/structure",
                f"recipe page needs the unified {required[0]!r} section (rules/recipe.md)",
            )
        if is_agentic:
            ov = AGENTIC_OVERVIEW_SECTIONS[lang]
            if ov not in titles:
                add(
                    "major",
                    None,
                    "docs/structure",
                    f"Agent & LLM recipe needs the unified {ov!r} section matching Embodied AI recipes",
                )
        for generic in sorted(titles & GENERIC_SECTIONS):
            add(
                "major",
                None,
                "docs/structure",
                f"remove the generic theory section {generic!r}; link to Reference instead",
            )

    if not any(skip in rel for skip in SKIP_TODO_PATHS):
        for number, line in enumerate(lines, 1):
            if TODO_PATTERN.search(line):
                add(
                    "major",
                    number,
                    "docs/no-todo",
                    "remove unfinished TODO / TBD / FIXME marker from published docs",
                )
            for pattern, advice in JARGON_PATTERNS:
                if pattern.search(line):
                    add("major", number, "docs/jargon", advice)

    # Apply code-density guard across ALL doc pages so prose never turns into
    # a wall of backticked variables, functions, classes, or paths.
    list_item_re = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
    for block in check_rst_markup.prose_blocks(source):
        if any(
            row.lstrip().startswith(("* -", "| ", "+-", "=="))
            for row in block.text.splitlines()
        ):
            continue
        chunks: list[tuple[int, str]] = []
        cur_lines: list[str] = []
        cur_start = block.lines[0]
        for line_no, raw_ln in zip(block.lines, block.text.splitlines()):
            if list_item_re.match(raw_ln) and cur_lines:
                chunks.append((cur_start, "\n".join(cur_lines)))
                cur_lines = [raw_ln]
                cur_start = line_no
            else:
                cur_lines.append(raw_ln)
        if cur_lines:
            chunks.append((cur_start, "\n".join(cur_lines)))

        for start_line, chunk in chunks:
            plain = MARKUP.sub(" X ", chunk)
            inline_literals = LITERAL.findall(chunk)
            if len(inline_literals) >= 6:
                add(
                    "minor",
                    start_line,
                    "docs/inline-code-density",
                    f"paragraph packs {len(inline_literals)} inline code literals; "
                    "avoid stacking variables, functions, classes, or paths in prose—"
                    "move structured specs into a table or simplify",
                )
            if kind != "publication":
                if lang == "zh":
                    char_count = len(re.sub(r"\s+", "", plain))
                    if char_count > 260:
                        add(
                            "minor",
                            start_line,
                            "docs/long-paragraph",
                            f"paragraph has {char_count} Chinese characters on a {kind} page; "
                            "split into short scannable paragraphs or a table",
                        )
                else:
                    word_count = len(plain.split())
                    if word_count > 120:
                        add(
                            "minor",
                            start_line,
                            "docs/long-paragraph",
                            f"paragraph has {word_count} words on a {kind} page; "
                            "split into short scannable paragraphs or a table",
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
