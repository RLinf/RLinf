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

"""Mechanical checks for recipe and evaluation-guide pages.

Enforces the onboarding and command conventions across ``examples/`` and
``evaluations/``:

- Copy-pasteable ``install.sh`` and ``hf download`` commands with no ``<...>``
  placeholders, no preceding ``cd``, and no ``--local-dir .`` / ``/path/to``.
- All ``install.sh`` and ``hf download`` blocks wrapped in ``.. tab-set::``
  (even for a single option) outside hardware-backend subsections.
- No legacy download commands (``huggingface-cli``, ``pip install huggingface-hub``,
  ``git lfs``), no ``CUDA_VISIBLE_DEVICES`` advice, and no repeated mirror
  comments inside bash code blocks.
- No decorative downloaded-file directory trees (``├──`` / ``└──``), no
  boilerplate ``runner.logger`` / ``video_base_dir`` YAML blocks, and no stale
  ``rollout.model.model_path`` alongside ``actor.model.model_path``.
- New documentation diagrams/images belong in ``RLinf/misc`` rather than under
  ``docs/**/_static/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import route

RECIPE_OR_EVAL_PAGE = re.compile(
    r"^docs/source-(?:en|zh)/rst_source/(?:start|examples|evaluations)/(?!reference/).+\.rst$"
)
SKIP_NAME = re.compile(r"(?:^|/)(?:index|_[^/]+|[^/]+_index)\.rst$")
HARDWARE_SECTION = re.compile(
    r"^(?:Run on Different Hardware Backends|在不同硬件后端上运行)\s*$",
    re.MULTILINE,
)
CODE_BLOCK = re.compile(r"^(\s*)\.\.\s+code(?:-block)?::\s*(\S*)\s*$")
TAB_SET = re.compile(r"^(\s*)\.\.\s+tab-set::\s*$")
TAB_ITEM = re.compile(r"^(\s*)\.\.\s+tab-item::\s*(.+?)\s*$")
DROPDOWN = re.compile(r"^(\s*)\.\.\s+dropdown::\s*(.+?)\s*$")
HEADING_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
HARDWARE_TAB_TITLE = re.compile(
    r"(?:AMD\s+ROCm|Ascend|昇腾|MUSA|Moore\s+Threads|摩尔线程)", re.IGNORECASE
)
DOCKER_TAB_OR_DROPDOWN = re.compile(
    r"^(?:Docker(?:\s+Image)?|使用\s*Docker|Docker\s*镜像)", re.IGNORECASE
)
HARDCODED_RLINF_IMAGE = re.compile(r"\brlinf/rlinf:[A-Za-z0-9_.-]+")

PLACEHOLDER = re.compile(r"<[A-Za-z0-9_./-]+>")
MIRROR_COMMENT = re.compile(
    r"^\s*#.*(?:--use-mirror|HF_ENDPOINT|hf-mirror|gh-proxy)", re.IGNORECASE
)
LEGACY_CMD = (
    (
        re.compile(r"\bhuggingface-cli\s+download\b"),
        "use ``hf download`` instead of ``huggingface-cli download``",
    ),
    (
        re.compile(r"\bpip\s+install\b[^\n]*huggingface[-_]hub"),
        "RLinf environments already install ``hf``; drop ``pip install huggingface-hub``",
    ),
    (
        re.compile(r"\bgit\s+lfs\s+(?:install|clone)\b"),
        "use ``hf download`` instead of ``git lfs``",
    ),
    (
        re.compile(r"\bgit\s+clone\s+https://huggingface\.co/"),
        "use ``hf download`` instead of cloning from Hugging Face",
    ),
    (
        re.compile(r"\bCUDA_VISIBLE_DEVICES\b"),
        "use ``cluster.component_placement`` instead of ``CUDA_VISIBLE_DEVICES``",
    ),
)
TREE_LINE = re.compile(r"^\s*.*(?:├──|└──)")
LOCAL_STATIC_IMAGE = re.compile(
    r"^docs/source-(?:en|zh)/_static/.*\.(?:svg|png|jpe?g|gif|webp|mp4)$",
    re.IGNORECASE,
)


def _iter_code_blocks(lines: list[str], cutoff_line: int) -> list[dict]:
    """Extract code blocks before ``cutoff_line`` with enclosing container info."""
    blocks: list[dict] = []
    # Track active directive scopes as (indent, kind)
    scopes: list[tuple[int, str]] = []
    i = 0
    total = min(len(lines), cutoff_line)
    while i < total:
        line = lines[i]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip()) if stripped else 0

        if stripped:
            while scopes and indent <= scopes[-1][0]:
                scopes.pop()

        if TAB_SET.match(line):
            scopes.append((indent, "tab-set"))
            i += 1
            continue
        if DROPDOWN.match(line):
            scopes.append((indent, "dropdown"))
            i += 1
            continue

        match = CODE_BLOCK.match(line)
        if match:
            block_indent = len(match.group(1))
            lang = (match.group(2) or "").lower()
            in_tabset = any(kind == "tab-set" for _, kind in scopes)
            in_dropdown = any(kind == "dropdown" for _, kind in scopes)
            start = i + 1  # 1-based line number of directive
            body: list[tuple[int, str]] = []
            i += 1
            # Skip directive options like :caption:
            while i < total and lines[i].strip().startswith(":"):
                i += 1
            while i < total:
                cur = lines[i]
                if cur.strip() == "":
                    body.append((i + 1, ""))
                    i += 1
                    continue
                cur_indent = len(cur) - len(cur.lstrip())
                if cur_indent <= block_indent:
                    break
                body.append((i + 1, cur))
                i += 1
            blocks.append(
                {
                    "line": start,
                    "lang": lang,
                    "body": body,
                    "in_tabset": in_tabset,
                    "in_dropdown": in_dropdown,
                }
            )
            continue
        i += 1
    return blocks


def _find_nested_tabsets(lines: list[str], cutoff_line: int) -> list[int]:
    """Return 1-based line numbers of nested ``.. tab-set::`` directives."""
    nested: list[int] = []
    scopes: list[tuple[int, str]] = []
    for idx, line in enumerate(lines[:cutoff_line], 1):
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        while scopes and indent <= scopes[-1][0]:
            scopes.pop()
        if TAB_SET.match(line):
            if any(kind == "tab-set" for _, kind in scopes):
                nested.append(idx)
            scopes.append((indent, "tab-set"))
    return nested


def _hardware_cutoff(lines: list[str]) -> int:
    """Line index (0-based) where the hardware-backends section begins, or len(lines)."""
    for idx in range(len(lines) - 1):
        if HARDWARE_SECTION.match(lines[idx]) and HEADING_UNDERLINE.match(
            lines[idx + 1]
        ):
            return idx
    return len(lines)


def check_static_assets(changes: list[route.Change]) -> list[route.Finding]:
    """Flag newly added images/diagrams under ``docs/**/_static/``."""
    findings: list[route.Finding] = []
    for change in changes:
        if change.status == "A" and LOCAL_STATIC_IMAGE.match(change.path):
            findings.append(
                route.Finding(
                    "major",
                    change.path,
                    None,
                    "docs/local-asset",
                    "upload documentation figures/diagrams to RLinf/misc under pic/ "
                    "and reference via raw.githubusercontent.com/RLinf/misc/main/pic/...",
                )
            )
    return findings


def check_page(path: Path, rel: str) -> list[route.Finding]:
    """Run strict recipe and evaluation-guide harness checks on one RST page."""
    if not RECIPE_OR_EVAL_PAGE.match(rel) or SKIP_NAME.search(rel):
        return []

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    cutoff = _hardware_cutoff(lines)
    blocks = _iter_code_blocks(lines, cutoff)
    findings: list[route.Finding] = []

    def add(severity: str, line: int | None, rule: str, message: str) -> None:
        findings.append(route.Finding(severity, rel, line, rule, message))

    for lineno in _find_nested_tabsets(lines, cutoff):
        add(
            "major",
            lineno,
            "docs/nested-tabset",
            "avoid nested ``.. tab-set::`` directives (stacked tab bars hurt readability); "
            "flatten into one single-level ``.. tab-set::`` or fold secondary options into ``.. dropdown::``",
        )

    # Check for dedicated Docker tabs/dropdowns or hardcoded RLinf image tags
    # outside non-NVIDIA hardware-backend tabs (ROCm/Ascend/MUSA).
    hw_tab_indent: int | None = None
    for idx, raw in enumerate(lines[:cutoff], 1):
        stripped = raw.strip()
        if not stripped:
            continue
        indent = len(raw) - len(raw.lstrip())
        if hw_tab_indent is not None and indent <= hw_tab_indent:
            hw_tab_indent = None
        tab_match = TAB_ITEM.match(raw)
        if tab_match:
            title = tab_match.group(2)
            if HARDWARE_TAB_TITLE.search(title):
                hw_tab_indent = len(tab_match.group(1))
                continue
            if DOCKER_TAB_OR_DROPDOWN.search(title):
                add(
                    "major",
                    idx,
                    "docs/hardcoded-docker",
                    "do not add a dedicated Docker tab; link to Docker Hub "
                    "(https://hub.docker.com/r/rlinf/rlinf) via _setup_common.rst or start/installation.rst",
                )
            continue
        if hw_tab_indent is not None:
            continue
        dd_match = DROPDOWN.match(raw)
        if dd_match and DOCKER_TAB_OR_DROPDOWN.search(dd_match.group(2)):
            add(
                "major",
                idx,
                "docs/hardcoded-docker",
                "do not add a dedicated Docker dropdown; link to Docker Hub "
                "(https://hub.docker.com/r/rlinf/rlinf) via _setup_common.rst or start/installation.rst",
            )
            continue
        if HARDCODED_RLINF_IMAGE.search(raw):
            add(
                "major",
                idx,
                "docs/hardcoded-docker",
                "avoid hardcoded ``rlinf/rlinf:<tag>`` image tags in recipes/eval guides; "
                "link to Docker Hub (https://hub.docker.com/r/rlinf/rlinf) instead",
            )

    for block in blocks:
        lang = block["lang"]
        body = block["body"]
        text = "\n".join(l for _, l in body)

        # 1. Shell / bash command blocks
        if lang in ("bash", "sh", "shell", "console", ""):
            has_install = False
            has_hf = False
            prev_non_empty = ""
            for lineno, raw in body:
                stripped = raw.strip()
                if not stripped:
                    continue

                if MIRROR_COMMENT.match(raw):
                    add(
                        "major",
                        lineno,
                        "docs/mirror-comment",
                        "remove commented mirror lines inside bash blocks; state "
                        "--use-mirror and HF_ENDPOINT once in prose or _setup_common.rst",
                    )

                for pat, msg in LEGACY_CMD:
                    if pat.search(stripped):
                        add("major", lineno, "docs/command-legacy", msg)

                if "requirements/install.sh" in stripped and not stripped.startswith(
                    "#"
                ):
                    has_install = True
                    if PLACEHOLDER.search(stripped):
                        add(
                            "major",
                            lineno,
                            "docs/command-placeholder",
                            "install.sh command contains a <...> placeholder; "
                            "provide a directly copy-pasteable command",
                        )

                if re.search(
                    r"\bhf\s+download\b", stripped
                ) and not stripped.startswith("#"):
                    has_hf = True
                    if PLACEHOLDER.search(stripped):
                        add(
                            "major",
                            lineno,
                            "docs/command-placeholder",
                            "hf download command contains a <...> placeholder; "
                            "provide a directly copy-pasteable command",
                        )
                    if re.match(r"^cd\s+", prev_non_empty):
                        add(
                            "major",
                            lineno,
                            "docs/command-cd",
                            "avoid ``cd`` before ``hf download``; pass ``--local-dir ./<name>`` "
                            "(or the repo-relative path) directly from the repo root",
                        )
                    if re.search(r"--local-dir\s+(?:\.(?:\s|$)|/path/to/)", stripped):
                        add(
                            "major",
                            lineno,
                            "docs/command-cd",
                            "use a concrete local directory like ``--local-dir ./<name>`` "
                            "instead of ``--local-dir .`` or ``/path/to/...``",
                        )

                prev_non_empty = stripped

            if (
                (has_install or has_hf)
                and not block["in_tabset"]
                and not block["in_dropdown"]
            ):
                add(
                    "major",
                    block["line"],
                    "docs/command-tabset",
                    "wrap install.sh and hf download commands in ``.. tab-set::`` "
                    "(even for a single option) and group companion setup/download commands together",
                )

        # 2. Decorative ASCII directory trees
        if lang in ("text", ""):
            tree_lines = [lineno for lineno, raw in body if TREE_LINE.match(raw)]
            if len(tree_lines) >= 3 and not block["in_dropdown"]:
                add(
                    "major",
                    tree_lines[0],
                    "docs/dir-tree",
                    "remove decorative downloaded-file directory trees; keep only "
                    "essential path instructions",
                )

        # 3. YAML blocks: boilerplate & stale rollout.model.model_path
        if lang in ("yaml", "yml"):
            non_empty = [l for _, l in body if l.strip()]
            has_rollout_mp = bool(
                re.search(r"rollout:\s*\n\s+model:\s*\n\s+model_path:", text)
            )
            has_actor_mp = bool(
                re.search(r"actor:\s*\n\s+model:\s*\n\s+model_path:", text)
            )
            if has_rollout_mp and has_actor_mp:
                add(
                    "major",
                    block["line"],
                    "docs/boilerplate-yaml",
                    "omit ``rollout.model.model_path`` when ``actor.model.model_path`` "
                    "is set; rollout inherits it automatically",
                )
            if "logger_backends:" in text and not block["in_dropdown"]:
                add(
                    "major",
                    block["line"],
                    "docs/boilerplate-yaml",
                    "do not repeat boilerplate ``runner.logger`` YAML blocks; link to "
                    "``reference/metrics`` in one sentence",
                )
            if "video_base_dir:" in text and not block["in_dropdown"]:
                add(
                    "major",
                    block["line"],
                    "docs/boilerplate-yaml",
                    "do not repeat boilerplate ``video_cfg`` YAML blocks; mention "
                    "``env.eval.video_cfg.save_video=True`` in one sentence",
                )
            if len(non_empty) >= 25 and not block["in_dropdown"]:
                add(
                    "minor",
                    block["line"],
                    "docs/long-yaml",
                    f"YAML block has {len(non_empty)} lines outside ``.. dropdown::``; "
                    "show only required paths up front and fold optional tuning into ``.. dropdown::``",
                )

    return findings
