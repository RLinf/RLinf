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

"""Chapter-specific structural and accuracy checks across RLinf docs.

Extends ``docs/check.py`` beyond recipes so that index pages, extending guides,
operational guides, concept pages, and algorithm/API references are all
enforced automatically during PR review:

- ``docs/raw-html-grid``: forbid raw HTML navigation card grids and hardcoded
  ``color: blue`` links; use ``sphinx-design`` ``.. grid::`` cards.
- ``docs/index-style``: index pages must use ``:hidden:`` toctrees and list
  every toctree target in body cards or tables.
- ``docs/path-exists``: repository file paths cited in prose or commands must
  exist on disk.
- ``docs/extending-no-pytest``: extending docs use standalone local CPU
  verification snippets, not boilerplate ``tests/unit_tests/`` / ``pytest``
  sections.
- ``docs/extending-structure``: extending guides open with an At-a-Glance
  ``list-table`` and close with a verification step.
- ``docs/guide-quality``: catch ultra-thin guide stubs and legacy commands in
  ``guides/``.
- ``docs/concept-purity``: keep ``.. autoclass::``, ultra-thin stubs, and PR
  migration notes out of ``concepts/``.
- ``docs/algo-keys``: YAML keys documented on ``reference/algorithms/`` pages
  must exist in the repository, and each algorithm page must link to a runnable
  config or recipe.
- ``docs/latex-sanity``: catch missing LaTeX subscripts and orphan paper
  equation tags in ``reference/``.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import route

SKIP_HISTORICAL = (
    "/resources/release_v0",
    "/resources/blog/",
    "/resources/publications/",
)

RAW_HTML_BLOCK = re.compile(r"^\s*\.\.\s+raw::\s+html\s*$")
RAW_HTML_BAD = re.compile(
    r"color:\s*blue|display:\s*grid[^\n]*(?:.|\n){0,400}<a\s+href=",
    re.IGNORECASE,
)

REPO_FILE_LITERAL = re.compile(
    r"``((?:rlinf|examples|evaluations|toolkits|requirements)/"
    r"[A-Za-z0-9_./-]+\.(?:py|yaml|yml|sh|toml|json))``"
)
REPO_SCRIPT_CMD = re.compile(
    r"\b(?:bash|sh|python3?)\s+"
    r"((?:examples|evaluations|toolkits|requirements|rlinf)/"
    r"[A-Za-z0-9_./-]+\.(?:sh|py))\b"
)
REPO_DIR_LITERAL = re.compile(
    r"``((?:rlinf|examples|evaluations|toolkits|requirements)/[A-Za-z0-9_/-]+)``"
)
BARE_FILE_REF = re.compile(
    r"(?:``|\*\*)([A-Za-z0-9_-]+\.(?:py|sh|yaml|yml))(?:``|\*\*)"
)
ILLUSTRATIVE_PATH = re.compile(
    r"(?:my_|point_reach|tiny_|custom_|example_|foo|bar|your_|placeholder|xxx|"
    r"dict_sync|mock_|calc_tool_worker|calc_agent_loop|config\.yaml$)"
)

EXTENDING_PYTEST = re.compile(
    r"tests/unit_tests/|\bpytest\b|Add Unit Tests|添加单元测试|Unit Tests\s*$"
)
EXTENDING_VERIFY = re.compile(r"Verif|Check|Test|验证|检查|自检")

CONCEPT_AUTODOC = re.compile(r"^\s*\.\.\s+auto(?:class|function|module)::")
CONCEPT_MIGRATION = re.compile(
    r"is now rejected|replace\s+``[^`]+``\s+with\s+``[^`]+``|"
    r"move from\s+``[^`]+``\s+to\s+``[^`]+``|"
    r"已替换为|不再接受|请将\s*``[^`]+``\s*替换为"
)

GUIDE_LEGACY_CMD = (
    (
        re.compile(r"\bhuggingface-cli\s+download\b"),
        "use ``hf download`` instead of ``huggingface-cli download``",
    ),
    (
        re.compile(r"\bgit\s+lfs\s+(?:install|clone)\b"),
        "use ``hf download`` instead of ``git lfs``",
    ),
    (
        re.compile(r"\bCUDA_VISIBLE_DEVICES\b"),
        "use ``cluster.component_placement`` instead of ``CUDA_VISIBLE_DEVICES``",
    ),
)
GUIDE_MIRROR_COMMENT = re.compile(
    r"^\s*#.*(?:--use-mirror|HF_ENDPOINT|hf-mirror|gh-proxy)", re.IGNORECASE
)

YAML_BLOCK_START = re.compile(r"^(\s*)\.\.\s+code(?:-block)?::\s*ya?ml\s*$")
YAML_KEY_LINE = re.compile(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s*:")
ALGO_RUNNABLE_REF = re.compile(r"examples/[A-Za-z0-9_./-]+\.yaml|:doc:`[^`]*examples/")
LATEX_BROKEN = (
    (
        re.compile(r"\\mathbb\{E\}\{|\\pi\{\\theta\}"),
        "missing subscript ``_`` in LaTeX expression (e.g. ``\\mathbb{E}_{...}`` or ``\\pi_{\\theta}``)",
    ),
    (
        re.compile(r"\\tag\{(?:[5-9]|[1-9]\d+)\}"),
        "orphan copy-pasted paper equation tag (use sequential tags starting from 1 or omit ``\\tag``)",
    ),
)


def _is_orphan(source: str) -> bool:
    return ":orphan:" in source[:240]


def _is_index(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name == "index.rst" or name.endswith("_index.rst")


@functools.cache
def repo_yaml_and_code_words(root: Path) -> frozenset[str]:
    """Identifier vocabulary across ``rlinf/``, ``examples/``, and ``evaluations/``."""
    words: set[str] = set()
    word_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    for folder in ("rlinf", "examples", "evaluations"):
        base = root / folder
        if not base.is_dir():
            continue
        for file_path in base.rglob("*"):
            if file_path.is_file() and file_path.suffix in (".py", ".yaml", ".yml"):
                try:
                    words.update(
                        word_re.findall(
                            file_path.read_text(encoding="utf-8", errors="ignore")
                        )
                    )
                except OSError:
                    continue
    return frozenset(words)


@functools.cache
def repo_basenames(root: Path) -> frozenset[str]:
    """Basenames of tracked/shipped files in code folders for bare-file checks."""
    names: set[str] = set()
    for folder in ("rlinf", "examples", "evaluations", "toolkits", "requirements"):
        base = root / folder
        if not base.is_dir():
            continue
        for file_path in base.rglob("*"):
            if file_path.is_file():
                names.add(file_path.name)
    return frozenset(names)


def _check_raw_html(rel: str, lines: list[str]) -> list[route.Finding]:
    findings: list[route.Finding] = []
    i = 0
    total = len(lines)
    while i < total:
        if RAW_HTML_BLOCK.match(lines[i]):
            start = i + 1
            indent = len(lines[i]) - len(lines[i].lstrip())
            body_lines: list[str] = []
            i += 1
            while i < total:
                cur = lines[i]
                if cur.strip() and (len(cur) - len(cur.lstrip())) <= indent:
                    break
                body_lines.append(cur)
                i += 1
            body = "\n".join(body_lines)
            if RAW_HTML_BAD.search(body):
                findings.append(
                    route.Finding(
                        "major",
                        rel,
                        start,
                        "docs/raw-html-grid",
                        "replace raw HTML grid / hardcoded ``color: blue`` with "
                        "``sphinx-design`` ``.. grid::`` cards (dark-mode safe and link-checked)",
                    )
                )
            continue
        i += 1
    return findings


def _check_index_page(rel: str, source: str, lines: list[str]) -> list[route.Finding]:
    """Check section/gallery index pages for clean routing and toctree coverage."""
    if rel in ("docs/source-en/index.rst", "docs/source-zh/index.rst"):
        return []
    findings: list[route.Finding] = []

    toc_targets: list[str] = []
    has_non_hidden_toctree = False
    for i, line in enumerate(lines):
        if line.strip() != ".. toctree::":
            continue
        indent = len(line) - len(line.lstrip())
        is_hidden = False
        for body in lines[i + 1 :]:
            stripped = body.strip()
            if stripped and (len(body) - len(body.lstrip())) <= indent:
                break
            if stripped == ":hidden:":
                is_hidden = True
            elif stripped and not stripped.startswith(":"):
                target = re.search(r"<([^>]+)>$", stripped)
                toc_targets.append(target.group(1) if target else stripped)
        if not is_hidden:
            has_non_hidden_toctree = True

    if has_non_hidden_toctree:
        findings.append(
            route.Finding(
                "major",
                rel,
                None,
                "docs/index-style",
                "mark ``.. toctree::`` as ``:hidden:`` on index pages and route "
                "readers via ``.. grid::`` cards or a ``.. list-table::`` in the body",
            )
        )

    if toc_targets and ".. grid::" not in source and ".. list-table::" not in source:
        findings.append(
            route.Finding(
                "major",
                rel,
                None,
                "docs/index-style",
                "index page routes without ``.. grid::`` cards or a ``.. list-table::``",
            )
        )

    body_without_toctree = re.sub(
        r"\.\.\s+toctree::.*?(?=\n\S|\Z)", "", source, flags=re.DOTALL
    )
    missing = [
        t
        for t in toc_targets
        if t not in body_without_toctree
        and t.split("/")[-1] not in body_without_toctree
    ]
    if missing:
        findings.append(
            route.Finding(
                "major",
                rel,
                None,
                "docs/index-style",
                f"toctree entry missing from page cards/table: {', '.join(missing)}",
            )
        )
    return findings


def _check_paths(
    rel: str, lines: list[str], root: Path, all_basenames: frozenset[str]
) -> list[route.Finding]:
    """Verify repository file paths cited in docs exist on disk."""
    findings: list[route.Finding] = []
    is_extending = "/rst_source/extending/" in rel
    active_dir: Path | None = None
    active_dir_line = -999

    for idx, line in enumerate(lines, 1):
        for pat in (REPO_FILE_LITERAL, REPO_SCRIPT_CMD):
            for match in pat.finditer(line):
                target = match.group(1)
                if ILLUSTRATIVE_PATH.search(target):
                    continue
                if not (root / target).exists():
                    findings.append(
                        route.Finding(
                            "major",
                            rel,
                            idx,
                            "docs/path-exists",
                            f"referenced repository path ``{target}`` does not exist",
                        )
                    )

        if is_extending:
            continue

        for dm in REPO_DIR_LITERAL.finditer(line):
            cand = root / dm.group(1).rstrip("/")
            if cand.is_dir():
                active_dir = cand
                active_dir_line = idx

        if active_dir is not None and idx - active_dir_line <= 8:
            for fm in BARE_FILE_REF.finditer(line):
                fname = fm.group(1)
                if ILLUSTRATIVE_PATH.search(fname):
                    continue
                if not (active_dir / fname).exists() and fname not in all_basenames:
                    rel_dir = active_dir.relative_to(root).as_posix()
                    findings.append(
                        route.Finding(
                            "major",
                            rel,
                            idx,
                            "docs/path-exists",
                            f"referenced file ``{fname}`` does not exist under ``{rel_dir}`` or in the repository",
                        )
                    )
    return findings


def _check_extending(rel: str, source: str, lines: list[str]) -> list[route.Finding]:
    """Enforce RLinf extending-guide standards (no pytest boilerplate, At-a-Glance, verify)."""
    findings: list[route.Finding] = []
    for idx, line in enumerate(lines, 1):
        if EXTENDING_PYTEST.search(line):
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    idx,
                    "docs/extending-no-pytest",
                    "do not include boilerplate ``tests/unit_tests/`` or ``pytest`` "
                    "sections in extending guides; provide a standalone local CPU verification snippet instead",
                )
            )

    head = "\n".join(lines[:70])
    if ".. list-table::" not in head:
        findings.append(
            route.Finding(
                "major",
                rel,
                None,
                "docs/extending-structure",
                "extending guide needs an At-a-Glance ``.. list-table::`` near the top of the page",
            )
        )
    if not EXTENDING_VERIFY.search(source):
        findings.append(
            route.Finding(
                "major",
                rel,
                None,
                "docs/extending-structure",
                "extending guide needs a runnable verification section/snippet",
            )
        )
    return findings


def _check_guide(rel: str, lines: list[str]) -> list[route.Finding]:
    """Catch ultra-thin stub guides and legacy commands/mirror clutter in guides/."""
    findings: list[route.Finding] = []
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 40:
        findings.append(
            route.Finding(
                "major",
                rel,
                None,
                "docs/guide-quality",
                f"guide has only {len(non_empty)} non-empty lines (stub); expand with "
                "a concrete config, expected effect, and verification steps or merge into a related guide",
            )
        )
    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if GUIDE_MIRROR_COMMENT.match(line):
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    idx,
                    "docs/mirror-comment",
                    "remove commented mirror lines inside bash blocks; state "
                    "--use-mirror and HF_ENDPOINT once in prose or start/installation.rst",
                )
            )
        for pat, msg in GUIDE_LEGACY_CMD:
            if pat.search(stripped):
                findings.append(
                    route.Finding("major", rel, idx, "docs/command-legacy", msg)
                )
    return findings


def _check_concept(rel: str, lines: list[str]) -> list[route.Finding]:
    """Keep Concepts pages conceptual: no autodoc dumps, stubs, or PR migration notes."""
    findings: list[route.Finding] = []
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 40:
        findings.append(
            route.Finding(
                "major",
                rel,
                None,
                "docs/concept-purity",
                f"concept page has only {len(non_empty)} non-empty lines (stub); "
                "expand or merge into an adjacent concept page",
            )
        )
    for idx, line in enumerate(lines, 1):
        if CONCEPT_AUTODOC.match(line):
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    idx,
                    "docs/concept-purity",
                    "move ``.. autoclass::`` / ``.. autofunction::`` out of Concepts into ``reference/api/``",
                )
            )
        if CONCEPT_MIGRATION.search(line):
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    idx,
                    "docs/concept-purity",
                    "remove PR migration history from concept prose; describe the current contract directly",
                )
            )
    return findings


def _check_reference(
    rel: str, source: str, lines: list[str], root: Path
) -> list[route.Finding]:
    """Verify algorithm YAML keys against repo code and check LaTeX syntax."""
    findings: list[route.Finding] = []

    has_tag_1 = "\\tag{1}" in source
    for idx, line in enumerate(lines, 1):
        for pat, msg in LATEX_BROKEN:
            if "\\tag{" in pat.pattern and has_tag_1:
                continue
            if pat.search(line):
                findings.append(
                    route.Finding("major", rel, idx, "docs/latex-sanity", msg)
                )

    if "/rst_source/reference/algorithms/" in rel and not _is_index(rel):
        known = repo_yaml_and_code_words(root)
        in_yaml = False
        yaml_indent = 0
        for idx, line in enumerate(lines, 1):
            m = YAML_BLOCK_START.match(line)
            if m:
                in_yaml = True
                yaml_indent = len(m.group(1))
                continue
            if in_yaml:
                if line.strip() and (len(line) - len(line.lstrip())) <= yaml_indent:
                    in_yaml = False
                    continue
                if line.strip().startswith("#"):
                    continue
                km = YAML_KEY_LINE.match(line)
                if km and km.group(1) not in known:
                    findings.append(
                        route.Finding(
                            "major",
                            rel,
                            idx,
                            "docs/algo-keys",
                            f"YAML key ``{km.group(1)}`` does not exist in ``rlinf/``, "
                            "``examples/``, or ``evaluations/``",
                        )
                    )
        if not ALGO_RUNNABLE_REF.search(source):
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    None,
                    "docs/algo-keys",
                    "algorithm reference page should link to at least one runnable "
                    "``examples/.../*.yaml`` config or recipe page",
                )
            )
    return findings


def check_page(path: Path, rel: str, root: Path) -> list[route.Finding]:
    """Run chapter-specific structural and accuracy checks on one RST page."""
    if any(skip in rel for skip in SKIP_HISTORICAL):
        return []
    source = path.read_text(encoding="utf-8")
    if _is_orphan(source):
        return []
    lines = source.splitlines()
    findings: list[route.Finding] = []

    findings += _check_raw_html(rel, lines)
    findings += _check_paths(rel, lines, root, repo_basenames(root))

    if _is_index(rel):
        findings += _check_index_page(rel, source, lines)
        return findings

    if "/rst_source/extending/" in rel:
        findings += _check_extending(rel, source, lines)
    elif "/rst_source/guides/" in rel:
        findings += _check_guide(rel, lines)
    elif "/rst_source/concepts/" in rel:
        findings += _check_concept(rel, lines)
    elif "/rst_source/reference/" in rel:
        findings += _check_reference(rel, source, lines, root)

    return findings
