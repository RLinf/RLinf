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

"""Check changed RLinf docs: the build, markup, names, links, pairing, and writing.

Usage::

    python3 .agents/skills/docs/check.py              # docs changed against main
    python3 .agents/skills/docs/check.py FILE ...     # specific pages
    python3 .agents/skills/docs/check.py --no-build   # skip the Sphinx build

Page findings count only on lines the change adds (``--all-lines`` counts every
line); the build always covers both trees. The build and the markup check run in
the docs virtualenv (``DOCS_VENV``, default ``<repo>/.docs-venv``), created from
``requirements/docs`` on first use.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import route  # noqa: E402
from build_docs import build, ensure_venv  # noqa: E402

REQUIREMENTS = route.REPO_ROOT / "requirements" / "docs" / "requirements.txt"
VENV = Path(os.environ.get("DOCS_VENV") or route.REPO_ROOT / ".docs-venv")

try:
    import docutils  # noqa: F401
except ImportError:
    python = ensure_venv(VENV, REQUIREMENTS)
    os.execv(str(python), [str(python), __file__, *sys.argv[1:]])

import check_doc_symbols  # noqa: E402
import check_recipe  # noqa: E402
import check_rst_markup  # noqa: E402
import check_sections  # noqa: E402
import check_writing  # noqa: E402

LANGS = {"docs/source-en/": "docs/source-zh/", "docs/source-zh/": "docs/source-en/"}
READMES = {"README.md": "README.zh-CN.md", "README.zh-CN.md": "README.md"}
RTD_LINK = re.compile(r"readthedocs\.io/(?:en|zh-cn)/latest/rst_source/")
WARNING_AT = re.compile(
    r"^(?P<path>/[^:\s]+)(?::(?P<line>\d+))?: (?:WARNING|ERROR|SEVERE): "
)
LIST_MARKER = re.compile(r"^(?:\* -|[-*+]|#\.|\d+\.)\s+")
TABLE_BORDER = re.compile(r"^\s*=+(?:\s+=+)+\s*$")


def relative(path: str) -> str:
    """A path relative to the repository when it lies inside it."""
    try:
        return Path(path).resolve().relative_to(route.REPO_ROOT).as_posix()
    except ValueError:
        return path


def counterpart(path: str) -> str | None:
    """The other language's file for a docs page or README, if it has one."""
    if path in READMES:
        return READMES[path]
    for prefix, other in LANGS.items():
        if path.startswith(prefix):
            return other + path[len(prefix) :]
    return None


PARITY_EXTS = (".rst", ".md", ".css", ".js")
SHARED_STATIC_PARITY = ("_static/css/custom.css", "_static/js/sidebar-nav.js")


def check_parity(changes: list[route.Change]) -> list[route.Finding]:
    changed = {c.path for c in changes}
    findings = []
    checked_shared: set[str] = set()
    for change in changes:
        other = counterpart(change.path)
        is_conf = change.path in ("docs/source-en/conf.py", "docs/source-zh/conf.py")
        if other is None or (not change.path.endswith(PARITY_EXTS) and not is_conf):
            continue
        exists = (route.REPO_ROOT / other).exists()
        if change.status == "D" and exists and other not in changed:
            findings.append(
                route.Finding(
                    "major",
                    change.path,
                    None,
                    "docs/parity",
                    f"deleted, but {other} still exists",
                )
            )
        elif change.status != "D" and not exists:
            findings.append(
                route.Finding(
                    "major", change.path, None, "docs/parity", f"{other} does not exist"
                )
            )
        elif change.status != "D" and other not in changed:
            findings.append(
                route.Finding(
                    "minor",
                    change.path,
                    None,
                    "docs/parity",
                    f"changed while {other} did not; check whether it needs the same change",
                )
            )
        for suffix in SHARED_STATIC_PARITY:
            if change.path.endswith(suffix) and suffix not in checked_shared:
                checked_shared.add(suffix)
                en_f = route.REPO_ROOT / "docs/source-en" / suffix
                zh_f = route.REPO_ROOT / "docs/source-zh" / suffix
                if en_f.exists() and zh_f.exists():
                    if en_f.read_bytes() != zh_f.read_bytes():
                        findings.append(
                            route.Finding(
                                "major",
                                change.path,
                                None,
                                "docs/parity",
                                f"EN and ZH copies of {suffix} differ; keep shared CSS/JS identical",
                            )
                        )
    return findings


def toctree_entries(path: Path) -> set[str]:
    """The targets of every ``.. toctree::`` on a page, without captions."""
    entries: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip() != ".. toctree::":
            continue
        indent = len(line) - len(line.lstrip())
        for body in lines[i + 1 :]:
            stripped = body.strip()
            if stripped and len(body) - len(body.lstrip()) <= indent:
                break
            if stripped and not stripped.startswith(":"):
                target = re.search(r"<([^>]+)>$", stripped)
                entries.add(target.group(1) if target else stripped)
    return entries


def check_toctrees(pages: list[str]) -> list[route.Finding]:
    findings = []
    pairs = {
        (page, counterpart(page))
        if page.startswith("docs/source-en/")
        else (counterpart(page), page)
        for page in pages
        if counterpart(page)
    }
    for en_page, zh_page in sorted(pairs):
        mine, theirs = route.REPO_ROOT / en_page, route.REPO_ROOT / zh_page
        if not (mine.exists() and theirs.exists()):
            continue
        page = en_page
        en, zh = toctree_entries(mine), toctree_entries(theirs)
        if en != zh:
            only = "; ".join(
                f"only in {lang}: {', '.join(sorted(s))}"
                for lang, s in (("EN", en - zh), ("ZH", zh - en))
                if s
            )
            findings.append(
                route.Finding(
                    "major", page, None, "docs/parity", f"toctrees differ ({only})"
                )
            )
    return findings


TAB_SET_LINE = re.compile(r"^\s*\.\.\s+tab-set::", re.MULTILINE)
TAB_ITEM_LINE = re.compile(r"^\s*\.\.\s+tab-item::", re.MULTILINE)
RST_IMAGE_URL = re.compile(
    r"^\s*(?:\.\.\s+(?:figure|image)::|:img-top:)\s+(https?://\S+)\s*$"
)
MD_IMAGE_URL = re.compile(
    r"""(?:<img\b[^>]*\bsrc=["'](https?://[^"']+)["']|!\[[^\]]*\]\((https?://[^)\s]+)\))"""
)
GITHUB_RAW_REDIRECT = re.compile(r"^https?://github\.com/[^/\s]+/[^/\s]+/raw/")


def _probe_url(url: str) -> tuple[int | None, str | None]:
    """Return ``(http_status, network_error)`` for an external image URL."""
    parts = urllib.parse.urlsplit(url)
    safe_url = urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc.encode("idna").decode("ascii"),
            urllib.parse.quote(parts.path, safe="/:@!$&'()*+,;=-._~%"),
            urllib.parse.quote(parts.query, safe="=/?:@!$&'()*+,;=-._~%"),
            urllib.parse.quote(parts.fragment, safe=""),
        )
    )
    headers = {"User-Agent": "RLinf-docs-check/1.0"}
    for method, extra in (("HEAD", {}), ("GET", {"Range": "bytes=0-0"})):
        req = urllib.request.Request(
            safe_url, method=method, headers={**headers, **extra}
        )
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                return resp.status, None
        except urllib.error.HTTPError as err:
            if method == "HEAD" and err.code in (403, 405, 501):
                continue
            return err.code, None
        except (urllib.error.URLError, OSError) as err:
            return None, str(err)
    return None, "probe failed"


def check_image_urls(
    changes: list[route.Change],
    selection: route.Selection,
    check_http: bool = True,
) -> list[route.Finding]:
    """Verify external figure/image URLs on checked lines (non-redirecting raw URL + HTTP 200)."""
    findings: list[route.Finding] = []
    targets: list[tuple[str, int, str]] = []
    for change in changes:
        if change.status == "D":
            continue
        is_rst = change.path.endswith(".rst") and change.path.startswith("docs/")
        is_md = change.path in ("README.md", "README.zh-CN.md")
        if not (is_rst or is_md):
            continue
        path = route.REPO_ROOT / change.path
        if not path.is_file():
            continue
        allowed = selection.lines(change)
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            if allowed is not None and lineno not in allowed:
                continue
            urls: list[str] = []
            if is_rst:
                m = RST_IMAGE_URL.match(line)
                if m:
                    urls.append(m.group(1))
            else:
                for m in MD_IMAGE_URL.finditer(line):
                    urls.append(m.group(1) or m.group(2))
            for url in urls:
                if GITHUB_RAW_REDIRECT.match(url):
                    findings.append(
                        route.Finding(
                            "major",
                            change.path,
                            lineno,
                            "docs/image-url",
                            f"use raw.githubusercontent.com instead of redirecting github.com/.../raw/ URL: {url}",
                        )
                    )
                targets.append((change.path, lineno, url))

    if not check_http or not targets:
        return findings

    cache: dict[str, tuple[int | None, str | None]] = {}
    network_unavailable = False
    for rel, lineno, url in targets:
        if network_unavailable:
            break
        if url not in cache:
            status, net_err = _probe_url(url)
            cache[url] = (status, net_err)
            if net_err is not None and len(cache) == 1:
                network_unavailable = True
                print("skipped: external image URL checks (network unavailable)")
                break
        status, _ = cache[url]
        if status is not None and status >= 400 and status != 429:
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    lineno,
                    "docs/image-url",
                    f"external image URL returned HTTP {status}: {url}",
                )
            )
    return findings


def check_duplicate_commands(pages: list[str]) -> list[route.Finding]:
    """Flag identical multi-line setup/launch bash blocks duplicated across changed recipe pages."""
    findings: list[route.Finding] = []
    seen: dict[tuple[str, str], str] = {}
    for rel in sorted(pages):
        if not check_recipe.RECIPE_OR_EVAL_PAGE.match(
            rel
        ) or check_recipe.SKIP_NAME.search(rel):
            continue
        lang = "zh" if rel.startswith("docs/source-zh/") else "en"
        path = route.REPO_ROOT / rel
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for block in check_recipe._iter_code_blocks(lines, len(lines)):
            if block["lang"] not in ("bash", "sh", "shell", "console", ""):
                continue
            cmds = [
                raw.strip()
                for _, raw in block["body"]
                if raw.strip() and not raw.strip().startswith("#")
            ]
            if len(cmds) < 2:
                continue
            norm = "\n".join(cmds)
            if (
                "requirements/install.sh" not in norm
                and not check_recipe.LAUNCH_SCRIPT.search(norm)
            ):
                continue
            key = (lang, norm)
            first = seen.get(key)
            if first is not None and first != rel:
                findings.append(
                    route.Finding(
                        "minor",
                        rel,
                        block["line"],
                        "docs/duplicate-commands",
                        f"setup/launch bash block duplicates {first}; link to the primary recipe or include a shared _*.rst partial",
                    )
                )
            elif first is None:
                seen[key] = rel
    return findings


def check_tabs(pages: list[str]) -> list[route.Finding]:
    """Ensure paired EN and ZH pages have the same number of tab-sets and tabs."""
    findings = []
    pairs = {
        (page, counterpart(page))
        if page.startswith("docs/source-en/")
        else (counterpart(page), page)
        for page in pages
        if counterpart(page)
    }
    for en_page, zh_page in sorted(pairs):
        mine, theirs = route.REPO_ROOT / en_page, route.REPO_ROOT / zh_page
        if not (mine.exists() and theirs.exists()):
            continue
        en_text, zh_text = (
            mine.read_text(encoding="utf-8"),
            theirs.read_text(encoding="utf-8"),
        )
        en_counts = (
            len(TAB_SET_LINE.findall(en_text)),
            len(TAB_ITEM_LINE.findall(en_text)),
        )
        zh_counts = (
            len(TAB_SET_LINE.findall(zh_text)),
            len(TAB_ITEM_LINE.findall(zh_text)),
        )
        if en_counts != zh_counts:
            findings.append(
                route.Finding(
                    "major",
                    en_page,
                    None,
                    "docs/tab-parity",
                    f"tab structure differs between EN ({en_counts[0]} tab-set, "
                    f"{en_counts[1]} tab-item) and ZH ({zh_counts[0]} tab-set, "
                    f"{zh_counts[1]} tab-item)",
                )
            )
    return findings


def _expand_file_args(files: list[str]) -> list[str]:
    """Expand any directory arguments into their ``*.rst`` files."""
    expanded: list[str] = []
    for raw in files:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = route.REPO_ROOT / candidate
        if candidate.is_dir():
            expanded.extend(
                p.resolve().relative_to(route.REPO_ROOT).as_posix()
                for p in sorted(candidate.rglob("*.rst"))
            )
        else:
            expanded.append(raw)
    return expanded


def is_cjk(char: str) -> bool:
    return unicodedata.east_asian_width(char) in ("W", "F")


def check_hard_wrap(page: Path, rel: str) -> list[route.Finding]:
    """Chinese prose that continues on the next source line renders a space."""
    findings = []
    source = page.read_text(encoding="utf-8")
    for block in check_rst_markup.prose_blocks(source):
        rows = block.text.split("\n")
        if any(TABLE_BORDER.match(row) for row in rows):
            continue
        for previous, current, number in zip(rows, rows[1:], block.lines[1:]):
            head = previous.lstrip()
            marker = LIST_MARKER.match(head)
            column = len(previous) - len(head) + (marker.end() if marker else 0)
            text = current[column:]
            if (
                previous.rstrip()
                and is_cjk(previous.rstrip()[-1])
                and len(current) - len(current.lstrip()) == column
                and text
                and is_cjk(text[0])
            ):
                findings.append(
                    route.Finding(
                        "minor",
                        rel,
                        number,
                        "docs/zh-hard-wrap",
                        "Chinese prose continues from the line above; join the lines",
                    )
                )
    return findings


def check_page(page: Path, rel: str, known: set[str]) -> list[route.Finding]:
    """Markup, names (on the pages that document RLinf's API), links, wrapping."""
    findings = [
        route.Finding(
            "critical" if f.level == "ERROR" else "major",
            rel,
            f.line,
            "docs/markup",
            f.message,
        )
        for f in check_rst_markup.check_file(page)
    ]
    findings += [
        route.Finding(
            "major",
            rel,
            number,
            "docs/symbol",
            f"``{token}`` names nothing in the repository; rename, delete, or "
            "add it to ALLOWED in check_doc_symbols.py if it is illustrative",
        )
        for number, token in check_doc_symbols.literals(page)
        if rel.startswith(check_doc_symbols.DOC_ROOTS) and token not in known
    ]
    for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
        if RTD_LINK.search(line):
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    number,
                    "docs/rtd-link",
                    "hardcoded Read the Docs link to an RLinf page; use :doc: or :ref:",
                )
            )
    if rel.startswith("docs/source-zh/"):
        findings += check_hard_wrap(page, rel)
    return findings


def check_build() -> list[route.Finding]:
    python = ensure_venv(VENV, REQUIREMENTS)
    findings = []
    with tempfile.TemporaryDirectory(prefix="rlinf-docs-") as output:
        for lang in ("en", "zh"):
            source = route.REPO_ROOT / "docs" / f"source-{lang}"
            print(f"building {source.relative_to(route.REPO_ROOT)}", flush=True)
            for record in build(python, source, Path(output) / lang):
                head, *rest = record.splitlines()
                where = WARNING_AT.match(head)
                path, line = f"docs/source-{lang}", None
                if where:
                    path = relative(where["path"])
                    line = int(where["line"]) if where["line"] else None
                    head = head[where.end() :]
                findings.append(
                    route.Finding(
                        "critical",
                        path,
                        line,
                        "docs/build",
                        " ".join([head, *[r.strip() for r in rest]])[:300],
                    )
                )
    return findings


def main() -> int:
    parser = route.check_parser(__doc__.splitlines()[0])
    parser.add_argument("--no-build", action="store_true", help="skip the Sphinx build")
    parser.add_argument(
        "--no-url-check",
        action="store_true",
        help="skip HTTP probes for external figure/image URLs",
    )
    args = parser.parse_args()
    if args.files:
        args.files = _expand_file_args(args.files)
    selection = route.select("docs", args)
    changes = selection.changes
    pages = [
        c.path
        for c in changes
        if c.status != "D" and c.path.endswith(".rst") and c.path.startswith("docs/")
    ]
    known = (
        check_doc_symbols.vocabulary(route.REPO_ROOT)
        | check_doc_symbols.ALLOWED
        | check_doc_symbols.EXTERNAL
        | check_doc_symbols.KNOWN_DRIFT
    )
    names = check_writing.code_names(route.REPO_ROOT)
    added = {c.path for c in changes if c.status == "A"}
    findings = (
        check_parity(changes)
        + check_toctrees(pages)
        + check_tabs(pages)
        + check_recipe.check_static_assets(changes)
        + check_image_urls(changes, selection, check_http=not args.no_url_check)
        + check_duplicate_commands(pages)
    )
    for page in pages:
        path = route.REPO_ROOT / page
        page_findings = check_page(path, page, known)
        page_findings += check_writing.check_page(path, page, names, page in added)
        page_findings += check_recipe.check_page(path, page)
        page_findings += check_sections.check_page(
            path, page, route.REPO_ROOT, is_new=page in added
        )
        findings += selection.relevant(page_findings)
    if args.no_build:
        print("skipped: the Sphinx build (--no-build)")
    elif any(c.path.startswith("docs/") for c in changes):
        findings += check_build()
    return route.report("docs", findings)


if __name__ == "__main__":
    raise SystemExit(main())
