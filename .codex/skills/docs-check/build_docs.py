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

"""Reproduce the Read the Docs build gate locally, for both languages.

``docs/source-en/.readthedocs.yaml`` and ``docs/source-zh/.readthedocs.yaml``
both set ``fail_on_warning: true``, so *any* Sphinx warning turns the
``docs/readthedocs.org:rlinf`` or ``:rlinf-cn`` check red.  The two projects
build independently, so a warning that only the Chinese page triggers — a
CJK punctuation problem, a missing ZH counterpart, a broken ``:doc:`` target
— fails ``rlinf-cn`` while ``rlinf`` stays green.  Always build **both**.

This script provisions a docs-only virtualenv from
``requirements/docs/requirements.txt``, builds each language, and reports the
warnings that would fail Read the Docs.  Autodoc import failures are filtered
by default: the docs venv deliberately does not install torch, so every
``.. autoclass::`` fails locally in a way it never does on Read the Docs.
Pass ``--strict-autodoc`` (in a full environment) to see them.

Usage::

    python3 .codex/skills/docs-check/build_docs.py             # en + zh
    python3 .codex/skills/docs-check/build_docs.py --lang zh
    python3 .codex/skills/docs-check/build_docs.py --keep-output

Exit code is 1 when either language reports a warning, 0 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Emitted for every autodoc target when the package itself is not installed.
AUTODOC_NOISE = re.compile(
    r"WARNING: autodoc: failed to import .*"
    r"|^No module named .*\[autodoc\.import_object\]"
    r"|^\s*the following exception was raised:",
    re.MULTILINE,
)
# A warning record starts at "<path>:<line>: WARNING" or a bare "WARNING:".
RECORD_START = re.compile(r"^(?:\S+:\d+: )?(?:WARNING|ERROR|SEVERE)[:\s]")


def repo_root() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()


def ensure_venv(venv: Path, requirements: Path) -> Path:
    """Create the docs venv if needed and return its python."""
    python = venv / "bin" / "python"
    if python.exists():
        probe = subprocess.run(
            [str(python), "-c", "import sphinx"], capture_output=True
        )
        if probe.returncode == 0:
            return python
    else:
        print(f"[build_docs] creating {venv}")
        if shutil.which("uv"):
            subprocess.run(["uv", "venv", str(venv), "--python", "3.11"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

    print(f"[build_docs] installing {requirements.name}")
    env = {**os.environ, "UV_HTTP_TIMEOUT": "180"}
    if shutil.which("uv"):
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "-r", str(requirements)],
            check=True,
            env=env,
        )
    else:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-q", "-r", str(requirements)],
            check=True,
        )
    return python


def split_records(stderr: str) -> list[str]:
    """Group Sphinx's stderr into one string per warning."""
    records: list[str] = []
    for line in stderr.splitlines():
        if RECORD_START.match(line) or not records:
            records.append(line)
        else:
            records[-1] += "\n" + line
    return [r for r in records if r.strip()]


def build(python: Path, source: Path, output: Path, strict_autodoc: bool) -> list[str]:
    """Build one language and return the warnings that matter."""
    result = subprocess.run(
        [
            str(python.parent / "sphinx-build"),
            "-b",
            "html",
            "-q",
            str(source),
            str(output),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"},
    )
    records = split_records(result.stderr)
    if not strict_autodoc:
        records = [r for r in records if not AUTODOC_NOISE.search(r)]
    if result.returncode != 0 and not records:
        records.append(
            f"sphinx-build exited {result.returncode}\n{result.stdout[-2000:]}"
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=("en", "zh", "both"), default="both")
    parser.add_argument(
        "--venv",
        type=Path,
        default=os.environ.get("DOCS_VENV"),
        help="virtualenv to build in (default: <repo>/.docs-venv)",
    )
    parser.add_argument(
        "--strict-autodoc",
        action="store_true",
        help="keep autodoc import warnings (only useful in a full env)",
    )
    parser.add_argument(
        "--keep-output", action="store_true", help="keep the generated HTML"
    )
    args = parser.parse_args()

    root = repo_root()
    venv = args.venv or root / ".docs-venv"
    python = ensure_venv(venv, root / "requirements" / "docs" / "requirements.txt")

    languages = ("en", "zh") if args.lang == "both" else (args.lang,)
    destination = (
        root / "docs" / "build" / "check"
        if args.keep_output
        else Path(tempfile.mkdtemp(prefix="rlinf-docs-"))
    )

    total = 0
    for language in languages:
        source = root / "docs" / f"source-{language}"
        print(f"[build_docs] building {source.name}")
        records = build(python, source, destination / language, args.strict_autodoc)
        total += len(records)
        for record in records:
            print(record)
        print(f"[build_docs] {source.name}: {len(records)} warning(s)")

    if not args.keep_output:
        shutil.rmtree(destination, ignore_errors=True)

    if total:
        print(
            f"\n{total} warning(s) — Read the Docs builds with fail_on_warning, "
            f"so this fails the docs check.",
            file=sys.stderr,
        )
        return 1
    print("\nBoth builds are clean." if len(languages) > 1 else "\nBuild is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
