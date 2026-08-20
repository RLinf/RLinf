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

"""Check that every name the docs put in code font exists in the code.

The build checks that a page renders and the link checker checks that a
``:doc:`` target resolves.  Neither reads what the page *says*.  A page can
describe a class that was renamed three commits ago, show a table whose shape
changed, or name a config key nobody reads, and every existing gate passes.

That is not hypothetical: ``concepts/realworld_envs.rst`` documented ``TASKS``
as a mapping of gym id to ``(env class, wrapper builder)`` for several commits
after the wrapper builder was deleted, and the docs build stayed clean.

So this reads every ``literal`` on every page and asks whether that name occurs
anywhere in the repository.  A name that does not is either renamed, deleted, or
a typo.  It is deliberately a spell-check rather than a resolver: it does not
know whether ``FrankaEnv.step`` still takes what the page claims, only that
nothing called ``FrankaEnv`` exists any more.  That catches the drift that
matters and costs one pass over the tree.

Prose in double backticks is reported too, because prose does not belong in
code font.  Genuinely illustrative names -- the ``ExampleArm`` of a tutorial --
go in ``ALLOWED`` below or in a ``--allow`` argument.

Usage::

    python3 .cursor/skills/docs-check/check_doc_symbols.py           # all docs
    python3 .cursor/skills/docs-check/check_doc_symbols.py FILE...   # subset
    python3 .cursor/skills/docs-check/check_doc_symbols.py --allow Foo --allow Bar

Exit code is 1 when any unknown name is reported, 0 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

#: Pages that describe this repository's own API, where a name going stale is
#: a defect. The example and guide pages name shell variables, metrics, device
#: nodes and dataset ids, which no amount of allowlisting makes tractable.
DOC_ROOTS = tuple(
    f"docs/source-{lang}/rst_source/{area}"
    for lang in ("en", "zh")
    for area in ("concepts", "extending", "reference")
)

#: Where a name is allowed to come from. Anything the repository ships.
CODE_ROOTS = (
    "rlinf",
    "examples",
    "evaluations",
    "tests",
    "toolkits",
    "requirements",
    "docs/source-en/conf.py",
    "docs/source-zh/conf.py",
)

#: Files worth reading for names. Binary and vendored trees are skipped.
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".toml", ".cfg", ".txt", ".md"}

#: Names a page invents for the reader: the class in a tutorial, the method in
#: a snippet the reader is meant to write. Real code will never define these.
ALLOWED = {
    "ExampleArm",
    "MyEnv",
    "MyRobot",
    "WipeEnv",
    "WipeConfig",
    "build_my_model",
    "my_new_model",
    "my_engine",
    "iter_XXXXXXX",
    "rollout_output_queue",  # a queue the reader names, not an API
    "train_step",  # the method the snippet asks the reader to write
}

#: Names another project owns. The integration pages have to spell them, and
#: this repository will never define them. Adding to this set is a claim that
#: the name belongs to somebody else -- check before you make it.
EXTERNAL = {
    # SGLang server and pipeline configuration
    "DreamZeroPipelineConfig",
    "disagg_role",
    "http_payload_format",
    "master_port_base",
    "max_sessions",
    "port_base",
    "router_url",
    # Megatron / mbridge
    "deepstack_visual_indexes",
}

#: Real drift: the docs name something this repository used to have. Listed so
#: the gate stays green for *new* drift while the backlog stays countable. The
#: fix is to correct the page, then delete the entry -- not to grow this set.
KNOWN_DRIFT: set[str] = set()

#: A literal worth checking looks like a Python name, not a sentence.
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")

#: Literals in RST are delimited by double backticks.
LITERAL = re.compile(r"``([^`\n]+)``")

#: Words, as any of these files spell them.
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def vocabulary(root: Path) -> set[str]:
    """Every word the shipped code and configuration use."""
    known: set[str] = set()
    for entry in CODE_ROOTS:
        target = root / entry
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = [
                path
                for path in target.rglob("*")
                if path.is_file() and path.suffix in CODE_SUFFIXES
            ]
        else:
            continue
        for path in files:
            try:
                known.update(WORD.findall(path.read_text(encoding="utf-8")))
            except (UnicodeDecodeError, OSError):
                continue
    return known


def literals(path: Path) -> list[tuple[int, str]]:
    """Every single-word literal on a page, with its line number."""
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for token in LITERAL.findall(line):
            token = token.strip()
            if IDENTIFIER.match(token):
                found.append((number, token))
    return found


def collect(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_dir():
            paths.extend(sorted(candidate.rglob("*.rst")))
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
        default=list(DOC_ROOTS),
        help="RST files or directories (default: docs/source-en docs/source-zh)",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="NAME",
        help="a name that is illustrative rather than real; repeatable",
    )
    args = parser.parse_args()

    known = vocabulary(ROOT) | ALLOWED | EXTERNAL | KNOWN_DRIFT | set(args.allow)
    pages = collect(args.targets or list(DOC_ROOTS))

    unknown: list[str] = []
    for page in pages:
        for number, token in literals(page):
            if token not in known:
                unknown.append(f"{page.relative_to(ROOT)}:{number}: {token}")

    for line in unknown:
        print(line)

    if unknown:
        print(
            f"\n{len(unknown)} name(s) in code font that the repository does not "
            "define. Rename, delete, or add to ALLOWED if illustrative.",
            file=sys.stderr,
        )
        return 1
    print(f"Every name in code font across {len(pages)} pages exists in the code.")
    if KNOWN_DRIFT:
        print(
            f"{len(KNOWN_DRIFT)} name(s) are known drift waiting to be fixed: "
            f"{', '.join(sorted(KNOWN_DRIFT))}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
