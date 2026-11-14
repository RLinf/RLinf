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

"""Check changed install, Docker, and e2e files against RLinf's install conventions.

Usage::

    python3 .agents/skills/install/check.py            # files changed against main
    python3 .agents/skills/install/check.py FILE ...

The line rules report only lines the change adds (``--all-lines`` reports every
line), so the exceptions already on ``main`` do not repeat. They are heuristics: a hit
is a line to review against the conventions in ``SKILL.md``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import route  # noqa: E402

INSTALL = "requirements/install.sh"
DOCKERFILE = "docker/Dockerfile"

PACKAGE_MANAGER = re.compile(r"^\s*(?:sudo\s+)?(?:apt-get|apt|dnf|yum|pacman)\s")
RAY_PIN = re.compile(r"uv pip install[^#]*\bray==")
TORCH_PIN = re.compile(r"uv pip install[^#]*\b(?:torch|torchvision|torchaudio)==")
FILE_WRITE = re.compile(r"sed -i|>> |>>\"|cp |tee ")
GITHUB_ORG = re.compile(r"github\.com/([A-Za-z0-9_-]+)")
PINNED = re.compile(r"@[0-9a-fA-F]{7,}|@v?[0-9]|@[A-Za-z][A-Za-z0-9._/-]+|-b\s")
RAW_CLONE = re.compile(r"(?:^|[^_])git clone")


def supported(text: str, name: str) -> set[str]:
    match = re.search(rf"^{name}=\(([^)]*)\)", text, re.MULTILINE)
    return set(match[1].replace('"', "").split()) if match else set()


def check_install_lines(lines: list[str]) -> list[route.Finding]:
    """The install.sh conventions, applied to every line."""

    def finding(severity: str, number: int, rule: str, message: str) -> route.Finding:
        return route.Finding(severity, INSTALL, number, rule, message)

    sanctioned = range(0)
    starts = {
        line.split("(")[0]: n
        for n, line in enumerate(lines, 1)
        if line.endswith("() {")
    }
    if "apply_torch_override" in starts and "install_uv" in starts:
        sanctioned = range(starts["apply_torch_override"], starts["install_uv"])

    findings = []
    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            continue
        if PACKAGE_MANAGER.search(line) and not re.search(
            r"command -v|detect_pkg_manager", line
        ):
            findings.append(
                finding(
                    "major",
                    number,
                    "install/sys-deps",
                    "system package installed in install.sh; add it to every "
                    "install_deps_<manager> in requirements/embodied/sys_deps.sh",
                )
            )
        if RAY_PIN.search(line):
            findings.append(
                finding(
                    "major",
                    number,
                    "install/core-dep",
                    "ray is a core dependency; do not pin it per env",
                )
            )
        if TORCH_PIN.search(line):
            findings.append(
                finding(
                    "minor",
                    number,
                    "install/core-dep",
                    "explicit torch pin; prefer --torch (apply_torch_override), or install "
                    "from outside the repo so RLinf's pyproject overrides do not apply",
                )
            )
        if (
            "pyproject.toml" in line
            and FILE_WRITE.search(line)
            and number not in sanctioned
        ):
            findings.append(
                finding(
                    "minor",
                    number,
                    "install/pyproject",
                    "edits a pyproject.toml; fine for a cloned dependency after cd, "
                    "otherwise route the change through apply_torch_override",
                )
            )
        orgs = GITHUB_ORG.findall(line)
        if (
            orgs
            and orgs[-1] != "RLinf"
            and not PINNED.search(line)
            and not re.search(
                r"git config|clone_or_reuse_repo\(\)|releases/download", line
            )
        ):
            findings.append(
                finding(
                    "major",
                    number,
                    "install/git-pin",
                    "git dependency is neither pinned (@<rev> or -b <branch>) nor forked "
                    "into github.com/RLinf; a later git checkout of a fixed hash also pins it",
                )
            )
        if RAW_CLONE.search(line) and not re.search(
            r'clone_or_reuse_repo|git clone "\$@"', line
        ):
            findings.append(
                finding(
                    "minor",
                    number,
                    "install/clone",
                    "raw git clone; use clone_or_reuse_repo unless this builds a multi-repo workspace",
                )
            )
    return findings


def check_syntax(path: str) -> list[route.Finding]:
    result = subprocess.run(
        ["bash", "-n", path], cwd=route.REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode == 0:
        return []
    return [
        route.Finding("critical", path, None, "install/syntax", result.stderr.strip())
    ]


def check_docker_stages(
    selection: route.Selection, current: str
) -> list[route.Finding]:
    """Every model or env the change adds to SUPPORTED_* has a Docker install.

    With ``--all-lines`` every supported name counts as added.
    """
    before = ""
    if not selection.all_lines:
        try:
            before = route.git("show", f"{route.merge_base(selection.base)}:{INSTALL}")
        except subprocess.CalledProcessError:
            pass
    dockerfile = (route.REPO_ROOT / DOCKERFILE).read_text(encoding="utf-8")
    findings = []
    for array, flag in (("SUPPORTED_MODELS", "--model"), ("SUPPORTED_ENVS", "--env")):
        for name in sorted(supported(current, array) - supported(before, array)):
            if not re.search(
                rf"{flag} {re.escape(name)}(?:[^a-zA-Z0-9_-]|$)",
                dockerfile,
                re.MULTILINE,
            ):
                findings.append(
                    route.Finding(
                        "major",
                        INSTALL,
                        None,
                        "install/docker-stage",
                        f"{name} joins {array} but {DOCKERFILE} has no '{flag} {name}' "
                        "install; add a stage, unless comparable targets on main have no "
                        "image (real-world hardware envs do not)",
                    )
                )
    return findings


def check_docker_layers(added: frozenset[int] | None) -> list[route.Finding]:
    """A stage that installs several venvs does it in one RUN, for uv hardlinks."""
    lines = (route.REPO_ROOT / DOCKERFILE).read_text(encoding="utf-8").splitlines()
    instructions: list[tuple[int, str]] = []
    first, text = None, ""
    for number, line in enumerate(lines, 1):
        if first is None and (not line.strip() or line.lstrip().startswith("#")):
            continue
        first = first or number
        text += line.rstrip().removesuffix("\\") + " "
        if not line.rstrip().endswith("\\"):
            instructions.append((first, text))
            first, text = None, ""
    stages: list[tuple[int, list[str]]] = []
    for number, instruction in instructions:
        if instruction.startswith("FROM "):
            stages.append((number, []))
        elif stages and instruction.startswith("RUN "):
            stages[-1][1].append(instruction)
    findings = []
    bounds = [n for n, _ in stages] + [len(lines) + 1]
    for (start, runs), end in zip(stages, bounds[1:]):
        installs = sum("requirements/install.sh" in run for run in runs)
        if installs > 1 and (added is None or added & set(range(start, end))):
            findings.append(
                route.Finding(
                    "major",
                    DOCKERFILE,
                    start,
                    "install/docker-layers",
                    f"{installs} RUN instructions call install.sh in this stage; chain them "
                    "in one RUN with && so uv can hardlink from its cache",
                )
            )
    return findings


def check_e2e_wiring(change: route.Change) -> list[route.Finding]:
    name = Path(change.path).stem
    workflows = (route.REPO_ROOT / ".github" / "workflows").glob("*.yml")
    if any(
        re.search(rf"\b{re.escape(name)}\b", w.read_text(encoding="utf-8"))
        for w in workflows
    ):
        return []
    return [
        route.Finding(
            "major",
            change.path,
            None,
            "install/e2e-wiring",
            "no workflow under .github/workflows runs this config; add it to the job that tests it",
        )
    ]


def main() -> int:
    args = route.check_parser(__doc__.splitlines()[0]).parse_args()
    selection = route.select("install", args)
    findings: list[route.Finding] = []
    for change in selection.changes:
        if change.status == "D":
            continue
        if change.path.startswith("requirements/") and change.path.endswith(".sh"):
            findings += check_syntax(change.path)
        if change.path == INSTALL:
            current = (route.REPO_ROOT / INSTALL).read_text(encoding="utf-8")
            findings += selection.relevant(check_install_lines(current.splitlines()))
            findings += check_docker_stages(selection, current)
        if change.path == DOCKERFILE:
            findings += check_docker_layers(selection.lines(change))
        if (
            (change.status == "A" or selection.all_lines)
            and change.path.startswith("tests/e2e_tests/")
            and change.path.endswith(".yaml")
        ):
            findings += check_e2e_wiring(change)
    return route.report("install", findings)


if __name__ == "__main__":
    raise SystemExit(main())
