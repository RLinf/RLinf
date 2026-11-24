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

"""Read shared version from the root RLinf ``pyproject.toml`` (or sdist ``PKG-INFO``)."""

from __future__ import annotations

import re
from pathlib import Path

from setuptools import setup

_HERE = Path(__file__).resolve().parent


def _read_root_pyproject_version(pyproject_path: Path) -> str | None:
    if not pyproject_path.is_file():
        return None
    text = pyproject_path.read_text(encoding="utf-8")
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if in_project:
            match = re.match(r'^version\s*=\s*["\']([^"\']+)["\']', stripped)
            if match:
                return match.group(1)
    return None


def _read_pkg_info_version(pkg_info_path: Path) -> str | None:
    if not pkg_info_path.is_file():
        return None
    for line in pkg_info_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
            if version:
                return version
    return None


def _resolve_version() -> str:
    version = _read_root_pyproject_version(_HERE.parent / "pyproject.toml")
    if version:
        return version
    version = _read_pkg_info_version(_HERE / "PKG-INFO")
    if version:
        return version
    raise RuntimeError(
        "Unable to determine rlinf-cli version from '../pyproject.toml' or 'PKG-INFO'."
    )


setup(version=_resolve_version())
