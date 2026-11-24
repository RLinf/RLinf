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

"""Zero-dependency global launcher for the ``rlinf`` CLI.

Locates the active RLinf repository checkout and delegates to its
``rlinf.cli.main`` module so globally installed ``rlinf`` (via
``uv tool install rlinf-cli`` or ``pip install rlinf-cli``) always stays in
sync with the current checkout and works before ``.venv`` exists.

``rlinf shell-init`` and ``rlinf use`` work directly even outside an RLinf
checkout.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Sequence

from .shell import SUPPORTED_SHELLS, handle_shell_init, handle_use


def _get_version() -> str:
    try:
        return importlib.metadata.version("rlinf-cli")
    except importlib.metadata.PackageNotFoundError:
        return "0.5.0.dev0"


def _is_rlinf_checkout(path: Path) -> bool:
    return (path / "rlinf" / "cli" / "main.py").is_file() and (
        path / "requirements" / "install.sh"
    ).is_file()


def find_checkout_root(start: Path | None = None) -> Path:
    """Locate an RLinf checkout from env vars or by walking up from ``start``."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    for env_key in ("RLINF_REPO_PATH", "REPO_PATH"):
        env_val = os.environ.get(env_key)
        if env_val:
            candidates.append(Path(env_val).expanduser().resolve())
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_rlinf_checkout(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not locate an RLinf repository checkout from '{cwd}'. "
        "Run inside an RLinf checkout ('git clone https://github.com/RLinf/RLinf.git && cd RLinf') "
        "or set RLINF_REPO_PATH / REPO_PATH."
    )


def _try_find_checkout_root() -> Path | None:
    try:
        return find_checkout_root()
    except FileNotFoundError:
        return None


def _run_standalone_shell_init(args: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="rlinf shell-init",
        description="Install (default), print, or remove the rlinf shell hook for automatic venv activation.",
    )
    parser.add_argument(
        "shell",
        nargs="?",
        choices=SUPPORTED_SHELLS,
        help="Target shell (defaults to detecting $SHELL).",
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print the shell hook snippet to stdout without modifying startup files.",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the shell hook from your shell startup file.",
    )
    parsed = parser.parse_args(list(args))
    return handle_shell_init(
        shell=parsed.shell,
        print_only=parsed.print_only,
        uninstall=parsed.uninstall,
    )


def _run_standalone_use(args: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="rlinf use",
        description="Switch to an RLinf virtual environment (requires shell hook for auto-activation).",
    )
    parser.add_argument(
        "venv",
        nargs="?",
        help="Virtual environment directory (defaults to .venv).",
    )
    parser.add_argument(
        "--list",
        dest="list_venvs",
        action="store_true",
        help="List discovered virtual environments (.venv, .venv-*).",
    )
    parsed = parser.parse_args(list(args))
    return handle_use(
        parsed.venv,
        list_venvs=parsed.list_venvs,
        repo_root=_try_find_checkout_root(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate ``rlinf`` CLI invocations to the current RLinf checkout."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args in (["-V"], ["--version"]):
        sys.stdout.write(f"rlinf {_get_version()}\n")
        sys.stdout.flush()
        return 0
    if raw_args and raw_args[0] == "shell-init":
        return _run_standalone_shell_init(raw_args[1:])
    if raw_args and raw_args[0] == "use":
        return _run_standalone_use(raw_args[1:])

    try:
        repo_root = find_checkout_root()
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: {exc}\n")
        sys.stderr.flush()
        return 2

    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from rlinf.cli.main import main as checkout_main

    return checkout_main(raw_args, repo_root=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
