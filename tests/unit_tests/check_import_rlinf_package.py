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

"""Import-check all modules under the rlinf package.

Two passes run. The first imports every module the skip list allows. The second
resolves each ``rlinf.*`` import statement against the source tree without
importing anything, so a module naming a package that does not exist is caught
even where the first pass cannot reach it.

Usage:
    python tests/unit_tests/check_import_rlinf_package.py --workers 16
    python tests/unit_tests/check_import_rlinf_package.py --no-test-modules rlinf/envs rlinf/models
"""

import argparse
import ast
import importlib
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DEFAULT_NO_TEST_MODULES = [
    "rlinf/envs",
    "rlinf/models",
    "rlinf/data/datasets/dreamzero",
    "rlinf/data/datasets/openpi",
    "rlinf/data/datasets/fastwam.py",
    "rlinf/data/datasets/recap/cfg_model.py",
    "rlinf/data/datasets/recap/utils.py",
    "rlinf/data/datasets/recap/value_dataset.py",
    "rlinf/workers/sft/fsdp_cfg_worker.py",
    "rlinf/workers/sft/fsdp_value_sft_worker.py",
    "rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py",
]


def _normalize_no_test_modules(raw_paths: list[str]) -> list[str]:
    return [
        path.strip().replace("\\", "/").rstrip("/")
        for path in raw_paths
        if path.strip()
    ]


def _should_skip_module(relative: Path, no_test_modules: list[str]) -> bool:
    module_path = f"rlinf/{relative.with_suffix('').as_posix()}"
    file_path = f"rlinf/{relative.as_posix()}"

    for skip_path in no_test_modules:
        normalized_skip = skip_path.removesuffix(".py")
        if module_path == normalized_skip or module_path.startswith(
            f"{normalized_skip}/"
        ):
            return True
        if file_path == skip_path:
            return True
    return False


def _discover_modules(rlinf_root: Path, no_test_modules: list[str]) -> list[str]:
    modules: set[str] = set()
    for py_file in rlinf_root.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue

        relative = py_file.relative_to(rlinf_root)
        if _should_skip_module(relative, no_test_modules):
            continue
        if py_file.name == "__init__.py":
            if relative.parent == Path("."):
                module_name = "rlinf"
            else:
                module_name = f"rlinf.{'.'.join(relative.parent.parts)}"
        else:
            module_name = f"rlinf.{'.'.join(relative.with_suffix('').parts)}"
        modules.add(module_name)

    return sorted(modules)


def _module_exists(rlinf_root: Path, dotted: str) -> bool:
    """Whether ``dotted`` names a module or package inside the rlinf source tree."""
    parts = dotted.split(".")[1:]
    if not parts:
        return True
    target = rlinf_root.joinpath(*parts)
    return target.is_dir() or target.with_suffix(".py").exists()


def _find_dangling_imports(
    rlinf_root: Path,
) -> tuple[list[tuple[str, int, str]], list[tuple[str, str]]]:
    """Find absolute ``rlinf.*`` imports that no file in the source tree provides.

    Only the module part of each statement is resolved; ``from rlinf.a.b import c``
    checks ``rlinf.a.b`` and says nothing about ``c``. Relative imports are left to
    the import pass, which resolves them the way Python does.

    Returns:
        tuple: ``(dangling, unreadable)``. ``dangling`` holds one
        ``(file, line, module)`` per unresolved import, ordered by file then line.
        ``unreadable`` holds one ``(file, reason)`` per file this pass could not
        parse; those are reported rather than skipped, because the files most
        likely to reach here are the ones the import pass never touches.
    """
    dangling: list[tuple[str, int, str]] = []
    unreadable: list[tuple[str, str]] = []
    for py_file in sorted(rlinf_root.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        relative = py_file.relative_to(rlinf_root.parent).as_posix()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError) as error:
            unreadable.append((relative, f"{type(error).__name__}: {error}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level or not node.module:
                    continue
                targets = [node.module]
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                continue
            for target in targets:
                if target != "rlinf" and not target.startswith("rlinf."):
                    continue
                if not _module_exists(rlinf_root, target):
                    dangling.append((relative, node.lineno, target))
    return sorted(dangling), sorted(unreadable)


def _import_module(module_name: str) -> tuple[str, str | None]:
    try:
        importlib.import_module(module_name)
        return module_name, None
    except Exception:
        return module_name, traceback.format_exc()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, max(4, (os.cpu_count() or 4) * 2)),
        help="Number of threads used for parallel import.",
    )
    parser.add_argument(
        "--no-test-modules",
        nargs="*",
        default=DEFAULT_NO_TEST_MODULES,
        help=(
            "Full paths under repo to skip, such as 'rlinf/envs' or "
            "'rlinf/path/to/module.py'."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    rlinf_root = repo_root / "rlinf"
    no_test_modules = _normalize_no_test_modules(args.no_test_modules)
    modules = _discover_modules(rlinf_root, no_test_modules)

    print(f"Discovered {len(modules)} modules under {rlinf_root}")
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_import_module, module) for module in modules]
        for future in as_completed(futures):
            module, err = future.result()
            if err is not None:
                failures.append((module, err))

    dangling, unreadable = _find_dangling_imports(rlinf_root)

    if failures:
        print(f"Import failures: {len(failures)}")
        for module, err in sorted(failures):
            print(f"\n[FAILED] {module}\n{err}")

    if dangling:
        print(f"Unresolved rlinf imports: {len(dangling)}")
        for file_path, lineno, module in dangling:
            print(f"[DANGLING] {file_path}:{lineno} imports {module}")

    if unreadable:
        print(f"Unparsable files: {len(unreadable)}")
        for file_path, reason in unreadable:
            print(f"[UNPARSABLE] {file_path}: {reason}")

    if failures or dangling or unreadable:
        return 1

    print("OK: all discovered rlinf modules imported successfully")
    print("OK: every rlinf import resolves inside the source tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
