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

Usage:
    python tests/unit_tests/test_import_rlinf_package.py --workers 16
    python tests/unit_tests/test_import_rlinf_package.py --no-test-modules rlinf/envs rlinf/models
"""

import argparse
import importlib
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DEFAULT_NO_TEST_MODULES = [
    "rlinf/envs",
    "rlinf/models",
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

    if failures:
        print(f"Import failures: {len(failures)}")
        for module, err in sorted(failures):
            print(f"\n[FAILED] {module}\n{err}")
        return 1

    print("OK: all discovered rlinf modules imported successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
