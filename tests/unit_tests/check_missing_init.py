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

from pathlib import Path


def main() -> int:
    # Resolve from script location to avoid depending on current working directory.
    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / "rlinf"
    missing = []
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
        if "__pycache__" in directory.parts:
            continue
        py_files = [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix == ".py" and p.name != "__init__.py"
        ]
        if py_files and not (directory / "__init__.py").exists():
            missing.append(directory.relative_to(repo_root))

    if missing:
        print("Missing __init__.py in:")
        for directory in sorted(missing):
            print(f" - {directory}")
        return 1

    print("OK: all python dirs have __init__.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
