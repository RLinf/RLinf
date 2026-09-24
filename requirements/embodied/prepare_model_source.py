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

"""Apply a versioned compatibility patch without replacing user modifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def prepare_source(source: Path, manifest_path: Path, managed: bool = False) -> None:
    """Validate a source revision and apply its exact compatibility patch once.

    Only an installer-managed clean checkout may change revisions. A caller's
    explicit checkout is validated in place; conflicting changes fail before
    applying the patch. The manifest records both original and patched hashes.
    """
    source = source.resolve()
    manifest = json.loads(manifest_path.read_text())

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(source), *args], text=True, capture_output=True
        )
        if result.returncode:
            raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    revision = manifest["revision"]
    actual_revision = git("rev-parse", "HEAD")
    if actual_revision != revision:
        if not managed or git("status", "--porcelain"):
            raise ValueError(
                f"Expected source revision {revision}, found {actual_revision}. "
                "Use a clean checkout at the expected revision or unset the "
                "source-path override to let the installer create one."
            )
        git("checkout", "--detach", revision)

    files = manifest["files"]
    unexpected = set(git("diff", "HEAD", "--name-only").splitlines()) - files.keys()
    if unexpected:
        raise ValueError(f"Source has unrelated tracked changes: {sorted(unexpected)}")

    def hashes() -> dict[str, str | None]:
        result = {}
        for name in files:
            path = (source / name).resolve()
            if not path.is_relative_to(source):
                raise ValueError(f"Source file escapes checkout: {name}")
            result[name] = (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            )
        return result

    before = {name: values["before_sha256"] for name, values in files.items()}
    after = {name: values["after_sha256"] for name, values in files.items()}
    current = hashes()
    if current == after:
        return
    if current != before:
        conflicts = [name for name in files if current[name] != before[name]]
        raise ValueError(
            f"Source differs from the pinned baseline or complete patch: {conflicts}. "
            "Existing files were preserved; use a separate clean checkout."
        )
    patch = (manifest_path.parent / manifest["patch"]).resolve()
    git("apply", "--check", str(patch))
    git("apply", str(patch))
    if hashes() != after:
        raise RuntimeError(
            "Patched source hashes do not match the compatibility manifest"
        )


def main() -> None:
    """Prepare the dependency checkout selected by the shared installer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--managed", action="store_true")
    args = parser.parse_args()
    prepare_source(args.source, args.manifest, managed=args.managed)
    print(f"Verified model source: {args.source.resolve()}")


if __name__ == "__main__":
    main()
