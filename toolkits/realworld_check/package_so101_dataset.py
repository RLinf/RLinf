# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Package and verify SO-101 LeRobot episodes for batch transfer.

This utility is deliberately local-only unless ``--rsync-destination`` is
provided. It never opens robot, camera, or policy-service devices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

try:
    from .check_so101_episode import check_dataset
except ImportError:  # Support direct ``python package_so101_dataset.py``.
    from check_so101_episode import check_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_files(dataset_path: Path) -> list[Path]:
    return sorted(path for path in dataset_path.rglob("*") if path.is_file())


def build_manifest(dataset_path: Path, archive_path: Path) -> dict[str, Any]:
    """Build a deterministic manifest for a finalized dataset."""
    info_path = dataset_path / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    files = [
        {"path": str(path.relative_to(dataset_path)), "sha256": _sha256(path)}
        for path in _relative_files(dataset_path)
    ]
    return {
        "schema_version": 1,
        "dataset_type": "lerobot",
        "robot_type": info.get("robot_type"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "archive": {
            "path": archive_path.name,
            "sha256": _sha256(archive_path) if archive_path.is_file() else None,
        },
        "files": files,
    }


def verify_manifest(dataset_path: Path, manifest_path: Path) -> list[str]:
    """Return manifest mismatches; an empty list means verification passed."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_files = {item["path"]: item["sha256"] for item in manifest["files"]}
    actual_files = {
        str(path.relative_to(dataset_path)): path
        for path in _relative_files(dataset_path)
    }
    for relative_path in sorted(expected_files.keys() - actual_files.keys()):
        errors.append(f"missing file: {relative_path}")
    for relative_path in sorted(actual_files.keys() - expected_files.keys()):
        errors.append(f"unexpected file: {relative_path}")
    for relative_path, expected_hash in expected_files.items():
        path = actual_files.get(relative_path)
        if path is not None and _sha256(path) != expected_hash:
            errors.append(f"hash mismatch: {relative_path}")
    archive = manifest.get("archive") or {}
    archive_path = manifest_path.parent / str(archive.get("path", ""))
    expected_archive_hash = archive.get("sha256")
    if expected_archive_hash:
        if not archive_path.is_file():
            errors.append(f"missing archive: {archive_path.name}")
        elif _sha256(archive_path) != expected_archive_hash:
            errors.append(f"archive hash mismatch: {archive_path.name}")
    return errors


def package_dataset(dataset_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Validate and package one dataset, returning archive and manifest paths."""
    checker_path = Path(__file__).with_name("check_so101_episode.py")
    report = check_dataset(
        dataset_path,
        checker_path.parent.parent.parent
        / "examples/embodiment/config/so101/validation_schema.yaml",
    )
    if not report.passed:
        raise ValueError("Dataset validation failed: " + "; ".join(report.errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{dataset_path.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(dataset_path, arcname=dataset_path.name, recursive=True)
    manifest = build_manifest(dataset_path, archive_path)
    manifest_path = output_dir / f"{dataset_path.name}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archive_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--rsync-destination")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    if args.verify_manifest:
        errors = verify_manifest(
            dataset_path, args.verify_manifest.expanduser().resolve()
        )
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print("manifest verification: PASS")
        return 0

    archive_path, manifest_path = package_dataset(
        dataset_path, args.output_dir.expanduser().resolve()
    )
    print(f"archive: {archive_path}")
    print(f"manifest: {manifest_path}")
    if args.rsync_destination:
        command = [
            "rsync",
            "-av",
            "--partial",
            str(archive_path),
            str(manifest_path),
            args.rsync_destination,
        ]
        print("rsync command: " + " ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
