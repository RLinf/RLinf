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

"""On-demand ZIP extraction, asset caching, and LRU eviction for RoboUniverse-10K."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import urlopen

from rlinf.envs.sim.robouniverse.catalog import TaskRecord

_ALLOWED_ASSET_PREFIXES = (
    "https://raw.githubusercontent.com/agilexrobotics/piper_isaac_sim/",
    "https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/resolve/",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LAST_USED_MARKER = ".rlinf_last_used"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_join(root: Path, rel_path: str) -> Path:
    """Resolve a relative archive path strictly inside ``root``."""
    candidate = Path(rel_path)
    if (
        not candidate.parts
        or candidate.is_absolute()
        or ".." in candidate.parts
        or "\\" in rel_path
    ):
        raise ValueError(
            f"Expected a safe relative path inside {root}, found {rel_path!r}."
        )
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Path {rel_path!r} escapes extraction root {root_resolved}.")
    return resolved


def _quote_url(url: str) -> str:
    """Percent-encode only the path segment of ``url``, preserving query parameters."""
    parts = urlsplit(url)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            quote(parts.path, safe="/%"),
            parts.query,
            parts.fragment,
        )
    )


@dataclass(frozen=True)
class MaterializedTask:
    """Unpacked task workspace ready for the RoboUniverse simulator worker."""

    record: TaskRecord
    task_dir: Path
    workspace_dir: Path
    runtime_dir: Path
    instruction: str
    env_spec: dict[str, Any]


class TaskMaterializer:
    """Own on-demand extraction, asset caching, and disk-bounded LRU cleanup."""

    def __init__(
        self,
        data_root: str | Path,
        work_dir: str | Path,
        *,
        tasks_dir: str | Path | None = None,
        asset_cache_dir: str | Path | None = None,
        max_cached_tasks: int | None = 32,
        verify_sha256: bool = True,
        offline: bool = False,
        download_timeout_s: float = 90.0,
    ) -> None:
        if max_cached_tasks is not None and max_cached_tasks < 1:
            raise ValueError(
                f"Expected max_cached_tasks >= 1 or None, found {max_cached_tasks}."
            )
        self.data_root = Path(data_root).expanduser().resolve()
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.tasks_dir = (
            Path(tasks_dir).expanduser().resolve()
            if tasks_dir is not None
            else self.work_dir / "tasks"
        )
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.asset_cache_dir = (
            Path(asset_cache_dir).expanduser().resolve()
            if asset_cache_dir is not None
            else self.work_dir / "asset_cache"
        )
        self.asset_cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cached_tasks = max_cached_tasks
        self.verify_sha256 = verify_sha256
        self.offline = offline
        self.download_timeout_s = download_timeout_s
        self._lock_path = self.tasks_dir / ".materialize.lock"

    def materialize(
        self,
        record: TaskRecord,
        *,
        protected_cases: frozenset[str] = frozenset(),
    ) -> MaterializedTask:
        """Return an unpacked :class:`MaterializedTask`, extracting on demand.

        Args:
            record: Catalog entry to unpack.
            protected_cases: Case names that LRU eviction must keep on disk.

        Returns:
            Verified task paths, natural-language instruction, and ``env_spec``.
        """
        task_dir = self.tasks_dir / record.case
        with self._lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if not self._is_complete(task_dir):
                    self._extract_atomically(record, task_dir)
                self._touch(task_dir)
                if self.max_cached_tasks is not None:
                    keep = set(protected_cases) | {record.case}
                    self._evict_lru(keep)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        return self._load_metadata(record, task_dir)

    def _is_complete(self, task_dir: Path) -> bool:
        return (
            task_dir / "environment" / "sim" / "workspace" / "env_spec.json"
        ).is_file()

    def _touch(self, task_dir: Path) -> None:
        (task_dir / _LAST_USED_MARKER).write_text(
            f"{time.time():.6f}\n", encoding="utf-8"
        )

    def _extract_atomically(self, record: TaskRecord, task_dir: Path) -> None:
        archive_path = _safe_join(self.data_root, record.archive)
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Expected task archive for {record.case!r} at {archive_path}, "
                "but the file does not exist."
            )
        if self.verify_sha256:
            actual_sha = _file_sha256(archive_path)
            if actual_sha != record.sha256:
                raise ValueError(
                    f"Archive SHA-256 mismatch for {record.case!r}: "
                    f"expected {record.sha256}, found {actual_sha}."
                )

        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{record.case}.", dir=self.tasks_dir)
        )
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    target = _safe_join(staging_dir, member.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    if member.filename.endswith(".sh") or target.name == "roboenv":
                        target.chmod(0o755)

            release_path = staging_dir / "release.json"
            if release_path.is_file():
                release = json.loads(release_path.read_text(encoding="utf-8"))
                if release.get("case") != record.case:
                    raise ValueError(
                        f"Expected release.json case {record.case!r}, "
                        f"found {release.get('case')!r}."
                    )

            upstream_path = staging_dir / "upstream.json"
            if upstream_path.is_file():
                upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
                for rel_target, (url, checksum) in upstream.items():
                    self._fetch_asset(rel_target, url, checksum, staging_dir)

            (staging_dir / "environment" / "sim" / "runtime").mkdir(
                parents=True, exist_ok=True
            )
            if task_dir.exists():
                shutil.rmtree(task_dir)
            os.replace(staging_dir, task_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def _fetch_asset(
        self, rel_target: str, url: str, checksum: str, dest: Path
    ) -> None:
        if not _SHA256_RE.fullmatch(checksum) or not url.startswith(
            _ALLOWED_ASSET_PREFIXES
        ):
            raise ValueError(
                f"Unrecognized upstream asset dependency for {rel_target!r}: "
                f"url={url!r}, checksum={checksum!r}."
            )
        cached = self.asset_cache_dir / checksum
        if not cached.is_file() or (
            self.verify_sha256 and _file_sha256(cached) != checksum
        ):
            if self.offline:
                raise FileNotFoundError(
                    f"Missing cached upstream asset {rel_target!r} (sha256={checksum}) "
                    f"in {self.asset_cache_dir} while offline=True."
                )
            for attempt in range(3):
                fd, tmp_name = tempfile.mkstemp(dir=self.asset_cache_dir)
                tmp_path = Path(tmp_name)
                try:
                    with (
                        os.fdopen(fd, "wb") as tmp_file,
                        urlopen(
                            _quote_url(url), timeout=self.download_timeout_s
                        ) as response,
                    ):
                        shutil.copyfileobj(response, tmp_file)
                    actual = _file_sha256(tmp_path)
                    if actual != checksum:
                        raise ValueError(
                            f"Upstream asset hash mismatch for {rel_target!r}: "
                            f"expected {checksum}, found {actual}."
                        )
                    os.replace(tmp_path, cached)
                    break
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
                finally:
                    tmp_path.unlink(missing_ok=True)

        target = _safe_join(dest, rel_target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)

    def _evict_lru(self, protected: set[str]) -> None:
        entries: list[tuple[float, Path]] = []
        for entry in self.tasks_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            marker = entry / _LAST_USED_MARKER
            try:
                timestamp = float(marker.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                timestamp = entry.stat().st_mtime
            entries.append((timestamp, entry))

        if len(entries) <= self.max_cached_tasks:
            return

        entries.sort(key=lambda item: item[0])
        to_remove = len(entries) - self.max_cached_tasks
        for _, entry in entries:
            if to_remove <= 0:
                break
            if entry.name in protected:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            to_remove -= 1

    @staticmethod
    def _load_metadata(record: TaskRecord, task_dir: Path) -> MaterializedTask:
        sim_dir = task_dir / "environment" / "sim"
        workspace_dir = sim_dir / "workspace"
        runtime_dir = sim_dir / "runtime"
        spec_path = workspace_dir / "env_spec.json"
        if not spec_path.is_file():
            raise FileNotFoundError(
                f"Expected env_spec.json for {record.case!r} at {spec_path}."
            )
        env_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        instruction_rel = env_spec.get("instruction_path", "task_instruction.md")
        instruction_path = _safe_join(workspace_dir, instruction_rel)
        if not instruction_path.is_file():
            instruction_path = task_dir / "instruction.md"
        if not instruction_path.is_file():
            raise FileNotFoundError(
                f"Expected task instruction file for {record.case!r} in {workspace_dir}."
            )
        instruction = instruction_path.read_text(encoding="utf-8").strip()
        return MaterializedTask(
            record=record,
            task_dir=task_dir,
            workspace_dir=workspace_dir,
            runtime_dir=runtime_dir,
            instruction=instruction,
            env_spec=env_spec,
        )
