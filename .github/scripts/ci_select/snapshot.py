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

"""Read-only views of a repository at one revision.

Every analysis reads files through a ``Snapshot`` so the same code runs on a
CI checkout (``FsSnapshot``) and on historical commits (``GitSnapshot``).
Parsed results are cached by blob id, which makes replaying many adjacent
commits cheap.
"""

import fnmatch
import os
import subprocess
from functools import cached_property
from pathlib import Path


class Snapshot:
    """The files of one revision, addressed by repo-relative POSIX path."""

    #: Stable id of a file's content, used as a cache key across snapshots.
    def blob_id(self, path: str) -> str | None:
        raise NotImplementedError

    def read(self, path: str) -> str | None:
        raise NotImplementedError

    @cached_property
    def files(self) -> frozenset[str]:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        return path in self.files

    @cached_property
    def dirs(self) -> frozenset[str]:
        out = set()
        for f in self.files:
            parts = f.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                out.add("/".join(parts[:i]))
        return frozenset(out)

    def under(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/") + "/"
        return sorted(f for f in self.files if f.startswith(prefix))

    def glob(self, pattern: str) -> list[str]:
        """Match like the shell: ``*`` stays within one path segment."""
        depth = pattern.count("/")
        return sorted(
            f
            for f in self.files
            if f.count("/") == depth and fnmatch.fnmatchcase(f, pattern)
        )


class GitSnapshot(Snapshot):
    def __init__(self, repo: "GitRepo", rev: str):
        self.repo = repo
        self.rev = rev
        self._tree = repo.tree(rev)

    def blob_id(self, path: str) -> str | None:
        return self._tree.get(path)

    def read(self, path: str) -> str | None:
        sha = self._tree.get(path)
        return None if sha is None else self.repo.blob(sha)

    @cached_property
    def files(self) -> frozenset[str]:
        return frozenset(self._tree)


class FsSnapshot(Snapshot):
    """The working tree, for running the selector inside a CI checkout."""

    def __init__(self, root: str):
        self.root = Path(root)

    def blob_id(self, path: str) -> str | None:
        p = self.root / path
        if not p.is_file():
            return None
        st = p.stat()
        return f"{path}:{st.st_mtime_ns}:{st.st_size}"

    def read(self, path: str) -> str | None:
        try:
            return (self.root / path).read_text(errors="replace")
        except (FileNotFoundError, IsADirectoryError):
            return None

    @cached_property
    def files(self) -> frozenset[str]:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            check=True,
        ).stdout.decode()
        return frozenset(
            p for p in out.split("\0") if p and os.path.isfile(self.root / p)
        )


class GitRepo:
    """Batched object reads over one long-lived ``git cat-file`` process."""

    def __init__(self, path: str):
        self.path = path
        self._cat = subprocess.Popen(
            ["git", "-C", path, "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._blobs: dict[str, str] = {}
        self._trees: dict[str, dict[str, str]] = {}

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", self.path, *args], capture_output=True, check=True, text=True
        ).stdout

    def tree(self, rev: str) -> dict[str, str]:
        if rev not in self._trees:
            out = subprocess.run(
                ["git", "-C", self.path, "ls-tree", "-r", "-z", rev],
                capture_output=True,
                check=True,
            ).stdout.decode()
            tree = {}
            for entry in out.split("\0"):
                if not entry:
                    continue
                meta, path = entry.split("\t", 1)
                mode, kind, sha = meta.split()
                if kind == "blob" and mode != "120000":
                    tree[path] = sha
            self._trees[rev] = tree
        return self._trees[rev]

    def blob(self, sha: str) -> str:
        if sha not in self._blobs:
            self._cat.stdin.write(sha.encode() + b"\n")
            self._cat.stdin.flush()
            header = self._cat.stdout.readline().split()
            size = int(header[2])
            data = self._cat.stdout.read(size)
            self._cat.stdout.read(1)
            self._blobs[sha] = data.decode(errors="replace")
        return self._blobs[sha]

    def snapshot(self, rev: str) -> GitSnapshot:
        return GitSnapshot(self, rev)
