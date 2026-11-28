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

"""Download the frozen starting weights for one Go2 locomotion task."""

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

# Each task maps to a public Git blob and the SHA-256 of its checkpoint.
BASELINES = {
    "A": (
        "5d7c6742e9c4016274c022702c1f1549a0fc5b5e",
        "3bead8f500d495b299b72724c8ac1e998d525f2d7984826916638bcfbd99a463",
    ),
    "B": (
        "ba47544424afbcb5bbc490f921117c829bbe090e",
        "75ce9c0f82178884444dd40331ecf34905b162e29f45b5a40d54cab3ad8891f3",
    ),
    "C": (
        "d1e7f630bc1c099f7991e6fe44912155c192a594",
        "c9cd3187d058b1ff1ae172c5a947adfff6b60922de451aa4184408b6ba67aff1",
    ),
}


def verify(path: Path, digest: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256 for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=BASELINES)
    parser.add_argument(
        "--data",
        type=Path,
        default=os.environ.get("GO2_ABC_DATA", "data/go2_abc"),
        help="Data root containing models/<task>/baseline.pt",
    )
    args = parser.parse_args()
    target = args.data / "models" / args.task / "baseline.pt"
    blob, digest = BASELINES[args.task]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            f"https://api.github.com/repos/QuincyQAQ/RLinf/git/blobs/{blob}",
            headers={"Accept": "application/vnd.github.raw+json"},
        )
        with tempfile.TemporaryDirectory(dir=target.parent) as staging:
            staged = Path(staging) / target.name
            with (
                urllib.request.urlopen(request, timeout=90) as stream,
                staged.open("wb") as output,
            ):
                shutil.copyfileobj(stream, output)
            verify(staged, digest)
            try:
                target.hardlink_to(staged)
            except FileExistsError:
                pass  # Concurrent downloads verify the same published file below.
    verify(target, digest)


if __name__ == "__main__":
    main()
