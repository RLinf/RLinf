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

"""Prepare the 40 G1 LAFAN1 retarget clips SuperMimic trains on.

Source: the ``g1/`` folder of the Hugging Face dataset
``lvhaidong/LAFAN1_Retargeting_Dataset`` at a pinned revision. Each CSV row
is root position (3), root quaternion xyzw (4) and 29 G1 joint angles at
30 Hz. Each clip becomes ``clip_<i>.pt`` (sorted by source name) with
``fps=30``; the loader resamples to the 50 Hz control rate.
``manifest.json`` records the revision, every source file's size and
SHA-256, and every output clip's SHA-256, and the loader checks it.

License: the dataset card declares no license. The retargets derive from
Ubisoft La Forge's LAFAN1, released under CC BY-NC-ND 4.0 (non-commercial,
no derivatives). Check those terms before downloading; pass
``--accept-license`` to confirm. Do not commit the data to a repository.

Download and convert::

    python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license

Convert CSVs fetched by other means (they are still checked against the
pinned revision's sums unless ``--skip-remote-check``)::

    python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 \\
        --source-dir /data/lafan1_csv --accept-license
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# ``python toolkits/supermimic/prepare_lafan1.py`` puts only this folder on sys.path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REPO_ID = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
SUBDIR = "g1"
EXPECTED_CLIPS = 40
SOURCE_FPS = 30.0
LICENSE_NOTE = (
    "Dataset card declares no license. Retargets derive from Ubisoft La Forge "
    "LAFAN1, CC BY-NC-ND 4.0 (non-commercial, no derivatives)."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, use_curl: bool, timeout: float) -> None:
    tmp = dest.with_name(dest.name + ".part")
    if use_curl:
        # -C - resumes a .part left by a dropped connection.
        subprocess.run(
            [
                "curl",
                "-4",
                "--http1.1",
                "-fsSL",
                "--retry",
                "5",
                "-C",
                "-",
                "-o",
                str(tmp),
                url,
            ],
            check=True,
            timeout=timeout,
        )
    else:
        with (
            urllib.request.urlopen(url, timeout=timeout) as response,
            open(tmp, "wb") as out,
        ):
            shutil.copyfileobj(response, out)
    os.replace(tmp, dest)


def remote_listing(
    endpoint: str, repo: str, revision: str, use_curl: bool, timeout: float
):
    """``[{name, bytes, sha256}]`` for the CSVs under ``g1/`` at ``revision``."""
    url = f"{endpoint}/api/datasets/{repo}/tree/{revision}/{SUBDIR}"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tree.json"
        _fetch(url, path, use_curl, timeout)
        entries = json.loads(path.read_text())
    files = []
    for entry in entries:
        name = Path(entry.get("path", "")).name
        if entry.get("type") != "file" or not name.endswith(".csv"):
            continue
        lfs = entry.get("lfs") or {}
        files.append(
            {
                "name": name,
                "bytes": int(lfs.get("size", entry.get("size", -1))),
                "sha256": lfs.get("oid"),
            }
        )
    return sorted(files, key=lambda item: item["name"])


def download(
    listing, endpoint, repo, revision, dest: Path, use_curl, timeout, attempts=5
):
    dest.mkdir(parents=True, exist_ok=True)
    for item in listing:
        target = dest / item["name"]
        url = f"{endpoint}/datasets/{repo}/resolve/{revision}/{SUBDIR}/{item['name']}"
        for attempt in range(attempts + 1):
            if target.is_file() and (
                not item["sha256"] or sha256_file(target) == item["sha256"]
            ):
                break
            if attempt == attempts:
                raise SystemExit(
                    f"could not fetch {item['name']} with a matching SHA-256"
                )
            target.unlink(missing_ok=True)
            print(
                f"[lafan1] fetching {item['name']} (attempt {attempt + 1})", flush=True
            )
            try:
                _fetch(url, target, use_curl, timeout)
            except (OSError, subprocess.SubprocessError) as error:
                print(f"[lafan1] {item['name']}: {error}", flush=True)
                time.sleep(min(2 * (attempt + 1), 10))


def check_sources(source_dir: Path, listing) -> list[dict]:
    """Every expected CSV present with the pinned size and sum."""
    found = sorted(p.name for p in source_dir.glob("*.csv"))
    if listing is None:
        return [
            {
                "name": n,
                "bytes": (source_dir / n).stat().st_size,
                "sha256": sha256_file(source_dir / n),
            }
            for n in found
        ]
    expected = [item["name"] for item in listing]
    missing = sorted(set(expected) - set(found))
    if missing:
        raise SystemExit(
            f"{source_dir} is missing {len(missing)} CSVs, e.g. {missing[:3]}"
        )
    checked = []
    for item in listing:
        path = source_dir / item["name"]
        size = path.stat().st_size
        digest = sha256_file(path)
        if item["bytes"] >= 0 and size != item["bytes"]:
            raise SystemExit(
                f"{path} has {size} bytes, revision {REVISION[:8]} has {item['bytes']}"
            )
        if item["sha256"] and digest != item["sha256"]:
            raise SystemExit(f"{path} SHA-256 does not match revision {REVISION[:8]}")
        checked.append({"name": item["name"], "bytes": size, "sha256": digest})
    return checked


def convert(source_dir: Path, sources: list[dict], out: Path) -> list[dict]:
    import numpy as np
    import torch

    from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM, G1_JOINT_NAMES, STATE_DIM
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
    from rlinf.envs.sim.g1_mimic.motion import decode_motion_payload

    q0 = asset_default_state()[7:36].clone().float()
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for index, item in enumerate(sources):
        raw = np.loadtxt(source_dir / item["name"], delimiter=",", dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 7 + ACTION_DIM:
            raise SystemExit(
                f"{item['name']} has shape {raw.shape}, expected [T, {7 + ACTION_DIM}]"
            )
        state = np.zeros((raw.shape[0], STATE_DIM), dtype=np.float32)
        state[:, 0:3] = raw[:, 0:3]
        # CSV quaternions are xyzw; the simulator state is wxyz.
        state[:, 3] = raw[:, 6]
        state[:, 4:7] = raw[:, 3:6]
        state[:, 7 : 7 + ACTION_DIM] = raw[:, 7:]
        payload = {
            "fps": SOURCE_FPS,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": q0,
            "state": torch.from_numpy(state),
            "source": f"{REPO_ID}@{REVISION} {SUBDIR}/{item['name']}",
        }
        name = f"clip_{index:03d}.pt"
        decoded, _ = decode_motion_payload(payload, name)
        torch.save(payload, out / name)
        quat_norm = np.linalg.norm(state[:, 3:7], axis=1)
        clips.append(
            {
                "file": name,
                "source": item["name"],
                "sha256": sha256_file(out / name),
                "frames_30hz": int(raw.shape[0]),
                "frames_50hz": int(decoded.shape[0]),
                "seconds": (raw.shape[0] - 1) / SOURCE_FPS,
                "root_z_range": [float(state[:, 2].min()), float(state[:, 2].max())],
                "quat_norm_range": [float(quat_norm.min()), float(quat_norm.max())],
            }
        )
        print(
            f"[lafan1] {name} <- {item['name']} ({clips[-1]['seconds']:.1f} s)",
            flush=True,
        )
    return clips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", required=True, help="directory for clip_*.pt and manifest.json"
    )
    parser.add_argument(
        "--source-dir", default="", help="existing CSV folder; skips download"
    )
    parser.add_argument(
        "--csv-cache", default="", help="where downloads go (default <out>/csv)"
    )
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument(
        "--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    )
    parser.add_argument(
        "--curl", action="store_true", help="download with curl -4 --http1.1"
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--skip-remote-check",
        action="store_true",
        help="with --source-dir, do not compare against the pinned revision",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=f"accept fewer than {EXPECTED_CLIPS} clips (not the paper set)",
    )
    parser.add_argument("--accept-license", action="store_true")
    args = parser.parse_args()

    print(f"[lafan1] {LICENSE_NOTE}")
    if not args.accept_license:
        parser.error("read the license note above, then pass --accept-license")
    endpoint = args.endpoint.rstrip("/")
    out = Path(args.out)
    listing = None
    if not (args.source_dir and args.skip_remote_check):
        listing = remote_listing(
            endpoint, REPO_ID, args.revision, args.curl, args.timeout
        )
        if len(listing) != EXPECTED_CLIPS and not args.allow_partial:
            raise SystemExit(
                f"revision {args.revision[:8]} lists {len(listing)} CSVs under {SUBDIR}/, "
                f"expected {EXPECTED_CLIPS}"
            )
    if args.source_dir:
        source_dir = Path(args.source_dir)
    else:
        source_dir = Path(args.csv_cache) if args.csv_cache else out / "csv"
        download(
            listing,
            endpoint,
            REPO_ID,
            args.revision,
            source_dir,
            args.curl,
            args.timeout,
        )
    sources = check_sources(source_dir, listing)
    if len(sources) != EXPECTED_CLIPS and not args.allow_partial:
        raise SystemExit(
            f"{source_dir} has {len(sources)} CSVs, expected {EXPECTED_CLIPS}"
        )
    clips = convert(source_dir, sources, out)
    manifest = {
        "dataset": REPO_ID,
        "revision": args.revision,
        "subdir": SUBDIR,
        "license_note": LICENSE_NOTE,
        "source_fps": SOURCE_FPS,
        "control_fps": 50.0,
        "q0": "mjlab G1 knees-bent keyframe (asset_default_state)",
        "converter": "toolkits/supermimic/prepare_lafan1.py",
        "checked_against_revision": listing is not None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "num_clips": len(clips),
        "total_seconds_30hz": sum(clip["seconds"] for clip in clips),
        "sources": sources,
        "clips": clips,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    hours = manifest["total_seconds_30hz"] / 3600.0
    print(
        f"[lafan1] wrote {len(clips)} clips, {hours:.2f} h, manifest {out / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
