#!/usr/bin/env python
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
"""
Build a deterministic per-task subset of a LeRobot v2.1 dataset, with
contiguous episode_index renumbering (required by LingBot-VA's loader).

What this does:
  1. Pick `per_task` episodes from each task (grouped via tasks.jsonl /
     parquet task_index) using a seeded RNG.
  2. Renumber the selected episodes 0..N-1 contiguously. This is mandatory:
     LatentLeRobotDataset indexes `episode_data_index['from']` positionally,
     so a non-contiguous episode_index field raises IndexError at training.
  3. Symlink the per-episode parquet + video files from the source dataset
     so the subset doesn't duplicate ~7 GB of mp4s on disk.
  4. Pre-create dangling latent symlinks under `<subset>/latents/.../
     episode_<subset_idx>_0_T.pth` pointing at `<src>/latents/.../
     episode_<orig_idx>_0_T.pth`. When extract_latents.py later runs on the
     subset, torch.save follows the symlink and writes the actual .pth into
     the parent dataset — so multiple subsets can share latents.
  5. Record `original_episode_index` on every line of the rewritten
     episodes.jsonl for traceability.

The subset's episodes.jsonl keeps the LingBot-VA `action_config` field that
was injected at conversion time; nothing in this script touches action_config
contents (other than re-keying the line by the new episode_index).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

CHUNK_DIR = "chunk-000"
CAM_KEYS = [
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def relative_symlink(target: Path, link: Path) -> None:
    """Symlink with a relative path, so the subset dir is portable
    alongside the parent."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    rel = os.path.relpath(target, start=link.parent)
    link.symlink_to(rel)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def task_index_for_episode(src: Path, episode_index: int) -> int:
    """Read a single value out of the parquet — cheap, avoids loading frames."""
    import pyarrow.parquet as pq
    pq_path = src / "data" / CHUNK_DIR / f"episode_{episode_index:06d}.parquet"
    tbl = pq.read_table(pq_path, columns=["task_index"])
    return int(tbl.column("task_index")[0].as_py())


def build_subset(src: Path, dst: Path, per_task: int, seed: int) -> None:
    if dst.exists():
        raise SystemExit(
            f"Destination exists: {dst}. Remove it first to avoid mixing runs."
        )
    src_meta = src / "meta"
    if not (src_meta / "info.json").exists():
        raise SystemExit(f"Source is not a LeRobot dataset: {src}")

    info = json.loads((src_meta / "info.json").read_text())
    episodes = read_jsonl(src_meta / "episodes.jsonl")
    episodes_stats = {
        e["episode_index"]: e for e in read_jsonl(src_meta / "episodes_stats.jsonl")
    }
    tasks = read_jsonl(src_meta / "tasks.jsonl")

    by_task: dict[int, list[int]] = defaultdict(list)
    for ep in episodes:
        ti = task_index_for_episode(src, ep["episode_index"])
        by_task[ti].append(ep["episode_index"])

    rng = np.random.default_rng(seed)
    selected: list[tuple[int, int]] = []  # (task_index, original_episode_index)
    for ti in sorted(by_task):
        pool = sorted(by_task[ti])
        if len(pool) < per_task:
            raise SystemExit(
                f"Task {ti} has only {len(pool)} episodes, asked for {per_task}"
            )
        picks = rng.choice(len(pool), size=per_task, replace=False)
        for idx in sorted(picks.tolist()):
            selected.append((ti, pool[idx]))
    print(f"[subset] selected {len(selected)} episodes "
          f"({len(by_task)} tasks × {per_task})")

    # Renumber.
    new_episodes: list[dict] = []
    new_episodes_stats: list[dict] = []
    new_to_old: dict[int, int] = {}
    cum = 0
    for new_idx, (_, old_idx) in enumerate(selected):
        new_to_old[new_idx] = old_idx
        ep = next(e for e in episodes if e["episode_index"] == old_idx)
        new_ep = dict(ep)
        new_ep["episode_index"] = new_idx
        new_ep["original_episode_index"] = old_idx
        new_episodes.append(new_ep)

        st = dict(episodes_stats[old_idx])
        st["episode_index"] = new_idx
        new_episodes_stats.append(st)
        cum += ep["length"]

    # Layout.
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "data" / CHUNK_DIR).mkdir(parents=True, exist_ok=True)
    for cam in CAM_KEYS:
        (dst / "videos" / CHUNK_DIR / cam).mkdir(parents=True, exist_ok=True)
        (dst / "latents" / CHUNK_DIR / cam).mkdir(parents=True, exist_ok=True)

    # tasks.jsonl: identical to parent (task_index references unchanged).
    write_jsonl(dst / "meta" / "tasks.jsonl", tasks)

    # info.json: update totals, keep schema.
    info_out = dict(info)
    info_out["total_episodes"] = len(new_episodes)
    info_out["total_frames"] = cum
    info_out["total_videos"] = len(new_episodes) * len(CAM_KEYS)
    info_out["splits"] = {"train": f"0:{len(new_episodes)}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info_out, indent=2))

    write_jsonl(dst / "meta" / "episodes.jsonl", new_episodes)
    write_jsonl(dst / "meta" / "episodes_stats.jsonl", new_episodes_stats)

    # Symlink parquet + mp4 files (renumbered name -> original path).
    for new_idx, old_idx in new_to_old.items():
        old_pq = src / "data" / CHUNK_DIR / f"episode_{old_idx:06d}.parquet"
        new_pq = dst / "data" / CHUNK_DIR / f"episode_{new_idx:06d}.parquet"
        relative_symlink(old_pq, new_pq)
        for cam in CAM_KEYS:
            old_mp4 = (src / "videos" / CHUNK_DIR / cam
                       / f"episode_{old_idx:06d}.mp4")
            new_mp4 = (dst / "videos" / CHUNK_DIR / cam
                       / f"episode_{new_idx:06d}.mp4")
            relative_symlink(old_mp4, new_mp4)

        # Pre-create dangling latent symlinks. extract_latents.py will write
        # through them; the actual bytes land in the parent.
        ep_length = next(e for e in new_episodes
                         if e["episode_index"] == new_idx)["length"]
        for cam in CAM_KEYS:
            old_latent = (src / "latents" / CHUNK_DIR / cam
                          / f"episode_{old_idx:06d}_0_{ep_length}.pth")
            new_latent = (dst / "latents" / CHUNK_DIR / cam
                          / f"episode_{new_idx:06d}_0_{ep_length}.pth")
            # Ensure parent's latent dir exists so the dangling target is in
            # a real directory (Python's open('wb') on a dangling symlink
            # creates the file at the target only if the target's parent dir
            # already exists).
            old_latent.parent.mkdir(parents=True, exist_ok=True)
            relative_symlink(old_latent, new_latent)

    # Save the selection manifest for auditing.
    manifest = {
        "src": str(src.resolve()),
        "seed": seed,
        "per_task": per_task,
        "selected": [
            {"new": ni, "original": old, "task_index": ti}
            for ni, (ti, old) in enumerate(selected)
        ],
    }
    (dst / "meta" / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"[subset] wrote {dst}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True,
                   help="Parent LeRobot dataset (output of convert_*.py)")
    p.add_argument("--dst", type=Path, required=True,
                   help="Output subset dataset directory")
    p.add_argument("--per-task", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_subset(args.src, args.dst, args.per_task, args.seed)
