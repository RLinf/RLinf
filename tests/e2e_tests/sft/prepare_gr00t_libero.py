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

"""Prepare one real LIBERO episode for the GR00T SFT integration test."""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gr00t-path", type=Path, required=True)
    args = parser.parse_args()
    snapshot_download(
        "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
        repo_type="dataset",
        revision="bf14d6258218d12c2e3c1a3b9922e163cdf6455d",
        local_dir=args.output_dir,
        allow_patterns=[
            "meta/*",
            "data/chunk-000/episode_000000.parquet",
            "videos/chunk-000/*/episode_000000.mp4",
        ],
    )
    meta = args.output_dir / "meta"
    episodes_path = meta / "episodes.jsonl"
    episode = json.loads(episodes_path.read_text().splitlines()[0])
    episodes_path.write_text(json.dumps(episode) + "\n")
    episodes_stats_path = meta / "episodes_stats.jsonl"
    episodes_stats_path.write_text(
        episodes_stats_path.read_text().splitlines()[0] + "\n"
    )
    info_path = meta / "info.json"
    info = json.loads(info_path.read_text())
    info.update(
        total_episodes=1, total_frames=episode["length"], splits={"train": "0:1"}
    )
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    shutil.copyfile(
        args.gr00t_path / "examples/LIBERO/modality.json", meta / "modality.json"
    )

    from gr00t.data.stats import generate_stats

    generate_stats(args.output_dir)


if __name__ == "__main__":
    main()
