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

"""Compute normalization statistics for StreamingVLA LIBERO SFT data."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import openpi.shared.normalize as normalize
import tqdm

from rlinf.models.embodiment.streamingvla import dataconfig as _config
from rlinf.utils.logging import get_logger
from toolkits.lerobot.calculate_norm_stats import (
    create_rlds_dataloader,
    create_torch_dataloader,
)


def _copy_action_statistics(norm_stats: dict[str, Any]) -> dict[str, Any]:
    """Copy action statistics to action states as required by StreamingVLA."""
    if "actions" not in norm_stats:
        raise KeyError("Cannot create action_states stats without actions stats.")
    return {**norm_stats, "action_states": norm_stats["actions"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute StreamingVLA norm stats from a LIBERO dataset."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--config-name", default="pi05_libero_sfp")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    """Compute stats and save ``norm_stats.json`` under the selected asset."""
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    config = _config.get_streamingvla_config(
        args.config_name,
        batch_size=args.batch_size,
        data_kwargs={
            "repo_id": args.repo_id,
            "assets": {
                "assets_dir": str(args.assets_dir),
                "asset_id": args.asset_id,
            },
            "use_action_states": True,
        },
    )
    config = dataclasses.replace(config, num_workers=args.num_workers)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            args.max_frames,
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            args.max_frames,
        )
    if num_batches <= 0:
        raise ValueError(
            "No complete norm-stat batch is available; increase max_frames or "
            "decrease batch_size."
        )

    running_stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        running_stats["state"].update(np.asarray(batch["state"]))
        running_stats["actions"].update(np.asarray(batch["actions"]))

    norm_stats = {
        key: running_stat.get_statistics()
        for key, running_stat in running_stats.items()
    }
    norm_stats = _copy_action_statistics(norm_stats)
    output_path = args.assets_dir.expanduser() / args.asset_id
    get_logger().info(f"Writing StreamingVLA norm stats to {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    main()
