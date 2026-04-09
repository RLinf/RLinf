#!/usr/bin/env python
# Copyright 2025 The RLinf Authors.
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

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement


def _write_scalar(log_root: str, tag: str, value: int | float) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "tensorboard is required for profile marker logging. Try `pip install tensorboard`."
        ) from exc

    # Keep consistent with `toolkits/auto_placement/util.py` extraction logic.
    tb_logdir = os.path.join(log_root, "tensorboard", "all")
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(tb_logdir)
    writer.add_scalar(tag, value, 0)
    writer.flush()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write TensorBoard markers for auto-placement profiling."
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--total-num-envs", type=int, required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config_path) / f"{args.config_name}.yaml"
    cfg = OmegaConf.create(OmegaConf.load(str(cfg_path)))

    # Mirror profiling-time overrides (previously in modify_cfg_for_profiling()).
    with open_dict(cfg):
        cfg.cluster.component_placement = {"actor": "all", "rollout": "all", "env": "all"}
        cfg.env.train.total_num_envs = int(args.total_num_envs)
        cfg.runner.logger.per_worker_log = True
        cfg.rollout.pipeline_stage_num = 1

    cfg = validate_cfg(cfg)
    placement = HybridComponentPlacement(cfg, Cluster())
    env_world_size = int(placement.get_world_size("env"))
    if env_world_size <= 0:
        raise ValueError(f"Invalid env_world_size={env_world_size}")

    env_num_per_instance = int(args.total_num_envs) // env_world_size
    _write_scalar(args.log_dir, "profile/env_num_per_instance", env_num_per_instance)
    _write_scalar(args.log_dir, "profile/total_num_envs", int(args.total_num_envs))


if __name__ == "__main__":
    main()

