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

import json
import os
from importlib.util import find_spec
from pathlib import Path

from omegaconf import DictConfig, open_dict

from rlinf.utils.data_iter_utils import ceildiv


def resolve_behavior_eval_config(cfg: DictConfig, chunk_size: int) -> None:
    """Fill in the step limits for a BEHAVIOR evaluation.

    Args:
        cfg: Environment configuration to update before rollout validation.
        chunk_size: Number of actions the policy predicts per chunk.
    """
    with open_dict(cfg):
        cfg.is_eval = True
    if cfg.max_episode_steps is None:
        # Read the official task statistics without importing the simulator on the driver.
        root = Path(
            os.environ.get("OMNIGIBSON_DATA_PATH", "../../datasets")
        ).expanduser()
        if not root.is_absolute():
            package = find_spec("omnigibson")
            if package is None:
                raise ModuleNotFoundError(
                    "Set OMNIGIBSON_DATA_PATH or install OmniGibson."
                )
            root = Path(package.origin).resolve().parent / root
        stats = root / "2026-challenge-task-instances" / "metadata" / "task.jsonl"
        task = cfg.omni_config.task.activity_name
        with stats.open() as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["task_name"] == task:
                    # The SDK tests its zero-based step counter before incrementing it.
                    cfg.max_episode_steps = int(row["length"] * 1.5) + 1
                    break
            else:
                raise ValueError(
                    f"No human demonstration length for {task!r} in {stats}"
                )
    if cfg.max_episode_steps < 1 or chunk_size < 1:
        raise ValueError(
            "BEHAVIOR episode length and action chunk size must be positive."
        )
    chunks = ceildiv(cfg.max_episode_steps, chunk_size)
    episode_budget = chunks * chunk_size
    if cfg.max_steps_per_rollout_epoch is None:
        cfg.max_steps_per_rollout_epoch = episode_budget
    elif cfg.max_steps_per_rollout_epoch <= 0:
        raise ValueError("BEHAVIOR max_steps_per_rollout_epoch must be positive.")
    elif (
        not cfg.auto_reset
        and not cfg.get("early_stop_on_done", False)
        and cfg.max_steps_per_rollout_epoch > episode_budget
    ):
        raise ValueError(
            f"With auto_reset=False, the episode fits in {chunks} action chunks "
            f"({episode_budget} rollout steps). Reduce max_steps_per_rollout_epoch "
            "or enable early_stop_on_done."
        )
