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

import hydra
from omegaconf import DictConfig

from rlinf.envs.layout import EnvLayout
from rlinf.utils.data_iter_utils import ceildiv


def resolve_step_limits(env_cfg: DictConfig, chunk_size: int) -> None:
    """Apply the challenge timeout after Hydra has composed this job's overrides."""
    if env_cfg.max_episode_steps is None:
        root = Path(
            os.environ.get("OMNIGIBSON_DATA_PATH", "../../datasets")
        ).expanduser()
        if not root.is_absolute():
            package = find_spec("omnigibson")
            if package is None or package.origin is None:
                raise ValueError("Set OMNIGIBSON_DATA_PATH to the BEHAVIOR dataset.")
            root = Path(package.origin).parent / root
        stats = root / "2026-challenge-task-instances/metadata/task.jsonl"
        task = env_cfg.omni_config.task.activity_name
        with stats.open() as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["task_name"] == task:
                    # The official timeout checks a zero-based step counter.
                    env_cfg.max_episode_steps = int(row["length"] * 1.5) + 1
                    break
            else:
                raise ValueError(f"No demonstration length for {task!r} in {stats}")
    if env_cfg.max_episode_steps < 1 or chunk_size < 1:
        raise ValueError("The episode length and action chunk size must be positive.")
    budget = ceildiv(env_cfg.max_episode_steps, chunk_size) * chunk_size
    if env_cfg.max_steps_per_rollout_epoch is None:
        env_cfg.max_steps_per_rollout_epoch = budget
    elif env_cfg.max_steps_per_rollout_epoch < 1:
        raise ValueError("max_steps_per_rollout_epoch must be positive.")
    elif (
        not env_cfg.auto_reset
        and not env_cfg.get("early_stop_on_done", False)
        and env_cfg.max_steps_per_rollout_epoch > budget
    ):
        raise ValueError(
            f"The episode fits in {budget} rollout steps. Reduce "
            "max_steps_per_rollout_epoch or enable early_stop_on_done."
        )


@hydra.main(
    version_base="1.1", config_path=".", config_name="behavior2026_openpi_pi05_eval"
)
def main(cfg: DictConfig) -> None:
    from evaluations.eval_embodied_agent import main as run_eval

    for env_cfg in EnvLayout.env_cfgs(cfg.env.eval, "env.eval").values():
        if env_cfg.env_type == "behavior2026":
            resolve_step_limits(env_cfg, cfg.rollout.model.num_action_chunks)
    run_eval(cfg)


if __name__ == "__main__":
    main()
