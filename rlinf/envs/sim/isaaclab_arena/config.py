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

"""Configuration contracts shared by validation and the Arena env worker."""

from omegaconf import DictConfig, OmegaConf


def validate_arena_config(
    cfg: DictConfig, *, evaluation: bool, action_chunk: int | None = None
) -> None:
    if cfg.get("group_size", 1) != 1 or any(
        cfg.get(key, False)
        for key in ("use_fixed_reset_state_ids", "use_ordered_reset_state_ids")
    ):
        raise ValueError(
            "Arena samples independent placements: use group_size=1 and disable fixed or ordered reset state IDs"
        )
    if OmegaConf.select(cfg, "data_collection.enabled", default=False):
        raise ValueError(
            "Isaac Lab-Arena cannot export episode datasets. Set data_collection.enabled=false; "
            "use video_cfg.save_video=true to record rollouts."
        )
    if cfg.ignore_terminations:
        raise ValueError("Isaac Lab-Arena requires ignore_terminations=false")
    if type(cfg.max_episode_steps) is not int or cfg.max_episode_steps < 1:
        raise ValueError(
            "Isaac Lab-Arena requires a positive integer max_episode_steps"
        )
    if cfg.auto_reset == evaluation:
        raise ValueError(
            "Isaac Lab-Arena requires auto_reset=false for evaluation and auto_reset=true for training"
        )
    if evaluation and cfg.max_steps_per_rollout_epoch < cfg.max_episode_steps:
        raise ValueError(
            "Arena evaluation needs max_steps_per_rollout_epoch >= max_episode_steps so every assigned episode finishes"
        )
    if not evaluation and action_chunk not in (None, 1):
        raise ValueError(
            "Isaac Lab-Arena training requires num_action_chunks=1 so every PPO action is executed"
        )
