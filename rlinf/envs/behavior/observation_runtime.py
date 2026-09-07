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

from __future__ import annotations

import torch


def parse_trunk_proprio_randomization(cfg):
    random_cfg = cfg.get("trunk_proprio_randomization", None)
    if random_cfg is None or not random_cfg.get("enabled", False):
        return None

    indices = list(random_cfg.get("indices", [236, 237, 238, 239]))
    low = list(random_cfg.get("low", [0.0, -0.45, 0.0, 0.0]))
    high = list(random_cfg.get("high", [0.7, 0.35, 0.18, 0.0]))
    fixed_values = list(random_cfg.get("fixed_values", [None, None, None, 0.0]))
    if not (len(indices) == len(low) == len(high) == len(fixed_values)):
        raise ValueError(
            "env.trunk_proprio_randomization indices, low, high, and "
            "fixed_values must have the same length."
        )
    return {
        "indices": [int(index) for index in indices],
        "low": [float(value) for value in low],
        "high": [float(value) for value in high],
        "fixed_values": [
            None if value is None else float(value) for value in fixed_values
        ],
    }


def sample_trunk_proprio_values(
    config: dict,
    *,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    dim = len(config["indices"])
    low = torch.tensor(config["low"], dtype=torch.float32)
    high = torch.tensor(config["high"], dtype=torch.float32)
    sampled = low + torch.rand(count, dim, generator=generator) * (high - low)
    for dim_idx, fixed_value in enumerate(config["fixed_values"]):
        if fixed_value is not None:
            sampled[:, dim_idx] = float(fixed_value)
    return sampled


def apply_trunk_proprio_values(
    states: torch.Tensor,
    *,
    config: dict,
    values: torch.Tensor,
    env_indices: list[int],
) -> torch.Tensor:
    states = states.clone()
    indices = config["indices"]
    if max(indices) >= states.shape[-1]:
        if states.shape[-1] < 7:
            return states
        indices = [3, 4, 5, 6]

    selected_values = values[env_indices].to(device=states.device, dtype=states.dtype)
    states[:, indices] = selected_values
    return states


def update_stage_prompts_from_info(
    stage_prompt_lists: list[list[str] | None],
    env_idx: int,
    info: dict | None,
) -> str | None:
    if not isinstance(info, dict):
        return None

    stage_info = info
    reward_info = info.get("reward")
    if isinstance(reward_info, dict):
        task_specific_info = reward_info.get("task_specific")
        if isinstance(task_specific_info, dict):
            stage_info = task_specific_info

    replay_info = info.get("replay_init")
    if isinstance(replay_info, dict):
        replay_prompts = replay_info.get("replay_stage_prompts")
        if isinstance(replay_prompts, (list, tuple)):
            stage_prompt_lists[env_idx] = [
                str(prompt).strip() for prompt in replay_prompts if str(prompt).strip()
            ]

    explicit_prompt = (
        stage_info.get("current_stage_prompt")
        or stage_info.get("stage_prompt")
        or stage_info.get("subtask_prompt")
    )
    if explicit_prompt:
        return str(explicit_prompt).strip()

    stage_idx = stage_info.get("current_stage_idx")
    if stage_idx is None:
        return None
    try:
        stage_idx = int(stage_idx)
    except (TypeError, ValueError):
        return None

    stage_prompts = stage_prompt_lists[env_idx]
    if not stage_prompts or stage_idx < 0 or stage_idx >= len(stage_prompts):
        return None
    return stage_prompts[stage_idx]


def compose_task_description(
    prompt_override: str | None,
    task_description: str,
    stage_prompt: str | None,
) -> str:
    if prompt_override is not None:
        return prompt_override
    if stage_prompt:
        return task_description + "\n" + stage_prompt
    return task_description


def task_descriptions_from_infos(
    *,
    num_envs: int,
    prompt_override: str | None,
    use_subtask_prompt: bool,
    task_description: str,
    stage_prompt_lists: list[list[str] | None],
    infos=None,
    env_indices=None,
) -> list[str]:
    """Build task descriptions while updating prompt state by global env row."""
    if prompt_override is not None:
        output_size = num_envs if env_indices is None else len(env_indices)
        return [prompt_override for _ in range(output_size)]
    if not use_subtask_prompt or infos is None:
        output_size = num_envs if env_indices is None else len(env_indices)
        return [task_description for _ in range(output_size)]
    output_env_indices = list(range(num_envs)) if env_indices is None else [
        int(index) for index in env_indices
    ]
    return [
        compose_task_description(
            prompt_override,
            task_description,
            update_stage_prompts_from_info(stage_prompt_lists, env_idx, info),
        )
        for env_idx, info in zip(output_env_indices, infos, strict=True)
    ]
