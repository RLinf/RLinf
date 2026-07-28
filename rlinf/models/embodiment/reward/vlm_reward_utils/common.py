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

from typing import Any

import torch
from transformers import AutoProcessor


def load_vlm_processor(model_path: str, subprocessor_kwargs: Any = None):
    """Load an HF processor and optionally override nested subprocessor attrs."""
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    for subprocessor_name, kwargs in dict(subprocessor_kwargs or {}).items():
        subprocessor = getattr(processor, subprocessor_name, None)
        if subprocessor is None:
            continue
        for key, value in dict(kwargs).items():
            if hasattr(subprocessor, key):
                setattr(subprocessor, key, value)
    return processor


def apply_gt_success_bonus(
    rewards: torch.Tensor,
    reward_input: dict[str, Any],
    gt_success_bonus: float,
) -> torch.Tensor:
    """Add a success bonus from env infos when configured."""
    if rewards is None or gt_success_bonus == 0.0:
        return rewards
    env_infos = (
        reward_input.get("env_infos") if isinstance(reward_input, dict) else None
    )
    if not isinstance(env_infos, dict):
        return rewards

    success = None
    final_info = env_infos.get("final_info", {})
    for info_dict in (
        env_infos,
        env_infos.get("episode"),
        final_info,
        final_info.get("episode") if isinstance(final_info, dict) else None,
    ):
        if not isinstance(info_dict, dict):
            continue
        for key in ("success", "success_at_end", "success_once"):
            value = info_dict.get(key)
            if value is not None:
                success = torch.as_tensor(value).reshape(-1).bool()
                break
        if success is not None:
            break

    if success is None or success.shape[0] != rewards.shape[0]:
        return rewards
    bonus = success.to(device=rewards.device, dtype=rewards.dtype)
    return rewards + (bonus * gt_success_bonus).view(-1, *([1] * (rewards.dim() - 1)))
