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


def load_vlm_processor(model_path: str, subprocessor_kwargs: Any = None):
    """Load an HF processor and optionally override nested subprocessor attrs."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    for subprocessor_name, kwargs in dict(subprocessor_kwargs or {}).items():
        subprocessor = getattr(processor, subprocessor_name, None)
        if subprocessor is None:
            continue
        for key, value in dict(kwargs).items():
            if hasattr(subprocessor, key):
                setattr(subprocessor, key, value)
    return processor


def _success_flags(infos: Any) -> torch.Tensor | None:
    if not isinstance(infos, dict):
        return None
    for candidate in (infos, infos.get("episode")):
        if not isinstance(candidate, dict):
            continue
        for key in ("success", "success_at_end", "success_once"):
            if candidate.get(key) is not None:
                return torch.as_tensor(candidate[key]).reshape(-1).bool()
    return None


def apply_gt_success_bonus(
    rewards: torch.Tensor,
    reward_input: dict[str, Any],
    gt_success_bonus: float,
) -> torch.Tensor:
    """Add per-step success bonuses, using final infos for ended episodes.

    With auto-reset, ``env_infos`` can describe the next episode. For entries
    marked by ``dones`` (or ``_final_info``), use ``final_info`` instead. Without
    a boundary mask, retain current-info precedence and the final-info fallback.
    """
    if rewards is None or gt_success_bonus == 0.0:
        return rewards
    env_infos = (
        reward_input.get("env_infos") if isinstance(reward_input, dict) else None
    )
    if not isinstance(env_infos, dict):
        return rewards

    def aligned(flags: torch.Tensor | None) -> torch.Tensor | None:
        if flags is None or flags.numel() != rewards.shape[0]:
            return None
        return flags.to(device=rewards.device)

    success = aligned(_success_flags(env_infos))
    final_success = aligned(_success_flags(env_infos.get("final_info")))
    if final_success is not None:
        boundary = reward_input.get("dones")
        if boundary is None:
            boundary = env_infos.get("_final_info")
        if boundary is None:
            if success is None:
                success = final_success
        else:
            boundary = torch.as_tensor(
                boundary, device=rewards.device, dtype=torch.bool
            )
            if boundary.ndim > 1:
                boundary = boundary.reshape(boundary.shape[0], -1).any(dim=-1)
            boundary = boundary.reshape(-1)
            valid_final = env_infos.get("_final_info")
            if valid_final is not None:
                valid_final = aligned(torch.as_tensor(valid_final).reshape(-1).bool())
                if valid_final is None:
                    boundary = torch.zeros_like(boundary)
                elif boundary.numel() == valid_final.numel():
                    boundary = boundary & valid_final
            if boundary.numel() == rewards.shape[0]:
                if success is None:
                    success = torch.zeros_like(final_success)
                success = torch.where(boundary, final_success, success)

    if success is None:
        return rewards
    bonus = success.to(dtype=rewards.dtype)
    return rewards + (bonus * gt_success_bonus).view(-1, *([1] * (rewards.dim() - 1)))
