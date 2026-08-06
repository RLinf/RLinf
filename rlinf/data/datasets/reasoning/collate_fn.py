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

"""Reasoning dataset collate function for RL training batches."""

from __future__ import annotations

from typing import Any

import torch


def collate_fn(data_list: list[Any]) -> dict[str, Any]:
    """Collate reasoning samples into a rollout-ready batch dict."""
    prompts = []
    lens = []
    for it in data_list:
        p = (
            it.prompt
            if isinstance(it.prompt, torch.Tensor)
            else torch.as_tensor(it.prompt, dtype=torch.long)
        )
        if p.dim() == 2 and p.size(0) == 1:
            p = p.squeeze(0)
        assert p.dim() == 1, (
            f"DatasetItem.prompt must be 1-D tensor, current shape is: {p.shape}"
        )
        prompts.append(p)
        lens.append(p.numel())

    if len(set(lens)) == 1:
        target_len = lens[0]
    else:
        target_len = min(lens)
        prompts = [p[-target_len:] if p.numel() > target_len else p for p in prompts]

    batch_prompt = torch.stack(prompts, dim=0)  # [B, L]
    batch_length = torch.tensor(
        [min(int(it.length), target_len) for it in data_list], dtype=torch.long
    )
    batch_idx = torch.tensor([int(it.idx) for it in data_list], dtype=torch.long)

    return {
        "prompt": batch_prompt,  # [B, L]
        "length": batch_length,  # [B]
        "answer": [it.answer for it in data_list],  # List[str]
        "idx": batch_idx,  # [B]
        "solution": [it.solution for it in data_list],  # List[Optional[str]]
        "image_data": [
            it.image_data for it in data_list
        ],  # List[Optional[List[bytes|str]]]
        "prompt_text": [it.prompt_text for it in data_list],  # List[Optional[str]]
        "meta": [it.meta for it in data_list],  # List[Optional[dict]]
        "multi_modal_inputs": [
            it.multi_modal_inputs for it in data_list
        ],  # List[Optional[dict]]
    }


__all__ = ["collate_fn"]
