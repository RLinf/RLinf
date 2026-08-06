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

"""VLM SFT collate function preserving multimodal fields."""

from __future__ import annotations

from typing import Any

import torch


def collate_fn(data_list: list[Any]) -> dict[str, Any]:
    """Collate VLM SFT samples and preserve multi-modal fields."""
    prompts = []
    lens = []
    attention_masks = []
    label_masks = []

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

        am = getattr(it, "attention_mask", None)
        am = (
            am
            if isinstance(am, torch.Tensor)
            else torch.as_tensor(am, dtype=torch.long)
        )
        if am.dim() == 2 and am.size(0) == 1:
            am = am.squeeze(0)
        attention_masks.append(am)

        lm = getattr(it, "label_mask", None)
        lm = (
            lm
            if isinstance(lm, torch.Tensor)
            else torch.as_tensor(lm, dtype=torch.bool)
        )
        if lm.dim() == 2 and lm.size(0) == 1:
            lm = lm.squeeze(0)
        label_masks.append(lm)

    if len(set(lens)) == 1:
        target_len = lens[0]
    else:
        target_len = max(lens)
        padded_prompts = []
        for p in prompts:
            if p.numel() < target_len:
                pad = target_len - p.numel()
                p = torch.nn.functional.pad(p, (pad, 0), value=0)
            padded_prompts.append(p)
        prompts = padded_prompts

        padded_attention = []
        for m in attention_masks:
            if m.numel() < target_len:
                pad = target_len - m.numel()
                m = torch.nn.functional.pad(m, (pad, 0), value=False)
            padded_attention.append(m)
        attention_masks = padded_attention

    padded_label = []
    for m, prompt_len in zip(label_masks, lens):
        if m.numel() < prompt_len:
            pad = prompt_len - m.numel()
            m = torch.nn.functional.pad(m, (0, pad), value=False)
        if m.numel() < target_len:
            pad = target_len - m.numel()
            m = torch.nn.functional.pad(m, (pad, 0), value=False)
        padded_label.append(m)
    label_masks = padded_label

    batch_prompt = torch.stack(prompts, dim=0)  # [B, L]
    batch_length = torch.tensor(
        [min(int(it.length), target_len) for it in data_list], dtype=torch.long
    )
    batch_idx = torch.tensor([int(it.idx) for it in data_list], dtype=torch.long)

    multi_modal_list = [
        it.multi_modal_inputs for it in data_list if it.multi_modal_inputs is not None
    ]
    multi_modal_inputs = {}
    if multi_modal_list:
        for key in multi_modal_list[0].keys():
            vals = [m[key] for m in multi_modal_list]
            if key == "pixel_values":
                multi_modal_inputs[key] = vals
            elif key == "image_grid_thw":
                multi_modal_inputs[key] = (
                    torch.cat(vals, dim=0)
                    if isinstance(vals[0], torch.Tensor)
                    else vals
                )
            else:
                raise ValueError(f"Unsupported multi_modal_input key: {key}")

    batch: dict[str, Any] = {
        "prompt": batch_prompt,
        "length": batch_length,
        "answer": [it.answer for it in data_list],
        "idx": batch_idx,
        "solution": [it.solution for it in data_list],
        "image_data": [it.image_data for it in data_list],
        "prompt_text": [it.prompt_text for it in data_list],
        "meta": [it.meta for it in data_list],
        "multi_modal_inputs": multi_modal_inputs,
    }
    batch["attention_mask"] = torch.stack(attention_masks, dim=0)
    batch["label_mask"] = torch.stack(label_masks, dim=0)
    return batch


__all__ = ["collate_fn"]
