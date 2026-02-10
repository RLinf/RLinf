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

import logging
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from rlinf.data.datasets.item import DatasetItem
from rlinf.data.datasets.math import MathDataset
from rlinf.data.datasets.vlm import VLMDatasetRegistry


def create_rl_dataset(
    config: DictConfig, tokenizer: AutoTokenizer
) -> tuple[Dataset, Dataset]:
    """Create rl datasets.

    Arguments:
        config: The RLinf config.
        tokenizer (Tokenizer): The tokenizer.

    Returns:
        train_dataset (Dataset): The training dataset.

        val_dataset (Dataset): The validation dataset.
    """

    if config.data.type == "math":
        logging.info(f"Using dataset class: {MathDataset.__name__}")

        train_dataset = MathDataset(
            data_paths=config.data.train_data_paths,
            config=config,
            tokenizer=tokenizer,
        )

        val_dataset = MathDataset(
            data_paths=config.data.val_data_paths,
            config=config,
            tokenizer=tokenizer,
        )

        return train_dataset, val_dataset
    elif config.data.type == "vision_language":
        # Prefer new factory-based VLM datasets; fallback to legacy if requested
        dataset_name = getattr(config.data, "dataset_name", None)
        lazy_loading = bool(getattr(config.data, "lazy_loading", False))

        logging.info(
            f"Using VLM dataset: name={dataset_name}, lazy_loading={lazy_loading}"
        )

        train_dataset = VLMDatasetRegistry.create(
            dataset_name,
            data_paths=config.data.train_data_paths,
            config=config,
            tokenizer=tokenizer,
        )
        val_dataset = VLMDatasetRegistry.create(
            dataset_name,
            data_paths=config.data.val_data_paths,
            config=config,
            tokenizer=tokenizer,
        )
        return train_dataset, val_dataset
    elif config.data.type == "robot_demo":
        from rlinf.data.replay_buffer import SACReplayBuffer

        train_dataset = SACReplayBuffer.create_from_demo(config.data.path)
        return train_dataset, None
    else:
        raise NotImplementedError(
            f"Unsupported dataset type {config.data.type}, only support ['math', 'vision_language', 'robot_demo']"
        )


def collate_fn(data_list: list["DatasetItem"]) -> dict[str, Any]:
    """
    Collate function for batching dataset items.
    """
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

    batch: dict[str, Any] = {
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
    return batch


def sft_collate_fn(data_list: list["DatasetItem"]) -> dict[str, Any]:
    """
    Collate function for batching dataset items.
    """
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
        # pad prompts on the left
        padded_prompts = []
        for p in prompts:
            if p.numel() < target_len:
                pad = target_len - p.numel()
                p = torch.nn.functional.pad(p, (pad, 0), value=0)
            padded_prompts.append(p)
        prompts = padded_prompts

        # attention_mask is padded on the left
        # for example attention_mask:
        # [True, True, True, True, True, True, False, False]
        padded_attention = []
        for m in attention_masks:
            if m.numel() < target_len:
                pad = target_len - m.numel()
                m = torch.nn.functional.pad(m, (pad, 0), value=False)
            padded_attention.append(m)
        attention_masks = padded_attention

    # label_mask is padded on the left
    # for example label_mask:
    # [True, True, True, True, False, Fasle, True, True]
    # for the labels,will be set:
    # [-100, -100, -100, -100, -100, -100, -100, -100]
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
        for k in multi_modal_list[0].keys():
            vals = [m[k] for m in multi_modal_list]
            multi_modal_inputs[k] = (
                torch.cat(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals
            )

    batch: dict[str, Any] = {
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
        "multi_modal_inputs": multi_modal_inputs,
    }

    batch["attention_mask"] = torch.stack(attention_masks, dim=0)
    batch["label_mask"] = torch.stack(label_masks, dim=0)
    return batch
