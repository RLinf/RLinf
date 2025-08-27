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

import json
import logging
import os
from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


def batch_pad_to_fixed_len(
    batch: List[torch.Tensor],
    max_batch_len: int,
    pad_token: int,
    left_pad: bool = False,
) -> torch.Tensor:
    if left_pad:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),  # pad on the left
                        seq,
                    ]
                )
                for seq in batch
            ]
        )
    else:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        seq,
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),
                    ]
                )
                for seq in batch
            ]
        )
    return batch_pad


class MathDataset(Dataset):
    def __init__(self, data_paths, config, tokenizer):
        super().__init__()
        self.data_paths = data_paths
        if isinstance(self.data_paths, str):
            self.data_paths = [self.data_paths]

        self.max_prompt_length = config.max_prompt_length
        self.tokenizer = tokenizer
        self.prompt_key = config.prompt_key

        self.data = self._load_data()
        if config.get("filter_prompt_by_length", False):
            total = len(self.data)
            filtered = []
            failed = 0

            for item in self.data:
                try:
                    _, L = self.encode(item[self.prompt_key])
                    if L <= self.max_prompt_length:
                        filtered.append(item)
                except Exception:
                    failed += 1

            self.data = filtered
            assert len(self.data) > 0, (
                f"No samples found within max_prompt_length={self.max_prompt_length}. "
                "Please check your dataset or increase max_prompt_length."
            )

            if failed > 0:
                logging.warning(
                    f"{failed} samples were skipped due to format issues "
                    f"(kept {len(self.data)} / {total})."
                )

    def _load_data(self):
        merged_data = []

        for path in self.data_paths:
            _, file_extension = os.path.splitext(path)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    if file_extension == ".jsonl":
                        merged_data.extend([json.loads(line.strip()) for line in file])
                    elif file_extension == ".json":
                        content = json.load(file)
                        if isinstance(content, list):
                            merged_data.extend(content)
                        else:
                            merged_data.append(content)
                    else:
                        print(f"Unsupport {file_extension}, skip: {path}")
            except Exception:
                raise RuntimeError("Load data error")

        return merged_data

    def __len__(self):
        return len(self.data)

    def encode(self, text):
        text_ids = self.tokenizer.encode(text)
        return text_ids, len(text_ids)

    def __getitem__(self, idx):
        """
        Return a single prompt.
        """

        prompt = self.data[idx][self.prompt_key]

        answer = self.data[idx]["solutions"]

        prompt_tokens, prompt_length = self.encode(prompt)
        prompt_tokens_tensor = torch.as_tensor(prompt_tokens, dtype=torch.int64)

        if prompt_length > self.max_prompt_length:
            print(
                f"prompt_tokens_tensor length {prompt_length} exceeds the max_prompt_length {self.max_prompt_length}",
            )
            prompt_tokens_tensor = prompt_tokens_tensor[: self.max_prompt_length]
            prompt_length = self.max_prompt_length

        prompt_tokens_tensor = batch_pad_to_fixed_len(
            [prompt_tokens_tensor],
            self.max_prompt_length,
            self.tokenizer.eos_token_id,
            left_pad=True,
        )[0]

        output = {
            "prompt": prompt_tokens_tensor,
            "length": prompt_length,
            "answer": answer,
            "idx": idx,
        }
        return output


def create_rl_dataset(data_config, tokenizer):
    """Create rl datasets.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.

    Returns:
        train_dataset (Dataset): The training dataset.

        val_dataset (Dataset): The validation dataset.
    """

    if data_config.type == "math":
        dataset_cls = MathDataset
    else:
        return None, None

    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    train_dataset = dataset_cls(
        data_paths=data_config.train_data_paths,
        config=data_config,
        tokenizer=tokenizer,
    )

    val_dataset = dataset_cls(
        data_paths=data_config.val_data_paths,
        config=data_config,
        tokenizer=tokenizer,
    )

    return train_dataset, val_dataset


def collate_fn(data_list: list[dict]) -> dict:
    r"""
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}
