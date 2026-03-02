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
from typing import Any, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from rlinf.data.datasets.item import DatasetItem
from rlinf.data.utils import batch_pad_to_fixed_len


class WideSeekR1_Dataset(Dataset):
    def __init__(
        self,
        data_paths: Union[str, list[str]],
        config: DictConfig,
        tokenizer: AutoTokenizer,
    ):
        super().__init__()
        self.data_paths = data_paths
        if isinstance(self.data_paths, str):
            self.data_paths = [self.data_paths]

        self.max_prompt_length = config.data.max_prompt_length
        self.tokenizer = tokenizer
        self.prompt_key = config.data.prompt_key
        self.answer_key = config.data.answer_key
        self.apply_chat_template = config.data.apply_chat_template
        self.data_size = config.data.get("data_size", None)
        self.is_markdown = config.data.get("is_markdown", False)
        self.unique_columns_key = config.data.get("unique_columns", "unique_columns")
        self.is_hybrid = config.data.get("is_hybrid", False)
        self.enable_zh = config.data.get("enable_zh", False)

        self.data = self._load_data()
        if self.data_size is not None and self.data_size >= 0:
            self.data = self.data[: self.data_size]
        if config.data.get("filter_prompt_by_length", False):
            total = len(self.data)
            filtered = []
            failed = 0
            for item in self.data:
                try:
                    prompt = item[self.prompt_key]
                    _, L = self.encode(prompt)
                    if L <= self.max_prompt_length:
                        filtered.append(item)
                    # breakpoint()
                except Exception as e:
                    logging.warning(
                        f"Failed to process item with error: {e}. Item: {item}"
                    )
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

    def _load_data(self) -> list[Any]:
        """
        Load and merge data from multiple files(json or jsonl).
        """
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

    def encode(self, text: str) -> tuple[list[int], int]:
        """
        Use tokenizer to encode the text and return the token ids and length.
        """
        text_ids = self.tokenizer.encode(text)
        return text_ids, len(text_ids)

    def __getitem__(self, idx):
        """
        Return a single prompt.
        """
        language = "en"
        if self.enable_zh:
            instance_id = self.data[idx].get("instance_id", "")
            if "zh" in str(instance_id) or self.data[idx].get("language", "en") == "zh":
                language = "zh"
        if not self.is_hybrid:
            prompt = self.data[idx][self.prompt_key]
            answer = self.data[idx][self.answer_key]

            if self.is_markdown:
                # Build answer dict from data
                answer_dict = {
                    "answer": answer,
                    "unique_columns": self.data[idx].get(self.unique_columns_key, []),
                    "is_markdown": self.is_markdown,
                    "instance_id": self.data[idx].get("instance_id", idx),
                    "language": language,
                }
                # Try to get evaluation info if available
                evaluation = self.data[idx].get("evaluation", None)
                if evaluation:
                    if isinstance(evaluation, str):
                        try:
                            evaluation = json.loads(evaluation)
                        except json.JSONDecodeError:
                            pass
                if isinstance(evaluation, dict):
                    answer_dict["required"] = evaluation.get("required", [])
                answer = answer_dict
            else:
                answer_dict = {
                    "answer": answer if isinstance(answer, list) else [answer],
                    "is_markdown": self.is_markdown,
                    "instance_id": self.data[idx].get("instance_id", idx),
                    "language": language,
                }
                answer = answer_dict
        else:
            prompt = self.data[idx][self.prompt_key]
            answer = self.data[idx][self.answer_key]
            is_markdown = self.data[idx].get("is_markdown", False)

            if is_markdown:
                # Build answer dict from data
                answer_dict = {
                    "answer": answer,
                    "unique_columns": self.data[idx].get(self.unique_columns_key, []),
                    "is_markdown": is_markdown,
                    "instance_id": self.data[idx].get("instance_id", idx),
                    "language": language,
                }
                # Try to get evaluation info if available
                evaluation = self.data[idx].get("evaluation", None)
                if evaluation:
                    if isinstance(evaluation, str):
                        try:
                            evaluation = json.loads(evaluation)
                        except json.JSONDecodeError:
                            pass
                if isinstance(evaluation, dict):
                    answer_dict["required"] = evaluation.get("required", [])
                answer = answer_dict
            else:
                answer_dict = {
                    "answer": answer if isinstance(answer, list) else [answer],
                    "is_markdown": is_markdown,
                    "instance_id": self.data[idx].get("instance_id", idx),
                    "language": language,
                }
                answer = answer_dict

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
        output = DatasetItem(
            prompt=prompt_tokens_tensor,
            length=prompt_length,
            answer=answer,
            idx=idx,
            image_data=[],
        )
        return output
