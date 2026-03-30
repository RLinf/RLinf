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
from typing import Union

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from rlinf.data.datasets.item import DatasetItem
from rlinf.data.datasets.reasoning import ReasoningDataset
from rlinf.data.utils import batch_pad_to_fixed_len


class WideSeekR1Dataset(ReasoningDataset):
    def __init__(
        self,
        data_paths: Union[str, list[str]],
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(data_paths, config, tokenizer)
        self.is_markdown = config.data.get("is_markdown", False)
        self.unique_columns_key = config.data.get("unique_columns", "unique_columns")
        self.is_hybrid = config.data.get("is_hybrid", False)
        self.enable_zh = config.data.get("enable_zh", False)

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
            logging.warning(
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
