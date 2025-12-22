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
from typing import Any, List, Tuple, Union, Dict

import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from rlinf.data.datasets.item import DatasetItem
from rlinf.data.datasets.utils import batch_pad_to_fixed_len

def get_tool_schemas():
    """
    从配置文件获取工具模式列表
    
    Args:
        tools_config_file: 工具配置文件路径
        
    Returns:
        List[Dict[str, Any]]: 工具模式列表
    """
    # logger.info(f"Loading tool schemas from config file: {tools_config_file}")
    
    return [{
        "type": "function",
        "function": {
            "name": "python_code_with_standard_io", 
            "description": "Execute Python code with standard input and capture standard output. This function takes a Python code string and an input string, provides the input string through standard input (stdin) to the code, and captures and returns any output produced through standard output (stdout). If the executed code raises an exception, the error message will be captured and returned instead.", 
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string", 
                        "description": "A string containing Python code to be executed. The code can read from standard input using the input() function."
                    }, 
                    "input": {
                        "type": "string", 
                        "description": "A string that will be provided as standard input to the code when it calls input()."
                    },
                }, 
                "required": ["code", "input"],
            }
        },
    }]

class Rstar2Dataset(Dataset):
    def __init__(
        self,
        data_paths: Union[str, List[str]],
        config: DictConfig,
        tokenizer: AutoTokenizer,
    ):
        """
        Initialize the MathDataset.

        This constructor loads the dataset from specified paths and applies optional filtering
        based on prompt length constraints.

        This method handles prompt formatting based on the `apply_chat_template` configuration:

        **When apply_chat_template = False:**
            - The prompt is used directly from the dataset without modification.
            - Note: Ensure the prompt format is correct and check if the dataset already
                contains tokenizer-specific special characters.
            - Expected dataset format:
                {
                    "prompt_key": <str>,
                    "answer_key": <str>,
                }

        **When apply_chat_template = True:**
            - The raw dataset is processed using tokenizer.apply_chat_template() to format
                the prompt according to the model's chat template.
            - Expected dataset format:
                {
                    "prompt_key": [{"content": <str>, "role": <str>}, ...],
                    "answer_key": <str>,
                }
            - After processing, the data is transformed to:
                {
                    "prompt_key": <str>,
                    "answer_key": <str>,
                }
        """
        super().__init__()
        self.data_paths = data_paths
        if isinstance(self.data_paths, str):
            self.data_paths = [self.data_paths]

        self.max_prompt_length = config.data.max_prompt_length
        self.tokenizer = tokenizer
        self.prompt_key = config.data.prompt_key
        self.answer_key = config.data.answer_key
        self.apply_chat_template = config.data.apply_chat_template
        if config.rollout.get("custom_chat_template", None) is not None:
            self.tokenizer.chat_template = config.rollout.custom_chat_template
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        self.tool_schemas = get_tool_schemas()

        self.data = self._load_data()
        if config.data.get("filter_prompt_by_length", False):
            total = len(self.data)
            filtered = []
            failed = 0

            for item in self.data:
                try:
                    prompt = item[self.prompt_key]
                    if self.apply_chat_template:
                        prompt = self.tokenizer.apply_chat_template(
                            prompt,
                            tools=self.tool_schemas,
                            add_generation_prompt=True,
                            tokenize=False,
                            **self.apply_chat_template_kwargs,
                        )
                        # save the convert data
                        item[self.prompt_key] = prompt
                    _, L = self.encode(prompt)
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

    def _load_data(self) -> List[Any]:
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

    def encode(self, text: str) -> Tuple[List[int], int]:
        """
        Use tokenizer to encode the text and return the token ids and length.
        """
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return text_ids, len(text_ids)

    def __getitem__(self, idx):
        """
        Return a single prompt.
        """

        prompt = self.data[idx][self.prompt_key]
        answer = self.data[idx][self.answer_key]

        # if answer is a string, convert it to a list
        if isinstance(answer, str):
            answer = [answer]

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
