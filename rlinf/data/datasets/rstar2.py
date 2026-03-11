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

from typing import Union

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from rlinf.data.datasets.reasoning import ReasoningDataset


def get_tool_schemas():
    """
    Load tool schemas from a configuration file.

    Returns:
        List[Dict[str, Any]]: List of tool schema dictionaries.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "python_code_with_standard_io",
                "description": "Execute Python code with standard input and capture standard output. This function takes a Python code string and an input string, provides the input string through standard input (stdin) to the code, and captures and returns any output produced through standard output (stdout). If the executed code raises an exception, the error message will be captured and returned instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "A string containing Python code to be executed. The code can read from standard input using the input() function.",
                        },
                        "input": {
                            "type": "string",
                            "description": "A string that will be provided as standard input to the code when it calls input().",
                        },
                    },
                    "required": ["code", "input"],
                },
            },
        }
    ]


class Rstar2Dataset(ReasoningDataset):
    def __init__(
        self,
        data_paths: Union[str, list[str]],
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        if config.rollout.get("custom_chat_template", None) is not None:
            self.chat_template = config.rollout.custom_chat_template
        self.apply_chat_template_kwargs = config.data.get(
            "apply_chat_template_kwargs", {}
        )
        self.tool_schemas = get_tool_schemas()
        super().__init__(data_paths, config, tokenizer)

    def apply_chat_template(self, texts: list[str]) -> list[str]:
        """
        Use tokenizer to apply chat template to the texts.
        """
        if self.chat_template is not None:
            self.tokenizer.chat_template = self.chat_template
            self.chat_template = None
        prompts = self.tokenizer.apply_chat_template(
            texts,
            tools=self.tool_schemas,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )
        return prompts

    def encode(self, text: str) -> tuple[list[int], int]:
        """
        Use tokenizer to encode the text and return the token ids and length.
        """
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return text_ids, len(text_ids)

    def encode_batch(self, texts: list[str]) -> list[int]:
        """
        Use tokenizer to encode the texts and return the token ids and length.
        """
        text_ids = self.tokenizer.batch_encode_plus(
            list(texts), add_special_tokens=False
        )["input_ids"]
        return text_ids
