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

"""A Qwen3.5 tokenizer stand-in: the model's chat template over a toy vocabulary."""

import re
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template

TEMPLATE = (Path(__file__).parent / "assets/qwen3_5_chat_template.jinja").read_text()
SPECIAL = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
# Special tokens are atomic and "ab" is one canonical token.
VOCABULARY = [*SPECIAL, "ab"]
_SPLIT = re.compile("(" + "|".join(map(re.escape, VOCABULARY)) + ")")


class Qwen35ToyTokenizer:
    """Renders with the Qwen3.5 chat template; other characters are one token each."""

    eos_token = "<|im_end|>"
    eos_token_id = VOCABULARY.index("<|im_end|>")
    unk_token_id = None
    chat_template = TEMPLATE

    def apply_chat_template(self, messages, tools=None, tokenize=False, **kwargs):
        assert not tokenize
        rendered, _ = render_jinja_template(
            [messages], tools=tools, chat_template=self.chat_template, **kwargs
        )
        return rendered[0]

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for part in _SPLIT.split(text):
            if part in VOCABULARY:
                ids.append(VOCABULARY.index(part))
            else:
                ids += [ord(char) + 100 for char in part]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(VOCABULARY[i] if i < 100 else chr(i - 100) for i in ids)

    def convert_tokens_to_ids(self, token):
        return VOCABULARY.index(token) if token in VOCABULARY else None


def qwen35_agentic_tokenizer(monkeypatch):
    """The Qwen3.5 AgenticTokenizer over the toy tokenizer, with thinking off.

    Reasoning markers come from sglang's registry, which these tests skip.
    """
    from rlinf.agents.tokenization import Qwen35AgenticTokenizer, get_agentic_tokenizer

    monkeypatch.setattr(Qwen35AgenticTokenizer, "reasoning_parser", None)
    return get_agentic_tokenizer(
        Qwen35ToyTokenizer(), "qwen3_5", {"enable_thinking": False}
    )
