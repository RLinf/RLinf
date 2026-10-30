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

"""Unit tests for the agentic tokenizer and the Qwen3.5 tool-call parser.

No real model is loaded: a stub tokenizer maps a small vocab, and the
reasoning marker ids are set on the instance to bypass the sglang
registry lookup.
"""

import asyncio
import json

import pytest

from rlinf.agents.tokenization.agentic_tokenizer import (
    AgenticTokenizer,
    get_agentic_tokenizer,
)


class _StubTokenizer:
    """Minimal id<->text mapping: no model files, no sglang, no network."""

    def __init__(self, vocab: dict[str, int]):
        self._s2i = dict(vocab)
        self._i2s = {i: s for s, i in self._s2i.items()}
        self.unk_token_id = None
        self.pad_token_id = 0
        self.eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._i2s[i] for i in ids)

    def convert_tokens_to_ids(self, token):
        return self._s2i.get(token)


# Reasoning marker ids (arbitrary ids present in the stub vocab so that
# decode() never hits an unknown id).
THINK_START, THINK_END = 100, 200

# Stub vocab shared by the truncation/offset tests.
VOCAB = {
    "a": 1,
    "b": 2,
    "c": 3,
    "</search>": 4,
    "<search>": 5,
    "今": 6,
    "天": 7,
    "😀": 8,
    "</sea": 9,
    "rch>": 10,
    "<t>": THINK_START,
    "</t>": THINK_END,
}


def _make_tokenizer(vocab=None, start_id=THINK_START, end_id=THINK_END):
    """Base AgenticTokenizer on a stub, with markers assigned directly."""
    tok = AgenticTokenizer(_StubTokenizer(vocab if vocab is not None else VOCAB))
    tok.reasoning_start_token_id = start_id
    tok.reasoning_end_token_id = end_id
    return tok


# ---------------------------------------------------------------------------
# Group 1: truncate_after_stop
# ---------------------------------------------------------------------------


def test_truncate_after_stop_hit_start_zero():
    tok = _make_tokenizer()
    ids = [1, 2, 4, 3]  # a b </search> c
    result, matched = tok.truncate_after_stop(ids, "</search>", start=0)
    assert matched is True
    assert result == ids[:3]
    assert "</search>" in tok.tokenizer.decode(result[0:])


def test_truncate_after_stop_no_match():
    tok = _make_tokenizer()
    ids = [1, 2, 3]
    result, matched = tok.truncate_after_stop(ids, "</search>", start=0)
    assert matched is False
    assert result == ids


def test_truncate_after_stop_stop_only_in_reasoning():
    # The stop string appears before `start` (in the reasoning block) only:
    # the content region must be left untouched.
    tok = _make_tokenizer()
    ids = [4, 1, 2]  # </search> a b
    result, matched = tok.truncate_after_stop(ids, "</search>", start=1)
    assert matched is False
    assert result == ids


def test_truncate_after_stop_multibyte_tokens():
    tok = _make_tokenizer()
    ids = [6, 7, 4, 8]  # 今 天 </search> 😀
    result, matched = tok.truncate_after_stop(ids, "</search>", start=0)
    assert matched is True
    assert result == ids[:3]


def test_truncate_after_stop_stop_across_tokens():
    # The stop string is assembled from two separate tokens.
    tok = _make_tokenizer()
    ids = [1, 9, 10, 3]  # a </sea rch> c
    result, matched = tok.truncate_after_stop(ids, "</search>", start=0)
    assert matched is True
    assert result == ids[:3]


def test_truncate_after_stop_stop_at_last_token():
    tok = _make_tokenizer()
    ids = [1, 2, 4]  # a b </search>
    result, matched = tok.truncate_after_stop(ids, "</search>", start=0)
    assert matched is True
    assert len(result) == len(ids)


def test_truncate_after_stop_hit_start_positive():
    tok = _make_tokenizer()
    ids = [4, 1, 4, 2]  # </search> a </search> b, content starts at 2
    result, matched = tok.truncate_after_stop(ids, "</search>", start=2)
    assert matched is True
    assert result == ids[:3]


# ---------------------------------------------------------------------------
# Group 2: content_start_offset
# ---------------------------------------------------------------------------


def test_offset_prompt_ends_with_opener():
    # Turn 1 with thinking on: the template's generation prompt ends with
    # the opener; the FIRST closer in the response ends the block.
    tok = _make_tokenizer()
    prompt = [1, THINK_START]
    resp = [1, 2, THINK_END, 3, THINK_END, 4]
    assert tok.content_start_offset(resp, prompt_ids=prompt) == 3


def test_offset_prompt_ends_with_closer():
    # enable_thinking=False: the prompt ends with a closer, the response
    # contains no opener -> no reasoning block to skip.
    tok = _make_tokenizer()
    prompt = [1, THINK_END]
    resp = [1, 2, 3]
    assert tok.content_start_offset(resp, prompt_ids=prompt) == 0


def test_offset_response_starts_with_opener():
    # Qwen3-style template: thinking on but the opener is generated by the
    # model at the start of the response.
    tok = _make_tokenizer()
    prompt = [1, 2, 3]  # no markers at all
    resp = [THINK_START, 1, THINK_END, 2]
    assert tok.content_start_offset(resp, prompt_ids=prompt) == 3


def test_offset_opener_mid_content():
    # An opener in the middle of the content is content, not a block start.
    tok = _make_tokenizer()
    prompt = [1, 2, 3]
    resp = [1, THINK_START, THINK_END, 2]
    assert tok.content_start_offset(resp, prompt_ids=prompt) == 0


def test_offset_turn_two_shape():
    # Turn 2+: the prompt tail is the raw tool text appended after the
    # previous turn's closer; no new block is opened.
    tok = _make_tokenizer()
    prompt = [THINK_END, 1, 2]
    resp = [1, 2]
    assert tok.content_start_offset(resp, prompt_ids=prompt) == 0


def test_offset_trailing_closer_keeps_truncation():
    # Regression: with thinking on, a trailing closer after the search must
    # not move content_start past the search and disable truncation.
    tok = _make_tokenizer()
    prompt = [1, THINK_START]
    # block, <search> </search>, trailing closer, b
    resp = [THINK_END, 5, 4, THINK_END, 2]
    offset = tok.content_start_offset(resp, prompt_ids=prompt)
    assert offset == 1
    result, matched = tok.truncate_after_stop(resp, "</search>", start=offset)
    assert matched is True
    assert result == resp[:3]  # cut after the real </search>, trailing junk dropped


def test_offset_search_in_reasoning_and_content():
    # A search tag inside the reasoning block must not be the cut point;
    # only the one in the content region counts.
    tok = _make_tokenizer()
    prompt = [1, THINK_START]
    # reasoning: <search>a</search>, content: <search>b</search>c
    resp = [5, 1, 4, THINK_END, 5, 2, 4, 3]
    offset = tok.content_start_offset(resp, prompt_ids=prompt)
    assert offset == 4
    result, matched = tok.truncate_after_stop(resp, "</search>", start=offset)
    assert matched is True
    assert result == resp[:7]  # cut after b's </search> (index 6), not a's (index 2)


def test_offset_no_closer_in_vocab():
    # Model without single-token reasoning markers (e.g. Qwen2.5): offset 0.
    tok = _make_tokenizer(start_id=None, end_id=None)
    assert tok.content_start_offset([1, 2, 3], prompt_ids=[1, THINK_START]) == 0


# ---------------------------------------------------------------------------
# Group 3: factory error path
# ---------------------------------------------------------------------------


def test_get_agentic_tokenizer_unknown_type():
    with pytest.raises(ValueError) as excinfo:
        get_agentic_tokenizer(_StubTokenizer(VOCAB), tokenizer_type="nope")
    msg = str(excinfo.value)
    assert "nope" in msg
    for valid in ("default", "qwen", "glm47"):
        assert valid in msg


# ---------------------------------------------------------------------------
# Qwen3.5: increments after an assistant turn
# ---------------------------------------------------------------------------


@pytest.fixture
def qwen35(monkeypatch):
    from _qwen35_chat import qwen35_agentic_tokenizer

    return qwen35_agentic_tokenizer(monkeypatch)


@pytest.mark.parametrize(
    "new_messages",
    [
        [
            {"role": "tool", "tool_call_id": "c", "content": "seen"},
            {"role": "user", "content": "observation"},
        ],
        [{"role": "user", "content": "next"}],
        [
            {"role": "tool", "tool_call_id": "c", "content": "a"},
            {"role": "tool", "tool_call_id": "d", "content": "b"},
        ],
    ],
)
@pytest.mark.parametrize("stopped_on_eos", [True, False])
def test_qwen35_appends_new_messages_as_the_template_renders_them(
    qwen35, new_messages, stopped_on_eos
):
    conversation = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "done"},
        *new_messages,
    ]
    full = qwen35.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    # The template ends the reply with "<|im_end|>\n" before the new messages.
    after_reply = full.split("done<|im_end|>", 1)[1]
    sampled = "<|im_start|>assistant\n<think>\n\n</think>\n\ndone"
    prefix_ids = qwen35.tokenizer(sampled + "<|im_end|>" * stopped_on_eos)["input_ids"]
    ids = qwen35.build_native_tool_response(prefix_ids, new_messages)
    assert ids[: len(prefix_ids)] == prefix_ids
    assert qwen35.tokenizer.decode(ids) == sampled + "<|im_end|>" + after_reply


def test_qwen35_tool_calls_round_trip_through_the_chat_template():
    from _qwen35_chat import Qwen35ToyTokenizer

    from rlinf.algorithms.registry import get_toolcall_parser

    properties = {
        "text": {"type": "string"},
        "enabled": {"type": "boolean"},
        "count": {"type": "integer"},
        "position": {"type": "array", "items": {"type": "number"}},
        "optional": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "move",
                "parameters": {"type": "object", "properties": properties},
            },
        }
    ]
    arguments = [
        {
            "text": "向前 < 2米\n\nthen stop",
            "enabled": True,
            "count": 2,
            "position": [1, -2.5],
            "optional": None,
        },
        {"text": "null", "enabled": False, "count": 0, "position": [], "optional": 1.5},
    ]
    reply = {
        "role": "assistant",
        "content": "Check position.",
        "tool_calls": [
            {"type": "function", "function": {"name": "move", "arguments": args}}
            for args in arguments
        ],
    }
    rendered = Qwen35ToyTokenizer().apply_chat_template(
        [{"role": "user", "content": "Move."}, reply], tools=tools
    )
    response = rendered.split("<|im_start|>assistant\n", 1)[1]
    response = response.split("</think>\n\n")[-1].removesuffix("<|im_end|>\n")
    text, calls = asyncio.run(get_toolcall_parser("qwen3.5")(response, tools=tools))
    assert "Check position." in text
    assert [json.loads(call.arguments) for call in calls] == arguments


@pytest.mark.parametrize(
    "body",
    [
        "<parameter=x>1",
        "<parameter=x>1</parameter><parameter=x>2</parameter>",
        "unparsed text<parameter=x>1</parameter>",
    ],
)
def test_malformed_qwen35_calls_stay_visible_as_text(body):
    from rlinf.algorithms.registry import get_toolcall_parser

    response = f"<tool_call><function=move>{body}</function></tool_call>"
    text, calls = asyncio.run(get_toolcall_parser("qwen3.5")(response))
    assert text == response and calls == []


def test_invalid_qwen35_arguments_are_not_coerced():
    from rlinf.algorithms.registry import get_toolcall_parser

    properties = {"enabled": {"type": "boolean"}, "position": {"type": "array"}}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "move",
                "parameters": {"type": "object", "properties": properties},
            },
        }
    ]
    response = (
        "<tool_call><function=move><parameter=enabled>maybe</parameter>"
        "<parameter=position>[NaN]</parameter></function></tool_call>"
    )
    _, calls = asyncio.run(get_toolcall_parser("qwen3.5")(response, tools=tools))
    assert json.loads(calls[0].arguments) == {"enabled": "maybe", "position": "[NaN]"}
