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

"""Unit tests for the agentic tokenizer.

No real model is loaded: a stub tokenizer maps a small vocab, and the
reasoning marker ids are set on the instance to bypass the sglang
registry lookup.
"""

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
# Group 4: TITO chat-template continuity (real tokenizer, optional)
# ---------------------------------------------------------------------------

# Incremental-turn token continuity against a whole-conversation render:
# render the first turn to the generation prompt, append the assistant
# message as the engine would emit it, then continue with
# build_native_tool_response. Skipped unless the model dir env var names a
# local checkout.
_TITO_CASES = [
    (
        "qwen",
        "RLINF_TEST_QWEN25_MODEL_DIR",
        "<|im_end|>",
    ),
    # A tool-calling GLM turn ends at the observation marker: the template
    # render stops at </tool_call> and the engine appends this token,
    # which build_native_tool_response then drops as the next boundary.
    (
        "glm47",
        "RLINF_TEST_GLM47_MODEL_DIR",
        "<|observation|>",
    ),
]


def _tito_messages(tool_call_parser: str) -> list[dict]:
    """A system / user / assistant(tool_calls) / tool conversation prefix.

    GLM's chat template iterates the tool-call arguments as a dict; Qwen
    accepts the OpenAI JSON-string form.
    """
    if tool_call_parser == "glm47":
        arguments = {"query": "reinforcement learning"}
        assistant = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "The user asks about RL; search for it.",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "search", "arguments": arguments},
                }
            ],
        }
    else:
        assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "reinforcement learning"}',
                    },
                }
            ],
        }
    return [
        {"role": "system", "content": "You answer with a search tool."},
        {"role": "user", "content": "What is reinforcement learning?"},
        assistant,
    ]


def _tito_streams(agentic, tokenizer_type, tool_messages):
    """The first-turn prompt, the engine's turn output, and the whole render.

    The engine's output for the assistant message is the template's own
    rendering of that message, minus the generation prompt that preceded it
    and any trailing newline the template adds after the end token; what the
    caller adds to or removes from its tail is what the engine would have
    emitted (or not) at the end of the turn.
    """
    messages = _tito_messages(tokenizer_type)
    first_prompt = agentic.apply_chat_template(
        messages[:2], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    whole = agentic.apply_chat_template(
        [*messages, *tool_messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    after_assistant = agentic.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_dict=False
    )
    engine_output = list(after_assistant[len(first_prompt) :])
    assert engine_output, "the assistant message rendered to no tokens"
    if agentic.tokenizer.decode(engine_output[-1:]) == "\n":
        engine_output = engine_output[:-1]
    return first_prompt, engine_output, whole


def _load_tito_agentic(env_var, tokenizer_type):
    import os

    model_dir = os.environ.get(env_var)
    if not model_dir:
        pytest.skip(f"no local model dir; set {env_var} to run this test")
    if not os.path.isdir(model_dir):
        # A set variable that names no directory is a configuration error:
        # skipping here would let these tests never run unnoticed.
        pytest.fail(f"{env_var} is set to {model_dir!r}, which is not a directory")

    from transformers import AutoTokenizer

    return get_agentic_tokenizer(
        AutoTokenizer.from_pretrained(model_dir), tokenizer_type=tokenizer_type
    )


@pytest.mark.parametrize(
    "tokenizer_type,env_var,end_token", _TITO_CASES, ids=["qwen", "glm47"]
)
def test_incremental_turns_match_whole_conversation_render(
    tokenizer_type, env_var, end_token
):
    agentic = _load_tito_agentic(env_var, tokenizer_type)
    tool_messages = [
        {"role": "tool", "name": "search", "content": "Reinforcement learning is ..."}
    ]
    first_prompt, engine_output, whole = _tito_streams(
        agentic, tokenizer_type, tool_messages
    )
    # The engine's own turn output: the template's assistant render plus
    # the marker the engine emits when the turn ends. Qwen's template
    # already ends at its end-of-turn token; GLM's stops at </tool_call>
    # and the engine appends the observation marker.
    if agentic.tokenizer.decode(engine_output[-1:]) != end_token:
        engine_output.append(agentic.tokenizer.convert_tokens_to_ids(end_token))
    assert agentic.tokenizer.decode(engine_output[-1:]) == end_token

    incremental = agentic.build_native_tool_response(
        list(first_prompt) + engine_output, tool_messages
    )
    assert incremental == whole, (
        f"{tokenizer_type}: incremental turn build diverges from the whole "
        f"conversation render at token "
        f"{next(i for i, (a, b) in enumerate(zip(incremental, whole)) if a != b)}"
        if incremental[: len(whole)] != whole
        else f"{tokenizer_type}: incremental build is longer than the whole render"
    )


@pytest.mark.parametrize(
    "tokenizer_type,env_var,end_token", _TITO_CASES, ids=["qwen", "glm47"]
)
def test_truncated_turn_continues_to_the_whole_render(
    tokenizer_type, env_var, end_token
):
    """A turn cut at max_new_tokens emits no end-of-turn marker, and the tool
    message that follows still has to continue the conversation token for
    token: the boundary the template renders before it must not be lost."""
    agentic = _load_tito_agentic(env_var, tokenizer_type)
    tool_messages = [
        {"role": "tool", "name": "search", "content": "Reinforcement learning is ..."}
    ]
    first_prompt, engine_output, whole = _tito_streams(
        agentic, tokenizer_type, tool_messages
    )
    if agentic.tokenizer.decode(engine_output[-1:]) == end_token:
        engine_output = engine_output[:-1]
    assert agentic.tokenizer.decode(engine_output[-1:]) != end_token

    incremental = agentic.build_native_tool_response(
        list(first_prompt) + engine_output, tool_messages
    )
    assert incremental == whole, (
        f"{tokenizer_type}: the tool turn after a truncated turn diverges "
        f"from the whole conversation render at token "
        f"{next(i for i, (a, b) in enumerate(zip(incremental, whole)) if a != b)}"
        if incremental[: len(whole)] != whole
        else f"{tokenizer_type}: incremental build is longer than the whole render"
    )
