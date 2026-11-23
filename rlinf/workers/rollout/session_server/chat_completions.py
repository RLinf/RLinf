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

"""The OpenAI chat-completions format: messages to token ids and back.

The chat route lives on
:class:`~rlinf.workers.rollout.session_server.server_worker.SessionServerWorker`,
which owns the session state, the metrics and the error shapes. This module
owns only the OpenAI side of that route: rendering messages with the model
family's tokenizer, reusing the engine output of the previous turn where the
conversation continues (token-in, token-out, TITO), and parsing the engine's
text back into an assistant message and a ``chat.completion`` body.
"""

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

from rlinf.agents.tokenization.agentic_tokenizer import AgenticTokenizer


class ChatCheckpoint(NamedTuple):
    """The state a turn started from, for a retried request to render from.

    Restoring a rollback means restoring all four at once, and the rollback
    that uses it consumes it: a conversation's checkpoint is None once one
    rollback has run.
    """

    messages: list[dict]
    last_input: Optional[list[int]]
    last_output: Optional[list[int]]
    tools: Optional[list]


@dataclass
class ChatConversation:
    """The server-side copy of one session's chat history.

    ``messages`` is the conversation as the server knows it, the assistant
    messages it parsed itself included. ``tools`` is the tool list the
    stored tokens were rendered under. ``checkpoint`` is the state the last
    turn started from, kept so a retried request renders from before that
    turn; it is None once a rollback has consumed it. The three are set
    together once a turn has run.
    """

    messages: list[dict]
    tools: Optional[list]
    checkpoint: Optional[ChatCheckpoint]


@dataclass
class RenderedInput:
    """One rendered request: its token ids and the history to store.

    ``messages`` is the conversation as it stands after this request's
    messages, without the answer: the stored history, plus what this request
    added to it. The server keeps its own copy of the earlier turns, so a
    client that drops a field it sent before (reasoning_content, say) does
    not erase it from the history.
    """

    input_ids: list[int]
    messages: list[dict]
    # The request repeats the last one, so the worker drops that turn.
    rollback: bool = False


@dataclass
class ParsedMessage:
    """What one engine output parses into."""

    message: dict
    has_reasoning: bool = False
    has_tool_calls: bool = False
    tool_call_parse_failed: bool = False


class ChatCompletions:
    """The OpenAI format of one model family: messages to tokens and back.

    Built once per session server. Every method is a pure conversion — the
    caller passes the conversation in and gets ids, a message, or a response
    body out — so the session's lifecycle stays with the worker.
    """

    def __init__(self, agentic_tokenizer: AgenticTokenizer, model_name: str):
        """
        Args:
            agentic_tokenizer: The model family's tokenizer, already
                configured with the chat template arguments.
            model_name: Reported as the response's model when the request
                names none.
        """
        self._tokenizer = agentic_tokenizer
        self._model_name = model_name

    def check_request(self, body: dict, sampling_params: dict) -> Optional[str]:
        """The request's contract with this entry, or why it is refused.

        Fields the entry ignores (the model is fixed by config, training
        logprobs ride the payload, ``seed`` has no engine support) are
        simply never read. A field sent as ``null`` counts as unset, as in
        the OpenAI API, so clients that send every optional parameter as
        ``null`` still get the configured sampling.

        Args:
            body: The request body.
            sampling_params: The sampling parameters this server would
                forward the request with.

        Returns:
            The reason to refuse the request with, or None when it is
            acceptable.
        """
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return "messages must be a non-empty list"
        if (body.get("n") or 1) > 1:
            return (
                "n > 1 is not supported: the trajectories of a group are "
                "separate sessions"
            )
        for field in ("temperature", "top_p", "top_k"):
            value = body.get(field)
            if value is not None and value != sampling_params.get(field):
                return (
                    f"{field} must equal the configured value "
                    f"({sampling_params.get(field)!r}); training logprobs are "
                    f"computed under the configured sampling"
                )
        tool_choice = body.get("tool_choice") or "auto"
        if tool_choice not in ("auto", "none"):
            return (
                f"tool_choice {tool_choice!r} is not supported; only 'auto' and 'none'"
            )
        if body.get("response_format") is not None:
            return "response_format is not supported by this endpoint"
        if body.get("chat_template_kwargs") not in (
            None,
            self._tokenizer.chat_template_kwargs,
        ):
            return (
                f"chat_template_kwargs must match the configured "
                f"{dict(self._tokenizer.chat_template_kwargs)!r}"
            )
        return None

    @staticmethod
    def sampling_overrides(body: dict) -> dict:
        """Map the request's token-limit and stop fields onto sampling.

        A field sent as ``null`` is unset, as in check_request: a null
        ``max_completion_tokens`` does not hide the ``max_tokens`` next to it.
        """
        overrides: dict = {}
        for field in ("max_completion_tokens", "max_tokens"):
            value = body.get(field)
            if value is not None:
                overrides["max_new_tokens"] = value
                break
        stop = body.get("stop")
        if stop is not None:
            overrides["stop"] = stop
        return overrides

    def render(
        self,
        conversation: Optional[ChatConversation],
        last_tokens: list[int],
        messages: list[dict],
        tools: Optional[list],
    ) -> RenderedInput:
        """The input ids for this request, per the TITO rules.

        A request that continues the stored conversation renders only its new
        messages on top of the previous turn's tokens. A retry of the last
        request rolls back instead: the returned ``rollback`` tells the
        worker to drop that turn from the training data, and the retried
        request renders from the checkpoint. A request that changes the tool
        list, or any other request, re-renders the whole conversation and
        starts a new segment.

        Args:
            conversation: The session's stored conversation, None before its
                first turn.
            last_tokens: The previous turn's input and output ids.
            messages: The request's messages.
            tools: The request's tools, or None.
        """
        stored = conversation.messages if conversation is not None else []
        stored_tools = conversation.tools if conversation is not None else None
        rollback = False
        # The stored tokens were rendered under the stored tool list, and a
        # checkpoint's tokens under its own, so a request whose tools differ
        # from the tokens it would continue renders the whole conversation.
        if (
            _tools_equal(tools, stored_tools)
            and conversation is not None
            and conversation.checkpoint is not None
            and stored
            and len(messages) == len(stored) - 1
            and _match_prefix(messages, stored[:-1]) is not None
        ):
            # The request repeats itself, so the client never used the last
            # turn's answer. Render from before that turn, under the tool
            # list those tokens were rendered with. The checkpoint is
            # consumed here: a further rollback has no state of its own to
            # restore, and such a request renders the whole conversation
            # instead.
            checkpoint = conversation.checkpoint
            rollback = True
            stored = list(checkpoint.messages)
            stored_tools = checkpoint.tools
            last_tokens = list(checkpoint.last_input or []) + list(
                checkpoint.last_output or []
            )
        if (
            _tools_equal(tools, stored_tools)
            and stored
            and len(messages) > len(stored)
            and _match_prefix(messages, stored) is not None
        ):
            suffix = messages[len(stored) :]
            clear_thinking = self._tokenizer.chat_template_kwargs.get(
                "clear_thinking", False
            )
            if not (clear_thinking and _has_user_message(suffix)):
                return RenderedInput(
                    self._tokenizer.build_native_tool_response(
                        list(last_tokens), suffix
                    ),
                    messages=[*stored, *suffix],
                    rollback=rollback,
                )
        return RenderedInput(
            self._tokenizer.apply_chat_template(
                _template_messages(messages),
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            ),
            messages=list(messages),
            rollback=rollback,
        )

    def decode_output(self, output_ids: list[int], matched_stop: Any = None) -> str:
        """Decode the output token ids into the text the engine would return.

        With ``rollout.detokenize: false`` the engines start with
        ``skip_tokenizer_init`` and return only ``output_ids``; without this
        decoding, every chat response would carry an empty message. The
        decoding uses the engine's default detokenizer settings and removes a
        matched stop string or stop token the same way the engine does, so
        the text equals what the engine returns with detokenization on.

        Args:
            output_ids: The turn's output token ids.
            matched_stop: The ``matched`` field of the engine's
                ``finish_reason``: the stop string or stop token id that
                ended the turn, or None.

        Returns:
            The decoded text.
        """
        if isinstance(matched_stop, int) and output_ids:
            # The engine drops the matched stop token before decoding.
            output_ids = output_ids[:-1]
        text = self._tokenizer.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        )
        if isinstance(matched_stop, str) and matched_stop:
            # The engine cuts a matched stop string from its text but keeps
            # the matching tokens in output_ids.
            cut = text.find(matched_stop)
            if cut != -1:
                text = text[:cut]
        return text

    def parse(
        self,
        text: Optional[str],
        input_ids: list[int],
        tools: Optional[list],
        tool_choice: Any,
    ) -> ParsedMessage:
        """The assistant message the engine's text parses into.

        Args:
            text: The engine's decoded output.
            input_ids: The rendered request, whose tail says whether the
                output starts inside a reasoning block.
            tools: The request's tools, or None.
            tool_choice: The request's tool_choice.
        """
        content = text or ""
        reasoning_content = None
        # Only models whose tokenizer has the reasoning markers get a
        # reasoning split. For any other model the markers are ordinary
        # text a model may write, and splitting on them would swallow the
        # rest of the output: an unclosed marker makes sglang's parser
        # treat everything after it — tool calls included — as reasoning,
        # leaving the agent an empty answer.
        if (
            self._tokenizer.reasoning_parser is not None
            and self._tokenizer.reasoning_end_token_id is not None
        ):
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            # A generation prompt that ends with the reasoning opener (GLM)
            # means the engine output starts inside the block: the parser
            # then needs force_reasoning, as sglang itself passes it.
            parser = ReasoningParser(
                model_type=self._tokenizer.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=self._tokenizer.prompt_ends_in_reasoning(input_ids),
            )
            reasoning, normal = parser.parse_non_stream(content)
            if reasoning:
                reasoning_content = reasoning
            if normal is not None:
                content = normal
        tool_calls = None
        tool_call_parse_failed = False
        if tools and tool_choice != "none" and self._tokenizer.tool_call_parser:
            tool_calls, content, tool_call_parse_failed = self._parse_tool_calls(
                content, tools
            )
        message: dict = {"role": "assistant", "content": content}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls
        return ParsedMessage(
            message=message,
            has_reasoning=bool(reasoning_content),
            has_tool_calls=bool(tool_calls),
            tool_call_parse_failed=tool_call_parse_failed,
        )

    def _parse_tool_calls(
        self, content: str, tools: list
    ) -> tuple[Optional[list], str, bool]:
        """Parse tool calls out of the content, degrading to plain text.

        Returns the parsed calls (None when there are none), the content
        without them, and whether the parse failed. A parse failure is not
        the turn's failure: the whole text answers as content, the same
        degradation sglang's own chat serving applies.

        A tool-call block the parser cannot read counts as a failure even
        though the parser only warns about it and reports no call: without
        this, the model's text would silently disappear, since the content
        the parser returns stops before the block. When the parser reads
        some calls of a turn and skips others, the read ones are returned
        and nothing is counted — it does not say how many it skipped.
        """
        from sglang.srt.entrypoints.openai.protocol import Function, Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        sglang_tools = [
            Tool(
                function=Function(
                    name=tool["function"]["name"],
                    description=tool["function"].get("description", ""),
                    parameters=tool["function"].get("parameters", {}),
                )
            )
            for tool in tools
            if tool.get("type", "function") == "function"
        ]
        parser = FunctionCallParser(sglang_tools, self._tokenizer.tool_call_parser)
        try:
            normal, calls = parser.parse_non_stream(content)
        except Exception:
            return None, content, True
        if not calls:
            return None, content, parser.has_tool_call(content)
        parsed = [
            {
                # The client echoes this id on its tool message, so it must
                # be unique within the turn: tool_index is the position of
                # the tool in the tools list, the same for two calls of one
                # tool.
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": call.name, "arguments": call.parameters},
            }
            for call in calls
        ]
        return parsed, normal, False

    def response_body(
        self, body: dict, parsed: ParsedMessage, result: dict, input_ids: list[int]
    ) -> dict:
        """The OpenAI chat.completion body for one finished turn.

        Args:
            body: The request body.
            parsed: The parsed engine output.
            result: The generate core's result for this turn.
            input_ids: The rendered request, for the prompt-token count.
        """
        if parsed.has_tool_calls:
            finish_reason = "tool_calls"
        elif (result.get("finish_reason") or {}).get("type") == "length":
            finish_reason = "length"
        else:
            finish_reason = "stop"
        output_ids = result["output_ids"]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or self._model_name,
            "choices": [
                {
                    "index": 0,
                    "message": parsed.message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(output_ids),
                "total_tokens": len(input_ids) + len(output_ids),
                "prompt_tokens_details": {
                    "cached_tokens": result.get("cached_tokens", 0)
                },
            },
        }

    @staticmethod
    def stream_body(response: dict) -> bytes:
        """One protocol-legal chunk carrying the whole turn, then DONE.

        Clients default to streaming; one delta is legal and all tool calls
        ride in it (some clients mis-assemble arguments split across
        chunks).
        """
        message = response["choices"][0]["message"]
        delta = {"role": message["role"], "content": message["content"]}
        if message.get("reasoning_content") is not None:
            delta["reasoning_content"] = message["reasoning_content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                {**tool_call, "index": index}
                for index, tool_call in enumerate(message["tool_calls"])
            ]
        chunk = {
            "id": response["id"],
            "object": "chat.completion.chunk",
            "created": response["created"],
            "model": response["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": response["choices"][0]["finish_reason"],
                }
            ],
            "usage": response["usage"],
        }
        return b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n"


def _has_user_message(messages: list[dict]) -> bool:
    return any(message.get("role") == "user" for message in messages)


def _template_messages(messages: list[dict]) -> list[dict]:
    """A copy of the messages with template-friendly tool call arguments.

    Some chat templates (GLM) treat ``function.arguments`` as an object,
    while OpenAI clients send the JSON string form. Renders only need the
    parsed form; the stored conversation keeps the request's own shape.
    """
    converted = []
    for message in messages:
        calls = message.get("tool_calls")
        if not calls:
            converted.append(message)
            continue
        new_calls = []
        for call in calls:
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except ValueError:
                    pass
            new_calls.append({**call, "function": function})
        converted.append({**message, "tool_calls": new_calls})
    return converted


def _match_prefix(request_messages: list[dict], stored: list[dict]) -> Optional[int]:
    """Where the request stops matching the stored conversation.

    Returns the length of the matching prefix when ``request_messages``
    starts with ``stored``, else None. A message's content counts as
    matching when both sides carry none (None, missing, and "" all mean no
    content). Tool-call arguments compare by their parsed JSON; an
    assistant's reasoning_content counts as matching when the request omits
    it or sends it empty, so a client that clears history thinking still
    continues the conversation.
    """
    if len(request_messages) < len(stored):
        return None
    for request, saved in zip(request_messages, stored):
        if request.get("role") != saved.get("role"):
            return None
        if (request.get("content") or "") != (saved.get("content") or ""):
            return None
        if not _tool_calls_equal(request.get("tool_calls"), saved.get("tool_calls")):
            return None
        if request.get("role") == "assistant":
            request_reasoning = request.get("reasoning_content")
            if request_reasoning and request_reasoning != saved.get(
                "reasoning_content"
            ):
                return None
    return len(stored)


def _tools_equal(request_tools: Any, stored_tools: Any) -> bool:
    """Whether a request's tool list matches the one the tokens were rendered
    under. No tools (None, missing, or an empty list) equals no tools."""
    request_none = request_tools is None or request_tools == []
    stored_none = stored_tools is None or stored_tools == []
    if request_none or stored_none:
        return request_none and stored_none
    return request_tools == stored_tools


def _tool_calls_equal(
    request_calls: Optional[list], stored_calls: Optional[list]
) -> bool:
    """Tool calls match on id, name, order, and parsed arguments. No tool
    calls (None, missing, or an empty list) equals no tool calls."""
    request_none = request_calls is None or request_calls == []
    stored_none = stored_calls is None or stored_calls == []
    if request_none or stored_none:
        return request_none and stored_none
    if not isinstance(request_calls, list) or not isinstance(stored_calls, list):
        return False
    if len(request_calls) != len(stored_calls):
        return False
    for request, saved in zip(request_calls, stored_calls):
        if request.get("id") != saved.get("id"):
            return False
        request_fn = request.get("function") or {}
        saved_fn = saved.get("function") or {}
        if request_fn.get("name") != saved_fn.get("name"):
            return False
        try:
            if json.loads(request_fn.get("arguments") or "{}") != json.loads(
                saved_fn.get("arguments") or "{}"
            ):
                return False
        except (TypeError, ValueError):
            if request_fn.get("arguments") != saved_fn.get("arguments"):
                return False
    return True
