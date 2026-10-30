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

"""Render RPent planner chat requests into prompt token IDs."""

from __future__ import annotations

import copy
import json

from rlinf.agents.tokenization import AgenticTokenizer


def template_messages(messages: list[dict]) -> list[dict]:
    """Return messages in the form the chat template accepts.

    Tool-call arguments become mappings; calls whose arguments are not a JSON
    object are kept as text in the message content.
    """
    messages = copy.deepcopy(messages)
    for message in messages:
        if message.get("content") is None:
            message["content"] = ""
        if not isinstance(message["content"], str):
            raise ValueError(
                f"RPent text prompts need string message content, got "
                f"{type(message['content']).__name__}"
            )
        calls, malformed = [], []
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            if isinstance(arguments, dict):
                call["function"]["arguments"] = arguments
                calls.append(call)
            else:
                malformed.append(json.dumps(call, ensure_ascii=False))
        if malformed:
            message["content"] += "\n" + "\n".join(malformed)
            message["tool_calls"] = calls
    return messages


def render_prompt(tokenizer: AgenticTokenizer, request: dict) -> list[int]:
    """Render a whole chat request with the model's chat template."""
    text = tokenizer.apply_chat_template(
        template_messages(request["messages"]),
        tools=request.get("tools"),
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.tokenizer(text, add_special_tokens=False)["input_ids"]


class AppendOnlyHistory:
    """Build each turn's prompt by appending to the previous turn's sampled tokens.

    A request that continues the recorded conversation (same earlier messages
    and tools, the recorded assistant reply, then only user or tool messages)
    becomes the previous prompt plus the sampled response plus the new
    messages, so every turn is a token prefix of the next. Any other request,
    such as a compacted history, is rendered in full and starts a new chain.
    """

    def __init__(self, tokenizer: AgenticTokenizer) -> None:
        self.tokenizer = tokenizer
        self._messages = None
        self._tools = None
        self._assistant_message = None
        self._ids = None

    def _continues(self, request: dict) -> bool:
        messages = request["messages"]
        count = 0 if self._messages is None else len(self._messages)
        return (
            self._messages is not None
            and request.get("tools") == self._tools
            and len(messages) > count + 1
            and messages[:count] == self._messages
            and template_messages([messages[count]])[0] == self._assistant_message
            and all(m.get("role") in {"user", "tool"} for m in messages[count + 1 :])
        )

    def render(self, request: dict) -> list[int]:
        """Return the prompt token IDs for a chat request."""
        if not self._continues(request):
            return render_prompt(self.tokenizer, request)
        new_messages = template_messages(request["messages"][len(self._messages) + 1 :])
        return self.tokenizer.build_native_tool_response(self._ids, new_messages)

    def record(
        self,
        request: dict,
        prompt_ids: list[int],
        response_ids: list[int],
        assistant_message: dict,
    ) -> None:
        """Make a sampled turn the base of the next request."""
        self._messages = copy.deepcopy(request["messages"])
        self._tools = copy.deepcopy(request.get("tools"))
        self._assistant_message = template_messages([assistant_message])[0]
        self._ids = [*prompt_ids, *response_ids]
