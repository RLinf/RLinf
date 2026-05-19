from __future__ import annotations

from typing import Any, Callable

# Compatibility shims for SGLang 0.4.6 only. These patch legacy OpenAI chat
# behavior that newer SGLang versions handle natively.
LEGACY_TOOL_CALL_PARSE_ERROR = b"Failed to parse fc related info to json format!"


def patch_chat_body_assistant_content(body: object) -> None:
    """Normalize OpenAI-compatible assistant tool-call messages for SGLang 0.4.6.

    LiteLLM/OpenAI clients may omit `content` when it is null. SGLang 0.4.6
    requires the key to be present on assistant messages, even though the value
    can be null.
    """
    if not isinstance(body, dict):
        return
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and "content" not in message
        ):
            message["content"] = None


def downgrade_legacy_tool_call_parse_error(
    response: Any,
    *,
    request: Any,
    ret: Any,
    created: Any,
    response_builder: Callable[..., Any],
    tool_call_parser: Any = None,
) -> Any:
    """Turn SGLang 0.4.6 tool-call parse failures into plain chat completions."""
    body = response.body or b""
    if response.status_code != 400 or LEGACY_TOOL_CALL_PARSE_ERROR not in body:
        return response

    req_no_tools = request.model_copy(update={"tool_choice": "none", "tools": None})
    return response_builder(
        req_no_tools,
        ret,
        created,
        tool_call_parser=tool_call_parser,
    )
