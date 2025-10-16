# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
import logging
import os

import regex as re

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLINF_LOGGING_LEVEL", "WARN"))


class FunctionCall:
    """Function call representation compatible with verl"""
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class ToolResponse:
    """Tool response representation compatible with verl"""
    def __init__(self, text: str = None):
        self.text = text


class ToolParser:
    """Base tool parser class with registry functionality"""
    _registry: dict[str, type["ToolParser"]] = {}

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """Extract tool calls from the responses.

        Args:
            responses_ids (List[int]): The ids of the responses.

        Returns:
            Tuple[str, List[FunctionCall]]: Content and extracted tool calls.
        """
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        if name not in cls._registry:
            raise ValueError(f"Unknown tool parser: {name}")
        return cls._registry[name](tokenizer)

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[ToolParser]) -> type[ToolParser]:
            cls._registry[name] = subclass
            return subclass
        return decorator


@ToolParser.register("rstar2_agent_hermes")
class RStar2AgentHermesToolParser(ToolParser):
    """Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"""

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}")
                ################################### rStar ###################################
                # keep the error msg as tool response
                function_calls.append(ToolResponse(text=f"Failed to decode tool call: {e}"))
                #############################################################################

        # remaining text exclude tool call tokens
        content = self.tool_call_regex.sub("", text)

        return content, function_calls
