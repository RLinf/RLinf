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

import json
import logging
import re

import regex

from rlinf.algorithms.registry import register_toolcall_parser
from rlinf.data.tool_call.tool_io_struct import (
    ToolRequest,
    ToolResponse,
)


@register_toolcall_parser("qwen2.5")
class Qwen25ToolCallParser:
    """Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"""

    def __init__(self):
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = regex.compile(
            r"<tool_call>(.*?)</tool_call>", regex.DOTALL
        )

    async def __call__(self, responses_text: str) -> tuple[str, list[ToolRequest]]:
        text = responses_text
        if (
            self.tool_call_start_token not in text
            or self.tool_call_end_token not in text
        ):
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(
                    ToolRequest(
                        name=name, arguments=json.dumps(arguments, ensure_ascii=False)
                    )
                )
            except Exception as e:
                logging.error(f"Failed to decode tool call: {e}")

        # remaing text exclude tool call tokens
        content = self.tool_call_regex.sub("", text)

        return content, function_calls


@register_toolcall_parser("searchr1-qwen")
class Searchr1QwenToolCallParser:
    def __init__(self) -> None:
        self.tool_call_start_token: str = "<search>"
        self.tool_call_end_token: str = "</search>"
        self.tool_call_regex = re.compile(r"<search>(.*?)</search>", re.DOTALL)

    async def __call__(self, response_text: str) -> tuple[str, list[ToolRequest]]:
        if (
            self.tool_call_start_token not in response_text
            or self.tool_call_end_token not in response_text
        ):
            return response_text, []
        matches = self.tool_call_regex.findall(response_text)
        function_calls = []
        if matches:
            match = matches[-1].strip()
            function_calls.append(
                ToolRequest(name="search", arguments={"keyword": match})
            )

        # remaining text exclude tool call tokens
        content = self.tool_call_regex.sub("", response_text)

        return content, function_calls


@register_toolcall_parser("rstar2-qwen")
class Rstar2QwenToolCallParser:
    def __init__(self) -> None:
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    async def __call__(
        self, response_text: str
    ) -> tuple[str, list[ToolRequest | ToolResponse]]:
        if (
            self.tool_call_start_token not in response_text
            or self.tool_call_end_token not in response_text
        ):
            return response_text, []

        matches = self.tool_call_regex.findall(response_text)
        return_function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                return_function_calls.append(
                    ToolRequest(name=name, arguments=arguments)
                )
            except Exception as e:
                return_function_calls.append(
                    ToolResponse(text=f"Failed to decode tool call: {e}")
                )

        return response_text, return_function_calls


@register_toolcall_parser("wideseek_r1-qwen")
class WideSeekQwenToolCallParser:
    """Tool-call parser for WideSeek-R1 planner/worker/single-agent roles."""

    def __init__(self) -> None:
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    @staticmethod
    def _parse_planner_calls(
        tool_name: str,
        tool_arguments: dict,
        max_workers_per_planner: int,
    ) -> list[ToolRequest]:
        if tool_name != "create_sub_agents":
            return []
        sub_agents = tool_arguments.get("sub_agents", [])
        if not isinstance(sub_agents, list):
            return []

        function_calls = []
        for sub_agent in sub_agents[:max_workers_per_planner]:
            if not isinstance(sub_agent, dict):
                continue
            prompt = sub_agent.get("prompt", "")
            if not isinstance(prompt, str) or not prompt:
                continue
            function_calls.append(
                ToolRequest(name="subtask", arguments={"subtask": prompt})
            )
        return function_calls

    @staticmethod
    def _parse_worker_calls(
        tool_name: str,
        tool_arguments: dict,
        max_toolcall_per_worker: int,
    ) -> list[ToolRequest]:
        function_calls = []
        if tool_name == "search":
            searches = tool_arguments.get("queries", [])
            if not isinstance(searches, list):
                return []
            for search_item in searches[:max_toolcall_per_worker]:
                if not isinstance(search_item, dict):
                    continue
                query = search_item.get("query", "")
                if not isinstance(query, str) or not query:
                    continue
                topk = search_item.get("count", None)
                if topk:
                    function_calls.append(
                        ToolRequest(
                            name="search",
                            arguments={"query": query, "topk": topk},
                        )
                    )
                else:
                    function_calls.append(
                        ToolRequest(name="search", arguments={"query": query})
                    )

        elif tool_name == "access":
            accesses = tool_arguments.get("urls", [])
            if not isinstance(accesses, list):
                return []
            for access_item in accesses[:max_toolcall_per_worker]:
                if not isinstance(access_item, dict):
                    continue
                url = access_item.get("url", "")
                if not isinstance(url, str) or not url:
                    continue
                info_to_extract = access_item.get("info_to_extract", None)
                function_calls.append(
                    ToolRequest(
                        name="access",
                        arguments={
                            "url": url,
                            "access_token": 25000,
                            "info_to_extract": info_to_extract,
                        },
                    )
                )
        return function_calls

    @staticmethod
    def _parse_single_calls(tool_name: str, tool_arguments: dict) -> list[ToolRequest]:
        if tool_name == "search":
            query = tool_arguments.get("query", "")
            if not isinstance(query, str) or not query:
                return []
            topk = tool_arguments.get("count", None)
            if topk:
                return [
                    ToolRequest(
                        name="search",
                        arguments={"query": query, "topk": topk},
                    )
                ]
            return [ToolRequest(name="search", arguments={"query": query})]

        if tool_name == "access":
            url = tool_arguments.get("url", "")
            if not isinstance(url, str) or not url:
                return []
            info_to_extract = tool_arguments.get("info_to_extract", None)
            return [
                ToolRequest(
                    name="access",
                    arguments={
                        "url": url,
                        "access_token": 25000,
                        "info_to_extract": info_to_extract,
                    },
                )
            ]
        return []

    async def __call__(
        self,
        response_text: str,
        *,
        role: str,
        max_workers_per_planner: int = 10,
        max_toolcall_per_worker: int = 5,
    ) -> tuple[str, list[ToolRequest]]:
        if (
            self.tool_call_start_token not in response_text
            or self.tool_call_end_token not in response_text
        ):
            return response_text, []

        matches = self.tool_call_regex.findall(response_text)
        if not matches:
            return response_text, []

        try:
            tool_call_json = json.loads(matches[0].strip())
        except Exception:
            return response_text, []

        if not isinstance(tool_call_json, dict):
            return response_text, []
        tool_name = tool_call_json.get("name")
        tool_arguments = tool_call_json.get("arguments", {})
        if not isinstance(tool_arguments, dict):
            return response_text, []

        if role == "planner":
            function_calls = self._parse_planner_calls(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                max_workers_per_planner=max_workers_per_planner,
            )
        elif role == "worker":
            function_calls = self._parse_worker_calls(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                max_toolcall_per_worker=max_toolcall_per_worker,
            )
        elif role == "single":
            function_calls = self._parse_single_calls(
                tool_name=tool_name, tool_arguments=tool_arguments
            )
        else:
            function_calls = []

        # remaining text exclude tool call tokens
        content = self.tool_call_regex.sub("", response_text)
        return content, function_calls
