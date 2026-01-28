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

import asyncio
import copy
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import regex as re
from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import AgentLoopOutput, AgentLoopWorker


@dataclass
class GenerateContext:
    tool_session_ids: dict[str, str] = field(default_factory=dict)


class Rstar2AgentLoopWorker(AgentLoopWorker):
    """Simple tool agent loop that can interact with tools."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        if self.cfg.rollout.get("custom_chat_template", None) is not None:
            self.tokenizer.chat_template = cfg.rollout.custom_chat_template

        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        max_total_len = int(self.cfg.actor.model.encoder_seq_length)
        self.max_resp_len = max(1, max_total_len - self.max_prompt_len)

        self.max_user_turns = cfg.agentloop.get("max_user_turns", 5)
        self.max_assistant_turns = cfg.agentloop.get("max_assistant_turns", 5)
        self.max_parallel_calls = cfg.agentloop.get("max_parallel_calls", 3)
        self.max_tool_response_length = cfg.agentloop.get(
            "max_tool_response_length", 500
        )
        self.tool_response_truncate_side = cfg.agentloop.get(
            "tool_response_truncate_side", "right"
        )
        self.apply_chat_template_kwargs = cfg.data.get("apply_chat_template_kwargs", {})
        self.system_prompt = self.tokenizer.apply_chat_template(
            [{}],
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

    def generate_context_create(self) -> dict[str, Any]:
        return GenerateContext()

    async def generate_context_release(
        self, generate_context: GenerateContext
    ) -> dict[str, Any]:
        for tool_worker_name, session_id in generate_context.tool_session_ids.items():
            if self.tool_channel_info_map[tool_worker_name].has_session:
                # tool need session
                await self.tool_session_release(tool_worker_name, session_id)

    async def tool_session_get(
        self, generate_context: GenerateContext, tool_name: str
    ) -> Any:
        tool_worker_name = self.tool_name_map[tool_name]
        tool_channel_info = self.tool_channel_info_map[tool_worker_name]
        if tool_worker_name in generate_context.tool_session_ids:
            return generate_context.tool_session_ids[tool_worker_name]
        session_id = uuid4().hex
        generate_context.tool_session_ids[tool_worker_name] = session_id
        if tool_channel_info.has_session:
            # tool need session
            await tool_channel_info.input_channel.put(
                ToolChannelRequest(session_id=session_id, request_type="session_start"),
                async_op=True,
            ).async_wait()
            self.log_debug("session_start put")
            response: ToolChannelResponse = await self.tool_worker_output_channel.get(
                session_id, async_op=True
            ).async_wait()
            assert response.success
            self.log_debug("session_start get")
        return session_id

    async def tool_session_release(self, tool_worker_name, session_id) -> str | dict:
        tool_channel_info = self.tool_channel_info_map[tool_worker_name]
        await tool_channel_info.input_channel.put(
            ToolChannelRequest(session_id=session_id, request_type="session_end"),
            async_op=True,
        ).async_wait()
        self.log_debug("session_end put")
        response: ToolChannelResponse = await self.tool_worker_output_channel.get(
            session_id, async_op=True
        ).async_wait()
        assert response.success
        self.log_debug("session_end get")

    async def tool_call(
        self,
        generate_context: GenerateContext,
        tool_request: ToolRequest | ToolChannelResponse,
    ) -> ToolResponse:
        if isinstance(tool_request, ToolChannelResponse):
            return ToolResponse(text=tool_request.result)
        elif tool_request.name not in self.tool_name_map:
            return ToolResponse(
                text=f"Error when executing tool: '{tool_request.name}'"
            )
        else:
            tool_name, tool_args = tool_request.name, tool_request.arguments
            tool_channel_info = self.tool_channel_info_map[
                self.tool_name_map[tool_name]
            ]
            tool_input_channel = tool_channel_info.input_channel
            session_id = await self.tool_session_get(generate_context, tool_name)
            await tool_input_channel.put(
                ToolChannelRequest(
                    session_id=session_id,
                    request_type="execute",
                    tool_name=tool_name,
                    tool_args=tool_args,
                ),
                async_op=True,
            ).async_wait()
            self.log_debug("tool execute put")
            response: ToolChannelResponse = await self.tool_worker_output_channel.get(
                session_id, async_op=True
            ).async_wait()
            self.log_debug("tool execute get")
            self.log_info(f"{response}")
            if isinstance(response.result, (list, dict)):
                result_text = json.dumps(response.result)
            else:
                result_text = str(response.result)

            if result_text and len(result_text) > self.max_tool_response_length:
                if self.tool_response_truncate_side == "left":
                    result_text = (
                        result_text[: self.max_tool_response_length] + "...(truncated)"
                    )
                elif self.tool_response_truncate_side == "right":
                    result_text = (
                        "(truncated)..." + result_text[-self.max_tool_response_length :]
                    )
                else:
                    length = self.max_tool_response_length // 2
                    result_text = (
                        result_text[:length]
                        + "...(truncated)..."
                        + result_text[-length:]
                    )
            return ToolResponse(text=result_text)

    async def extract_tool_calls(
        self, response_text
    ) -> tuple[str, list[ToolRequest | ToolChannelResponse]]:
        tool_call_start_token: str = "<tool_call>"
        tool_call_end_token: str = "</tool_call>"
        tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

        if (
            tool_call_start_token not in response_text
            or tool_call_end_token not in response_text
        ):
            return response_text, []

        matches = tool_call_regex.findall(response_text)
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
                    ToolChannelResponse(
                        success=False, result=f"Failed to decode tool call: {e}"
                    )
                )

        return response_text, return_function_calls

    async def demo_tool_call(self, response_text: str):
        generate_context: GenerateContext = self.generate_context_create()
        _, tool_requests = await self.extract_tool_calls(response_text)
        # Execute tools in parallel with history propagation
        tool_requests = tool_requests[: self.max_parallel_calls]
        total_tool_responses, filtered_tool_requests, pending_pos = [], [], []
        for i, tool_request in enumerate(tool_requests):
            if isinstance(tool_request, ToolResponse):
                total_tool_responses.append(tool_request)
            else:
                total_tool_responses.append(None)
                pending_pos.append(i)
                filtered_tool_requests.append(tool_request)
        tool_requests = filtered_tool_requests

        tasks = []
        for tool_request in tool_requests[: self.max_parallel_calls]:
            tasks.append(self.tool_call(generate_context, tool_request))
        tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)
        for i, tool_response in zip(
            pending_pos[: self.max_parallel_calls], tool_responses, strict=False
        ):
            total_tool_responses[i] = tool_response
        tool_responses = total_tool_responses
        return tool_responses[0]

    async def run_one_query(self, prompt_ids: list[int]) -> AgentLoopOutput:
        prompt_ids = copy.deepcopy(prompt_ids)
        orig_prompt_ids = copy.deepcopy(prompt_ids)
        orig_prompt_text = self.tokenizer.decode(orig_prompt_ids)
        generate_context: GenerateContext = self.generate_context_create()
        trace_uuid = uuid4().hex
        trace_prints: list[dict] = []
        if self.print_outputs:
            trace_prints.append({"uuid": trace_uuid, "prompt": orig_prompt_text})
        response_mask = []
        response_logprobs = None
        if self.return_logprobs:
            response_logprobs = []
        user_turns, assistant_turns = 0, 0
        while True:
            # Generate response from LLM
            max_resp_len = self.max_resp_len - (len(prompt_ids) - len(orig_prompt_ids))

            generate_result = await self.generate(
                prompt_ids, sampling_params={"max_new_tokens": max_resp_len}
            )
            response_ids = generate_result["output_ids"]
            if self.return_logprobs:
                generate_logprobs = generate_result["logprobs"]
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]
                if self.return_logprobs:
                    generate_logprobs = generate_logprobs[:max_resp_len]

            response_text = self.tokenizer.decode(response_ids)

            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)  # 1 for LLM generated tokens
            if self.return_logprobs:
                response_logprobs += generate_logprobs
            assistant_turns += 1
            if len(response_ids) == max_resp_len:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break
            # Extract tool calls from response
            _, tool_requests = await self.extract_tool_calls(response_text)
            if len(tool_requests) == 0:
                break
            # Execute tools in parallel with history propagation
            tool_requests = tool_requests[: self.max_parallel_calls]
            total_tool_responses, filtered_tool_requests, pending_pos = [], [], []
            for i, tool_request in enumerate(tool_requests):
                if isinstance(tool_request, ToolResponse):
                    total_tool_responses.append(tool_request)
                else:
                    total_tool_responses.append(None)
                    pending_pos.append(i)
                    filtered_tool_requests.append(tool_request)
            tool_requests = filtered_tool_requests

            tasks = []
            for tool_request in tool_requests[: self.max_parallel_calls]:
                tasks.append(self.tool_call(generate_context, tool_request))
            tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)
            for i, tool_response in zip(
                pending_pos[: self.max_parallel_calls], tool_responses, strict=False
            ):
                total_tool_responses[i] = tool_response
            tool_responses = total_tool_responses
            if any(isinstance(item, Exception) for item in tool_responses):
                assert False
            # Convert tool responses to messages and tokenize
            tool_messages = []
            for tool_response in tool_responses:
                message = {"role": "tool", "content": tool_response.text}
                tool_messages.append(message)

            # Tokenize tool responses
            tool_response_ids = self.tokenizer.apply_chat_template(
                tool_messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]
            max_tool_resp_len = self.max_resp_len - (
                len(prompt_ids) - len(orig_prompt_ids)
            )
            if len(tool_response_ids) > max_tool_resp_len:
                break
            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)  # 0 for tool response tokens
            if self.return_logprobs:
                response_logprobs += [0.0] * len(tool_response_ids)

            user_turns += 1

        # Separate prompt and response
        response_ids = prompt_ids[len(orig_prompt_ids) :]
        extra_fields = {"num_turns": user_turns + assistant_turns + 1}
        if self.print_outputs:
            trace_prints.append(
                {
                    "uuid": trace_uuid,
                    "generate": self.tokenizer.decode(response_ids),
                    "assistant_turns": user_turns + assistant_turns + 1,
                }
            )
        return AgentLoopOutput(
            prompt_ids=orig_prompt_ids,
            prompt_text=orig_prompt_text,
            response_ids=response_ids,
            response_text=self.tokenizer.decode(response_ids),
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            trace_prints=trace_prints,
            extra_fields=extra_fields,
        )
