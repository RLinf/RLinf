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
import random
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

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


class MCPAgentLoopWorker(AgentLoopWorker):
    """
    An agent loop worker that can interact with mcp tools with session.
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)

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
            response: ToolChannelResponse = await self.tool_worker_output_channel.get(
                session_id, async_op=True
            ).async_wait()
            assert response.success
        return session_id

    async def tool_session_release(self, tool_worker_name, session_id) -> str | dict:
        tool_channel_info = self.tool_channel_info_map[tool_worker_name]
        await tool_channel_info.input_channel.put(
            ToolChannelRequest(session_id=session_id, request_type="session_end"),
            async_op=True,
        ).async_wait()
        response: ToolChannelResponse = await self.tool_worker_output_channel.get(
            session_id, async_op=True
        ).async_wait()
        assert response.success

    async def tool_call(
        self, generate_context: GenerateContext, tool_request: ToolRequest
    ) -> ToolResponse:
        tool_name, tool_args = tool_request.name, tool_request.arguments
        tool_channel_info = self.tool_channel_info_map[self.tool_name_map[tool_name]]
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
        response: ToolChannelResponse = await self.tool_worker_output_channel.get(
            session_id, async_op=True
        ).async_wait()
        assert response.success
        if isinstance(response.result, (list, dict)):
            result_text = json.dumps(response.result)
        else:
            result_text = str(response.result)
        return ToolResponse(
            text=result_text,
        )

    async def extract_tool_calls(self, response_text) -> tuple[str, list[ToolRequest]]:
        # random tool call
        return_function_calls = random.choice(
            [
                [
                    ToolRequest(
                        name="write_file",
                        arguments={
                            "path": "/projects/test/mcp_written.txt",
                            "content": f"Written by mcp at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        },
                    )
                ],
                [
                    ToolRequest(
                        name="list_directory", arguments={"path": "/projects/test"}
                    )
                ],
            ]
        )

        return response_text, return_function_calls

    async def run_one_query(self, prompt_ids: list[int]) -> AgentLoopOutput:
        orig_prompt_ids = copy.deepcopy(prompt_ids)
        generate_context: GenerateContext = self.generate_context_create()
        trace_prints = []
        response_mask = []
        try:
            # 5 is a magic number in this demo.
            for _ in range(5):
                # Generate response from LLM
                max_prompt_len = int(self.cfg.data.get("max_prompt_length", 1024))
                max_total_len = int(self.cfg.actor.model.encoder_seq_length)
                max_resp_len = max(1, max_total_len - max_prompt_len)

                if len(prompt_ids) > max_prompt_len:
                    prompt_ids = prompt_ids[-max_prompt_len:]

                generate_result = await self.generate(prompt_ids)
                response_ids = generate_result["output_ids"]
                if len(response_ids) > max_resp_len:
                    response_ids = response_ids[:max_resp_len]
                response_text = self.tokenizer.decode(response_ids)

                prompt_ids += response_ids
                response_mask += [1] * len(response_ids)  # 1 for LLM generated tokens

                # Extract tool calls from response
                _, tool_requests = await self.extract_tool_calls(response_text)

                # Execute tools in parallel with history propagation
                tasks = []
                for tool_request in tool_requests:
                    tasks.append(self.tool_call(generate_context, tool_request))
                tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)

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
                )
                prompt_ids += tool_response_ids
                response_mask += [0] * len(
                    tool_response_ids
                )  # 0 for tool response tokens
                if self.print_outputs:
                    # add anything you want to print
                    trace_prints.append(
                        {"generate": response_text, "tool_resp": tool_messages}
                    )

            # Separate prompt and response
            response_ids = prompt_ids[-len(orig_prompt_ids) :]

            return AgentLoopOutput(
                prompt_ids=orig_prompt_ids,
                prompt_text=self.tokenizer.decode(orig_prompt_ids),
                response_ids=response_ids,
                response_mask=response_mask,
                trace_prints=trace_prints,
            )
        finally:
            await self.generate_context_release(generate_context)
