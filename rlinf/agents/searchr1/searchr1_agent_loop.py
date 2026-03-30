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
import json
from typing import Any
from uuid import uuid4

from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
)
from rlinf.scheduler import Channel
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import AgentLoopWorker


class Searchr1ToolAgentLoopWorker(AgentLoopWorker):
    """Simple tool agent loop that can interact with tools."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        max_total_len = int(self.cfg.runner.seq_length)
        self.max_resp_len = max(1, max_total_len - self.max_prompt_len)

        assert self.toolcall_parser is not None, (
            "toolcall_parser must be set in searchr1"
        )

        # Inserting tool info requires re-encode token_ids, so the recompute_logprobs must be true.
        if self.cfg.runner.task_type != "reasoning_eval":
            assert self.cfg.algorithm.recompute_logprobs, (
                "search r1 must use recompute_logprobs"
            )

    async def state_less_tool_call_with_channel(
        self,
        input_channel: Channel,
        output_channel: Channel,
        tool_name: str,
        tool_args: dict,
    ) -> ToolChannelResponse:
        """state-less tool call with channel, used for demo"""
        session_id = uuid4().hex
        await input_channel.put(
            ToolChannelRequest(
                session_id=session_id,
                request_type="execute",
                tool_name=tool_name,
                tool_args=tool_args,
            ),
            async_op=True,
        ).async_wait()
        return await output_channel.get(session_id, async_op=True).async_wait()

    async def tool_call(self, tool_request: ToolRequest) -> ToolResponse:
        tool_name, tool_args = tool_request.name, tool_request.arguments
        tool_channel_info = self.tool_channel_info_map[self.tool_name_map[tool_name]]
        channel_response = await self.state_less_tool_call_with_channel(
            tool_channel_info.input_channel,
            self.tool_worker_output_channel,
            tool_name,
            tool_args,
        )

        # no failure in this demo
        assert channel_response.success
        if isinstance(channel_response.result, (list, dict)):
            result_text = json.dumps(channel_response.result)
        else:
            result_text = str(channel_response.result)
        return ToolResponse(
            text=result_text,
        )

    def pre_process(self, prompt_ids: list[int]) -> dict[str, Any]:
        return {"turn": 0}

    async def generate_llm_response(
        self,
        generate_context: dict[str, Any],
        trace_prints,
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
    ):
        turn = generate_context["turn"]
        if turn >= self.cfg.agentloop.maxturn:
            return False, [], [], [], None
        # Generate response from LLM
        max_resp_len = self.max_resp_len - (
            len(turn_prompt_ids) - len(problem_prompt_ids)
        )
        generate_result = await self.generate(
            turn_prompt_ids, sampling_params={"max_new_tokens": max_resp_len}
        )
        llm_response_ids: list[int] = generate_result["output_ids"]

        if len(llm_response_ids) > max_resp_len:
            llm_response_ids = llm_response_ids[:max_resp_len]
        llm_response_text = self.tokenizer.decode(llm_response_ids)

        # split </search> manually
        if "</search>" in llm_response_text:
            llm_response_text = llm_response_text.split("</search>")[0] + "</search>"
            llm_response_ids: list[int] = self.tokenizer.encode(llm_response_text)

        is_continue = len(llm_response_ids) != max_resp_len
        llm_response_mask = [1] * len(llm_response_ids)  # 1 for LLM generated tokens
        return (
            is_continue,
            llm_response_ids,
            llm_response_mask,
            None,
            llm_response_text,
        )

    async def generate_tool_response(
        self,
        generate_context: dict[str, Any],
        trace_prints,
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
        llm_response_ids: list[int],
        llm_response_text: str,
    ):
        # Extract tool calls from response
        _, tool_requests = await self.toolcall_parser(llm_response_text)
        if len(tool_requests) == 0:
            return False, [], [], None

        # Execute tools in parallel with history propagation
        tasks = []
        for tool_request in tool_requests:
            tasks.append(self.tool_call(tool_request))
        tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)

        # Convert tool responses to messages and tokenize
        tool_messages = []
        for tool_response in tool_responses:
            message = {"role": "tool", "content": tool_response.text}
            tool_messages.append(message)

        # Tokenize tool responses
        tool_response_ids: list[int] = self.tokenizer.encode(
            tool_messages[0]["content"], add_special_tokens=False
        )
        max_tool_resp_len = self.max_resp_len - (
            len(turn_prompt_ids) + len(llm_response_ids) - len(problem_prompt_ids)
        )
        if len(tool_response_ids) > max_tool_resp_len:
            return False, [], [], None

        tool_response_mask = [0] * len(tool_response_ids)
        if self.print_outputs:
            # add anything you want to print
            trace_prints.append(
                {
                    "decode_prompt": self.tokenizer.decode(turn_prompt_ids),
                    "generate": llm_response_text,
                    "tool_resp": tool_messages,
                }
            )
        generate_context["turn"] += 1
        return (
            True,
            tool_response_ids,
            tool_response_mask,
            None,
        )
