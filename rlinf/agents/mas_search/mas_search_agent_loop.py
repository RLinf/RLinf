# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
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
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)
from rlinf.algorithms.rewards.searchr1 import compute_score
from rlinf.algorithms.rewards import get_reward_class

class MasSearchAgentLoopWorker(MultiAgentLoopWorker):
    """
    Agent loop worker that combines search-r1's <search>keyword</search> extraction
    logic with multi agent system's component structure.
    """

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
        self.reward = get_reward_class(self.cfg.reward.reward_type)(cfg.reward)

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

    async def pre_process_query(self, prompt_ids: list[int], answer: str) -> tuple[list[int], dict[str, Any]]:
        return prompt_ids[: self.max_prompt_len], {"answer": answer, "turn": 0}

    async def post_process_query(
        self, generate_context: dict[str, Any], output: AgentLoopOutput
    ) -> dict[str, Any]:
        # Extract final answer from complete response using searchr1's extract_solution
        final_response_ids = generate_context["final_response_ids"]
        final_response_text = self.tokenizer.decode(final_response_ids)
        answer = generate_context["answer"]
        reward_score = compute_score(final_response_text, answer)
        reward_score2 = self.reward.get_reward([final_response_text], [answer])[0]
        assert reward_score == reward_score2, "Reward scores do not match"

        for single_turn_output in output.single_turn_outputs:
            single_turn_output.reward_score = reward_score

        return output

    async def generate_llm_response(
        self,
        generate_context: dict[str, Any],
        trace_prints: list[dict],
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
    ):
        llm_output = None
        generate_context["final_response_ids"] = turn_prompt_ids[len(problem_prompt_ids):]

        if generate_context["turn"] >= self.cfg.agentloop.maxturn:
            return False, None, None, llm_output

        # Generate response from LLM
        max_resp_len = self.max_resp_len - (len(turn_prompt_ids) - len(problem_prompt_ids))

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
            llm_response_ids = self.tokenizer.encode(llm_response_text)

        llm_output = AgentLoopOutput(
            prompt_ids=copy.deepcopy(turn_prompt_ids),
            response_ids=llm_response_ids,
        )
        generate_context["final_response_ids"] = turn_prompt_ids[len(problem_prompt_ids):] + llm_response_ids

        if len(llm_response_ids) == max_resp_len:
            return False, None, None, llm_output

        return True, llm_response_ids, llm_response_text, llm_output

    async def generate_tool_response(
        self,
        generate_context: dict[str, Any],
        trace_prints: list[dict],
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
        llm_response_ids,
        llm_response_text,
    ):
        # Extract tool calls from response
        _, tool_requests = await self.extract_tool_calls(llm_response_text)
        if tool_requests == []:
            return False, None

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
            len(turn_prompt_ids) - len(problem_prompt_ids)
        )
        if len(tool_response_ids) > max_tool_resp_len:
            return False, None

        next_turn_prompt_ids = turn_prompt_ids + llm_response_ids + tool_response_ids
        if self.print_outputs:
            # add anything you want to print
            trace_prints.append(
                {
                    "prompt": self.tokenizer.decode(turn_prompt_ids),
                    "generate": llm_response_text,
                    "tool_resp": tool_messages,
                }
            )
        generate_context["turn"] += 1
        return True, next_turn_prompt_ids


    async def run_one_query_turn(
        self,
        output_buffer: list[AgentLoopOutput],
        generate_context: dict[str, Any],
        trace_prints: list[dict],
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
    ):
        (
            is_continue,
            llm_response_ids,
            llm_response_text,
            llm_output,
        ) = await self.generate_llm_response(
            generate_context,
            trace_prints,
            problem_prompt_ids,
            turn_prompt_ids,
        )

        if llm_output is not None:
            output_buffer.append(llm_output)

        if not is_continue:
            return False, None

        (
            is_continue,
            next_turn_prompt_ids,
        ) = await self.generate_tool_response(
            generate_context,
            trace_prints,
            problem_prompt_ids,
            turn_prompt_ids,
            llm_response_ids,
            llm_response_text,
        )

        return is_continue, next_turn_prompt_ids

    async def run_one_query(self, *args, **kwargs) -> MultiAgentLoopOutput:
        prompt_ids, generate_context = await self.pre_process_query(*args)
        problem_prompt_ids = copy.deepcopy(prompt_ids)
        output_buffer = []
        trace_prints = []
        while True:
            (
                is_continue,
                prompt_ids,
            ) = await self.run_one_query_turn(
                output_buffer,
                generate_context,
                trace_prints,
                problem_prompt_ids,
                prompt_ids,
            )
            if not is_continue:
                break

        output = MultiAgentLoopOutput(
            single_turn_outputs=output_buffer,
            trace_prints=trace_prints,
        )

        return await self.post_process_query(generate_context, output)

