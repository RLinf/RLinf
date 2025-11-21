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
import random
import re
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
from rlinf.workers.agent.agent_loop import MultiTurnAgentLoopWorker, AgentLoopOutput, MultiTurnAgentLoopOutput

class MultiRoleAgentLoopWorker(MultiTurnAgentLoopWorker):
    """Simple tool agent loop that can interact with tools."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        max_total_len = int(self.cfg.actor.model.encoder_seq_length)
        self.max_resp_len = max(1, max_total_len - self.max_prompt_len)

        self.think_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.answer_regex = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

        # 5 is a magic number in this demo.
        self.max_planner_turns = self.cfg.agentloop.get("max_planner_turns", 5)
        self.max_worker_turns = self.cfg.agentloop.get("max_worker_turns", 5)

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

    async def extract_tool_calls(self, response_text) -> tuple[str, list[ToolRequest]]:
        # random tool call
        return_function_calls = random.choice(
            [
                [ToolRequest(name="fake_tool", arguments={"arg1": "value1"})],
                [ToolRequest(name="fake_tool", arguments={"arg2": "value2"})],
                [], # no tool calls
            ]
        )

        return response_text, return_function_calls

    async def worker_call(self, worker_request: ToolRequest, worker_system_prompts: dict[str, str]) -> ToolResponse:
        worker_name, worker_args = worker_request.name, worker_request.arguments
        if worker_name not in worker_system_prompts:
            output_buffer, trace_prints = [], []
            final_answer = f"Invalid worker role: {worker_name}, should be one of {worker_system_prompts.keys()}"
            return output_buffer, trace_prints, final_answer
        if 'task_prompt' not in worker_request.arguments:
            output_buffer, trace_prints = [], []
            final_answer = f"Invalid worker arguments: {worker_request.arguments}, should contain 'task_prompt'"
            return output_buffer, trace_prints, final_answer
        task_prompt = worker_args['task_prompt']
        output_buffer, trace_prints, final_answer = await self.run_one_query_role(worker_system_prompts[worker_name], task_prompt, worker_name, self.max_worker_turns, self.tool_call)
        return output_buffer, trace_prints, final_answer

    async def extract_worker_calls(self, response_text) -> tuple[str, list[ToolRequest]]:
        # random tool call
        return_worker_calls = random.choice(
            [
                [ToolRequest(name="worker1", arguments={"task_prompt": "value1"})],
                [ToolRequest(name="worker2", arguments={"task_prompt": "value2"})],
                [], # no worker calls
            ]
        )

        return response_text, return_worker_calls

    async def run_one_query_role(self, system_prompt: str, task_prompt: list[int], role_name: str, max_turns: int, tool_call_fn: callable) -> tuple[list[AgentLoopOutput], list[Any], str]:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task_prompt}]
        prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        trace_prints = []

        output_buffer = []
        final_answer = ""
        for _ in range(max_turns):
            # Generate response from LLM
            max_resp_len = self.max_resp_len - len(prompt_ids)
            generate_result = await self.generate(
                prompt_ids, sampling_params={"max_new_tokens": max_resp_len}
            )
            response_ids = generate_result["output_ids"]
            assert generate_result["logprobs"] is not None
            output_buffer.append(AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                is_end=generate_result["finish_reason"] == "length",
                response_logprobs=generate_result["logprobs"],
                extra_fields=dict(
                    role=role_name,
                ),
            ))
            if generate_result["finish_reason"] == "length":
                final_answer = None
                break
            response_text = self.tokenizer.decode(response_ids)
            if self.print_outputs:
                # add anything you want to print
                trace_prints.append({"messages": messages, f"{role_name}_generate": response_text})
            response_text = self.think_regex.sub("", response_text)
            response_ids = self.tokenizer.encode(response_text)
            prompt_ids += response_ids

            if self.answer_regex.search(response_text):
                final_answer = self.answer_regex.search(response_text).group(1)
                break

            tool_messages = []
            if role_name == "planner":
                # Extract worker calls from response
                _, worker_requests = await self.extract_worker_calls(response_text)

                # Execute workers in parallel with history propagation
                tasks = []
                for worker_request in worker_requests:
                    tasks.append(tool_call_fn(worker_request))
                worker_responses: list = await asyncio.gather(*tasks)

                # Convert tool responses to messages and tokenize
                for worker_response in worker_responses:
                    worker_output_buffer, worker_trace_prints, worker_final_answer = worker_response
                    output_buffer.extend(worker_output_buffer)
                    trace_prints.extend(worker_trace_prints)
                    message = {"role": "tool", "content": worker_final_answer}
                    tool_messages.append(message)
            else:
                # Extract tool calls from response
                _, tool_requests = await self.extract_tool_calls(response_text)

                # Execute tools in parallel with history propagation
                tasks = []
                for tool_request in tool_requests:
                    tasks.append(tool_call_fn(tool_request))
                tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)

                # Convert tool responses to messages and tokenize
                for tool_response in tool_responses:
                    message = {"role": "tool", "content": tool_response.text}
                    tool_messages.append(message)

            # Tokenize tool responses
            tool_response_ids = self.get_tool_response_ids(tool_messages)
            max_tool_resp_len = self.max_resp_len - len(prompt_ids)
            if len(tool_response_ids) > max_tool_resp_len:
                final_answer = None
                break

            prompt_ids += tool_response_ids
            if self.print_outputs:
                # add anything you want to print
                trace_prints[-1]["tool_resp"] = tool_messages

        return output_buffer, trace_prints, final_answer

    async def run_one_query(self, input_prompts: dict[str, Any]):
        # planner_system_prompt: str = input_prompts["planner_system_prompt"]
        # worker_system_prompts: dict[str, str] = input_prompts["worker_system_prompts"]
        # task_prompt: str = input_prompts["task_prompt"]

        planner_system_prompt = """you are a planner. sub workers are tools."""
        worker_system_prompts = {
            "worker1": "you are a worker1. you are a worker to solve the problem, and you can call tools to help you.",
            "worker2": "you are a worker2. you are a worker to solve the problem, and you can call tools to help you.",
        }
        task_prompt = "please write a program to solve the problem. the problem is: 1 + 1 = ?"
        worker_call_fn = lambda worker_request: self.worker_call(worker_request, worker_system_prompts)
        output_buffer, trace_prints, final_answer = await self.run_one_query_role(planner_system_prompt, task_prompt, "planner", self.max_planner_turns, worker_call_fn)

        for single_turn_output in output_buffer:
            single_turn_output.reward_score = 0.0

        return MultiTurnAgentLoopOutput(
            single_turn_outputs=output_buffer,
            trace_prints=trace_prints,
            extra_fields=dict(
                final_answer=final_answer,
            ),
        )
