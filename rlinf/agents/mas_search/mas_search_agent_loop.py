# Copyright 2025 The RLinf Authors.
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
import re
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
    MultiTurnAgentLoopOutput,
    MultiTurnAgentLoopWorker,
)


class MasSearchAgentLoopWorker(MultiTurnAgentLoopWorker):
    """
    Agent loop worker that combines search-r1's <search>keyword</search> extraction
    logic with multiturn_demo's component structure.
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        max_total_len = int(self.cfg.actor.model.encoder_seq_length)
        self.max_resp_len = max(1, max_total_len - self.max_prompt_len)

        # Search tool call tokens
        self.tool_call_start_token: str = "<search>"
        self.tool_call_end_token: str = "</search>"
        self.tool_call_regex = re.compile(r"<search>(.*?)</search>", re.DOTALL)

        # Max turns for multi-turn interaction
        self.max_turns = self.cfg.agentloop.get("max_turns", 5)

        self.return_logprobs = True

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
        """
        Extract tool calls from response text using <search>keyword</search> format.
        """
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

    async def run_one_query(self, prompt_ids: list[int], *, answer) -> AgentLoopOutput:
        output_buffer = []
        orig_prompt_ids = copy.deepcopy(prompt_ids)
        trace_prints = []
        response_mask = []
        all_response_ids = []
        for _ in range(self.max_turns):
            # Generate response from LLM
            max_resp_len = self.max_resp_len - (len(prompt_ids) - len(orig_prompt_ids))

            generate_result = await self.generate(
                prompt_ids, sampling_params={"max_new_tokens": max_resp_len}
            )
            generate_prompt_ids = copy.deepcopy(prompt_ids)
            response_ids = generate_result["output_ids"]
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]
            response_text = self.tokenizer.decode(response_ids)

            # # split </search> manually
            # if "</search>" in response_text:
            #     response_text = response_text.split("</search>")[0] + "</search>"
            #     response_ids = self.tokenizer.encode(response_text)

            output_buffer.append(
                AgentLoopOutput(
                    prompt_ids=copy.deepcopy(prompt_ids),
                    response_ids=copy.deepcopy(response_ids),
                    prompt_text=self.tokenizer.decode(prompt_ids),
                    response_text=response_text,
                    response_mask=response_mask,
                    response_logprobs=generate_result["logprobs"],
                )
            )

            prompt_ids += response_ids
            all_response_ids.extend(response_ids)
            response_mask += [1] * len(response_ids)

            if len(response_ids) == max_resp_len:
                break

            # Extract tool calls from response text
            content, function_calls = await self.extract_tool_calls(response_text)

            if function_calls == []:
                break

            # Execute tools in parallel with history propagation
            tasks = []
            for tool_request in function_calls:
                tasks.append(self.tool_call(tool_request))
            tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)

            # Convert tool responses to messages and tokenize
            tool_messages = []
            for tool_response in tool_responses:
                message = {"role": "tool", "content": tool_response.text}
                tool_messages.append(message)
            tool_response_ids = self.tokenizer.encode(
                tool_messages[0]["content"], add_special_tokens=False
            )
            max_tool_resp_len = self.max_resp_len - (
                len(prompt_ids) - len(orig_prompt_ids)
            )
            if len(tool_response_ids) > max_tool_resp_len:
                break
            prompt_ids += tool_response_ids
            all_response_ids.extend(tool_response_ids)
            response_mask += [0] * len(tool_response_ids)

            if self.print_outputs:
                # add anything you want to print
                trace_prints.append(
                    {
                        "decode_prompt": self.tokenizer.decode(generate_prompt_ids),
                        "generate": response_text,
                        "tool_resp": tool_messages,
                    }
                )
        # Build complete response text from all turns
        complete_response_text = self.tokenizer.decode(all_response_ids)

        # Extract final answer from complete response using searchr1's extract_solution
        from rlinf.algorithms.rewards.searchr1 import compute_score

        reward_score = compute_score(complete_response_text, answer)

        for single_turn_output in output_buffer:
            single_turn_output.reward_score = reward_score

        return MultiTurnAgentLoopOutput(
            single_turn_outputs=output_buffer,
            trace_prints=[],
        )
