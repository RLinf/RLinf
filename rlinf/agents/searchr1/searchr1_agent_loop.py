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
from typing import Any

from omegaconf import DictConfig

from rlinf.algorithms.rewards.searchr1 import compute_score
from rlinf.data.tool_call.tool_io_struct import ToolResponse
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)


class Searchr1AgentLoopWorker(MultiAgentLoopWorker):
    """
    Search-R1 agent loop
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

    async def pre_process_query(
        self, prompt_ids: list[int], answer: str
    ) -> tuple[list[int], dict[str, Any]]:
        return (
            prompt_ids[: self.max_prompt_len],
            {
                "answer": answer,
                "turn": 0,
                "all_llm_response_ids": [],  # accumulate only LLM-generated tokens for reward
            },
        )

    async def post_process_query(
        self, generate_context: dict[str, Any], output: MultiAgentLoopOutput
    ) -> MultiAgentLoopOutput:
        # Compute reward from all LLM-generated tokens (excluding tool responses)
        final_response_text = self.tokenizer.decode(
            generate_context["all_llm_response_ids"]
        )
        reward_score = compute_score(
            final_response_text, generate_context["answer"], do_print=False
        )

        for single_turn_output in output.single_turn_outputs:
            single_turn_output.reward_score = reward_score

        # Store trajectory-level info for eval
        output.extra_fields["llm_reward"] = reward_score
        output.extra_fields["response_text"] = final_response_text
        output.extra_fields["prompt_text"] = self.tokenizer.decode(
            generate_context.get("problem_prompt_ids", [])
        )
        # Per-turn details: each turn's input and output text
        turns = []
        for single_turn_output in output.single_turn_outputs:
            turns.append(
                {
                    "input": self.tokenizer.decode(single_turn_output.prompt_ids),
                    "output": self.tokenizer.decode(single_turn_output.response_ids),
                }
            )
        output.extra_fields["turns"] = turns

        return output

    async def generate_llm_response(
        self,
        generate_context: dict[str, Any],
        trace_prints: list[dict],
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
    ):
        llm_output = None

        if generate_context["turn"] >= self.cfg.agentloop.max_turns:
            return False, None, None, llm_output

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
            llm_response_ids = self.tokenizer.encode(llm_response_text)

        llm_output = AgentLoopOutput(
            prompt_ids=copy.deepcopy(turn_prompt_ids),
            response_ids=llm_response_ids,
        )
        generate_context["all_llm_response_ids"] += llm_response_ids

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
        _, tool_requests = await self.toolcall_parser(llm_response_text)
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
            len(turn_prompt_ids) + len(llm_response_ids) - len(problem_prompt_ids)
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

    def gen_extra_fields(self, task_results, answer):
        if self.is_eval:
            # Eval: store per-traj reward/text/turns and group-level answer
            llm_rewards = []
            response_texts = []
            prompt_texts = []
            turns_list = []
            for task_result in task_results:
                llm_rewards.append(task_result.extra_fields.get("llm_reward", 0.0))
                response_texts.append(task_result.extra_fields.get("response_text", ""))
                prompt_texts.append(task_result.extra_fields.get("prompt_text", ""))
                turns_list.append(task_result.extra_fields.get("turns", []))
            extra_fields_traj = {
                "llm_reward": llm_rewards,
                "response_text": response_texts,
                "prompt_text": prompt_texts,
                "turns": turns_list,
            }
            extra_fields_group = {"answer": answer}
            return None, extra_fields_traj, extra_fields_group, {}

        # Training: idx_to_sub_traj for loss scaling (single agent, all 0s)
        idx_to_sub_traj = []
        for task_result in task_results:
            for _ in task_result.single_turn_outputs:
                idx_to_sub_traj.append(0)
        extra_fields_train = {"idx_to_sub_traj": idx_to_sub_traj}
        return None, None, None, extra_fields_train

    async def run_one_query(self, *args, **kwargs) -> MultiAgentLoopOutput:
        prompt_ids, generate_context = await self.pre_process_query(*args, **kwargs)
        problem_prompt_ids = copy.deepcopy(prompt_ids)
        generate_context["problem_prompt_ids"] = problem_prompt_ids
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
