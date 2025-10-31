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
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from omegaconf import DictConfig
from transformers import AutoTokenizer

from rlinf.data.io_struct import (
    RolloutRequest,
    RolloutResult,
)
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.tool_worker import ToolChannelInfo
from rlinf.workers.rollout.utils import green


@dataclass
class AgentLoopOutput:
    """Agent loop output."""

    """Prompt token ids."""
    prompt_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_ids: list[int]
    """Prompt text decoded from prompt_ids"""
    prompt_text: str = ""
    """Response text decoded from response_ids"""
    response_text: str = ""
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_mask: list[int] = None
    """Log probabilities for the response tokens."""
    response_logprobs: Optional[list[float]] = None
    """Number of chat turns, including user, assistant, tool."""
    num_turns: int = 0
    """Debug information to print."""
    trace_prints: list[Any] = field(default_factory=list)
    """Extra fields for dynamic addition."""
    extra_fields: dict[str, Any] = field(default_factory=dict)


class AgentLoopWorker(Worker):
    """
    Abstract agent loop worker.

    Subclasses must implement the run_one_query method.
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__()
        self.cfg = cfg
        self.print_outputs = cfg.agentloop.print_outputs

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.rollout.model_dir)

    def init_worker(
        self,
        generate_input_channel: Channel,
        generate_output_channel: Channel,
        tool_channel_info_map: dict[str, ToolChannelInfo],
        tool_name_map: dict[str, str],
        tool_worker_output_channel: Channel,
    ):
        self.generate_input_channel = generate_input_channel
        self.generate_output_channel = generate_output_channel
        # tool worker name to tool channel info.
        self.tool_channel_info_map = tool_channel_info_map
        # tool name to tool worker. a tool worker may have multiple tools.
        self.tool_name_map = tool_name_map
        self.tool_worker_output_channel = tool_worker_output_channel

    async def generate(self, prompt_ids: list[int]):
        channel_key = uuid4().hex
        await self.generate_input_channel.put(
            {"channel_key": channel_key, "prompt_ids": prompt_ids}, async_op=True
        ).async_wait()
        result = await self.generate_output_channel.get(
            channel_key, async_op=True
        ).async_wait()
        return result

    def print_agent_outputs(
        self,
        prompt_texts: str,
        trace_prints: list[Any],
    ):
        print_texts = [
            f"{green('Prompt')}         : {prompt_texts!r}",
        ]
        for trace_print in trace_prints:
            print_texts.append(f"{green('Trace print')}    : {trace_print!r}")
        print(*print_texts, sep="\n")

    async def run_agentloop_rollout_group(
        self,
        input_ids: list[int],
        answers: list[Any],
        group_size: int,
        output_channel: Channel,
    ):
        """
        Run the agent loop for a group of queries.
        """
        rollout_tasks = []
        # grpo group_size
        for _ in range(group_size):
            task = asyncio.create_task(self.run_one_query(input_ids))
            rollout_tasks.append(task)

        task_results = await asyncio.gather(*rollout_tasks)
        rollout_result = self.get_rollout_result(task_results, answers)
        await output_channel.put(rollout_result, async_op=True).async_wait()

    async def run_agentloop_rollout(
        self, input_channel: Channel, output_channel: Channel
    ):
        """
        Run the agent loop for multiple queries.
        """
        with self.worker_timer():
            rollout_request: RolloutRequest = input_channel.get()

            send_output_tasks = []
            for input_ids, answers in zip(
                rollout_request.input_ids, rollout_request.answers
            ):
                send_output_tasks.append(
                    asyncio.create_task(
                        self.run_agentloop_rollout_group(
                            input_ids, answers, rollout_request.n, output_channel
                        ),
                    )
                )

            await asyncio.gather(*send_output_tasks)

    def get_rollout_result(
        self, task_results: list[AgentLoopOutput], answers
    ) -> RolloutResult:
        """
        Collect group task results into a RolloutResult.
        """
        if self.print_outputs:
            for task_result in task_results:
                if len(task_result.trace_prints) > 0:
                    self.print_agent_outputs(
                        task_result.prompt_text, task_result.trace_prints
                    )
        # Clip to model limits to avoid mask/position size mismatch
        max_prompt_len = int(self.cfg.data.max_prompt_length)
        max_total_len = int(self.cfg.actor.model.encoder_seq_length)
        max_resp_len = max(1, max_total_len - max_prompt_len)

        prompt_ids = [r.prompt_ids[:max_prompt_len] for r in task_results]
        response_ids = [r.response_ids[:max_resp_len] for r in task_results]
        prompt_lengths = [len(p) for p in prompt_ids]
        response_lengths = [len(o) for o in response_ids]
        response_mask = [r.response_mask[:max_resp_len] for r in task_results]
        is_end = [True for _ in task_results]
        answers = [answers] * len(task_results)
        return RolloutResult(
            num_sequence=len(task_results),
            group_size=len(task_results),
            prompt_lengths=prompt_lengths,
            prompt_ids=prompt_ids,
            response_lengths=response_lengths,
            response_ids=response_ids,
            is_end=is_end,
            answers=answers,
            response_mask=response_mask,
        )

    async def run_one_query(self, prompt_ids: list[int], **kwargs) -> AgentLoopOutput:
        raise NotImplementedError("Subclasses must implement this method")
