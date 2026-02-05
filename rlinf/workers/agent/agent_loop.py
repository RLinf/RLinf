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
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from omegaconf import DictConfig
from transformers import AutoTokenizer

from rlinf.data.io_struct import (
    RolloutRequest,
    DynamicRolloutResult,
)
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.tool_worker import ToolChannelInfo
from rlinf.workers.rollout.utils import green


from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
)
import json
import pandas as pd
import traceback
import re
import numpy as np



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
    response_mask: Optional[list[int]] = None
    """Log probabilities for the response tokens."""
    response_logprobs: Optional[list[float]] = None
    """Number of chat turns, including user, assistant, tool."""
    num_turns: int = 0
    """Whether the sequence ends."""
    is_end: bool = False
    """Reward score for the trajectory."""
    reward_score: Optional[float] = None
    """Debug information to print."""
    trace_prints: list[Any] = field(default_factory=list)
    """Extra fields for dynamic addition."""
    extra_fields: dict[str, Any] = field(default_factory=dict)
    """Tool call information for this turn."""
    tool_call_info: Optional[dict[str, int]] = None
    """task failed"""
    context_failed: bool = False
    turn_repeat_failed: bool = False
    max_turn_limit_failed: bool = False


@dataclass
class MultiTurnAgentLoopOutput:
    """Multi agent loop output."""

    """Single-turn agent loop outputs."""
    single_turn_outputs: list[AgentLoopOutput]
    """Single-turn agent loop outputs that used for training"""
    train_buffer: list[AgentLoopOutput]    
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
        self.return_logprobs = not cfg.algorithm.recompute_logprobs
        assert self.return_logprobs
        self.is_dynamic_rollout_batch = cfg.agentloop.get("is_dynamic_rollout_batch", False)
        if self.is_dynamic_rollout_batch:
            assert isinstance(self, MultiTurnAgentLoopWorker), "agent loop worker must be MultiTurnAgentLoopWorker if is_dynamic_rollout_batch is True"
            # assert not self.return_logprobs, "recompute_logprobs must be False if is_dynamic_rollout_batch is True"

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.rollout.model.model_path)

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

    async def state_less_tool_call_with_channel(
        self,
        input_channel: Channel,
        output_channel: Channel,
        tool_name: str,
        tool_args: dict,
    ) -> ToolChannelResponse:
        """Execute stateless tool call via channel."""
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
        """Execute a tool call (search or access)."""
        tool_name, tool_args = tool_request.name, tool_request.arguments
        tool_channel_info = self.tool_channel_info_map[self.tool_name_map[tool_name]]
        channel_response = await self.state_less_tool_call_with_channel(
            tool_channel_info.input_channel,
            self.tool_worker_output_channel,
            tool_name,
            tool_args,
        )

        # No failure in this demo
        if isinstance(channel_response.result, (list, dict)):
            result_text = json.dumps(channel_response.result)
        else:
            result_text = str(channel_response.result)
        return ToolResponse(text=result_text)

    def get_tool_response_ids(self, tool_messages: list[dict]):
        """
        To append correct tool response ids.
        For some agents use custom chat template and special tokens, you should use custom method to override it.
        """
        wo_messages = [{"role": "user", "content": "hi"}]
        wi_messages = [*wo_messages, *tool_messages]
        wo_ids = self.tokenizer.apply_chat_template(
            wo_messages, add_generation_prompt=False, tokenize=True
        )
        wi_ids = self.tokenizer.apply_chat_template(
            wi_messages, add_generation_prompt=True, tokenize=True
        )
        return wi_ids[len(wo_ids) :]

class MultiTurnAgentLoopWorker(AgentLoopWorker):
    """Multi-turn agent loop worker."""

    def init_worker(
        self,
        generate_input_channel: Channel,
        generate_output_channel: Channel,
        tool_channel_info_map: dict[str, ToolChannelInfo],
        tool_name_map: dict[str, str],
        tool_worker_output_channel: Channel,
        solid_generate_input_channels: dict[str, Channel] = {},
    ):
        super().init_worker(generate_input_channel, generate_output_channel, tool_channel_info_map, tool_name_map, tool_worker_output_channel)
        self.solid_generate_input_channels = solid_generate_input_channels

    async def generate(
        self, prompt_ids: list[int], sampling_params: Optional[dict] = None, rollout_name: str = None
    ):
        channel_key = uuid4().hex
        if rollout_name is None:
            input_channel = self.generate_input_channel
        else:
            input_channel = self.solid_generate_input_channels[rollout_name]
        await input_channel.put(
            {
                "channel_key": channel_key,
                "prompt_ids": prompt_ids,
                "sampling_params": sampling_params,
            },
            async_op=True,
        ).async_wait()
        result = await self.generate_output_channel.get(
            channel_key, async_op=True
        ).async_wait()
        return result
    
    async def run_agentloop_rollout_group(
        self,
        input_dict: dict,
        answer: str,
        group_size: int,
        output_channel: Channel,
        is_eval: bool = False,
    ):
        """
        Run the agent loop for a group of queries.

        Args:
            input_dict: Input dictionary containing 'input_ids' and 'answer'
            answer: Ground truth answer
            group_size: Number of rollouts per query (k samples for pass@k)
            output_channel: Channel to output results
            is_eval: If True, output evaluation-friendly format instead of DynamicRolloutResult
        """
        rollout_tasks = []
        # grpo group_size
        for _ in range(group_size):
            task = asyncio.create_task(self.run_one_query(copy.deepcopy(input_dict), answer=answer))
            rollout_tasks.append(task)

        task_results = await asyncio.gather(*rollout_tasks)

        # For eval mode, allow multiple samples (group_size=k) to compute pass@k and avg@k
        if is_eval:
            rollout_result = self.get_rollout_result_eval(task_results, answer)
        else:
            rollout_result = self.get_rollout_result(task_results, answer)

        await output_channel.put(rollout_result, async_op=True).async_wait()

    async def run_agentloop_rollout(
        self, input_channel: Channel, output_channel: Channel, is_eval: bool = False
    ):
        """
        Run the agent loop for multiple queries.

        Args:
            input_channel: Channel to receive rollout requests
            output_channel: Channel to output results
            is_eval: If True, output evaluation-friendly format
        """
        with self.worker_timer():
            rollout_request: RolloutRequest = input_channel.get()

            send_output_tasks = []
            for input_ids, answer in zip(
                rollout_request.input_ids, rollout_request.answers
            ):
                send_output_tasks.append(
                    asyncio.create_task(
                        self.run_agentloop_rollout_group(
                            input_ids, answer, rollout_request.n, output_channel, is_eval=is_eval
                        ),
                    )
                )

            await asyncio.gather(*send_output_tasks)

    def get_rollout_result(
        self, task_results: list[MultiTurnAgentLoopOutput], answer: str
    ) -> DynamicRolloutResult:
        """
        Collect group task results into a DynamicRolloutResult.
        """
        if self.print_outputs:
            for task_result in task_results:
                if len(task_result.trace_prints) > 0:
                    self.print_agent_outputs(None, task_result.trace_prints)
        idx_to_traj = []
        prompt_lengths = []
        response_lengths = []
        input_ids = []
        rollout_logprobs = []
        is_end = []
        rewards = []

        # Collect tool call counts per turn
        turn_subtask_counts = []
        turn_search_counts = []
        turn_access_counts = []

        # Track valid turns for computing averages
        num_valid_planner_turns = 0
        num_valid_worker_turns = 0

        # Collect eval_metrics per trajectory
        eval_metrics = []
        total_turn_list_metric = []
        roles = []
        role_group_sizes = []
        role_group_size = 0

        for idx, task_result in enumerate(task_results):
            # Extract eval_metric from extra_fields for this trajectory
            eval_metric = task_result.extra_fields.get("eval_metric", None)
            total_turn_list = task_result.extra_fields.get("total_turn_list", None)
            eval_metrics.append(eval_metric)
            total_turn_list_metric.append(total_turn_list)
            have_role = False
            for single_turn_output in task_result.train_buffer:
                single_turn_output: AgentLoopOutput
                idx_to_traj.append(idx)
                prompt_lengths.append(len(single_turn_output.prompt_ids))
                response_lengths.append(len(single_turn_output.response_ids))
                input_ids.append(single_turn_output.prompt_ids + single_turn_output.response_ids)
                rollout_logprobs.append(single_turn_output.response_logprobs)
                is_end.append(single_turn_output.is_end)
                rewards.append(single_turn_output.reward_score)
                role = single_turn_output.extra_fields['role']
                roles.append(role)
                if self.train_roles and role in self.train_roles:
                    have_role = True
            
            if have_role:
                role_group_size += 1

            for single_turn_output in task_result.single_turn_outputs:
                # Collect tool call info (keep all turns but track valid ones)
                if single_turn_output.tool_call_info is not None:
                    role = single_turn_output.tool_call_info.get("role", "")
                    subtask_count = single_turn_output.tool_call_info.get("subtask", 0)
                    search_count = single_turn_output.tool_call_info.get("search", 0)
                    access_count = single_turn_output.tool_call_info.get("access", 0)

                    turn_subtask_counts.append(subtask_count)
                    turn_search_counts.append(search_count)
                    turn_access_counts.append(access_count)

                    # Track valid turns by role
                    if role == "planner":
                        assert subtask_count > 0
                        num_valid_planner_turns += 1
                    elif role == "worker" or role == "single":
                        assert (search_count > 0 or access_count > 0)
                        num_valid_worker_turns += 1
                else:
                    # Invalid turn - pad with zeros
                    turn_subtask_counts.append(0)
                    turn_search_counts.append(0)
                    turn_access_counts.append(0)

        role_group_sizes = [role_group_size] * len(idx_to_traj)
        
        return DynamicRolloutResult(
            num_sequence=len(idx_to_traj),
            group_size=len(task_results),
            idx_to_traj=idx_to_traj,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            input_ids=input_ids,
            rollout_logprobs=rollout_logprobs,
            is_end=is_end,
            rewards=rewards,
            roles=roles,
            role_group_sizes=role_group_sizes,
            turn_subtask_counts=turn_subtask_counts,
            turn_search_counts=turn_search_counts,
            turn_access_counts=turn_access_counts,
            num_valid_planner_turns=num_valid_planner_turns,
            num_valid_worker_turns=num_valid_worker_turns,
            eval_metrics=eval_metrics,
            total_turn_list_metric=total_turn_list_metric,
        )

    def get_rollout_result_eval(
        self, task_results: list[MultiTurnAgentLoopOutput], answer: str
    ) -> dict:
        """
        Collect task results into an evaluation-friendly dictionary format.

        Args:
            task_results: List of MultiTurnAgentLoopOutput from multiple samples (k samples for pass@k)
            answer: Ground truth answer

        Returns:
            Dictionary with human-readable evaluation data including:
            - Individual sample data (k samples)
            - All metrics computed per sample (EM, F1, LLM)
        """
        group_size = len(task_results)

        # Collect data for all k samples
        samples_data = []
        for idx, task_result in enumerate(task_results):
            if self.print_outputs:
                if len(task_result.trace_prints) > 0:
                    self.print_agent_outputs(None, task_result.trace_prints)

            # Collect all turns with readable text for this sample
            turns = []
            for single_turn_output in task_result.single_turn_outputs:
                single_turn_output: AgentLoopOutput
                turn_data = {
                    "prompt_text": single_turn_output.prompt_text,
                    "response_text": single_turn_output.response_text,
                    "prompt_ids_length": len(single_turn_output.prompt_ids),
                    "response_ids_length": len(single_turn_output.response_ids),
                    "is_end": single_turn_output.is_end,
                    "reward_score": single_turn_output.reward_score,
                    "role": single_turn_output.extra_fields.get("role", "unknown"),
                    "tool_call_info": single_turn_output.tool_call_info,
                }
                turns.append(turn_data)

            # Extract evaluation metrics for this sample
            eval_metric = task_result.extra_fields.get("eval_metric", {})

            final_answer = task_result.extra_fields.get("final_answer", None)
            if final_answer is not None:
                if isinstance(final_answer, pd.DataFrame):
                    final_answer = final_answer.to_dict(orient='records')

            # Extract total_turn_list for MAS workflow
            total_turn_list = task_result.extra_fields.get("total_turn_list", None)

            sample_data = {
                "sample_idx": idx,
                "num_turns": len(turns),
                "turns": turns,
                "origin_question": task_result.extra_fields.get("origin_question", None),
                "final_answer": final_answer,
                "final_answer_text": task_result.extra_fields.get("final_answer_text", None),
                "planner_summary": task_result.extra_fields.get("planner_summary", None),
                # Evaluation metrics for this sample
                "eval_metric": eval_metric,
                # Total turn list for MAS: last element is main agent turns, others are subagent turns
                "total_turn_list": total_turn_list,
            }
            samples_data.append(sample_data)

        # Create evaluation result dictionary with all k samples
        eval_result = {
            "group_size": group_size,
            "answer": answer,
            "samples": samples_data,  # All k samples with their metrics
        }
        self.log_info(f"finish question id {answer['instance_id']}")
        return eval_result

    async def run_one_query(self, *args, **kwargs) -> MultiTurnAgentLoopOutput:
        raise NotImplementedError("Subclasses must implement this method")
