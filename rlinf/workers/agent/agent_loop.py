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
from typing import Any, Optional
from uuid import uuid4

from omegaconf import DictConfig
from transformers import AutoTokenizer

from rlinf.data.io_struct import (
    DynamicRolloutResult,
    RolloutRequest,
    RolloutResult,
)
from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
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



@dataclass
class MultiTurnAgentLoopOutput:
    """Multi agent loop output."""

    """Single-turn agent loop outputs."""
    single_turn_outputs: list[AgentLoopOutput]
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
        if cfg.runner.task_type == "reasoning_eval":
            self.return_logprobs = False
        else:
            self.return_logprobs = not cfg.algorithm.recompute_logprobs
        self.is_dynamic_rollout_batch = cfg.agentloop.get(
            "is_dynamic_rollout_batch", False
        )
        if self.is_dynamic_rollout_batch:
            assert isinstance(self, MultiTurnAgentLoopWorker), (
                "agent loop worker must be MultiTurnAgentLoopWorker if is_dynamic_rollout_batch is True"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.rollout.model.model_path)

    def init_worker(
        self,
        generate_input_channel: Channel,
        generate_output_channel: Channel,
        tool_channel_info_map: dict[str, ToolChannelInfo],
        tool_name_map: dict[str, str],
        tool_worker_output_channel: Channel,
        solid_generate_input_channels: dict[str, Channel] = {},
    ):
        self.generate_input_channel = generate_input_channel
        self.generate_output_channel = generate_output_channel
        # tool worker name to tool channel info.
        self.tool_channel_info_map = tool_channel_info_map
        # tool name to tool worker. a tool worker may have multiple tools.
        self.tool_name_map = tool_name_map
        self.tool_worker_output_channel = tool_worker_output_channel
        # for calling another llm without training.
        self.solid_generate_input_channels = solid_generate_input_channels

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: Optional[dict] = None,
        rollout_name: Optional[str] = None,
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

    def print_agent_outputs(
        self,
        prompt_texts: Optional[str],
        trace_prints: list[Any],
    ):
        print_texts = []
        if prompt_texts is not None:
            print_texts = [
                f"{green('Prompt')}         : {prompt_texts!r}",
            ]
        for trace_print in trace_prints:
            print_texts.append(f"{green('Trace print')}    : {trace_print!r}")
        print(*print_texts, sep="\n")

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

    async def run_agentloop_rollout_group(
        self,
        input_ids: list[int],
        answer: str,
        group_size: int,
        output_channel: Channel,
    ):
        """
        Run the agent loop for a group of queries.
        """
        rollout_tasks = []
        # grpo group_size
        for _ in range(group_size):
            task = asyncio.create_task(self.run_one_query(copy.deepcopy(input_ids)))
            rollout_tasks.append(task)

        task_results = await asyncio.gather(*rollout_tasks)
        rollout_result = self.get_rollout_result(task_results, answer)
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
            for input_ids, answer in zip(
                rollout_request.input_ids, rollout_request.answers
            ):
                send_output_tasks.append(
                    asyncio.create_task(
                        self.run_agentloop_rollout_group(
                            input_ids, answer, rollout_request.n, output_channel
                        ),
                    )
                )

            await asyncio.gather(*send_output_tasks)

    def get_rollout_result(
        self, task_results: list[AgentLoopOutput], answer: str
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
        max_total_len = int(self.cfg.runner.seq_length)
        max_resp_len = max(1, max_total_len - max_prompt_len)

        prompt_ids = [r.prompt_ids for r in task_results]
        prompt_texts = [r.prompt_text for r in task_results]
        response_ids = [r.response_ids for r in task_results]
        response_texts = [r.response_text for r in task_results]
        prompt_lengths = [len(p) for p in prompt_ids]
        response_lengths = [len(o) for o in response_ids]
        response_mask = None
        if all(r.response_mask is not None for r in task_results):
            response_mask = [r.response_mask[:max_resp_len] for r in task_results]

        # prompt_lengths and response_lengths should be clipped to max_prompt_len and max_resp_len to avoid mask/position size mismatch
        assert max(prompt_lengths) <= max_prompt_len, (
            "prompt_lengths should be clipped to max_prompt_len"
        )
        assert max(response_lengths) <= max_resp_len, (
            "response_lengths should be clipped to max_resp_len"
        )
        response_logprobs = None
        if self.return_logprobs:
            response_logprobs = [
                r.response_logprobs[:max_resp_len] for r in task_results
            ]
        is_end = [True for _ in task_results]
        answers = [answer] * len(task_results)
        return RolloutResult(
            num_sequence=len(task_results),
            group_size=len(task_results),
            prompt_lengths=prompt_lengths,
            prompt_ids=prompt_ids,
            prompt_texts=prompt_texts,
            response_lengths=response_lengths,
            response_ids=response_ids,
            response_texts=response_texts,
            is_end=is_end,
            answers=answers,
            response_mask=response_mask,
            rollout_logprobs=response_logprobs,
        )

    async def run_one_query(self, prompt_ids: list[int], **kwargs) -> AgentLoopOutput:
        raise NotImplementedError("Subclasses must implement this method")


class MultiTurnAgentLoopWorker(AgentLoopWorker):
    """Multi-turn agent loop worker."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.extra_keys_turn = None
        self.extra_keys_traj = None
        self.is_eval = self.cfg.runner.task_type == "reasoning_eval"

    async def run_agentloop_rollout_group(
        self,
        input_dict: dict,
        answer: str,
        group_size: int,
        output_channel: Channel,
    ):
        """
        Run the agent loop for a group of queries.

        Args:
            input_dict: Input dictionary containing 'input_ids' and 'answer'
            answer: Ground truth answer
            group_size: Number of rollouts per query (k samples for pass@k)
            output_channel: Channel to output results
        """
        rollout_tasks: list[asyncio.Task[MultiTurnAgentLoopOutput]] = []
        # grpo group_size
        for _ in range(group_size):
            task = asyncio.create_task(
                self.run_one_query(copy.deepcopy(input_dict), answer=answer)
            )
            rollout_tasks.append(task)

        task_results = await asyncio.gather(*rollout_tasks)

        # For eval mode, allow multiple samples (group_size=k) to compute pass@k and avg@k
        extra_fields = self.gen_extra_fields(task_results, answer)
        if self.is_eval:
            rollout_result = self.get_rollout_result(task_results, *extra_fields, use_no_training=False)
        else:
            rollout_result = self.get_rollout_result(task_results, *extra_fields)
        agent_metrics = self.get_rollout_metrics(rollout_result)

        await output_channel.put(rollout_result, async_op=True).async_wait()
        return agent_metrics

    async def run_agentloop_rollout(
        self, input_channel: Channel, output_channel: Channel,
    ):
        """
        Run the agent loop for multiple queries.

        Args:
            input_channel: Channel to receive rollout requests
            output_channel: Channel to output results
        """
        with self.worker_timer():
            rollout_request: RolloutRequest = input_channel.get()

            send_output_tasks: list[asyncio.Task[dict]] = []
            for input_ids, answer in zip(
                rollout_request.input_ids, rollout_request.answers
            ):
                task = asyncio.create_task(
                    self.run_agentloop_rollout_group(
                        input_ids,
                        answer,
                        rollout_request.n,
                        output_channel,
                    ),
                )
                send_output_tasks.append(task)

            agent_metrics_list = await asyncio.gather(*send_output_tasks)
            agent_metrics = self.post_process_metric(agent_metrics_list)
            return agent_metrics

    def gen_extra_fields(
        self,
        task_results: list[MultiTurnAgentLoopOutput],
        answer: str, 
    ) -> Optional[dict]:
        """Collect extra fields emitted by per-turn and per-trajectory outputs.

        Args:
            task_results: Grouped multi-turn outputs for one query.
            answer: Ground-truth answer metadata for the group.

        Returns:
            Tuple-like structure of turn/traj/group/train extra fields.
        """
        extra_fields_turn = None
        if self.extra_keys_turn is not None:
            extra_fields_turn = {k: list() for k in self.extra_keys_turn}
        extra_fields_traj = None
        if self.extra_keys_traj is not None:
            extra_fields_traj = {k: list() for k in self.extra_keys_traj}

        for task_result in task_results:
            for single_turn_output in task_result.single_turn_outputs:
                single_turn_output: AgentLoopOutput
                if self.extra_keys_turn is not None:
                    for k in self.extra_keys_turn:
                        v = single_turn_output.extra_fields.get(k, None)
                        extra_fields_turn[k].append(v)
            if self.extra_keys_traj is not None:
                for k in self.extra_keys_traj:
                    v = task_result.extra_fields.get(k, None)
                    extra_fields_traj[k].append(v)
        return extra_fields_turn, extra_fields_traj, None, None

    def post_process_metric(self, agent_metrics_list: list[dict]):
        """Merge per-query metrics, including weighted stats across workers.

        Args:
            agent_metrics_list: Metrics returned by each query rollout.

        Returns:
            Aggregated metric dictionary on rank 0, or empty dict on no data.
        """
        if self._world_size > 1:
            if self._rank == 0:
                for i in range(1, self._world_size):
                    recv_list = self.recv(self._group_name, i)
                    agent_metrics_list.extend(recv_list)
            else:
                self.send(agent_metrics_list, self._group_name, 0)
                return
        if len(agent_metrics_list) == 0:
            return {}

        all_keys = set()
        for metric_dict in agent_metrics_list:
            all_keys.update(metric_dict.keys())

        visible_keys = sorted(k for k in all_keys if not k.startswith("__stat/"))
        whole_metrics = {}
        for k in visible_keys:
            sum_key = f"__stat/sum/{k}"
            cnt_key = f"__stat/count/{k}"
            has_weighted_stat = any(
                (sum_key in metric_dict) or (cnt_key in metric_dict)
                for metric_dict in agent_metrics_list
            )

            if has_weighted_stat:
                weighted_sum = sum(
                    float(metric_dict[sum_key])
                    for metric_dict in agent_metrics_list
                )
                weighted_cnt = sum(
                    float(metric_dict[cnt_key])
                    for metric_dict in agent_metrics_list
                )
                whole_metrics[k] = weighted_sum / weighted_cnt if weighted_cnt > 0 else 0.0
                continue

            values_list = [metric_dict[k] for metric_dict in agent_metrics_list if k in metric_dict]
            if len(values_list) == 0:
                continue
            if "/max/" in k:
                whole_metrics[k] = max(values_list)
            else:
                whole_metrics[k] = sum(values_list) / len(values_list)
        return whole_metrics

    def get_rollout_metrics(
        self,
        rollout_result: DynamicRolloutResult,
    ) -> dict:
        """Hook for subclasses to compute task-specific rollout metrics."""
        return {}

    def get_rollout_result(
        self,
        task_results: list[MultiTurnAgentLoopOutput],
        extra_fields_turn: Optional[dict],
        extra_fields_traj: Optional[dict],
        extra_fields_group: Optional[dict],
        extra_fields_train: dict,
        use_no_training=True,
    ) -> DynamicRolloutResult:
        """Collect a group of turn-level outputs into `DynamicRolloutResult`.

        Args:
            task_results: Multi-turn outputs for one query group.
            extra_fields_turn: Turn-level extra fields.
            extra_fields_traj: Trajectory-level extra fields.
            extra_fields_group: Group-level extra fields.
            extra_fields_train: Training-only fields (e.g. regroup indices).
            use_no_training: Whether to drop turns marked as non-training.

        Returns:
            A packed `DynamicRolloutResult` ready for downstream training/eval.
        """
        if self.print_outputs:
            for task_result in task_results:
                if len(task_result.trace_prints) > 0:
                    self.print_agent_outputs(None, task_result.trace_prints)
        self.log_info(f"finish question id {task_results[0].extra_fields['instance_id']}")

        idx_to_traj = []
        prompt_lengths = []
        response_lengths = []
        input_ids = []
        rollout_logprobs = None
        if self.return_logprobs:
            rollout_logprobs = []
        is_end = []
        rewards = []

        # Flatten all retained turns while keeping trajectory mapping.
        for idx, task_result in enumerate(task_results):
            for single_turn_output in task_result.single_turn_outputs:
                single_turn_output: AgentLoopOutput
                if use_no_training and single_turn_output.extra_fields["not_training"] == True:
                    continue
                idx_to_traj.append(idx)
                prompt_lengths.append(len(single_turn_output.prompt_ids))
                response_lengths.append(len(single_turn_output.response_ids))
                input_ids.append(
                    single_turn_output.prompt_ids + single_turn_output.response_ids
                )
                if self.return_logprobs:
                    assert len(single_turn_output.response_logprobs) == len(single_turn_output.response_ids), (
                        "response_logprobs should have the same length as response_ids"
                    )
                    rollout_logprobs.append(single_turn_output.response_logprobs)
                is_end.append(single_turn_output.is_end)
                rewards.append(single_turn_output.reward_score)

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
            extra_fields_turn=extra_fields_turn,
            extra_fields_traj=extra_fields_traj,
            extra_fields_group=extra_fields_group,
            extra_fields_train=extra_fields_train,
        )

    async def run_one_query(self, *args, **kwargs) -> MultiTurnAgentLoopOutput:
        """Run one query and return a multi-turn output (subclass must implement)."""
        raise NotImplementedError("Subclasses must implement this method")
