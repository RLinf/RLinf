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

import pandas as pd
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
        self, prompt_ids: list[int], sampling_params: Optional[dict] = None
    ):
        channel_key = uuid4().hex
        await self.generate_input_channel.put(
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
        extra_fields_group = self.gen_extra_fields_group(task_results, answer)
        agent_metrics = None
        rollout_result_train = self.get_rollout_result(task_results, extra_fields_group)
        rollout_result_eval = self.get_rollout_result(task_results, extra_fields_group, use_no_training=False)
        import dataclasses
        import torch
        def to_obj(obj):
            if dataclasses.is_dataclass(obj):
                return {
                    '__cls__': obj.__class__.__name__,
                    **to_obj(dataclasses.asdict(obj)),
                }
            if isinstance(obj, dict):
                return {k: to_obj(v) for k, v in obj.items()}
            if isinstance(obj, tuple):
                return tuple((to_obj(i) for i in obj))
            if isinstance(obj, list):
                return list((to_obj(i) for i in obj))
            if isinstance(obj, (int, float, str, bool, torch.Tensor, type(None))):
                return obj
            if isinstance(obj, (pd.DataFrame)):
                return obj.to_numpy().tolist()
            else:
                assert False, f"zcy_dbg: bad_type: {obj.__class__.__name__}"

        def diff_keys(obj_a, obj_b):
            dict_obj_a = to_obj(obj_a)
            dict_obj_b = to_obj(obj_b)
            min_key =  set(dict_obj_a.keys()) & set(dict_obj_b.keys())
            max_key =  set(dict_obj_a.keys()) | set(dict_obj_b.keys())
            result_min = {
                k: [dict_obj_a[k], dict_obj_b[k]]
                for k in min_key
                    if dict_obj_a[k] != dict_obj_b[k]
            }
            result_max = {
                k: None
                for k in (max_key - min_key)
            }
            return {**result_min, **result_max}

        def diff_idx(obj_a, obj_b):
            list_obj_a = to_obj(obj_a)
            list_obj_b = to_obj(obj_b)
            min_len = min(len(list_obj_a), len(list_obj_b))
            max_len = max(len(list_obj_a), len(list_obj_b))
            result_min = {
                i: [list_obj_a[i], list_obj_b[i]]
                for i in range(min_len) if list_obj_a[i] != list_obj_b[i]
            }
            result_max = {
                i: None
                for i in range(min_len, max_len)
            }
            return {**result_min, **result_max}

        try:
            """rollout_result"""
            rollout_result_old = self.get_rollout_result_old(task_results, extra_fields_group)
            # rollout_result_train = get_rollout_result(self, task_results, extra_fields_group)
            # rollout_result_old = get_rollout_result_old(self, task_results, extra_fields_group)
            # rollout_result_old = rollout_result = rollout_result
            assert to_obj(rollout_result_old) == to_obj(rollout_result_train)

            """eval_result"""
            from rlinf.agents.wideseek_r1.eval_runner import WideSeekR1AgentEvalRunner
            eval_result = WideSeekR1AgentEvalRunner.extract_eval_result(None, rollout_result_eval)

            """eval_result 1"""
            eval_result_old = self.get_rollout_result_eval_old(task_results, answer)
            # eval_result = WideSeekR1AgentEvalRunner.extract_eval_result(None, self.get_rollout_result_old(task_results, extra_fields_group))
            # eval_result = WideSeekR1AgentEvalRunner.extract_eval_result(None, get_rollout_result(self, task_results, extra_fields_group, False)[0])
            # eval_result_old = self.get_rollout_result_eval_old(task_results, answer)
            # eval_result_old = get_rollout_result_eval_old(self, task_results, answer)
            assert to_obj(eval_result_old) == to_obj(eval_result)

            """eval_result 2"""
            eval_result_new = MultiTurnAgentLoopWorker.get_rollout_result_eval(rollout_result_eval)
            # eval_result_new = get_rollout_result_eval(rollout_result_eval)
            assert to_obj(eval_result_old) == to_obj(eval_result_new)

            print("[run_agentloop_rollout_group] all assert passsed!")
        except Exception as e:
            print("zcy_dbg: assert False, goto breakpoint")
            breakpoint()
            not_equal_k = {
                k: [[to_obj(rollout_result_old)[k], to_obj(rollout_result)[k]]]
                for k in set([*to_obj(rollout_result_old).keys(), *to_obj(rollout_result).keys()])
                    if to_obj(rollout_result_old)[k] != to_obj(rollout_result)[k]
            }

            not_equal_k = {
                k: [[to_obj(eval_result_old)[k], to_obj(eval_result)[k]]]
                for k in set([*to_obj(eval_result_old), *to_obj(eval_result)])
                    if to_obj(eval_result_old)[k] != to_obj(eval_result)[k]
            }
            # to_obj(eval_result_old)["answer"] == to_obj(eval_result)["answer"]
            to_obj(eval_result_old)["samples"] == to_obj(eval_result)["samples"]
            # len(to_obj(eval_result_old)["samples"]) == len(to_obj(eval_result)["samples"])
            not_equal_idx = {
                i: [to_obj(eval_result_old)["samples"][i], to_obj(eval_result)["samples"][i]]
                for i in range(len(to_obj(eval_result_old)["samples"]))
                    if to_obj(eval_result_old)["samples"][i] != to_obj(eval_result)["samples"][i]
            }
            sample_idx = 0
            to_obj(eval_result_old)["samples"][sample_idx] == to_obj(eval_result)["samples"][sample_idx]
            not_equal_k = {
                k: [to_obj(eval_result_old)["samples"][sample_idx][k], to_obj(eval_result)["samples"][sample_idx][k]]
                for k in ['sample_idx', 'num_turns', 'turns', 'origin_question', 'final_answer', 'final_answer_text', 'planner_summary', 'eval_metric', 'total_turn_list']
                    if to_obj(eval_result_old)["samples"][sample_idx][k] != to_obj(eval_result)["samples"][sample_idx][k]
            } # ['num_turns', 'turns', 'origin_question', 'final_answer', 'planner_summary']

            not_equal_turns_len = {
                i: [len(to_obj(eval_result_old)["samples"][sample_idx]["turns"]), len(to_obj(eval_result)["samples"][sample_idx]["turns"])]
                for i in range(len(to_obj(eval_result_old)["samples"][sample_idx]["turns"]))
                    if len(to_obj(eval_result_old)["samples"][sample_idx]["turns"]) != len(to_obj(eval_result)["samples"][sample_idx]["turns"])
            }
            not_equal_idx = {
                i: [to_obj(eval_result_old)["samples"][sample_idx]["turns"][i], to_obj(eval_result)["samples"][sample_idx]["turns"][i]]
                for i in range(len(to_obj(eval_result_old)["samples"][sample_idx]["turns"]))
                    if to_obj(eval_result_old)["samples"][sample_idx]["turns"][i] != to_obj(eval_result)["samples"][sample_idx]["turns"][i]
            }
            turns_idx = 2
            not_equal_k = {
                k: [[to_obj(eval_result_old)["samples"][sample_idx]["turns"][turns_idx][k], to_obj(eval_result)["samples"][sample_idx]["turns"][turns_idx][k]]]
                for k in to_obj(eval_result_old)["samples"][sample_idx]["turns"][turns_idx].keys()
                    if to_obj(eval_result_old)["samples"][sample_idx]["turns"][turns_idx][k] != to_obj(eval_result)["samples"][sample_idx]["turns"][turns_idx][k]
            } # ['prompt_text', 'response_text', 'prompt_ids_length', 'response_ids_length', 'is_end', 'reward_score', 'role', 'tool_call_info']

        if self.is_eval:
            rollout_result = rollout_result_eval
        else:
            rollout_result = rollout_result_train
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
            # breakpoint()
            return agent_metrics

    def gen_extra_fields_group(self, task_results: list[MultiTurnAgentLoopOutput], answer: str) -> Optional[dict]:
        return None

    def post_process_metric(self, agent_metrics_list: list[dict]):
        if self._world_size > 1:
            if self._rank == 0:
                for i in range(1, self._world_size):
                    recv_list = self.recv(self._group_name, i)
                    agent_metrics_list.extend(recv_list)
            else:
                self.send(agent_metrics_list, self._group_name, 0)
                return
        keys_list = [i.keys() for i in agent_metrics_list]
        all_keys = set([j for i in keys_list for j in i])
        assert all((len(all_keys) == len(keys) for keys in keys_list))

        whole_metrics = {}
        for k in all_keys:
            values_list = [i[k] for i in agent_metrics_list]
            if "agent/turn/mean/" in k or "agent/traj/mean/" in k:
                whole_metrics[k] = sum(values_list) / len(values_list)
            elif "agent/turn/max/" in k or "agent/traj/max/" in k:
                whole_metrics[k] = max(values_list)
            else:
                assert False
        return whole_metrics

    def get_rollout_metrics(
        self,
        rollout_result: DynamicRolloutResult,
    ) -> dict:
        return {}

    def get_rollout_result_old(
        self,
        task_results: list[MultiTurnAgentLoopOutput],
        extra_fields_group: Optional[dict]=None,
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
        rollout_logprobs = None
        if self.return_logprobs:
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

        extra_fields_turn = None
        if self.extra_keys_turn is not None:
            extra_fields_turn = {k: list() for k in self.extra_keys_turn}
        extra_fields_traj = None
        if self.extra_keys_traj is not None:
            extra_fields_traj = {k: list() for k in self.extra_keys_traj}

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
                role = single_turn_output.extra_fields["role"]
                roles.append(role)
                if self.train_roles and role in self.train_roles:
                    have_role = True

            if have_role:
                role_group_size += 1

            for single_turn_output in task_result.single_turn_outputs:
                # Collect tool call info (keep all turns but track valid ones)
                single_turn_output: AgentLoopOutput
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
                        assert search_count > 0 or access_count > 0
                        num_valid_worker_turns += 1
                else:
                    # Invalid turn - pad with zeros
                    turn_subtask_counts.append(0)
                    turn_search_counts.append(0)
                    turn_access_counts.append(0)

            for single_turn_output in task_result.single_turn_outputs:
                if self.extra_keys_turn is not None:
                    for k in self.extra_keys_turn:
                        v = single_turn_output.extra_fields.get(k, None)
                        extra_fields_turn[k].append(v)
            if self.extra_keys_traj is not None:
                for k in self.extra_keys_traj:
                    v = task_result.extra_fields.get(k, None)
                    extra_fields_traj[k].append(v)

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
            extra_fields_turn=extra_fields_turn,
            extra_fields_traj=extra_fields_traj,
            extra_fields_group=extra_fields_group,
        )

    def get_rollout_result(
        self,
        task_results: list[MultiTurnAgentLoopOutput],
        extra_fields_group: Optional[dict]=None,
        use_no_training=True,
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
        rollout_logprobs = None
        if self.return_logprobs:
            rollout_logprobs = []
        is_end = []
        rewards = []

        # Collect eval_metrics per trajectory
        roles = []

        extra_fields_turn = None
        if self.extra_keys_turn is not None:
            extra_fields_turn = {k: list() for k in self.extra_keys_turn}
        extra_fields_traj = None
        if self.extra_keys_traj is not None:
            extra_fields_traj = {k: list() for k in self.extra_keys_traj}

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

            for single_turn_output in task_result.single_turn_outputs:
                if self.extra_keys_turn is not None:
                    for k in self.extra_keys_turn:
                        v = single_turn_output.extra_fields.get(k, None)
                        extra_fields_turn[k].append(v)
                        if k == "role" and single_turn_output.extra_fields["not_training"] != True:
                            roles.append(v)
            if self.extra_keys_traj is not None:
                for k in self.extra_keys_traj:
                    v = task_result.extra_fields.get(k, None)
                    extra_fields_traj[k].append(v)

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
            # train
            # roles=extra_fields_turn["role"],
            roles=roles,
            role_group_sizes=[extra_fields_group["role_group_size"]] * len(idx_to_traj),
            # metrics in train
            turn_subtask_counts=extra_fields_turn["subtask_count"],
            turn_search_counts=extra_fields_turn["search_count"],
            turn_access_counts=extra_fields_turn["access_count"],
            num_valid_planner_turns=sum(extra_fields_traj["num_valid_planner_turns"]),
            num_valid_worker_turns=sum(extra_fields_traj["num_valid_worker_turns"]),
            # eval & metrics in train
            eval_metrics=extra_fields_traj["eval_metric"],
            total_turn_list_metric=extra_fields_traj["total_turn_list"],
        )

    def get_rollout_result_eval_old(
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
                    final_answer = final_answer.to_dict(orient="records")

            # Extract total_turn_list for MAS workflow
            total_turn_list = task_result.extra_fields.get("total_turn_list", None)

            sample_data = {
                "sample_idx": idx,
                "num_turns": len(turns),
                "turns": turns,
                "origin_question": task_result.extra_fields.get(
                    "origin_question", None
                ),
                "final_answer": final_answer,
                "final_answer_text": task_result.extra_fields.get(
                    "final_answer_text", None
                ),
                "planner_summary": task_result.extra_fields.get(
                    "planner_summary", None
                ),
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

    @staticmethod
    def get_rollout_result_eval(
        rollout_result: DynamicRolloutResult,
        log_info=None,
    ) -> dict:
        group_size = rollout_result.group_size

        eval_metrics = rollout_result.eval_metrics or [None] * group_size
        total_turn_list_metric = (
            rollout_result.total_turn_list_metric or [None] * group_size
        )

        samples_data: list[dict] = []
        for traj_idx in range(group_size):
            eval_metric = (
                eval_metrics[traj_idx] if traj_idx < len(eval_metrics) else None
            ) or {}
            total_turn_list = (
                total_turn_list_metric[traj_idx]
                if traj_idx < len(total_turn_list_metric)
                else None
            )

            turn_idxes = [i for i, j in enumerate(rollout_result.idx_to_traj) if j == traj_idx]
            turns = []
            for turn_idx in turn_idxes:
                turn_data = {
                    "prompt_text": rollout_result.extra_fields_turn["prompt_text"][turn_idx],
                    "response_text": rollout_result.extra_fields_turn["response_text"][turn_idx],
                    "prompt_ids_length": rollout_result.prompt_lengths[turn_idx],
                    "response_ids_length": rollout_result.response_lengths[turn_idx],
                    "is_end": rollout_result.is_end[turn_idx],
                    "reward_score": rollout_result.rewards[turn_idx],
                    "role": rollout_result.extra_fields_turn["role"][turn_idx],
                    "tool_call_info": rollout_result.extra_fields_turn["tool_call_info"][turn_idx],
                }
                turns.append(turn_data)

            final_answer = rollout_result.extra_fields_traj["final_answer"][traj_idx]
            if isinstance(final_answer, pd.DataFrame):
                final_answer = final_answer.to_dict(orient="records")
            samples_data.append(
                {
                    "sample_idx": traj_idx,
                    "num_turns": len(turn_idxes),
                    "turns": turns,
                    "origin_question": rollout_result.extra_fields_traj["origin_question"][traj_idx],
                    "final_answer": final_answer,
                    "final_answer_text": rollout_result.extra_fields_traj["final_answer_text"][traj_idx],
                    "planner_summary": rollout_result.extra_fields_traj["planner_summary"][traj_idx],
                    "eval_metric": eval_metric,
                    "total_turn_list": total_turn_list,
                }
            )

        answer = rollout_result.extra_fields_group["answer"]
        eval_result = {"group_size": group_size, "answer": answer, "samples": samples_data}
        if isinstance(answer, dict) and "instance_id" in answer and log_info is not None:
            log_info(f"finish question id {answer['instance_id']}")
        return eval_result

    async def run_one_query(self, *args, **kwargs) -> MultiTurnAgentLoopOutput:
        raise NotImplementedError("Subclasses must implement this method")
