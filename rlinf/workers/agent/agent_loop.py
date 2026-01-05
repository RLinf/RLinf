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
    RolloutResult,
    DynamicRolloutResult,
)
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.tool_worker import ToolChannelInfo
from rlinf.workers.rollout.utils import green

from rlinf.agents.utils.reward import compute_score_em, compute_score_f1, extract_final_answer

from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
)
import json
import pandas as pd
import traceback
import requests
import time
import re
from pandas.api.types import is_integer_dtype, is_float_dtype
import aiohttp
import threading


class SGLangClient:
    """SGLang API client with connection pooling."""

    # Class-level shared session for connection pooling
    _shared_session = None
    _session_lock = threading.Lock()

    @classmethod
    async def get_session(cls):
        """Get or create shared aiohttp session with connection pooling."""
        if cls._shared_session is None or cls._shared_session.closed:
            with cls._session_lock:
                if cls._shared_session is None or cls._shared_session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=100,  # Max total connections
                        limit_per_host=50,  # Max connections per host
                        ttl_dns_cache=300,  # DNS cache TTL
                        enable_cleanup_closed=True,
                    )
                    cls._shared_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=30),
                        trust_env=False
                    )
        return cls._shared_session

    def __init__(self, llm_ip: str, llm_port: str, llm_type: str):
        self.llm_ip = llm_ip
        self.llm_port = llm_port
        self.llm_type = llm_type

    async def call_sglang_api(self, messages: list) -> str:
        """
        Call SGLang API with connection pooling.

        Args:
            llm_ip: LLM server IP
            llm_port: LLM server port
            llm_type: LLM model type
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Response text from the API, or None if failed
        """
        url = f"http://{self.llm_ip}:{self.llm_port}/v1/chat/completions"
        data = {
            "model": self.llm_type,
            "messages": messages,
        }

        max_retries = 5
        retry_count = 0
        session = await self.get_session()

        while retry_count < max_retries:
            try:
                async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        result = await response.json()
                        result_text = result['choices'][0]['message']['content']
                        return result_text
                    else:
                        response_text = await response.text()
                        print(f"[ERROR] SGLangClient: Failed calling sglang: {response.status}, response: {response_text}")
            except Exception as e:
                print(f"[ERROR] SGLangClient: Exception error in calling sglang: {e}")

            retry_count += 1
            await asyncio.sleep(10)

        print(f"[ERROR] SGLangClient: Failed after {max_retries} retries")
        return None


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
        self.return_logprobs = not cfg.algorithm.recompute_logprobs
        assert self.return_logprobs
        self.is_dynamic_rollout_batch = cfg.agentloop.get("is_dynamic_rollout_batch", False)
        if self.is_dynamic_rollout_batch:
            assert isinstance(self, MultiTurnAgentLoopWorker), "agent loop worker must be MultiTurnAgentLoopWorker if is_dynamic_rollout_batch is True"
            # assert not self.return_logprobs, "recompute_logprobs must be False if is_dynamic_rollout_batch is True"

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.rollout.model.model_path)
        self.reward_type = self.cfg.reward.get("reward_type", "EM").upper()    
        self.reward_eval = self.cfg.reward.get("eval_metric", [])
        self.reward_eval = [m.upper() for m in self.reward_eval]
        self.is_widesearch = self.cfg.data.get("is_widesearch", False)

        self.use_llm_judge_api = self.cfg.agentloop.get("use_llm_judge_api", False)
        if self.use_llm_judge_api:
            llm_ip = self.cfg.agentloop.get("llm_ip", "")
            llm_port = self.cfg.agentloop.get("llm_port", "")
            llm_type = self.cfg.agentloop.get("llm_type", "")            
            self.sgl_client = SGLangClient(llm_ip, llm_port, llm_type)

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
        # for multi agent model, use a different agent with no training.
        # such as a 8b planner with training and a 4b worker without training.
        self.solid_generate_input_channels = solid_generate_input_channels

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
    ) -> RolloutResult:
        """
        Collect group task results into a RolloutResult.
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

        for idx, task_result in enumerate(task_results):
            # Extract eval_metric from extra_fields for this trajectory
            eval_metric = task_result.extra_fields.get("eval_metric", None)
            eval_metrics.append(eval_metric)
            for single_turn_output in task_result.single_turn_outputs:
                single_turn_output: AgentLoopOutput
                idx_to_traj.append(idx)
                prompt_lengths.append(len(single_turn_output.prompt_ids))
                response_lengths.append(len(single_turn_output.response_ids))
                input_ids.append(single_turn_output.prompt_ids + single_turn_output.response_ids)
                rollout_logprobs.append(single_turn_output.response_logprobs)
                is_end.append(single_turn_output.is_end)
                rewards.append(single_turn_output.reward_score)

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
            turn_subtask_counts=turn_subtask_counts,
            turn_search_counts=turn_search_counts,
            turn_access_counts=turn_access_counts,
            num_valid_planner_turns=num_valid_planner_turns,
            num_valid_worker_turns=num_valid_worker_turns,
            eval_metrics=eval_metrics,
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
            }
            samples_data.append(sample_data)

        # Create evaluation result dictionary with all k samples
        eval_result = {
            "group_size": group_size,
            "answer": answer,
            "samples": samples_data,  # All k samples with their metrics
        }

        return eval_result

    async def run_one_query(self, *args, **kwargs) -> MultiTurnAgentLoopOutput:
        raise NotImplementedError("Subclasses must implement this method")

    async def get_final_reward_score(self, origin_question, extract_answer, label_answer, is_markdown = False):
        em_score = None
        f1_score = None
        llm_score = None
        reward_score = 0.0
        format = True

        if is_markdown:
            reward_score, eval_metric, format = await self.evaluate_markdown(extract_answer, label_answer)
            return reward_score, eval_metric, format
        if label_answer is not None and extract_answer is not None:
            # Compute all metrics specified in eval_metric
            if "EM" in self.reward_eval:
                em_score = compute_score_em(extract_answer, label_answer)

            if "F1" in self.reward_eval:
                f1_score = compute_score_f1(extract_answer, label_answer)

            if "LLM" in self.reward_eval:
                # Use LLM as judge
                llm_score = await self.verify_answer_with_llm_judge(
                    question=origin_question,
                    predicted_answer=extract_answer,
                    correct_answer=label_answer,
                    use_llm_judge_api=self.use_llm_judge_api,
                )

            # Determine which metric to use as reward based on reward_type
            if self.reward_type == "EM":
                reward_score = em_score if em_score is not None else compute_score_em(extract_answer, label_answer)
            elif self.reward_type == "F1":
                reward_score = f1_score if f1_score is not None else compute_score_f1(extract_answer, label_answer)
            elif self.reward_type == "LLM":
                reward_score = llm_score if llm_score is not None else await self.verify_answer_with_llm_judge(
                    question=origin_question,
                    predicted_answer=extract_answer,
                    correct_answer=label_answer,
                    use_llm_judge_api=self.use_llm_judge_api,
                )
            else:
                raise ValueError(f"Illegal reward func {self.reward_type}")

        else:
            reward_score = 0.0

        eval_metric = {
            "EM": em_score,
            "F1": f1_score,
            "LLM": llm_score
        }
        return reward_score, eval_metric, format
    
    async def verify_answer_with_llm_judge(
        self,
        question: str,
        predicted_answer: str,
        correct_answer: list,
        use_llm_judge_api = False,
    ) -> float:

        """Use LLM as judge to verify if predicted answer is equivalent to correct answer.

        Args:
            question: Original question
            predicted_answer: The model's predicted answer
            correct_answer: The ground truth answer
            output_buffer: Buffer to store the judge's generation output

        Returns:
            Score: 1.0 if correct, 0.0 if incorrect
        """
        from rlinf.agents.prompt.prompt import LLM_JUDGE_PROMPT
        
        if len(correct_answer) == 1:
        # Format the judge prompt
            judge_prompt_text = LLM_JUDGE_PROMPT.format(
                question=question,
                correct_answer=correct_answer[0],
                response=predicted_answer
            )
        else:
            judge_prompt_text = LLM_JUDGE_PROMPT.format(
                question=question,
                correct_answer=correct_answer,
                response=predicted_answer
            )            

        # Create messages for the judge
        judge_messages = [
            {'role': "system", "content": "You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer."},
            {"role": "user", "content": judge_prompt_text}
        ]

        # Apply chat template
        if use_llm_judge_api:
            judge_response_text = await self.sgl_client.call_sglang_api(judge_messages)
            judge_prompt_ids = []
            judge_response_ids = []
        else:
            judge_prompt_ids = self.tokenizer.apply_chat_template(
                judge_messages, tokenize=True, add_generation_prompt=True
            )

            generate_result = await self.generate(
                judge_prompt_ids, sampling_params={"max_new_tokens": 8192}
            )

            judge_response_ids = generate_result["output_ids"]
            judge_response_text = self.tokenizer.decode(judge_response_ids)

        # Parse the judge's response
        # The judge should respond with "Correct" or "Incorrect"
        judge_response_clean = judge_response_text.strip().lower()
        # Check if the response contains "correct" (but not "incorrect")
        if "correct" in judge_response_clean and "incorrect" not in judge_response_clean:
            return 1.0
        else:
            return 0.0

    async def evaluate_markdown(self, extract_answer, label_answer):
        """Evaluate markdown table answer against ground truth.

        Args:
            extract_answer: pd.DataFrame extracted from model output
            label_answer: dict containing:
                - answer: Either pd.DataFrame or str (markdown string)
                - unique_columns: list of primary key columns
                - required: list of required columns (optional, defaults to all)

        Returns:
            tuple: (reward_score, eval_metric dict)
        """

        # Helper function to normalize column names
        def norm_column(col: str) -> str: 
            if not self.is_widesearch:
                return col.strip().lower()
            else:
                return col.strip().lower().replace(" ", "")
            # return col.strip().lower().replace(" ", "")

        # Helper function to calculate F1 score
        def calc_f1(precision, recall):
            epsilon = 1e-9
            return (
                (2 * precision * recall / (precision + recall))
                if (precision + recall > epsilon)
                else 0.0
            )

        def normalize_series_to_str(s: pd.Series) -> pd.Series:
            s0 = s.astype(str).str.strip()
            num = pd.to_numeric(s0, errors="coerce")
            if num.notna().any():
                return num.map(lambda x: "" if pd.isna(x) else f"{x:g}")
            else:
                return s0
            
        # Initialize metrics
        score = 0.0
        precision_by_row = 0.0
        recall_by_row = 0.0
        f1_by_row = 0.0
        precision_by_item = 0.0
        recall_by_item = 0.0
        f1_by_item = 0.0
        msg = ""
        eval_metrics = {
            "score": 0.0,
            "precision_by_row": 0.0,
            "recall_by_row": 0.0,
            "f1_by_row": 0.0,
            "precision_by_item": 0.0,
            "recall_by_item": 0.0,
            "f1_by_item": 0.0,
            "search_precision_by_item": 0.0,
            "search_recall_by_item": 0.0,
            "search_f1_by_item": 0.0,                        
        }
        metrics = list(set(self.reward_eval) | set([self.reward_type]))
        error_eval_return = {metric: eval_metrics for metric in metrics}

        try:
            # Parse label_answer
            if isinstance(label_answer, dict):
                answer_markdown = label_answer.get("answer", "")
                unique_columns = label_answer.get("unique_columns", [])
                required_columns = label_answer.get("required", [])
            else:
                # If label_answer is a string, assume it's markdown
                answer_markdown = label_answer
                unique_columns = []
                required_columns = []

            # Convert answer_markdown to DataFrame if it's a string
            if isinstance(answer_markdown, str):
                answer_df = extract_final_answer(answer_markdown, mode='markdown')
                if answer_df is None:
                    msg = "Failed to parse label answer markdown"
                    self.log_error(msg)                    
                    return 0.0, error_eval_return, False
            elif isinstance(answer_markdown, pd.DataFrame):
                answer_df = answer_markdown
            else:
                msg = f"Invalid label answer type: {type(answer_markdown)}"
                self.log_error(msg)
                return 0.0, error_eval_return, False

            # Validate extract_answer
            if not isinstance(extract_answer, pd.DataFrame) or extract_answer.empty:
                msg = f"Extracted answer is None or not a DataFrame, it's {extract_answer}"
                # self.log_error(msg)
                return 0.0, error_eval_return, False

            response_df = copy.deepcopy(extract_answer)

            # Normalize column names
            answer_df.columns = [norm_column(col) for col in answer_df.columns]
            response_df.columns = [norm_column(col) for col in response_df.columns]

            # Normalize unique_columns and required_columns
            unique_columns = [norm_column(col) for col in unique_columns]

            if not required_columns:
                required_columns = list(answer_df.columns)
            else:
                required_columns = [norm_column(col) for col in required_columns] # widesearch requir columns: " " -> ""

            # Check if response has required columns
            if not set(required_columns).issubset(set(response_df.columns)):
                # Try primary key preprocessing to map column names
                column_map = await self.primary_key_preprocess(
                    list(response_df.columns), required_columns, self.use_llm_judge_api
                )
                # breakpoint()
                response_df.rename(columns=column_map, inplace=True)

            if not set(required_columns).issubset(set(response_df.columns)):
                msg = f"required_columns {required_columns} != response_df {list(response_df.columns)}"
                # self.log_error(msg)
                return 0.0, error_eval_return, False

            # Preprocess: convert all columns to string, but bufore make sure 1 = 1.0
            # for col in required_columns:
            #     try:
            #         answer_type = answer_df[col].dtype 
            #         response_type = response_df[col].dtype
            #     except Exception:
            #         self.log_error("error type",answer_df[col].dtype, response_df[col].dtype)
            #         answer_type = None
            #         response_type = None
                
            #     self.log_info(f"xzxuan type is {answer_type}, and {response_type}")
            #     if (response_type == float and answer_type == int) or (
            #         response_type == int and answer_type == float
            #     ):
            #         if response_type == int:
            #             response_df[col] = response_df[col].astype(float)
            #         elif answer_type == int:
            #             answer_df[col] = answer_df[col].astype(float)

            #     answer_df[col] = answer_df[col].astype(str)
            #     response_df[col] = response_df[col].astype(str)

            for col in required_columns:
                answer_df[col] = normalize_series_to_str(answer_df[col])
                response_df[col] = normalize_series_to_str(response_df[col])

            # Remove duplicates based on unique columns
            if unique_columns:
                response_df.drop_duplicates(subset=unique_columns, inplace=True)
                answer_df.drop_duplicates(subset=unique_columns, inplace=True)

                # Preprocess primary keys using LLM
                for col in unique_columns:
                    primary_key_map = await self.primary_key_preprocess(
                        response_df[col].tolist(),
                        answer_df[col].tolist(),
                        self.use_llm_judge_api
                    )
                    response_df[col + "_before_map"] = response_df[col]
                    response_df[col] = response_df[col].apply(
                        lambda x: primary_key_map.get(x, x)
                    )

            # TODO: any preprocess? at least norm str! how can int is equal to float? like 1.0 == 1?
            # for col, item in query.evaluation["eval_pipeline"].items():
            #     preprocess_func_name_list = item.get("preprocess", [])
            #     for preprocess_func_name in preprocess_func_name_list:
            #         response_df[col] = response_df[col].apply(
            #             lambda x: preprocess_call(x, preprocess_func_name)
            #         )
            #         answer_df[col] = answer_df[col].apply(
            #             lambda x: preprocess_call(x, preprocess_func_name)
            #         )


            # Inner join to find matching rows
            df_inner = pd.merge(
                answer_df,
                response_df,
                on=unique_columns,
                how="inner",
                suffixes=("_query", "_response"),
            )

            # Initialize score DataFrames for each metric type in self.reward_eval
            df_inner_scores = {}

            for metric_type in metrics:
                df_inner_scores[metric_type] = pd.DataFrame(index=df_inner.index)

            llm_tasks = []
            llm_columns = []
            
            # Process each column
            for col in required_columns:
                if col in unique_columns:
                    # Primary keys must match exactly for all metrics
                    for metric_type in metrics:
                        df_inner_scores[metric_type][f"{col}_score"] = 1.0
                else:
                    # Compute scores for each metric type
                    for metric_type in metrics:
                        if metric_type == "EM":
                            scores = [
                                compute_score_em(row[col + "_response"], row[col + "_query"])
                                for _, row in df_inner.iterrows()
                            ]
                            df_inner_scores["EM"][f"{col}_score"] = scores

                        elif metric_type == "F1":
                            scores = [
                                compute_score_f1(row[col + "_response"], row[col + "_query"])
                                for _, row in df_inner.iterrows()
                            ]
                            df_inner_scores["F1"][f"{col}_score"] = scores

                        elif metric_type == "LLM":
                            # Collect for batch LLM judge
                            responses = df_inner[col + "_response"].tolist()
                            targets = df_inner[col + "_query"].tolist()
                            llm_tasks.append(
                                self.llm_judge_column(responses, targets, self.use_llm_judge_api)
                            )
                            llm_columns.append(col)

            # Execute all LLM calls in parallel
            if llm_tasks:
                llm_results = await asyncio.gather(*llm_tasks)
                # Assign results back to df_inner_scores["LLM"]
                for col, scores in zip(llm_columns, llm_results):
                    df_inner_scores["LLM"][f"{col}_score"] = scores

            # Calculate metrics for each evaluation method
            num_pred_rows = len(response_df)
            num_gt_rows = len(answer_df)
            num_pred_items = num_pred_rows * len(required_columns)
            num_gt_items = num_gt_rows * len(required_columns)

            eval_metric = {}
            for metric_type in metrics:
                df_score = df_inner_scores[metric_type]

                # Row-level metrics
                row_scores = df_score.min(axis=1)
                tp_by_row = row_scores.sum()
                precision_by_row = tp_by_row / num_pred_rows if num_pred_rows > 0 else 0.0
                recall_by_row = tp_by_row / num_gt_rows if num_gt_rows > 0 else 0.0
                f1_by_row = calc_f1(precision_by_row, recall_by_row)

                # Item-level metrics
                tp_by_item = df_score.sum().sum()
                precision_by_item = tp_by_item / num_pred_items if num_pred_items > 0 else 0.0
                recall_by_item = tp_by_item / num_gt_items if num_gt_items > 0 else 0.0
                f1_by_item = calc_f1(precision_by_item, recall_by_item)

                # Search-specific item-level metrics (non-primary-key columns only)
                # This reflects pure search ability without trivial primary key matches
                non_pk_columns = [col for col in required_columns if col not in unique_columns]
                if non_pk_columns:
                    # Select only non-primary-key column scores
                    search_score_cols = [f"{col}_score" for col in non_pk_columns]
                    df_search_score = df_score[search_score_cols]

                    # Compute search metrics
                    tp_search = df_search_score.sum().sum()
                    num_pred_search = num_pred_rows * len(non_pk_columns)
                    num_gt_search = num_gt_rows * len(non_pk_columns)

                    search_precision_by_item = tp_search / num_pred_search if num_pred_search > 0 else 0.0
                    search_recall_by_item = tp_search / num_gt_search if num_gt_search > 0 else 0.0
                    search_f1_by_item = calc_f1(search_precision_by_item, search_recall_by_item)
                else:
                    # If all columns are primary keys, search metrics are 0
                    search_precision_by_item = 0.0
                    search_recall_by_item = 0.0
                    search_f1_by_item = 0.0

                if (
                    precision_by_item == recall_by_item == 1.0
                    and precision_by_row == recall_by_row == 1.0
                ): 
                    score = 1.0

                eval_metric[metric_type] = {
                    "score": score,  
                    "precision_by_row": precision_by_row,
                    "recall_by_row": recall_by_row,
                    "f1_by_row": f1_by_row,
                    "precision_by_item": precision_by_item,
                    "recall_by_item": recall_by_item,
                    "f1_by_item": f1_by_item,
                    # Search-specific metrics (non-primary-key columns only)
                    "search_precision_by_item": search_precision_by_item,
                    "search_recall_by_item": search_recall_by_item,
                    "search_f1_by_item": search_f1_by_item
                }

            msg = f"Evaluated with {len(metrics)} metrics: {metrics}"

        except Exception as e:
            msg = f"Evaluation error: {traceback.format_exc()}"            
            self.log_error(msg)
            return 0.0, error_eval_return, False

        # Select reward score based on self.reward_type
        reward_score = eval_metric[self.reward_type]["f1_by_item"]
        # self.log_info(eval_metric)
        return reward_score, eval_metric, True

    async def llm_judge_column(self, responses: list, targets: list, criterion: str = None, use_llm_judge_api = False) -> list:
        if criterion is None:
            criterion = "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence."

        if not responses:
            return []

        # Widesearch's eval_column_prompt
        eval_column_prompt = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by step according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
    "idx_0": score,
    "idx_1": score,
    ...
}}
```

{criterion}
"""

        user_prompt = """Here is the response you need to judge, please make sure to analyze each item step by step before providing the final scoring results.

{response}
"""

        # Build response dict
        response_dict = {}
        for idx, (resp, tar) in enumerate(zip(responses, targets)):
            response_dict[f"idx_{idx}"] = {"response": str(resp), "target": str(tar)}

        # Format prompt
        system_prompt = eval_column_prompt.format(
            criterion=criterion,
        )

        user_prompt = user_prompt.format(
            response = response_dict
        )
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if use_llm_judge_api:
            result_text = await self.sgl_client.call_sglang_api(messages)
        else:
            # Apply chat template
            prompt_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )

            # Generate
            generate_result = await self.generate(
                prompt_ids, sampling_params={"max_new_tokens": 4096}
            )

            result_text = self.tokenizer.decode(generate_result["output_ids"])

        try:
            pat = r"```json\s*(\{.*?\})\s*```"
            matches = re.findall(pat, result_text, re.DOTALL)
            if matches:
                json_str = matches[-1]
                score_dict = json.loads(json_str)
                score_list = [
                    float(score_dict.get(f"idx_{idx}", 0))
                    for idx in range(len(responses))
                ]
            else:
                # Parsing failed, default to 0
                score_list = [0.0] * len(responses)
        except Exception as e:
            # If any error, default to 0
            score_list = [0.0] * len(responses)

        # Ensure correct length
        if len(score_list) != len(responses):
            score_list = [0.0] * len(responses)

        return score_list

    async def primary_key_preprocess(self, response_list, reference_list, use_llm_judge_api = False):
        primary_key_map = {}

        # The prompt template from widesearch
        primary_key_preprocess_prompt = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```
"""

        user_prompt = """
The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}
"""

        # Format the prompt
        system_prompt = primary_key_preprocess_prompt

        user_prompt = user_prompt.format(
            response=response_list, reference=reference_list
        )

        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if use_llm_judge_api:
            result_text = await self.sgl_client.call_sglang_api(messages)
        else:
            # Apply chat template
            prompt_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )

            # Generate
            generate_result = await self.generate(
                prompt_ids, sampling_params={"max_new_tokens": 4096}
            )

            result_text = self.tokenizer.decode(generate_result["output_ids"])

        # Parse JSON from result
        try:
            pat = r"```json\s*(\{.*?\})\s*```"
            matches = re.findall(pat, result_text, re.DOTALL)
            if matches:
                json_str = matches[-1]
                transform_map = json.loads(json_str)
                primary_key_map.update(transform_map)
        except Exception:
            pass

        return primary_key_map
