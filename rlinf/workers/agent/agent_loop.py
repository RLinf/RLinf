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
import contextvars
import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from omegaconf import DictConfig

from rlinf.agents.tool_call.schema import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
)
from rlinf.algorithms.registry import get_toolcall_parser
from rlinf.data.schema.reasoning_requests import RolloutRequest
from rlinf.data.schema.reasoning_results import (
    DiscardedGroup,
    DynamicRolloutResult,
    RolloutResult,
)
from rlinf.models.tokenization.hf import hf_tokenizer
from rlinf.scheduler import Channel, Worker
from rlinf.utils.metric_utils import aggregate_agent_metrics
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.session_server_client import (
    SessionBinding,
    SessionDiscardedError,
    SessionServerClient,
)
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
    """Weight version the rollout instance had loaded when the generate
    call producing this turn's response started, and when it ended.
    None when the rollout did not stamp versions (synchronous training).
    """
    version_start: Optional[int] = None
    version_end: Optional[int] = None


@dataclass
class MultiAgentLoopOutput:
    """Multi agent loop output."""

    """Single-turn agent loop outputs."""
    single_turn_outputs: list[AgentLoopOutput]
    """Debug information to print."""
    trace_prints: list[Any] = field(default_factory=list)
    """Extra fields for dynamic addition."""
    extra_fields: dict[str, Any] = field(default_factory=dict)


class GroupDiscarded(RuntimeError):
    """A group's data is unusable, affecting only that group.

    Raise this only when the task code is certain the failure spoils this
    one group and nothing else; the continuous loop turns it into a
    discard record and moves on. Every other exception terminates
    training.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class AgentLoopWorker(Worker):
    """
    Abstract agent loop worker.

    Subclasses must implement the run_one_query method.
    """

    # Class-level default so generate dispatch is safe even when the
    # instance is built without running ``__init__`` (e.g. ``object.__new__``
    # in unit tests); ``init_worker`` installs the real transport.
    session_server_client = None

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
            assert isinstance(self, MultiAgentLoopWorker), (
                "agent loop worker must be MultiAgentLoopWorker if is_dynamic_rollout_batch is True"
            )

        self.tokenizer = hf_tokenizer(
            cfg.rollout.model.model_path,
            trust_remote_code=cfg.rollout.model.trust_remote_code,
        )
        self.toolcall_parser = None
        if cfg.agentloop.get("toolcall_parser", None) is not None:
            self.toolcall_parser = get_toolcall_parser(cfg.agentloop.toolcall_parser)

        # Task running run_agentloop_rollout_continuously, None while idle.
        self._agentloop_task = None
        self._agent_metric_channel = None
        self._producer_error = None
        self._inflight_groups = set()
        # Session server transport, set by init_worker when the runner hands
        # over session server URLs instead of a generate channel pair.
        self.session_server_client: Optional[SessionServerClient] = None
        # The session of the trajectory whose task is currently running;
        # set per trajectory by the session-path group runner, read by
        # generate.
        self._session_binding = contextvars.ContextVar(
            "rlinf_agent_session_binding", default=None
        )
        # Idle-time accounting for the continuous loop: seconds with no
        # in-flight group, accumulated over the window since the last
        # pop_idle_stats call.
        self._idle_seconds = 0.0
        self._idle_window_start = None
        self._idle_since = None

    def init_worker(
        self,
        generate_input_channel: Channel,
        generate_output_channel: Channel,
        tool_channel_info_map: dict[str, ToolChannelInfo],
        tool_name_map: dict[str, str],
        tool_worker_output_channel: Channel,
        solid_generate_input_channels: dict[str, Channel] = {},
        session_server_urls: Optional[list[str]] = None,
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
        # With URLs, generate goes over HTTP to the session servers.
        self.session_server_client = None
        if session_server_urls:
            session_server_cfg = self.cfg.get("session_server", {}) or {}
            self.session_server_client = SessionServerClient(
                session_server_urls,
                generate_timeout_s=float(
                    session_server_cfg.get("generate_timeout_s", 600)
                ),
                keepalive_timeout_s=float(
                    session_server_cfg.get("keep_alive_s", 5)
                )
                - 1.0,
            )

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: Optional[dict] = None,
        rollout_name: Optional[str] = None,
    ):
        if rollout_name is not None:
            # Solid targets are frozen models outside the training loop;
            # they never enter a session and produce no payload.
            return await self._generate_on_solid_channel(
                prompt_ids, sampling_params, rollout_name
            )
        if self.session_server_client is not None:
            binding = self._session_binding.get()
            assert binding is not None, (
                "the session server transport is active but no session is "
                "bound to this task; the group runner binds one per "
                "trajectory"
            )
            return await self._generate_on_session(binding, prompt_ids, sampling_params)
        return await self._generate_on_channel(prompt_ids, sampling_params)

    async def _generate_on_channel(
        self, prompt_ids: list[int], sampling_params: Optional[dict]
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
        if isinstance(result, dict) and "error" in result:
            # The rollout side failed to generate and reported back via the
            # channel; fail loudly instead of continuing.
            raise RuntimeError(f"Rollout generate request failed: {result['error']}")
        return result

    async def _generate_on_solid_channel(
        self,
        prompt_ids: list[int],
        sampling_params: Optional[dict],
        rollout_name: str,
    ):
        channel_key = uuid4().hex
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
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Rollout generate request failed: {result['error']}")
        return result

    async def _generate_on_session(
        self,
        binding: SessionBinding,
        prompt_ids: list[int],
        sampling_params: Optional[dict],
    ):
        try:
            result = await self.session_server_client.generate(
                binding.group_id, binding.session_id, prompt_ids, sampling_params
            )
        except SessionDiscardedError as exc:
            # Same meaning as the channel path's error report: the group
            # is unusable and the discard is already reported server-side.
            raise GroupDiscarded(exc.reason) from exc
        binding.record_turn(prompt_ids, result["output_ids"], sampling_params)
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
        # return_dict=False: transformers 5 returns a BatchEncoding here, whose
        # [0] is an Encoding rather than a token id, and the slice below needs the
        # plain list[int]. The argument is accepted from transformers 4.46 on.
        wo_ids = self.tokenizer.apply_chat_template(
            wo_messages, add_generation_prompt=False, tokenize=True, return_dict=False
        )
        wi_ids = self.tokenizer.apply_chat_template(
            wi_messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        return wi_ids[len(wo_ids) :]

    async def run_agentloop_rollout_group(
        self,
        input_ids: list[int],
        answer: str,
        group_size: int,
        output_channel: Channel,
        group_id: Optional[int] = None,
        task_name: Optional[str] = None,
    ):
        """
        Run the agent loop for a group of queries.
        """
        rollout_tasks = []
        # grpo group_size
        for _ in range(group_size):
            task = asyncio.create_task(self.run_one_query(copy.deepcopy(input_ids)))
            rollout_tasks.append(task)

        try:
            task_results = await asyncio.gather(*rollout_tasks)
            rollout_result = self.get_rollout_result(task_results, answer)
            if group_id is not None:
                rollout_result.group_id = group_id
                rollout_result.task_name = task_name
            await output_channel.put(rollout_result, async_op=True).async_wait()
        except GroupDiscarded as e:
            # Only this failure mode is per-group: record the discard and
            # let the group end normally. The remaining trajectories of
            # this group are pointless now; cancel them instead of
            # letting them finish generating and throwing the data away.
            if group_id is None:
                raise
            for task in rollout_tasks:
                task.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)
            await output_channel.put(
                DiscardedGroup(group_id=group_id, reason=e.reason),
                async_op=True,
            ).async_wait()

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
            return {}

    async def run_agentloop_rollout_continuously(
        self,
        input_channel: Channel,
        output_channel: Channel,
        agent_metric_channel: Channel,
    ):
        """Serve rollout requests until stopped; used by the async runner.

        Each request spawns one task per (input_ids, answer) group; finished
        groups put `{"bucket": ..., "metrics": ...}` on
        `agent_metric_channel`. Stops via `stop()` or a group failure, both
        by cancelling this method's task; a clean stop returns normally, a
        group failure re-raises here.
        """
        assert self._agentloop_task is None, (
            "agent loop task is not None but the continuous loop is called."
        )
        self._agentloop_task = asyncio.current_task()
        self._producer_error = None
        self._agent_metric_channel = agent_metric_channel
        self._inflight_groups: set[asyncio.Task] = set()
        self._idle_seconds = 0.0
        self._idle_window_start = time.monotonic()
        self._idle_since = self._idle_window_start
        try:
            while True:
                request = await input_channel.get(async_op=True).async_wait()
                group_ids = request.group_ids
                task_names = request.task_names
                for index, (input_ids, answer) in enumerate(
                    zip(request.input_ids, request.answers)
                ):
                    group_id = group_ids[index] if group_ids is not None else None
                    task_name = task_names[index] if task_names is not None else None
                    task = asyncio.create_task(
                        self.run_agentloop_rollout_group(
                            input_ids,
                            answer,
                            request.n,
                            output_channel,
                            group_id,
                            task_name,
                        )
                    )
                    task.add_done_callback(self._on_group_done)
                    self._mark_inflight_transition(False)
                    self._inflight_groups.add(task)
        except asyncio.CancelledError:
            # Cancelled either by stop() or, via _on_group_done, by a group
            # failure; in the latter case re-raise the recorded error.
            if self._producer_error is not None:
                raise self._producer_error
            return
        finally:
            # In-flight groups are dropped on any exit: the loop only stops
            # when training ended or a group failed. Keep the loop task
            # reference until they are awaited: a group ending with a
            # non-cancel exception after the reference is cleared would make
            # _on_group_done cancel None and lose the original error.
            for task in self._inflight_groups:
                task.cancel()
            if self._inflight_groups:
                await asyncio.gather(*self._inflight_groups, return_exceptions=True)
            self._agentloop_task = None

    def _on_group_done(self, task: asyncio.Task):
        self._inflight_groups.discard(task)
        if not self._inflight_groups:
            self._mark_inflight_transition(True)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            agent_metrics = task.result()
            if agent_metrics is not None and agent_metrics:
                # `bucket` is reserved for grouping agent metrics by the weight
                # version that generated the group; it is always None here, and the
                # runner aggregates only None-keyed records.
                self._agent_metric_channel.put(
                    {"bucket": None, "metrics": agent_metrics}, async_op=True
                )
        elif self._producer_error is None:
            # Done callbacks cannot raise: record the error, cancel the
            # loop task, and let it re-raise the error on unwind.
            self.log_error(f"Agent loop group failed:\n{type(exc).__name__}: {exc}")
            self._producer_error = exc
            self._agentloop_task.cancel()

    def stop(self):
        """Stop the continuous agent loop; in-flight groups are cancelled."""
        if self._agentloop_task is not None and not self._agentloop_task.done():
            self._agentloop_task.cancel()

    def num_inflight_groups(self) -> int:
        """In-flight group count."""
        return len(self._inflight_groups)

    def _mark_inflight_transition(self, now_idle: bool):
        """Update idle accounting at an empty <-> non-empty transition.

        A transition to non-empty closes the idle segment that started at
        `_idle_since`; a transition to idle opens a new one. Transitions to
        the state we are already in are ignored, so callers need not check.
        """
        if now_idle:
            if self._idle_since is None:
                self._idle_since = time.monotonic()
        else:
            if self._idle_since is not None:
                self._idle_seconds += time.monotonic() - self._idle_since
                self._idle_since = None

    def pop_idle_stats(self) -> tuple[float, float]:
        """Idle seconds and window seconds since the last call; resets both.

        An ongoing idle segment counts up to the call time and continues
        from there in the next window. Called between training steps, so
        the window covers one full step.
        """
        now = time.monotonic()
        window = now - self._idle_window_start if self._idle_window_start else 0.0
        idle = self._idle_seconds
        if self._idle_since is not None:
            idle += now - self._idle_since
            self._idle_since = now
        self._idle_seconds = 0.0
        self._idle_window_start = now
        return idle, window

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

    def pre_process(self, prompt_ids: list[int]) -> dict[str, Any]:
        return {}

    async def post_process(
        self, generate_context: dict[str, Any], output: AgentLoopOutput
    ) -> dict[str, Any]:
        return output

    async def run_one_query_turn(
        self,
        generate_context: dict[str, Any],
        trace_prints: list[dict],
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
    ):
        (
            is_continue,
            llm_response_ids,
            llm_response_mask,
            llm_response_logprobs,
            llm_response_text,
        ) = await self.generate_llm_response(
            generate_context,
            trace_prints,
            problem_prompt_ids,
            turn_prompt_ids,
        )

        if not is_continue:
            return (
                False,
                llm_response_ids,
                llm_response_mask,
                llm_response_logprobs,
            )

        (
            is_continue,
            tool_response_ids,
            tool_response_mask,
            tool_response_logprobs,
        ) = await self.generate_tool_response(
            generate_context,
            trace_prints,
            problem_prompt_ids,
            turn_prompt_ids,
            llm_response_ids,
            llm_response_text,
        )
        append_ids = llm_response_ids + tool_response_ids
        append_mask = llm_response_mask + tool_response_mask
        append_logprobs = None
        if self.return_logprobs:
            append_logprobs = llm_response_logprobs + tool_response_logprobs

        return (
            is_continue,
            append_ids,
            append_mask,
            append_logprobs,
        )

    async def run_one_query(self, prompt_ids: list[int]) -> AgentLoopOutput:
        prompt_ids = prompt_ids[: self.max_prompt_len]
        problem_prompt_ids = copy.deepcopy(prompt_ids)
        generate_context = self.pre_process(prompt_ids)
        trace_prints = []
        response_mask = []
        response_logprobs = None
        if self.return_logprobs:
            response_logprobs = []
        while True:
            (
                is_continue,
                append_ids,
                append_mask,
                append_logprobs,
            ) = await self.run_one_query_turn(
                generate_context,
                trace_prints,
                problem_prompt_ids,
                prompt_ids,
            )
            prompt_ids += append_ids
            response_mask += append_mask
            if self.return_logprobs:
                response_logprobs += append_logprobs
            if not is_continue:
                break

        # Separate prompt and response
        response_ids = prompt_ids[len(problem_prompt_ids) :]

        output = AgentLoopOutput(
            prompt_ids=problem_prompt_ids,
            prompt_text=self.tokenizer.decode(problem_prompt_ids),
            response_ids=response_ids,
            response_text=self.tokenizer.decode(response_ids),
            response_mask=response_mask,
            trace_prints=trace_prints,
            response_logprobs=response_logprobs,
        )
        return await self.post_process(generate_context, output)


class MultiAgentLoopWorker(AgentLoopWorker):
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
        session_server_cfg = cfg.get("session_server", {}) or {}
        self._verify_assembly = bool(session_server_cfg.get("verify_assembly", False))

    async def run_agentloop_rollout_group(
        self,
        input_dict: dict,
        answer: str,
        group_size: int,
        output_channel: Channel,
        group_id: Optional[int] = None,
        task_name: Optional[str] = None,
    ):
        """
        Run the agent loop for a group of queries.

        Args:
            input_dict: Input dictionary containing 'input_ids' and 'answer'
            answer: Ground truth answer
            group_size: Number of rollouts per query (k samples for pass@k)
            output_channel: Channel receiving each result
            group_id: Id of the query group, carried on the result so the
                receiver can route it; None in synchronous training.
            task_name: Agent task name of the group, carried on the result
                like group_id.
        """
        if self.session_server_client is not None:
            return await self._run_session_server_group(
                input_dict, answer, group_size, group_id, task_name
            )
        rollout_tasks: list[asyncio.Task[MultiAgentLoopOutput]] = []
        # grpo group_size
        for _ in range(group_size):
            task = asyncio.create_task(
                self.run_one_query(copy.deepcopy(input_dict), answer=answer)
            )
            rollout_tasks.append(task)

        try:
            task_results = await asyncio.gather(*rollout_tasks)

            # For eval mode, allow multiple samples (group_size=k) to compute pass@k and avg@k
            extra_fields = self.gen_extra_fields(task_results, answer)
            rollout_result = self.get_rollout_result(task_results, *extra_fields)
            if group_id is not None:
                rollout_result.group_id = group_id
                rollout_result.task_name = task_name
            agent_metrics = self.get_rollout_metrics(rollout_result)

            await output_channel.put(rollout_result, async_op=True).async_wait()
        except GroupDiscarded as e:
            # Only this failure mode is per-group: record the discard and
            # let the group end normally. The remaining trajectories of
            # this group are pointless now; cancel them instead of
            # letting them finish generating and throwing the data away.
            if group_id is None:
                raise
            for task in rollout_tasks:
                task.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)
            agent_metrics = {}
            await output_channel.put(
                DiscardedGroup(group_id=group_id, reason=e.reason),
                async_op=True,
            ).async_wait()
        return agent_metrics

    async def _run_session_server_group(
        self,
        input_dict: dict,
        answer: str,
        group_size: int,
        group_id: Optional[int],
        task_name: Optional[str],
    ) -> dict:
        """Run one query group over the session server transport.

        This side owns the lifecycle — one session per trajectory, close
        with the kept lengths and extra fields, reward, abort on failure —
        while the session server writes the training payload and reports
        the group itself.
        """
        assert group_id is not None and task_name is not None, (
            "the session server path needs the group and task identity the "
            f"buffer worker stamps on its requests; got group_id={group_id!r}, "
            f"task_name={task_name!r}"
        )
        rollout_tasks = [
            asyncio.create_task(
                self._run_session_query(
                    input_dict, answer, group_id, traj_index, task_name, group_size
                )
            )
            for traj_index in range(group_size)
        ]
        try:
            bindings_outputs = await asyncio.gather(*rollout_tasks)
        except (GroupDiscarded, SessionDiscardedError) as exc:
            # The discard is reported by the session server — it discarded
            # the group already, or does so on this abort. The remaining
            # trajectories of the group are pointless now; cancel them.
            for task in rollout_tasks:
                task.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)
            await self.session_server_client.abort(group_id, exc.reason)
            return {}
        task_results = [output for _, output in bindings_outputs]
        extra_turn, extra_traj, extra_group, extra_train = self.gen_extra_fields(
            task_results, answer
        )
        assert extra_turn is None and extra_traj is None and extra_group is None, (
            "the session server path does not support turn / traj / group "
            f"extra fields; got turn={extra_turn!r}, traj={extra_traj!r}, "
            f"group={extra_group!r}"
        )
        verify_result = (
            self._verify_reference(
                bindings_outputs, (extra_turn, extra_traj, extra_group, extra_train)
            )
            if self._verify_assembly
            else None
        )
        # extra_train values run trajectory-major then turn order; each
        # trajectory's close carries its own slice of every field.
        traj_fields_slices: list[dict] = []
        start = 0
        for output in task_results:
            end = start + len(output.single_turn_outputs)
            traj_fields_slices.append(
                {
                    name: values[start:end]
                    for name, values in (extra_train or {}).items()
                }
            )
            start = end
        try:
            last_index = group_size - 1
            for index, (binding, output) in enumerate(bindings_outputs):
                kept_lens = self._declared_kept_lens(binding, output)
                verify = verify_result if index == last_index else None
                await self.session_server_client.close(
                    group_id,
                    binding.session_id,
                    kept_lens,
                    traj_fields_slices[index],
                    verify,
                )
                await self.session_server_client.reward(
                    group_id, binding.session_id, self._trajectory_reward(output)
                )
        except SessionDiscardedError as exc:
            # The group was voided server-side (a payload write failed
            # after the last generate); its discard record is already with
            # the buffer worker, so this side drops the group quietly.
            self.log_warning(
                f"Dropping group {group_id} at close/reward: the session "
                f"server discarded it ({exc.reason})."
            )
            return {}
        return {}

    async def _run_session_query(
        self,
        input_dict: dict,
        answer: str,
        group_id: int,
        traj_index: int,
        task_name: str,
        group_size: int,
    ) -> tuple[SessionBinding, MultiAgentLoopOutput]:
        """Run one trajectory with its session bound to this task."""
        session_id = await self.session_server_client.create_session(
            group_id, traj_index, group_size, task_name
        )
        binding = SessionBinding(
            group_id=group_id, traj_index=traj_index, session_id=session_id
        )
        token = self._session_binding.set(binding)
        try:
            output = await self.run_one_query(copy.deepcopy(input_dict), answer=answer)
        finally:
            self._session_binding.reset(token)
        return binding, output

    def _declared_kept_lens(
        self, binding: SessionBinding, output: MultiAgentLoopOutput
    ) -> dict[int, int]:
        """The kept length per turn, asserted against the call records.

        A task whose reported outputs disagree with the calls it made —
        a turn missing, a prompt rewritten, a response that is not a
        prefix of the engine's output — fails here, before the server
        slices the payload, instead of assembling wrong training data.
        """
        outputs = output.single_turn_outputs
        assert len(outputs) == len(binding.turns), (
            f"a trajectory closed with {len(outputs)} turn outputs after "
            f"{len(binding.turns)} generate calls; the session server path "
            "requires them to correspond one to one"
        )
        kept_lens: dict[int, int] = {}
        for turn_index, (turn_output, record) in enumerate(zip(outputs, binding.turns)):
            if turn_output.extra_fields.get("not_training", False):
                raise RuntimeError(
                    "the session server path does not support not_training turns"
                )
            assert turn_output.prompt_ids == record["input_ids"], (
                f"turn {turn_index} of trajectory {binding.traj_index} "
                f"reported prompt ids {turn_output.prompt_ids!r} but sent "
                f"{record['input_ids']!r} to generate"
            )
            kept = len(turn_output.response_ids)
            assert record["output_ids"][:kept] == turn_output.response_ids, (
                f"turn {turn_index} of trajectory {binding.traj_index} "
                f"reported a response that is not a prefix of the engine's "
                f"output; reported {kept} tokens, engine output starts "
                f"{record['output_ids'][:kept]!r}"
            )
            kept_lens[turn_index] = kept
        return kept_lens

    def _trajectory_reward(self, output: MultiAgentLoopOutput) -> float:
        """The trajectory-level score, asserted uniform across its turns.

        The reward route takes one scalar per trajectory; a task scoring
        turns differently needs per-turn verdict rewards, which the session
        server path does not support.
        """
        scores = [turn.reward_score for turn in output.single_turn_outputs]
        assert scores and all(score is not None for score in scores), (
            "every turn must carry a reward_score before the session path "
            f"closes its trajectory; got {scores!r}"
        )
        assert all(score == scores[0] for score in scores), (
            "turn reward scores must agree within a trajectory to report "
            f"one scalar; got {scores!r}"
        )
        return scores[0]

    def _verify_reference(
        self,
        bindings_outputs: list[tuple[SessionBinding, MultiAgentLoopOutput]],
        extra_fields: tuple,
    ) -> dict:
        """The reference for the assembly verification, group-level.

        Rides with the last close. ``result`` is what the payload-assembled
        result is compared against field by field; the actor checks the
        server's forwarded sampling params by merging
        ``caller_sampling_params`` over the config defaults.
        """
        task_results = [output for _, output in bindings_outputs]
        reference = self.get_rollout_result(task_results, *extra_fields)
        return {
            "result": {
                "num_sequence": reference.num_sequence,
                "group_size": reference.group_size,
                "idx_to_traj": reference.idx_to_traj,
                "input_ids": reference.input_ids,
                "prompt_lengths": reference.prompt_lengths,
                "response_lengths": reference.response_lengths,
                "rollout_logprobs": reference.rollout_logprobs,
                "rewards": reference.rewards,
                "is_end": reference.is_end,
                "extra_fields_train": reference.extra_fields_train,
            },
            "caller_sampling_params": {
                binding.traj_index: [
                    record["sampling_params"] for record in binding.turns
                ]
                for binding, _ in bindings_outputs
            },
        }

    async def run_agentloop_rollout(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        """
        Run the agent loop for multiple queries.

        Args:
            input_channel: Channel to receive rollout requests
            output_channel: Channel receiving each result
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
        task_results: list[MultiAgentLoopOutput],
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
            extra_fields_turn = {k: [] for k in self.extra_keys_turn}
        extra_fields_traj = None
        if self.extra_keys_traj is not None:
            extra_fields_traj = {k: [] for k in self.extra_keys_traj}

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
        return extra_fields_turn, extra_fields_traj, None, {}

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
        return aggregate_agent_metrics(agent_metrics_list)

    def get_rollout_metrics(
        self,
        rollout_result: DynamicRolloutResult,
    ) -> dict:
        """Hook for subclasses to compute task-specific rollout metrics."""
        return {}

    def get_rollout_result(
        self,
        task_results: list[MultiAgentLoopOutput],
        extra_fields_turn: Optional[dict],
        extra_fields_traj: Optional[dict],
        extra_fields_group: Optional[dict],
        extra_fields_train: dict,
    ) -> DynamicRolloutResult:
        """Collect a group of turn-level outputs into `DynamicRolloutResult`.

        Args:
            task_results: Multi-turn outputs for one query group.
            extra_fields_turn: Turn-level extra fields.
            extra_fields_traj: Trajectory-level extra fields.
            extra_fields_group: Group-level extra fields.
            extra_fields_train: Training-only fields (e.g. regroup indices).

        Returns:
            A packed `DynamicRolloutResult` ready for downstream training/eval.
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
        version_start = []
        version_end = []

        # Flatten all retained turns while keeping trajectory mapping.
        for idx, task_result in enumerate(task_results):
            for single_turn_output in task_result.single_turn_outputs:
                single_turn_output: AgentLoopOutput
                if single_turn_output.extra_fields.get("not_training", False):
                    continue
                idx_to_traj.append(idx)
                prompt_lengths.append(len(single_turn_output.prompt_ids))
                response_lengths.append(len(single_turn_output.response_ids))
                input_ids.append(
                    single_turn_output.prompt_ids + single_turn_output.response_ids
                )
                if self.return_logprobs:
                    assert len(single_turn_output.response_logprobs) == len(
                        single_turn_output.response_ids
                    ), "response_logprobs should have the same length as response_ids"
                    rollout_logprobs.append(single_turn_output.response_logprobs)
                is_end.append(single_turn_output.is_end)
                rewards.append(single_turn_output.reward_score)
                version_start.append(single_turn_output.version_start)
                version_end.append(single_turn_output.version_end)

        # Weight versions are either stamped on every turn or on none: mixing
        # would silently drop the version-less turns from staleness statistics.
        if any(v is not None for v in version_start + version_end):
            assert all(v is not None for v in version_start) and all(
                v is not None for v in version_end
            ), (
                "agent loop outputs mix turns with weight versions and turns "
                "without (version_start/version_end)"
            )
        else:
            version_start = None
            version_end = None

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
            version_start=version_start,
            version_end=version_end,
            extra_fields_turn=extra_fields_turn,
            extra_fields_traj=extra_fields_traj,
            extra_fields_group=extra_fields_group,
            extra_fields_train=extra_fields_train,
        )

    async def pre_process_query(self, prompt_ids: list[int]) -> dict[str, Any]:
        raise NotImplementedError("pre_process_query is not implemented")

    async def post_process_query(
        self, generate_context: dict[str, Any], output: MultiAgentLoopOutput
    ) -> dict[str, Any]:
        raise NotImplementedError("post_process_query is not implemented")

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
        prompt_ids, generate_context = await self.pre_process_query(*args, **kwargs)
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
