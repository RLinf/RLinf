# Copyright 2025 The RLinf Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import json
import re
from typing import Optional

from omegaconf import DictConfig

from rlinf.agents.wideseek_r1.utils.metrics import _compute_rollout_metrics
from rlinf.agents.wideseek_r1.utils.prompt_utils import (
    get_access_summary_messages,
    get_access_summary_tool_message,
    get_access_tool_message,
    get_first_turn_hint,
    get_next_turn_hint,
    get_planner_subtask_failed_message,
    get_planner_subtask_result_message,
    get_prompt_planner,
    get_prompt_single_agent,
    get_prompt_worker,
    get_search_tool_message,
)
from rlinf.agents.wideseek_r1.utils.reward import (
    credit_assignment,
    extract_final_answer,
    get_final_reward_score,
)
from rlinf.agents.wideseek_r1.utils.sglang_client import SGLangClient
from rlinf.agents.wideseek_r1.utils.tool_description import (
    tools_description_en,
    tools_description_zh,
)
from rlinf.data.io_struct import DynamicRolloutResult
from rlinf.data.tool_call.tool_io_struct import (
    ToolRequest,
    ToolResponse,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)


class WideSeekR1AgentLoopWorker(MultiAgentLoopWorker):
    """Multi-turn WideSeek-R1 agent worker for MAS and single-agent workflows."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.extra_keys_turn = [
            "subtask_count",
            "search_count",
            "access_count",
            "tool_call_info",
            "prompt_text",
            "response_text",
            "role",
        ]
        self.extra_keys_traj = [
            "origin_question",
            "final_answer",
            "final_answer_text",
            "num_valid_planner_turns",
            "num_valid_worker_turns",
            "total_turn_list",
            "final_answer_format",
            "llm_reward",
        ]

        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        self.max_total_len = int(self.cfg.runner.seq_length)

        self.use_access_summary = self.cfg.tools.get("use_access_summary", False)

        self.use_fixed_rollout = cfg.rollout.get("use_fixed_worker", False)
        self.fixed_role = self.cfg.agentloop.get("fixed_role", None)
        if self.use_fixed_rollout:
            assert self.fixed_role

        self.workflow = self.cfg.agentloop.get("workflow", "mas")
        self.is_hybrid = self.cfg.data.get("is_hybrid", False)

        llm_ip = self.cfg.agentloop.get("llm_ip", "")
        llm_port = self.cfg.agentloop.get("llm_port", "")
        llm_type = self.cfg.agentloop.get("llm_type", "")
        self.sgl_client = SGLangClient(llm_ip, llm_port, llm_type)
        assert self.return_logprobs if not self.is_eval else True

    async def extract_tool_calls(
        self, response_text: str, role: str
    ) -> tuple[str, list[ToolRequest]]:
        """Parse the first `<tool_call>` block and convert it to internal requests.

        Args:
            response_text: Decoded model response that may contain tool-call JSON.
            role: Current role (`planner`, `worker`, or `single`).

        Returns:
            A tuple of `(tool_requests, tool_call_info)` where `tool_call_info`
            summarizes subtask/search/access counts for metrics.
        """
        function_calls = []

        # Extract all <tool_call> tags
        tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        max_workers_per_planner = self.cfg.agentloop.get("max_workers_per_planner", 10)
        max_toolcall_per_worker = self.cfg.agentloop.get("max_toolcall_per_worker", 5)

        tool_call_match = tool_call_regex.findall(response_text)

        subtask_count = 0
        search_count = 0
        access_count = 0

        if tool_call_match:
            tool_call_str = tool_call_match[0]
            try:
                # Parse JSON from tool call
                tool_call_json = json.loads(tool_call_str.strip())
                tool_name = tool_call_json.get("name")
                tool_arguments = tool_call_json.get("arguments", {})

                if role == "planner":
                    # Planner: handle create_sub_agents tool
                    if tool_name == "create_sub_agents":
                        # Extract sub_agents array
                        sub_agents = tool_arguments.get("sub_agents", [])
                        for sub_agent in sub_agents[:max_workers_per_planner]:
                            # Skip if not a dict or missing prompt
                            if not isinstance(sub_agent, dict):
                                continue
                            prompt = sub_agent.get("prompt", "")
                            if not prompt:
                                continue
                            function_calls.append(
                                ToolRequest(
                                    name="subtask", arguments={"subtask": prompt}
                                )
                            )
                            subtask_count += 1
                elif role == "worker":
                    # Worker: handle search and access tools
                    if tool_name == "search":
                        # Extract searches array
                        searches = tool_arguments.get("queries", [])
                        for search_item in searches[:max_toolcall_per_worker]:
                            # Skip if not a dict or missing query
                            if not isinstance(search_item, dict):
                                continue
                            query = search_item.get("query", "")
                            if not query:
                                continue
                            topk = search_item.get("count", None)
                            if topk:
                                function_calls.append(
                                    ToolRequest(
                                        name="search",
                                        arguments={"query": query, "topk": topk},
                                    )
                                )
                            else:
                                function_calls.append(
                                    ToolRequest(
                                        name="search", arguments={"query": query}
                                    )
                                )
                            search_count += 1

                    elif tool_name == "access":
                        # Extract accesses array
                        accesses = tool_arguments.get("urls", [])
                        for access_item in accesses[:max_toolcall_per_worker]:
                            # Skip if not a dict or missing url
                            if not isinstance(access_item, dict):
                                continue
                            url = access_item.get("url", "")
                            info_to_extract = access_item.get("info_to_extract", None)
                            if not url:
                                continue
                            function_calls.append(
                                ToolRequest(
                                    name="access",
                                    arguments={
                                        "url": url,
                                        "access_token": 25000,
                                        "info_to_extract": info_to_extract,
                                    },
                                )
                            )
                            access_count += 1
                elif role == "single":
                    if tool_name == "search":
                        query = tool_arguments.get("query", "")
                        if query:
                            topk = tool_arguments.get("count", None)
                            if topk:
                                function_calls.append(
                                    ToolRequest(
                                        name="search",
                                        arguments={"query": query, "topk": topk},
                                    )
                                )
                            else:
                                function_calls.append(
                                    ToolRequest(
                                        name="search", arguments={"query": query}
                                    )
                                )
                            search_count = 1

                    elif tool_name == "access":
                        url = tool_arguments.get("url", "")
                        if url:
                            info_to_extract = tool_arguments.get(
                                "info_to_extract", None
                            )
                            function_calls.append(
                                ToolRequest(
                                    name="access",
                                    arguments={
                                        "url": url,
                                        "access_token": 25000,
                                        "info_to_extract": info_to_extract,
                                    },
                                )
                            )
                            access_count = 1
            except Exception:
                pass

        tool_call_info = {
            "subtask": subtask_count,
            "search": search_count,
            "access": access_count,
            "role": role,
        }
        if function_calls == []:
            tool_call_info = None
        return function_calls, tool_call_info

    async def access_sumamry(self, info_to_extract, page_content):
        """Summarize access content to keep context compact for follow-up turns.

        Args:
            info_to_extract: Focus information requested by the worker.
            page_content: Raw page text returned by the access tool.

        Returns:
            A short summary string for tool feedback.
        """
        if page_content == "No More Information is Found for this URL.":
            return "No useful Information is Found under this URL."

        messages = get_access_summary_messages(info_to_extract, page_content)
        result_text = await self.sgl_client.call_sglang_api(messages)
        return result_text

    async def worker_call(
        self,
        worker_request: ToolRequest,
        main_task: str,
        is_markdown: bool,
        language: str,
        sub_traj_id: int,
    ) -> tuple[list[AgentLoopOutput], str]:
        """Execute one planner-created subtask through the worker role loop.

        Args:
            worker_request: Planner output converted to a `subtask` tool request.
            main_task: Original user question for worker grounding.
            is_markdown: Whether this sample expects markdown-table final answers.
            language: Prompt language (`en` or `zh`).
            sub_traj_id: Sub-trajectory index used for training regrouping.

        Returns:
            Worker turn outputs, worker summary text, turn statistics, and failure flag.
        """
        assert worker_request.name == "subtask", (
            f"Expected 'subtask' tool, got {worker_request.name}"
        )
        assert "subtask" in worker_request.arguments, (
            f"Missing 'subtask' in arguments: {worker_request.arguments}"
        )
        assert sub_traj_id > 0
        subtask = worker_request.arguments["subtask"]

        (
            worker_outputs_buffer,
            answer_text,
            total_turn_list,
            task_failed,
            _,
        ) = await self.run_one_query_role(
            question=subtask,
            role="worker",
            sub_traj_id=sub_traj_id,
            main_task=main_task,
            is_markdown=is_markdown,
            language=language,
        )
        return worker_outputs_buffer, answer_text, total_turn_list, task_failed

    def _set_max_turns(self, role: str) -> int:
        if role == "planner":
            return self.cfg.agentloop.get("max_planner_turns", 10)
        if role == "single":
            return self.cfg.agentloop.get("max_sa_turns", 50)
        if role == "worker":
            return self.cfg.agentloop.get("max_worker_turns", 20)
        raise ValueError(f"illegal role {role}")

    def _build_message_history_and_tools(
        self,
        origin_question: str,
        role: str,
        is_markdown: bool,
        language: str,
        main_task: str | None,
    ) -> tuple[list[dict], list[dict]]:
        """Build role-specific prompt history and exposed tool descriptions.

        Args:
            origin_question: Query text for this role loop.
            role: Current role (`planner`, `worker`, or `single`).
            is_markdown: Whether markdown answer format is required.
            language: Prompt language identifier (`en` or `zh`).
            main_task: Parent task text required for worker prompts.

        Returns:
            A tuple of `(message_history, tools)` for chat-template rendering.
        """
        tools_description = (
            tools_description_zh if language == "zh" else tools_description_en
        )
        if role == "planner":
            message_history = get_prompt_planner(
                origin_question, is_markdown=is_markdown, language=language
            )
            tools = [tools_description["create_sub_agents"]]
        elif role == "worker":
            assert main_task is not None, "Worker must have main_task provided"
            message_history = get_prompt_worker(
                main_task, origin_question, language=language
            )
            tools = [tools_description["search"], tools_description["access"]]
        elif role == "single":
            message_history = get_prompt_single_agent(
                origin_question, is_markdown=is_markdown, language=language
            )
            tools = [
                tools_description["search_single_agent"],
                tools_description["access_single_agent"],
            ]
        else:
            raise ValueError(f"Invalid role: {role}")
        return message_history, tools

    def _mark_role_failed_turns(
        self,
        *,
        output_buffer: list[AgentLoopOutput],
        role: str,
        turn_idx: int,
        max_turns: int,
        succ_end: bool,
        context_failed: bool,
        tool_response_failed: bool,
    ) -> bool:
        """Apply failure flags to turns for one role and return task failure.

        Args:
            output_buffer: Collected per-turn outputs for this role execution.
            role: Current role whose turns should be marked.
            turn_idx: Last executed loop index (zero-based).
            max_turns: Maximum allowed turns for this role.
            succ_end: Whether the role loop ended successfully.
            context_failed: Whether prompt/response length hit context limit.
            tool_response_failed: Whether tool feedback exceeded available space.

        Returns:
            Boolean task failure indicator for this role execution.
        """
        max_turn_limit_failed = (
            not succ_end
            and not context_failed
            and not tool_response_failed
            and turn_idx + 1 >= max_turns
        )

        if max_turn_limit_failed:
            for turn in output_buffer:
                if turn.extra_fields["role"] == role:
                    turn.extra_fields["max_turn_limit_failed"] = True

        if context_failed or tool_response_failed:
            for turn in output_buffer:
                if turn.extra_fields["role"] == role:
                    turn.extra_fields["context_failed"] = True

        if (
            context_failed
            and len(output_buffer) >= 1
            and len(output_buffer[-1].response_ids) >= 8000
        ):
            output_buffer[-1].extra_fields["turn_repeat_failed"] = True

        task_failed = max_turn_limit_failed or context_failed or tool_response_failed
        assert task_failed != succ_end
        return task_failed

    async def run_one_query_role(
        self,
        question: str,
        role: str,
        sub_traj_id: int,
        main_task: str | None = None,
        is_markdown: bool = False,
        language: str = "en",
    ) -> tuple[list[AgentLoopOutput], str]:
        """Run one query under a specific role until stop, failure, or turn budget.

        Args:
            question: Role-specific input question (main query or subtask).
            role: One of `planner`, `worker`, or `single`.
            sub_traj_id: Sub-trajectory id for downstream regrouping.
            main_task: Original task text required when `role == "worker"`.
            is_markdown: Whether markdown answer format is required.
            language: Prompt language.

        Returns:
            Tuple of `(output_buffer, answer_text, total_turn_list, task_failed, succ_end)`.
        """

        origin_question = question
        output_buffer = []
        total_turn_list = []

        message_history, tools = self._build_message_history_and_tools(
            origin_question=origin_question,
            role=role,
            is_markdown=is_markdown,
            language=language,
            main_task=main_task,
        )
        max_turns = self._set_max_turns(role=role)

        turn_hint = get_first_turn_hint(max_turns=max_turns, language=language)
        assert message_history[-1]["role"] == "user"
        message_history[-1]["content"] = message_history[-1]["content"] + turn_hint

        prompt_ids = self.tokenizer.apply_chat_template(
            message_history, tokenize=True, add_generation_prompt=True, tools=tools
        )
        prompt_ids = prompt_ids[: self.max_total_len]

        # Initialize tracking variables
        context_failed = False
        tool_response_failed = False

        succ_end = False
        sub_traj_num = 0

        turn_idx = -1
        for turn_idx in range(max_turns):
            max_resp_len = self.max_total_len - len(prompt_ids)
            if max_resp_len <= 0:
                context_failed = True
                break

            if role == self.fixed_role and self.use_fixed_rollout:
                generate_result = await self.generate(
                    prompt_ids,
                    sampling_params={"max_new_tokens": max_resp_len},
                    rollout_name="subworker",
                )
                generate_result["logprobs"] = [0.0] * len(generate_result["output_ids"])
            else:
                generate_result = await self.generate(
                    prompt_ids,
                    sampling_params={"max_new_tokens": max_resp_len},
                )

            response_ids = generate_result["output_ids"]
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]

            response_text = self.tokenizer.decode(response_ids)

            tool_requests, tool_call_info = await self.extract_tool_calls(
                response_text, role=role
            )

            output_buffer.append(
                AgentLoopOutput(
                    prompt_ids=copy.deepcopy(prompt_ids),
                    response_ids=copy.deepcopy(response_ids),
                    prompt_text=copy.deepcopy(self.tokenizer.decode(prompt_ids)),
                    response_text=response_text,
                    is_end=generate_result["finish_reason"] == "length",
                    response_logprobs=generate_result["logprobs"]
                    if self.return_logprobs
                    else None,
                    extra_fields={
                        "role": role,
                        "idx_to_sub_traj": sub_traj_id,
                        "context_failed": False,
                        "max_turn_limit_failed": False,
                        "turn_repeat_failed": False,
                    },
                    tool_call_info=tool_call_info
                    if tool_call_info
                    else None,  # if passed, must have tool call
                )
            )

            prompt_ids += response_ids

            if len(response_ids) == max_resp_len:
                context_failed = True
                break

            # Extract tool calls
            if tool_requests == []:
                succ_end = True
                break

            # Handle tool calls based on role
            tasks = []
            tool_messages = []
            worker_buffer = []
            worker_turn_list = []
            if role == "planner":
                assert sub_traj_id == 0
                # Planner fans out multiple sub-agents in parallel.
                for i, tool_request in enumerate(tool_requests, start=1):
                    tasks.append(
                        self.worker_call(
                            tool_request,
                            origin_question,
                            is_markdown,
                            language,
                            sub_traj_id + i + sub_traj_num,
                        )
                    )
                sub_traj_num += len(tasks)
                worker_results = await asyncio.gather(*tasks)

                tool_messages_text = []
                for idx, (
                    worker_outputs_buffer,
                    worker_summary,
                    total_turn_list_worker,
                    task_failed,
                ) in enumerate(worker_results):
                    worker_buffer.extend(worker_outputs_buffer)
                    worker_turn_list.extend(total_turn_list_worker)
                    # assert len(worker_outputs_buffer) == sum(total_turn_list_worker) and len(total_turn_list_worker) >=1
                    # Format tool response with both request and result
                    subtask_text = tool_requests[idx].arguments["subtask"]
                    if not task_failed:
                        tool_messages_text.append(
                            get_planner_subtask_result_message(
                                subtask_idx=idx + 1,
                                subtask_text=subtask_text,
                                worker_summary=worker_summary,
                                language=language,
                            )
                        )
                    else:
                        tool_messages_text.append(
                            get_planner_subtask_failed_message(
                                subtask_idx=idx + 1,
                                subtask_text=subtask_text,
                                language=language,
                            )
                        )

                turn_hint = get_next_turn_hint(
                    next_turn_idx=turn_idx + 2,
                    max_turns=max_turns,
                    language=language,
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": "\n\n".join(tool_messages_text) + turn_hint,
                    }
                )

            else:
                # Worker/single executes search/access tools in parallel.
                for tool_request in tool_requests:
                    tasks.append(self.tool_call(tool_request))
                tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)

                tool_messages_text = []
                access_summary_jobs = []
                for idx, (tool_request, tool_response) in enumerate(
                    zip(tool_requests, tool_responses)
                ):
                    # Include the original request and the result
                    if tool_request.name == "search":
                        query = tool_request.arguments["query"]
                        tool_messages_text.append(
                            get_search_tool_message(
                                query=query,
                                search_result=tool_response.text,
                                language=language,
                            )
                        )
                    elif tool_request.name == "access":
                        url = tool_request.arguments["url"]
                        info_to_extract = tool_request.arguments["info_to_extract"]
                        page_content = tool_response.text
                        if self.use_access_summary:
                            tool_messages_text.append(None)
                            coro = self.access_sumamry(info_to_extract, page_content)
                            access_summary_jobs.append(
                                (idx, url, info_to_extract, coro)
                            )
                        else:
                            tool_messages_text.append(
                                get_access_tool_message(
                                    url=url,
                                    page_content=page_content,
                                    language=language,
                                )
                            )
                    else:
                        raise ValueError(
                            f"Unknown tool request name: {tool_request.name}"
                        )

                if self.use_access_summary and access_summary_jobs:
                    coros = [job[-1] for job in access_summary_jobs]
                    summaries = await asyncio.gather(*coros)
                    for job, summary in zip(access_summary_jobs, summaries):
                        idx, url, info_to_extract, _ = job
                        tool_messages_text[idx] = get_access_summary_tool_message(
                            url=url,
                            info_to_extract=info_to_extract,
                            summary=summary,
                            language=language,
                        )

                turn_hint = get_next_turn_hint(
                    next_turn_idx=turn_idx + 2,
                    max_turns=max_turns,
                    language=language,
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": "\n\n".join(tool_messages_text) + turn_hint,
                    }
                )

            # Tokenize tool responses
            tool_response_ids = self.get_tool_response_ids(tool_messages)
            max_tool_resp_len = self.max_total_len - len(prompt_ids)
            if len(tool_response_ids) >= max_tool_resp_len:
                tool_response_failed = True
                break

            prompt_ids += tool_response_ids
            output_buffer.extend(worker_buffer)
            total_turn_list.extend(worker_turn_list)

        task_failed = self._mark_role_failed_turns(
            output_buffer=output_buffer,
            role=role,
            turn_idx=turn_idx,
            max_turns=max_turns,
            succ_end=succ_end,
            context_failed=context_failed,
            tool_response_failed=tool_response_failed,
        )

        # Generate summary
        if role == "planner":
            answer_text = response_text.split("<|im_end|>")[0]
        elif role == "worker":
            answer_text = (
                response_text.split("</think>")[-1].split("<|im_end|>")[0].strip()
            )
        elif role == "single":
            answer_text = response_text.split("<|im_end|>")[0]

        if role == "worker":
            total_turn_list.append(turn_idx + 1)  # with no summary
        else:
            total_turn_list.append(turn_idx + 1)
        return output_buffer, answer_text, total_turn_list, task_failed, succ_end

    async def run_one_query(self, prompt_ids: list[int], *, answer) -> AgentLoopOutput:
        """Run one sample end-to-end and attach reward/training metadata.

        Args:
            prompt_ids: Tokenized query prompt from the dataset.
            answer: Label payload used for format extraction and reward scoring.

        Returns:
            A multi-turn output object containing all turns and trajectory metadata.
        """
        sub_traj_id = 0
        origin_question = self.tokenizer.decode(prompt_ids)
        language = answer.get("language", "en")
        if self.workflow == "sa":
            role = "single"
        else:
            role = "planner"

        is_markdown = answer["is_markdown"]

        (
            output_buffer,
            answer_text,
            total_turn_list,
            task_failed,
            succ_end,
        ) = await self.run_one_query_role(
            question=origin_question,
            role=role,
            sub_traj_id=sub_traj_id,
            is_markdown=is_markdown,
            language=language,
        )

        if is_markdown:
            final_answer_extract = extract_final_answer(answer_text, mode="markdown")
        else:
            final_answer_extract = extract_final_answer(answer_text, mode="boxed")

        # credit assignment
        norm_column = self.cfg.data.get("norm_column", False)
        llm_reward, format = await get_final_reward_score(
            origin_question,
            final_answer_extract,
            answer,
            is_markdown,
            norm_column,
            self.sgl_client,
        )

        output_buffer, train_buffer, final_answer_format, reward_score = (
            credit_assignment(
                agentloop_config=self.cfg.agentloop,
                output_buffer=output_buffer,
                llm_reward=llm_reward,
                succ_end=succ_end,
                answer_format=final_answer_extract is not None and format is True,
            )
        )

        for single_turn_output in output_buffer:
            single_turn_output.reward_score = reward_score
        for single_turn_output in train_buffer:
            single_turn_output.reward_score = reward_score

        for single_turn_output in output_buffer:
            single_turn_output.extra_fields["not_training"] = (
                False if self.is_eval else True
            )
        for single_turn_output in train_buffer:
            single_turn_output.extra_fields["not_training"] = False

        # Track valid turns for computing averages
        num_valid_planner_turns = 0
        num_valid_worker_turns = 0

        for single_turn_output in output_buffer:
            # Collect tool call info (keep all turns but track valid ones)
            single_turn_output: AgentLoopOutput
            subtask_count = 0
            search_count = 0
            access_count = 0
            if single_turn_output.tool_call_info is not None:
                role = single_turn_output.tool_call_info.get("role", "")
                subtask_count = single_turn_output.tool_call_info.get("subtask", 0)
                search_count = single_turn_output.tool_call_info.get("search", 0)
                access_count = single_turn_output.tool_call_info.get("access", 0)

                # Track valid turns by role
                if role == "planner":
                    assert subtask_count > 0
                    num_valid_planner_turns += 1
                elif role == "worker" or role == "single":
                    assert search_count > 0 or access_count > 0
                    num_valid_worker_turns += 1
            single_turn_output.extra_fields["subtask_count"] = subtask_count
            single_turn_output.extra_fields["search_count"] = search_count
            single_turn_output.extra_fields["access_count"] = access_count
            single_turn_output.extra_fields["tool_call_info"] = (
                single_turn_output.tool_call_info
            )
            single_turn_output.extra_fields["prompt_text"] = (
                single_turn_output.prompt_text
            )
            single_turn_output.extra_fields["response_text"] = (
                single_turn_output.response_text
            )

        output = MultiAgentLoopOutput(
            single_turn_outputs=output_buffer,
            trace_prints=[],  # Can add message_history tracking if needed
            extra_fields={
                "final_answer": final_answer_extract,
                "final_answer_text": answer_text,
                "reward": reward_score,
                "origin_question": origin_question,
                "llm_reward": llm_reward,
                "total_turn_list": total_turn_list if self.workflow == "mas" else None,
                "instance_id": answer["instance_id"],
                "num_valid_planner_turns": num_valid_planner_turns,
                "num_valid_worker_turns": num_valid_worker_turns,
                "final_answer_format": final_answer_format,
            },
        )
        return output

    def gen_extra_fields(
        self,
        task_results: list[MultiAgentLoopOutput],
        answer: str,
    ) -> Optional[dict]:
        """Build extra fields for turn/traj/group scopes and training regrouping.

        Args:
            task_results: Grouped rollout samples for the same input question.
            answer: Ground-truth answer payload for this group.

        Returns:
            Extra field dicts for turn-level, trajectory-level, group-level,
            and training-only fields.
        """
        extra_fields_turn, extra_fields_traj, *_ = super().gen_extra_fields(
            task_results, answer
        )

        roles = []
        for task_result in task_results:
            for single_turn_output in task_result.single_turn_outputs:
                if self.extra_keys_turn is not None:
                    for k in self.extra_keys_turn:
                        v = single_turn_output.extra_fields.get(k, None)
                        if (
                            k == "role"
                            and not single_turn_output.extra_fields["not_training"]
                        ):
                            roles.append(v)
        extra_fields_turn = {**extra_fields_turn, "roles": roles}

        extra_fields_group = {
            "answer": answer,
            "num_valid_planner_turns": sum(
                extra_fields_traj["num_valid_planner_turns"]
            ),
            "num_valid_worker_turns": sum(extra_fields_traj["num_valid_worker_turns"]),
        }

        idx_to_sub_traj = []
        for task_result in task_results:
            sub_traj_map = {}
            for single_turn_output in task_result.single_turn_outputs:
                if single_turn_output.extra_fields["not_training"]:
                    continue
                role_idx = single_turn_output.extra_fields["idx_to_sub_traj"]
                if role_idx not in sub_traj_map:
                    sub_traj_map[role_idx] = len(sub_traj_map)
                idx_to_sub_traj.append(sub_traj_map[role_idx])
        extra_fields_train = {"idx_to_sub_traj": idx_to_sub_traj}

        return (
            extra_fields_turn,
            extra_fields_traj,
            extra_fields_group,
            extra_fields_train,
        )

    def get_rollout_metrics(
        self,
        rollout_result: DynamicRolloutResult,
    ) -> dict:
        """Compute wideseek rollout metrics from packed dynamic rollout outputs.

        Args:
            rollout_result: Dynamic rollout structure produced by this worker.

        Returns:
            Aggregated metric dictionary for logging.
        """
        if self.is_eval:
            return {}

        rollout_batch = {
            "turn_subtask_counts": rollout_result.extra_fields_turn["subtask_count"],
            "turn_search_counts": rollout_result.extra_fields_turn["search_count"],
            "turn_access_counts": rollout_result.extra_fields_turn["access_count"],
            "num_valid_planner_turns": sum(
                rollout_result.extra_fields_traj["num_valid_planner_turns"]
            ),
            "num_valid_worker_turns": sum(
                rollout_result.extra_fields_traj["num_valid_worker_turns"]
            ),
            "total_turn_list_metric": rollout_result.extra_fields_traj[
                "total_turn_list"
            ],
            "final_answer_format": rollout_result.extra_fields_traj[
                "final_answer_format"
            ],
        }
        return _compute_rollout_metrics(
            rollout_batch=rollout_batch,
            idx_to_traj=rollout_result.idx_to_traj,
            num_trajectories=int(rollout_result.group_size),
        )
