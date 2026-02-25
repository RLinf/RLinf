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
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from rlinf.data.io_struct import DynamicRolloutResult
from rlinf.agents.wideseek_r1.utils.metrics import _compute_eval_metrics, _compute_mas_turn_metrics, _compute_tool_call_metrics
from rlinf.agents.wideseek_r1.utils.sglang_client import SGLangClient
from rlinf.agents.wideseek_r1.utils.reward import (
    compute_score_em,
    compute_score_f1,
    extract_final_answer,
)
from rlinf.agents.wideseek_r1.utils.tool_description import (
    tools_description_en,
    tools_description_zh,
)
from rlinf.agents.wideseek_r1.utils.prompt_utils import (
    get_access_summary_messages,
    get_prompt_planner,
    get_prompt_single_agent,
    get_prompt_worker,
)
from rlinf.data.tool_call.tool_io_struct import (
    ToolRequest,
    ToolResponse,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiTurnAgentLoopOutput,
    MultiTurnAgentLoopWorker,
)
from contextvars import ContextVar


class WideSeekR1AgentLoopWorker(MultiTurnAgentLoopWorker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.extra_keys_turn = ["subtask_count", "search_count", "access_count", "tool_call_info", "prompt_text", "response_text", "role"]
        self.extra_keys_traj = ["origin_question", "planner_summary", "final_answer", "final_answer_text", "num_valid_planner_turns", "num_valid_worker_turns", "eval_metric", "total_turn_list"]

        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        self.max_total_len = int(self.cfg.actor.model.encoder_seq_length)

        self.use_access_summary = self.cfg.tools.get("use_access_summary", False)

        self.use_fixed_rollout = cfg.rollout.get("use_fixed_worker", False)
        self.fixed_role = self.cfg.agentloop.get("fixed_role", None)
        if self.use_fixed_rollout:
            assert self.fixed_role

        self.workflow = self.cfg.agentloop.get("workflow", "mas")
        self.is_hybrid = self.cfg.data.get("is_hybrid", False)
        self.reward_type = self.cfg.reward.get("reward_type", None)
        if self.reward_type:
            self.reward_type = self.reward_type.upper()
        self.reward_eval = self.cfg.reward.get("eval_metric", [])
        self.reward_eval = [m.upper() for m in self.reward_eval]
        self.is_widesearch = self.cfg.data.get("is_widesearch", False)
        
        self.use_llm_judge_api = True
        llm_ip = self.cfg.agentloop.get("llm_ip", "")
        llm_port = self.cfg.agentloop.get("llm_port", "")
        llm_type = self.cfg.agentloop.get("llm_type", "")
        self.sgl_client = SGLangClient(llm_ip, llm_port, llm_type)

        self.train_roles = self.cfg.agentloop.get("train_roles", None)
        

    async def extract_tool_calls(
        self, response_text: str, role: str
    ) -> tuple[str, list[ToolRequest]]:
        """Extract tool calls from response based on role using Qwen's JSON format.

        Args:
            response_text: The response text to extract from
            role: Agent role ('planner' or 'worker')

        Returns:
            list_of_tool_requests
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
        sub_traj_id: int
    ) -> tuple[list[AgentLoopOutput], str]:
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

    async def run_one_query_role(
        self,
        question: str,
        role: str,
        sub_traj_id: int,
        main_task: str = None,
        is_markdown: bool = False,
        language: str = "en",
    ) -> tuple[list[AgentLoopOutput], str]:

        origin_question = question
        output_buffer = []
        total_turn_list = []

        tools_description = (
            tools_description_zh if language == "zh" else tools_description_en
        )
        # Build message history with appropriate prompts
        if role == "planner":
            message_history = get_prompt_planner(
                origin_question, is_markdown=is_markdown, language=language
            )
            # Planner uses subtask tool
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

        if role == "planner":
            max_turns = self.cfg.agentloop.get("max_planner_turns", 10)
        elif role == "single":
            max_turns = self.cfg.agentloop.get("max_sa_turns", 50)
        elif role == "worker":
            max_turns = self.cfg.agentloop.get("max_worker_turns", 20)
        else:
            raise ValueError(f"illegal role {role}")

        turn_hint = (
            f"\n\nThis is your first turn to answer the question. You must finish your answer within {max_turns} turns"
            if language == "en"
            else f"\n\n这是你回答该问题的第一轮。你必须在 {max_turns} 轮之内完成你的回答"
        )
        assert message_history[-1]["role"] == "user"
        message_history[-1]["content"] = message_history[-1]["content"] + turn_hint

        prompt_ids = self.tokenizer.apply_chat_template(
            message_history, tokenize=True, add_generation_prompt=True, tools=tools
        )
        prompt_ids = prompt_ids[: self.max_total_len]

        # Initialize tracking variables
        context_failed = False
        max_turn_limit_failed = False
        tool_response_failed = False

        succ_end = False
        sub_traj_num = 0

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

            # Store output
            assert generate_result["logprobs"] is not None
            output_buffer.append(
                AgentLoopOutput(
                    prompt_ids=copy.deepcopy(prompt_ids),
                    response_ids=copy.deepcopy(response_ids),
                    prompt_text=copy.deepcopy(self.tokenizer.decode(prompt_ids)),
                    response_text=response_text,
                    is_end=generate_result["finish_reason"] == "length",
                    response_logprobs=generate_result["logprobs"],
                    extra_fields=dict(role=role, idx_to_sub_traj=sub_traj_id),
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
                for i, tool_request in enumerate(tool_requests, start=1):
                    tasks.append(
                        self.worker_call(
                            tool_request, origin_question, is_markdown, language, sub_traj_id + i + sub_traj_num
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
                            f"# Subtask {idx + 1}:\n{subtask_text}\n# Result:\n{worker_summary}"
                            if language == "en"
                            else f"# 子任务 {idx + 1}:\n{subtask_text}\n# 结果:\n{worker_summary}"
                        )
                    else:
                        tool_messages_text.append(
                            f"# Subtask {idx + 1}:\n{subtask_text}\n# Result:\nThe current subagent exceeded its context window limit while executing this subtask, which caused the failure. Please retry."
                            if language == "en"
                            else f"# 子任务 {idx + 1}:\n{subtask_text}\n# 结果:\n当前子智能体在执行该子任务时超出其上下文窗口限制，导致失败。请重试。"
                        )

                turn_hint = (
                    f"\n\nYour next answer will be on turn {turn_idx + 2}. You MUST finish the entire answer by turn {max_turns}."
                    if language == "en"
                    else f"\n\n你的下一次回答将是第 {turn_idx + 2} 轮。你必须在第 {max_turns} 轮之内完成整个回答。"
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": "\n\n".join(tool_messages_text) + turn_hint,
                    }
                )

            else:
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
                            f"# Search query:\n{query}\n# Result:\n{tool_response.text}"
                            if language == "en"
                            else f"# 搜索查询:\n{query}\n# 结果:\n{tool_response.text}"
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
                                f"# Access URL:\n{url}\n# Result:\n{page_content}"
                                if language == "en"
                                else f"# 访问URL:\n{url}\n# 结果:\n{page_content}"
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
                        tool_messages_text[idx] = (
                            f"# Access URL:\n{url}\n# Info to extract:\n{info_to_extract}\n# Result:\n{summary}"
                            if language == "en"
                            else f"# 访问URL:\n{url}\n# 需要提取的信息:\n{info_to_extract}\n# 结果:\n{summary}"
                        )

                turn_hint = (
                    f"\n\nYour next answer will be on turn {turn_idx + 2}. You MUST finish the entire answer by turn {max_turns}."
                    if language == "en"
                    else f"\n\n你的下一次回答将是第 {turn_idx + 2} 轮。你必须在第 {max_turns} 轮之内完成整个回答。"
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

        if not succ_end and not context_failed and not tool_response_failed:
            if turn_idx + 1 >= max_turns:
                max_turn_limit_failed = True

        if max_turn_limit_failed:
            for turn in output_buffer:
                if turn.extra_fields["role"] == role:
                    turn.max_turn_limit_failed = True

        if context_failed or tool_response_failed:
            for turn in output_buffer:
                if turn.extra_fields["role"] == role:
                    turn.context_failed = True

        if (
            context_failed
            and len(output_buffer) >= 1
            and len(output_buffer[-1].response_ids) >= 8000
        ):
            output_buffer[-1].turn_repeat_failed = True

        task_failed = max_turn_limit_failed or context_failed or tool_response_failed
        assert task_failed != succ_end

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

        final_answer_text = None
        if is_markdown:
            final_answer_extract = extract_final_answer(answer_text, mode="markdown")
        else:
            final_answer_extract = extract_final_answer(answer_text, mode="boxed")

        # credit assignment
        reward_score = 0.0
        orm_em_score, eval_metric, format = await self.get_final_reward_score(
            origin_question, final_answer_extract, answer, is_markdown=is_markdown
        )

        if self.is_hybrid:
            eval_metric = {}  # FIXME:
        eval_metric.update(
            {
                "format_score": {
                    "call_search_reward": 0,
                    "final_answer_format": 0,
                    "length_penalty": 0,
                    "context_failed": 0,
                    "turn_repeat_failed": 0,
                    "max_turn_limit_failed": 0,
                    "abs_call_search_reward": 0,
                    "abs_final_answer_format": 0,
                    "abs_length_penalty": 0,
                    "abs_em_f1_score": 0,
                }
            }
        )

        # credit assignment
        search_credit = 0.0
        length_penalty = 0.0

        self.format_reward = self.cfg.agentloop.get("format_reward", 0.0)
        self.call_search_reward = self.cfg.agentloop.get("call_search_reward", 0.0)
        self.length_limit = self.cfg.agentloop.get("length_limit", 5000)
        self.max_length_limit = self.cfg.agentloop.get("max_length_limit", 7000)
        self.length_penalty = self.cfg.agentloop.get("length_penalty", 0.0)

        for turn_output in output_buffer:
            tool_call_info = turn_output.tool_call_info
            if tool_call_info is None:
                continue
            if tool_call_info.get("access", 0) > 0:
                search_credit = self.call_search_reward
                eval_metric["format_score"]["call_search_reward"] = 1
                break

        max_response_len = max(
            len(turn_output.response_ids) for turn_output in output_buffer
        )
        if max_response_len > self.length_limit:
            t = (max_response_len - self.length_limit) / (
                self.max_length_limit - self.length_limit
            )
            t = max(0.0, min(1.0, t))
            length_penalty = t * (self.length_penalty)
            # reward_score -= penalty
            eval_metric["format_score"]["length_penalty"] = t

        one_turn_failed = False

        for turn in output_buffer:
            if turn.max_turn_limit_failed == True:
                eval_metric["format_score"]["max_turn_limit_failed"] = 1
            if turn.turn_repeat_failed == True:
                eval_metric["format_score"]["turn_repeat_failed"] = 1
            if turn.context_failed == True:
                eval_metric["format_score"]["context_failed"] = 1

            if turn.turn_repeat_failed:
                one_turn_failed = True

        train_buffer = []
        if final_answer_extract is not None and format == True:
            flag = False
            for turn in output_buffer:
                if (
                    turn.context_failed or turn.max_turn_limit_failed
                ) and turn.extra_fields["role"] != "worker":
                    # main agent or sa failed but extract good format
                    flag = True

            if not flag:
                for turn in output_buffer:
                    if not (turn.context_failed or turn.max_turn_limit_failed):
                        train_buffer.append(turn)

                reward_score = (
                    orm_em_score + self.format_reward + search_credit - length_penalty
                )

                eval_metric["format_score"]["abs_final_answer_format"] = (
                    self.format_reward
                )
                eval_metric["format_score"]["abs_em_f1_score"] = orm_em_score
                eval_metric["format_score"]["abs_call_search_reward"] = search_credit
                eval_metric["format_score"]["abs_length_penalty"] = -length_penalty
                eval_metric["format_score"]["final_answer_format"] = 1
            else:
                for turn in output_buffer:
                    if (
                        turn.context_failed or turn.max_turn_limit_failed
                    ) and turn.extra_fields["role"] != "worker":
                        train_buffer.append(turn)

                reward_score = 0.0

        else:
            reward_score = 0.0
            if succ_end:
                train_buffer.append(output_buffer[-1])

            if one_turn_failed:
                for turn in output_buffer:
                    if turn.turn_repeat_failed:
                        if turn not in train_buffer:
                            train_buffer.append(turn)
            else:
                for turn in output_buffer:
                    if turn.max_turn_limit_failed or turn.context_failed:
                        assert not turn.turn_repeat_failed
                        if turn not in train_buffer:
                            train_buffer.append(turn)

        sorted_train_buffer = []
        for single_turn_output in output_buffer:
            if single_turn_output not in sorted_train_buffer and single_turn_output in train_buffer:
                sorted_train_buffer.append(single_turn_output) # TODO: 为啥这样？

        for single_turn_output in output_buffer:
            single_turn_output.reward_score = reward_score
        for single_turn_output in train_buffer:
            single_turn_output.reward_score = reward_score

        for single_turn_output in output_buffer:
            single_turn_output.extra_fields["not_training"] = True
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
            single_turn_output.extra_fields["tool_call_info"] = single_turn_output.tool_call_info
            single_turn_output.extra_fields["prompt_text"] = single_turn_output.prompt_text
            single_turn_output.extra_fields["response_text"] = single_turn_output.response_text

        output = MultiTurnAgentLoopOutput(
            single_turn_outputs=output_buffer,
            train_buffer=sorted_train_buffer, # FIXME:
            trace_prints=[],  # Can add message_history tracking if needed
            extra_fields=dict(
                final_answer=final_answer_extract,
                final_answer_text=final_answer_text,
                planner_summary=answer_text,
                reward=reward_score,
                origin_question=origin_question,
                eval_metric=eval_metric,
                total_turn_list=total_turn_list if self.workflow == "mas" else None,
                instance_id=answer["instance_id"],
                num_valid_planner_turns=num_valid_planner_turns,
                num_valid_worker_turns=num_valid_worker_turns,
            ),
        )
        return output

    def gen_extra_fields(
        self,
        task_results: list[MultiTurnAgentLoopOutput],
        answer: str, 
    ) -> Optional[dict]:
        extra_fields_turn, extra_fields_traj, *_ = super().gen_extra_fields(task_results, answer)

        roles = []
        for task_result in task_results:
            for single_turn_output in task_result.single_turn_outputs:
                if self.extra_keys_turn is not None:
                    for k in self.extra_keys_turn:
                        v = single_turn_output.extra_fields.get(k, None)
                        # extra_fields_turn[k].append(v) # FIXME: 这里要注释，不然重复了？
                        if k == "role" and single_turn_output.extra_fields["not_training"] != True:
                            roles.append(v)
        extra_fields_turn = {**extra_fields_turn, "roles": roles}

        role_group_size = 0
        for task_result in task_results:
            have_role = False
            for single_turn_output in task_result.single_turn_outputs:
                single_turn_output: AgentLoopOutput
                role = single_turn_output.extra_fields["role"]
                if self.train_roles and role in self.train_roles:
                    have_role = True
                    break
            if have_role:
                role_group_size += 1
        extra_fields_group = dict(
            answer=answer,
            role_group_size=role_group_size,
            num_valid_planner_turns=sum(extra_fields_traj["num_valid_planner_turns"]),
            num_valid_worker_turns=sum(extra_fields_traj["num_valid_worker_turns"]),
        )

        idx_to_sub_traj = []
        # 0, 1, 1, 2, 2, 0
        # sub_traj_map = 0, 1, 2, 
        # idx_to_sub_traj: 0, 1, 1, 2, 2, 0
        for task_result in task_results:
            sub_traj_map = {}
            for single_turn_output in task_result.single_turn_outputs:
                if single_turn_output.extra_fields["not_training"] == True:
                    continue
                # 有可能不连续了？因为continue跳过了一些,好像不影响
                role_idx = single_turn_output.extra_fields["idx_to_sub_traj"]
                if role_idx not in sub_traj_map:
                    sub_traj_map[role_idx] = len(sub_traj_map)
                idx_to_sub_traj.append(sub_traj_map[role_idx])
        extra_fields_train = {"idx_to_sub_traj": idx_to_sub_traj}

        return extra_fields_turn, extra_fields_traj, extra_fields_group, extra_fields_train

    def get_rollout_metrics(
        self,
        rollout_result: DynamicRolloutResult,
    ) -> dict:
        try:
            # TODO: 这一段意义是？
            for k1, v2 in {
                "turn_subtask_counts":      rollout_result.extra_fields_turn["subtask_count"],
                "turn_search_counts":       rollout_result.extra_fields_turn["search_count"],
                "turn_access_counts":       rollout_result.extra_fields_turn["access_count"],
                "num_valid_planner_turns":  rollout_result.extra_fields_group["num_valid_planner_turns"],
                "num_valid_worker_turns":   rollout_result.extra_fields_group["num_valid_worker_turns"],
                "eval_metrics":             rollout_result.extra_fields_traj["eval_metric"],
                "total_turn_list_metric":   rollout_result.extra_fields_traj["total_turn_list"],
            }.items():
                v1 = getattr(rollout_result, k1, None)
                if v1 is not None:
                    assert v1 == v2
        except Exception as e:
            breakpoint()

        if self.is_eval:
            return {}

        rollout_batch_1 = dict(
            turn_subtask_counts=rollout_result.extra_fields_turn["subtask_count"],
            turn_search_counts=rollout_result.extra_fields_turn["search_count"],
            turn_access_counts=rollout_result.extra_fields_turn["access_count"],
            num_valid_planner_turns=sum(rollout_result.extra_fields_traj["num_valid_planner_turns"]),
            num_valid_worker_turns=sum(rollout_result.extra_fields_traj["num_valid_worker_turns"]),
        )

        rollout_batch_2 = dict(
            eval_metrics=rollout_result.extra_fields_traj["eval_metric"],
        )

        rollout_batch_3 = dict(
            total_turn_list_metric=rollout_result.extra_fields_traj["total_turn_list"],
        )

        # idx_to_traj = rollout_batch["idx_to_traj"]
        idx_to_traj = rollout_result.idx_to_traj
        num_trajectories = max(idx_to_traj) + 1
        tool_call_metrics = _compute_tool_call_metrics(
            rollout_batch_1, idx_to_traj, int(num_trajectories)
        )
        # FIXME: 如何写device和dp group？不需要allreduce？
        
        # eval_metrics_dict = _compute_eval_metrics(
        #     rollout_batch_2, device, data_parallel_group
        # )
        # mas_turn_metrics = _compute_mas_turn_metrics(
        #     rollout_batch_3, device, data_parallel_group
        # )
        agent_metrics = {
            **tool_call_metrics,
            # **eval_metrics_dict,
            # **mas_turn_metrics,
        }
        return agent_metrics

    async def get_final_reward_score(
        self, origin_question, extract_answer, label_answer, is_markdown=False
    ):
        format = True
        metrics = list(set(self.reward_eval) | set([self.reward_type]))
        if metrics == []:
            return 0.0, {}, format
        if is_markdown:
            reward_score, eval_metric, format = await self.evaluate_markdown(
                extract_answer, label_answer
            )
            return reward_score, eval_metric, format

        label_answer = label_answer["answer"]
        eval_metric = dict.fromkeys(metrics, 0)
        if label_answer is not None and extract_answer is not None:
            for metric in metrics:
                if metric == "EM":
                    em_score = compute_score_em(extract_answer, label_answer)
                    eval_metric[metric] = em_score

                if metric == "F1":
                    f1_score = compute_score_f1(extract_answer, label_answer)
                    eval_metric[metric] = f1_score

                if metric == "LLM":
                    # Use LLM as judge
                    llm_score = await self.verify_answer_with_llm_judge(
                        question=origin_question,
                        predicted_answer=extract_answer,
                        correct_answer=label_answer,
                        use_llm_judge_api=self.use_llm_judge_api,
                    )
                    eval_metric[metric] = llm_score

            reward_score = eval_metric[self.reward_type]

        else:
            reward_score = 0.0

        return reward_score, eval_metric, format

    async def verify_answer_with_llm_judge(
        self,
        question: str,
        predicted_answer: str,
        correct_answer: list,
        use_llm_judge_api=False,
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
        from rlinf.agents.wideseek_r1.utils.prompt import LLM_JUDGE_PROMPT

        if len(correct_answer) == 1:
            # Format the judge prompt
            judge_prompt_text = LLM_JUDGE_PROMPT.format(
                question=question,
                correct_answer=correct_answer[0],
                response=predicted_answer,
            )
        else:
            judge_prompt_text = LLM_JUDGE_PROMPT.format(
                question=question,
                correct_answer=correct_answer,
                response=predicted_answer,
            )

        # Create messages for the judge
        judge_messages = [
            {
                "role": "system",
                "content": "You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.",
            },
            {"role": "user", "content": judge_prompt_text},
        ]

        # Apply chat template
        if use_llm_judge_api:
            judge_response_text = await self.sgl_client.call_sglang_api(judge_messages)
            judge_prompt_ids = []
            judge_response_ids = []
        else:
            # assert False, "please use call llm judge api."
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
        if (
            "correct" in judge_response_clean
            and "incorrect" not in judge_response_clean
        ):
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

        def normalize_metric_dict(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, np.generic):
                    out[k] = v.item()
                else:
                    out[k] = v
            return out

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
        error_eval_return = dict.fromkeys(metrics, eval_metrics)

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
                answer_df = extract_final_answer(
                    answer_markdown, mode="markdown", strict=False
                )
                if answer_df is None:
                    msg = "Failed to parse label answer markdown"
                    self.log_warning(msg)
                    return 0.0, error_eval_return, False
            elif isinstance(answer_markdown, pd.DataFrame):
                answer_df = answer_markdown
            else:
                msg = f"Invalid label answer type: {type(answer_markdown)}"
                self.log_warning(msg)
                return 0.0, error_eval_return, False

            # Validate extract_answer
            if not isinstance(extract_answer, pd.DataFrame) or extract_answer.empty:
                msg = f"Extracted answer is None or not a DataFrame, it's {extract_answer}"
                # self.log_warning(msg)
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
                required_columns = [
                    norm_column(col) for col in required_columns
                ]  # widesearch requir columns: " " -> ""

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
                # self.log_warning(msg)
                return 0.0, error_eval_return, False

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
                        self.use_llm_judge_api,
                    )
                    response_df[col + "_before_map"] = response_df[col]
                    response_df[col] = response_df[col].apply(
                        lambda x: primary_key_map.get(x, x)
                    )

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
                                compute_score_em(
                                    row[col + "_response"], row[col + "_query"]
                                )
                                for _, row in df_inner.iterrows()
                            ]
                            df_inner_scores["EM"][f"{col}_score"] = scores

                        elif metric_type == "F1":
                            scores = [
                                compute_score_f1(
                                    row[col + "_response"], row[col + "_query"]
                                )
                                for _, row in df_inner.iterrows()
                            ]
                            df_inner_scores["F1"][f"{col}_score"] = scores

                        elif metric_type == "LLM":
                            # Collect for batch LLM judge
                            responses = df_inner[col + "_response"].tolist()
                            targets = df_inner[col + "_query"].tolist()
                            llm_tasks.append(
                                self.llm_judge_column(
                                    responses, targets, self.use_llm_judge_api
                                )
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
                precision_by_row = (
                    tp_by_row / num_pred_rows if num_pred_rows > 0 else 0.0
                )
                recall_by_row = tp_by_row / num_gt_rows if num_gt_rows > 0 else 0.0
                f1_by_row = calc_f1(precision_by_row, recall_by_row)

                # Item-level metrics
                tp_by_item = df_score.sum().sum()
                precision_by_item = (
                    tp_by_item / num_pred_items if num_pred_items > 0 else 0.0
                )
                recall_by_item = tp_by_item / num_gt_items if num_gt_items > 0 else 0.0
                f1_by_item = calc_f1(precision_by_item, recall_by_item)

                # Search-specific item-level metrics (non-primary-key columns only)
                # This reflects pure search ability without trivial primary key matches
                non_pk_columns = [
                    col for col in required_columns if col not in unique_columns
                ]
                if non_pk_columns:
                    # Select only non-primary-key column scores
                    search_score_cols = [f"{col}_score" for col in non_pk_columns]
                    df_search_score = df_score[search_score_cols]

                    # Compute search metrics
                    tp_search = df_search_score.sum().sum()
                    num_pred_search = num_pred_rows * len(non_pk_columns)
                    num_gt_search = num_gt_rows * len(non_pk_columns)

                    search_precision_by_item = (
                        tp_search / num_pred_search if num_pred_search > 0 else 0.0
                    )
                    search_recall_by_item = (
                        tp_search / num_gt_search if num_gt_search > 0 else 0.0
                    )
                    search_f1_by_item = calc_f1(
                        search_precision_by_item, search_recall_by_item
                    )
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

                eval_metric[metric_type] = normalize_metric_dict(
                    {
                        "score": score,
                        "precision_by_row": precision_by_row,
                        "recall_by_row": recall_by_row,
                        "f1_by_row": f1_by_row,
                        "precision_by_item": precision_by_item,
                        "recall_by_item": recall_by_item,
                        "f1_by_item": f1_by_item,
                        "search_precision_by_item": search_precision_by_item,
                        "search_recall_by_item": search_recall_by_item,
                        "search_f1_by_item": search_f1_by_item,
                    }
                )

            msg = f"Evaluated with {len(metrics)} metrics: {metrics}"

        except Exception:
            msg = f"Evaluation error: {traceback.format_exc()}"
            self.log_warning(msg)
            return 0.0, error_eval_return, False

        # Select reward score based on self.reward_type
        reward_score = eval_metric[self.reward_type]["f1_by_item"]
        # self.log_info(eval_metric)
        return reward_score, eval_metric, True

    async def llm_judge_column(
        self, responses: list, targets: list, use_llm_judge_api=False
    ) -> list:
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

        user_prompt = user_prompt.format(response=response_dict)
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if use_llm_judge_api:
            result_text = await self.sgl_client.call_sglang_api(messages)
        else:
            # assert False, "please use call llm judge api."
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
        except Exception:
            # If any error, default to 0
            score_list = [0.0] * len(responses)

        # Ensure correct length
        if len(score_list) != len(responses):
            score_list = [0.0] * len(responses)
        return score_list

    async def primary_key_preprocess(
        self, response_list, reference_list, use_llm_judge_api=False
    ):
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
            {"role": "user", "content": user_prompt},
        ]

        if use_llm_judge_api:
            result_text = await self.sgl_client.call_sglang_api(messages)
        else:
            # Apply chat template
            # assert False, "please use call llm judge api."
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
