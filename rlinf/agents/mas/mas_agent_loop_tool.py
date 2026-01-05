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
from typing import Any
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
    MultiTurnAgentLoopWorker,
    AgentLoopOutput,
    MultiTurnAgentLoopOutput,
)
from rlinf.agents.prompt.tools import get_prompt_planner, get_prompt_worker, get_prompt_single_agent, tools_description
from rlinf.agents.utils.summary_utils import (
    get_hint_extraction_prompt,
    get_final_answer_extraction_prompt,
    generate_summarize_prompt,
)
from rlinf.agents.utils.reward import compute_score_em, compute_score_f1, extract_final_answer


class MASToolAgentLoopWorker_Tool_Parallel(MultiTurnAgentLoopWorker):
    """Multi-Agent System tool agent loop with planner-worker architecture.

    Architecture:
    - Planner: Calls 'subtask' tool to spawn workers
    - Worker: Calls 'search' and 'access' tools to gather information
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        self.max_total_len = int(self.cfg.actor.model.encoder_seq_length)
        # self.max_resp_len = max(1, self.max_total_len - self.max_prompt_len)

        # Configuration
        self.max_planner_turns = self.cfg.agentloop.get("max_planner_turns", 5)
        self.max_worker_turns = self.cfg.agentloop.get("max_worker_turns", 5)
        self.max_workers_per_planner = self.cfg.agentloop.get("max_workers_per_planner", 1)
        self.max_toolcall_per_worker = self.cfg.agentloop.get("max_toolcall_per_worker", 1)
        self.hint_generation = self.cfg.agentloop.get("hint_generation", False)
        self.final_answer_extraction = self.cfg.agentloop.get("final_answer_extraction", False)
        
        self.format_reward = self.cfg.agentloop.get("format_reward", 0.0)
        self.call_search_reward = self.cfg.agentloop.get("call_search_reward", 0.0)
        self.main_summary = self.cfg.agentloop.get("main_summary", False)
        self.length_limit = self.cfg.agentloop.get("length_limit", 5000)
        self.max_length_limit = self.cfg.agentloop.get("max_length_limit", 7000)
        self.length_penalty = self.cfg.agentloop.get("length_penalty", 0.0)
        self.max_length_penalty = self.cfg.agentloop.get("max_length_penalty", 0.0)        

        # Role filtering for training
        self.train_roles = self.cfg.agentloop.get("train_roles", None)
        if self.train_roles is not None and not isinstance(self.train_roles, list):
            self.train_roles = list(self.train_roles) if self.train_roles else None

        # Tool extraction regexes for Qwen's JSON format
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

        self.use_sub_rollout = cfg.rollout.get('use_sub_worker', False)

        # Markdown table evaluation mode
        self.is_markdown = self.cfg.data.get("is_markdown", False)
        self.use_tool = self.cfg.agentloop.get("use_tool", True)

        self.workflow = self.cfg.agentloop.get("workflow", "mas") 

    async def extract_tool_calls(self, response_text: str, role: str) -> tuple[str, list[ToolRequest]]:
        """Extract tool calls from response based on role using Qwen's JSON format.

        Args:
            response_text: The response text to extract from
            role: Agent role ('planner' or 'worker')

        Returns:
            list_of_tool_requests
        """
        function_calls = []

        # Extract all <tool_call> tags
        tool_call_match = self.tool_call_regex.findall(response_text)

        subtask_count = 0
        search_count = 0
        access_count = 0

        if tool_call_match:
            tool_call_str = tool_call_match[0] # FIXME:可能是个list，我认为目前先取0？
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
                        for sub_agent in sub_agents[:self.max_workers_per_planner]:
                            # Skip if not a dict or missing prompt
                            if not isinstance(sub_agent, dict):
                                continue
                            prompt = sub_agent.get("prompt", "")
                            if not prompt:
                                continue
                            function_calls.append(
                                ToolRequest(name='subtask', arguments={"subtask": prompt})
                            )
                            subtask_count += 1
                elif role == "worker":
                    # Worker: handle search and access tools
                    if tool_name == "search":
                        # Extract searches array
                        searches = tool_arguments.get("queries", [])
                        for search_item in searches[:self.max_toolcall_per_worker]:
                            # Skip if not a dict or missing query
                            if not isinstance(search_item, dict):
                                continue
                            query = search_item.get("query", "")
                            if not query:
                                continue
                            topk = search_item.get("count", None)
                            if topk:
                                function_calls.append(
                                    ToolRequest(name='search', arguments={"query": query, "topk": topk})
                                )
                            else:
                                function_calls.append(
                                    ToolRequest(name='search', arguments={"query": query})
                                )
                            search_count += 1

                    elif tool_name == "access":
                        # Extract accesses array
                        accesses = tool_arguments.get("urls", [])
                        for access_item in accesses[:self.max_toolcall_per_worker]:
                            # Skip if not a dict or missing url
                            if not isinstance(access_item, dict):
                                continue
                            url = access_item.get("url", "")
                            if not url:
                                continue
                            function_calls.append(
                                ToolRequest(name='access', arguments={"url": url})
                            )
                            access_count += 1
                elif role == "single":
                    if tool_name == "search":
                        query = tool_arguments.get("query", "")
                        if query:
                            topk = tool_arguments.get("count", None)
                            if topk:
                                function_calls.append(
                                    ToolRequest(name='search', arguments={"query": query, "topk": topk})
                                )
                            else:
                                function_calls.append(
                                    ToolRequest(name='search', arguments={"query": query})
                                )
                            search_count = 1

                    elif tool_name == "access":
                        url = tool_arguments.get("url", "")
                        if url:
                            function_calls.append(
                                ToolRequest(
                                    name="access",
                                    arguments={"url": url},
                                )
                            )
                            access_count = 1
            except Exception as e:
                # self.log_error(f"debug info in extract {e}, the tool call match is {tool_call_match}")
                pass

        tool_call_info={"subtask": subtask_count, "search": search_count, "access": access_count, "role": role}
        if function_calls == []:
            tool_call_info = None  
        return function_calls, tool_call_info

    async def generate_hints(self, question: str, output_buffer: list) -> str:
        hint_messages = get_hint_extraction_prompt(question)
        hint_prompt_ids = self.tokenizer.apply_chat_template(
            hint_messages, tokenize=True, add_generation_prompt=True
        )

        # Generate hints
        max_hint_len = self.max_total_len - len(hint_prompt_ids)
        assert max_hint_len > 0, f"Prompt too long for hint generation, {self.max_total_len}, {len(hint_prompt_ids)}"
        generate_result = await self.generate(
            hint_prompt_ids, sampling_params={"max_new_tokens": max_hint_len}
        )

        hint_ids = generate_result["output_ids"]
        if len(hint_ids) > max_hint_len:
            hint_ids = hint_ids[:max_hint_len]

        # Split the eos token for pure hint text
        hint_text = self.tokenizer.decode(hint_ids).split('<|im_end|>')[0]
        
        output_buffer.append(AgentLoopOutput(
            prompt_ids=hint_prompt_ids,
            response_ids=hint_ids,
            is_end=generate_result["finish_reason"] == "length",
            response_logprobs=generate_result["logprobs"],
            extra_fields=dict(
                role='hint',
            ),
            prompt_text=self.tokenizer.decode(hint_prompt_ids), # For debug
            response_text=self.tokenizer.decode(hint_ids)
        ))        
        return hint_text

    async def get_final_answer_llm(self, task_description: str, summary_text: str, output_buffer) -> str:
        """Extract and format final answer from summary using local LLM.

        Args:
            task_description: Original task question
            summary_text: Agent's summary of work

        Returns:
            Answer text
        """
        extraction_prompt = get_final_answer_extraction_prompt(task_description, summary_text, self.is_markdown)
        extraction_prompt_ids = self.tokenizer.apply_chat_template(
            extraction_prompt, tokenize=True, add_generation_prompt=True
        )

        # Extract answer
        max_answer_len = self.max_total_len - len(extraction_prompt_ids)
        assert max_answer_len > 0, f"Prompt too long for answer extraction, {self.max_total_len}, {len(extraction_prompt_ids)}"
        generate_result = await self.generate(
            extraction_prompt_ids, sampling_params={"max_new_tokens": max_answer_len}
        )

        answer_ids = generate_result["output_ids"]
        if len(answer_ids) > max_answer_len:
            answer_ids = answer_ids[:max_answer_len]

        answer_text = self.tokenizer.decode(answer_ids)

        output_buffer.append(AgentLoopOutput(
            prompt_ids=extraction_prompt_ids,
            response_ids=answer_ids,
            is_end=generate_result["finish_reason"] == "length",
            response_logprobs=generate_result["logprobs"],
            extra_fields=dict(
                role='final_answer',
            ),
            prompt_text=self.tokenizer.decode(extraction_prompt_ids), # For debug
            response_text=answer_text,
        ))
        return answer_text

    async def generate_summary(
        self,
        prompt_ids: list[int],
        task_description: str,
        main_task: str,
        output_buffer: list,
        task_failed: bool = False,
        role: str = 'planner'
    ) -> tuple[list[int], str]:
        """Generate final summary for the agent's work.

        Args:
            prompt_ids: Current conversation token IDs
            task_description: Original task description
            task_failed: Whether the task failed (hit limits)

        Returns:
            Tuple of (updated_prompt_ids, summary_text)
        """
        summary_prompt = generate_summarize_prompt(task_description, main_task, task_failed, self.final_answer_extraction, role, self.is_markdown)
        summary_messages = [{"role": "user", "content": summary_prompt}]
        summary_prompt_ids = self.get_tool_response_ids(summary_messages)

        max_summary_len = self.max_total_len - len(prompt_ids) - len(summary_prompt_ids) - 10

        if max_summary_len <= 0:
            # we need to cut promptids
            reserve_output_token = 1024
            prompt_ids = prompt_ids[:len(prompt_ids) - (len(summary_prompt_ids) + 10 + reserve_output_token)]
            prompt_ids += summary_prompt_ids
        
        else:
            prompt_ids += summary_prompt_ids

        max_summary_len = self.max_total_len - len(prompt_ids) - 10

        if role == 'worker' and self.use_sub_rollout:
            generate_result = await self.generate(
                prompt_ids, sampling_params={"max_new_tokens": max_summary_len}, rollout_name="subworker",
            )
            generate_result["logprobs"] = [0.0] * len(generate_result["output_ids"])
        else:
            generate_result = await self.generate(
                prompt_ids, sampling_params={"max_new_tokens": max_summary_len},
            )   

        response_ids = generate_result["output_ids"]
        if len(response_ids) > max_summary_len:
            response_ids = response_ids[:max_summary_len]

        summary_text = self.tokenizer.decode(response_ids).split('<|im_end|>')[0]

        output_buffer.append(AgentLoopOutput(
            prompt_ids=copy.deepcopy(prompt_ids),
            response_ids=copy.deepcopy(response_ids),
            is_end=generate_result["finish_reason"] == "length",
            response_logprobs=generate_result["logprobs"],
            extra_fields=dict(
                role='summary',
            ),
            prompt_text=self.tokenizer.decode(prompt_ids), # For debug
            response_text=self.tokenizer.decode(response_ids)
        ))                

        return summary_text

    async def worker_call(self, worker_request: ToolRequest, main_task: str) -> tuple[list[AgentLoopOutput], str]:
        assert worker_request.name == 'subtask', f"Expected 'subtask' tool, got {worker_request.name}"
        assert 'subtask' in worker_request.arguments, f"Missing 'subtask' in arguments: {worker_request.arguments}"

        subtask = worker_request.arguments["subtask"]

        worker_outputs_buffer, summary_text = await self.run_one_query_role(
            question=subtask,
            role='worker',
            main_task=main_task,
        )
        return worker_outputs_buffer, summary_text

    async def run_one_query_role(
        self,
        question: str,
        role: str,
        main_task: str = None,
    ) -> tuple[list[AgentLoopOutput], str]:
        origin_question = question
        output_buffer = []

        hint_text = None
        if role == 'planner' and self.hint_generation:
            hint_text = await self.generate_hints(origin_question, output_buffer)

        # Build message history with appropriate prompts
        if role == 'planner':
            message_history = get_prompt_planner(origin_question, hint_text, final_answer_extraction = self.final_answer_extraction, is_markdown=self.is_markdown, use_tool = self.use_tool)
            # Planner uses subtask tool
            tools = [tools_description["create_sub_agents"]]
        elif role == 'worker':
            assert main_task is not None, "Worker must have main_task provided"
            message_history = get_prompt_worker(main_task, origin_question, is_parallel=True)
            # Worker uses search and access tools
            tools = [tools_description["search"], tools_description["access"]]
        elif role == 'single':
            message_history = get_prompt_single_agent(origin_question)
            tools = [tools_description["search_single_agent"], tools_description["access_single_agent"]]                                                 
        else:
            raise ValueError(f"Invalid role: {role}")

        # Apply chat template with tools
        if self.use_tool:
            prompt_ids = self.tokenizer.apply_chat_template(
                message_history, tokenize=True, add_generation_prompt=True, tools=tools
            )
        else:
            prompt_ids = self.tokenizer.apply_chat_template(
                message_history, tokenize=True, add_generation_prompt=True
            )    
        prompt_ids = prompt_ids[:self.max_total_len] 

        # Initialize tracking variables
        task_failed = False
        max_turns = self.max_planner_turns if role == 'planner' else self.max_worker_turns

        for turn_idx in range(max_turns):
            max_resp_len = self.max_total_len - len(prompt_ids)
            if max_resp_len <= 0:
                task_failed = True
                break
                
            if role == 'worker' and self.use_sub_rollout:
                generate_result = await self.generate(
                    prompt_ids, sampling_params={"max_new_tokens": max_resp_len}, rollout_name="subworker",
                )
                generate_result["logprobs"] = [0.0] * len(generate_result["output_ids"])
            else:
                generate_result = await self.generate(
                    prompt_ids, sampling_params={"max_new_tokens": max_resp_len},
                )    

            response_ids = generate_result["output_ids"]
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]

            response_text = self.tokenizer.decode(response_ids)

            tool_requests, tool_call_info = await self.extract_tool_calls(response_text, role=role)

            # Store output
            assert generate_result["logprobs"] is not None
            output_buffer.append(AgentLoopOutput(
                prompt_ids=copy.deepcopy(prompt_ids),
                response_ids=copy.deepcopy(response_ids),
                prompt_text=copy.deepcopy(self.tokenizer.decode(prompt_ids)),
                response_text=response_text,
                is_end=generate_result["finish_reason"] == "length",
                response_logprobs=generate_result["logprobs"],
                extra_fields=dict(role=role),
                tool_call_info=tool_call_info if tool_call_info else None # if passed, must have tool call
            ))

            prompt_ids += response_ids            

            if len(response_ids) == max_resp_len:
                task_failed = True
                break            

            # Extract tool calls
            if tool_requests == []:
                break

            # Handle tool calls based on role
            tasks = []
            tool_messages = []
            worker_buffer = []
            if role == "planner":
                for tool_request in tool_requests:
                    tasks.append(self.worker_call(tool_request, origin_question))
                worker_results = await asyncio.gather(*tasks)

                tool_messages_text = []
                for idx, (worker_outputs_buffer, worker_summary) in enumerate(worker_results):
                    worker_buffer.extend(worker_outputs_buffer) 

                    # Format tool response with both request and result
                    subtask_text = tool_requests[idx].arguments["subtask"] 
                    tool_messages_text.append(f"Subtask {idx + 1}: {subtask_text}\nResult: {worker_summary}")
                tool_messages.append({"role": "tool", "content": "\n\n".join(tool_messages_text)}) # FIXME:放在一个里面更好吧

            else:
                for tool_request in tool_requests:
                    tasks.append(self.tool_call(tool_request))
                tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)
                
                tool_messages_text = []
                for idx, (tool_request, tool_response) in enumerate(zip(tool_requests, tool_responses)):
                    # Include the original request and the result
                    if tool_request.name == 'search':
                        query = tool_request.arguments["query"]
                        tool_messages_text.append(f"Search query: {query}\nResult: {tool_response.text}")
                    elif tool_request.name == 'access':
                        url = tool_request.arguments["url"]
                        tool_messages_text.append(f"Access URL: {url}\nResult: {tool_response.text}")
                    else:
                        raise ValueError(f"Unknown tool request name: {tool_request.name}")

                tool_messages.append({"role": "tool", "content": "\n\n".join(tool_messages_text)})

            # Tokenize tool responses
            tool_response_ids = self.get_tool_response_ids(tool_messages)
            max_tool_resp_len = self.max_total_len - len(prompt_ids)
            if len(tool_response_ids) > max_tool_resp_len:
                task_failed = True
                break                      

            prompt_ids += tool_response_ids
            output_buffer.extend(worker_buffer)

        # Check if hit max turns
        if turn_idx + 1 >= max_turns:
            task_failed = True

        # Generate summary
        if role == 'planner' and not self.main_summary:
            summary_text = response_text.split('<|im_end|>')[0]
        elif role == 'worker':
            summary_text = await self.generate_summary(
                prompt_ids=prompt_ids,
                task_description=origin_question,
                main_task=main_task,
                output_buffer=output_buffer,
                task_failed=task_failed, # true if hit token limits or max turns
                role=role
            )
            summary_text = summary_text.split("</think>")[-1].strip()
        elif role == 'single':
            summary_text = response_text.split('<|im_end|>')[0]

        return output_buffer, summary_text

    async def run_one_query(self, prompt_ids: list[int], *, answer) -> AgentLoopOutput:
        origin_question = self.tokenizer.decode(prompt_ids)

        if self.workflow == "sa":
            role = 'single'
        else:
            role = 'planner'

        output_buffer, planner_summary = await self.run_one_query_role(
            question=origin_question,
            role=role,
        )

        final_answer_text = None
        final_answer_extract = None
        if self.final_answer_extraction:
            # if enable final_answer_extraction, the sumamry contains no boxed answer
            final_answer_text = await self.get_final_answer_llm(
                task_description=origin_question,
                summary_text=planner_summary,
                output_buffer=output_buffer,
            )
        else:
            final_answer_text = planner_summary

        if self.is_markdown:
            final_answer_extract = extract_final_answer(final_answer_text, mode='markdown')
        else:
            final_answer_extract = extract_final_answer(final_answer_text, mode='boxed')            

        # Filter turns by role if train_roles is specified
        if self.train_roles is not None and len(self.train_roles) > 0:
            filtered_output_buffer = []
            for turn_output in output_buffer:
                turn_role = turn_output.extra_fields.get("role", None)
                if turn_role in self.train_roles:
                    filtered_output_buffer.append(turn_output)
            output_buffer = filtered_output_buffer


        reward_score, eval_metric, format = await self.get_final_reward_score(
            origin_question, final_answer_extract, answer, is_markdown=self.is_markdown
        )


        # credit assignment
        if final_answer_extract is not None and format == True:
            reward_score += self.format_reward

        for turn_output in output_buffer:
            tool_call_info = turn_output.tool_call_info
            if tool_call_info is None:
                continue
            if tool_call_info.get("search", 0) + tool_call_info.get("access", 0) > 0:
                reward_score += self.call_search_reward
                break

        for turn_output in output_buffer:
            response_len = len(turn_output.response_ids)
            if response_len > self.length_limit:
                t = (response_len - self.length_limit) / (self.max_length_limit - self.length_limit)
                t = max(0.0, min(1.0, t)) 
                penalty = self.length_penalty + t * (self.max_length_penalty - self.length_penalty)
                reward_score -= penalty
                break

        for single_turn_output in output_buffer:
            single_turn_output.reward_score = reward_score

        output = MultiTurnAgentLoopOutput(
            single_turn_outputs=output_buffer,
            trace_prints=[],  # Can add message_history tracking if needed
            extra_fields=dict(
                final_answer=final_answer_extract,
                final_answer_text=final_answer_text,
                planner_summary=planner_summary,
                reward=reward_score,
                origin_question=origin_question,
                eval_metric=eval_metric
            ),
        )
        return output
