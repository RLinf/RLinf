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

from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import (
    ToolRequest,
    ToolResponse,
)
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

import pandas as pd
import traceback
import re
from rlinf.agents.mas.sglang_client import SGLangClient

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

        if final_answer_extract is not None:
            if isinstance(final_answer_extract, pd.DataFrame):
                final_answer_extract = final_answer_extract.to_dict(orient='records')

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
