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
import logging
import os
from typing import Any, Optional
from uuid import uuid4
from typing import Dict, List
from dataclasses import dataclass, field
from megatron.core import parallel_state
from rlinf.scheduler import Channel, Worker
from transformers import AutoTokenizer
from omegaconf import DictConfig
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.data.io_struct import (
    RolloutRequest,
    RolloutResult,
)

from .tool_parser import FunctionCall, ToolParser, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLINF_LOGGING_LEVEL", "WARN"))

@dataclass
class AgentLoopOutput:
    """Agent loop output."""
    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""

class ToolAgentLoop(Worker):
    """Simple tool agent loop that can interact with tools.

    重构为普通类：不再继承 Worker，也不使用 WorkerGroup/Channel。
    通过注入的 rollout 直接调用 agenerate，输入/输出均为 token ids。
    """

    def __init__(
        self, 
        cfg: DictConfig, 
        placement: ModelParallelComponentPlacement,
    ):
        Worker.__init__(self)
        # 初始化AgentLoopBase的功能
        self.cfg = cfg
        self.component_placement = placement
        self.loop = asyncio.get_running_loop()
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.rollout.model_dir)
        if self.cfg.rollout.get("custom_chat_template", None) is not None:
            self.tokenizer.chat_template = cfg.rollout.custom_chat_template
        # Configuration
        self.max_user_turns = cfg.agentloop.get("max_user_turns", 5)
        self.max_assistant_turns = cfg.agentloop.get("max_assistant_turns", 5)
        self.max_parallel_calls = cfg.agentloop.get("max_parallel_calls", 3)
        self.max_tool_response_length = cfg.agentloop.get("max_tool_response_length", 500)
        self.tool_response_truncate_side = cfg.agentloop.get("tool_response_truncate_side", "right")
        self.return_logprobs = cfg.agentloop.get("return_logprobs", False)
        
        # Initialize tool parser
        self.tool_parser = ToolParser.get_tool_parser("hermes", self.tokenizer)
        
        # Tools registry: 存储工具实例（在worker本地初始化）
        self.tools = None
        # 存储工具配置（可序列化）
        self.tool_config = None
        self._tools_initialized = False
        self.tool_schemas = None
    
        # 持有 rollout（AsyncSGLangWorker 的 group 句柄）
        self.rollout = None

        # Chat template configuration
        self.apply_chat_template_kwargs = cfg.data.get("apply_chat_template_kwargs", {})

        self.system_prompt = self.tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **self.apply_chat_template_kwargs
        )
        
    def _pre_process_rollout_request(
        self, request: RolloutRequest
    ) -> List[List[RolloutRequest]]:
        """Split rollout request into smaller groups (reference: SGLangWorker._pre_process_rollout_request).

        Returns: List[List[RolloutRequest]]
        - Outer list: batches
        - Inner list: requests aligned to group_size
        """
        group_size = request.n
        repeated_request = request.repeat()

        # Derive rollout batch size similar to AsyncSGLangWorker
        per_gpu = self.cfg.algorithm.get("rollout_batch_size_per_gpu", None)
        if per_gpu is None:
            rollout_batch_size = None
        else:
            tp = getattr(self.component_placement, "rollout_tp_size", 1)
            pp = getattr(self.component_placement, "rollout_pipeline_parallel_size", 1)
            rollout_batch_size = per_gpu * tp * pp

        if rollout_batch_size is not None:
            assert len(repeated_request.input_ids) % rollout_batch_size == 0, (
                f"rollout_batch_size {rollout_batch_size} must divide the total number of requests {len(repeated_request)}"
            )
            num_batch = len(repeated_request.input_ids) // rollout_batch_size
        else:
            num_batch = 1

        split_requests = repeated_request.split(num_batch)
        if self.component_placement.is_disaggregated:
            num_prompts_per_request = len(split_requests[0].input_ids) // group_size
            return [r.split(num_prompts_per_request) for r in split_requests]
        else:
            return [r.split(1) for r in split_requests]

    
    async def run_agentloop_rollout(self, input_channel: Channel, output_channel: Channel, tool_config, rollout):
        """运行AgentLoop rollout - 参考SGLangWorker的rollout方法
        
        Args:
            input_channel: 输入通道
            output_channel: 输出通道
            tool_config: 工具配置（可序列化的配置路径或配置字典），而不是工具实例
            rollout: rollout worker实例
        """
        # 存储工具配置而不是工具实例
        self.tool_config = tool_config
        self.rollout = rollout
        
        # 在worker本地初始化工具（如果尚未初始化）
        if not self._tools_initialized:
            await self._initialize_tools_locally()
            self._tools_initialized = True
        # 从输入通道获取请求 - 参考SGLangWorker
        request: RolloutRequest = input_channel.get()
        # output_channel.gpu_lock.acquire()
        # Repeat prompts based on the group_size config
        requests = self._pre_process_rollout_request(request)
        self.log_info(
            f"outer_loop size: {len(requests)}, group_size: {len(requests[0])}"
        )
        with self.worker_timer():
            for request_groups in requests:
                rollout_tasks = []
                for group in request_groups:
                    # 为当前group创建任务（直接调用本地 ToolAgentLoop）
                    
                    for message in group.raw_prompts:
                        task = asyncio.create_task(
                            self.run(message)
                        )
                        rollout_tasks.append(task)

                # 等待所有任务完成，保持顺序
                task_results = await asyncio.gather(*rollout_tasks)
                results = []
                for result in task_results:
                    results.append(result)

                # 汇总为 RolloutResult 对象，供后续 actor 使用
                # Clip to model limits to avoid mask/position size mismatch
                max_prompt_len = int(self.cfg.data.max_prompt_length)
                max_total_len = int(self.cfg.actor.model.encoder_seq_length)
                max_resp_len = max(1, max_total_len - max_prompt_len)

                prompt_ids = [r.prompt_ids[:max_prompt_len] for r in results]
                response_ids = [r.response_ids[:max_resp_len] for r in results]
                prompt_lengths = [len(p) for p in prompt_ids]
                response_lengths = [len(o) for o in response_ids]
                response_mask = [r.response_mask[:max_resp_len] for r in results]
                is_end = [True for _ in results]
                # print(f"len(results): {len(results)}")
                rollout_obj = RolloutResult(
                    num_sequence=len(results),
                    group_size=group.n,
                    prompt_lengths=prompt_lengths,
                    prompt_ids=prompt_ids,
                    response_lengths=response_lengths,
                    response_ids=response_ids,
                    is_end=is_end,
                    answers=group.answers,
                    response_mask=response_mask,
                )

                # 将结果发送到输出通道（回退为直接发送）
                await output_channel.put(
                    rollout_obj, async_op=True
                ).async_wait()
        
            # self.rollout.offload_engine().wait()

    async def run(self, messages: List[Dict[str, Any]], **kwargs) -> AgentLoopOutput:
        """Run the tool agent loop with token ids as input, return ids.

        - 使用 rollout.agenerate 直接生成 response_ids（token ids）。
        - 解析工具调用并执行，工具响应通过 chat template 转为 token ids 继续对话。
        """
        if self.rollout is None:
            raise RuntimeError("Rollout worker is not provided to ToolAgentLoop")

        response_mask = []
        response_logprobs = []
        user_turns, assistant_turns = 0, 0
        # print(f"messages: {messages}")
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                tools=self.tool_schemas,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        history_tool_calls = []
        while True:
            # Generate response from LLM
            # 截断上下文，避免超过模型最大长度
            max_prompt_len = int(self.cfg.data.get("max_prompt_length", 1024))
            max_total_len = int(self.cfg.actor.model.encoder_seq_length)
            max_resp_len = max(1, max_total_len - max_prompt_len)

            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]

            response_ids, log_probs = await self._agenerate(prompt_ids)
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]

            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)  # 1 for LLM generated tokens
            if log_probs:
                response_logprobs += log_probs
            # print(f"response_mask: {len(response_mask)}")
            assistant_turns += 1
            # Check termination conditions
            if len(response_mask) >= self.cfg.rollout.get("response_length", 1024):
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break
            
            # Extract tool calls from response
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            
            if not tool_calls:
                break
            print(f"tool_calls: {tool_calls}")
            # Execute tools in parallel with history propagation
            tool_calls = tool_calls[: self.max_parallel_calls]
            total_tool_responses, filtered_tool_calls, pending_pos = [], [], []
            for i, tool_call in enumerate(tool_calls):
                if isinstance(tool_call, ToolResponse):
                    total_tool_responses.append(tool_call)
                else:
                    total_tool_responses.append(None)
                    pending_pos.append(i)
                    filtered_tool_calls.append(tool_call)
            tool_calls = filtered_tool_calls
            tasks = []
            tools_kwargs = kwargs.get("tools_kwargs", {})
            for tool_call in tool_calls[: self.max_parallel_calls]:
                tools_kwargs_copy = copy.deepcopy(tools_kwargs)
                tools_kwargs_copy["history_tool_calls"] = list(history_tool_calls)
                tasks.append(self._call_tool(tool_call, tools_kwargs_copy))
                history_tool_calls.append(tool_call)
            tool_responses = await asyncio.gather(*tasks)
            for i, tool_response in zip(pending_pos[: self.max_parallel_calls], tool_responses, strict=False):
                total_tool_responses[i] = tool_response
            tool_responses = total_tool_responses
            if any(isinstance(item, Exception) for item in tool_responses):
                break
            # Convert tool responses to messages and tokenize
            tool_messages = []
            for tool_response in tool_responses:
                if isinstance(tool_response, dict):
                    text = tool_response.get("text", "")
                else:
                    text = str(tool_response) if tool_response is not None else ""
                message = {"role": "tool", "content": text}
                tool_messages.append(message)
            
            # Tokenize tool responses
            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    tool_messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]
            if len(response_mask) + len(tool_response_ids) >= self.cfg.rollout.get("response_length", 1024):
                break
            # Add tool response tokens
            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)  # 0 for tool response tokens
            if response_logprobs:
                response_logprobs += [0.0] * len(tool_response_ids)
            user_turns += 1
        
        # Separate prompt and response
        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]
        
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            num_turns=user_turns + assistant_turns + 1,
        )

    async def _agenerate(self, prompt_ids: list[int]) -> list[int]:
        import random
        instance_num = self.cfg.rollout.get("rollout_instance_num", 2)
        sglang_instance_id = random.randint(0, max(1, instance_num) - 1)
        generate_result = await self.rollout.execute_on(sglang_instance_id).agenerate(prompt_ids).async_wait()
        # print(generate_result)
        # 规范化返回：可能是 List[Dict] 或 List[List[Dict]] 或 Dict
        res_obj = generate_result
        if isinstance(res_obj, list):
            # 取第一个worker返回
            res_obj = res_obj[0] if len(res_obj) > 0 else {}
            # 若仍是列表（嵌套），再取第一项
            if isinstance(res_obj, list):
                res_obj = res_obj[0] if len(res_obj) > 0 else {}
        logprobs = []
        if self.return_logprobs:
            logprobs = [
                [item[0] for item in res["meta_info"]["output_token_logprobs"]]
                for res in generate_result
            ]
        # 现在应为 Dict
        if isinstance(res_obj, dict):
            return (res_obj.get("response_ids") or res_obj.get("output_ids", [])), logprobs
        return [], logprobs

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any] | None = None) -> str | dict:
        """Call a tool and return the response.

        Supports rstar2-style rich tools with create/execute/release, and returns text or {"text": ...}.
        """
        tools_kwargs = tools_kwargs or {}
        tool, instance_id = None, None
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            has_rich_iface = all(
                hasattr(tool, attr) and callable(getattr(tool, attr))
                for attr in ("create", "execute", "release")
            )

            if has_rich_iface:
                create_kwargs = tools_kwargs.get(tool_name, {}).get("create_kwargs", {})
                history_tool_calls = tools_kwargs.get("history_tool_calls", [])
                instance_id, _ = await tool.create(create_kwargs=create_kwargs, history_tool_calls=history_tool_calls)
                tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)
                text = getattr(tool_execution_response, "text", None) or ""
                # truncate
                if text and len(text) > self.max_tool_response_length:
                    if self.tool_response_truncate_side == "left":
                        text = text[: self.max_tool_response_length] + "...(truncated)"
                    elif self.tool_response_truncate_side == "right":
                        text = "(truncated)..." + text[-self.max_tool_response_length :]
                    else:
                        length = self.max_tool_response_length // 2
                        text = text[:length] + "...(truncated)..." + text[-length:]
                return {"text": text}
            else:
                # Not supported without rich interface in this integration
                return f"Error: Tool '{tool_name}' does not support rich interface"

        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return f"Error when executing tool: {e}"
        finally:
            try:
                if tool and instance_id and hasattr(tool, "release") and callable(getattr(tool, "release")):
                    await tool.release(instance_id)
            except Exception:
                logger.warning("Failed to release tool instance", exc_info=True)

    async def _initialize_tools_locally(self):
        """在worker本地初始化工具实例
        
        这个方法在Ray worker中被调用，用于在本地创建工具实例，
        避免从主进程传递不可序列化的工具对象。
        """
        if self.tool_config is None:
            logger.warning("No tool config provided, tools will not be initialized")
            self.tools = {}
            return
        
        try:
            logger.info(f"Initializing tools locally in worker with config: {self.tool_config}")
            
            if isinstance(self.tool_config, str):
                # 如果是配置文件路径
                from toolkits.rstar2.tools.tool_registry import load_tools_from_config, get_tool_schemas_from_config
                self.tools = await load_tools_from_config(self.tool_config)
                self.tool_schemas = get_tool_schemas_from_config(self.tool_config)
            else:
                logger.error(f"Unsupported tool_config type: {type(self.tool_config)}")
                self.tools = {}
            
            logger.info(f"Tools initialized locally: {list(self.tools.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools locally: {e}", exc_info=True)
            self.tools = {}
    
    def init_worker(self):
        # 兼容旧接口（无实际操作）
        logger.info("ToolAgentLoop ready")

    # Remove built-in mock tools; this loop expects injected rstar2-style tools.
