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
from megatron.core import parallel_state

from .agent_loop import AgentLoopOutput
from .tool_parser import FunctionCall, ToolParser, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLINF_LOGGING_LEVEL", "WARN"))


class ToolAgentLoop:
    """Simple tool agent loop that can interact with tools.

    重构为普通类：不再继承 Worker，也不使用 WorkerGroup/Channel。
    通过注入的 rollout 直接调用 agenerate，输入/输出均为 token ids。
    """

    def __init__(
        self, 
        config, 
        placement,
        **kwargs
    ):
        # 初始化AgentLoopBase的功能
        self.config = config
        self._placement = placement
        self.loop = asyncio.get_event_loop()
        
        # 自己创建tokenizer，参考SGLangWorker的实现
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.rollout.model_dir)
        
        # Configuration
        self.max_user_turns = config.rollout.get("max_user_turns", 5)
        self.max_assistant_turns = config.rollout.get("max_assistant_turns", 5)
        self.max_parallel_calls = config.rollout.get("max_parallel_calls", 3)
        self.max_tool_response_length = config.rollout.get("max_tool_response_length", 500)
        self.tool_response_truncate_side = config.rollout.get("tool_response_truncate_side", "right")
        
        # Initialize tool parser
        self.tool_parser = ToolParser.get_tool_parser("hermes", self.tokenizer)
        
        # Tools registry: expect rstar2-style tools to be injected via kwargs["tools"].
        # If not provided, keep empty dict (no mock tools).
        self.tools = kwargs.get("tools", {})
        
        # Log tool information for debugging
        if self.tools:
            logger.info(f"ToolAgentLoop initialized with {len(self.tools)} tools: {list(self.tools.keys())}")
            # Check if tools are shared instances
            for tool_name, tool in self.tools.items():
                if hasattr(tool, 'request_processor') and tool.request_processor:
                    is_running = getattr(tool.request_processor, '_running', False)
                    logger.debug(f"Tool {tool_name}: request_processor running={is_running}")
        else:
            logger.info("ToolAgentLoop initialized with no tools")
        
        # 持有 rollout（AsyncSGLangWorker 的 group 句柄）
        self.rollout = kwargs.get("rollout")

        # Chat template configuration
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})

        self.system_prompt = self.tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **self.apply_chat_template_kwargs
        )

    async def run(self, prompt_ids: list[int], sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the tool agent loop with token ids as input, return ids.

        - 使用 rollout.agenerate 直接生成 response_ids（token ids）。
        - 解析工具调用并执行，工具响应通过 chat template 转为 token ids 继续对话。
        """
        if self.rollout is None:
            raise RuntimeError("Rollout worker is not provided to ToolAgentLoop")

        request_id = uuid4().hex
        response_mask = []
        response_logprobs = []
        user_turns, assistant_turns = 0, 0
        
        # Main conversation loop
        history_tool_calls = []
        while True:
            # Generate response from LLM
            # 截断上下文，避免超过模型最大长度
            max_prompt_len = int(self.config.data.get("max_prompt_length", 1024))
            max_total_len = int(self.config.actor.model.encoder_seq_length)
            max_resp_len = max(1, max_total_len - max_prompt_len)

            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]

            response_ids = await self._agenerate(prompt_ids, sampling_params)
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]

            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)  # 1 for LLM generated tokens
            # print(f"response_mask: {len(response_mask)}")
            assistant_turns += 1
            # Check termination conditions
            if len(response_mask) >= self.config.rollout.get("response_length", 1024):
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break
            
            # Extract tool calls from response
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                break
            
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
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break
            # Add tool response tokens
            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)  # 0 for tool response tokens
            user_turns += 1
        
        # Separate prompt and response
        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]
        
        # Decode text for output
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data={},
            num_turns=user_turns + assistant_turns + 1,
            prompt_text=prompt_text,
            response_text=response_text,
        )

    async def _agenerate(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]:
        import random
        instance_num = self.config.rollout.get("rollout_instance_num", 1)
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

        # 现在应为 Dict
        if isinstance(res_obj, dict):
            return res_obj.get("response_ids") or res_obj.get("output_ids", [])
        return []

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

    def init_worker(self):
        # 兼容旧接口（无实际操作）
        logger.info("ToolAgentLoop ready")

    # Remove built-in mock tools; this loop expects injected rstar2-style tools.
