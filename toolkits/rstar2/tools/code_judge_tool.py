# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from functools import partial, wraps
from typing import Any, Optional
from uuid import uuid4

import aiohttp

from .request_processor import RequestProcessor
from .code_judge_utils import generate_tool_call_code, generate_tool_call_input, run_tool_calls_on_server_async


# Compatibility decorator for rollout_trace_op (no-op in RLinf)
def rollout_trace_op(func):
    """Compatibility decorator for verl's rollout_trace_op"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper


class ToolResponse:
    def __init__(self, text: str | None = None, image=None, video=None):
        self.text = text
        self.image = image
        self.video = video


class OpenAIFunctionToolSchema:
    """Placeholder for OpenAI function tool schema compatibility"""
    def __init__(self, **kwargs):
        self.schema = kwargs


class BaseTool:
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        self.config = config
        self.tool_schema = tool_schema


class CodeJudgeTool(BaseTool):
    """
    CodeJudgeTool compatible with both verl and RLinf frameworks.
    Supports both config-based initialization (verl style) and direct parameter initialization (RLinf style).
    """
    def __init__(self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None, 
                 name: str = None, *, host_addr: str = "localhost", host_port: str | int = "8000", 
                 batch_size: int = 1, concurrency: int = 1, batch_timeout_seconds: float = 30.0):
        # Support both verl-style (config dict) and RLinf-style (direct params) initialization
        if config is None and name is not None:
            # RLinf-style initialization
            config = {
                "host_addr": host_addr,
                "host_port": host_port,
                "request_processor_batch_size": batch_size,
                "request_processor_concurrency": concurrency,
                "request_processor_batch_timeout_seconds": batch_timeout_seconds,
            }
            self.name = name
        else:
            # verl-style initialization
            self.name = getattr(self, 'name', 'code_judge_tool')
            
        super().__init__(config or {}, tool_schema)
        self._instance_dict = {}

        host_addr = self.config.get("host_addr", "localhost")
        host_port = self.config.get("host_port", "8000")
        run_jupyter_tool_calls_on_server_async = partial(
            run_tool_calls_on_server_async,
            generate_tool_call_code=generate_tool_call_code,
            generate_tool_call_input=generate_tool_call_input,
            host_addr=host_addr,
            host_port=host_port,
        )
        request_processor_batch_size = self.config.get("request_processor_batch_size", 1)
        request_processor_concurrency = self.config.get("request_processor_concurrency", 1)
        request_processor_batch_timeout_seconds = self.config.get("request_processor_batch_timeout_seconds", 30)
        tool_connector = aiohttp.TCPConnector(
            limit=request_processor_concurrency, force_close=True, enable_cleanup_closed=True
        )
        tool_timeout = aiohttp.ClientTimeout(total=60)
        tool_session = aiohttp.ClientSession(connector=tool_connector, timeout=tool_timeout)
        self.request_processor = RequestProcessor(
            batch_size=request_processor_batch_size,
            batch_timeout_seconds=request_processor_batch_timeout_seconds,
            session=tool_session,
            concurrency=request_processor_concurrency,
            batch_submit_func=run_jupyter_tool_calls_on_server_async,
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def _start_request_processor(self):
        if not self.request_processor._running:
            await self.request_processor.start()

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class SimJupyterTool(CodeJudgeTool):
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        assert "history_tool_calls" in kwargs, "history_tool_calls must be provided in kwargs"
        await self._start_request_processor()
        history_tool_calls = []
        for history_tool_call in kwargs["history_tool_calls"]:
            if history_tool_call.name == "jupyter_code":
                try:
                    arguments = json.loads(history_tool_call.arguments)
                    assert len(arguments) == 1 and "code" in arguments
                    history_tool_calls.append(
                        {
                            "name": "jupyter_code",
                            "arguments": {
                                "code": arguments["code"],
                            },
                        }
                    )
                except Exception:
                    pass

        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
            "history_tool_calls": history_tool_calls,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        tool_call = {
            "name": "jupyter_code",
            "arguments": {
                "code": code,
            },
            "history_tool_calls": self._instance_dict[instance_id]["history_tool_calls"],
        }
        result_text = await self.request_processor.send_request(tool_call)
        return ToolResponse(text=result_text), 0.0, {}
        
    # Add RLinf-style execute method for compatibility
    async def __call__(self, **kwargs):
        """RLinf-style tool execution interface"""
        # Create a temporary instance for RLinf-style calls
        instance_id = "temp_" + str(uuid4())
        await self.create(instance_id, **kwargs)
        try:
            result, cost, info = await self.execute(instance_id, kwargs)
            return result.text if result.text else ""
        finally:
            await self.release(instance_id)


class PythonTool(CodeJudgeTool):
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        await self._start_request_processor()

        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        input_data = parameters.get("input", "")
        tool_call = {
            "name": "python_code_with_standard_io",
            "arguments": {
                "code": code,
                "input": input_data,
            },
        }
        result_text = await self.request_processor.send_request(tool_call)
        return ToolResponse(text=result_text), 0.0, {}
        
    # Add RLinf-style execute method for compatibility
    async def __call__(self, **kwargs):
        """RLinf-style tool execution interface"""
        # Create a temporary instance for RLinf-style calls
        instance_id = "temp_" + str(uuid4())
        await self.create(instance_id, **kwargs)
        try:
            result, cost, info = await self.execute(instance_id, kwargs)
            return result.text if result.text else ""
        finally:
            await self.release(instance_id)