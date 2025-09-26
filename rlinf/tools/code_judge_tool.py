import json
from functools import partial
from typing import Any, Optional
from uuid import uuid4

import aiohttp

from .request_processor import RequestProcessor
from .code_judge_utils import run_tool_calls_on_server_async


class ToolResponse:
    def __init__(self, text: str | None = None, image=None, video=None):
        self.text = text
        self.image = image
        self.video = video


class BaseTool:
    def __init__(self, name: str):
        self.name = name


class CodeJudgeTool(BaseTool):
    def __init__(self, name: str, *, host_addr: str = "localhost", host_port: str | int = "8000", batch_size: int = 1, concurrency: int = 1, batch_timeout_seconds: float = 30.0):
        super().__init__(name)
        # Defer aiohttp session and RequestProcessor initialization to `create()` when an event loop is guaranteed.
        self._session: aiohttp.ClientSession | None = None
        self._request_processor: RequestProcessor | None = None
        self._instance_dict: dict[str, dict] = {}
        # Store configs for lazy init
        self._cfg_host_addr = host_addr
        self._cfg_host_port = host_port
        self._cfg_batch_size = batch_size
        self._cfg_concurrency = concurrency
        self._cfg_batch_timeout_seconds = batch_timeout_seconds

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        # Lazy init network stack here to ensure running event loop exists
        if self._request_processor is None:
            connector = aiohttp.TCPConnector(limit=self._cfg_concurrency, force_close=True, enable_cleanup_closed=True)
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            self._request_processor = RequestProcessor(
                batch_size=self._cfg_batch_size,
                batch_timeout_seconds=self._cfg_batch_timeout_seconds,
                session=self._session,
                concurrency=self._cfg_concurrency,
                batch_submit_func=partial(
                    run_tool_calls_on_server_async,
                    host_addr=self._cfg_host_addr,
                    host_port=self._cfg_host_port,
                ),
            )
        if not getattr(self._request_processor, "_running", False):
            await self._request_processor.start()
        history_tool_calls = kwargs.get("history_tool_calls", [])
        self._instance_dict[instance_id] = {
            "history_tool_calls": history_tool_calls,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        # Map to a single tool_call and send via RequestProcessor
        tool_call = {
            "name": self.name,
            "arguments": {
                **parameters,
            },
        }
        assert self._request_processor is not None, "RequestProcessor is not initialized. Call create() first."
        text = await self._request_processor.send_request(tool_call)
        return ToolResponse(text=text), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


