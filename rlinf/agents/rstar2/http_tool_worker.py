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

import aiohttp
import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import ToolChannelRequest, ToolChannelResponse
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker

from rlinf.agents.rstar2.http_code_judge_tool import (
    ToolBase,
    JupyterTool,
    LeanTool,
    PythonTool,
)


class HttpToolWorker(ToolWorker):
    TOOL_SET_CLS = [JupyterTool, LeanTool, PythonTool]

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.request_processor_task = None
        self.client_session = None
        self.active_tasks = set()
        self.tools: Dict[str, ToolBase] = {}

        # TODO: make it as config
        self.concurrency_limit = self.cfg.tools.codejudge.concurrency_limit
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

        self._logger = None
        self.init_tools()

    def init_tools(self):
        for tool_cls in self.TOOL_SET_CLS:
            tool = tool_cls(self.cfg)
            self.tools[tool.name] = tool

    @property
    def logger(self):
        """Lazy initialization of logger."""
        if self._logger is None:
            self._logger = logging.getLogger(f"{self.__class__.__name__}")
        return self._logger

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        """Initialize the worker with communication channels."""
        super().init_worker(input_channel, output_channel)

    def start_server(self):
        """Start the request processor task."""
        self.is_running = True
        try:
            loop = asyncio.get_running_loop()
            self.request_processor_task = loop.create_task(self._process_requests())
            tool_connector = aiohttp.TCPConnector(
                limit=self.concurrency_limit, force_close=True, enable_cleanup_closed=True
            )
            tool_timeout = aiohttp.ClientTimeout(total=60)
            self.client_session = aiohttp.ClientSession(connector=tool_connector, timeout=tool_timeout)
        except RuntimeError:
            # No event loop running, will be created later
            self.request_processor_task = None
            self.client_session = None

    def stop_server(self):
        """Cleanup worker resources."""
        if self.logger:
            self.log_info(f"Cleaning up {self.__class__.__name__}")

        self.is_running = False

        # Cancel all activate tasks
        if self.active_tasks:
            try:
                loop = asyncio.get_running_loop()
                loop.run_until_complete(asyncio.wait(self.active_tasks, timeout=30.0))
            except RuntimeError:
                self.logger.warning(f"{self.__class__.__name__}.stop_server: try to cancel all activate tasksm but no running loop.")
            finally:
                self.active_tasks = set()

        # Cancel request processor task
        if (
            hasattr(self, "request_processor_task")
            and self.request_processor_task
            and not self.request_processor_task.done()
        ):
            self.request_processor_task.cancel()
        
        if self.client_session and not self.client_session.closed:
            try:
                loop = asyncio.get_running_loop()
                loop.run_until_complete(self.client_session.close())
            except RuntimeError:
                self.logger.warning(f"{self.__class__.__name__}.stop_server: try to close session but no running loop.")
            finally:
                self.client_session = None

    async def _process_requests(self):
        async def send_request(session_id: str, request: ToolChannelRequest):
            assert request.request_type == "execute", \
                f"{self.__class__.__name__} is stateless right now, only `execute` is accepted."
            assert request.tool_name in self.tools, \
                f"{self.__class__.__name__}: {request.tool_name} not registered."
            tool = self.tools[request.tool_name]
            response = await tool.execute(request, self._send_request)
            await self.output_channel.put(
                response, key=session_id, async_op=True
            ).async_wait()
            self.logger.info(f"{self.__class__.__name__}._process_requests: send tool response")

        while self.is_running:
            request: ToolChannelRequest = await self.input_channel.get(
                async_op=True
            ).async_wait()
            if request.request_type != "execute":
                await self.output_channel.put(
                    ToolChannelResponse(success=True), key=request.session_id, async_op=True
                ).async_wait()
            else:
                asyncio.create_task(self._safe_run_task(send_request(session_id=request.session_id, request=request), request=request))

    async def _safe_run_task(self, coro, request: ToolChannelRequest):
        """Catch and log the error from the task"""
        task = asyncio.current_task()
        self.active_tasks.add(task)
        async with self.semaphore:
            try:
                await coro
            except Exception as e:
                self.logger.error(
                    f"Exception occurred while processing tool request {request.tool_name} "
                    f"(Type: {request.request_type}): {e}",
                    exc_info=True
                )
            finally:
                self.active_tasks.remove(task)

    async def _send_request(self, url, data):
        async with self.client_session.post(url, json=data) as response:
            response.raise_for_status()
            response_json = await response.json()
        return response_json