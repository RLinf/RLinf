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
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Optional

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import (
    MCPRequest,
    MCPRequestType,
    MCPResponse,
    MCPSessionState,
    ToolChannelRequest,
    ToolChannelResponse,
)
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker


class FileSystemMCPSession:
    """MCP session manager for handling MCP server communication."""

    def __init__(
        self, session_id: str, config: DictConfig, mount_dir: Optional[str] = None
    ):
        self.session_id = session_id
        self.config = config
        self.mount_dir = mount_dir
        self.state = MCPSessionState.INITIALIZING

        # Session resources
        self.client_session: Optional[ClientSession] = (
            None  # not persisted across requests
        )
        self.server_process: Optional[subprocess.Popen] = None
        self.temp_dir: Optional[str] = None
        self.docker_image: Optional[str] = None

        # Statistics
        self.created_at = time.time()
        self.last_activity = time.time()
        self.requests_processed = 0

        self._logger = None

    @property
    def logger(self):
        """Lazy initialization of logger."""
        if self._logger is None:
            self._logger = logging.getLogger(f"FileSystemMCPSession.{self.session_id}")
        return self._logger

    async def start(self) -> bool:
        """Start the MCP session."""
        try:
            self.logger.info(f"Starting MCP session {self.session_id}")

            # Use configured mount directory or create temporary one
            if self.mount_dir:
                # Ensure the mount directory exists
                os.makedirs(self.mount_dir, exist_ok=True)
                self.temp_dir = self.mount_dir
                self.logger.info(f"Using configured mount directory: {self.temp_dir}")
            else:
                # Create temporary directory for MCP server
                self.temp_dir = tempfile.mkdtemp(
                    prefix=f"mcp_filesystem_{self.session_id}_"
                )
                self.logger.info(f"Created temp directory: {self.temp_dir}")

            # Get Docker image from config (store for per-request use)
            self.docker_image = self.config.get("mcp", {}).get(
                "docker_image", "mcp/filesystem:1.0.2"
            )

            # Mark as ready: we will open/close stdio per request
            self.state = MCPSessionState.CONNECTED
            self.logger.info(
                f"MCP session {self.session_id} prepared successfully (per-request connection)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error starting MCP session {self.session_id}: {e}")
            self.state = MCPSessionState.FAILED
            return False

    async def _create_server_params(self) -> StdioServerParameters:
        """Create server parameters for docker stdio connection."""
        return StdioServerParameters(
            command="docker",
            args=[
                "run",
                "--rm",
                "-i",
                "--mount",
                f"type=bind,src={self.temp_dir},dst=/projects/test",
                self.docker_image or "mcp/filesystem:1.0.2",
                "/projects",
            ],
        )

    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process an MCP request."""
        start_time = time.time()
        self.last_activity = start_time

        try:
            if self.state != MCPSessionState.CONNECTED:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="MCP session not connected",
                    execution_time=time.time() - start_time,
                )

            self.logger.debug(f"Processing MCP request: {request.request_type}")

            server_params = await self._create_server_params()
            # Open stdio and client session per request to ensure enter/exit on same task
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Process based on request type
                    if request.request_type == MCPRequestType.LIST_TOOLS:
                        result = await session.list_tools()
                        result_data = {
                            "tools": [
                                {"name": tool.name, "description": tool.description}
                                for tool in result.tools
                            ]
                        }
                    elif request.request_type == MCPRequestType.CALL_TOOL:
                        if not request.tool_name:
                            raise ValueError("Tool name is required for tool calls")
                        result = await session.call_tool(
                            request.tool_name, request.tool_arguments or {}
                        )
                        content = []
                        for item in result.content:
                            if hasattr(item, "text"):
                                content.append(item.text)
                            else:
                                content.append(str(item))
                        result_data = {
                            "content": content,
                            "structured_content": getattr(
                                result, "structuredContent", None
                            ),
                        }
                    elif request.request_type == MCPRequestType.LIST_RESOURCES:
                        result = await session.list_resources()
                        result_data = {
                            "resources": [
                                {"uri": resource.uri, "name": resource.name}
                                for resource in result.resources
                            ]
                        }
                    elif request.request_type == MCPRequestType.READ_RESOURCE:
                        if not request.resource_uri:
                            raise ValueError(
                                "Resource URI is required for resource reading"
                            )
                        from pydantic import AnyUrl

                        result = await session.read_resource(
                            AnyUrl(request.resource_uri)
                        )
                        content = []
                        for item in result.contents:
                            if hasattr(item, "text"):
                                content.append(item.text)
                            else:
                                content.append(str(item))
                        result_data = {"content": content}
                    elif request.request_type == MCPRequestType.LIST_PROMPTS:
                        result = await session.list_prompts()
                        result_data = {
                            "prompts": [
                                {"name": prompt.name, "description": prompt.description}
                                for prompt in result.prompts
                            ]
                        }
                    elif request.request_type == MCPRequestType.GET_PROMPT:
                        if not request.prompt_name:
                            raise ValueError(
                                "Prompt name is required for prompt getting"
                            )
                        result = await session.get_prompt(
                            request.prompt_name, request.prompt_arguments or {}
                        )
                        messages = []
                        for msg in result.messages:
                            message_data = {"role": msg.role, "content": []}
                            for content_item in msg.content:
                                if hasattr(content_item, "text"):
                                    message_data["content"].append(
                                        {"type": "text", "text": content_item.text}
                                    )
                                else:
                                    message_data["content"].append(
                                        {
                                            "type": str(type(content_item).__name__),
                                            "data": str(content_item),
                                        }
                                    )
                            messages.append(message_data)
                        result_data = {"messages": messages}
                    else:
                        raise ValueError(
                            f"Unknown request type: {request.request_type}"
                        )

            self.requests_processed += 1
            execution_time = time.time() - start_time

            return MCPResponse(
                request_id=request.request_id,
                success=True,
                result=result_data,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error processing MCP request {request.request_id}: {e}")

            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                execution_time=execution_time,
            )

    async def cleanup(self):
        """Cleanup MCP session resources."""
        try:
            # No long-lived client session to close here; per-request handles its own cleanup

            # Only cleanup temp directory if it was created by us (not configured)
            if self.temp_dir and not self.mount_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            elif self.mount_dir:
                self.logger.info(
                    f"Preserved configured mount directory: {self.temp_dir}"
                )

            self.state = MCPSessionState.TERMINATED
            self.logger.info(f"MCP session {self.session_id} cleaned up")

        except Exception as e:
            self.logger.error(f"Error cleaning up MCP session {self.session_id}: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "requests_processed": self.requests_processed,
            "uptime": time.time() - self.created_at,
        }


class MCPFilesystemClientWorker(ToolWorker):
    """MCP Filesystem Client Worker for interacting with MCP filesystem servers."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.is_running = False

        # MCP session management
        self.sessions: dict[str, FileSystemMCPSession] = {}
        self._session_lock = None

        # Get mount directory from config
        self.mount_dir = cfg.get("mcp", {}).get("mount_dir", None)
        if self.mount_dir:
            # Ensure mount directory exists
            os.makedirs(self.mount_dir, exist_ok=True)

    @property
    def session_lock(self):
        """Lazy initialization of session lock."""
        if self._session_lock is None:
            self._session_lock = threading.RLock()
        return self._session_lock

    @property
    def logger(self):
        """Lazy initialization of logger."""
        if self._logger is None:
            self._logger = logging.getLogger("MCPFilesystemClientWorker")
        return self._logger

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        """Initialize the worker with communication channels."""
        super().init_worker(input_channel, output_channel)

        # Initialize components
        self.sessions = {}

        self.log_info("MCP Filesystem Client Worker initialized")
        return self

    def start_server(self):
        """Start the request processor task."""
        self.is_running = True
        try:
            loop = asyncio.get_running_loop()
            self.request_processor_task = loop.create_task(self._process_requests())
        except RuntimeError:
            # No event loop running, will be created later
            self.request_processor_task = None

    async def session_start(self, request: ToolChannelRequest) -> bool:
        session = await self._get_or_create_session(
            request.session_id, f"/projects/{request.session_id}"
        )
        if not session:
            response = ToolChannelResponse(
                success=False, result="Failed to create MCP session"
            )
        else:
            response = ToolChannelResponse(
                success=True,
            )
        await self.output_channel.put(
            response, key=request.session_id, async_op=True
        ).async_wait()

    async def session_end(self, request: ToolChannelRequest) -> bool:
        await self._remove_session(request.session_id)
        response = ToolChannelResponse(
            success=True,
        )
        await self.output_channel.put(
            response, key=request.session_id, async_op=True
        ).async_wait()

    async def session_execute(self, request: ToolChannelRequest) -> bool:
        request_work = MCPRequest(
            request_id=str(uuid.uuid4()),
            request_type=MCPRequestType.CALL_TOOL,
            tool_name=request.tool_name,
            tool_arguments=request.tool_args,
            timeout=30,
            metadata={
                "session_id": request.session_id,
                "mount_dir": f"/projects/{request.session_id}",
            },
        )
        result = await self._process_mcp_request(request_work)
        response = ToolChannelResponse(
            success=True,
            result=result.result,
        )
        await self.output_channel.put(
            response, key=request.session_id, async_op=True
        ).async_wait()

    async def _process_requests(self):
        """Process incoming requests asynchronously."""
        self.log_info("Starting async request processor")

        while self.is_running:
            try:
                # Get request from input channel
                request: ToolChannelRequest = await self.input_channel.get(
                    async_op=True
                ).async_wait()
                self.logger.info(
                    f"MCPFilesystemClientWorker._process_requests: got request: {request}"
                )
                request_type = request.request_type
                if request_type == "session_start":
                    asyncio.create_task(self.session_start(request))
                elif request_type == "session_end":
                    asyncio.create_task(self.session_end(request))
                elif request_type == "execute":
                    asyncio.create_task(self.session_execute(request))
                else:
                    assert False, f"Unsupported request type: {request_type}"

            except Exception as e:
                if "QueueEmpty" not in str(e):
                    self.logger.error(f"Error processing request: {e}")
                await asyncio.sleep(0.1)

    async def _cleanup_all_sessions(self):
        """Async helper to cleanup all sessions using gather."""
        tasks = []
        for session_id in list(self.sessions.keys()):
            tasks.append(self._remove_session(session_id))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_mcp_request(self, request: MCPRequest) -> MCPResponse:
        """Process a single MCP request."""
        session_id = request.metadata.get("session_id", "default")
        # Determine mount_dir for this request: prefer request metadata, fallback to worker default
        requested_mount_dir = request.metadata.get(
            "mount_dir", getattr(self, "mount_dir", None)
        )

        try:
            # Get or create session
            session = await self._get_or_create_session(session_id, requested_mount_dir)

            if not session:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="Failed to create MCP session",
                )

            # Process request
            return await session.process_request(request)

        except Exception as e:
            self.logger.error(f"Error processing MCP request: {e}")
            return MCPResponse(
                request_id=request.request_id, success=False, error_message=str(e)
            )

    async def _get_or_create_session(
        self, session_id: str, mount_dir: Optional[str]
    ) -> Optional[FileSystemMCPSession]:
        """Get existing session or create new one. Each session_id is bound to one mount_dir."""
        async with asyncio.Lock():
            if session_id in self.sessions:
                session = self.sessions[session_id]
                # Enforce one mount_dir per session
                if (
                    mount_dir
                    and session.mount_dir
                    and os.path.abspath(session.mount_dir) != os.path.abspath(mount_dir)
                ):
                    self.logger.error(
                        f"Session '{session_id}' already bound to mount_dir '{session.mount_dir}',"
                        f" but requested '{mount_dir}'. One mount_dir per session is required."
                    )
                    return None
                if session.state == MCPSessionState.CONNECTED:
                    return session
                else:
                    # Remove failed/terminated session
                    await self._remove_session(session_id)

            # Create new session with mount directory
            # If mount_dir not provided, fallback to worker default; else ensure it exists
            effective_mount_dir = mount_dir or getattr(self, "mount_dir", None)
            # If still None, allocate a per-session temporary directory and bind it
            if effective_mount_dir is None:
                try:
                    effective_mount_dir = tempfile.mkdtemp(
                        prefix=f"mcp_session_{session_id}_"
                    )
                    self.log_info(
                        f"Allocated temp mount_dir for session '{session_id}': {effective_mount_dir}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to allocate temp mount_dir for session '{session_id}': {e}"
                    )
                    return None
            if effective_mount_dir:
                try:
                    os.makedirs(effective_mount_dir, exist_ok=True)
                except Exception as e:
                    self.logger.error(
                        f"Failed to ensure mount_dir '{effective_mount_dir}': {e}"
                    )
                    return None

            session = FileSystemMCPSession(session_id, self.cfg, effective_mount_dir)

            if await session.start():
                self.sessions[session_id] = session
                self.log_info(
                    f"Created new MCP session: {session_id}, mount_dir: {effective_mount_dir}"
                )
                return session
            else:
                self.logger.error(f"Failed to create MCP session: {session_id}")
                return None

    async def _remove_session(self, session_id: str):
        """Remove session from manager."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            await session.cleanup()
            self.log_info(f"Removed MCP session: {session_id}")

    def stop_server(self):
        """Cleanup worker resources."""
        if self.logger:
            self.log_info("Cleaning up MCP Filesystem Client Worker")

        self.is_running = False

        # Cancel request processor task
        if (
            hasattr(self, "request_processor_task")
            and self.request_processor_task
            and not self.request_processor_task.done()
        ):
            self.request_processor_task.cancel()

        # Cleanup all sessions
        if self.sessions:
            try:
                loop = asyncio.get_running_loop()
                # Schedule coroutine task (not a Future) to avoid TypeError
                loop.create_task(self._cleanup_all_sessions())
            except RuntimeError:
                # No event loop running, cleanup will happen when loop starts
                pass

        return self

    def get_stats(self) -> dict[str, Any]:
        """Get worker statistics."""
        session_stats = []
        for session in self.sessions.values():
            session_stats.append(session.get_stats())

        return {
            "worker_type": "MCPFilesystemClientWorker",
            "is_running": self.is_running,
            "sessions": session_stats,
            "total_sessions": len(self.sessions),
        }

    def __getstate__(self):
        """Custom serialization to exclude non-serializable objects."""
        state = self.__dict__.copy()
        # Only keep serializable attributes
        serializable_keys = ["cfg", "is_running"]
        return {k: v for k, v in state.items() if k in serializable_keys}

    def __setstate__(self, state):
        """Custom deserialization to restore non-serializable objects."""
        self.__dict__.update(state)
        # Initialize non-serializable attributes to None
        self.sessions = {}
        self._session_lock = None
        self.request_processor_task = None
        self.input_channel = None
        self.output_channel = None
