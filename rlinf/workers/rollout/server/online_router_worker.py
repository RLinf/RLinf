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
import random
import time
import uuid
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from omegaconf.dictconfig import DictConfig
from pydantic import BaseModel

from rlinf.scheduler import Worker
from rlinf.utils.placement import ComponentPlacement
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker


class CompleteRequest(BaseModel):
    """Complete request model."""

    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0
    stop: Optional[list[str]] = None
    stream: Optional[bool] = False


class CompleteResponse(BaseModel):
    """Complete response model."""

    id: str
    choices: list[dict[str, Any]]
    model: str
    created: int
    object: str = "text_completion"


class OnlineRouterWorker(Worker):
    """Online router worker with FastAPI server for handling complete and completionTrack requests."""

    def __init__(self, cfg: DictConfig, placement: ComponentPlacement):
        Worker.__init__(self)

        self._cfg = cfg

        # Configuration
        self._server_host = cfg.server.online_router.get("host", "0.0.0.0")
        self._server_port = cfg.server.online_router.get("port", 8081)
        self._rollout_instance_num = placement.rollout_dp_size
        self._sampling_params = SGLangWorker.get_sampling_param_from_config(
            self._cfg.algorithm.sampling_params
        )
        if "stop" in self._cfg.algorithm.sampling_params:
            self._sampling_params["stop"] = self._cfg.algorithm.sampling_params["stop"]

        # Sync weight state management
        self._sync_model_lock = asyncio.Lock()
        self._sync_model_in_progress = False
        self._pending_requests: list[asyncio.Future] = []

        # Request synchronization state
        self._sync_in_progress = False
        self._old_requests_complete = asyncio.Event()
        self._new_requests_blocked = asyncio.Event()
        self._new_requests_blocked.set()  # Initially allow new requests
        self._blocked_requests: list[asyncio.Future] = []

        # Request tracking
        self._active_requests: dict[str, asyncio.Future] = {}

        # Setup FastAPI routes
        self._setup_routes()
        self._server_task = None

    def _setup_routes(self):
        """Setup FastAPI routes."""
        app = FastAPI(title="OnlineRouterWorker", version="1.0.0")
        app.add_api_route("/v1/completions", self._handle_complete, methods=["POST"])

        # Init the HTTP server
        self._server = uvicorn.Server(
            uvicorn.Config(
                app, host=self._server_host, port=self._server_port, log_level="info"
            )
        )

    def server_start(self):
        """Start service."""
        assert self._server_task is None

        # Start server in background task
        self._server_task = asyncio.create_task(self._server.serve())

        self.log_info(f"service started on {self._server_host}:{self._server_port}")

    async def server_stop(self):
        """Stop service."""
        assert self._server_task is not None

        # Stop the HTTP server
        self._server.should_exit = True

        # Wait the HTTP server to stop
        await self._server_task

        self._server_task = None
        self.log_info("service stopped")

    async def _handle_complete(self, request: CompleteRequest):
        """Handle complete requests with synchronization support."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Check if sync is in progress
        if self._sync_in_progress:
            # Wait for old requests to complete
            await self._old_requests_complete.wait()
            # Block new requests during sync
            await self._new_requests_blocked.wait()

        # Create future for this request
        future = asyncio.Future()
        self._active_requests[request_id] = future

        try:
            # Forward request to rollout worker
            sglang_instance_id = random.randint(0, self._rollout_instance_num - 1)
            if request.stop is not None:
                sampling_params = copy.deepcopy(self._sampling_params)
                sampling_params["stop"] = request.stop
            generate_result = (
                await self.rollout_worker.execute_on(sglang_instance_id)
                .async_generate(prompt=request.prompt, sampling_params=sampling_params)
                .async_wait()
            )
            generate_result = generate_result[0][0]

            if not request.stream:
                # Create response
                response = CompleteResponse(
                    id=str(request_id),
                    choices=[
                        {
                            "text": generate_result["text"],
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": generate_result["meta_info"][
                                "finish_reason"
                            ]["type"],
                        }
                    ],
                    created=int(start_time),
                    model="test-model",
                    object="text_completion",
                )
            else:

                def generate_stream():
                    # Send final chunk with finish_reason
                    final_data = {
                        "id": request_id,
                        "object": "text_completion.chunk",
                        "created": int(start_time),
                        "model": "test-model",
                        "choices": [
                            {
                                "text": generate_result["text"],
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"
                    yield "data: [DONE]\n\n"

                response = StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                    },
                )

            # Set future result
            future.set_result(response)
            return response

        finally:
            # Clean up
            if request_id in self._active_requests:
                del self._active_requests[request_id]

    async def init_worker(self, rollout_worker: SGLangWorker):
        """Initialize the worker."""
        self.rollout_worker = rollout_worker

    async def sync_model_start(self):
        """Start model synchronization. Block new requests and wait for old ones to complete."""
        async with self._sync_model_lock:
            assert not self._sync_in_progress

            self.log_info("Starting model synchronization...")
            self._sync_in_progress = True

            # Clear the event to block new requests
            self._new_requests_blocked.clear()

            # Wait for all existing requests to complete
            if self._active_requests:
                self.log_info(
                    f"Waiting for {len(self._active_requests)} active requests to complete..."
                )
                # Wait for all active requests to finish
                await asyncio.gather(
                    *self._active_requests.values(), return_exceptions=True
                )

            # Set event to indicate old requests are complete
            self._old_requests_complete.set()
            self.log_info("All old requests completed, sync can proceed")

    async def sync_model_end(self):
        """End model synchronization. Resume processing of blocked requests."""
        async with self._sync_model_lock:
            assert self._sync_in_progress

            self.log_info("Ending model synchronization...")

            # Reset sync state
            self._sync_in_progress = False
            self._old_requests_complete.clear()

            # Allow new requests to proceed
            self._new_requests_blocked.set()

            self.log_info("Model synchronization completed, new requests can proceed")
