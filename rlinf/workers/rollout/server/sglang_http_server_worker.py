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
import random
import time
import uuid
from typing import Any, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from omegaconf.dictconfig import DictConfig
from pydantic import BaseModel
from rlinf.scheduler import Worker
from rlinf.scheduler.worker.worker_group import WorkerGroup
from rlinf.utils.placement import ComponentPlacement
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker


class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False


class SGLangHTTPServerWorker(Worker):
    """HTTP server worker for SGLangWorker to provide OpenAI-compatible chat completions API."""

    def __init__(self, cfg: DictConfig, placement: ComponentPlacement):
        Worker.__init__(self)

        self._cfg = cfg

        # Configuration - EXACTLY like online_router_worker.py line 67-68
        self._server_host = cfg.server.sglang_http.get("host", "0.0.0.0")
        self._server_port = cfg.server.sglang_http.get("port", 8020)
        self._rollout_instance_num = placement.rollout_dp_size
        self._sampling_params = SGLangWorker.get_sampling_param_from_config(
            self._cfg.algorithm.sampling_params
        )
        if "stop" in self._cfg.algorithm.sampling_params:
            self._sampling_params["stop"] = self._cfg.algorithm.sampling_params["stop"]

        # Initialize tokenizer for converting prompt to prompt_token_ids
        # AgentLightning requires prompt_token_ids in the response
        from transformers import AutoTokenizer
        model_path = self._cfg.rollout.model.model_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Setup FastAPI routes
        self._setup_routes()
        self._server_task = None

    def _setup_routes(self):
        """Setup FastAPI routes."""
        app = FastAPI(title="SGLangHTTPServerWorker", version="1.0.0")
        app.add_api_route(
            "/v1/chat/completions", self._handle_chat_completion, methods=["POST"]
        )
        app.add_api_route("/health", self._handle_health, methods=["GET"])
        app.add_api_route("/", self._handle_root, methods=["GET"])

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

    async def _handle_chat_completion(self, request: ChatCompletionRequest):
        """Handle chat completions requests."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Check if rollout_worker is initialized
        if not hasattr(self, 'rollout_worker') or self.rollout_worker is None:
            self.log_error("Rollout worker not initialized")
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Rollout worker not initialized. Please ensure init_worker() has been called.",
                        "type": "ServiceUnavailable"
                    }
                }
            )

        try:
            # 1. Convert OpenAI messages to prompt string
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            prompt_text = (
                "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                + "\nassistant:"
            )

            # 2. Tokenize prompt to get prompt_token_ids (for AgentLightning)
            # Format: list of lists (batch format) - like _validate_weight_at_first
            tokenizer_output = self._tokenizer([prompt_text])
            prompt_token_ids = tokenizer_output.input_ids
            # Ensure it's a list of lists (batch format)
            if hasattr(prompt_token_ids, 'tolist'):
                prompt_token_ids = prompt_token_ids.tolist()
            elif isinstance(prompt_token_ids, list) and len(prompt_token_ids) > 0 and isinstance(prompt_token_ids[0], int):
                # Single list, wrap it
                prompt_token_ids = [prompt_token_ids]
            # Extract single prompt_token_ids (we're processing one request)
            prompt_token_ids_single = prompt_token_ids[0]

            # 3. Forward request to rollout worker
            sglang_instance_id = random.randint(0, self._rollout_instance_num - 1)
            
            # Build sampling_params
            sampling_params = copy.deepcopy(self._sampling_params)
            if request.stop is not None:
                sampling_params["stop"] = request.stop
            
            # Handle other request-level overrides for chat completion API
            if request.temperature is not None:
                sampling_params["temperature"] = request.temperature
            if request.max_tokens is not None:
                sampling_params["max_new_tokens"] = request.max_tokens
            if request.top_p is not None:
                sampling_params["top_p"] = request.top_p
            if request.top_k is not None:
                sampling_params["top_k"] = request.top_k
            
            # Pass prompt string directly (EXACTLY like online_router_worker.py line 153)
            # SGLang engine will tokenize on GPU, avoiding CPU tensor issues
            generate_result = (
                await self.rollout_worker.execute_on(sglang_instance_id)
                .async_generate(prompt=prompt_text, sampling_params=sampling_params)
                .async_wait()
            )
            generate_result = generate_result[0][0]

            # 4. Extract result from SGLang
            # SGLang returns: {"text": str, "output_ids": list[int], "meta_info": {...}}
            response_text = generate_result.get("text", "")
            response_ids = generate_result.get("output_ids", [])
            meta_info = generate_result.get("meta_info", {})
            finish_reason_info = meta_info.get("finish_reason", {})
            finish_reason = finish_reason_info.get("type", "stop")

            # 5. Build OpenAI-format response with AgentLightning-compatible fields
            # AgentLightning expects:
            # - prompt_token_ids: List[int] (single list)
            # - response_token_ids: List[List[int]] (list of lists, will take [0])
            response = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(start_time),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids_single),
                    "completion_tokens": len(response_ids),
                    "total_tokens": len(prompt_token_ids_single) + len(response_ids),
                },
                # AgentLightning-compatible fields
                "prompt_token_ids": prompt_token_ids_single,  # List[int]
                "response_token_ids": [response_ids],  # List[List[int]], AgentLightning will take [0]
            }

            return response

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            # 添加详细的错误日志
            self.log_error(f"Error in chat completion (request_id={request_id}): {error_detail}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"Generation failed: {str(e)}",
                        "type": type(e).__name__,
                        "detail": error_detail  # 添加详细错误信息到响应中
                    }
                },
            )

    async def _handle_health(self):
        """Health check endpoint."""
        return {"status": "healthy", "model": "sglang-model"}

    async def _handle_root(self):
        """Root endpoint with service information."""
        return {
            "service": "SGLang HTTP Server",
            "model": "sglang-model",
            "endpoints": ["/v1/chat/completions", "/health"],
        }

    async def init_worker(self, rollout_worker: WorkerGroup[SGLangWorker]):
        """Initialize the worker."""
        self.rollout_worker = rollout_worker
