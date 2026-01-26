
import asyncio
import copy
import time
import uuid
from typing import Any, List, Literal, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from omegaconf import DictConfig
from pydantic import BaseModel

from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker


class ChatMessage(BaseModel):

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False


class SGLangWorkerWithHTTPServer(SGLangWorker):

    def __init__(
        self,
        config: DictConfig,
        placement: ModelParallelComponentPlacement,
        weight_reload: Literal["sync", "cpu", None] = "sync",
        config_rollout: Optional[DictConfig] = None,
        enable_http_server: bool = True,
        http_server_host: str = "0.0.0.0",
        http_server_port: int = 8020,
    ):
        # Call parent __init__ first
        super().__init__(config, placement, weight_reload, config_rollout)

        # HTTP server configuration
        self._enable_http_server = enable_http_server
        self._http_server_host = http_server_host
        self._http_server_port = http_server_port
        self._http_server = None
        self._http_server_task = None
        self._http_app = None

        if self._enable_http_server:
            self._setup_http_routes()

    def _setup_http_routes(self):
        app = FastAPI(title="SGLangWorker-HTTP", version="1.0.0")

        @app.post("/v1/chat/completions")
        async def handle_chat_completion(request: ChatCompletionRequest):
            return await self._handle_chat_completion(request)

        @app.get("/health")
        async def handle_health():
            return {"status": "healthy", "model": "sglang-model"}

        @app.get("/")
        async def handle_root():
            return {
                "service": "SGLang HTTP Server",
                "model": "sglang-model",
                "endpoints": ["/v1/chat/completions", "/health"],
            }

        self._http_app = app
        self._http_server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=self._http_server_host,
                port=self._http_server_port,
                log_level="warning",
                access_log=False,
            )
        )

    async def _handle_chat_completion(self, request: ChatCompletionRequest):

        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            prompt_text = (
                "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                + "\nassistant:"
            )

            prompt_token_ids = self._tokenizer(prompt_text).input_ids
            if hasattr(prompt_token_ids, "tolist"):
                prompt_token_ids = prompt_token_ids.tolist() 
                
            sampling_params = copy.deepcopy(self._sampling_params)
            if request.stop is not None:
                sampling_params["stop"] = request.stop
            if request.temperature is not None:
                sampling_params["temperature"] = request.temperature
            if request.max_tokens is not None:
                sampling_params["max_new_tokens"] = int(request.max_tokens)
            if request.top_p is not None:
                sampling_params["top_p"] = request.top_p
            if request.top_k is not None:
                sampling_params["top_k"] = request.top_k

            engine_results = await self._engine.async_generate(
                prompt=prompt_text,
                sampling_params=sampling_params,
                return_logprob=False,
            )
            
            result = engine_results[0] if isinstance(engine_results, list) and len(engine_results) > 0 else engine_results

            response_text = result.get("text", "")
            response_ids = result.get("output_ids", [])
            meta_info = result.get("meta_info", {})
            finish_reason_info = meta_info.get("finish_reason", {})
            finish_reason = finish_reason_info.get("type", "stop")

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
                "prompt_token_ids": prompt_token_ids_single,  # List[int]
                "response_token_ids": [response_ids],  # List[List[int]], AgentLightning will take [0]
            }

            return response

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            self.log_error(f"Error in chat completion (request_id={request_id}): {error_detail}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"Generation failed: {str(e)}",
                        "type": type(e).__name__,
                        "detail": error_detail,
                    }
                },
            )

    def http_server_start(self):

        if not self._enable_http_server:
            return

        if self._http_server_task is not None:
            self.log_warning("HTTP server is already running")
            return

        # Start server in background task
        self._http_server_task = asyncio.create_task(self._http_server.serve())
        self.log_info(f"HTTP server started on {self._http_server_host}:{self._http_server_port}")

    async def http_server_stop(self):
"
        if not self._enable_http_server or self._http_server_task is None:
            return

        self._http_server.should_exit = True
        await self._http_server_task
        self._http_server_task = None
        self.log_info("HTTP server stopped")

    async def init_worker(self, start_http_server: bool = False):

        await super().init_worker()

        if self._enable_http_server and start_http_server and self._rank == 0:
            self.http_server_start()

    def shutdown(self):

        if self._enable_http_server and self._http_server_task is not None:
            self._http_server.should_exit = True
            # Note: This is synchronous shutdown, async cleanup should be done elsewhere
            # The task will be cleaned up when the event loop stops

        # Call parent shutdown
        super().shutdown()

