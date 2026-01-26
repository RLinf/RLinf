
import asyncio
import copy
import dataclasses
import time
import uuid
from typing import Any, List, Literal, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from omegaconf import DictConfig
from pydantic import BaseModel
from sglang.srt.server_args import ServerArgs

from rlinf.config import torch_dtype_from_precision
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.sglang import Engine
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
        super().__init__(config, placement, weight_reload, config_rollout)

        self._enable_http_server = enable_http_server
        self._http_server_host = http_server_host or self._cfg.get("server", {}).get("sglang_http", {}).get("host", "0.0.0.0")
        base_port = self._cfg.get("server", {}).get("sglang_http", {}).get("port", http_server_port)
        self._http_server_port = base_port + self._rank
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
                return_logprob=self._return_logprobs,
            )
            
            result = engine_results[0] if isinstance(engine_results, list) and len(engine_results) > 0 else engine_results

            response_text = result.get("text", "")
            response_ids = result.get("output_ids", [])
            meta_info = result.get("meta_info", {})
            finish_reason_info = meta_info.get("finish_reason", {})
            finish_reason = finish_reason_info.get("type", "stop")

            logprobs = None
            if self._return_logprobs:
                logprobs = [item[0] for item in meta_info["output_token_logprobs"]]

            choice_logprobs = None
            if logprobs is not None:
                choice_logprobs = {
                    "content": [
                        {
                            "token": "",
                            "logprob": logprob_value,
                            "top_logprobs": []
                        }
                        for logprob_value in logprobs
                    ]
                }

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
                        "logprobs": choice_logprobs,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": len(response_ids),
                    "total_tokens": len(prompt_token_ids) + len(response_ids),
                },
                "prompt_token_ids": prompt_token_ids,  
                "response_token_ids": [response_ids],  
            }

            return response

    def _init_engine(self):
        """Override parent method to add tool_call_parser support."""
        use_cudagraph = not self._cfg_rollout.enforce_eager

        load_format = "dummy"
        if self.weight_reload == "sync":
            if self._cfg_rollout.validate_weight or getattr(
                self._cfg_rollout, "validate_weight_first_sync", False
            ):
                load_format = "auto"
        else:
            load_format = "auto"

            

        server_args = ServerArgs(
            model_path=self._cfg_rollout.model.model_path,
            disable_cuda_graph=not use_cudagraph,
            cuda_graph_max_bs=min(
                self._cfg_rollout.cuda_graph_max_bs,
                self._cfg_rollout.max_running_requests,
            ),
            tp_size=self._cfg_rollout.tensor_parallel_size,
            mem_fraction_static=self._cfg_rollout.gpu_memory_utilization,
            enable_memory_saver=use_cudagraph,
            enable_torch_compile=self._cfg_rollout.sglang.use_torch_compile,
            torch_compile_max_bs=min(
                self._cfg_rollout.sglang.torch_compile_max_bs,
                self._cfg_rollout.max_running_requests,
            ),
            load_format=load_format,
            dtype=torch_dtype_from_precision(self._cfg_rollout.model.precision),
            skip_tokenizer_init=not self._cfg_rollout.detokenize,
            decode_log_interval=self._cfg_rollout.sglang.decode_log_interval,
            attention_backend=self._cfg_rollout.sglang.attention_backend,
            log_level="warning",
            max_running_requests=self._cfg_rollout.max_running_requests,
            dist_init_addr=f"127.0.0.1:{str(self.acquire_free_port())}",
            tool_call_parser = self._cfg_rollout.sglang.get("tool_call_parser", None),
        )

        self.log_on_first_rank(f"{server_args=}")
        self._engine = Engine(
            **dataclasses.asdict(server_args),
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

        if self._enable_http_server and start_http_server:
            self.http_server_start()

    def shutdown(self):

        if self._enable_http_server and self._http_server_task is not None:
            self._http_server.should_exit = True
            # Note: This is synchronous shutdown, async cleanup should be done elsewhere
            # The task will be cleaned up when the event loop stops

        # Call parent shutdown
        super().shutdown()

    def get_server_address(self) -> str:
        """Get the HTTP server address for this worker."""
        if not self._enable_http_server:
            return None
        return f"{self._http_server_host}:{self._http_server_port}"

