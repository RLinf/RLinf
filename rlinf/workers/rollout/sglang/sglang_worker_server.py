
import asyncio
import dataclasses
import time
from typing import List, Literal, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from omegaconf import DictConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.template_manager import TemplateManager

from rlinf.config import torch_dtype_from_precision
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.sglang import Engine
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker


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
        self._openai_serving_chat = None

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

    def _init_openai_serving(self):
        tokenizer_manager = self._engine.tokenizer_manager
        template_manager = TemplateManager()
        template_manager.initialize_templates(
            tokenizer_manager=tokenizer_manager,
            model_path=self._cfg_rollout.model.model_path,
        )
        self._openai_serving_chat = OpenAIServingChat(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
        )

    async def _handle_chat_completion(self, request: ChatCompletionRequest):
        try:

            if self._return_logprobs:
                request.logprobs = True

            if request.temperature is None and "temperature" in self._sampling_params:
                request.temperature = self._sampling_params["temperature"]
            if request.max_tokens is None and "max_new_tokens" in self._sampling_params:
                request.max_tokens = self._sampling_params["max_new_tokens"]
            if request.top_p is None and "top_p" in self._sampling_params:
                request.top_p = self._sampling_params["top_p"]
            if request.top_k is None and "top_k" in self._sampling_params:
                request.top_k = self._sampling_params["top_k"]
            
           
            adapted_request, _ = self._openai_serving_chat._convert_to_internal_request(
                request
            )
            adapted_request.return_logprob = self._return_logprobs 
            prompt_token_ids = None
            if hasattr(adapted_request, "input_ids") and adapted_request.input_ids is not None:
                prompt_token_ids = adapted_request.input_ids
                if hasattr(prompt_token_ids, 'tolist'):
                    prompt_token_ids = prompt_token_ids.tolist()
            
            generator = self._openai_serving_chat.tokenizer_manager.generate_request(
                adapted_request
            )
            result = await generator.__anext__()
            
            if not isinstance(result, list):
                result = [result]


            response = self._openai_serving_chat._build_chat_response(
                request,
                result,
                int(time.time()),
            )
            
            response_dict = response.model_dump(exclude_none=True)     


            if result and len(result) > 0 and "output_ids" in result[0]:
                response_dict["response_token_ids"] = [result[0]["output_ids"]]

            if prompt_token_ids is not None:
                response_dict["prompt_token_ids"] = prompt_token_ids
            
            return response_dict

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(e),
                        "type": type(e).__name__
                    }
                }
            )

    def _init_engine(self):
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

        self._http_server_task = asyncio.create_task(self._http_server.serve())
        self.log_info(f"HTTP server started on {self._http_server_host}:{self._http_server_port}")

    async def http_server_stop(self):
        if not self._enable_http_server or self._http_server_task is None:
            return

        self._http_server.should_exit = True
        await self._http_server_task
        self._http_server_task = None
        self.log_info("HTTP server stopped")

    async def init_worker(self, start_http_server: bool = False):
        await super().init_worker()
        
        if self._enable_http_server:
            self._init_openai_serving()
            if start_http_server:
                self.http_server_start()

    def get_server_address(self) -> str:
        if not self._enable_http_server:
            return None
        host = self._http_server_host
        if host == "0.0.0.0":
            import ray.util

            host = ray.util.get_node_ip_address()
        return f"{host}:{self._http_server_port}"


