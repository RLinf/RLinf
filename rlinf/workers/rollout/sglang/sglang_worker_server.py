# Copyright 2026 The RLinf Authors.
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
import time
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from omegaconf import DictConfig
from starlette.responses import Response

from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

try:
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.managers.template_manager import TemplateManager
except ImportError:
    from sglang.srt.openai_api.adapter import (
        v1_chat_generate_request,
        v1_chat_generate_response,
    )
    from sglang.srt.openai_api.protocol import ChatCompletionRequest

    OpenAIServingChat = None  # type: ignore[misc, assignment]
    TemplateManager = None  # type: ignore[misc, assignment]

_LEGACY_OPENAI = OpenAIServingChat is None


def _patch_chat_body_assistant_content(body: object) -> None:
    """LiteLLM/OpenAI clients may omit `content` when null; older SGLang (e.g. 0.4.x)
    ChatCompletionMessageGenericParam requires the key (value may be null). 0.5.2+
    uses Field(default=None); this patch is harmless there."""
    msgs = body.get("messages")
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant" and "content" not in m:
            m["content"] = None


class SGLangWorkerWithHTTPServer(SGLangWorker):
    def __init__(
        self,
        config: DictConfig,
        placement: ModelParallelComponentPlacement,
        weight_reload: Literal["sync", "cpu", None] = "sync",
        config_rollout: Optional[DictConfig] = None,
        http_server_host: str = "0.0.0.0",
        http_server_port: int = 8020,
    ):
        super().__init__(config, placement, weight_reload, config_rollout)

        server_cfg = (self._cfg.rollout.get("sglang") or {}).get("server") or {}
        self._http_server_host = http_server_host or str(
            server_cfg.get("host", "0.0.0.0")
        )
        self._http_server_port = (
            int(server_cfg.get("port", http_server_port)) + self._rank
        )
        self._http_server = None
        self._http_server_task = None
        self._http_app = None
        self._openai_serving_chat = None

        self._setup_http_routes()

    def _setup_http_routes(self):
        app = FastAPI(title="SGLangWorker-HTTP", version="1.0.0")

        @app.post("/v1/chat/completions")
        async def handle_chat_completion(raw_request: Request):
            body = await raw_request.json()
            _patch_chat_body_assistant_content(body)
            request = ChatCompletionRequest.model_validate(body)
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

    def _init_openai_serving(self) -> None:
        if _LEGACY_OPENAI:
            return

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
                request.top_logprobs = 1

            if request.temperature is None and "temperature" in self._sampling_params:
                request.temperature = self._sampling_params["temperature"]
            if request.max_tokens is None and "max_new_tokens" in self._sampling_params:
                request.max_tokens = self._sampling_params["max_new_tokens"]
            if request.top_p is None and "top_p" in self._sampling_params:
                request.top_p = self._sampling_params["top_p"]
            if request.top_k is None and "top_k" in self._sampling_params:
                request.top_k = self._sampling_params["top_k"]

            tokenizer_manager = self._engine.tokenizer_manager
            if _LEGACY_OPENAI:
                adapted_request, _ = v1_chat_generate_request(
                    [request],
                    tokenizer_manager,
                    request_ids=[getattr(request, "rid", None)],
                )
            else:
                adapted_request, _ = (
                    self._openai_serving_chat._convert_to_internal_request(request)
                )

            adapted_request.return_logprob = self._return_logprobs
            prompt_token_ids = None
            if (
                hasattr(adapted_request, "input_ids")
                and adapted_request.input_ids is not None
            ):
                prompt_token_ids = adapted_request.input_ids
                if hasattr(prompt_token_ids, "tolist"):
                    prompt_token_ids = prompt_token_ids.tolist()

            generator = tokenizer_manager.generate_request(adapted_request)
            result = await generator.__anext__()

            if not isinstance(result, list):
                result = [result]

            created = int(time.time())
            if _LEGACY_OPENAI:
                response = v1_chat_generate_response(
                    request,
                    result,
                    created,
                    tool_call_parser=self._cfg_rollout.sglang.get(
                        "tool_call_parser", None
                    ),
                )
            else:
                response = self._openai_serving_chat._build_chat_response(
                    request, result, created
                )

            # Align SGLang 0.4.x behavior with 0.5.2: tool-call parse failure should not
            # fail the entire request. Legacy adapter returns ORJSONResponse(400) for this
            # specific case; we downgrade it to a normal chat completion response.
            if isinstance(response, Response):
                body = getattr(response, "body", b"") or b""
                status_code = int(getattr(response, "status_code", 0) or 0)
                if (
                    status_code == 400
                    and b"Failed to parse fc related info to json format!" in body
                ):
                    # Reuse SGLang 0.4.x formatter to avoid subtle field mismatches:
                    # rerun response building with tool calls disabled.
                    tool_call_parser = self._cfg_rollout.sglang.get(
                        "tool_call_parser", None
                    )
                    try:
                        req_no_tools = request.model_copy(
                            update={"tool_choice": "none", "tools": None}
                        )
                    except Exception:
                        req_no_tools = request
                        setattr(req_no_tools, "tool_choice", "none")
                        setattr(req_no_tools, "tools", None)

                    response_chat_completion = v1_chat_generate_response(
                        req_no_tools,
                        result,
                        created,
                        tool_call_parser=tool_call_parser,
                    )
                    response = response_chat_completion
            response_dict = response.model_dump(exclude_none=True)

            if result and len(result) > 0 and "output_ids" in result[0]:
                response_dict["response_token_ids"] = [result[0]["output_ids"]]

            if prompt_token_ids is not None:
                response_dict["prompt_token_ids"] = prompt_token_ids

            return response_dict

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": type(e).__name__}},
            )

    def http_server_start(self):
        if self._http_server_task is not None:
            self.log_warning("HTTP server is already running")
            return

        self._http_server_task = asyncio.create_task(self._http_server.serve())
        self.log_info(
            f"HTTP server started on {self._http_server_host}:{self._http_server_port}"
        )

    async def http_server_stop(self):
        if self._http_server_task is None:
            return

        self._http_server.should_exit = True
        await self._http_server_task
        self._http_server_task = None
        self.log_info("HTTP server stopped")

    async def init_worker(self):
        await super().init_worker()

        self._init_openai_serving()
        self.http_server_start()

    def get_server_address(self) -> str:
        host = self._http_server_host
        if host == "0.0.0.0":
            import ray.util

            host = ray.util.get_node_ip_address()
        return f"{host}:{self._http_server_port}"
