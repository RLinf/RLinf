from __future__ import annotations

import dataclasses
import json
import multiprocessing
from types import SimpleNamespace
from typing import Any, Callable, List, Optional

import torch
import yaml
from fastapi import Request
from fastapi.responses import ORJSONResponse
from sglang.srt.entrypoints import http_server as _http_server

try:
    from sglang.srt.entrypoints.openai import serving_chat as _serving_chat
except ImportError:
    _serving_chat = None
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.server_args import ServerArgs
from sglang.version import __version__ as _sglang_version

from rlinf.hybrid_engines.sglang.common.io_struct import (
    SyncHFWeightInput,
    TaskMethodInput,
)
from rlinf.hybrid_engines.sglang.server.openai_compat import (
    downgrade_legacy_tool_call_parse_error,
    patch_chat_body_assistant_content,
)

ORIG__launch_server = _http_server.launch_server
_patch_applied: bool = False
_smg_init_kwargs: dict[str, Any] | None = None
_rollout_return_logprobs: bool = False
_openai_response_tokens: bool = False
_legacy_adapter: Any | None = None

if _serving_chat is None:
    try:
        from sglang.srt.openai_api import adapter as _legacy_adapter
    except ImportError:
        _legacy_adapter = None


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.dtype):
        return str(obj)
    return obj


def set_smg_init_context(**kwargs) -> None:
    global _smg_init_kwargs
    _smg_init_kwargs = dict(kwargs)


def _patch_openai_serving_chat() -> None:
    _orig_chat_handle_request = _serving_chat.OpenAIServingChat.handle_request

    async def _handle_request_force_logprobs(
        self: Any, request: Any, raw_request: Any
    ) -> Any:
        if _rollout_return_logprobs and _openai_response_tokens:
            request.logprobs = True
            request.top_logprobs = getattr(request, "top_logprobs", None) or 1
        return await _orig_chat_handle_request(self, request, raw_request)

    _serving_chat.OpenAIServingChat.handle_request = _handle_request_force_logprobs
    _orig_build_chat_response = _serving_chat.OpenAIServingChat._build_chat_response

    def _build_chat_response_with_token_ids(
        self: Any,
        request: Any,
        ret: List[dict],
        created: int,
    ) -> Any:
        response = _orig_build_chat_response(self, request, ret, created)
        if not _openai_response_tokens:
            return response
        if not ret or not hasattr(response, "model_dump"):
            return response
        if "output_ids" not in ret[0]:
            return response
        out = response.model_dump(exclude_none=True)
        out["response_token_ids"] = [ret[0]["output_ids"]]
        return ORJSONResponse(content=out)

    _serving_chat.OpenAIServingChat._build_chat_response = (
        _build_chat_response_with_token_ids
    )

    _orig_handle_non_streaming = (
        _serving_chat.OpenAIServingChat._handle_non_streaming_request
    )

    async def _handle_non_streaming_request_with_prompt_token_ids(
        self: Any,
        adapted_request: Any,
        request: Any,
        raw_request: Any,
    ) -> Any:
        if not _openai_response_tokens:
            return await _orig_handle_non_streaming(
                self, adapted_request, request, raw_request
            )
        prompt_token_ids = None
        if (
            hasattr(adapted_request, "input_ids")
            and adapted_request.input_ids is not None
        ):
            pt = adapted_request.input_ids
            prompt_token_ids = pt.tolist() if hasattr(pt, "tolist") else list(pt)
        response = await _orig_handle_non_streaming(
            self, adapted_request, request, raw_request
        )
        if prompt_token_ids is None:
            return response
        if isinstance(response, ORJSONResponse):
            try:
                payload = json.loads(response.body.decode("utf-8"))
            except (
                AttributeError,
                TypeError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ):
                return response
            payload["prompt_token_ids"] = prompt_token_ids
            return ORJSONResponse(content=payload)
        if hasattr(response, "model_dump"):
            try:
                out = response.model_dump(exclude_none=True)
                out["prompt_token_ids"] = prompt_token_ids
                return ORJSONResponse(content=out)
            except Exception:
                return response
        return response

    _serving_chat.OpenAIServingChat._handle_non_streaming_request = (
        _handle_non_streaming_request_with_prompt_token_ids
    )


def _patch_legacy_openai_adapter() -> None:
    _orig_v1_chat_completions = getattr(_legacy_adapter, "v1_chat_completions", None)
    _orig_v1_chat_generate_request = _legacy_adapter.v1_chat_generate_request
    _orig_v1_chat_generate_response = _legacy_adapter.v1_chat_generate_response

    if _orig_v1_chat_completions is not None:

        async def _v1_chat_completions_with_assistant_content(*args, **kwargs) -> Any:
            raw_request = args[1] if len(args) > 1 else kwargs.get("raw_request")
            if raw_request is not None and hasattr(raw_request, "json"):
                try:
                    patch_chat_body_assistant_content(await raw_request.json())
                except Exception:
                    pass
            return await _orig_v1_chat_completions(*args, **kwargs)

        _legacy_adapter.v1_chat_completions = (
            _v1_chat_completions_with_assistant_content
        )
        if hasattr(_http_server, "v1_chat_completions"):
            _http_server.v1_chat_completions = (
                _v1_chat_completions_with_assistant_content
            )

    def _v1_chat_generate_request_with_logprobs(*args, **kwargs) -> Any:
        requests = args[0] if args else kwargs.get("requests")
        if _rollout_return_logprobs and _openai_response_tokens and requests:
            for request in requests:
                request.logprobs = True
                request.top_logprobs = getattr(request, "top_logprobs", None) or 1

        adapted_request, *rest = _orig_v1_chat_generate_request(*args, **kwargs)
        if not _openai_response_tokens:
            return (adapted_request, *rest)
        prompt_token_ids = getattr(adapted_request, "input_ids", None)
        if prompt_token_ids is not None and requests:
            if hasattr(prompt_token_ids, "tolist"):
                prompt_token_ids = prompt_token_ids.tolist()
            for request in requests:
                setattr(request, "_rlinf_prompt_token_ids", prompt_token_ids)
        return (adapted_request, *rest)

    def _v1_chat_generate_response_with_token_ids(*args, **kwargs) -> Any:
        response = _orig_v1_chat_generate_response(*args, **kwargs)
        request = args[0] if args else kwargs.get("request")
        ret = args[1] if len(args) > 1 else kwargs.get("ret")
        created = args[2] if len(args) > 2 else kwargs.get("created")
        tool_call_parser = kwargs.get("tool_call_parser", None)
        if isinstance(response, ORJSONResponse):
            response = downgrade_legacy_tool_call_parse_error(
                response,
                request=request,
                ret=ret,
                created=created,
                response_builder=_orig_v1_chat_generate_response,
                tool_call_parser=tool_call_parser,
            )
        if not _openai_response_tokens:
            return response
        prompt_token_ids = getattr(request, "_rlinf_prompt_token_ids", None)
        if not ret or ("output_ids" not in ret[0] and prompt_token_ids is None):
            return response
        if isinstance(response, ORJSONResponse):
            try:
                payload = json.loads(response.body.decode("utf-8"))
            except (
                AttributeError,
                TypeError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ):
                return response
            if "output_ids" in ret[0]:
                payload["response_token_ids"] = [ret[0]["output_ids"]]
            if prompt_token_ids is not None:
                payload["prompt_token_ids"] = prompt_token_ids
            return ORJSONResponse(content=payload)
        if hasattr(response, "model_dump"):
            out = response.model_dump(exclude_none=True)
            if "output_ids" in ret[0]:
                out["response_token_ids"] = [ret[0]["output_ids"]]
            if prompt_token_ids is not None:
                out["prompt_token_ids"] = prompt_token_ids
            return ORJSONResponse(content=out)
        return response

    _legacy_adapter.v1_chat_generate_request = _v1_chat_generate_request_with_logprobs
    _legacy_adapter.v1_chat_generate_response = (
        _v1_chat_generate_response_with_token_ids
    )
    if hasattr(_http_server, "v1_chat_generate_request"):
        _http_server.v1_chat_generate_request = _v1_chat_generate_request_with_logprobs
    if hasattr(_http_server, "v1_chat_generate_response"):
        _http_server.v1_chat_generate_response = (
            _v1_chat_generate_response_with_token_ids
        )


def _apply_patch() -> None:
    global _patch_applied
    if _patch_applied:
        return
    _patch_applied = True
    app = _http_server.app

    if _serving_chat is not None:
        _patch_openai_serving_chat()
    elif _legacy_adapter is not None:
        _patch_legacy_openai_adapter()

    @app.post("/sync_hf_weight")
    async def sync_hf_weight() -> Any:
        global_state = _http_server._global_state
        await global_state.tokenizer_manager.sync_hf_weight(SyncHFWeightInput())
        return {"status": "ok"}

    @app.post("/release_memory_occupation")
    async def release_memory_occupation() -> Any:
        global_state = _http_server._global_state
        obj = TaskMethodInput(
            method_name="release_memory_occupation",
            args=(ReleaseMemoryOccupationReqInput(),),
            kwargs={},
        )
        await global_state.tokenizer_manager.run_task_method(obj)
        return {"status": "ok"}

    @app.post("/resume_memory_occupation")
    async def resume_memory_occupation() -> Any:
        global_state = _http_server._global_state
        obj = TaskMethodInput(
            method_name="resume_memory_occupation",
            args=(ResumeMemoryOccupationReqInput(),),
            kwargs={},
        )
        await global_state.tokenizer_manager.run_task_method(obj)
        return {"status": "ok"}

    @app.post("/run_task_method")
    async def run_task_method(request: Request) -> Any:
        global_state = _http_server._global_state
        body = await request.json()
        method_name = body["method_name"]
        args = body.get("args", [])
        kwargs = body.get("kwargs", _smg_init_kwargs if not args else {})
        obj = TaskMethodInput(method_name=method_name, args=args, kwargs=kwargs)
        res = await global_state.tokenizer_manager.run_task_method(obj)
        return {"status": "ok", "result": res}

    @app.post("/init_rlinf_worker")
    async def init_rlinf_worker(request: Request) -> Any:
        global_state = _http_server._global_state
        body = await request.json()
        from omegaconf import OmegaConf

        from rlinf.scheduler import WorkerAddress
        from rlinf.utils.placement import (
            PlacementMode,
            RolloutSyncMode,
            placement_mode_to_rollout_sync_mode,
        )

        parent_address = WorkerAddress.from_name(body["parent_worker_name"])
        weight_reload = body.get("weight_reload", "sync")

        if weight_reload == "sync":
            placement_dict = body.get("placement", {})
            pm = placement_dict.get("placement_mode", 1)
            placement_mode = PlacementMode(pm) if isinstance(pm, int) else pm
            rollout_sync_mode = placement_dict.get("_rollout_sync_mode")
            if rollout_sync_mode is None:
                rollout_sync_mode = placement_mode_to_rollout_sync_mode(placement_mode)
            elif isinstance(rollout_sync_mode, int):
                rollout_sync_mode = RolloutSyncMode(rollout_sync_mode)
            placement = SimpleNamespace(
                rollout_dp_size=int(placement_dict.get("rollout_dp_size", 1)),
                actor_tp_size=int(placement_dict.get("actor_tp_size", 1)),
                actor_pp_size=int(placement_dict.get("actor_pp_size", 1)),
                actor_world_size=int(placement_dict.get("actor_world_size", 1)),
                rollout_tp_size=int(placement_dict.get("rollout_tp_size", 1)),
                rollout_world_size=int(placement_dict.get("rollout_world_size", 1)),
                placement_mode=placement_mode,
                _rollout_sync_mode=rollout_sync_mode,
            )
            cfg_yaml = body.get("cfg_yaml")
            if not cfg_yaml:
                raise ValueError(
                    "init_rlinf_worker requires non-empty cfg_yaml (full resolved config)."
                )
            cfg = OmegaConf.create(yaml.safe_load(cfg_yaml))
            args = (parent_address, weight_reload, placement, cfg)
        else:
            args = (parent_address, weight_reload)

        obj = TaskMethodInput(
            method_name="init_rlinf_worker",
            args=args,
            kwargs={},
        )
        await global_state.tokenizer_manager.run_task_method(obj)
        return {"status": "ok"}

    @app.get("/server_info")
    async def server_info() -> Any:
        global_state = _http_server._global_state
        internal_states = await global_state.tokenizer_manager.get_internal_state()
        payload = {
            **dataclasses.asdict(global_state.tokenizer_manager.server_args),
            **global_state.scheduler_info,
            "internal_states": internal_states,
            "version": _sglang_version,
        }
        return _make_json_safe(payload)


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
    rollout_return_logprobs: bool = False,
    openai_response_tokens: bool = False,
):
    global _rollout_return_logprobs, _openai_response_tokens
    _rollout_return_logprobs = rollout_return_logprobs
    _openai_response_tokens = openai_response_tokens
    _apply_patch()
    return ORIG__launch_server(
        server_args=server_args,
        pipe_finish_writer=pipe_finish_writer,
        launch_callback=launch_callback,
    )


__all__ = ["launch_server", "set_smg_init_context"]
