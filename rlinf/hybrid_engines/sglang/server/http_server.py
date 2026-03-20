from __future__ import annotations

import dataclasses
import multiprocessing
from types import SimpleNamespace
from typing import Any, Callable, List, Optional

import torch
import yaml
from fastapi import Request
from fastapi.responses import ORJSONResponse
from sglang.srt.entrypoints import http_server as _http_server
from sglang.srt.entrypoints.openai import serving_chat as _serving_chat
from sglang.srt.server_args import ServerArgs
from sglang.version import __version__ as _sglang_version

from rlinf.hybrid_engines.sglang.common.io_struct import (
    SyncHFWeightInput,
    TaskMethodInput,
)

ORIG__launch_server = _http_server.launch_server
_patch_applied: bool = False
_smg_init_kwargs: dict[str, Any] | None = None
_rollout_return_logprobs: bool = False


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_make_json_safe(x) for x in obj)
    return obj


def set_smg_init_context(**kwargs) -> None:
    global _smg_init_kwargs
    _smg_init_kwargs = dict(kwargs)


def _apply_patch() -> None:
    global _patch_applied
    if _patch_applied:
        return
    _patch_applied = True
    app = _http_server.app
    create_error_response = _http_server._create_error_response  # type: ignore[attr-defined]

    # Force logprobs on chat requests when rollout_return_logprobs is True (e.g. autogen doesn't send; same as worker_server).
    _orig_chat_handle_request = _serving_chat.OpenAIServingChat.handle_request

    async def _handle_request_force_logprobs(
        self: Any, request: Any, raw_request: Any
    ) -> Any:
        if _rollout_return_logprobs:
            request.logprobs = True
            request.top_logprobs = getattr(request, "top_logprobs", None) or 1
        return await _orig_chat_handle_request(self, request, raw_request)

    _serving_chat.OpenAIServingChat.handle_request = _handle_request_force_logprobs

    # Patch chat response to include response_token_ids for AgentLightning (router_server mode).
    # Stock SGLang only returns standard OpenAI fields; without this, client gets no token_ids -> response_lengths all 0.
    _orig_build_chat_response = _serving_chat.OpenAIServingChat._build_chat_response

    def _build_chat_response_with_token_ids(
        self: Any,
        request: Any,
        ret: List[dict],
        created: int,
    ) -> Any:
        response = _orig_build_chat_response(self, request, ret, created)
        if not ret or not hasattr(response, "model_dump"):
            return response
        if "output_ids" not in ret[0]:
            return response
        out = response.model_dump(exclude_none=True)
        out["response_token_ids"] = [ret[0]["output_ids"]]
        return ORJSONResponse(content=out)

    _serving_chat.OpenAIServingChat._build_chat_response = _build_chat_response_with_token_ids

    @app.post("/sync_hf_weight")
    async def sync_hf_weight() -> Any:  # type: ignore[override]
        try:
            global_state = _http_server._global_state  # type: ignore[attr-defined]
            if global_state is None:
                raise RuntimeError("s_http._global_state is not initialized yet")
            await global_state.tokenizer_manager.sync_hf_weight(SyncHFWeightInput())  # type: ignore[attr-defined]
            return {"status": "ok"}
        except Exception as e:  # noqa: BLE001
            return create_error_response(e)

    @app.post("/release_memory_occupation")
    async def release_memory_occupation() -> Any:  # type: ignore[override]
        try:
            global_state = _http_server._global_state  # type: ignore[attr-defined]
            if global_state is None:
                raise RuntimeError("s_http._global_state is not initialized yet")
            obj = TaskMethodInput(
                method_name="release_memory_occupation",
                args=(ReleaseMemoryOccupationReqInput(),),
                kwargs={},
            )
            await global_state.tokenizer_manager.run_task_method(obj)  # type: ignore[attr-defined]
            return {"status": "ok"}
        except Exception as e:  # noqa: BLE001
            return create_error_response(e)

    @app.post("/resume_memory_occupation")
    async def resume_memory_occupation() -> Any:  # type: ignore[override]
        try:
            global_state = _http_server._global_state  # type: ignore[attr-defined]
            if global_state is None:
                raise RuntimeError("s_http._global_state is not initialized yet")
            obj = TaskMethodInput(
                method_name="resume_memory_occupation",
                args=(ResumeMemoryOccupationReqInput(),),
                kwargs={},
            )
            await global_state.tokenizer_manager.run_task_method(obj)  # type: ignore[attr-defined]
            return {"status": "ok"}
        except Exception as e:  # noqa: BLE001
            return create_error_response(e)

    @app.post("/run_task_method")
    async def run_task_method(request: Request) -> Any:  # type: ignore[override]
        try:
            global_state = _http_server._global_state  # type: ignore[attr-defined]
            if global_state is None:
                raise RuntimeError("s_http._global_state is not initialized yet")
            body = await request.json()
            method_name = body["method_name"]
            args = body.get("args", [])
            kwargs = body.get("kwargs", _smg_init_kwargs if not args else {})
            obj = TaskMethodInput(method_name=method_name, args=args, kwargs=kwargs)
            res = await global_state.tokenizer_manager.run_task_method(obj)  # type: ignore[attr-defined]
            return {"status": "ok", "result": res}
        except Exception as e:  # noqa: BLE001
            return create_error_response(e)

    @app.post("/init_rlinf_worker")
    async def init_rlinf_worker(request: Request) -> Any:  # type: ignore[override]
        try:
            global_state = _http_server._global_state  # type: ignore[attr-defined]
            if global_state is None:
                raise RuntimeError("s_http._global_state is not initialized yet")
            body = await request.json()
            from omegaconf import OmegaConf
            from rlinf.scheduler import WorkerAddress
            from rlinf.utils.placement import PlacementMode

            parent_address = WorkerAddress.from_name(body["parent_worker_name"])
            weight_reload = body.get("weight_reload", "sync")

            if weight_reload == "sync":
                placement_dict = body.get("placement", {})
                pm = placement_dict.get("placement_mode", 1)
                placement = SimpleNamespace(
                    rollout_dp_size=int(placement_dict.get("rollout_dp_size", 1)),
                    actor_tp_size=int(placement_dict.get("actor_tp_size", 1)),
                    actor_pp_size=int(placement_dict.get("actor_pp_size", 1)),
                    actor_world_size=int(placement_dict.get("actor_world_size", 1)),
                    rollout_tp_size=int(placement_dict.get("rollout_tp_size", 1)),
                    rollout_world_size=int(placement_dict.get("rollout_world_size", 1)),
                    placement_mode=PlacementMode(pm) if isinstance(pm, int) else pm,
                )
                cfg_yaml = body.get("cfg_yaml")
                if not cfg_yaml:
                    raise ValueError(
                        "init_rlinf_worker requires non-empty cfg_yaml (full resolved config)."
                    )
                cfg = OmegaConf.create(yaml.safe_load(cfg_yaml))
                args = (parent_address, weight_reload, placement, cfg)
            else:
                # "cpu" 或 None：与 SGLangWorker 一致，只传 2 个参数
                args = (parent_address, weight_reload)

            obj = TaskMethodInput(
                method_name="init_rlinf_worker",
                args=args,
                kwargs={},
            )
            await global_state.tokenizer_manager.run_task_method(obj)  # type: ignore[attr-defined]
            return {"status": "ok"}
        except Exception as e:  # noqa: BLE001
            return create_error_response(e)

    @app.get("/server_info")
    async def server_info() -> Any:  # type: ignore[override]
        global_state = _http_server._global_state  # type: ignore[attr-defined]
        if global_state is None:
            raise RuntimeError("s_http._global_state is not initialized yet")
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
):
    global _rollout_return_logprobs
    _rollout_return_logprobs = rollout_return_logprobs
    _apply_patch()
    return ORIG__launch_server(
        server_args=server_args,
        pipe_finish_writer=pipe_finish_writer,
        launch_callback=launch_callback,
    )


__all__ = ["launch_server", "set_smg_init_context"]
