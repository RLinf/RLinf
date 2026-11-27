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

"""Private pickle-based control routes for SRT and SGLang Diffusion servers.

Authenticate before unpickling: all control routes can execute untrusted pickle.
Both SRT and SGLang Diffusion keep control routes on loopback and mount
``PeerWhitelistMiddleware`` when binding 0.0.0.0 for router/session access.
HTTP does not encrypt tokens or payloads; trust the server's pickled responses.
"""

import base64
import pickle
import secrets
from typing import Any

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel

from rlinf.utils.logging import get_logger


class RlinfPayload(BaseModel):
    """Body of every /rlinf/* route: a pickled request object, b64-wrapped."""

    payload_b64: str


def _decode(payload: RlinfPayload) -> Any:
    return pickle.loads(base64.b64decode(payload.payload_b64))


def _encode(result: Any) -> dict:
    return {"payload_b64": base64.b64encode(pickle.dumps(result)).decode()}


def _manager(request: Request):
    manager = getattr(request.app.state, "rlinf_manager", None)
    if manager is not None:
        return manager
    from sglang.srt.entrypoints.http_server import get_global_state

    return get_global_state().tokenizer_manager


async def rlinf_run_task_method(payload: RlinfPayload, request: Request) -> dict:
    """Run a scheduler method through the runtime's control manager."""
    result = await _manager(request).run_task_method(_decode(payload), request)
    return _encode(result)


async def rlinf_sync_hf_weight(payload: RlinfPayload, request: Request) -> dict:
    """Trigger an actor-to-scheduler weight sync over RLinf communication."""
    result = await _manager(request).sync_hf_weight(_decode(payload), request)
    return {"status": "ok", **(result if isinstance(result, dict) else {})}


async def rlinf_abort_generation(payload: RlinfPayload, request: Request) -> dict:
    """Abort all in-flight generations."""
    await _manager(request).abort_generation(_decode(payload), request)
    return {"status": "ok"}


def add_rlinf_routes(app, *, control_token: str, manager=None) -> None:
    """Mount the three /rlinf/* control routes on a sglang FastAPI app.

    Must be called before uvicorn starts serving the app.
    """
    if not control_token:
        raise ValueError("RLinf control routes require a nonempty authentication token")

    async def authenticate(request: Request) -> None:
        supplied = request.headers.get("X-RLinf-Control-Token", "")
        if not secrets.compare_digest(supplied.encode(), control_token.encode()):
            raise HTTPException(status_code=401, detail="Invalid RLinf control token")

    dependencies = [Depends(authenticate)]
    if manager is not None:
        app.state.rlinf_manager = manager
    app.add_api_route(
        "/rlinf/run_task_method",
        rlinf_run_task_method,
        methods=["POST"],
        dependencies=dependencies,
    )
    app.add_api_route(
        "/rlinf/sync_hf_weight",
        rlinf_sync_hf_weight,
        methods=["POST"],
        dependencies=dependencies,
    )
    app.add_api_route(
        "/rlinf/abort_generation",
        rlinf_abort_generation,
        methods=["POST"],
        dependencies=dependencies,
    )
    get_logger().info(
        "RLinf control routes mounted: /rlinf/run_task_method, "
        "/rlinf/sync_hf_weight, /rlinf/abort_generation "
        "(token authentication enabled)"
    )
