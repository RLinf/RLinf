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

"""RLinf control-plane routes mounted onto sglang's FastAPI app.

Server-mode rollout drives a sglang HTTP server from the RLinf rollout
worker over loopback. Three capabilities of the patched TokenizerManager
(`run_task_method` / `sync_hf_weight` / `abort_generation`) exist only as
Python methods, so they need matching HTTP routes. They are mounted with
`app.add_api_route` before `launch_server` starts uvicorn — no
monkeypatching, one less import-order dependency.

Requests carry arbitrary Python objects (worker address, placement,
config) that the ZMQ path already pickles today, so the body is
``{"payload_b64": base64(pickle.dumps(obj))}`` and each handler unpickles
it before forwarding to the TokenizerManager.

Security: these routes unpickle the request body — an RCE surface. The
server must not be reachable from untrusted networks; server-mode rollout
binds to 127.0.0.1.

Note: routes are registered on the in-process ``app`` object, so they
serve on the default single-tokenizer path (``uvicorn.run(app, ...)``)
only, not in ``--tokenizer-worker-num > 1`` worker processes. RLinf never
enables that mode.
"""

import base64
import pickle
from typing import Any

from fastapi import Request
from pydantic import BaseModel


class RlinfPayload(BaseModel):
    """Body of every /rlinf/* route: a pickled request object, b64-wrapped."""

    payload_b64: str


def _decode(payload: RlinfPayload) -> Any:
    return pickle.loads(base64.b64decode(payload.payload_b64))


def _encode(result: Any) -> dict:
    return {"payload_b64": base64.b64encode(pickle.dumps(result)).decode()}


def _tokenizer_manager():
    from sglang.srt.entrypoints.http_server import get_global_state

    return get_global_state().tokenizer_manager


async def rlinf_run_task_method(payload: RlinfPayload, request: Request) -> dict:
    """Run a method in the scheduler via the TokenizerManager."""
    result = await _tokenizer_manager().run_task_method(_decode(payload), request)
    return _encode(result)


async def rlinf_sync_hf_weight(payload: RlinfPayload, request: Request) -> dict:
    """Trigger an actor→rollout NCCL weight sync round."""
    await _tokenizer_manager().sync_hf_weight(_decode(payload), request)
    return {"status": "ok"}


async def rlinf_abort_generation(payload: RlinfPayload, request: Request) -> dict:
    """Abort all in-flight generations."""
    await _tokenizer_manager().abort_generation(_decode(payload), request)
    return {"status": "ok"}


def add_rlinf_routes(app) -> None:
    """Mount the three /rlinf/* control routes on a sglang FastAPI app.

    Must be called before uvicorn starts serving the app (i.e. before
    ``launch_server``).
    """
    app.add_api_route("/rlinf/run_task_method", rlinf_run_task_method, methods=["POST"])
    app.add_api_route("/rlinf/sync_hf_weight", rlinf_sync_hf_weight, methods=["POST"])
    app.add_api_route(
        "/rlinf/abort_generation", rlinf_abort_generation, methods=["POST"]
    )
    print(
        "RLinf control routes mounted: /rlinf/run_task_method, "
        "/rlinf/sync_hf_weight, /rlinf/abort_generation",
        flush=True,
    )
