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

"""A thread-hosted stand-in for an sglang server, for tests.

Adapted from the ``MockSGLangServer`` and ``UvicornThreadServer`` of
github.com/radixark/miles (Apache-2.0), ``miles/utils/test_utils/``. Unlike
the miles mock this one needs no tokenizer: it answers ``/generate`` with
token ids derived deterministically from the request, so tests can build a
later turn's input from an earlier turn's output.

The recorded requests (path, headers, body) let a test assert what the
session server forwarded: the routing-key header, the merged sampling
params, and the stripped request block.
"""

import asyncio
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def find_available_port() -> int:
    """Return a port the kernel currently hands out for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class UvicornThreadServer:
    """Run a uvicorn server on a daemon thread until :meth:`stop`."""

    def __init__(self, app, host: str, port: int):
        self._app = app
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        config = uvicorn.Config(
            self._app, host=self.host, port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)

        def run() -> None:
            asyncio.run(self._server.serve())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        self._wait_for_port_open()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _wait_for_port_open(self) -> None:
        for _ in range(100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    if sock.connect_ex((self.host, self.port)) == 0:
                        return
            except OSError:
                pass
            time.sleep(0.1)
        raise RuntimeError(f"Failed to start server on {self.url}")


class MockSGLangServer:
    """Answer /generate, /health, and /server_info like an sglang server.

    Output ids are derived deterministically from the input ids and the
    requested length, so consecutive turns can be chained in a test. The
    reported ``weight_version`` and per-request latency are configurable
    attributes; the latency can be raised mid-test to keep a generate in
    flight while the test does something else.
    """

    def __init__(self, host: str = "127.0.0.1", port: int | None = None):
        self.host = host
        self.port = port if port is not None else find_available_port()
        self.weight_version: str = "5"
        self.latency_s: float = 0.0
        self.request_log: list[dict] = []

        self.app = FastAPI()
        self._server = UvicornThreadServer(self.app, self.host, self.port)
        self._setup_routes()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.stop()

    def _setup_routes(self) -> None:
        @self.app.post("/generate")
        async def generate(request: Request):
            headers = dict(request.headers.items())
            body = await request.json()
            self.request_log.append(
                {"path": "/generate", "headers": headers, "body": body}
            )
            if self.latency_s > 0:
                await asyncio.sleep(self.latency_s)
            return JSONResponse(content=self._generate_response(body))

        @self.app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        @self.app.get("/server_info")
        async def server_info():
            return JSONResponse(
                content={
                    "model_name": "mock-model",
                    "weight_version": self.weight_version,
                }
            )

    def _generate_response(self, body: dict) -> dict:
        """Deterministic output ids honoring sampling_params.max_new_tokens."""
        input_ids = body.get("input_ids", [])
        sampling_params = body.get("sampling_params") or {}
        max_new_tokens = int(sampling_params.get("max_new_tokens", 8))
        seed = input_ids[0] if input_ids else 0
        output_ids = [
            (seed + 1000 + index * 7) % 30000 for index in range(max_new_tokens)
        ]
        meta_info = {
            "finish_reason": {"type": "stop", "length": max_new_tokens},
            "weight_version": self.weight_version,
            "cached_tokens": 0,
            "prompt_tokens": len(input_ids),
            "completion_tokens": max_new_tokens,
        }
        if body.get("return_logprob"):
            meta_info["output_token_logprobs"] = [
                [-0.01 * index, token_id] for index, token_id in enumerate(output_ids)
            ]
        return {"output_ids": output_ids, "meta_info": meta_info}
