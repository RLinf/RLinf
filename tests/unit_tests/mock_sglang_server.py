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
    """Return a port the kernel currently assigns for binding."""
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
        # Texts returned (and detokenized) instead of the id-derived text,
        # one per /generate call, oldest first; the last one repeats once
        # the script runs out. Empty means every response derives its text
        # from the ids.
        self.scripted_texts: list[str] = []
        # Output ids answered as-is with no text, for an engine started
        # without a tokenizer (``rollout.detokenize`` off). Takes
        # precedence over the two text attributes above.
        self.scripted_output_ids: list[int] = []
        # ("first turn text", "later turn text"): a request whose input is no
        # longer than the first one seen is a conversation's opening turn and
        # gets the first text, a longer one continues it and gets the second.
        # Lets a test script tool-call-then-answer for every conversation.
        self.scripted_by_turn: tuple[str, str] | None = None
        self._scripted_first_len: int | None = None
        self._generated_calls = 0

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
        response = {"output_ids": output_ids, "meta_info": meta_info}
        if self.scripted_output_ids:
            # An engine without a tokenizer answers ids only: no "text" key
            # at all, which is what a caller decoding them itself must see.
            response["output_ids"] = list(self.scripted_output_ids)
            return response
        if self.scripted_by_turn is not None:
            first_text, later_text = self.scripted_by_turn
            input_len = len(body.get("input_ids", []))
            if self._scripted_first_len is None:
                self._scripted_first_len = input_len
            text = first_text if input_len <= self._scripted_first_len else later_text
            response["text"] = text
            response["output_ids"] = self._ids_for_text(text, body.get("input_ids", []))
            return response
        if self.scripted_texts:
            index = min(self._generated_calls, len(self.scripted_texts) - 1)
            text = self.scripted_texts[index]
            self._generated_calls += 1
            response["text"] = text
            # Keep the ids consistent with the text for callers that use
            # them (the chat entry parses the text and reports lengths from
            # the ids).
            tokenizer_args = body.get("input_ids", [])
            response["output_ids"] = self._ids_for_text(text, tokenizer_args)
        return response

    def _ids_for_text(self, text: str, input_ids: list[int]) -> list[int]:
        """Deterministic ids for a scripted text (id per character)."""
        seed = input_ids[0] if input_ids else 0
        return [(seed + 1000 + ord(char)) % 30000 for char in text]
