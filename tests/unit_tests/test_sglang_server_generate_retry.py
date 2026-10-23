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

"""Unit tests for server-backend generation error handling.

1. Connection-level errors (ECONNRESET from the keep-alive race) are
   retried by ``SGLangServerBackend.async_generate``.
2. A generate failure is delivered to the requesting agent loop as an
   ``{"error": ...}`` payload on the same channel key, and
   ``AgentLoop.generate`` turns that payload into a ``RuntimeError``.

No sglang or Ray is launched; a local aiohttp test server provides the
/generate endpoint.
"""

import asyncio
from unittest import mock

import aiohttp
from aiohttp import web

from rlinf.workers.agent.agent_loop import AgentLoopWorker
from rlinf.workers.rollout.sglang.backends import SGLangServerBackend


def _make_backend() -> SGLangServerBackend:
    """A server backend whose client is stubbed out (no subprocess)."""
    backend = SGLangServerBackend.__new__(SGLangServerBackend)
    backend._log_info = lambda msg: None
    return backend


class _ResettingServer:
    """Local aiohttp server whose first /generate closes the connection."""

    def __init__(self):
        self.hit_count = 0
        self.runner = None

    async def _handle_generate(self, request):
        self.hit_count += 1
        if self.hit_count == 1:
            # Drop the TCP connection without a response, like uvicorn
            # closing a keep-alive connection mid-request.
            request.transport.abort()
            return
        return web.json_response({"output_ids": [1, 2, 3]})

    async def start(self):
        app = web.Application()
        app.router.add_post("/generate", self._handle_generate)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        return f"http://127.0.0.1:{port}"

    async def stop(self):
        if self.runner is not None:
            await self.runner.cleanup()


def test_async_generate_retries_connection_reset():
    """First request is dropped (ECONNRESET); the retry succeeds."""

    async def run():
        server = _ResettingServer()
        url = await server.start()
        try:
            from rlinf.utils.http_client import InferenceHTTPClient

            client = InferenceHTTPClient(url)
            backend = _make_backend()
            backend._client = client
            result = await backend.async_generate(
                prompt="hello",
                sampling_params={"max_new_tokens": 4},
                image_data=None,
                return_logprob=False,
            )
            assert result["output_ids"] == [1, 2, 3]
            assert server.hit_count == 2
            await client.aclose()
        finally:
            await server.stop()

    asyncio.run(run())


class _FakeChannel:
    """Minimal channel double: records puts, serves stored payloads."""

    def __init__(self):
        self.items = {}

    def put(self, item, key=None, async_op=False):
        self.items[key] = item

        async def _wait():
            return None

        return mock.MagicMock(async_wait=_wait)

    def get(self, key=None, async_op=False):
        item = self.items.pop(key)

        async def _wait():
            return item

        return mock.MagicMock(async_wait=_wait)


def test_generate_failure_surfaces_as_error_payload():
    """Retries exhausted -> channel gets {"error": ...} -> AgentLoop raises."""

    async def run():
        output_channel = _FakeChannel()
        input_channel = _FakeChannel()
        worker = AgentLoopWorker.__new__(AgentLoopWorker)
        worker.generate_input_channel = input_channel
        worker.generate_output_channel = output_channel
        worker.solid_generate_input_channels = {}
        worker.tool_channel_info_map = {}
        worker.tool_name_map = {}
        worker.tool_worker_output_channel = None

        # The rollout side: retries exhausted, error payload goes to the
        # channel.
        from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

        sglang_worker = SGLangWorker.__new__(SGLangWorker)
        sglang_worker._sampling_params = {"max_new_tokens": 4}
        sglang_worker._return_logprobs = False
        sglang_worker._is_model_instance_entry = True
        # Bypass Worker.__init__ (no logger); capture errors directly.
        sglang_worker.log_error = mock.MagicMock()

        async def _always_fail(**kwargs):
            raise aiohttp.ClientOSError(104, "Connection reset by peer")

        backend = mock.MagicMock()
        backend.async_generate = _always_fail
        sglang_worker._backend = backend

        await sglang_worker.generate_and_send(
            output_channel=output_channel,
            channel_key="key-1",
            prompt_ids=[1, 2, 3],
        )
        assert "key-1" in output_channel.items
        payload = output_channel.items["key-1"]
        assert isinstance(payload, dict) and "error" in payload
        assert "ClientOSError" in payload["error"]
        sglang_worker.log_error.assert_called_once()

        # The agent-loop side: the error payload becomes a RuntimeError.
        # Redirect the loop's input puts so the error payload stored under
        # "key-1" above is what generate gets back for its own request key.
        async def _noop_wait():
            return None

        def _redirect_put(item, key=None, async_op=False):
            output_channel.items[item["channel_key"]] = output_channel.items["key-1"]
            return mock.MagicMock(async_wait=_noop_wait)

        input_channel.put = _redirect_put
        try:
            await worker.generate(prompt_ids=[1, 2, 3])
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "Rollout generate request failed" in str(e)

    asyncio.run(run())
