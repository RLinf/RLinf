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

"""The session-scoped OpenAI endpoint that external agents call."""

import asyncio
import json

import httpx

from rlinf.workers.agent.openai_session_server import (
    OpenAISessionServer,
    chat_completion,
)

CALL = {"id": "call_0_0", "type": "function", "function": {"name": "look"}}


def serve(handler, scenario):
    """Run ``scenario(server, client)`` against a server on a free local port."""

    async def main():
        server = OpenAISessionServer(
            handler,
            bind_host="127.0.0.1",
            port=0,
            advertise_host="127.0.0.1",
        )
        await server.start()
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                return await scenario(server, client)
        finally:
            await server.stop()

    return asyncio.run(asyncio.wait_for(main(), timeout=30))


def test_requests_reach_open_sessions_one_at_a_time():
    release = asyncio.Event()

    async def handler(state, body):
        state.append(body["messages"])
        await release.wait()
        return chat_completion("c", "policy", {"role": "assistant"}, "stop", 1, 1)

    async def scenario(server, client):
        requests = []
        url = server.open("episode", requests) + "/chat/completions"
        body = {"model": "policy", "messages": [{"role": "user", "content": "go"}]}
        first = asyncio.create_task(client.post(url, json=body))
        while not requests:
            await asyncio.sleep(0.01)
        concurrent = await client.post(url, json=body)
        release.set()
        statuses = [(await first).status_code, concurrent.status_code]
        await server.close("episode")
        closed = await client.post(url, json=body)
        return statuses, closed, requests

    (answered, concurrent), closed, requests = serve(handler, scenario)
    assert (answered, concurrent, closed.status_code) == (200, 409, 404)
    assert closed.json()["error"]["code"] == "session_not_found"
    assert len(requests) == 1


def test_close_waits_for_the_request_in_progress():
    release = asyncio.Event()

    async def handler(state, body):
        await release.wait()
        state.append("answered")
        return chat_completion("c", "policy", {"role": "assistant"}, "stop", 1, 1)

    async def scenario(server, client):
        state = []
        url = server.open("episode", state) + "/chat/completions"
        body = {"model": "policy", "messages": []}
        request = asyncio.create_task(client.post(url, json=body))
        await asyncio.sleep(0.1)
        close = asyncio.create_task(server.close("episode"))
        await asyncio.sleep(0.1)
        waited = not close.done()
        release.set()
        await close
        return waited, list(state), (await request).status_code

    waited, state_at_close, status = serve(handler, scenario)
    assert waited and state_at_close == ["answered"] and status == 200


def test_streaming_request_receives_the_completion_as_one_event():
    async def handler(state, body):
        message = {"role": "assistant", "content": None, "tool_calls": [CALL]}
        return chat_completion("c", "policy", message, "tool_calls", 3, 2)

    async def scenario(server, client):
        url = server.open("episode", None) + "/chat/completions"
        body = {"model": "policy", "messages": [], "stream": True}
        return await client.post(url, json=body)

    response = serve(handler, scenario)
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line[len("data: ") :] for line in response.text.split("\n\n") if line]
    delta, finish = (json.loads(event)["choices"][0] for event in events[:2])
    assert delta["delta"]["tool_calls"] == [{"index": 0, **CALL}]
    assert finish["finish_reason"] == "tool_calls"
    assert json.loads(events[1])["usage"]["total_tokens"] == 5
    assert events[2] == "[DONE]"
