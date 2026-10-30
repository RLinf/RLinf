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

"""OpenAI chat-completions endpoint for agents that run outside RLinf."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web


class SessionRequestError(Exception):
    """A request the handler rejects; the agent receives an OpenAI error object."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def chat_completion(
    completion_id: str,
    model: str,
    message: dict,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """Return an OpenAI chat completion with one choice."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@dataclass
class _Session:
    state: Any
    error: Exception | None = None
    # Set while no request of the session is being answered.
    idle: asyncio.Event = field(default_factory=asyncio.Event)


class OpenAISessionServer:
    """Serve ``POST /sessions/{session_id}/v1/chat/completions``.

    The owner opens a session for each agent run, gives the agent the session's
    base URL, and closes the session when the run ends. Each request is passed
    to ``handler`` with the state of its session, one request per session at a
    time: a request for an unknown session gets 404 and a concurrent one 409.
    Streaming requests receive the whole completion as one event.
    """

    def __init__(
        self,
        handler: Callable[[Any, dict], Awaitable[dict]],
        *,
        bind_host: str,
        port: int,
        advertise_host: str,
    ) -> None:
        self._handler = handler
        self._bind_host = bind_host
        self._port = port
        self._advertise_host = advertise_host
        self._sessions: dict[str, _Session] = {}
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Start serving; port zero binds an available port atomically."""
        if self._runner is not None:
            return
        app = web.Application()
        app.router.add_post("/sessions/{session_id}/v1/chat/completions", self._chat)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, self._bind_host, self._port).start()
        self._port = runner.addresses[0][1]
        self._runner = runner

    async def stop(self) -> None:
        """Stop serving; open sessions then receive no more requests."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def open(self, session_id: str, state: Any) -> str:
        """Serve a new session and return the base URL its agent should use."""
        if session_id in self._sessions:
            raise ValueError(f"session {session_id} is already open")
        session = _Session(state)
        session.idle.set()
        self._sessions[session_id] = session
        return f"http://{self._advertise_host}:{self._port}/sessions/{session_id}/v1"

    async def close(self, session_id: str) -> Exception | None:
        """Stop serving a session and return the first error its handler raised.

        A request already being answered runs to completion first, so the
        session's state is final once this returns.
        """
        session = self._sessions.pop(session_id)
        await session.idle.wait()
        return session.error

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        session = self._sessions.get(request.match_info["session_id"])
        if session is None:
            return self._error(404, "session_not_found", "the session is not open")
        if not session.idle.is_set():
            return self._error(
                409, "concurrent_request", "the session is answering a request"
            )
        session.idle.clear()
        try:
            body = await request.json()
            completion = await self._handler(session.state, body)
        except SessionRequestError as error:
            return self._error(error.status, error.code, str(error))
        except Exception as error:
            # The agent sees a server error; the owner gets the cause at close.
            session.error = session.error or error
            return self._error(500, "internal_error", repr(error))
        finally:
            session.idle.set()
        if not body.get("stream"):
            return web.json_response(completion)
        return await self._stream(request, completion)

    @staticmethod
    def _error(status: int, code: str, message: str) -> web.Response:
        return web.json_response(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": code,
                }
            },
            status=status,
        )

    @staticmethod
    async def _stream(request: web.Request, completion: dict) -> web.StreamResponse:
        choice = completion["choices"][0]
        delta = dict(choice["message"])
        if "tool_calls" in delta:
            delta["tool_calls"] = [
                {"index": idx, **call} for idx, call in enumerate(delta["tool_calls"])
            ]
        common = {key: completion[key] for key in ("id", "created", "model")}
        common["object"] = "chat.completion.chunk"
        chunks = [
            {
                **common,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                **common,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
                ],
                "usage": completion["usage"],
            },
        ]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in chunks:
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response
