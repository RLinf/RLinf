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


from __future__ import annotations

from typing import Any, Optional

import aiohttp
import requests

from rlinf.utils import internal_http


class InferenceHTTPClient:
    """Thin HTTP client for an sglang router or server base URL.

    Both sync and async methods are provided. Async methods reuse a
    single lazily-created :class:`aiohttp.ClientSession`, so call
    :meth:`aclose` (or use ``async with``) to release sockets.


    Example::

        # sync
        client = InferenceHTTPClient("http://router:30000")
        out = client.generate(prompt="Hello", sampling_params={"max_new_tokens": 16})

        # async
        async with InferenceHTTPClient("http://router:30000") as c:
            pending = {asyncio.create_task(c.async_generate(prompt=p)) for p in prompts}
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    handle(task.result())
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout: float = 10.0,
        max_connections: int = 1024 * 16,
    ):
        self.base_url = base_url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.max_connections = max_connections
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list[int]] = None,
        sampling_params: Optional[dict] = None,
        return_logprob: bool = False,
    ) -> dict:
        return self._post(
            "/generate",
            self._generate_body(prompt, input_ids, sampling_params, return_logprob),
        )

    def chat_completion(
        self,
        messages: list[dict],
        model: str = "sglang-model",
        **kwargs: Any,
    ) -> dict:
        body = {"model": model, "messages": messages, **kwargs}
        return self._post("/v1/chat/completions", body)

    def health(self) -> bool:
        try:
            r = internal_http.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------
    async def async_generate(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list[int]] = None,
        sampling_params: Optional[dict] = None,
        return_logprob: bool = False,
    ) -> dict:
        return await self._apost(
            "/generate",
            self._generate_body(prompt, input_ids, sampling_params, return_logprob),
        )

    async def async_chat_completion(
        self,
        messages: list[dict],
        model: str,
        **kwargs: Any,
    ) -> dict:
        body = {"model": model, "messages": messages, **kwargs}
        return await self._apost("/v1/chat/completions", body)

    async def async_health(self) -> bool:
        session = self._get_or_create_session()
        try:
            async with session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, TimeoutError):
            return False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def _get_or_create_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # limit         = total in-flight conns across all hosts
            # limit_per_host=0  → no separate per-host ceiling, so `limit` rules
            # enable_cleanup_closed helps reclaim sockets when peers hang up.
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=0,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def aclose(self) -> None:
        """Close the underlying aiohttp session, if one was created."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "InferenceHTTPClient":
        self._get_or_create_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_body(
        prompt: Optional[str],
        input_ids: Optional[list[int]],
        sampling_params: Optional[dict],
        return_logprob: bool,
    ) -> dict:
        body: dict = {"return_logprob": return_logprob}
        if prompt is not None:
            body["text"] = prompt
        if input_ids is not None:
            body["input_ids"] = input_ids
        if sampling_params is not None:
            body["sampling_params"] = sampling_params
        return body

    def _post(self, path: str, body: dict) -> dict:
        # (connect, read) tuple: bound the TCP connect phase only;
        # let the response take as long as it needs.
        resp = internal_http.post(
            f"{self.base_url}{path}",
            json=body,
            timeout=(self.connect_timeout, None),
        )
        resp.raise_for_status()
        return resp.json()

    async def _apost(self, path: str, body: dict) -> dict:
        session = self._get_or_create_session()
        async with session.post(
            f"{self.base_url}{path}",
            json=body,
            timeout=aiohttp.ClientTimeout(
                total=None,
                sock_connect=self.connect_timeout,
                sock_read=None,
            ),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
