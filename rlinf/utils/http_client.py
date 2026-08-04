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

import os
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import aiohttp
import requests

_PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


@contextmanager
def no_proxy_env() -> Iterator[None]:
    """Temporarily unset HTTP proxy env vars.

    Removes ``http_proxy`` / ``https_proxy`` / ``all_proxy`` (both cases)
    from ``os.environ`` for the duration of the block, and restores the
    original values on exit. Any subprocess spawned inside the block
    inherits the stripped env, which is useful when the child's HTTP
    client (e.g. sglang-router's Rust reqwest, or sglang server's
    tokenizer-manager IPC) would otherwise tunnel intra-cluster traffic
    through a user-configured proxy.
    """
    saved = {var: os.environ.pop(var, None) for var in _PROXY_ENV_VARS}
    try:
        yield
    finally:
        for var, val in saved.items():
            if val is not None:
                os.environ[var] = val


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
        return self.post(
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
        return self.post("/v1/chat/completions", body)

    def health(self) -> bool:
        try:
            r = requests.get(
                f"{self.base_url}/health",
                timeout=5,
                proxies={"http": None, "https": None},
            )
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

    def post(
        self,
        path: str,
        body: dict,
        *,
        msgpack: bool = False,
        timeout_s: Optional[float] = None,
        max_retries: int = 0,
        retry_backoff_s: float = 0.0,
    ) -> dict:
        """POST ``body`` to ``path`` and return the decoded response dict.

        JSON by default. With ``msgpack=True`` the body (including torch tensors
        / numpy arrays) is serialized with the sglang msgpack codec and the
        response is msgpack-decoded; transient 5xx / connection errors are then
        retried with linear backoff. Proxies are always stripped. Heavy deps
        (sglang codec, torch, tianshou) are imported lazily so this module stays
        importable without them.
        """
        url = f"{self.base_url}{path}"
        retries = max(0, int(max_retries))
        retry_statuses = {500, 502, 503, 504}
        if msgpack:
            from sglang.multimodal_gen.runtime.entrypoints.vla.protocol import (
                pack_msgpack,
                unpack_msgpack,
            )

            ct = "application/msgpack"
            request_kwargs = {
                "data": pack_msgpack(self._to_msgpackable(body)),
                "headers": {"Content-Type": ct, "Accept": ct},
                "timeout": timeout_s,
            }
        else:
            # (connect, read) tuple: bound the TCP connect phase only.
            request_kwargs = {
                "json": body,
                "timeout": (self.connect_timeout, timeout_s),
            }

        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            is_last = attempt >= retries
            try:
                resp = requests.post(
                    url, proxies={"http": None, "https": None}, **request_kwargs
                )
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if is_last:
                    raise RuntimeError(
                        f"POST {url} failed after {retries + 1} attempt(s): {exc}"
                    ) from exc
                self._sleep_before_retry(attempt, retry_backoff_s)
                continue
            if resp.status_code in retry_statuses and not is_last:
                last_error = RuntimeError(
                    f"status={resp.status_code}, body={resp.text[:500]}"
                )
                self._sleep_before_retry(attempt, retry_backoff_s)
                continue
            if not resp.ok:
                detail = f"status={resp.status_code}, body={resp.text[:500]}"
                if last_error is not None:
                    detail += (
                        f" (after {retries + 1} attempt(s); prior error: {last_error})"
                    )
                raise RuntimeError(f"POST {url} failed: {detail}")
            if msgpack:
                if "msgpack" not in resp.headers.get("content-type", "").lower():
                    raise RuntimeError(
                        "expected a msgpack response, got content-type="
                        f"{resp.headers.get('content-type')!r}"
                    )
                return unpack_msgpack(resp.content)
            return resp.json()

    @staticmethod
    def _sleep_before_retry(attempt: int, retry_backoff_s: float) -> None:
        if retry_backoff_s > 0:
            time.sleep(retry_backoff_s * float(attempt + 1))

    @staticmethod
    def _to_msgpackable(value: Any) -> Any:
        """Recursively convert torch tensors / tianshou Batch to plain types."""
        import torch

        try:
            from tianshou.data import Batch

            if isinstance(value, Batch):
                value = value.__getstate__()
        except ImportError:
            pass
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, Mapping):
            return {
                str(k): InferenceHTTPClient._to_msgpackable(v) for k, v in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [InferenceHTTPClient._to_msgpackable(v) for v in value]
        return value

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
