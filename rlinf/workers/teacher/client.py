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

"""Bounded HTTP connections to immutable Teacher replicas."""

import asyncio
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch

from rlinf.data.schema.distillation import TeacherScoreRequest, TeacherTargets

from .backend import TeacherModelInfo


@dataclass(frozen=True)
class TeacherEndpoint:
    """Serializable service address and model identity."""

    url: str
    teacher_key: str
    model_info: TeacherModelInfo


class TeacherClient:
    """Limit in-flight scoring and match every response to its request."""

    def __init__(
        self,
        endpoints: list[TeacherEndpoint],
        *,
        max_concurrency: int = 8,
        max_total_concurrency: int = 16,
        timeout: float = 300,
        max_payload_bytes: int = 256 * 1024 * 1024,
    ):
        if (
            not endpoints
            or max_total_concurrency < 1
            or max_concurrency < 1
            or timeout <= 0
            or max_payload_bytes < 1
        ):
            raise ValueError(
                "TeacherClient requires endpoints and positive resource limits"
            )
        self.endpoints = defaultdict(list)
        for endpoint in endpoints:
            replicas = self.endpoints[endpoint.teacher_key]
            if replicas and replicas[0].model_info != endpoint.model_info:
                raise ValueError(
                    f"Replicas of Teacher {endpoint.teacher_key!r} disagree on model identity"
                )
            replicas.append(endpoint)
        self._total_limit = asyncio.Semaphore(max_total_concurrency)
        self._limits = {
            key: asyncio.Semaphore(max_concurrency) for key in self.endpoints
        }
        self._requests = {}
        self._pending = {}
        self._timeout = timeout
        self._max_payload_bytes = max_payload_bytes
        self._session = None
        self._next = defaultdict(int)
        self._closed = False

    def begin_step(self) -> None:
        """Release completed request identities after the synchronous step barrier."""
        if self._pending:
            raise RuntimeError(
                "Expected all Teacher requests to finish before starting a new step"
            )
        self._requests.clear()

    async def score(self, request: TeacherScoreRequest) -> TeacherTargets:
        """Score one request; errors fail the caller instead of dropping samples."""
        import aiohttp

        if self._closed:
            raise RuntimeError("TeacherClient is closed")
        key = request.teacher_key
        if key not in self.endpoints:
            raise ValueError(
                f"Unknown Teacher {key!r}; expected one of {list(self.endpoints)}"
            )
        replicas = self.endpoints[key]
        request.validate(replicas[0].model_info.vocab_size)
        signature = hashlib.sha256(json.dumps(asdict(request)).encode()).digest()
        pending = self._pending.get(request.request_id)
        if pending is not None:
            previous, task = pending
            if previous != signature:
                raise ValueError(
                    f"Conflicting content for request {request.request_id}"
                )
            return await asyncio.shield(task)
        previous = self._requests.get(request.request_id)
        if previous is not None and previous != signature:
            raise ValueError(f"Conflicting content for request {request.request_id}")

        async def dispatch():
            try:
                async with self._limits[key], self._total_limit:
                    first = self._next[key]
                    self._next[key] += 1
                    for offset in range(len(replicas)):
                        endpoint = replicas[(first + offset) % len(replicas)]
                        try:
                            return await self._score_endpoint(
                                endpoint, request, retry=offset > 0
                            )
                        except aiohttp.ClientResponseError as error:
                            if error.status not in {502, 503, 504} or offset + 1 == len(
                                replicas
                            ):
                                raise
                        except (TimeoutError, aiohttp.ClientConnectionError):
                            if offset + 1 == len(replicas):
                                raise
            finally:
                self._pending.pop(request.request_id, None)
                self._requests[request.request_id] = signature

        task = asyncio.create_task(dispatch())
        self._pending[request.request_id] = signature, task
        return await asyncio.shield(task)

    async def score_many(
        self, requests: list[TeacherScoreRequest]
    ) -> list[TeacherTargets]:
        """Score across model pools, preserving submission order."""
        return await asyncio.gather(*(self.score(request) for request in requests))

    async def _score_endpoint(self, endpoint, request, *, retry):
        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
        async with self._session.post(
            endpoint.url + "/score",
            json=asdict(request),
            headers={"X-Teacher-Retry": str(int(retry))},
        ) as response:
            response.raise_for_status()
            chunks, total = [], 0
            async for chunk in response.content.iter_chunked(64 * 1024):
                total += len(chunk)
                if total > self._max_payload_bytes:
                    raise ValueError(
                        f"Teacher request {request.request_id}: response exceeds byte budget"
                    )
                chunks.append(chunk)
        data = json.loads(b"".join(chunks))
        result = TeacherTargets(
            request_id=data["request_id"],
            teacher_key=data["teacher_key"],
            model_fingerprint=data["model_fingerprint"],
            target_format=data["target_format"],
            student_policy_version=data["student_policy_version"],
            token_ids=None
            if data["token_ids"] is None
            else torch.tensor(data["token_ids"], dtype=torch.long),
            logprobs=torch.tensor(data["logprobs"], dtype=torch.float32),
            valid_mask=torch.tensor(data["valid_mask"], dtype=torch.bool),
        )
        if result.model_fingerprint != endpoint.model_info.model_fingerprint:
            raise ValueError(
                f"Teacher request {request.request_id}: model fingerprint changed"
            )
        result.validate(request, endpoint.model_info.vocab_size)
        return result

    async def close(self) -> None:
        """Close local connections, leaving the remote model owned by its runtime."""
        self._closed = True
        pending = [task for _, task in self._pending.values()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if self._session is not None:
            session, self._session = self._session, None
            await session.close()
