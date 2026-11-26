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

"""Scheduler-owned frozen Teacher service."""

import asyncio
from collections import Counter
from dataclasses import asdict
from time import monotonic

from rlinf.algorithms.distillation.config import (
    DistillationLossConfig,
    TeacherModelConfig,
)
from rlinf.data.schema.distillation import TeacherScoreRequest, TeacherTargets
from rlinf.scheduler import Worker

from .client import TeacherEndpoint


class TeacherWorker(Worker):
    """Own engine initialization, scoring HTTP endpoint and idempotent shutdown."""

    def __init__(
        self,
        teacher_key: str,
        config: TeacherModelConfig,
        loss_config: DistillationLossConfig,
    ):
        super().__init__()
        self.teacher_key = teacher_key
        self.config = config
        self.loss_config = loss_config
        self.backend = None
        self._runner = None
        self._metrics = Counter()
        self.endpoint = None
        self._limit = asyncio.Semaphore(config.inference.get("max_concurrency", 8))

    async def init_worker(self) -> TeacherEndpoint:
        """Publish the endpoint only after the requested score shape is supported."""
        import ray
        from aiohttp import web

        if self.endpoint is not None:
            return self.endpoint
        if self.config.inference.get("backend", "sglang") == "sglang":
            from .sglang_backend import SGLangTeacherBackend

            backend_cls = SGLangTeacherBackend
        else:
            from .vllm_backend import VLLMTeacherBackend

            backend_cls = VLLMTeacherBackend
        self.backend = backend_cls(self.config, self.teacher_key)
        try:
            info = await self.backend.initialize()
            mode = self.loss_config.canonical_loss_mode
            topk = self.loss_config.topk if "topk" in mode else None
            if topk is not None and topk > info.vocab_size:
                raise ValueError(
                    f"Teacher topk must not exceed model output vocabulary size {info.vocab_size}, got {topk}"
                )
            selected = None
            version = None
            if (
                topk is not None
                and topk > 0
                and self.loss_config.topk_source == "student"
            ):
                selected = (tuple(range(topk)),) * 2
                version = "initialization"
            if self.loss_config.mc_k > 1:
                selected = ((0,) * self.loss_config.mc_k,) * 2
            probe = TeacherScoreRequest(
                "initialization",
                "initialization",
                "initialization",
                self.teacher_key,
                (0, 0),
                (False, True),
                topk,
                info.tokenizer_fingerprint,
                selected,
                version,
                mc_k=self.loss_config.mc_k,
            )
            await self.backend.score(probe)
            app = web.Application(
                client_max_size=self.config.inference.get(
                    "max_request_bytes", 64 * 1024 * 1024
                )
            )
            app.router.add_post("/score", self._handle_score)
            self._runner = web.AppRunner(app, access_log=None)
            await self._runner.setup()
            port = self.acquire_free_port()
            await web.TCPSite(self._runner, "0.0.0.0", port).start()
            self.endpoint = TeacherEndpoint(
                f"http://{ray.util.get_node_ip_address()}:{port}",
                self.teacher_key,
                info,
            )
            return self.endpoint
        except BaseException:
            try:
                await self.stop()
            except Exception:
                self.log_error("Teacher cleanup failed during initialization rollback")
            raise

    async def score(self, request: TeacherScoreRequest) -> TeacherTargets:
        """Score one sample using the frozen model and bounded engine concurrency."""
        if self.backend is None:
            raise RuntimeError("Teacher is not initialized")
        queued = monotonic()
        self._metrics["requests"] += 1
        self._metrics["tokens"] += len(request.input_ids)
        async with self._limit:
            started = monotonic()
            self._metrics["queue_seconds"] += started - queued
            try:
                return await self.backend.score(request)
            except Exception:
                self._metrics["failures"] += 1
                raise
            finally:
                self._metrics["score_seconds"] += monotonic() - started

    def take_metrics(self) -> dict:
        """Drain replica counters at a completed synchronous update boundary."""
        metrics = self._metrics.copy()
        self._metrics.clear()
        return metrics

    async def _handle_score(self, http_request):
        from aiohttp import web

        if http_request.headers.get("X-Teacher-Retry") == "1":
            self._metrics["retries"] += 1
        payload = await http_request.json()
        request = TeacherScoreRequest(**payload)
        try:
            result = await self.score(request)
        except Exception as error:
            self.log_error(
                f"Teacher request {request.request_id} ({self.teacher_key}) failed: {type(error).__name__}"
            )
            raise web.HTTPBadRequest(
                text=f"Teacher request {request.request_id} ({self.teacher_key}) failed"
            ) from None
        data = asdict(result)
        for key in ("token_ids", "logprobs", "valid_mask"):
            if data[key] is not None:
                data[key] = data[key].tolist()
        return web.json_response(data)

    async def stop(self) -> None:
        """Stop accepting requests and release the frozen inference engine."""
        self.endpoint = None
        try:
            if self._runner is not None:
                runner, self._runner = self._runner, None
                await runner.cleanup()
        finally:
            if self.backend is not None:
                backend, self.backend = self.backend, None
                await backend.close()
