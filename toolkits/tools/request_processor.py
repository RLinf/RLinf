import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable

import aiohttp


BatchSubmitFunc = Callable[[list[Any], aiohttp.ClientSession], Awaitable[list[Any]]]


class RequestProcessor:
    """
    Minimal async batching processor adapted for RLinf to send tool calls in batches.
    """

    def __init__(self, batch_size: int, batch_timeout_seconds: float, session: aiohttp.ClientSession, concurrency: int, batch_submit_func: BatchSubmitFunc):
        if batch_size <= 0 or concurrency <= 0:
            raise ValueError("batch_size and concurrency must be positive")
        self._batch_size = batch_size
        self._batch_timeout_seconds = batch_timeout_seconds
        self._session = session
        self._concurrency = concurrency
        self._batch_submit_func = batch_submit_func

        self._submission_queue = asyncio.Queue()
        self._pending_requests: dict[str, dict[str, Any]] = {}
        self._semaphore = asyncio.Semaphore(concurrency)
        self._sender_workers: list[asyncio.Task] = []
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        self._sender_workers = [asyncio.create_task(self._sender_worker()) for _ in range(self._concurrency)]

    async def stop(self):
        if not self._running:
            return
        self._running = False
        await self._submission_queue.join()
        for w in self._sender_workers:
            w.cancel()
        await asyncio.gather(*self._sender_workers, return_exceptions=True)

    async def send_request(self, payload: Any, timeout: float | None = None):
        if not self._running:
            raise RuntimeError("RequestProcessor is not running. Call .start() first.")
        req_id = str(uuid.uuid4())
        fut = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = {"future": fut, "payload": payload}
        await self._submission_queue.put(req_id)
        return await asyncio.wait_for(fut, timeout=timeout)

    async def _sender_worker(self):
        while self._running or not self._submission_queue.empty():
            batch_ids: list[str] = []
            try:
                first = await asyncio.wait_for(self._submission_queue.get(), timeout=self._batch_timeout_seconds)
                self._submission_queue.task_done()
                batch_ids.append(first)
                while len(batch_ids) < self._batch_size:
                    try:
                        nxt = self._submission_queue.get_nowait()
                        self._submission_queue.task_done()
                        batch_ids.append(nxt)
                    except asyncio.QueueEmpty:
                        break
            except asyncio.TimeoutError:
                if not batch_ids:
                    continue

            if batch_ids:
                async with self._semaphore:
                    await self._perform_send_batch(batch_ids)

    async def _perform_send_batch(self, batch_ids: list[str]):
        valid_ids = [i for i in batch_ids if i in self._pending_requests]
        if not valid_ids:
            return
        payloads = [self._pending_requests[i]["payload"] for i in valid_ids]
        results = await self._batch_submit_func(payloads, self._session)
        for i, res in zip(valid_ids, results, strict=False):
            fut = self._pending_requests[i]["future"]
            if not fut.done():
                fut.set_result(res)
            del self._pending_requests[i]


