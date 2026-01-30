# Copyright 2025 The RLinf Authors.
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

import asyncio
import random
import threading
import time
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from typing import Any

from rlinf.scheduler import Channel, Cluster, NodePlacementStrategy, Worker

try:
    from ray.util.queue import Queue as RayQueue
except ImportError:
    RayQueue = None  # type: ignore[misc, assignment]


@dataclass
class BenchmarkConfig:
    num_messages: int = 2000
    num_warmup_messages: int = 2
    payload_size: int = 1024 * 1024  # bytes
    channel_maxsize: int = 0
    enable_thread_interference: bool = False
    num_noise_threads: int = 2


class Producer(Worker):
    def __init__(self):
        super().__init__()
        self._noise_started = False

    @staticmethod
    def _progress(i: int, total: int, prefix: str) -> None:
        if total <= 0:
            return
        # Update ~50 times across the run to keep logs reasonable.
        step = max(1, total // 50)
        if (i % step) != 0 and i != total - 1:
            return
        pct = (i + 1) / total
        bar_len = 40
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(
            f"\r[{prefix}] |{bar}| {pct * 100:6.2f}% ({i + 1}/{total})",
            end="",
            flush=True,
        )
        if i == total - 1:
            print()

    def prefill(self, channel: Channel, cfg: BenchmarkConfig) -> int:
        """Fill the channel with `num_messages` synchronously (no timing)."""
        payload = b"x" * cfg.payload_size
        for _ in range(cfg.num_messages):
            channel.put(payload)
        return cfg.num_messages

    def start_cpu_noise(self, cfg: BenchmarkConfig) -> None:
        """Optionally start background CPU-burning threads on this worker."""
        if self._noise_started or not cfg.enable_thread_interference:
            return
        self._noise_started = True

        def _burn():
            while True:
                x = 0.0
                for _ in range(100_000):
                    x += 1.0

        for _ in range(max(1, cfg.num_noise_threads)):
            t = threading.Thread(target=_burn, daemon=True)
            t.start()

    def warmup(self, channel: Channel, cfg: BenchmarkConfig, async_mode: bool) -> None:
        """Un-timed warmup for puts to build up channel state and JITs."""
        payload = b"x" * cfg.payload_size
        if async_mode:

            async def _warmup() -> None:
                for _ in range(cfg.num_warmup_messages):
                    work = channel.put(payload, async_op=True)
                    await work.async_wait()

            asyncio.run(_warmup())
        else:
            for _ in range(cfg.num_warmup_messages):
                channel.put(payload)

    def run_sync(self, channel: Channel, cfg: BenchmarkConfig) -> float:
        """Synchronous put: each put blocks until finished."""
        payload = b"x" * cfg.payload_size
        with self.worker_timer("producer_sync"):
            for i in range(cfg.num_messages):
                channel.put(payload)
                self._progress(i, cfg.num_messages, "put  sync ")
        return self.pop_execution_time("producer_sync")

    def run_async(self, channel: Channel, cfg: BenchmarkConfig) -> float:
        """Async put using asyncio: await put(..., async_op=True).async_wait()."""

        async def _run() -> None:
            payload = b"x" * cfg.payload_size
            for i in range(cfg.num_messages):
                work = channel.put(payload, async_op=True)
                await work.async_wait()
                self._progress(i, cfg.num_messages, "put  async")

        with self.worker_timer("producer_async"):
            asyncio.run(_run())
        return self.pop_execution_time("producer_async")

    def run_sync_keys(
        self, channel: Channel, cfg: BenchmarkConfig, keys: list[int | str]
    ) -> float:
        """Synchronous put with per-message keys."""
        assert len(keys) == cfg.num_messages
        payload = b"x" * cfg.payload_size
        with self.worker_timer("producer_sync_keys"):
            for i, key in enumerate(keys):
                channel.put(payload, key=key)
                self._progress(i, cfg.num_messages, "putK sync")
        return self.pop_execution_time("producer_sync_keys")

    def run_async_keys(
        self, channel: Channel, cfg: BenchmarkConfig, keys: list[int | str]
    ) -> float:
        """Async put with per-message keys using asyncio."""
        assert len(keys) == cfg.num_messages
        payload = b"x" * cfg.payload_size

        async def _run() -> None:
            for i, key in enumerate(keys):
                work = channel.put(payload, key=key, async_op=True)
                await work.async_wait()
                self._progress(i, cfg.num_messages, "putK async")

        with self.worker_timer("producer_async_keys"):
            asyncio.run(_run())
        return self.pop_execution_time("producer_async_keys")

    def prefill_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> int:
        """Fill ray.util.queue.Queue with num_messages (no timing)."""
        if RayQueue is None:
            return 0
        payload = b"x" * cfg.payload_size
        for _ in range(cfg.num_messages):
            queue.put(payload)
        return cfg.num_messages

    def run_sync_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> float:
        """Synchronous put on ray.util.queue.Queue."""
        if RayQueue is None:
            return 0.0
        payload = b"x" * cfg.payload_size
        with self.worker_timer("producer_sync_ray_queue"):
            for i in range(cfg.num_messages):
                queue.put(payload)
                self._progress(i, cfg.num_messages, "rayQ put ")
        return self.pop_execution_time("producer_sync_ray_queue")

    def run_async_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> float:
        """Async put on ray.util.queue.Queue (put_async returns coroutine)."""
        if RayQueue is None:
            return 0.0
        payload = b"x" * cfg.payload_size

        async def _run():
            for i in range(cfg.num_messages):
                await queue.put_async(payload)
                if i % max(1, cfg.num_messages // 50) == 0 or i == cfg.num_messages - 1:
                    self._progress(i, cfg.num_messages, "rayQ putA")

        with self.worker_timer("producer_async_ray_queue"):
            asyncio.run(_run())
        return self.pop_execution_time("producer_async_ray_queue")


class Consumer(Worker):
    def __init__(self):
        super().__init__()
        self._noise_started = False

    @staticmethod
    def _progress(i: int, total: int, prefix: str) -> None:
        if total <= 0:
            return
        step = max(1, total // 50)
        if (i % step) != 0 and i != total - 1:
            return
        pct = (i + 1) / total
        bar_len = 40
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(
            f"\r[{prefix}] |{bar}| {pct * 100:6.2f}% ({i + 1}/{total})",
            end="",
            flush=True,
        )
        if i == total - 1:
            print()

    def run_sync(self, channel: Channel, cfg: BenchmarkConfig) -> float:
        with self.worker_timer("consumer_sync"):
            for i in range(cfg.num_messages):
                _ = channel.get()
                self._progress(i, cfg.num_messages, "get  sync ")
        return self.pop_execution_time("consumer_sync")

    def run_async(self, channel: Channel, cfg: BenchmarkConfig) -> float:
        """Async get using asyncio: await get(async_op=True).async_wait()."""

        async def _run() -> None:
            for i in range(cfg.num_messages):
                work = channel.get(async_op=True)
                await work.async_wait()
                self._progress(i, cfg.num_messages, "get  async")

        with self.worker_timer("consumer_async"):
            asyncio.run(_run())
        return self.pop_execution_time("consumer_async")

    def run_sync_keys(
        self, channel: Channel, cfg: BenchmarkConfig, keys: list[int | str]
    ) -> float:
        """Synchronous get with per-message keys."""
        assert len(keys) == cfg.num_messages
        with self.worker_timer("consumer_sync_keys"):
            for i, key in enumerate(keys):
                _ = channel.get(key=key)
                self._progress(i, cfg.num_messages, "getK sync")
        return self.pop_execution_time("consumer_sync_keys")

    def run_async_keys(
        self, channel: Channel, cfg: BenchmarkConfig, keys: list[int | str]
    ) -> float:
        """Async get with per-message keys using asyncio."""
        assert len(keys) == cfg.num_messages

        async def _run() -> None:
            for i, key in enumerate(keys):
                work = channel.get(key=key, async_op=True)
                await work.async_wait()
                self._progress(i, cfg.num_messages, "getK async")

        with self.worker_timer("consumer_async_keys"):
            asyncio.run(_run())
        return self.pop_execution_time("consumer_async_keys")

    def warmup(self, channel: Channel, cfg: BenchmarkConfig, async_mode: bool) -> None:
        """Un-timed warmup for gets."""
        if async_mode:

            async def _warmup() -> None:
                for _ in range(cfg.num_warmup_messages):
                    work = channel.get(async_op=True)
                    await work.async_wait()

            asyncio.run(_warmup())
        else:
            for _ in range(cfg.num_warmup_messages):
                _ = channel.get()

    def run_sync_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> float:
        """Synchronous get on ray.util.queue.Queue."""
        if RayQueue is None:
            return 0.0
        with self.worker_timer("consumer_sync_ray_queue"):
            for i in range(cfg.num_messages):
                _ = queue.get()
                self._progress(i, cfg.num_messages, "rayQ get ")
        return self.pop_execution_time("consumer_sync_ray_queue")

    def run_async_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> float:
        """Async get on ray.util.queue.Queue (get_async returns coroutine)."""
        if RayQueue is None:
            return 0.0

        async def _run():
            for i in range(cfg.num_messages):
                _ = await queue.get_async()
                if i % max(1, cfg.num_messages // 50) == 0 or i == cfg.num_messages - 1:
                    self._progress(i, cfg.num_messages, "rayQ getA")

        with self.worker_timer("consumer_async_ray_queue"):
            asyncio.run(_run())
        return self.pop_execution_time("consumer_async_ray_queue")

    def start_cpu_noise(self, cfg: BenchmarkConfig) -> None:
        """Optionally start background CPU-burning threads on this worker."""
        if self._noise_started or not cfg.enable_thread_interference:
            return
        self._noise_started = True

        def _burn():
            while True:
                x = 0.0
                for _ in range(100_000):
                    time.sleep(0.001)
                    x += 1.0

        for _ in range(max(1, cfg.num_noise_threads)):
            t = threading.Thread(target=_burn, daemon=True)
            t.start()


def run_benchmark(cfg: BenchmarkConfig) -> None:
    # Initialize a single-node cluster first so Channel/Workers share the same Ray cluster.
    cluster = Cluster(num_nodes=1)

    placement = NodePlacementStrategy(node_ranks=[0])
    producer_group = Producer.create_group().launch(
        cluster=cluster, placement_strategy=placement, name="channel_perf_producer"
    )
    consumer_group = Consumer.create_group().launch(
        cluster=cluster, placement_strategy=placement, name="channel_perf_consumer"
    )

    if cfg.enable_thread_interference:
        print(
            f"\n[Info] Enabling thread interference with "
            f"{cfg.num_noise_threads} background threads per worker."
        )
        producer_group.start_cpu_noise(cfg).wait()
        consumer_group.start_cpu_noise(cfg).wait()

    # Single shared channel for all tests.
    channel = Channel.create(
        name="pressure_demo_channel_shared",
        maxsize=cfg.channel_maxsize,
        distributed=False,
    )

    def reset_channel() -> None:
        """Drain all remaining messages from the shared channel."""
        from asyncio import QueueEmpty

        while True:
            try:
                channel.get_nowait()
            except QueueEmpty:
                break

    # Ray util queue for comparison (ray.util.queue.Queue).
    ray_queue = None
    if RayQueue is not None:
        ray_queue = RayQueue(maxsize=cfg.channel_maxsize or 0)

    def reset_ray_queue() -> None:
        """Drain ray.util.queue.Queue."""
        if ray_queue is None:
            return
        while True:
            try:
                ray_queue.get_nowait()
            except QueueEmpty:
                break

    def ray_queue_round(async_mode: bool) -> dict[str, float] | None:
        """Mixed put+get on ray.util.queue.Queue (same pattern as one_round)."""
        if ray_queue is None:
            return None
        reset_ray_queue()
        start_wall = time.perf_counter()
        if async_mode:
            prod_res = producer_group.run_async_ray_queue(ray_queue, cfg)
            cons_res = consumer_group.run_async_ray_queue(ray_queue, cfg)
        else:
            prod_res = producer_group.run_sync_ray_queue(ray_queue, cfg)
            cons_res = consumer_group.run_sync_ray_queue(ray_queue, cfg)
        prod_time = prod_res.wait()[0]
        cons_time = cons_res.wait()[0]
        wall = time.perf_counter() - start_wall
        msgs = float(cfg.num_messages)
        total_mb = (cfg.num_messages * cfg.payload_size) / (1024 * 1024)
        return {
            "producer_time": prod_time,
            "consumer_time": cons_time,
            "wall_time": wall,
            "throughput_msg_per_sec": msgs / wall if wall > 0 else float("inf"),
            "throughput_mb_per_sec": total_mb / wall if wall > 0 else float("inf"),
            "producer_latency_ms": (prod_time / msgs * 1e3)
            if msgs > 0
            else float("inf"),
            "consumer_latency_ms": (cons_time / msgs * 1e3)
            if msgs > 0
            else float("inf"),
        }

    def put_only_ray_queue_round(async_mode: bool) -> dict[str, float] | None:
        """Put-only on ray.util.queue.Queue: producer only."""
        if ray_queue is None:
            return None
        reset_ray_queue()
        start_wall = time.perf_counter()
        if async_mode:
            prod_res = producer_group.run_async_ray_queue(ray_queue, cfg)
        else:
            prod_res = producer_group.run_sync_ray_queue(ray_queue, cfg)
        prod_time = prod_res.wait()[0]
        wall = time.perf_counter() - start_wall
        msgs = float(cfg.num_messages)
        total_mb = (cfg.num_messages * cfg.payload_size) / (1024 * 1024)
        return {
            "producer_time": prod_time,
            "consumer_time": 0.0,
            "wall_time": wall,
            "throughput_msg_per_sec": msgs / wall if wall > 0 else float("inf"),
            "throughput_mb_per_sec": total_mb / wall if wall > 0 else float("inf"),
            "producer_latency_ms": (prod_time / msgs * 1e3)
            if msgs > 0
            else float("inf"),
            "consumer_latency_ms": 0.0,
        }

    def get_only_ray_queue_round(async_mode: bool) -> dict[str, float] | None:
        """Get-only on ray.util.queue.Queue (prefill then time consumer)."""
        if ray_queue is None:
            return None
        reset_ray_queue()
        producer_group.prefill_ray_queue(ray_queue, cfg).wait()
        start_wall = time.perf_counter()
        if async_mode:
            cons_res = consumer_group.run_async_ray_queue(ray_queue, cfg)
        else:
            cons_res = consumer_group.run_sync_ray_queue(ray_queue, cfg)
        cons_time = cons_res.wait()[0]
        wall = time.perf_counter() - start_wall
        msgs = float(cfg.num_messages)
        total_mb = (cfg.num_messages * cfg.payload_size) / (1024 * 1024)
        return {
            "consumer_time": cons_time,
            "wall_time": wall,
            "throughput_msg_per_sec": msgs / wall if wall > 0 else float("inf"),
            "throughput_mb_per_sec": total_mb / wall if wall > 0 else float("inf"),
            "consumer_latency_ms": (cons_time / msgs * 1e3)
            if msgs > 0
            else float("inf"),
        }

    def one_round(async_mode: bool) -> dict[str, float]:
        reset_channel()

        # Warmup both producer and consumer.
        producer_group.warmup(channel, cfg, async_mode).wait()
        consumer_group.warmup(channel, cfg, async_mode).wait()

        start_wall = time.perf_counter()
        if async_mode:
            prod_res = producer_group.run_async(channel, cfg)
            cons_res = consumer_group.run_async(channel, cfg)
        else:
            prod_res = producer_group.run_sync(channel, cfg)
            cons_res = consumer_group.run_sync(channel, cfg)

        prod_time = prod_res.wait()[0]
        cons_time = cons_res.wait()[0]
        wall = time.perf_counter() - start_wall

        msgs = float(cfg.num_messages)
        total_mb = (cfg.num_messages * cfg.payload_size) / (1024 * 1024)
        return {
            "producer_time": prod_time,
            "consumer_time": cons_time,
            "wall_time": wall,
            "throughput_msg_per_sec": msgs / wall if wall > 0 else float("inf"),
            "throughput_mb_per_sec": total_mb / wall if wall > 0 else float("inf"),
            "producer_latency_ms": (prod_time / msgs * 1e3)
            if msgs > 0
            else float("inf"),
            "consumer_latency_ms": (cons_time / msgs * 1e3)
            if msgs > 0
            else float("inf"),
        }

    def put_only_round(async_mode: bool) -> dict[str, float]:
        """Measure pure put performance: producer only, no consumer."""
        reset_channel()
        producer_group.warmup(channel, cfg, async_mode).wait()
        consumer_group.warmup(channel, cfg, async_mode).wait()
        reset_channel()

        start_wall = time.perf_counter()
        if async_mode:
            prod_res = producer_group.run_async(channel, cfg)
        else:
            prod_res = producer_group.run_sync(channel, cfg)
        prod_time = prod_res.wait()[0]
        wall = time.perf_counter() - start_wall

        msgs = float(cfg.num_messages)
        total_mb = (cfg.num_messages * cfg.payload_size) / (1024 * 1024)
        return {
            "producer_time": prod_time,
            "consumer_time": 0.0,
            "wall_time": wall,
            "throughput_msg_per_sec": msgs / wall if wall > 0 else float("inf"),
            "throughput_mb_per_sec": total_mb / wall if wall > 0 else float("inf"),
            "producer_latency_ms": (prod_time / msgs * 1e3)
            if msgs > 0
            else float("inf"),
            "consumer_latency_ms": 0.0,
        }

    def get_only_round(async_mode: bool) -> dict[str, float]:
        """Measure pure get performance on a pre-filled channel."""
        reset_channel()
        # Warmup on an empty channel.
        producer_group.warmup(channel, cfg, async_mode).wait()
        consumer_group.warmup(channel, cfg, async_mode).wait()
        # Ensure warmup traffic is drained.
        reset_channel()

        # Prefill all messages synchronously so consumer only measures get-side cost.
        producer_group.prefill(channel, cfg).wait()

        start_wall = time.perf_counter()
        if async_mode:
            cons_res = consumer_group.run_async(channel, cfg)
        else:
            cons_res = consumer_group.run_sync(channel, cfg)

        cons_time = cons_res.wait()[0]
        wall = time.perf_counter() - start_wall

        msgs = float(cfg.num_messages)
        total_mb = (cfg.num_messages * cfg.payload_size) / (1024 * 1024)
        return {
            "consumer_time": cons_time,
            "wall_time": wall,
            "throughput_msg_per_sec": msgs / wall if wall > 0 else float("inf"),
            "throughput_mb_per_sec": total_mb / wall if wall > 0 else float("inf"),
            "consumer_latency_ms": (cons_time / msgs * 1e3)
            if msgs > 0
            else float("inf"),
        }

    def random_key_round(async_mode: bool) -> dict[str, float]:
        """Put/get with a shuffled key schedule to stress key-based routing."""
        reset_channel()
        # Generate a random but deterministic key schedule shared by producer and consumer.
        num_distinct_keys = min(128, cfg.num_messages)
        keys = [i % num_distinct_keys for i in range(cfg.num_messages)]
        random.shuffle(keys)

        start_wall = time.perf_counter()
        if async_mode:
            prod_res = producer_group.run_async_keys(channel, cfg, keys)
            cons_res = consumer_group.run_async_keys(channel, cfg, keys)
        else:
            prod_res = producer_group.run_sync_keys(channel, cfg, keys)
            cons_res = consumer_group.run_sync_keys(channel, cfg, keys)

        prod_time = prod_res.wait()[0]
        cons_time = cons_res.wait()[0]
        wall = time.perf_counter() - start_wall

        msgs = float(cfg.num_messages)
        total_mb = (cfg.num_messages * cfg.payload_size) / (1024 * 1024)
        return {
            "producer_time": prod_time,
            "consumer_time": cons_time,
            "wall_time": wall,
            "throughput_msg_per_sec": msgs / wall if wall > 0 else float("inf"),
            "throughput_mb_per_sec": total_mb / wall if wall > 0 else float("inf"),
            "producer_latency_ms": (prod_time / msgs * 1e3)
            if msgs > 0
            else float("inf"),
            "consumer_latency_ms": (cons_time / msgs * 1e3)
            if msgs > 0
            else float("inf"),
        }

    print(f"Running channel pressure benchmark with config: {cfg}")

    print("\n[Start] Channel under pressure (sync)")
    sync_stats = one_round(async_mode=False)
    print("\n[Start] Channel under pressure (async)")
    async_stats = one_round(async_mode=True)

    # Put-only benchmark (producer only).
    print("\n[Start] Put-only benchmark (sync)")
    sync_put_stats = put_only_round(async_mode=False)
    print("\n[Start] Put-only benchmark (async)")
    async_put_stats = put_only_round(async_mode=True)

    # Get-only benchmark (channel is already full before measurement).
    print("\n[Start] Get-only benchmark (sync)")
    sync_get_stats = get_only_round(async_mode=False)
    print("\n[Start] Get-only benchmark (async)")
    async_get_stats = get_only_round(async_mode=True)

    # Random-key benchmark (key-based routing stress).
    print("\n[Start] Random-key benchmark (sync)")
    sync_randkey_stats = random_key_round(async_mode=False)
    print("\n[Start] Random-key benchmark (async)")
    async_randkey_stats = random_key_round(async_mode=True)

    # Ray util.queue.Queue comparison (same shared queue for all Ray rounds).
    sync_rayq_stats = None
    async_rayq_stats = None
    sync_rayq_put_stats = None
    async_rayq_put_stats = None
    sync_rayq_get_stats = None
    async_rayq_get_stats = None
    if ray_queue is not None:
        print("\n[Start] Ray util.queue.Queue under pressure (sync)")
        sync_rayq_stats = ray_queue_round(async_mode=False)
        print("\n[Start] Ray util.queue.Queue under pressure (async)")
        async_rayq_stats = ray_queue_round(async_mode=True)
        print("\n[Start] Ray util.queue.Queue put-only (sync)")
        sync_rayq_put_stats = put_only_ray_queue_round(async_mode=False)
        print("\n[Start] Ray util.queue.Queue put-only (async)")
        async_rayq_put_stats = put_only_ray_queue_round(async_mode=True)
        print("\n[Start] Ray util.queue.Queue get-only (sync)")
        sync_rayq_get_stats = get_only_ray_queue_round(async_mode=False)
        print("\n[Start] Ray util.queue.Queue get-only (async)")
        async_rayq_get_stats = get_only_ray_queue_round(async_mode=True)

    def fmt(s: dict[str, float]) -> str:
        return (
            f"producer={s['producer_time']:.4f}s, "
            f"consumer={s['consumer_time']:.4f}s, "
            f"wall={s['wall_time']:.4f}s, "
            f"throughput={s['throughput_msg_per_sec']:.1f} msg/s, "
            f"bandwidth={s['throughput_mb_per_sec']:.2f} MB/s, "
            f"producer_latency={s['producer_latency_ms']:.3f} ms/msg, "
            f"consumer_latency={s['consumer_latency_ms']:.3f} ms/msg"
        )

    def fmt_get(s: dict[str, float]) -> str:
        return (
            f"consumer={s['consumer_time']:.4f}s, "
            f"wall={s['wall_time']:.4f}s, "
            f"throughput={s['throughput_msg_per_sec']:.1f} msg/s, "
            f"bandwidth={s['throughput_mb_per_sec']:.2f} MB/s, "
            f"consumer_latency={s['consumer_latency_ms']:.3f} ms/msg"
        )

    print("\n=== Channel under pressure (sync) ===")
    print(fmt(sync_stats))
    print("\n=== Channel under pressure (async) ===")
    print(fmt(async_stats))
    print("\n=== Put-only benchmark (sync) ===")
    print(fmt(sync_put_stats))
    print("\n=== Put-only benchmark (async) ===")
    print(fmt(async_put_stats))
    print("\n=== Get-only benchmark (sync) ===")
    print(fmt_get(sync_get_stats))
    print("\n=== Get-only benchmark (async) ===")
    print(fmt_get(async_get_stats))
    print("\n=== Random-key benchmark (sync) ===")
    print(fmt(sync_randkey_stats))
    print("\n=== Random-key benchmark (async) ===")
    print(fmt(async_randkey_stats))

    # Ray util.queue.Queue comparison results.
    if sync_rayq_stats is not None:
        print("\n=== Ray util.queue.Queue under pressure (sync) ===")
        print(fmt(sync_rayq_stats))
    if async_rayq_stats is not None:
        print("\n=== Ray util.queue.Queue under pressure (async) ===")
        print(fmt(async_rayq_stats))
    if sync_rayq_put_stats is not None:
        print("\n=== Ray util.queue.Queue put-only (sync) ===")
        print(fmt(sync_rayq_put_stats))
    if async_rayq_put_stats is not None:
        print("\n=== Ray util.queue.Queue put-only (async) ===")
        print(fmt(async_rayq_put_stats))
    if sync_rayq_get_stats is not None:
        print("\n=== Ray util.queue.Queue get-only (sync) ===")
        print(fmt_get(sync_rayq_get_stats))
    if async_rayq_get_stats is not None:
        print("\n=== Ray util.queue.Queue get-only (async) ===")
        print(fmt_get(async_rayq_get_stats))


if __name__ == "__main__":
    run_benchmark(BenchmarkConfig())
