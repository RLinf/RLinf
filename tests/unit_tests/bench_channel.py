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

import argparse
import asyncio
import random
import re
import threading
import time
from dataclasses import dataclass, field
from queue import Empty as QueueEmpty
from typing import Any

import torch
from ray.util.queue import Queue as RayQueue

from rlinf.scheduler import Channel, Cluster, NodePlacementStrategy, Worker

# Available test names for --tests selection.
AVAILABLE_TESTS = frozenset(
    {
        "channel",  # Channel under pressure (sync + async)
        "put_only",
        "get_only",
        "random_key",
    }
)

# Payload types for --payload-type.
PAYLOAD_TYPES = frozenset(
    {
        "bytes",
        "cpu_tensor",
        "gpu_tensor",
        "tensor_list",
        "tensor_dict",
        "tensor_dataclass",
        "tensor_list_dataclass",
        "tensor_dict_dataclass",
    }
)

# Size units (binary: 1024-based)
_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


def parse_size(s: str | int) -> int:
    """Parse a size string with optional unit (B, KB, MB, GB) into bytes.

    Examples: "1024", "1KB", "64KB", "1MB", "1GB"
    """
    if isinstance(s, int):
        return s
    s = str(s).strip().upper()
    if not s:
        raise ValueError("Empty size string")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([KMG]?B?)$", s)
    if not m:
        raise ValueError(f"Invalid size format: {s!r}. Use e.g. 1024, 1KB, 1MB, 1GB")
    num_str, unit = m.groups()
    num = float(num_str)
    unit = unit or "B"
    if unit == "K":
        unit = "KB"
    elif unit == "M":
        unit = "MB"
    elif unit == "G":
        unit = "GB"
    if unit not in _SIZE_UNITS:
        raise ValueError(f"Unknown unit: {unit}. Use B, KB, MB, GB")
    return int(num * _SIZE_UNITS[unit])


@dataclass
class TensorPayload:
    """Dataclass with a tensor field for benchmarking optimized dataclass put/get."""

    id: int
    payload: torch.Tensor
    note: str


@dataclass
class TensorListPayload:
    """Dataclass with a list of tensors for benchmarking channel put/get."""

    id: int
    payload_list: list
    note: str


@dataclass
class TensorDictPayload:
    """Dataclass with a dict of tensors for benchmarking channel put/get."""

    id: int
    payload_dict: dict
    note: str


@dataclass
class BenchmarkConfig:
    num_messages: int = 2000
    num_warmup_messages: int = 2
    payload_size: int = 1024 * 1024  # bytes (or approximate for tensors)
    channel_maxsize: int = 0
    enable_thread_interference: bool = False
    num_noise_threads: int = 2
    payload_type: str = "bytes"  # bytes | cpu_tensor | gpu_tensor | tensor_list | tensor_dict | tensor_dataclass | tensor_list_dataclass | tensor_dict_dataclass
    payload_device: str = "auto"  # "auto" (cuda if available else cpu) | "cpu" | "cuda"
    enabled_tests: frozenset[str] = field(default_factory=lambda: AVAILABLE_TESTS)
    enable_ray_queue: bool = False  # Run ray.util.queue.Queue comparison for same tests


def _resolve_device(device: str) -> str:
    """Resolve device: 'auto' -> 'cuda' if available else 'cpu'; 'cpu'/'cuda' as-is."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device cuda requested but CUDA is not available")
    return device


def _create_tensor_payload(
    payload_type: str,
    size_bytes: int,
    payload_device: str = "auto",
) -> (
    torch.Tensor
    | list[torch.Tensor]
    | dict[str, torch.Tensor]
    | TensorPayload
    | TensorListPayload
    | TensorDictPayload
):
    """Create a tensor payload of given type and approximate size in bytes.

    For tensor_list, tensor_dict, tensor_dataclass, tensor_list_dataclass, and
    tensor_dict_dataclass, device controls where tensors live: 'auto' (cuda if
    available else cpu), 'cpu', or 'cuda'. cpu_tensor and gpu_tensor ignore
    device and always use cpu/cuda.
    """
    payload_device = _resolve_device(payload_device)
    torch_device = torch.device(payload_device) if payload_device == "cuda" else "cpu"

    # float32: 4 bytes per element
    num_elements = max(1, size_bytes // 4)
    shape = (num_elements,)

    if payload_type == "cpu_tensor":
        return torch.ones(shape, dtype=torch.float32, device="cpu")

    if payload_type == "gpu_tensor":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU tensor benchmark requested but CUDA is not available"
            )
        return torch.ones(
            shape, dtype=torch.float32, device=torch.device("cuda")
        ).contiguous()

    if payload_type == "tensor_list":
        n = max(1, num_elements // 8)
        return [
            torch.ones((n,), dtype=torch.float32, device=torch_device).contiguous()
            for _ in range(8)
        ]

    if payload_type == "tensor_dict":
        n = max(1, num_elements // 8)
        return {
            f"t{i}": torch.ones(
                (n,), dtype=torch.float32, device=torch_device
            ).contiguous()
            for i in range(8)
        }

    if payload_type == "tensor_dataclass":
        tensor = torch.ones(
            shape, dtype=torch.float32, device=torch_device
        ).contiguous()
        return TensorPayload(id=0, payload=tensor, note="bench")

    if payload_type == "tensor_list_dataclass":
        payload_list = [
            torch.ones(
                (num_elements,), dtype=torch.float32, device=torch_device
            ).contiguous()
        ]
        return TensorListPayload(id=0, payload_list=payload_list, note="bench")

    if payload_type == "tensor_dict_dataclass":
        payload_dict = {
            "data": torch.ones(
                (num_elements,), dtype=torch.float32, device=torch_device
            ).contiguous()
        }
        return TensorDictPayload(id=0, payload_dict=payload_dict, note="bench")

    raise ValueError(f"Unknown payload_type: {payload_type}")


class Producer(Worker):
    def __init__(self):
        super().__init__()
        self._noise_started = False

    def _get_payload(self, cfg: BenchmarkConfig) -> Any:
        """Create payload based on config payload_type and device."""
        if cfg.payload_type == "bytes":
            return b"x" * cfg.payload_size
        return _create_tensor_payload(
            cfg.payload_type, cfg.payload_size, cfg.payload_device
        )

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
        payload = self._get_payload(cfg)
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
        payload = 1
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
        payload = self._get_payload(cfg)
        with self.worker_timer("producer_sync"):
            for i in range(cfg.num_messages):
                channel.put(payload)
                self._progress(i, cfg.num_messages, "put  sync ")
        return self.pop_execution_time("producer_sync")

    def run_async(self, channel: Channel, cfg: BenchmarkConfig) -> float:
        """Async put using asyncio: await put(..., async_op=True).async_wait()."""
        payload = self._get_payload(cfg)

        async def _run() -> None:
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
        payload = self._get_payload(cfg)
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
        payload = self._get_payload(cfg)

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
        payload = self._get_payload(cfg)
        for _ in range(cfg.num_messages):
            queue.put(payload)
        return cfg.num_messages

    def run_sync_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> float:
        """Synchronous put on ray.util.queue.Queue."""
        if RayQueue is None:
            return 0.0
        payload = self._get_payload(cfg)
        with self.worker_timer("producer_sync_ray_queue"):
            for i in range(cfg.num_messages):
                queue.put(payload)
                self._progress(i, cfg.num_messages, "rayQ put ")
        return self.pop_execution_time("producer_sync_ray_queue")

    def run_async_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> float:
        """Async put on ray.util.queue.Queue (put_async returns coroutine)."""
        if RayQueue is None:
            return 0.0
        payload = self._get_payload(cfg)

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
    ray_queue = RayQueue(
        maxsize=cfg.channel_maxsize or 0,
        actor_options={
            "runtime_env": {
                "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
            }
        },
    )

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

    enabled = cfg.enabled_tests

    # Results storage
    sync_stats = async_stats = None
    sync_put_stats = async_put_stats = None
    sync_get_stats = async_get_stats = None
    sync_randkey_stats = async_randkey_stats = None
    sync_rayq_stats = async_rayq_stats = None
    sync_rayq_put_stats = async_rayq_put_stats = None
    sync_rayq_get_stats = async_rayq_get_stats = None

    print(f"Running channel pressure benchmark with config: {cfg}")
    payload_info = f"payload_type: {cfg.payload_type}"
    if cfg.payload_type in (
        "tensor_list",
        "tensor_dict",
        "tensor_dataclass",
        "tensor_list_dataclass",
        "tensor_dict_dataclass",
    ):
        payload_info += f", device: {cfg.payload_device}"
    print(
        f"Enabled tests: {sorted(enabled)}, {payload_info}"
        + (
            ", ray_queue comparison enabled"
            if cfg.enable_ray_queue and ray_queue
            else ""
        )
    )

    # Ray util.queue.Queue comparison runs for same tests when enabled
    run_ray_queue = cfg.enable_ray_queue and ray_queue is not None

    # Channel under pressure
    if "channel" in enabled:
        print("\n[Start] Channel under pressure (sync)")
        sync_stats = one_round(async_mode=False)
        print("\n[Start] Channel under pressure (async)")
        async_stats = one_round(async_mode=True)

    # Put-only benchmark (producer only).
    if "put_only" in enabled:
        print("\n[Start] Put-only benchmark (sync)")
        sync_put_stats = put_only_round(async_mode=False)
        print("\n[Start] Put-only benchmark (async)")
        async_put_stats = put_only_round(async_mode=True)

    # Get-only benchmark (channel is already full before measurement).
    if "get_only" in enabled:
        print("\n[Start] Get-only benchmark (sync)")
        sync_get_stats = get_only_round(async_mode=False)
        print("\n[Start] Get-only benchmark (async)")
        async_get_stats = get_only_round(async_mode=True)

    # Random-key benchmark (key-based routing stress).
    if "random_key" in enabled:
        print("\n[Start] Random-key benchmark (sync)")
        sync_randkey_stats = random_key_round(async_mode=False)
        print("\n[Start] Random-key benchmark (async)")
        async_randkey_stats = random_key_round(async_mode=True)

    # Ray util.queue.Queue comparison for same tests when enabled.
    if run_ray_queue and "channel" in enabled:
        print("\n[Start] Ray util.queue.Queue under pressure (sync)")
        sync_rayq_stats = ray_queue_round(async_mode=False)
        print("\n[Start] Ray util.queue.Queue under pressure (async)")
        async_rayq_stats = ray_queue_round(async_mode=True)
    if run_ray_queue and "put_only" in enabled:
        print("\n[Start] Ray util.queue.Queue put-only (sync)")
        sync_rayq_put_stats = put_only_ray_queue_round(async_mode=False)
        print("\n[Start] Ray util.queue.Queue put-only (async)")
        async_rayq_put_stats = put_only_ray_queue_round(async_mode=True)
    if run_ray_queue and "get_only" in enabled:
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

    if sync_stats is not None:
        print("\n=== Channel under pressure (sync) ===")
        print(fmt(sync_stats))
    if async_stats is not None:
        print("\n=== Channel under pressure (async) ===")
        print(fmt(async_stats))
    if sync_put_stats is not None:
        print("\n=== Put-only benchmark (sync) ===")
        print(fmt(sync_put_stats))
    if async_put_stats is not None:
        print("\n=== Put-only benchmark (async) ===")
        print(fmt(async_put_stats))
    if sync_get_stats is not None:
        print("\n=== Get-only benchmark (sync) ===")
        print(fmt_get(sync_get_stats))
    if async_get_stats is not None:
        print("\n=== Get-only benchmark (async) ===")
        print(fmt_get(async_get_stats))
    if sync_randkey_stats is not None:
        print("\n=== Random-key benchmark (sync) ===")
        print(fmt(sync_randkey_stats))
    if async_randkey_stats is not None:
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark configuration and test selection."""
    parser = argparse.ArgumentParser(
        description="Channel pressure benchmark: measure put/get throughput for bytes and tensors."
    )
    parser.add_argument(
        "--tests",
        "-t",
        type=str,
        default=None,
        help=(
            "Comma-separated list of tests to run. "
            f"Available: {', '.join(sorted(AVAILABLE_TESTS))}. "
            "Default: run all tests."
        ),
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="Print available test names and exit.",
    )
    parser.add_argument(
        "--num-messages",
        type=int,
        default=2000,
        help="Number of messages (default: 2000).",
    )
    parser.add_argument(
        "--payload-size",
        type=parse_size,
        default="1MB",
        help="Payload size with optional unit: B, KB, MB, GB (default: 1MB).",
    )
    parser.add_argument(
        "--payload-type",
        type=str,
        default="bytes",
        choices=sorted(PAYLOAD_TYPES),
        help=f"Payload type: {', '.join(sorted(PAYLOAD_TYPES))}.",
    )
    parser.add_argument(
        "--payload-device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help=(
            "Device for tensor payloads (tensor_list, tensor_dict, tensor_dataclass, "
            "tensor_list_dataclass, tensor_dict_dataclass): auto (cuda if available "
            "else cpu), cpu, or cuda (default: auto). cpu_tensor and gpu_tensor "
            "ignore this and always use cpu/cuda."
        ),
    )
    parser.add_argument(
        "--channel-maxsize",
        type=int,
        default=0,
        help="Channel max size, 0 = unbounded (default: 0).",
    )
    parser.add_argument(
        "--enable-thread-interference",
        action="store_true",
        help="Enable background CPU-burning threads to stress test under interference.",
    )
    parser.add_argument(
        "--num-noise-threads",
        type=int,
        default=2,
        help="Number of noise threads per worker (default: 2).",
    )
    parser.add_argument(
        "--ray-queue",
        action="store_true",
        help="Run ray.util.queue.Queue comparison for the same tests (channel, put_only, get_only).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the channel benchmark."""
    args = parse_args()

    if args.list_tests:
        print("Available tests:")
        for name in sorted(AVAILABLE_TESTS):
            print(f"  {name}")
        return

    enabled_tests = AVAILABLE_TESTS
    if args.tests is not None:
        requested = {s.strip() for s in args.tests.split(",") if s.strip()}
        invalid = requested - AVAILABLE_TESTS
        if invalid:
            raise SystemExit(
                f"Unknown test(s): {invalid}. Available: {', '.join(sorted(AVAILABLE_TESTS))}"
            )
        enabled_tests = requested

    if args.payload_type == "gpu_tensor" and not torch.cuda.is_available():
        raise SystemExit("Payload type gpu_tensor requested but CUDA is not available.")
    if args.payload_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Device cuda requested but CUDA is not available.")

    payload_size = parse_size(args.payload_size)

    cfg = BenchmarkConfig(
        num_messages=args.num_messages,
        num_warmup_messages=2,
        payload_size=payload_size,
        channel_maxsize=args.channel_maxsize,
        enable_thread_interference=args.enable_thread_interference,
        num_noise_threads=args.num_noise_threads,
        payload_type=args.payload_type,
        payload_device=args.payload_device,
        enabled_tests=enabled_tests,
        enable_ray_queue=args.ray_queue,
    )
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
