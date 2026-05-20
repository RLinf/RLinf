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

"""Broadcast bandwidth benchmark (intra-node or inter-node).

Topology:

  * Sender: one worker on node 0.
  * Receivers: ``--num-receivers`` workers (default 8).
    - 1-node cluster: receivers are colocated on node 0 (intra-node).
    - >=2-node cluster: receivers run on node 1 (inter-node).

The benchmark infers the node count from the already-running Ray cluster
(``Cluster()`` with auto-detect), so the only prerequisite is that
``ray start`` has been run on every node you want to include. Workers are
placed by node rank via ``NodePlacementStrategy``, so the benchmark works
on clusters with or without accelerators. The device is selected per run
via ``--device``: ``cpu`` (default), ``auto`` (use an accelerator if the
worker was allocated one, otherwise fall back to CPU), or an explicit
accelerator type such as ``cuda``.

The benchmark compares three strategies for fan-out from one sender to many
receivers:

  1. ``broadcast``: ``Worker.broadcast(...)`` collective (the default
     hybrid IPC/NCCL path).
  2. ``loop send/recv``: the sender issues an async ``send`` to each
     receiver and then awaits every handle; each receiver issues a single
     async ``recv`` and waits on it.
  3. ``ring``: ``Worker.broadcast(...)`` with
     ``CollectiveGroupOptions(use_ring_broadcast=True)``. The source sends
     each tensor to the first receiver; the first receiver async-broadcasts
     tensor K to the other receivers while it concurrently receives tensor
     K+1 from the source.

Note that ring broadcast is force-disabled when the source shares a device
with any receiver (the src->first-receiver hop relies on plain
``_send``/``_recv``, which NCCL forbids on the same GPU) and when the
payload is on the accelerator and every participant has the same accelerator
type and model (a native CCL collective is at least as good). Run on a
multi-node cluster, mixed-vendor cluster, or CPU payload to actually
exercise the ring path; otherwise the ``ring`` row falls back to the same
collective as ``broadcast``.

The ring algorithm only pipelines across tensors, so ``--num-tensors``
controls how many tensors the per-size payload is split into (and thus how
many recv/bcast steps overlap).

Before timing starts a one-shot warmup exchange is run so the collective
group (broadcast) and each point-to-point pair are established outside the
measurement window.

Prerequisite: a Ray cluster must already be running. For an intra-node test
``ray start --head`` on the local machine is enough; for an inter-node test
make sure at least two nodes are part of the cluster (see
``ray_utils/start_ray.sh`` and the multi-node instructions in ``AGENTS.md``).
The script attaches to the existing cluster via ``Cluster(num_nodes=...)``
and is launched only on the head node.

Examples
--------
CPU benchmark on the currently-attached Ray cluster (node count
auto-detected)::

    python tests/unit_tests/bench_broadcast.py

GPU broadcast with a custom sweep::

    python tests/unit_tests/bench_broadcast.py \\
        --device cuda --sizes 1MB,16MB,256MB,1GB --num-iters 20
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from rlinf.scheduler import (
    Cluster,
    CollectiveGroupOptions,
    NodePlacementStrategy,
    Worker,
)

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

SENDER_GROUP_NAME = "bench_broadcast_sender"
RECEIVER_GROUP_NAME = "bench_broadcast_receiver"

# Binary (1024-based) size units, mirroring bench_channel.py for consistency.
_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
}


def parse_size(s: str | int) -> int:
    """Parse a size string with optional unit (B, KB, MB, GB) into bytes.

    Examples
    --------
    >>> parse_size("1024")
    1024
    >>> parse_size("16MB")
    16777216
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


def format_size(num_bytes: int) -> str:
    """Render ``num_bytes`` with a binary unit suffix (B/KB/MB/GB)."""
    for unit, scale in (
        ("GB", _SIZE_UNITS["GB"]),
        ("MB", _SIZE_UNITS["MB"]),
        ("KB", _SIZE_UNITS["KB"]),
    ):
        if num_bytes >= scale:
            return f"{num_bytes / scale:.2f} {unit}"
    return f"{num_bytes} B"


@dataclass
class BenchmarkConfig:
    """Static benchmark configuration shared between the driver and workers."""

    sizes_bytes: list[int]
    num_iters: int
    num_warmup: int
    dtype: str
    device: str  # "cpu" | "cuda" | "auto"
    num_receivers: int
    num_tensors: int = 1
    """If > 1, split each size into a list of this many tensors. The ring
    broadcast pipelines recv-K with broadcast-(K-1), so more tensors -> more
    overlap. ``loop`` always uses a single tensor so it keeps the recv-side
    preallocation optimisation."""
    methods: tuple[str, ...] = ("broadcast", "loop", "ring")
    """Which fan-out strategies to run, in order."""


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


class _BroadcastBenchWorker(Worker):
    """Common base: resolve the requested device and stage tensors on it."""

    def __init__(self):
        super().__init__()
        self._device_obj: torch.device = torch.device("cpu")

    def set_device(self, device_spec: str) -> str:
        """Resolve and pin the per-worker device for this benchmark run.

        ``device_spec`` is one of ``"cpu"``, ``"auto"``, or an explicit
        accelerator type like ``"cuda"``. ``"auto"`` uses an accelerator if
        the worker was allocated one, otherwise falls back to CPU.
        """
        accel_available = self.has_accelerator and Worker.torch_platform is not None
        if device_spec == "cpu":
            self._device_obj = torch.device("cpu")
        elif device_spec == "auto":
            if accel_available:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                Worker.torch_platform.set_device(local_rank)
                self._device_obj = torch.device(
                    f"{Worker.torch_device_type}:"
                    f"{Worker.torch_platform.current_device()}"
                )
            else:
                self._device_obj = torch.device("cpu")
        else:
            if not accel_available:
                raise RuntimeError(
                    f"Device {device_spec!r} requested but this worker has no "
                    "accelerator. Use --device cpu or --device auto."
                )
            if device_spec != Worker.torch_device_type:
                raise RuntimeError(
                    f"Device {device_spec!r} requested but worker's accelerator "
                    f"backend is {Worker.torch_device_type!r}."
                )
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            Worker.torch_platform.set_device(local_rank)
            self._device_obj = torch.device(
                f"{device_spec}:{Worker.torch_platform.current_device()}"
            )
        return str(self._device_obj)

    def _device(self) -> torch.device:
        return self._device_obj

    def _sync(self) -> None:
        """Synchronize the local device so wall-clock timings include device work."""
        if (
            self._device_obj.type != "cpu"
            and self.has_accelerator
            and Worker.torch_platform is not None
        ):
            Worker.torch_platform.synchronize()

    def _make_payload(self, num_elements: int, dtype_name: str) -> torch.Tensor:
        dtype = _DTYPES[dtype_name]
        tensor = torch.empty(num_elements, dtype=dtype, device=self._device())
        if dtype.is_floating_point:
            tensor.fill_(1.0)
        else:
            tensor.fill_(1)
        return tensor

    def _make_list_payload(
        self, num_elements: int, dtype_name: str, num_tensors: int
    ) -> torch.Tensor | list[torch.Tensor]:
        """Return either a single tensor or a list of ``num_tensors`` tensors
        whose total element count equals ``num_elements`` (split as evenly as
        possible). Lists exercise the per-tensor recv/bcast pipeline in the
        ring path; single tensors keep parity with the existing benchmark
        shape.
        """
        if num_tensors <= 1:
            return self._make_payload(num_elements, dtype_name)
        base, rem = divmod(num_elements, num_tensors)
        sizes = [base + (1 if i < rem else 0) for i in range(num_tensors)]
        return [self._make_payload(s, dtype_name) for s in sizes if s > 0]

    @staticmethod
    def _broadcast_options(use_ring: bool) -> Optional[CollectiveGroupOptions]:
        if not use_ring:
            return None
        return CollectiveGroupOptions(use_ring_broadcast=True)


class SenderWorker(_BroadcastBenchWorker):
    """The sole broadcast source. Lives as a single worker on node 0."""

    def setup_warmup(self, num_receivers: int, prime_ring: bool = False) -> None:
        """One-shot communication that establishes every collective group.

        Runs a single tiny broadcast (forms the broadcast collective), one
        sync ``send`` to each receiver (forms each point-to-point
        collective), and – when ``prime_ring`` is True – one ring broadcast
        so the ring hop / dst sub-groups are created outside the timed
        window. After this call returns it is safe to use async P2P.
        """
        groups = [
            (SENDER_GROUP_NAME, [0]),
            (RECEIVER_GROUP_NAME, list(range(num_receivers))),
        ]
        tiny = self._make_payload(num_elements=1, dtype_name="float32")
        self.broadcast(tiny, groups=groups)
        for r in range(num_receivers):
            self.send(tiny, dst_group_name=RECEIVER_GROUP_NAME, dst_rank=r)
        if prime_ring:
            ring_opts = CollectiveGroupOptions(use_ring_broadcast=True)
            self.broadcast([tiny], groups=groups, options=ring_opts)
        self._sync()

    def run_broadcast(
        self,
        num_receivers: int,
        num_elements: int,
        dtype_name: str,
        num_warmup: int,
        num_iters: int,
        num_tensors: int = 1,
        use_ring_broadcast: bool = False,
    ) -> dict[str, Any]:
        """Run warmup + timed ``broadcast`` calls for a single tensor size.

        When ``use_ring_broadcast`` is True the call goes through the ring
        broadcast path (``CollectiveGroupOptions.use_ring_broadcast``). When
        ``num_tensors > 1`` the payload is a list of tensors so the ring's
        per-tensor recv/bcast pipeline can overlap consecutive tensors.
        """
        groups = [
            (SENDER_GROUP_NAME, [0]),
            (RECEIVER_GROUP_NAME, list(range(num_receivers))),
        ]
        payload = self._make_list_payload(num_elements, dtype_name, num_tensors)
        options = self._broadcast_options(use_ring_broadcast)

        for _ in range(num_warmup):
            self.broadcast(payload, groups=groups, options=options)
            self._sync()

        per_iter: list[float] = []
        for _ in range(num_iters):
            self._sync()
            t0 = time.perf_counter()
            self.broadcast(payload, groups=groups, options=options)
            self._sync()
            per_iter.append(time.perf_counter() - t0)

        return {"per_iter_seconds": per_iter}

    def run_loop_send_recv(
        self,
        num_receivers: int,
        num_elements: int,
        dtype_name: str,
        num_warmup: int,
        num_iters: int,
    ) -> dict[str, Any]:
        """Async ``send`` fan-out: send to each receiver, then await all."""
        payload = self._make_payload(num_elements, dtype_name)

        def one_round() -> None:
            handles = []
            for r in range(num_receivers):
                handles.append(
                    self.send(
                        payload,
                        dst_group_name=RECEIVER_GROUP_NAME,
                        dst_rank=r,
                        async_op=True,
                    )
                )
            for h in handles:
                h.wait()

        for _ in range(num_warmup):
            one_round()
            self._sync()

        per_iter: list[float] = []
        for _ in range(num_iters):
            self._sync()
            t0 = time.perf_counter()
            one_round()
            self._sync()
            per_iter.append(time.perf_counter() - t0)

        return {"per_iter_seconds": per_iter}


class ReceiverWorker(_BroadcastBenchWorker):
    """One of ``--num-receivers`` workers on node 1."""

    def setup_warmup(self, num_receivers: int, prime_ring: bool = False) -> None:
        """Mirror image of ``SenderWorker.setup_warmup``."""
        groups = [
            (SENDER_GROUP_NAME, [0]),
            (RECEIVER_GROUP_NAME, list(range(num_receivers))),
        ]
        self.broadcast(None, groups=groups)
        self.recv(src_group_name=SENDER_GROUP_NAME, src_rank=0)
        if prime_ring:
            ring_opts = CollectiveGroupOptions(use_ring_broadcast=True)
            self.broadcast(None, groups=groups, options=ring_opts)
        self._sync()

    def run_broadcast(
        self,
        num_receivers: int,
        num_elements: int,
        dtype_name: str,
        num_warmup: int,
        num_iters: int,
        num_tensors: int = 1,
        use_ring_broadcast: bool = False,
    ) -> dict[str, Any]:
        """Receiver-side mirror of ``SenderWorker.run_broadcast``.

        ``options`` must match the sender so the ring-vs-hybrid dispatch
        agrees across ranks (see ``CollectiveGroup.broadcast`` docstring).
        ``num_tensors`` itself is irrelevant on the receive side – the
        payload shape is exchanged via metadata – but ``use_ring_broadcast``
        must mirror the sender's option.
        """
        groups = [
            (SENDER_GROUP_NAME, [0]),
            (RECEIVER_GROUP_NAME, list(range(num_receivers))),
        ]
        del num_elements, dtype_name, num_tensors
        options = self._broadcast_options(use_ring_broadcast)

        for _ in range(num_warmup):
            self.broadcast(None, groups=groups, options=options)
            self._sync()

        per_iter: list[float] = []
        for _ in range(num_iters):
            self._sync()
            t0 = time.perf_counter()
            received = self.broadcast(None, groups=groups, options=options)
            self._sync()
            per_iter.append(time.perf_counter() - t0)
            del received

        return {"per_iter_seconds": per_iter, "rank": self._rank}

    def run_loop_send_recv(
        self,
        num_receivers: int,
        num_elements: int,
        dtype_name: str,
        num_warmup: int,
        num_iters: int,
    ) -> dict[str, Any]:
        """One async ``recv`` per iteration paired with the sender's send.

        ``recv`` (unlike ``recv_tensor``) does not require a preallocated
        buffer; the destination tensor is allocated inside the call from the
        sender's metadata, so the resulting timings include that allocation.
        """
        del (
            num_receivers,
            num_elements,
            dtype_name,
        )  # recv discovers shape/dtype itself.

        def one_round() -> None:
            h = self.recv(
                src_group_name=SENDER_GROUP_NAME,
                src_rank=0,
                async_op=True,
            )
            h.wait()

        for _ in range(num_warmup):
            one_round()
            self._sync()

        per_iter: list[float] = []
        for _ in range(num_iters):
            self._sync()
            t0 = time.perf_counter()
            one_round()
            self._sync()
            per_iter.append(time.perf_counter() - t0)

        return {"per_iter_seconds": per_iter, "rank": self._rank}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _summarise(
    per_iter_seconds: list[float], num_bytes_per_iter: int
) -> dict[str, float]:
    """Aggregate per-iteration timings into mean/min/max + bandwidth."""
    if not per_iter_seconds:
        return {
            "mean_s": float("nan"),
            "min_s": float("nan"),
            "max_s": float("nan"),
            "mean_gbps": float("nan"),
        }
    mean_s = sum(per_iter_seconds) / len(per_iter_seconds)
    min_s = min(per_iter_seconds)
    max_s = max(per_iter_seconds)
    if mean_s > 0:
        mean_gbps = (num_bytes_per_iter / mean_s) / _SIZE_UNITS["GB"]
    else:
        mean_gbps = float("inf")
    return {
        "mean_s": mean_s,
        "min_s": min_s,
        "max_s": max_s,
        "mean_gbps": mean_gbps,
    }


def _print_row(
    label: str,
    role: str,
    summary: dict[str, float],
    size_str: str,
    num_iters: int,
    wall_elapsed: Optional[float] = None,
) -> None:
    parts = [
        f"{label:<10s}",
        f"{role:<10s}",
        f"{size_str:>10s}",
        f"iters={num_iters:<4d}",
        f"mean={summary['mean_s'] * 1e3:>9.3f} ms",
        f"min={summary['min_s'] * 1e3:>9.3f} ms",
        f"max={summary['max_s'] * 1e3:>9.3f} ms",
        f"BW={summary['mean_gbps']:>7.3f} GB/s",
    ]
    if wall_elapsed is not None:
        parts.append(f"wall={wall_elapsed:>7.3f} s")
    print(" | ".join(parts))


_METHOD_DISPATCH: dict[str, dict[str, Any]] = {
    "broadcast": {"fn": "run_broadcast", "use_ring": False, "list_payload": True},
    "ring": {"fn": "run_broadcast", "use_ring": True, "list_payload": True},
    "loop": {"fn": "run_loop_send_recv", "use_ring": False, "list_payload": False},
}


def _run_one_method(
    method: str,
    sender_group: Any,
    receiver_group: Any,
    cfg: BenchmarkConfig,
    num_elements: int,
    actual_bytes: int,
    size_str: str,
) -> None:
    """Drive a single benchmark method (``broadcast``, ``ring``, or ``loop``) for one size."""
    if method not in _METHOD_DISPATCH:
        raise ValueError(f"Unknown benchmark method: {method!r}")
    spec = _METHOD_DISPATCH[method]
    fn_name = spec["fn"]

    # Only the broadcast/ring path accepts ring options + tensor lists; the
    # loop method always uses a single tensor (the recv side uses ``recv``
    # and allocates its destination buffer inside the call).
    if spec["fn"] == "run_broadcast":
        extra_kwargs: dict[str, Any] = {
            "num_tensors": cfg.num_tensors,
            "use_ring_broadcast": spec["use_ring"],
        }
    else:
        extra_kwargs = {}

    wall_start = time.perf_counter()
    sender_handle = getattr(sender_group, fn_name)(
        cfg.num_receivers,
        num_elements,
        cfg.dtype,
        cfg.num_warmup,
        cfg.num_iters,
        **extra_kwargs,
    )
    receiver_handle = getattr(receiver_group, fn_name)(
        cfg.num_receivers,
        num_elements,
        cfg.dtype,
        cfg.num_warmup,
        cfg.num_iters,
        **extra_kwargs,
    )
    sender_result = sender_handle.wait()[0]
    receiver_results = receiver_handle.wait()
    wall_elapsed = time.perf_counter() - wall_start

    sender_summary = _summarise(sender_result["per_iter_seconds"], actual_bytes)
    receiver_per_iter_flat: list[float] = []
    for r in receiver_results:
        receiver_per_iter_flat.extend(r["per_iter_seconds"])
    receiver_summary = _summarise(receiver_per_iter_flat, actual_bytes)

    _print_row(method, "sender", sender_summary, size_str, cfg.num_iters, wall_elapsed)
    _print_row(
        method,
        f"recv x{cfg.num_receivers}",
        receiver_summary,
        size_str,
        cfg.num_iters,
    )


def run_benchmark(cfg: BenchmarkConfig) -> None:
    """Launch worker groups, run a setup warmup, and sweep over tensor sizes.

    The cluster size is auto-detected from the already-running Ray cluster
    (``Cluster(num_nodes=0)``), so the benchmark uses whichever nodes have
    joined via ``ray start``.
    """
    cluster = Cluster(num_nodes=0)

    if cluster.num_nodes < 1:
        raise SystemExit(
            "bench_broadcast requires at least 1 node; got "
            f"num_nodes={cluster.num_nodes}."
        )

    # On a single-node cluster, both sender and receivers live on node 0
    # (intra-node test). With >=2 nodes, receivers move to node 1.
    receiver_node_rank = 0 if cluster.num_nodes == 1 else 1
    sender_placement = NodePlacementStrategy(node_ranks=[0])
    receiver_placement = NodePlacementStrategy(
        node_ranks=[receiver_node_rank] * cfg.num_receivers,
    )

    sender_group = SenderWorker.create_group().launch(
        cluster=cluster,
        name=SENDER_GROUP_NAME,
        placement_strategy=sender_placement,
    )
    receiver_group = ReceiverWorker.create_group().launch(
        cluster=cluster,
        name=RECEIVER_GROUP_NAME,
        placement_strategy=receiver_placement,
    )

    sender_devices = sender_group.set_device(cfg.device).wait()
    receiver_devices = receiver_group.set_device(cfg.device).wait()
    print(
        f"[setup] requested device={cfg.device!r}; "
        f"sender devices={sender_devices}, receiver devices={receiver_devices}"
    )

    # --- One-shot setup warmup to establish every collective group ---------
    prime_ring = "ring" in cfg.methods
    print(
        "[setup] establishing broadcast and point-to-point collective groups"
        + (" (and ring sub-groups)" if prime_ring else "")
        + "..."
    )
    warmup_start = time.perf_counter()
    sender_warmup = sender_group.setup_warmup(cfg.num_receivers, prime_ring=prime_ring)
    receiver_warmup = receiver_group.setup_warmup(
        cfg.num_receivers, prime_ring=prime_ring
    )
    sender_warmup.wait()
    receiver_warmup.wait()
    print(f"[setup] done in {time.perf_counter() - warmup_start:.3f} s")

    dtype = _DTYPES[cfg.dtype]
    bytes_per_element = dtype.itemsize

    topology = (
        "intra-node (all on node 0)"
        if cluster.num_nodes == 1
        else f"inter-node (sender on node 0, receivers on node {receiver_node_rank})"
    )
    header = (
        f"\nBroadcast / ring / loop-send-recv benchmark: 1 sender -> "
        f"{cfg.num_receivers} receivers, {topology}\n"
        f"dtype={cfg.dtype}, device={cfg.device}, "
        f"num_tensors={cfg.num_tensors}, "
        f"per-size warmup={cfg.num_warmup}, "
        f"per-size iters={cfg.num_iters}"
    )
    print(header)
    print("-" * 100)

    for size_bytes in cfg.sizes_bytes:
        num_elements = max(1, size_bytes // bytes_per_element)
        actual_bytes = num_elements * bytes_per_element
        size_str = format_size(actual_bytes)

        for method in cfg.methods:
            _run_one_method(
                method,
                sender_group,
                receiver_group,
                cfg,
                num_elements,
                actual_bytes,
                size_str,
            )
        print("-" * 100)

    sender_group._close()
    receiver_group._close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the broadcast benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fan-out from 1 sender to N receivers: broadcast vs "
            "ring vs loop send/recv. The node count is auto-detected from "
            "the already-running Ray cluster: on a 1-node cluster all "
            "workers run on node 0 (intra-node test); on a >=2-node "
            "cluster the sender stays on node 0 and the receivers move to "
            "node 1 (inter-node test)."
        )
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="1KB,16KB,256KB,1MB,16MB,64MB,256MB,1GB",
        help=(
            "Comma-separated list of tensor sizes. Each entry supports "
            "B/KB/MB/GB suffixes (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of timed iterations per (size, method) (default: %(default)d).",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=1,
        help=(
            "Per-size untimed warmup iterations per method "
            "(default: %(default)d). A one-shot setup warmup is always run "
            "before the size sweep to establish collective groups."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=sorted(_DTYPES.keys()),
        help="Tensor dtype (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Tensor device: 'cpu', 'auto' (accelerator if the worker has one, "
            "else cpu), or an explicit accelerator type like 'cuda' "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--num-receivers",
        type=int,
        default=8,
        help="Number of receiver workers on node 1 (default: %(default)d).",
    )
    parser.add_argument(
        "--num-tensors",
        type=int,
        default=1,
        help=(
            "Split each size into this many tensors. >1 lets the ring "
            "broadcast overlap recv-K with broadcast-(K-1) "
            "(default: %(default)d)."
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="broadcast,ring,loop",
        help=(
            "Comma-separated subset of {broadcast, ring, loop}, in the "
            "order they should run per size (default: %(default)s)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the broadcast benchmark."""
    args = parse_args()

    sizes_bytes = [parse_size(s) for s in args.sizes.split(",") if s.strip()]
    if not sizes_bytes:
        raise SystemExit("--sizes must contain at least one entry.")

    methods_tuple = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    if not methods_tuple:
        raise SystemExit("--methods must contain at least one entry.")
    unknown = [m for m in methods_tuple if m not in _METHOD_DISPATCH]
    if unknown:
        raise SystemExit(
            f"Unknown method(s) {unknown!r}; choose from {sorted(_METHOD_DISPATCH)}."
        )

    if args.num_tensors < 1:
        raise SystemExit("--num-tensors must be >= 1.")

    cfg = BenchmarkConfig(
        sizes_bytes=sizes_bytes,
        num_iters=args.num_iters,
        num_warmup=args.num_warmup,
        dtype=args.dtype,
        device=args.device,
        num_receivers=args.num_receivers,
        num_tensors=args.num_tensors,
        methods=methods_tuple,
    )
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
