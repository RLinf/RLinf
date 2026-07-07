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

import dataclasses
import math
import os
import pickle
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import ray
import torch
from omegaconf import DictConfig, OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from ..cluster import Cluster

# Rough per-tensor protocol overhead for shape, dtype, and size metadata.
_TENSOR_METADATA_OVERHEAD = 256
_DEFAULT_PROXY_NAME = "NetEmulationProxy"
_DEFAULT_PROXY_NUM_CPUS = 1
_DEFAULT_PROXY_POLL_INTERVAL_MS = 1
_DEFAULT_SYMMETRIC_LINKS = True
_MEGABITS_TO_BYTES = 1_000_000.0 / 8.0
_MILLISECONDS_TO_SECONDS = 1000.0


def _contains_tensor(payload: Any) -> bool:
    """Check whether *payload* or any nested element contains a torch.Tensor."""
    if isinstance(payload, torch.Tensor):
        return True
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        return any(
            _contains_tensor(getattr(payload, f.name))
            for f in dataclasses.fields(payload)
        )
    if isinstance(payload, Mapping):
        return any(_contains_tensor(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_contains_tensor(item) for item in payload)
    return False


def _count_tensors(payload: Any) -> int:
    """Count the number of tensor leaves inside *payload*."""
    if isinstance(payload, torch.Tensor):
        return 1
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        return sum(
            _count_tensors(getattr(payload, f.name))
            for f in dataclasses.fields(payload)
        )
    if isinstance(payload, Mapping):
        return sum(_count_tensors(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(_count_tensors(item) for item in payload)
    return 0


def _pickle_part_size(obj: Any) -> int:
    """Estimate the wire size of a non-tensor object via pickle."""
    if obj is None:
        return 0
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return max(1, len(repr(obj)))


def estimate_payload_size_bytes(payload: Any) -> int:
    """Estimate payload size using tensor bytes plus pickle'd metadata."""
    if payload is None:
        return 0

    if isinstance(payload, torch.Tensor):
        return payload.numel() * payload.element_size()

    if _contains_tensor(payload):
        tensor_data_size = _estimate_tensor_data_size(payload)
        num_tensors = _count_tensors(payload)
        metadata_size = _estimate_metadata_size(payload)
        return (
            tensor_data_size + num_tensors * _TENSOR_METADATA_OVERHEAD + metadata_size
        )

    return _pickle_part_size(payload)


def _estimate_tensor_data_size(payload: Any) -> int:
    """Sum of raw tensor data sizes (no metadata) inside *payload*."""
    if isinstance(payload, torch.Tensor):
        return payload.numel() * payload.element_size()
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        return sum(
            _estimate_tensor_data_size(getattr(payload, f.name))
            for f in dataclasses.fields(payload)
        )
    if isinstance(payload, Mapping):
        return sum(_estimate_tensor_data_size(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(_estimate_tensor_data_size(item) for item in payload)
    return 0


def _estimate_metadata_size(payload: Any) -> int:
    """Estimate the size of non-tensor metadata (keys, struct info, piggyback, etc.)."""
    if isinstance(payload, torch.Tensor):
        return 0
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        fields = dataclasses.fields(payload)
        field_names_size = _pickle_part_size([f.name for f in fields])
        fields_meta = sum(
            _estimate_metadata_size(getattr(payload, f.name)) for f in fields
        )
        return field_names_size + fields_meta
    if isinstance(payload, Mapping):
        keys_size = _pickle_part_size(list(payload.keys()))
        values_meta = sum(_estimate_metadata_size(v) for v in payload.values())
        return keys_size + values_meta
    if isinstance(payload, (list, tuple)):
        return sum(_estimate_metadata_size(item) for item in payload)
    return _pickle_part_size(payload)


@dataclass(frozen=True)
class ProxyConfig:
    """Configuration used to launch the global net-emulation proxy actor."""

    node_rank: int | None
    num_cpus: int
    name: str
    poll_interval_ms: int = _DEFAULT_PROXY_POLL_INTERVAL_MS


@dataclass(frozen=True)
class CrossDCPair:
    """One emulated directed link between a source and destination endpoint."""

    src: str
    dst: str
    delay_ms: float


@dataclass(frozen=True)
class BandwidthGroup:
    """Endpoints that share the same emulated bandwidth budget."""

    members: tuple[str, ...]
    bandwidth_mbps: float


@dataclass(frozen=True)
class NetEmulationConfig:
    """Top-level configuration for application-level network emulation."""

    enabled: bool
    symmetric: bool
    proxy: ProxyConfig
    crossdc_pairs: tuple[CrossDCPair, ...]
    bandwidth_groups: tuple[BandwidthGroup, ...]

    @classmethod
    def from_cfg(
        cls, cfg: DictConfig | dict[str, Any] | None
    ) -> "NetEmulationConfig | None":
        """Build a normalized config from a Hydra/OmegaConf or plain dict."""
        if cfg is None:
            return None
        cfg_dict = (
            OmegaConf.to_container(cfg, resolve=True)
            if isinstance(cfg, DictConfig)
            else cfg
        )
        if not isinstance(cfg_dict, dict):
            return None
        if not cfg_dict.get("enabled", False):
            return None

        proxy_dict = cfg_dict.get("proxy") or {}
        proxy_node_rank = proxy_dict.get("node_rank")
        proxy = ProxyConfig(
            node_rank=int(proxy_node_rank) if proxy_node_rank is not None else None,
            num_cpus=int(proxy_dict.get("num_cpus", _DEFAULT_PROXY_NUM_CPUS)),
            name=str(proxy_dict.get("name", _DEFAULT_PROXY_NAME)),
            poll_interval_ms=int(
                proxy_dict.get("poll_interval_ms", _DEFAULT_PROXY_POLL_INTERVAL_MS)
            ),
        )
        crossdc_pairs: list[CrossDCPair] = []
        for item in cfg_dict.get("crossdc_pairs", []):
            src_endpoints = cls._expand_endpoints(item["src"], field_name="src")
            dst_endpoints = cls._expand_endpoints(item["dst"], field_name="dst")
            delay_ms = float(item["delay_ms"])
            for src in src_endpoints:
                for dst in dst_endpoints:
                    crossdc_pairs.append(
                        CrossDCPair(
                            src=src,
                            dst=dst,
                            delay_ms=delay_ms,
                        )
                    )
        bandwidth_groups = tuple(
            BandwidthGroup(
                members=tuple(str(member) for member in item["members"]),
                bandwidth_mbps=float(item["bandwidth_mbps"]),
            )
            for item in cfg_dict.get("bandwidth_groups", [])
        )
        return cls(
            enabled=True,
            symmetric=bool(cfg_dict.get("symmetric", _DEFAULT_SYMMETRIC_LINKS)),
            proxy=proxy,
            crossdc_pairs=tuple(crossdc_pairs),
            bandwidth_groups=bandwidth_groups,
        )

    @staticmethod
    def _expand_endpoints(
        value: str | list[Any] | tuple[Any, ...], field_name: str
    ) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple)):
            endpoints = tuple(str(item) for item in value)
            if endpoints:
                return endpoints
        raise ValueError(
            "net_emulation.crossdc_pairs entries must define a non-empty "
            f"string or list for '{field_name}'"
        )

    def to_proxy_payload(self) -> dict[str, Any]:
        """Serialize the config into the proxy actor payload format."""
        return {
            "symmetric": self.symmetric,
            "crossdc_pairs": [dataclasses.asdict(pair) for pair in self.crossdc_pairs],
            "bandwidth_groups": [
                dataclasses.asdict(group) for group in self.bandwidth_groups
            ],
        }


@dataclass
class _ReservationRequest:
    src: str
    dst: str
    size_bytes: int
    request_time: float
    done: threading.Event
    ready_at: float | None = None
    error: Exception | None = None


class NetEmulationProxy:
    """Centralized control-plane scheduler for emulated network permissions."""

    _MIN_POLL_INTERVAL_SECONDS = 0.0005
    _RESERVATION_POLL_INTERVAL_SECONDS = 0.1

    @staticmethod
    def _normalize_endpoint(name: str) -> str:
        """Normalize endpoint names by stripping a trailing ``Group`` suffix."""
        parts = name.split(":", 1)
        group = parts[0]
        if group.endswith("Group"):
            group = group[: -len("Group")]
        return group + (":" + parts[1] if len(parts) > 1 else "")

    def _ready(self) -> None:
        return None

    def __init__(
        self,
        config: dict[str, Any],
        poll_interval_ms: int = _DEFAULT_PROXY_POLL_INTERVAL_MS,
    ):
        """Initialize the proxy state and start the reservation scheduler thread."""
        self._poll_interval_s = max(
            self._MIN_POLL_INTERVAL_SECONDS,
            poll_interval_ms / _MILLISECONDS_TO_SECONDS,
        )
        self._pending_requests: queue.Queue[_ReservationRequest] = queue.Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._delay_by_pair: dict[tuple[str, str], float] = {}
        self._endpoint_to_bw_group: dict[str, str] = {}
        self._bw_by_group: dict[str, float] = {}
        self._uplink_next_free: dict[str, float] = {}
        self._downlink_next_free: dict[str, float] = {}

        self._load_config(config)

        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="net-emulation-proxy",
            daemon=True,
        )
        self._scheduler_thread.start()

    def _load_config(self, config: dict[str, Any]) -> None:
        symmetric = bool(config.get("symmetric", _DEFAULT_SYMMETRIC_LINKS))

        for idx, group in enumerate(config.get("bandwidth_groups", [])):
            group_id = f"group-{idx}"
            bandwidth_mbps = float(group["bandwidth_mbps"])
            bandwidth_bytes_per_s = bandwidth_mbps * _MEGABITS_TO_BYTES
            self._bw_by_group[group_id] = bandwidth_bytes_per_s
            self._uplink_next_free[group_id] = 0.0
            self._downlink_next_free[group_id] = 0.0
            for endpoint in group["members"]:
                self._endpoint_to_bw_group[self._normalize_endpoint(str(endpoint))] = (
                    group_id
                )

        for pair in config.get("crossdc_pairs", []):
            src = self._normalize_endpoint(str(pair["src"]))
            dst = self._normalize_endpoint(str(pair["dst"]))
            delay_s = float(pair["delay_ms"]) / _MILLISECONDS_TO_SECONDS
            self._delay_by_pair[(src, dst)] = delay_s
            if symmetric:
                self._delay_by_pair[(dst, src)] = delay_s

    def reserve(
        self,
        src: str,
        dst: str,
        size_bytes: int,
    ) -> float:
        """Reserve network capacity and return the remaining wait time in seconds."""
        request = _ReservationRequest(
            src=src,
            dst=dst,
            size_bytes=max(0, int(size_bytes)),
            request_time=time.monotonic(),
            done=threading.Event(),
        )
        self._pending_requests.put(request)
        while not request.done.wait(timeout=self._RESERVATION_POLL_INTERVAL_SECONDS):
            pass
        if request.error is not None:
            raise request.error
        assert request.ready_at is not None
        remaining = request.ready_at - time.monotonic()
        return max(remaining, 0.0)

    def shutdown(self) -> None:
        """Stop the background scheduler thread."""
        self._stop_event.set()
        self._scheduler_thread.join(timeout=1.0)

    def _scheduler_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._pending_requests.get(timeout=self._poll_interval_s)
            except queue.Empty:
                continue
            try:
                request.ready_at = self._schedule_request(request)
            except Exception as exc:  # pragma: no cover - defensive path
                request.error = exc
            finally:
                request.done.set()

    def _schedule_request(self, request: _ReservationRequest) -> float:
        norm_src = self._normalize_endpoint(request.src)
        norm_dst = self._normalize_endpoint(request.dst)
        lookup_key = (norm_src, norm_dst)
        delay_s = self._delay_by_pair.get(lookup_key)
        if delay_s is None:
            return request.request_time

        src_group = self._endpoint_to_bw_group.get(norm_src)
        dst_group = self._endpoint_to_bw_group.get(norm_dst)
        bw_u = self._bw_by_group.get(src_group, math.inf)
        bw_d = self._bw_by_group.get(dst_group, math.inf)

        with self._lock:
            t0 = request.request_time
            # Emulated uplink: serialize sends from the same source under the
            # configured sender-side bandwidth.
            t_u_start = (
                max(t0, self._uplink_next_free.get(src_group, 0.0)) if src_group else t0
            )
            t_u_finish = (
                t_u_start + (request.size_bytes / bw_u)
                if math.isfinite(bw_u) and bw_u > 0
                else t_u_start
            )
            if src_group:
                self._uplink_next_free[src_group] = t_u_finish

            # Delay queue: shift the whole transfer by the configured link delay.
            first_bit_arrive = t_u_start + delay_s
            last_bit_arrive = t_u_finish + delay_s

            # Emulated downlink: serialize receives at the destination when the
            # receiver-side bandwidth becomes the bottleneck.
            t_d_start = max(
                first_bit_arrive, self._downlink_next_free.get(dst_group, 0.0)
            )
            t_d_finish = (
                t_d_start + (request.size_bytes / bw_d)
                if math.isfinite(bw_d) and bw_d > 0
                else t_d_start
            )
            ready_at = max(last_bit_arrive, t_d_finish)
            if dst_group:
                self._downlink_next_free[dst_group] = ready_at
            return ready_at


class NetEmulationClient:
    """Client helper that coordinates with the global net-emulation proxy."""

    _PROXY_READY_TIMEOUT_SECONDS = 60.0
    _RESERVE_TIMEOUT_SECONDS = 120.0
    _PROXY_MAX_CONCURRENCY = 32

    def __init__(self, config: NetEmulationConfig):
        """Create a client and connect it to the shared proxy actor."""
        self._config = config
        self._proxy = self._get_or_create_proxy(config)

    @staticmethod
    def _resolve_proxy_launch_node(cluster: Cluster, config: NetEmulationConfig):
        """Pick the node where the global proxy should be created."""
        runtime_context = ray.get_runtime_context()
        current_node_id = runtime_context.get_node_id()
        for node_rank in range(cluster.num_nodes):
            node = cluster.get_node_info(node_rank)
            if node.ray_id == current_node_id:
                return node

        if config.proxy.node_rank is not None:
            return cluster.get_node_info(config.proxy.node_rank)

        return cluster.get_node_info(0)

    @staticmethod
    def _get_or_create_proxy(config: NetEmulationConfig):
        handle = None

        try:
            handle = ray.get_actor(config.proxy.name, namespace=Cluster.NAMESPACE)
        except ValueError:
            pass

        if handle is None:
            cluster = Cluster()
            node = NetEmulationClient._resolve_proxy_launch_node(cluster, config)
            remote_proxy = ray.remote(NetEmulationProxy)
            proxy_env_vars = node.env_vars.copy()
            path_env_merge_mode = Cluster.get_path_env_merge_mode(proxy_env_vars)
            if "PYTHONPATH" in os.environ:
                proxy_env_vars = Cluster.merge_worker_env_vars(
                    proxy_env_vars,
                    {"PYTHONPATH": os.environ["PYTHONPATH"]},
                    path_env_merge_mode,
                )
            options = {
                "name": config.proxy.name,
                "num_cpus": config.proxy.num_cpus,
                "max_concurrency": NetEmulationClient._PROXY_MAX_CONCURRENCY,
                "scheduling_strategy": NodeAffinitySchedulingStrategy(
                    node_id=node.ray_id,
                    soft=False,
                ),
                "runtime_env": {
                    "py_executable": node.python_interpreter_path,
                    "env_vars": proxy_env_vars,
                },
            }
            try:
                handle = remote_proxy.options(**options).remote(
                    config.to_proxy_payload(),
                    config.proxy.poll_interval_ms,
                )
            except ValueError:
                handle = ray.get_actor(config.proxy.name, namespace=Cluster.NAMESPACE)

        ray.get(
            handle._ready.remote(),
            timeout=NetEmulationClient._PROXY_READY_TIMEOUT_SECONDS,
        )
        return handle

    def wait_until_allowed(self, src: str, dst: str, size_bytes: int) -> None:
        """Block until the emulated transfer is allowed to proceed."""
        try:
            reserve_ref = self._proxy.reserve.remote(src, dst, size_bytes)
            remaining = ray.get(
                reserve_ref,
                timeout=NetEmulationClient._RESERVE_TIMEOUT_SECONDS,
            )
        except ray.exceptions.GetTimeoutError:
            raise
        if remaining > 0:
            time.sleep(remaining)
