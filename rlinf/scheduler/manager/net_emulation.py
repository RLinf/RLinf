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

import dataclasses
import math
import pickle
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from .manager import Manager

# Rough per-tensor protocol overhead for shape, dtype, and size metadata.
_TENSOR_METADATA_OVERHEAD = 256
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


class NetEmulationManager(Manager):
    """A global manager that emulates cross-worker network latency and bandwidth.

    Like the other scheduler managers, it is a single Ray actor pinned to node rank 0
    and reached from any process via :meth:`get_proxy`. The cluster launches it at
    startup when ``cluster.net_emulation.enabled`` is set.

    Senders call the class method :meth:`wait_before_send` right before handing a
    payload to the transport. It asks the manager for a transmission slot on the
    emulated link and sleeps for however long the manager says the transfer would
    have taken. Because every reservation is booked on the manager's own clock, the
    per-link delay and the per-group bandwidth budget stay consistent across the
    whole cluster. The API is a no-op until the manager has been launched, so it is
    safe to leave on the send path of runs that do not emulate the network.
    """

    MANAGER_NAME = "NetEmulationManager"

    # Process-local cache for the send path (never touched on the manager actor).
    _unavailable: bool = False

    # =============================== Manager (server) side ===============================

    def __init__(self, cfg: DictConfig | dict[str, Any]):
        """Build the emulated link table from the ``cluster.net_emulation`` config."""
        config = NetEmulationConfig.from_cfg(cfg)
        assert config is not None, (
            "NetEmulationManager was launched but cluster.net_emulation is disabled "
            "or empty. It must only be launched when net emulation is enabled."
        )
        self._lock = threading.Lock()
        self._delay_by_pair: dict[tuple[str, str], float] = {}
        self._endpoint_to_bw_group: dict[str, str] = {}
        self._bw_by_group: dict[str, float] = {}
        self._uplink_next_free: dict[str, float] = {}
        self._downlink_next_free: dict[str, float] = {}

        for idx, group in enumerate(config.bandwidth_groups):
            group_id = f"group-{idx}"
            self._bw_by_group[group_id] = group.bandwidth_mbps * _MEGABITS_TO_BYTES
            self._uplink_next_free[group_id] = 0.0
            self._downlink_next_free[group_id] = 0.0
            for endpoint in group.members:
                self._endpoint_to_bw_group[self._normalize_endpoint(endpoint)] = (
                    group_id
                )

        for pair in config.crossdc_pairs:
            src = self._normalize_endpoint(pair.src)
            dst = self._normalize_endpoint(pair.dst)
            delay_s = pair.delay_ms / _MILLISECONDS_TO_SECONDS
            self._delay_by_pair[(src, dst)] = delay_s
            if config.symmetric:
                self._delay_by_pair[(dst, src)] = delay_s

    @staticmethod
    def _normalize_endpoint(name: str) -> str:
        """Normalize endpoint names by stripping a trailing ``Group`` suffix."""
        parts = name.split(":", 1)
        group = parts[0]
        if group.endswith("Group"):
            group = group[: -len("Group")]
        return group + (":" + parts[1] if len(parts) > 1 else "")

    def reserve(self, src: str, dst: str, size_bytes: int) -> float:
        """Book a transmission slot and return how long the sender must still wait.

        Args:
            src (str): Name of the sending worker.
            dst (str): Name of the receiving worker.
            size_bytes (int): Estimated wire size of the payload.

        Returns:
            float: Remaining wait in seconds, ``0.0`` when the link is not emulated.
        """
        norm_src = self._normalize_endpoint(src)
        norm_dst = self._normalize_endpoint(dst)
        delay_s = self._delay_by_pair.get((norm_src, norm_dst))
        if delay_s is None:
            return 0.0

        size_bytes = max(0, int(size_bytes))
        src_group = self._endpoint_to_bw_group.get(norm_src)
        dst_group = self._endpoint_to_bw_group.get(norm_dst)
        bw_u = self._bw_by_group.get(src_group, math.inf)
        bw_d = self._bw_by_group.get(dst_group, math.inf)

        with self._lock:
            t0 = time.monotonic()
            # Emulated uplink: serialize sends from the same source under the
            # configured sender-side bandwidth.
            t_u_start = (
                max(t0, self._uplink_next_free.get(src_group, 0.0)) if src_group else t0
            )
            t_u_finish = (
                t_u_start + (size_bytes / bw_u)
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
                t_d_start + (size_bytes / bw_d)
                if math.isfinite(bw_d) and bw_d > 0
                else t_d_start
            )
            ready_at = max(last_bit_arrive, t_d_finish)
            if dst_group:
                self._downlink_next_free[dst_group] = ready_at

        return max(ready_at - time.monotonic(), 0.0)

    # ============================= Client-facing send side =============================

    @classmethod
    def _get(cls):
        """Return the manager proxy, or None if net emulation was never launched."""
        if cls._unavailable:
            return None
        try:
            return cls.get_proxy(no_wait=True)
        except Exception:
            cls._unavailable = True
            return None

    @classmethod
    def wait_before_send(cls, src: str, dst: str, *payloads: Any) -> None:
        """Block for the time the emulated ``src`` -> ``dst`` transfer would take.

        No-op when net emulation is disabled or the link is not configured.

        Args:
            src (str): Name of the sending worker.
            dst (str): Name of the receiving worker.
            *payloads: Objects about to be sent, used to estimate the wire size.
        """
        proxy = cls._get()
        if proxy is None:
            return
        size_bytes = sum(estimate_payload_size_bytes(payload) for payload in payloads)
        remaining = proxy.reserve(src, dst, size_bytes)
        if remaining > 0:
            time.sleep(remaining)
