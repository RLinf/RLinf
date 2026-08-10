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

from rlinf.scheduler.cluster.utils import parse_rank_config

from .manager import Manager

# Rough per-tensor protocol overhead for shape, dtype, and size metadata.
_TENSOR_METADATA_OVERHEAD = 256
_DEFAULT_SYMMETRIC_LINKS = True
_MEGABITS_TO_BYTES = 1_000_000.0 / 8.0
_MILLISECONDS_TO_SECONDS = 1000.0


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
                members=cls._expand_endpoints(item["members"], field_name="members"),
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
        values = (value,) if isinstance(value, str) else value
        if isinstance(values, (list, tuple)) and values:
            endpoints = []
            for item in values:
                group, ranks = str(item).split(":", 1)
                endpoints.extend(
                    f"{group}:{rank}"
                    for rank in parse_rank_config(ranks, rank_type=field_name)
                )
            return tuple(endpoints)
        raise ValueError(
            "net_emulation endpoint entries must define a non-empty "
            f"string or list for '{field_name}'"
        )


class NetEmulationManager(Manager):
    """Global manager that emulates cross-worker network latency and bandwidth.

    Like the other scheduler managers, it is a single Ray actor pinned to node rank 0
    and reached from any process via :meth:`get_proxy`. Unlike them it is optional:
    the cluster launches it only when ``cluster.net_emulation.enabled`` is set, so
    callers acquire it with ``get_proxy(no_wait=True)`` and read the resulting
    ``ValueError`` as "emulation is off".

    Senders book a transmission slot with :meth:`reserve` before handing a payload to
    the transport and sleep for the wait it returns. Because every reservation is
    booked on the manager's own clock, the per-link delay and the per-group bandwidth
    budget stay consistent across the whole cluster. Payload sizes are measured by the
    sender through :meth:`estimate_payload_size_bytes`, since shipping the payload to
    the manager would cost the very bandwidth being emulated.
    """

    MANAGER_NAME = "NetEmulationManager"

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
        """Book a point-to-point transmission and return the sender's wait.

        Args:
            src (str): Name of the sending worker.
            dst (str): Name of the receiving worker.
            size_bytes (int): Estimated wire size of the payload.

        Returns:
            float: Remaining wait in seconds, ``0.0`` when the link is not emulated.
        """
        return self._book(src, [dst], size_bytes)

    def reserve_broadcast(self, src: str, dsts: list[str], size_bytes: int) -> float:
        """Book a one-to-many transmission and return the sender's wait.

        The payload leaves the source once, so the sender-side bandwidth is charged
        a single time; each distinct receiving bandwidth group is charged once, on
        the assumption that a broadcast crosses every emulated link once and fans
        out locally afterwards. The wait is the slowest destination's, which is what
        gates the collective.

        Args:
            src (str): Name of the broadcasting worker.
            dsts (list[str]): Names of the receiving workers.
            size_bytes (int): Estimated wire size of the payload.

        Returns:
            float: Remaining wait in seconds, ``0.0`` when no destination is on an
            emulated link.
        """
        return self._book(src, dsts, size_bytes)

    def _book(self, src: str, dsts: list[str], size_bytes: int) -> float:
        """Reserve capacity for one transfer from *src* to every emulated *dsts*."""
        norm_src = self._normalize_endpoint(src)
        links: list[tuple[str, float]] = []
        for dst in dsts:
            norm_dst = self._normalize_endpoint(dst)
            delay_s = self._delay_by_pair.get((norm_src, norm_dst))
            if delay_s is not None:
                links.append((norm_dst, delay_s))
        if not links:
            return 0.0

        size_bytes = max(0, int(size_bytes))
        src_group = self._endpoint_to_bw_group.get(norm_src)
        bw_u = self._bw_by_group.get(src_group, math.inf)

        with self._lock:
            t0 = time.monotonic()
            # Emulated uplink: serialize sends from the same source under the
            # configured sender-side bandwidth. Charged once per transfer.
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

            ready_at = t0
            charged_dst_groups: set[str] = set()
            for norm_dst, delay_s in links:
                # Delay queue: shift the whole transfer by the configured link delay.
                first_bit_arrive = t_u_start + delay_s
                last_bit_arrive = t_u_finish + delay_s

                # Emulated downlink: serialize receives at the destination when the
                # receiver-side bandwidth becomes the bottleneck.
                dst_group = self._endpoint_to_bw_group.get(norm_dst)
                if dst_group is not None and dst_group in charged_dst_groups:
                    ready_at = max(ready_at, self._downlink_next_free[dst_group])
                    continue
                bw_d = self._bw_by_group.get(dst_group, math.inf)
                t_d_start = max(
                    first_bit_arrive, self._downlink_next_free.get(dst_group, 0.0)
                )
                t_d_finish = (
                    t_d_start + (size_bytes / bw_d)
                    if math.isfinite(bw_d) and bw_d > 0
                    else t_d_start
                )
                dst_ready_at = max(last_bit_arrive, t_d_finish)
                if dst_group is not None:
                    self._downlink_next_free[dst_group] = dst_ready_at
                    charged_dst_groups.add(dst_group)
                ready_at = max(ready_at, dst_ready_at)

        return max(ready_at - time.monotonic(), 0.0)

    # ============================ Payload size estimation ============================
    # These run in the sender's process, not on the manager actor.

    @classmethod
    def estimate_payload_size_bytes(cls, payload: Any) -> int:
        """Estimate the wire size of *payload* in bytes.

        Args:
            payload (Any): Object about to be sent. Tensors are measured from their
                storage, everything else from its pickled form.

        Returns:
            int: Estimated number of bytes on the wire.
        """
        if payload is None:
            return 0

        if isinstance(payload, torch.Tensor):
            return payload.numel() * payload.element_size()

        if cls._contains_tensor(payload):
            return (
                cls._estimate_tensor_data_size(payload)
                + cls._count_tensors(payload) * _TENSOR_METADATA_OVERHEAD
                + cls._estimate_metadata_size(payload)
            )

        return cls._pickle_part_size(payload)

    @classmethod
    def _contains_tensor(cls, payload: Any) -> bool:
        """Check whether *payload* or any nested element contains a torch.Tensor."""
        if isinstance(payload, torch.Tensor):
            return True
        if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
            return any(
                cls._contains_tensor(getattr(payload, f.name))
                for f in dataclasses.fields(payload)
            )
        if isinstance(payload, Mapping):
            return any(cls._contains_tensor(v) for v in payload.values())
        if isinstance(payload, (list, tuple)):
            return any(cls._contains_tensor(item) for item in payload)
        return False

    @classmethod
    def _count_tensors(cls, payload: Any) -> int:
        """Count the number of tensor leaves inside *payload*."""
        if isinstance(payload, torch.Tensor):
            return 1
        if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
            return sum(
                cls._count_tensors(getattr(payload, f.name))
                for f in dataclasses.fields(payload)
            )
        if isinstance(payload, Mapping):
            return sum(cls._count_tensors(v) for v in payload.values())
        if isinstance(payload, (list, tuple)):
            return sum(cls._count_tensors(item) for item in payload)
        return 0

    @classmethod
    def _estimate_tensor_data_size(cls, payload: Any) -> int:
        """Sum of raw tensor data sizes (no metadata) inside *payload*."""
        if isinstance(payload, torch.Tensor):
            return payload.numel() * payload.element_size()
        if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
            return sum(
                cls._estimate_tensor_data_size(getattr(payload, f.name))
                for f in dataclasses.fields(payload)
            )
        if isinstance(payload, Mapping):
            return sum(cls._estimate_tensor_data_size(v) for v in payload.values())
        if isinstance(payload, (list, tuple)):
            return sum(cls._estimate_tensor_data_size(item) for item in payload)
        return 0

    @classmethod
    def _estimate_metadata_size(cls, payload: Any) -> int:
        """Estimate the size of non-tensor metadata (keys, struct info, piggyback)."""
        if isinstance(payload, torch.Tensor):
            return 0
        if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
            fields = dataclasses.fields(payload)
            field_names_size = cls._pickle_part_size([f.name for f in fields])
            fields_meta = sum(
                cls._estimate_metadata_size(getattr(payload, f.name)) for f in fields
            )
            return field_names_size + fields_meta
        if isinstance(payload, Mapping):
            keys_size = cls._pickle_part_size(list(payload.keys()))
            values_meta = sum(cls._estimate_metadata_size(v) for v in payload.values())
            return keys_size + values_meta
        if isinstance(payload, (list, tuple)):
            return sum(cls._estimate_metadata_size(item) for item in payload)
        return cls._pickle_part_size(payload)

    @classmethod
    def _pickle_part_size(cls, obj: Any) -> int:
        """Estimate the wire size of a non-tensor object via pickle."""
        if obj is None:
            return 0
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return max(1, len(repr(obj)))
