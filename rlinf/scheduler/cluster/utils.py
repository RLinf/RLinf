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


import atexit
import contextlib
import dataclasses
import importlib
import logging
import os
import re
import threading
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TextIO

import torch

# Type for a single tensor field value in a dataclass (used for send/recv).
TensorFieldValue = (
    torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor, ...]
    | dict[str, torch.Tensor]
)
# Metadata for flatten/unflatten: (field_name, 'tensor'|'list'|'tuple'|'dict', None|length|list_of_keys).
DataclassTensorFieldsMetadata = list[
    tuple[str, str, Optional[int] | Optional[list[str]]]
]


@contextlib.contextmanager
def without_http_proxies():
    """Temporarily set no_proxy for Ray state API calls."""
    prev_no_proxy_upper = os.environ.get("NO_PROXY", None)
    prev_no_proxy_lower = os.environ.get("no_proxy", None)

    # Ray state APIs query the dashboard via HTTP.
    ray_address = None
    try:
        import ray.dashboard.utils

        ray_address = ray.dashboard.utils.get_address_for_submission_client(None)
    except Exception:
        ray_address = None

    if ray_address:
        if "http://" in ray_address:
            ray_address = ray_address.replace("http://", "")
        elif "https://" in ray_address:
            ray_address = ray_address.replace("https://", "")
        if ":" in ray_address:
            ray_address = ray_address.split(":")[0]
        os.environ["NO_PROXY"] = ray_address
        os.environ["no_proxy"] = ray_address

    try:
        yield
    finally:
        if prev_no_proxy_upper is not None:
            os.environ["NO_PROXY"] = prev_no_proxy_upper
        else:
            os.environ.pop("NO_PROXY", None)
        if prev_no_proxy_lower is not None:
            os.environ["no_proxy"] = prev_no_proxy_lower
        else:
            os.environ.pop("no_proxy", None)


def load_user_extension_module(logger: Optional[logging.Logger] = None) -> None:
    """Import ``RLINF_EXT_MODULE`` and call ``register()`` when defined."""
    from .cluster import Cluster, ClusterEnvVar

    ext_module_name = Cluster.get_sys_env_var(ClusterEnvVar.EXT_MODULE)
    if ext_module_name is None:
        return

    log = logger or logging.getLogger(__name__)
    try:
        ext_module = importlib.import_module(ext_module_name)
        if hasattr(ext_module, "register"):
            ext_module.register()
            log.debug(
                "Loaded extension module '%s' and called register()",
                ext_module_name,
            )
        else:
            log.warning(
                "Extension module '%s' has no register() function",
                ext_module_name,
            )
    except ImportError as e:
        log.warning(
            "Failed to import extension module '%s': %s",
            ext_module_name,
            e,
        )
    except Exception:
        log.exception("Error loading extension module '%s'", ext_module_name)


class DistributedRayLogCollector:
    """Collect and split Ray worker logs into per-worker files.

    The collector registers allocated workers, resolves their Ray worker ids,
    and tails only the corresponding Ray worker log files directly.

    Output format:
    ``<output_dir>/<group_name>/rank_<rank>.log``

    Workers in the same worker group share a folder; the group name is derived
    from worker_name (WorkerAddress format: group:rank1:rank2...).
    """

    def __init__(
        self,
        logger,
        output_dir: str | Path,
        namespace: Optional[str] = None,
        poll_interval_s: float = 1.0,
    ):
        """Initialize the distributed Ray log collector.

        Args:
            logger: Logger used for collector status and warning messages.
            output_dir (str | Path): Root directory for split output logs.
            namespace (Optional[str]): Ray namespace used to disambiguate actor
                lookup when resolving actor IDs by actor name.
            poll_interval_s (float): Poll interval in seconds for tailing files.
        """
        self._logger: logging.Logger = logger
        self._output_dir = Path(output_dir)
        self._namespace = namespace
        self._poll_interval_s = max(0.1, poll_interval_s)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file_offsets: dict[Path, int] = {}
        self._output_files: dict[tuple[str, str], TextIO] = {}
        self._log_file_map: dict[Path, tuple[str, str]] = {}
        self._registered_workers: dict[str, dict[str, Any]] = {}
        self._registry_lock = threading.Lock()
        self._started = False

    def __getstate__(self) -> dict[str, Any]:
        """Return a Ray/pickle-safe state.

        Runtime-only resources such as thread/event/lock and open files are
        recreated after unpickling and should not be serialized.
        """
        state = self.__dict__.copy()
        # Runtime-only objects are not pickleable and are safe to reconstruct.
        state.pop("_stop_event", None)
        state.pop("_thread", None)
        state.pop("_registry_lock", None)
        state["_output_files"] = {}
        state["_file_offsets"] = {}
        state["_log_file_map"] = {}
        state["_started"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore collector state after unpickling."""
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._thread = None
        self._registry_lock = threading.Lock()
        self._output_files = {}
        self._file_offsets = {}
        self._log_file_map = {}
        self._started = False

    @staticmethod
    def _sanitize_path_component(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    @staticmethod
    def _get_group_name(worker_name: str) -> str:
        """Extract worker group name from worker_name (WorkerAddress format: group:rank1:rank2...)."""
        return worker_name.split(":", 1)[0]

    @staticmethod
    def _get_ray_logs_dir() -> Optional[Path]:
        from ray._private import worker

        if worker._global_node is None:
            return None
        return Path(worker._global_node.get_logs_dir_path())

    def start(self) -> bool:
        """Start the background collector thread.

        Returns:
            bool: True if the collector is started, False otherwise.
        """
        logs_dir = self._get_ray_logs_dir()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="ray-log-collector",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        atexit.register(self._atexit_stop)
        self._logger.info(
            f"Started distributed log collector. Reading from {logs_dir}, writing split logs to {self._output_dir}."
        )
        return True

    def _atexit_stop(self) -> None:
        """Called by atexit to ensure logs are flushed on normal process exit."""
        if self._started:
            self.stop()

    def _drain_remaining(self, logs_dir: Path) -> None:
        """Synchronously tail any remaining log content from Ray worker files."""
        try:
            # For draining we want to read from the beginning for any new files.
            self._process_once(logs_dir, new_file_offset_from_start=True)
        except Exception as e:
            self._logger.warning(f"Distributed log collector drain failed: {e}")

    def stop(self, join_timeout_s: float = 10.0) -> None:
        """Stop the collector thread and flush remaining logs.

        Performs a final drain of Ray worker log files so that logs are complete
        even when the main process exits before the tail loop finishes.

        Args:
            join_timeout_s (float): Max seconds to wait for thread join.
        """
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)
        logs_dir = self._get_ray_logs_dir()
        if logs_dir is not None:
            self._drain_remaining(logs_dir)
        for handle in self._output_files.values():
            try:
                handle.close()
            except Exception:
                pass
        self._output_files.clear()
        self._started = False
        try:
            atexit.unregister(self._atexit_stop)
        except Exception:
            pass

    def _process_once(
        self, logs_dir: Path, *, new_file_offset_from_start: bool
    ) -> None:
        """Process a single iteration of resolving workers and tailing log files.

        Args:
            logs_dir (Path): Ray logs directory.
            new_file_offset_from_start (bool): If True, start reading new files
                from offset 0; if False, start from the current end of file.
        """
        self._resolve_registered_workers(logs_dir)
        with self._registry_lock:
            bound_files = list(self._log_file_map.items())
        for log_file, (worker_name, rank) in bound_files:
            if not log_file.exists():
                continue
            if log_file not in self._file_offsets:
                if new_file_offset_from_start:
                    self._file_offsets[log_file] = 0
                else:
                    self._file_offsets[log_file] = log_file.stat().st_size
            try:
                with log_file.open("r", encoding="utf-8", errors="replace") as fp:
                    fp.seek(self._file_offsets[log_file])
                    while True:
                        line = fp.readline()
                        if not line:
                            break
                        handle = self._get_output_handle(worker_name, rank)
                        handle.write(line)
                        handle.flush()
                    self._file_offsets[log_file] = fp.tell()
            except OSError:
                # If a file disappears between listing and opening, skip it.
                continue

    def _get_output_handle(self, worker_name: str, rank: str) -> TextIO:
        key = (worker_name, rank)
        if key in self._output_files:
            return self._output_files[key]
        group_name = self._get_group_name(worker_name)
        safe_group_name = self._sanitize_path_component(group_name)
        group_dir = self._output_dir / safe_group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        output_path = group_dir / f"rank_{rank}.log"
        self._output_files[key] = output_path.open("a", encoding="utf-8")
        return self._output_files[key]

    @staticmethod
    def _state_get(actor_state: Any, *keys: str) -> Optional[Any]:
        if isinstance(actor_state, dict):
            for key in keys:
                if key in actor_state:
                    return actor_state[key]
            return None
        for key in keys:
            if hasattr(actor_state, key):
                return getattr(actor_state, key)
        return None

    def _resolve_worker_id_from_workers_api(
        self, pid: int, node_id: Optional[str] = None
    ) -> Optional[str]:
        """Resolve worker_id from Ray worker states using pid/node_id filters."""
        from ray.util.state import list_workers

        filters = [("pid", "=", str(pid))]
        if node_id is not None:
            filters.append(("node_id", "=", str(node_id)))
        with without_http_proxies():
            worker_states = list_workers(filters=filters)
        for worker_state in worker_states:
            worker_id = self._state_get(worker_state, "worker_id", "workerId")
            if worker_id is not None:
                return str(worker_id)
        return None

    def register_worker(
        self,
        worker_name: str,
        rank: int,
        actor_handle: Optional = None,
        actor_id: Optional[str] = None,
    ) -> None:
        """Register worker metadata for direct source-file log binding."""
        with self._registry_lock:
            self._registered_workers[worker_name] = {
                "worker_name": worker_name,
                "rank": str(rank),
                "actor_handle": actor_handle,
                "actor_id": actor_id,
                "job_id": None,
                "pid": None,
                "worker_id": None,
                "resolved": False,
            }

    def _resolve_registered_workers(self, logs_dir: Path) -> None:
        from ray.util.state import list_actors

        with self._registry_lock:
            workers = list(self._registered_workers.values())

        for meta in workers:
            if meta["resolved"]:
                continue
            if meta["actor_id"] is None and meta["actor_handle"] is not None:
                raw_actor_id = getattr(meta["actor_handle"], "_actor_id", None)
                if raw_actor_id is not None and hasattr(raw_actor_id, "hex"):
                    meta["actor_id"] = raw_actor_id.hex()

            filters = [("NAME", "=", meta["worker_name"])]
            if self._namespace is not None:
                filters.append(("RAY_NAMESPACE", "=", self._namespace))
            with without_http_proxies():
                actor_states = list_actors(filters=filters)
            if len(actor_states) == 0:
                continue
            actor_state = actor_states[0]

            if not meta["job_id"]:
                job_id = self._state_get(actor_state, "job_id", "jobId")
                if job_id is not None:
                    meta["job_id"] = str(job_id)

            if meta["pid"] is None:
                pid_value = self._state_get(
                    actor_state, "pid", "process_id", "processId"
                )
                if pid_value is not None:
                    try:
                        meta["pid"] = int(pid_value)
                    except (TypeError, ValueError):
                        pass

            if not meta["worker_id"] and meta["pid"] is not None:
                node_id = self._state_get(actor_state, "node_id", "nodeId")
                worker_id = self._resolve_worker_id_from_workers_api(
                    pid=meta["pid"],
                    node_id=str(node_id) if node_id is not None else None,
                )
                if worker_id is not None:
                    meta["worker_id"] = worker_id

            if meta["pid"] is None:
                continue

            # Ray creates worker logs as worker-{worker_id}-{job_id?}-{pid}.{out|err}
            # job_id can be nil (ffffffff), missing, or set - support all cases
            files_set: set[Path] = set()
            suffixes = ("out", "err")
            if meta["worker_id"]:
                for suffix in suffixes:
                    # No job_id: worker-{worker_id}-{pid}.{suffix}
                    files_set.add(
                        logs_dir / f"worker-{meta['worker_id']}-{meta['pid']}.{suffix}"
                    )
                    # With any job_id: worker-{worker_id}-{job_id}-{pid}.{suffix}
                    files_set.update(
                        logs_dir.glob(
                            f"worker-{meta['worker_id']}-*-{meta['pid']}.{suffix}"
                        )
                    )
            files = list(files_set)
            with self._registry_lock:
                for file in files:
                    if file in self._log_file_map:
                        continue
                    self._log_file_map[file] = (
                        meta["worker_name"],
                        str(meta["rank"]),
                    )
                if files:
                    meta["resolved"] = True

    def _run(self) -> None:
        logs_dir = self._get_ray_logs_dir()
        if logs_dir is None:
            return
        while not self._stop_event.is_set():
            try:
                # In the background loop, skip historical content for new files.
                self._process_once(logs_dir, new_file_offset_from_start=False)
            except Exception as e:
                self._logger.warning(f"Distributed log collector iteration failed: {e}")
            self._stop_event.wait(self._poll_interval_s)


class DataclassProtocol(Protocol):
    """Protocol for dataclasses to enable type checking."""

    __dataclass_fields__: dict
    __dataclass_params__: dict
    __post_init__: Optional[Callable]


def parse_rank_config(
    rank_config: str | int,
    available_ranks: Optional[list[int]] = None,
    rank_type: Optional[str] = None,
) -> list[int]:
    """Parse a rank configuration string into a list of ranks.

    Args:
        rank_config (str | int): The rank configuration string, e.g., "0-3,5,7-9" or "all".
        available_ranks (Optional[list[int]]): The list of available ranks.
        rank_type (Optional[str]): The type of rank being parsed (for error messages).

    Returns:
        list[int]: The list of ranks.
    """
    ranks = set()
    if available_ranks is not None:
        available_ranks = sorted(available_ranks)
    # If the rank config is a single number
    # Omegaconf will parse it as an integer instead of a string
    rank_config = str(rank_config)
    if rank_config.lower() == "all":
        assert available_ranks is not None, (
            'When rank_config is "all", available_ranks must be provided.'
        )
        ranks = list(set(available_ranks))
    else:
        # First split by comma
        rank_ranges = rank_config.split(",")
        for rank_range in rank_ranges:
            rank_range = rank_range.strip()
            if rank_range == "":
                continue
            # Then split by hyphen to get the start and end of the range
            rank_range = rank_range.split("-")
            try:
                if len(rank_range) == 1:
                    start_rank = int(rank_range[0])
                    end_rank = start_rank
                elif len(rank_range) == 2:
                    start_rank = int(rank_range[0])
                    end_rank = int(rank_range[1])
                else:
                    raise ValueError
            except (ValueError, IndexError):
                raise ValueError(
                    f'Invalid rank format {rank_config} for {rank_type}, expected format: "a,b,c-d" or "all"'
                )
            assert end_rank >= start_rank, (
                f"Start rank {start_rank} must be less than or equal to end rank {end_rank} in rank config {rank_config} for {rank_type}."
            )
            if available_ranks is not None:
                assert available_ranks[0] <= start_rank <= available_ranks[-1], (
                    f'Start rank {start_rank} in rank config string "{rank_config}" must be within the available {rank_type if rank_type is not None else ""} ranks {available_ranks}.'
                )
                assert available_ranks[0] <= end_rank <= available_ranks[-1], (
                    f'End rank {end_rank} in rank config string "{rank_config}" must be within the available {rank_type if rank_type is not None else ""} ranks {available_ranks}.'
                )
            ranks.update(range(start_rank, end_rank + 1))
    ranks = list(ranks)
    return sorted(ranks)


def dataclass_arg_check(
    dataclass: DataclassProtocol,
    kwargs: dict,
    no_check_unknown: bool = False,
    error_suffix: str = "",
):
    """Check if the kwargs contain only valid fields for the given dataclass.

    Args:
        dataclass (DataclassProtocol): The dataclass to check against.
        kwargs (dict): The keyword arguments to check.
        no_check_unknown (bool): Whether to skip checking for unknown fields.
        error_suffix (str): Additional error message suffix.
    """
    args = set(kwargs.keys())
    valid_args = set(dataclass.__dataclass_fields__.keys())

    missing_args = valid_args - args
    unknown_args = args - valid_args
    missing_required_args = []
    for missing_arg in missing_args:
        field_info = dataclass.__dataclass_fields__[missing_arg]
        if (
            field_info.default is dataclasses.MISSING
            and field_info.default_factory is dataclasses.MISSING
        ):
            missing_required_args.append(missing_arg)

    assert not missing_required_args, (
        f"Missing fields '{missing_required_args}' detected {error_suffix}. Only got: {kwargs.keys()}."
    )
    if not no_check_unknown:
        assert not unknown_args, (
            f"Unknown fields '{unknown_args}' detected {error_suffix}. Valid fields are: {valid_args}."
        )

    return missing_required_args, unknown_args, valid_args


def extract_dataclass_tensor_fields(
    obj: Any,
) -> tuple[
    dict[str, TensorFieldValue], list[torch.Tensor], DataclassTensorFieldsMetadata
]:
    """Extract fields of a dataclass that are tensors or list/tuple/dict of tensors.

    Supported field types:
        - torch.Tensor
        - list[torch.Tensor] (all elements must be tensors)
        - tuple[torch.Tensor, ...] (all elements must be tensors)
        - dict[str, torch.Tensor] (all values must be tensors)

    Returns:
        (fields_dict, tensors_list, metadata): fields_dict maps field names to their value(s);
        tensors_list is a flat list of all tensors in field order for send/wire format;
        metadata describes each field's kind for unflatten on recv.
    """
    if not is_dataclass(obj):
        return {}, [], []
    result: dict[str, TensorFieldValue] = {}
    tensors_list: list[torch.Tensor] = []
    metadata: DataclassTensorFieldsMetadata = []
    for f in fields(obj):
        val = getattr(obj, f.name)
        if isinstance(val, torch.Tensor):
            result[f.name] = val
            tensors_list.append(val)
            metadata.append((f.name, "tensor", None))
        elif isinstance(val, (list, tuple)) and all(
            isinstance(item, torch.Tensor) for item in val
        ):
            # Preserve list vs tuple; flatten/unflatten will distinguish for wire format.
            result[f.name] = val
            tensors_list.extend(val)
            kind = "list" if isinstance(val, list) else "tuple"
            metadata.append((f.name, kind, len(val)))
        elif isinstance(val, dict) and all(
            isinstance(v, torch.Tensor) for v in val.values()
        ):
            result[f.name] = val
            keys = list(val.keys())
            tensors_list.extend(val[k] for k in keys)
            metadata.append((f.name, "dict", keys))
    return result, tensors_list, metadata


def unflatten_dataclass_tensor_fields(
    metadata: DataclassTensorFieldsMetadata,
    flat_tensors: list[torch.Tensor],
) -> dict[str, TensorFieldValue]:
    """Reconstruct a dict of tensor fields from metadata and flat tensor list (from recv)."""
    result: dict[str, TensorFieldValue] = {}
    idx = 0
    for name, kind, extra in metadata:
        if kind == "tensor":
            result[name] = flat_tensors[idx]
            idx += 1
        elif kind == "list":
            n = extra if isinstance(extra, int) else 0
            result[name] = flat_tensors[idx : idx + n]
            idx += n
        elif kind == "tuple":
            n = extra if isinstance(extra, int) else 0
            result[name] = tuple(flat_tensors[idx : idx + n])
            idx += n
        elif kind == "dict":
            keys = extra if isinstance(extra, list) else []
            result[name] = dict(zip(keys, flat_tensors[idx : idx + len(keys)]))
            idx += len(keys)
        else:
            raise ValueError(f"Unknown metadata kind for field {name}: {kind}")
    if idx != len(flat_tensors):
        raise ValueError(
            f"Metadata consumed {idx} tensors but flat list has {len(flat_tensors)}"
        )
    return result
