from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import logging

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
import os
import pickle
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock
from unittest.mock import patch

import pytest
import ray
from omegaconf import OmegaConf

import rlinf.scheduler.hardware.accelerators.nvidia_gpu as nv_module
from rlinf.scheduler import (
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Tracer,
    Worker,
    WorkerAddress,
)
from rlinf.scheduler.cluster.utils import DistributedRayLogCollector
from rlinf.scheduler.hardware.accelerators.intel_gpu import IntelGPUManager
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager
from rlinf.scheduler.manager.coll_manager import CollectiveManager
from rlinf.scheduler.manager.manager import Manager


def accelerator_is_available():
    """Return whether the Worker accelerator backend is available."""
    return (
        Worker.torch_platform is not None
        and hasattr(Worker.torch_platform, "is_available")
        and Worker.torch_platform.is_available()
    )


# Fixture to provide a ClusterResource instance for the test session
@pytest.fixture(scope="module")
def cluster():
    """Provides a ClusterResource instance for the tests."""
    # Use a small, fixed number of GPUs for consistent testing
    return Cluster(num_nodes=1)


# A basic Worker class for testing purposes
class BasicTestWorker(Worker):
    """A simple Worker implementation for testing basic functionality."""

    def __init__(self, arg1=None):
        super().__init__()
        self.arg1 = arg1
        self.initialized = True

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size

    def get_init_arg(self):
        return self.arg1


# A WorkerGroup-decorated class for testing distributed functionality
class DistributedTestWorker(Worker):
    """A WorkerGroup for testing distributed operations."""

    def __init__(self):
        super().__init__()

    def get_env_info(self):
        """Returns a dictionary of environment information for the worker."""
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "node_id": self._cluster_node_rank,
            "gpu_id": self._local_accelerator_rank,
            "node_local_rank": self._node_local_rank,
        }

    def sum_with_rank(self, value):
        """Adds the worker's rank to the given value."""
        return value + self._rank


class TestClusterResource:
    """Tests for the ClusterResource class."""

    def test_cluster_initialization(self, cluster: Cluster):
        """Verify that the cluster is initialized with correct properties."""
        assert cluster._num_nodes == 1
        if accelerator_is_available():
            assert cluster.num_accelerators >= 1


class TestWorkerAddress:
    """Tests for the WorkerAddress class."""

    def test_worker_address_naming(self):
        """Verify that WorkerAddress generates correct names."""
        addr = WorkerAddress("MyWorkerGroup", 5)
        assert addr.root_group_name == "MyWorkerGroup"
        assert addr.rank == 5
        assert addr.get_name() == "MyWorkerGroup:5"


class StaticMethodWorker(Worker):
    """A Worker exposing a public staticmethod, wrapped by WorkerMeta."""

    @staticmethod
    def add(first: int, second: int) -> int:
        return first + second


class TestWorkerMeta:
    """Tests for the WorkerMeta method wrapping."""

    def test_staticmethod_stays_a_staticmethod(self):
        """Verify WorkerMeta does not turn a staticmethod into an instance method."""
        assert isinstance(StaticMethodWorker.__dict__["add"], staticmethod)

    def test_staticmethod_call_does_not_rebind_self(self):
        """Verify calling a staticmethod through an instance passes no self."""
        worker = object.__new__(StaticMethodWorker)
        assert worker.add(1, 2) == 3
        assert StaticMethodWorker.add(1, 2) == 3


class TestManagerNamespace:
    """Tests for manager namespace propagation."""

    def test_manager_runtime_env_vars_include_cluster_namespace(self):
        """Verify manager runtime env always includes the cluster namespace."""
        with mock.patch.object(Cluster, "NAMESPACE", "test-namespace"):
            runtime_env = Manager.get_runtime_env_vars()

        assert runtime_env["CLUSTER_NAMESPACE"] == "test-namespace"

    def test_sync_cluster_namespace_from_env(self):
        """Verify manager syncs the cluster namespace from its runtime env."""
        with mock.patch.object(Cluster, "NAMESPACE", "original-namespace"):
            with mock.patch.dict(
                os.environ, {"CLUSTER_NAMESPACE": "env-namespace"}, clear=False
            ):
                CollectiveManager()
                assert Cluster.NAMESPACE == "env-namespace"


class TestWorkerGroup:
    """Tests for the WorkerGroup class and its interactions."""

    def test_worker_group_creation(self, cluster: Cluster):
        """Verify that a WorkerGroup can be created successfully."""
        if accelerator_is_available():
            num_workers = cluster.num_accelerators
        else:
            num_workers = 1
        worker_group = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="dist_test_1"
        )

        # Check that the correct number of actors were created
        assert len(worker_group.worker_info_list) == num_workers

        # Verify that we can get results from the workers
        results = worker_group.get_env_info().wait()
        assert len(results) == num_workers
        ranks = sorted([info["rank"] for info in results])
        assert ranks == list(range(num_workers))

    def test_execute_on_all_workers(self, cluster: Cluster):
        """Test calling a method on all workers in a group."""
        if accelerator_is_available():
            num_workers = cluster.num_accelerators
        else:
            num_workers = 1
        worker_group = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="dist_test_2"
        )

        base_value = 10
        results = worker_group.sum_with_rank(base_value).wait()

        assert len(results) == num_workers
        expected_results = sorted([base_value + i for i in range(num_workers)])
        assert sorted(results) == expected_results

    def test_execute_on_specific_ranks(self, cluster: Cluster):
        """Test calling a method on a subset of workers in a group."""
        if accelerator_is_available():
            placement = PackedPlacementStrategy(0, cluster.num_accelerators - 1)
        else:
            placement = NodePlacementStrategy([0] * 8)
        worker_group = DistributedTestWorker.create_group().launch(
            cluster=cluster, placement_strategy=placement, name="dist_test_3"
        )

        target_ranks = (0, 1)
        base_value = 20
        results = (
            worker_group.execute_on(*target_ranks).sum_with_rank(base_value).wait()
        )

        assert len(results) == len(target_ranks)
        expected_results = sorted([base_value + rank for rank in target_ranks])
        assert sorted(results) == expected_results

    def test_multiple_worker_groups(self, cluster: Cluster):
        """Test the creation and operation of multiple independent worker groups."""
        if accelerator_is_available():
            num_workers = cluster.num_accelerators
        else:
            num_workers = 1
        group1 = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="multi_group_1"
        )
        group2 = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="multi_group_2"
        )

        # Call a method on group 1
        results1 = group1.sum_with_rank(100).wait()
        assert len(results1) == num_workers
        assert sorted(results1) == [100 + i for i in range(num_workers)]

        # Call a method on group 2
        results2 = group2.sum_with_rank(200).wait()
        assert len(results2) == num_workers
        assert sorted(results2) == [200 + i for i in range(num_workers)]


class TestLoadUserExtensions:
    """Tests for the Worker._load_user_extensions method."""

    def _create_mock_worker(self):
        """Create a minimal mock worker instance for testing _load_user_extensions."""
        worker = object.__new__(Worker)
        return worker

    def test_no_action_when_env_var_not_set(self):
        """Verify no action is taken when RLINF_EXT_MODULE is not set."""
        worker = self._create_mock_worker()
        os.environ.pop("RLINF_EXT_MODULE", None)

        with mock.patch("importlib.import_module") as mock_import:
            worker._load_user_extensions()
            mock_import.assert_not_called()

    def test_extension_module_loaded_and_register_called(self):
        """Verify extension module is loaded and register() is called."""
        worker = self._create_mock_worker()
        mock_module = types.ModuleType("mock_ext_module")
        mock_module.register = mock.Mock()

        with mock.patch.dict(os.environ, {"RLINF_EXT_MODULE": "mock_ext_module"}):
            with mock.patch("importlib.import_module", return_value=mock_module):
                worker._load_user_extensions()
                mock_module.register.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])


class LocalRayLogCollector(DistributedRayLogCollector):
    """Collector variant using local temp directories for testing."""

    def __init__(self, *args, logs_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self._logs_dir = logs_dir

    def _get_ray_logs_dir(self) -> Path:
        return self._logs_dir

    def _resolve_registered_workers(self, logs_dir: Path) -> None:
        # Keep worker-to-log mapping fully controlled by each test.
        return


def _new_collector(logs_dir: Path, output_dir: Path) -> LocalRayLogCollector:
    logger = logging.getLogger("test.distributed_log_collector")
    return LocalRayLogCollector(
        logger=logger,
        output_dir=output_dir,
        logs_dir=logs_dir,
        poll_interval_s=0.1,
    )


def test_collector_is_pickleable_even_after_start(tmp_path: Path):
    logs_dir = tmp_path / "ray_logs"
    output_dir = tmp_path / "split_logs"
    logs_dir.mkdir(parents=True)

    collector = _new_collector(logs_dir=logs_dir, output_dir=output_dir)
    assert collector.start() is True
    try:
        payload = pickle.dumps(collector)
        restored = pickle.loads(payload)
    finally:
        collector.stop()

    assert isinstance(restored, LocalRayLogCollector)
    assert restored._thread is None
    assert restored._started is False


def test_process_once_reads_from_start_and_appends_incrementally(tmp_path: Path):
    logs_dir = tmp_path / "ray_logs"
    output_dir = tmp_path / "split_logs"
    logs_dir.mkdir(parents=True)

    worker_log = logs_dir / "worker-w1-j1-123.out"
    worker_log.write_text("line-1\nline-2\n", encoding="utf-8")

    collector = _new_collector(logs_dir=logs_dir, output_dir=output_dir)
    collector._log_file_map[worker_log] = ("actor_group:0", "0")

    collector._process_once(logs_dir, new_file_offset_from_start=True)
    out_path = output_dir / "actor_group" / "rank_0.log"
    assert out_path.read_text(encoding="utf-8") == "line-1\nline-2\n"

    with worker_log.open("a", encoding="utf-8") as fp:
        fp.write("line-3\n")
    collector._process_once(logs_dir, new_file_offset_from_start=False)
    assert out_path.read_text(encoding="utf-8") == "line-1\nline-2\nline-3\n"

    collector.stop()


def test_process_once_skips_historical_content_for_new_file_in_loop(tmp_path: Path):
    logs_dir = tmp_path / "ray_logs"
    output_dir = tmp_path / "split_logs"
    logs_dir.mkdir(parents=True)

    worker_log = logs_dir / "worker-w2-j2-456.err"
    worker_log.write_text("old-line\n", encoding="utf-8")

    collector = _new_collector(logs_dir=logs_dir, output_dir=output_dir)
    collector._log_file_map[worker_log] = ("collector_group:1", "1")

    # Background loop behavior: new files start at EOF (historical lines are skipped).
    collector._process_once(logs_dir, new_file_offset_from_start=False)
    out_path = output_dir / "collector_group" / "rank_1.log"
    assert not out_path.exists()

    with worker_log.open("a", encoding="utf-8") as fp:
        fp.write("new-line\n")
    collector._process_once(logs_dir, new_file_offset_from_start=False)
    assert out_path.read_text(encoding="utf-8") == "new-line\n"

    collector.stop()


def test_start_stop_drains_remaining_logs_from_beginning(tmp_path: Path):
    logs_dir = tmp_path / "ray_logs"
    output_dir = tmp_path / "split_logs"
    logs_dir.mkdir(parents=True)

    collector = LocalRayLogCollector(
        logger=logging.getLogger("test.distributed_log_collector.stop"),
        output_dir=output_dir,
        logs_dir=logs_dir,
        poll_interval_s=2.0,
    )
    assert collector.start() is True

    # Let the thread run one iteration with an empty mapping, then wait.
    time.sleep(0.2)

    worker_log = logs_dir / "worker-w3-j3-789.out"
    worker_log.write_text("late-line-1\nlate-line-2\n", encoding="utf-8")
    collector._log_file_map[worker_log] = ("late_group:0", "0")

    # stop() performs drain with new_file_offset_from_start=True, so it should
    # capture complete content even if the background thread did not process it yet.
    collector.stop()

    out_path = output_dir / "late_group" / "rank_0.log"
    assert out_path.read_text(encoding="utf-8") == "late-line-1\nlate-line-2\n"


class CollectorIntegrationWorker(Worker):
    """Worker used for integration testing with Cluster and Ray."""

    def __init__(self):
        super().__init__()

    def emit_test_log(self, token: str) -> int:
        self.log_info(f"collector-integration-token {token}")
        print(f"collector-integration-stdout {token}", flush=True)
        print(f"collector-integration-stderr {token}", file=sys.stderr, flush=True)
        return os.getpid()


def _reset_cluster_singleton() -> None:
    if ray.is_initialized():
        ray.shutdown()
    if hasattr(Cluster, "_instance"):
        instance = getattr(Cluster, "_instance")
        if instance is not None:
            instance._has_initialized = False
        delattr(Cluster, "_instance")
    Cluster.NAMESPACE = Cluster.SYS_NAME


def test_cluster_launch_collects_real_worker_logs(tmp_path: Path):
    out_dir = tmp_path / "cluster_logs"
    token = f"collector-token-{uuid.uuid4().hex}"
    _reset_cluster_singleton()
    tests_root = Path(__file__).resolve().parent
    python_path_entries = [str(tests_root)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_entries.append(existing_pythonpath)
    python_path_value = os.pathsep.join(python_path_entries)
    cluster_cfg = OmegaConf.create(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "train",
                    "node_ranks": "0",
                    "env_configs": [
                        {
                            "node_ranks": "0",
                            "python_interpreter_path": sys.executable,
                            "env_vars": [{"PYTHONPATH": python_path_value}],
                        }
                    ],
                }
            ],
        }
    )

    cluster = None
    worker_group = None
    try:
        cluster = Cluster(cluster_cfg=cluster_cfg, distributed_log_dir=str(out_dir))
        worker_group = CollectorIntegrationWorker.create_group().launch(
            cluster=cluster,
            placement_strategy=NodePlacementStrategy([0], node_group_label="train"),
            name="collector_integration_group",
        )
        pids = worker_group.emit_test_log(token).wait()
        assert len(pids) == 1
        worker_pid = pids[0]

        collector = cluster._distributed_log_collector
        assert collector is not None
        logs_dir = collector._get_ray_logs_dir()
        assert logs_dir is not None

        collector._stop_event.set()
        if collector._thread is not None:
            collector._thread.join(timeout=10)

        candidates = []
        bind_deadline = time.time() + 30
        while time.time() < bind_deadline and len(candidates) == 0:
            candidates = (
                list(logs_dir.glob(f"worker-*-*-{worker_pid}.out"))
                + list(logs_dir.glob(f"worker-*-{worker_pid}.out"))
                + list(logs_dir.glob(f"worker-*-*-{worker_pid}.err"))
                + list(logs_dir.glob(f"worker-*-{worker_pid}.err"))
            )
            if len(candidates) == 0:
                time.sleep(0.5)
        assert len(candidates) > 0, (
            "Did not find Ray worker log files for launched worker pid "
            f"{worker_pid} under {logs_dir}."
        )
        for candidate in candidates:
            collector._log_file_map[candidate] = ("collector_integration_group:0", "0")
            # Drop any offset the background loop may have recorded so the
            # from-start read below begins at byte 0 and captures the token.
            collector._file_offsets.pop(candidate, None)

        target_log = out_dir / "collector_integration_group" / "rank_0.log"
        deadline = time.time() + 30
        content = ""
        while time.time() < deadline:
            collector._process_once(logs_dir, new_file_offset_from_start=True)
            if target_log.exists():
                content = target_log.read_text(encoding="utf-8")
                if token in content:
                    break
            time.sleep(0.5)
        assert token in content, (
            "Did not find emitted integration token in collected worker log "
            f"within timeout. Current content:\n{content}"
        )
        collector.stop()
    finally:
        if worker_group is not None:
            worker_group._close()
        _reset_cluster_singleton()


class FakeProxy:
    """Stand-in for the tracer ManagerProxy that records events in memory."""

    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class TestTracerManager:
    """Server-side test: the Tracer manager writes events to a JSONL file."""

    def test_record_and_finalize(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "trace.jsonl")
            # Construct the manager as a plain object (no Ray actor needed).
            tracer = Tracer(path)
            tracer.record({"name": "a", "ph": "B", "ts": 1, "pid": "p", "tid": "main"})
            tracer.record({"name": "a", "ph": "E", "ts": 2, "pid": "p", "tid": "main"})
            assert tracer.finalize() == os.path.abspath(path)

            with open(path) as f:
                events = [json.loads(line) for line in f]
            assert [e["ph"] for e in events] == ["B", "E"]
            assert all(e["name"] == "a" for e in events)


class TestTracerEmit:
    """Client-side test: the emit API forwards well-formed events, or no-ops."""

    def teardown_method(self):
        Tracer._unavailable = False
        Tracer._labeled = False

    def test_emit_forwards_events(self, monkeypatch):
        proxy = FakeProxy()
        monkeypatch.setattr(Tracer, "_get", classmethod(lambda cls: proxy))
        monkeypatch.setattr(Tracer, "_pid", staticmethod(lambda: "driver"))
        Tracer._labeled = False

        with Tracer.trace_span("step", cat="runner", args={"i": 0}):
            Tracer.trace_begin("inner", cat="actor")
            Tracer.trace_end("inner", cat="actor")

        @Tracer.trace_func(cat="dec")
        def fn():
            return 7

        assert fn() == 7

        # A one-time process_name metadata event labels the process.
        meta = [e for e in proxy.events if e["ph"] == "M"]
        assert len(meta) == 1
        assert meta[0]["args"]["name"] == "driver"

        # Duration events are well-formed and balanced per name.
        spans = [e for e in proxy.events if e["ph"] in ("B", "E")]
        assert all({"name", "cat", "ph", "ts", "pid", "tid"} <= e.keys() for e in spans)
        step_begin = next(e for e in spans if e["name"] == "step" and e["ph"] == "B")
        assert step_begin["cat"] == "runner" and step_begin["args"] == {"i": 0}
        for name in ("step", "inner", "fn"):
            phs = sorted(e["ph"] for e in spans if e["name"] == name)
            assert phs == ["B", "E"]

    def test_disabled_is_noop(self):
        # When the tracer manager is unavailable, every emit API is a no-op.
        Tracer._unavailable = True

        Tracer.trace_begin("noop")
        Tracer.trace_end("noop")
        with Tracer.trace_span("noop"):
            pass

        @Tracer.trace_func
        def fn():
            return 42

        assert fn() == 42
        assert Tracer._get() is None


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])


@pytest.fixture(autouse=True)
def _reset_profiling_flag():
    """Reset module-level flag between tests for isolation."""
    nv_module._nv_profiling_active = False
    yield
    nv_module._nv_profiling_active = False


class TestStartStop:
    """NvidiaGPUManager.start_profiling / stop_profiling drive torch.cuda.profiler."""

    def test_start_sets_active_and_calls_cuda_profiler_start(self):
        with patch("torch.cuda.profiler.start") as mock_start:
            NvidiaGPUManager.start_profiling(step_idx=5)
        assert NvidiaGPUManager.is_profiling_active() is True
        mock_start.assert_called_once_with()

    def test_stop_clears_active_and_calls_cuda_profiler_stop(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()
        with patch("torch.cuda.profiler.stop") as mock_stop:
            NvidiaGPUManager.stop_profiling()
        assert NvidiaGPUManager.is_profiling_active() is False
        mock_stop.assert_called_once_with()

    def test_double_start_is_idempotent(self):
        with patch("torch.cuda.profiler.start") as mock_start:
            NvidiaGPUManager.start_profiling()
            NvidiaGPUManager.start_profiling()
        assert mock_start.call_count == 1

    def test_stop_without_start_is_noop(self):
        with patch("torch.cuda.profiler.stop") as mock_stop:
            NvidiaGPUManager.stop_profiling()
        mock_stop.assert_not_called()


class TestProfilingRangeSync:
    """NvidiaGPUManager.profiling_range on sync code is transparent off / wraps on."""

    def test_passes_through_when_inactive(self):
        results = []
        with NvidiaGPUManager.profiling_range("test/op"):
            results.append(42)
        assert results == [42]
        assert NvidiaGPUManager.is_profiling_active() is False

    def test_emits_range_when_active(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        with (
            patch("torch.cuda.nvtx.range_push") as mock_range_push,
            patch("torch.cuda.nvtx.range_pop") as mock_range_pop,
        ):
            with NvidiaGPUManager.profiling_range("test/op", color="green"):
                pass

        mock_range_push.assert_called_once_with("test/op")
        mock_range_pop.assert_called_once_with()

    def test_ends_range_on_exception(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        with (
            patch("torch.cuda.nvtx.range_push") as mock_range_push,
            patch("torch.cuda.nvtx.range_pop") as mock_range_pop,
        ):
            with pytest.raises(ValueError, match="oops"):
                with NvidiaGPUManager.profiling_range("test/op"):
                    raise ValueError("oops")

        mock_range_push.assert_called_once_with("test/op")
        mock_range_pop.assert_called_once_with()


class TestProfilingRangeAsync:
    """profiling_range works correctly inside async coroutines."""

    def test_passes_through_when_inactive(self):
        async def coro():
            with NvidiaGPUManager.profiling_range("test/async_op"):
                return 99

        assert asyncio.run(coro()) == 99

    def test_emits_range_when_active(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        async def coro():
            with NvidiaGPUManager.profiling_range("test/async_op"):
                return 7

        with (
            patch("torch.cuda.nvtx.range_push") as mock_range_push,
            patch("torch.cuda.nvtx.range_pop") as mock_range_pop,
        ):
            result = asyncio.run(coro())

        assert result == 7
        mock_range_push.assert_called_once_with("test/async_op")
        mock_range_pop.assert_called_once_with()


def test_get_torch_platform_adds_ipc_collect_when_missing(monkeypatch):
    torch_module = ModuleType("torch")
    xpu_platform = SimpleNamespace()
    torch_module.xpu = xpu_platform
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    result = IntelGPUManager.get_torch_platform()

    assert result is xpu_platform
    assert hasattr(xpu_platform, "ipc_collect")
    assert callable(xpu_platform.ipc_collect)
    assert xpu_platform.ipc_collect() is None


def test_get_torch_platform_keeps_existing_ipc_collect(monkeypatch):
    torch_module = ModuleType("torch")
    sentinel = object()

    def _existing_ipc_collect():
        return sentinel

    xpu_platform = SimpleNamespace(ipc_collect=_existing_ipc_collect)
    torch_module.xpu = xpu_platform
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    result = IntelGPUManager.get_torch_platform()

    assert result is xpu_platform
    assert xpu_platform.ipc_collect is _existing_ipc_collect
    assert xpu_platform.ipc_collect() is sentinel


_PATCHER_UNDER_TEST = "_rlinf_utils_patcher_under_test"
_TEST_MODULE_NAMES = (
    "flash_attn",
    "rlinf_test_missing_cuda_dep",
    "rlinf_test_missing_cuda_pkg",
    "rlinf_test_loaded_dep",
)


def _load_patcher_module():
    patcher_path = (
        Path(__file__).resolve().parents[2] / "rlinf" / "utils" / "patcher.py"
    )
    spec = importlib.util.spec_from_file_location(
        _PATCHER_UNDER_TEST,
        patcher_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Patcher = _load_patcher_module().Patcher


def _remove_test_modules() -> None:
    for root_name in _TEST_MODULE_NAMES:
        for module_name in list(sys.modules):
            if module_name == root_name or module_name.startswith(f"{root_name}."):
                del sys.modules[module_name]


@pytest.fixture(autouse=True)
def clean_patcher_state():
    originals = {
        name: sys.modules[name] for name in _TEST_MODULE_NAMES if name in sys.modules
    }
    _remove_test_modules()
    Patcher.clear()
    yield
    Patcher.clear()
    _remove_test_modules()
    sys.modules.update(originals)


def test_skip_import_registers_stub_immediately():
    result = Patcher.skip_import("rlinf_test_missing_cuda_dep")

    assert result is Patcher
    stub_module = sys.modules["rlinf_test_missing_cuda_dep"]
    assert stub_module.__name__ == "rlinf_test_missing_cuda_dep"
    assert stub_module.some_kernel() is None


def test_skip_import_supports_submodule_imports():
    Patcher.skip_import("rlinf_test_missing_cuda_pkg")

    submodule = importlib.import_module("rlinf_test_missing_cuda_pkg.bert_padding")
    namespace = {}
    exec(
        "from rlinf_test_missing_cuda_pkg.bert_padding import unpad_input",
        namespace,
    )

    assert submodule is sys.modules["rlinf_test_missing_cuda_pkg.bert_padding"]
    assert namespace["unpad_input"]() is None


def test_skip_import_preserves_loaded_modules():
    loaded_module = types.ModuleType("rlinf_test_loaded_dep")
    loaded_module.existing_value = object()
    sys.modules["rlinf_test_loaded_dep"] = loaded_module

    Patcher.skip_import("rlinf_test_loaded_dep")

    assert sys.modules["rlinf_test_loaded_dep"] is loaded_module


def test_clear_stub_import_removes_stub_tree_only():
    Patcher.skip_import("rlinf_test_missing_cuda_dep")
    importlib.import_module("rlinf_test_missing_cuda_dep.bert_padding")

    result = Patcher.clear_stub_import("rlinf_test_missing_cuda_dep")

    assert result is Patcher
    assert "rlinf_test_missing_cuda_dep" not in sys.modules
    assert "rlinf_test_missing_cuda_dep.bert_padding" not in sys.modules


def test_clear_stub_import_preserves_real_modules():
    real_module = types.ModuleType("rlinf_test_loaded_dep")
    real_module.__file__ = "/site-packages/rlinf_test_loaded_dep/__init__.py"
    sys.modules["rlinf_test_loaded_dep"] = real_module

    Patcher.clear_stub_import("rlinf_test_loaded_dep")

    assert sys.modules["rlinf_test_loaded_dep"] is real_module


def test_clear_stub_import_removes_flash_attn_stub_tree_only():
    Patcher.skip_import("flash_attn")
    importlib.import_module("flash_attn.bert_padding")

    Patcher.clear_stub_import("flash_attn")

    assert "flash_attn" not in sys.modules
    assert "flash_attn.bert_padding" not in sys.modules


def test_clear_stub_import_preserves_real_flash_attn_module():
    real_flash_attn = types.ModuleType("flash_attn")
    real_flash_attn.__file__ = "/site-packages/flash_attn/__init__.py"
    sys.modules["flash_attn"] = real_flash_attn

    Patcher.clear_stub_import("flash_attn")

    assert sys.modules["flash_attn"] is real_flash_attn


@pytest.fixture
def cpu_openpi_ppo_workers(monkeypatch):
    """Use real worker initialization with a small model/backend boundary."""
    import random

    import numpy as np
    import torch

    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.rollout.hf import huggingface_worker

    # This CPU backend has no remote tracing manager.
    monkeypatch.setattr(Tracer, "_unavailable", True)

    class TinyPolicy(torch.nn.Linear):
        def __init__(self):
            super().__init__(7, 32)
            random.random()
            np.random.rand()

        def predict_action_batch(self, env_obs, **kwargs):
            return torch.randn(2, 5, 7), {
                "python_draw": random.randint(0, 100000),
                "numpy_draw": float(np.random.rand()),
            }

    class CPUActor(EmbodiedFSDPActor):
        def __init__(self, cfg, rank):
            self.cfg, self._rank = cfg, rank
            self.enable_offload = False

        def setup_model_and_optimizer(self):
            self.model = TinyPolicy()
            self.optimizer = torch.optim.AdamW(self.model.parameters())

    class CPURollout(huggingface_worker.MultiStepRolloutWorker):
        def __init__(self, cfg, rank):
            self.cfg, self._rank = cfg, rank
            self.model_cfg = cfg.actor.model
            self.expert_model = self.rlt_feature_model = None
            self.enable_cuda_graph = self.enable_offload = False
            self.enable_opd = self.enable_dagger = False
            self.algorithm_cfg = {}
            self._accelerator_type = None
            self._timer_metrics = {}

    monkeypatch.setattr(huggingface_worker, "get_model", lambda cfg: TinyPolicy())

    def build(role, seed=0, rank=0, model_type="openpi", rollout_seed=None):
        cfg = OmegaConf.create(
            {
                "actor": {"seed": seed, "model": {"model_type": model_type}},
                "rollout": {"model": {}},
                "runner": {},
            }
        )
        if rollout_seed is not None:
            cfg.rollout.seed = rollout_seed
        return (CPUActor if role == "actor" else CPURollout)(cfg, rank)

    return build


def _ppo_seed_draws():
    import random

    import numpy as np
    import torch

    return random.randint(0, 100000), float(np.random.rand()), torch.randn(2, 10, 32)


def _assert_ppo_seed_draws_equal(left, right):
    import torch

    assert left[:2] == right[:2]
    torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)


@pytest.mark.parametrize("role", ["actor", "rollout"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_openpi_ppo_worker_seed_reproduces_model_and_stream(
    cpu_openpi_ppo_workers, role, seed
):
    import torch

    from rlinf.utils.utils import seed_everything

    expected_model = expected_draws = None
    for unrelated_seed in [123, 456]:
        seed_everything(unrelated_seed)
        worker = cpu_openpi_ppo_workers(role, seed=seed, rank=1)
        worker.init_worker()
        model = worker.model if role == "actor" else worker.hf_model
        if expected_model is None:
            expected_model = {k: v.clone() for k, v in model.state_dict().items()}
            expected_draws = _ppo_seed_draws()
        else:
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected_model[key], rtol=0, atol=0)
            _assert_ppo_seed_draws_equal(_ppo_seed_draws(), expected_draws)
        assert torch.initial_seed() == seed + 1


def test_openpi_ppo_actor_ranks_share_critic_initialization(cpu_openpi_ppo_workers):
    import torch

    first = cpu_openpi_ppo_workers("actor", seed=1, rank=0)
    first.init_worker()
    weights = {k: v.clone() for k, v in first.model.state_dict().items()}
    noise = _ppo_seed_draws()[2]
    second = cpu_openpi_ppo_workers("actor", seed=1, rank=1)
    second.init_worker()
    for key, value in second.model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    assert not torch.equal(noise, _ppo_seed_draws()[2])


@pytest.mark.parametrize("rollout_seed", [None, 0, 73])
def test_openpi_rollout_explicit_seed_and_rank_control_sampling(
    cpu_openpi_ppo_workers, rollout_seed
):
    import torch

    from rlinf.utils.utils import seed_everything

    worker = cpu_openpi_ppo_workers(
        "rollout", seed=11, rank=2, rollout_seed=rollout_seed
    )
    worker.init_worker()
    actions, result = worker.predict({}, mode="train")
    expected_seed = (11 if rollout_seed is None else rollout_seed) + 2
    seed_everything(expected_seed)
    expected_actions, expected_result = worker.hf_model.predict_action_batch({})
    torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
    assert result == expected_result | {"expert_label_flag": False}


def test_openpi_rollout_restored_rng_is_not_reseeded_by_prediction(
    cpu_openpi_ppo_workers,
):
    import torch

    from rlinf.utils.utils import get_rng_state, set_rng_state

    worker = cpu_openpi_ppo_workers("rollout", seed=2, rank=1)
    worker.init_worker()
    worker.predict({}, mode="train")
    saved = get_rng_state()
    expected_actions, expected_result = worker.predict({}, mode="eval")
    restored = cpu_openpi_ppo_workers("rollout", seed=2, rank=1)
    restored.init_worker()
    set_rng_state(saved)
    actions, result = restored.predict({}, mode="eval")
    torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
    assert result == expected_result


@pytest.mark.parametrize("role", ["actor", "rollout"])
def test_non_openpi_ppo_worker_preserves_previous_rng_behavior(
    cpu_openpi_ppo_workers, role
):
    import torch

    from rlinf.utils.utils import get_rng_state, seed_everything, set_rng_state

    seed_everything(999)
    initial = get_rng_state()
    worker = cpu_openpi_ppo_workers(role, seed=0, model_type="openvla")
    worker.init_worker()
    actual = _ppo_seed_draws()
    set_rng_state(initial)
    model = worker.model if role == "actor" else worker.hf_model
    type(model)()
    _assert_ppo_seed_draws_equal(_ppo_seed_draws(), actual)
    assert torch.initial_seed() == 999


@pytest.fixture
def cpu_openpi_sft_worker():
    """Exercise the SFT initialization lifecycle with a small CPU model."""
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    class CPUWorker(FSDPVlaSftWorker):
        def __init__(self, seed=0, rank=0, model_type="openpi"):
            self.cfg = OmegaConf.create(
                {
                    "actor": {
                        "seed": seed,
                        "model": {"model_type": model_type},
                        "optim": {"total_training_steps": 30000},
                    }
                }
            )
            self._rank = rank

        def setup_model_and_optimizer(self):
            import torch

            self.model = torch.nn.Linear(7, 32)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2.5e-5)

    return CPUWorker


def _sft_cpu_random_draws():
    import random

    import numpy as np
    import torch

    return (
        random.random(),
        np.random.rand(),
        torch.randn(2, 10, 32),
        torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0)).sample((2,)),
    )


def _assert_sft_random_draws_equal(left, right):
    import torch

    assert left[:2] == right[:2]
    for a, b in zip(left[2:], right[2:], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_openpi_sft_seed_controls_initialization_and_training(
    cpu_openpi_sft_worker, seed
):
    import torch

    from rlinf.utils.utils import seed_everything

    seed_everything(987)
    first = cpu_openpi_sft_worker(seed=seed, rank=1)
    first.init_worker()
    expected_weights = {k: v.clone() for k, v in first.model.state_dict().items()}
    expected_draws = _sft_cpu_random_draws()
    seed_everything(654)
    second = cpu_openpi_sft_worker(seed=seed, rank=1)
    second.init_worker()
    for k, value in second.model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[k], rtol=0, atol=0)
    _assert_sft_random_draws_equal(_sft_cpu_random_draws(), expected_draws)
    assert torch.initial_seed() == seed + 1


def test_openpi_sft_ranks_share_initial_weights_and_have_distinct_noise(
    cpu_openpi_sft_worker,
):
    import torch

    first = cpu_openpi_sft_worker(seed=0, rank=0)
    first.init_worker()
    initial = {k: v.clone() for k, v in first.model.state_dict().items()}
    noise0 = _sft_cpu_random_draws()
    second = cpu_openpi_sft_worker(seed=0, rank=1)
    second.init_worker()
    for k, value in second.model.state_dict().items():
        torch.testing.assert_close(value, initial[k], rtol=0, atol=0)
    assert not torch.equal(noise0[2], _sft_cpu_random_draws()[2])


def test_openpi_sft_checkpoint_rng_supersedes_fresh_initialization(
    cpu_openpi_sft_worker,
):
    from rlinf.utils.utils import get_rng_state, set_rng_state

    worker = cpu_openpi_sft_worker(seed=1, rank=1)
    worker.init_worker()
    _sft_cpu_random_draws()
    saved = get_rng_state()
    expected = _sft_cpu_random_draws()
    restored = cpu_openpi_sft_worker(seed=1, rank=1)
    restored.init_worker()
    set_rng_state(saved)
    _assert_sft_random_draws_equal(_sft_cpu_random_draws(), expected)


def test_other_vla_sft_initialization_preserves_existing_rng_behavior(
    cpu_openpi_sft_worker,
):
    import torch

    from rlinf.utils.utils import get_rng_state, seed_everything, set_rng_state

    seed_everything(999)
    initial = get_rng_state()
    worker = cpu_openpi_sft_worker(seed=0, model_type="openvla")
    worker.init_worker()
    actual = _sft_cpu_random_draws()
    set_rng_state(initial)
    torch.nn.Linear(7, 32)
    _assert_sft_random_draws_equal(_sft_cpu_random_draws(), actual)
    assert torch.initial_seed() == 999


def test_stack_cube_worker_preflight_accepts_complete_formal_sft_export(tmp_path):
    from toolkits.lerobot.verify_stack_cube_workers import checkpoint_actor

    actor = tmp_path / "global_step_2000/actor"
    weights = actor / "model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"export")
    marker = actor / "checkpoint_metadata.json"
    marker.write_text(json.dumps({"step": 2000, "complete": True}))
    assert checkpoint_actor(actor)[0] == actor
    assert checkpoint_actor(actor.parent)[1]["step"] == 2000
    marker.write_text(json.dumps({"step": 2000, "complete": False}))
    with pytest.raises(ValueError, match="complete SFT"):
        checkpoint_actor(actor)
    marker.write_text(json.dumps({"step": 2000, "complete": True}))
    weights.unlink()
    with pytest.raises(ValueError, match="full_weights.pt"):
        checkpoint_actor(actor)


@pytest.fixture
def cpu_ppo_checkpoint(tmp_path):
    import random

    import numpy as np
    import torch
    from torch.distributed.checkpoint import save

    actor = tmp_path / "global_step_5/actor"
    actor.mkdir(parents=True)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    state = {
        "model": model.state_dict(),
        "optimizers": optimizer.state_dict(),
        "lr_schedulers": [scheduler.state_dict()],
        "rng": (
            {
                "cpu": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
        ),
        "fsdp_version": 1,
    }
    save({"fsdp_checkpoint": state}, checkpoint_id=actor / "dcp_checkpoint")
    weights = actor / "model_state_dict/full_weights.pt"
    weights.parent.mkdir()
    torch.save(model.state_dict(), weights)
    return actor


def test_stack_cube_worker_preflight_accepts_native_ppo_export(cpu_ppo_checkpoint):
    from toolkits.lerobot.verify_stack_cube_workers import checkpoint_actor

    actor, metadata = checkpoint_actor(cpu_ppo_checkpoint.parent, checkpoint_kind="ppo")
    assert actor == cpu_ppo_checkpoint
    assert metadata["step"] == 5
    assert metadata["complete"] is None
    assert metadata["completion_marker_present"] is False
    assert metadata["dcp_storage_ranges_complete"] is True
    assert metadata["dcp_backing_files_bytes"]
    with pytest.raises(FileNotFoundError):
        checkpoint_actor(cpu_ppo_checkpoint)


def test_stack_cube_worker_preflight_accepts_marked_ppo_inference_export(tmp_path):
    import torch

    from rlinf.hybrid_engines.fsdp.strategy.checkpoint import (
        validate_checkpoint_for_resume,
    )
    from toolkits.lerobot.verify_stack_cube_workers import checkpoint_actor

    actor = tmp_path / "published_ppo_model"
    weights = actor / "model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    torch.save({"weight": torch.ones(2)}, weights)
    (actor / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "step": 10,
                "complete": True,
                "checkpoint_kind": "ppo",
                "save_training_state": False,
            }
        )
    )
    _, metadata = checkpoint_actor(actor, checkpoint_kind="ppo")
    assert metadata["inference_only_export"] is True
    with pytest.raises(ValueError, match="kind differs"):
        checkpoint_actor(actor)
    with pytest.raises(ValueError, match="inference weights only"):
        validate_checkpoint_for_resume(str(actor))


@pytest.mark.parametrize(
    "fault",
    [
        "truncated",
        "missing_file",
        "missing_rng",
        "sft_marker",
        "incomplete_marker",
        "missing_export",
    ],
)
def test_stack_cube_worker_preflight_rejects_incomplete_ppo_export(
    cpu_ppo_checkpoint, fault
):
    from torch.distributed.checkpoint import FileSystemReader

    from toolkits.lerobot.verify_stack_cube_workers import checkpoint_actor

    actor = cpu_ppo_checkpoint
    directory = actor / "dcp_checkpoint"
    if fault in {"truncated", "missing_file"}:
        shard = next(directory.glob("*.distcp"))
        if fault == "truncated":
            shard.write_bytes(b"")
        else:
            shard.unlink()
    elif fault == "missing_rng":
        metadata = FileSystemReader(directory).read_metadata()
        del metadata.state_dict_metadata["fsdp_checkpoint.rng"]
        with (directory / ".metadata").open("wb") as stream:
            pickle.dump(metadata, stream)
    elif fault == "sft_marker":
        (actor / "openpi_loader_state.json").write_text("{}")
    elif fault == "incomplete_marker":
        (actor / "checkpoint_metadata.json").write_text(
            json.dumps({"step": 5, "complete": False})
        )
    else:
        (actor / "model_state_dict/full_weights.pt").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        checkpoint_actor(actor, checkpoint_kind="ppo")


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "test",
        "ood",
        "duplicate",
        "duplicate_id",
        "missing_id",
        "overlap",
        "short_train",
    ],
)
def test_stack_cube_worker_preflight_keeps_formal_validation_independent(
    tmp_path, fault
):
    from toolkits.lerobot.verify_stack_cube_workers import request_cohorts

    train = [
        {
            "request_id": f"request-{seed}",
            "seed": seed,
            "split": "train",
            "profile": "formal",
        }
        for seed in range(32)
    ]
    val = [
        {
            "request_id": f"request-{seed}",
            "seed": seed,
            "split": "val",
            "profile": "formal",
        }
        for seed in range(100, 127)
    ]
    if fault == "test":
        val[0]["split"] = "test"
    elif fault == "ood":
        val[0]["profile"] = "ood_position"
    elif fault == "duplicate":
        val[1]["seed"] = val[0]["seed"]
    elif fault == "duplicate_id":
        val[1]["request_id"] = val[0]["request_id"]
    elif fault == "missing_id":
        val[0].pop("request_id")
    elif fault == "overlap":
        train[0]["seed"] = val[0]["seed"]
    elif fault == "short_train":
        train.pop()
    train_path, val_path = tmp_path / "train.json", tmp_path / "val.json"
    train_path.write_text(json.dumps({"requests": train}))
    val_path.write_text(json.dumps({"requests": val}))
    if fault is None:
        assert request_cohorts(train_path, val_path) == (train, val)
    else:
        with pytest.raises(ValueError):
            request_cohorts(train_path, val_path)


def test_stack_cube_worker_separate_interpreter_preserves_policy_protocol(tmp_path):
    from omegaconf import OmegaConf

    from toolkits.lerobot.verify_stack_cube_workers import configuration

    paths = {}
    for name in [
        "checkpoint",
        "norm_stats",
        "train_requests",
        "val_requests",
        "panda_urdf",
    ]:
        path = tmp_path / name
        path.mkdir() if name == "checkpoint" else path.write_text("{}")
        paths[name] = path
    arguments = SimpleNamespace(
        **paths,
        dependency_path=tmp_path / "model-reader",
        output=tmp_path / "output",
        gpu=3,
        seed=0,
        dry_run=True,
        env_python=None,
    )
    shared = OmegaConf.to_container(configuration(arguments), resolve=True)
    arguments.env_python = tmp_path / "simulator-python"
    separate = OmegaConf.to_container(configuration(arguments), resolve=True)
    assert len(separate["cluster"]["node_groups"]) == 2
    model_group, sim_group = separate["cluster"]["node_groups"]
    assert sim_group["env_configs"][0]["python_interpreter_path"] == str(
        arguments.env_python
    )
    assert {"PYTHONPATH": str(Path(__file__).resolve().parents[2])} in sim_group[
        "env_configs"
    ][0]["env_vars"]
    assert (
        separate["cluster"]["component_placement"]["actor,rollout"]["node_group"]
        == model_group["label"]
    )
    assert (
        separate["cluster"]["component_placement"]["env"]["node_group"]
        == sim_group["label"]
    )
    for cfg in (shared, separate):
        cfg["cluster"].pop("node_groups")
        cfg["cluster"].pop("component_placement")
    assert shared == separate
