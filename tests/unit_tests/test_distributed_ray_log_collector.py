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

import logging
import os
import pickle
import sys
import time
import uuid
from pathlib import Path

import ray
from omegaconf import OmegaConf

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.scheduler.cluster.utils import DistributedRayLogCollector


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
