# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the collection launcher, Ray owner and worker without robot hardware."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


def simulated_driver():
    import ray

    import rlinf.envs.real as real
    from rlinf.scheduler.worker.worker_group import WorkerGroupFuncResult

    real.RealWorldEnv = object
    from examples.embodiment.collect_real_data import DataCollector, _run_collection

    markers = Path(os.environ["RLINF_LOG_DIR"])

    @ray.remote
    class SimulatedCollector:
        def __init__(self):
            self.collector = object.__new__(DataCollector)
            c = self.collector
            c._run_future = None
            c._resources_closed = False
            c._stop_requested = threading.Event()
            c.log_info = print
            release = threading.Event()

            def wait_for_b():
                (markers / "waiting").touch()
                if not release.wait(60):
                    raise RuntimeError("stop RPC never arrived")
                raise KeyboardInterrupt

            c._run_collection = wait_for_b
            c.env = SimpleNamespace(
                request_stop=release.set, close=lambda: (markers / "parked").touch()
            )
            c.replay_buffer = SimpleNamespace(
                close=lambda: (markers / "flushed").touch()
            )

        async def run(self):
            return await self.collector.run()

        async def stop(self):
            return await self.collector.stop()

    # Use a private Ray instance. Never submit work to an existing robot cluster.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ray-sigint-") as ray_dir:
        ray.init(
            address="local",
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            _temp_dir=ray_dir,
        )
        try:
            actor = SimulatedCollector.remote()

            def handle(method):
                return WorkerGroupFuncResult(
                    None, [method.remote()], "collection", "test"
                )

            _run_collection(
                lambda: SimpleNamespace(
                    run=lambda: handle(actor.run), stop=lambda: handle(actor.stop)
                )
            )
            assert (markers / "parked").exists() and (markers / "flushed").exists()
            print("parking acknowledged before driver exit", flush=True)
        finally:
            ray.shutdown()


def test_collect_launcher_sigint(tmp_path):
    root = Path(__file__).resolve().parents[3]
    # Only substitute the driver's hardware setup; keep the real shell pipeline.
    shim = tmp_path / "python"
    shim.write_text('#!/bin/sh\nexec "$TEST_PYTHON" "$TEST_DRIVER"\n')
    shim.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "TEST_PYTHON": sys.executable,
        "TEST_DRIVER": str(Path(__file__).resolve()),
        "RLINF_LOG_DIR": str(tmp_path),
        "RAY_ADDRESS": "local",
    }
    proc = subprocess.Popen(
        ["bash", "examples/embodiment/collect_data.sh"],
        cwd=root,
        env=env,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 90
        while not (tmp_path / "waiting").exists():
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        assert os.getpgid(proc.pid) == proc.pid
        os.killpg(proc.pid, signal.SIGINT)
        assert proc.wait(timeout=30) == 0
        assert (
            "parking acknowledged before driver exit"
            in (tmp_path / "run_embodiment.log").read_text()
        )
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)


if __name__ == "__main__":
    simulated_driver()
