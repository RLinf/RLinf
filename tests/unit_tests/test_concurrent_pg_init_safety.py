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


from __future__ import annotations

import gc
import multiprocessing as mp
import socket
import threading
from datetime import timedelta

import pytest
import torch.distributed as dist

from rlinf.scheduler.collective.multi_channel_pg import (
    MultiChannelProcessGroup,
    _process_group_lifecycle_guard,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _stress_concurrent_create_process_group(
    num_threads: int = 16,
    trials: int = 4,
    iters_per_trial: int = 8,
) -> dict:
    """Create/destroy GLOO PGs from many threads; must survive without abort."""
    errors: list[str] = []
    lock = threading.Lock()

    for trial in range(trials):
        # Unique ports per thread: mirrors production peer-group rebuilds and
        # avoids TCP port reuse stalls when create/destroy is serialized.
        ports = [_free_port() for _ in range(num_threads)]
        barrier = threading.Barrier(num_threads)

        def worker(thread_id: int) -> None:
            port = ports[thread_id]
            for step in range(iters_per_trial):
                barrier.wait()
                try:
                    pg = MultiChannelProcessGroup._create_process_group(
                        backend="gloo",
                        init_method=f"tcp://127.0.0.1:{port}",
                        world_size=1,
                        rank=0,
                        group_name=f"safe_gloo_{thread_id}_{trial}_{step}",
                        timeout=timedelta(seconds=5),
                    )
                    try:
                        with _process_group_lifecycle_guard():
                            dist.destroy_process_group(pg)
                    except Exception:
                        pass
                    del pg
                except Exception as exc:
                    with lock:
                        errors.append(repr(exc))
                barrier.wait()
                gc.collect()

        threads = [
            threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    return {
        "ok": True,
        "errors": errors[:20],
        "num_errors": len(errors),
    }


def _stress_child(result_queue: mp.Queue) -> None:
    """Child entrypoint so SIGABRT cannot tear down the pytest process."""
    result_queue.put(_stress_concurrent_create_process_group())


def _run_stress_in_subprocess(timeout_s: float = 180.0) -> dict:
    """Run PG stress in a spawned child and surface abort as a pytest failure."""
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_stress_child, args=(result_queue,))
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=10)
        pytest.fail(
            f"Concurrent GLOO ProcessGroup stress timed out after {timeout_s}s"
        )
    if proc.exitcode != 0:
        # Negative exitcode means killed by signal (-6 == SIGABRT).
        pytest.fail(
            "Child aborted during concurrent GLOO ProcessGroup init "
            f"(exitcode={proc.exitcode}); likely pybind11_object_dealloc SIGABRT."
        )
    assert not result_queue.empty(), "Child exited 0 but produced no result"
    return result_queue.get()


class TestConcurrentProcessGroupInitSafety:
    """Concurrent GLOO PG init must not abort the process when serialization is on."""

    def test_concurrent_gloo_pg_init_does_not_abort_worker(self):
        # Run in a subprocess so a pybind SIGABRT is reported via exitcode
        # instead of killing pytest through WorkerGroup's SIGUSR1 failure path.
        results = _run_stress_in_subprocess()
        assert results["ok"] is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
