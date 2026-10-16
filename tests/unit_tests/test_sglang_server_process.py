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

"""Unit tests for the sglang server subprocess parent watchdog.

Covers the R13.7 fix: the spawned sglang server child must die — together
with its whole process group — once the process that spawned it is gone,
so a SIGKILLed Ray actor cannot leak the server tree (dp controller,
detokenizer, schedulers).

No sglang is launched. The real ``_run_sglang_server`` entrypoint is
exercised with ``server_type="srt"``; its heavyweight imports only happen
after the watchdog is up, so a watchdog kill wins the race and the child
dies without ever loading sglang.
"""

import multiprocessing as mp
import os
import signal
import time

from rlinf.workers.rollout.sglang_server import process as proc_mod

# Watchdog interval; tests wait a multiple of it plus slack.
WATCH_INTERVAL = proc_mod._PARENT_WATCH_INTERVAL


def _spawner_entry(child_pid_queue: mp.Queue) -> None:
    """Intermediate process: spawn the server child, report its pid, wait.

    Mirrors a Ray actor: it owns the server child but has no cleanup
    callback that runs on SIGKILL.
    """
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=proc_mod._run_sglang_server,
        args=("srt", {"model_path": "/stub"}, 0, None),
        kwargs={
            "apply_rlinf_patch": False,
            "dist_init_addr": None,
            "parent_pid": os.getpid(),
        },
        daemon=False,
    )
    proc.start()
    child_pid_queue.put(proc.pid)
    proc.join()


def _legacy_spawner_entry(child_pid_queue: mp.Queue) -> None:
    """Spawn the server child the legacy way: no ``parent_pid`` passed."""
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=proc_mod._run_sglang_server,
        args=("srt", {"model_path": "/stub"}, 0, None),
        kwargs={"apply_rlinf_patch": False, "dist_init_addr": None},
        daemon=False,
    )
    proc.start()
    child_pid_queue.put(proc.pid)
    proc.join()


def _watchdog_entry(parent_pid: int) -> None:
    """Run the real watchdog helper in its own process group."""
    os.setpgrp()
    proc_mod._watch_parent(parent_pid)


def _pgid_of(pid: int) -> int:
    with open(f"/proc/{pid}/stat") as f:
        stat = f.read()
    return int(stat[stat.rfind(")") + 2 :].split()[2])


def _group_alive(pgids: set[int]) -> bool:
    """Whether any live (non-zombie) process is in one of the groups."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
            fields = stat[stat.rfind(")") + 2 :].split()
            pgrp, state = int(fields[2]), fields[0]
        except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
            continue
        if pgrp in pgids and state != "Z":
            return True
    return False


def _wait_group_gone(pgids: set[int], timeout: float) -> bool:
    deadline = time.time() + timeout
    while _group_alive(pgids) and time.time() < deadline:
        time.sleep(0.5)
    return not _group_alive(pgids)


def test_child_dies_when_parent_killed():
    """SIGKILL the spawner; the server child's whole group must die."""
    ctx = mp.get_context("spawn")
    child_pid_queue = ctx.Queue()
    spawner = ctx.Process(target=_spawner_entry, args=(child_pid_queue,), daemon=False)
    spawner.start()
    child_pid = child_pid_queue.get(timeout=60)

    # Let the child settle: setpgrp + watchdog thread up (its sglang
    # imports may still be in flight; the watchdog kills regardless).
    time.sleep(2 * WATCH_INTERVAL)
    child_pgid = _pgid_of(child_pid)
    assert _group_alive({child_pgid}), "server child group should be alive"

    # SIGKILL the spawner — no cleanup callback ever runs.
    os.kill(spawner.pid, signal.SIGKILL)
    spawner.join(timeout=10)

    assert _wait_group_gone({child_pgid}, timeout=10 * WATCH_INTERVAL + 10), (
        "sglang server process group must die once its spawner is SIGKILLed"
    )


def test_child_survives_while_parent_alive():
    """The watchdog must not fire while the parent is alive."""
    ctx = mp.get_context("spawn")
    child_pid_queue = ctx.Queue()
    spawner = ctx.Process(target=_spawner_entry, args=(child_pid_queue,), daemon=False)
    spawner.start()
    child_pid = child_pid_queue.get(timeout=60)
    try:
        # Wait well past several watchdog intervals.
        time.sleep(4 * WATCH_INTERVAL + 1)
        _pgid_of(child_pid)  # raises if the child died
        assert spawner.is_alive()
    finally:
        spawner.terminate()
        spawner.join(timeout=10)
        # Best-effort cleanup of anything this test left behind.
        try:
            os.killpg(_pgid_of(child_pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, FileNotFoundError):
            pass


def test_watchdog_precheck_kills_group():
    """The watchdog's pre-check fires immediately on a wrong parent pid."""
    ctx = mp.get_context("spawn")
    holder = ctx.Process(
        target=_watchdog_entry,
        args=(os.getpid() + 100000,),  # never matches anyone real
        daemon=False,
    )
    holder.start()
    holder.join(timeout=15)
    assert holder.exitcode in (-signal.SIGKILL, 255), (
        f"watchdog child should have been SIGKILLed, got {holder.exitcode}"
    )


def test_no_watchdog_without_parent_pid():
    """Legacy call site (no ``parent_pid``) must not start the watchdog.

    The spawned child stays alive with its parent untouched until the
    test tears it down explicitly.
    """
    ctx = mp.get_context("spawn")
    child_pid_queue = ctx.Queue()
    spawner = ctx.Process(
        target=_legacy_spawner_entry, args=(child_pid_queue,), daemon=False
    )
    spawner.start()
    child_pid = child_pid_queue.get(timeout=60)
    try:
        # Well past several watchdog intervals: the child must still be
        # alive (no pre-check fired) and its parent (the spawner) too.
        time.sleep(4 * WATCH_INTERVAL + 1)
        _pgid_of(child_pid)  # raises if the child died
        assert spawner.is_alive()
    finally:
        spawner.terminate()
        spawner.join(timeout=10)
        try:
            os.killpg(_pgid_of(child_pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, FileNotFoundError):
            pass
