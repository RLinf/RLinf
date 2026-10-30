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

"""Cross-node rendezvous and readiness signaling for sglang server mode.

One model instance spanning N nodes must launch its sglang server processes
with an identical ``dist_init_addr`` pointing at a free port on the entry
(node_rank 0) node. Everything except that host:port pair is locally
derivable, so the only inter-process exchange is a single broadcast.

The same instance group carries two more signals from the entry: "ready"
or "failed" after startup, so non-entry ranks fail fast instead of timing
out; and "rollout finished" after each step's offload, which non-entry
ranks wait for while holding their device lock.
"""

import asyncio
from typing import Callable, Optional

import ray

from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.placement import (
    ModelParallelComponentPlacement,
    ModelParallelEvalComponentPlacement,
)

_Placement = ModelParallelComponentPlacement | ModelParallelEvalComponentPlacement

_DEFAULT_TIMEOUT_S = 300.0
"""Rendezvous timeout.

Generous on purpose: the broadcast only completes once every node's rollout
worker reaches ``initialize()``, which includes importing heavy modules.
(CollectiveGroupOptions has no timeout knob, so we enforce it ourselves.)
"""

_READINESS_TIMEOUT_S = 1200.0
"""Non-entry readiness-wait timeout.

Same order of magnitude as the entry's ``/health`` wait (900s in
``_wait_for_http_health``) plus margin for the entry's worker-registration
RPC: if the entry is healthy but slow, non-entry ranks must not time out
first.
"""

_READY = "ready"
_FAILED = "failed"
_ROLLOUT_DONE = "rollout_done"
_REASON_TRUNCATE = 500


def _instance_comm_group(
    worker: Worker, placement: _Placement
) -> tuple[list, tuple, int]:
    """Derive the instance-internal broadcast (groups, src, instance id).

    All ranks of one model instance derive identical values locally from
    the placement (single source of truth; consecutive ranks per instance,
    entry at node_rank 0).
    """
    nnodes = placement.rollout_nnodes_per_model_instance
    model_instance_id = placement.rollout_model_instance_id(worker._rank)
    group_name = worker.worker_address.root_group_name
    entry_rank = placement.rollout_model_instance_entry_process_rank(model_instance_id)
    ranks = [entry_rank + i for i in range(nnodes)]
    return [(group_name, ranks)], (group_name, entry_rank), model_instance_id


async def negotiate_model_instance_dist_addr(
    worker: Worker,
    placement: _Placement,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> tuple[str, int]:
    """Negotiate the shared rendezvous address of one model instance.

    Returns ``(host, port)`` - a 2-tuple, NOT a pre-joined "host:port"
    string: sglang wants one field (``dist_init_addr``) while the future
    vLLM path wants two separate flags; each caller formats its own.

    Args:
        worker: This rollout worker process (any rank of the instance).
        placement: The rollout component placement. Single source of
            truth: instance id, node rank and the entry process rank are
            all derived from ``worker._rank`` through it, so the local
            formula "entry rank = instance id * nnodes" exists only here
            via ``rollout_model_instance_entry_process_rank``.
        timeout_s: Fail (rather than hang forever) if any rank of the
            instance does not reach this call within the budget.

    Returns:
        (host, port) of the entry node. Single-node-per-instance layouts
        get ("127.0.0.1", free_port).
    """
    if placement.rollout_nnodes_per_model_instance == 1:
        # Degenerate case: no rendezvous, no broadcast, loopback address.
        return ("127.0.0.1", worker.acquire_free_port())

    groups, src, _ = _instance_comm_group(worker, placement)

    if placement.rollout_node_rank_in_model_instance(worker._rank) == 0:
        payload = (
            ray.util.get_node_ip_address(),
            worker.acquire_free_port(),
        )
    else:
        payload = None

    work = worker.broadcast(object=payload, groups=groups, src=src, async_op=True)
    try:
        result = await asyncio.wait_for(work.async_wait(), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        # AsyncFuncWork cannot be cancelled; its thread stays blocked,
        # acceptable because the run stops on this error.
        nnodes = placement.rollout_nnodes_per_model_instance
        model_instance_id = placement.rollout_model_instance_id(worker._rank)
        node_rank = placement.rollout_node_rank_in_model_instance(worker._rank)
        group_ranks = groups[0][1]
        raise RuntimeError(
            f"Rendezvous for model instance {model_instance_id} timed out "
            f"after {timeout_s}s (node_rank {node_rank} of "
            f"{nnodes}, worker ranks {group_ranks}). Another "
            f"node of this instance likely failed to reach initialize()."
        ) from e
    return result


async def broadcast_instance_readiness(
    worker: Worker,
    placement: _Placement,
    *,
    ready: bool,
    reason: Optional[str] = None,
    timeout_s: float = 60.0,
) -> None:
    """Entry-side readiness signal for the rest of the instance.

    Sends ``("ready",)`` or ``("failed", reason)`` over the instance's
    internal communication group. Entry rank only; must NOT be entered on
    single-node layouts (the group never forms there).

    Raises RuntimeError on timeout - for the "ready" signal that means the
    instance is broken anyway; callers broadcasting "failed" typically wrap
    this call best-effort so the original error is not masked.
    """
    groups, src, _ = _instance_comm_group(worker, placement)
    payload = (_READY,) if ready else (_FAILED, (reason or "")[:_REASON_TRUNCATE])
    work = worker.broadcast(object=payload, groups=groups, src=src, async_op=True)
    try:
        await asyncio.wait_for(work.async_wait(), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise RuntimeError(
            f"Readiness signal ({_READY if ready else _FAILED}) broadcast "
            f"timed out after {timeout_s}s; another rank of this model "
            f"instance is not participating in the broadcast."
        ) from e


async def _wait_for_entry_signal(
    worker: Worker,
    placement: _Placement,
    *,
    child_alive: Callable[[], bool],
    poll_child_failure: Callable[[], Optional[tuple[Optional[int], Optional[str]]]],
    handle_payload: Callable[[object], None],
    timeout_s: Optional[float],
    timeout_error: Callable[[], RuntimeError],
) -> None:
    """Wait for the entry's broadcast while watching this rank's subprocess.

    Shared by the readiness wait and the per-step rollout wait: the
    broadcast from the entry rank of this model instance races a liveness
    poll of this rank's own sglang server subprocess. Whichever completes
    first wins:

    - broadcast arrives -> ``handle_payload(payload)`` decides the outcome;
    - own subprocess exits first -> raise, naming the instance, node rank,
      exit code and whatever ``launch_server`` wrote to the ready pipe
      (may be empty; the full traceback lives in this rank's worker log).

    Args:
        worker: This (non-entry) rollout worker process.
        placement: The rollout component placement (layout source).
        child_alive: Liveness probe for this rank's sglang server
            subprocess (see ``SGLangServerProcess.is_child_alive``).
        poll_child_failure: Dead-child probe returning
            ``(exitcode, pipe_error)`` or ``None`` while alive.
        handle_payload: Validates the broadcast payload; raises on
            unexpected payloads, returns for accepted ones.
        timeout_s: Total budget or ``None`` to wait indefinitely (the
            per-step rollout wait has no reliable upper bound).
        timeout_error: Builds the error raised when ``timeout_s`` elapses
            with neither the broadcast nor a subprocess exit observed;
            must be non-``None`` whenever ``timeout_s`` is not ``None``.
    """
    assert (timeout_s is None) == (timeout_error is None), (
        f"timeout_s and timeout_error must be set together (timeout_s={timeout_s!r})."
    )
    groups, src, model_instance_id = _instance_comm_group(worker, placement)
    node_rank = placement.rollout_node_rank_in_model_instance(worker._rank)
    nnodes = placement.rollout_nnodes_per_model_instance
    work = worker.broadcast(object=None, groups=groups, src=src, async_op=True)

    async def _poll_child() -> Optional[tuple[Optional[int], Optional[str]]]:
        while True:
            if not child_alive():
                return poll_child_failure()
            await asyncio.sleep(1.0)

    waiter = asyncio.ensure_future(work.async_wait())
    poller = asyncio.ensure_future(_poll_child())
    try:
        done, _ = await asyncio.wait(
            {waiter, poller}, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
        )
        if poller in done:
            # Do NOT cancel the broadcast work (AsyncFuncWork has no
            # cancel; the run is expected to stop after this error).
            failure = poller.result()
            exitcode, child_error = failure if failure else (None, None)
            raise RuntimeError(
                f"sglang server subprocess of model instance "
                f"{model_instance_id} exited on node_rank {node_rank} of "
                f"{nnodes} (exitcode {exitcode}): "
                f"{child_error or '<no exception captured>'}. See this "
                f"rank's worker log for the full traceback."
            )
        if waiter in done:
            handle_payload(waiter.result())
            return
        raise timeout_error()
    finally:
        poller.cancel()


async def wait_for_instance_readiness(
    worker: Worker,
    placement: _Placement,
    *,
    child_alive: Callable[[], bool],
    poll_child_failure: Callable[[], Optional[tuple[Optional[int], Optional[str]]]],
    timeout_s: float = _READINESS_TIMEOUT_S,
) -> None:
    """Non-entry-side wait for the entry's readiness signal.

    Races three outcomes instead of returning after the 2s spawn check:

    - entry broadcast ``("ready",)`` -> return normally;
    - entry broadcast ``("failed", reason)`` -> raise, naming the entry;
    - own sglang subprocess exits first -> raise with the instance id,
      node rank, exit code and whatever ``launch_server`` wrote to the
      ready pipe (may be empty; full traceback lives in this rank's
      worker log).

    Args:
        worker: This (non-entry) rollout worker process.
        placement: The rollout component placement (layout source).
        child_alive: Liveness probe for this rank's sglang server
            subprocess (see ``SGLangServerProcess.is_child_alive``).
        poll_child_failure: Dead-child probe returning
            ``(exitcode, pipe_error)`` or ``None`` while alive.
        timeout_s: Total budget; same order of magnitude as the entry's
            ``/health`` wait so a slow-but-healthy entry is not mistaken
            for a failure.
    """
    groups, src, model_instance_id = _instance_comm_group(worker, placement)
    node_rank = placement.rollout_node_rank_in_model_instance(worker._rank)
    nnodes = placement.rollout_nnodes_per_model_instance

    def _handle_readiness(payload: object) -> None:
        if isinstance(payload, tuple) and payload[:1] == (_READY,):
            return
        if isinstance(payload, tuple) and payload[:1] == (_FAILED,):
            raise RuntimeError(
                f"Entry node of model instance {model_instance_id} "
                f"failed to start: {payload[1]}"
            )
        raise RuntimeError(
            f"Unexpected readiness payload {payload!r} from the entry "
            f"of model instance {model_instance_id}."
        )

    def _timeout_error() -> RuntimeError:
        return RuntimeError(
            f"Readiness wait for model instance {model_instance_id} timed "
            f"out after {timeout_s}s (node_rank {node_rank} of {nnodes}): "
            f"the entry node's /health neither passed nor reported "
            f"failure, and the local sglang subprocess is still alive."
        )

    await _wait_for_entry_signal(
        worker,
        placement,
        child_alive=child_alive,
        poll_child_failure=poll_child_failure,
        handle_payload=_handle_readiness,
        timeout_s=timeout_s,
        timeout_error=_timeout_error,
    )


async def signal_rollout_finished(
    worker: Worker,
    placement: _Placement,
) -> None:
    """Entry side: tell the non-entry ranks this step's rollout is done.

    Sent on every step after offload, or without it when nothing is
    offloaded. No timeout: every rank of the instance enters ``rollout()``
    on every step.
    """
    groups, src, _ = _instance_comm_group(worker, placement)
    work = worker.broadcast(
        object=(_ROLLOUT_DONE,), groups=groups, src=src, async_op=True
    )
    await work.async_wait()


async def wait_for_rollout_finished(
    worker: Worker,
    placement: _Placement,
    *,
    child_alive: Callable[[], bool],
    poll_child_failure: Callable[[], Optional[tuple[Optional[int], Optional[str]]]],
) -> None:
    """Non-entry side: wait for the entry's rollout-finished signal.

    No timeout, since a step's rollout has no reliable upper bound; a
    failing entry surfaces through Ray. Watching this rank's own sglang
    subprocess reports a local server death within seconds, which the
    entry would only see later as a stuck collective.
    """
    _, _, model_instance_id = _instance_comm_group(worker, placement)

    def _handle_payload(payload: object) -> None:
        if not (isinstance(payload, tuple) and payload[:1] == (_ROLLOUT_DONE,)):
            raise RuntimeError(
                f"Unexpected rollout-finished payload {payload!r} from the "
                f"entry of model instance {model_instance_id}."
            )

    await _wait_for_entry_signal(
        worker,
        placement,
        child_alive=child_alive,
        poll_child_failure=poll_child_failure,
        handle_payload=_handle_payload,
        timeout_s=None,
        timeout_error=None,
    )
