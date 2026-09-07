# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import inspect
import logging
from types import MethodType
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from rlinf.config import validate_weight_sync_overlap_cfg
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
from rlinf.runners.async_weight_sync_mixin import AsyncWeightSyncMixin
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)

# Both async runners must pick the overlap up from the mixin, not from their own
# copy of it.
RUNNER_CLASSES = (AsyncEmbodiedRunner, AsyncPPOEmbodiedRunner)


class _Handle:
    def __init__(self, done: bool = False) -> None:
        self.is_done = done
        self.wait_calls = 0

    def done(self) -> bool:
        return self.is_done

    def wait(self) -> None:
        self.wait_calls += 1
        self.is_done = True


class _Actor:
    def __init__(self, handles: list[_Handle]) -> None:
        self.handles = handles
        self.sync_calls = 0

    def sync_model_to_rollout(self) -> _Handle:
        handle = self.handles[self.sync_calls]
        self.sync_calls += 1
        return handle


class _Rollout:
    def __init__(
        self,
        blocking_handles: list[_Handle],
        background_handles: list[_Handle],
    ) -> None:
        self.blocking_handles = blocking_handles
        self.background_handles = background_handles
        self.blocking_calls = 0
        self.background_calls = 0

    def sync_model_from_actor(self) -> _Handle:
        handle = self.blocking_handles[self.blocking_calls]
        self.blocking_calls += 1
        return handle

    def request_actor_sync_model(self) -> _Handle:
        handle = self.background_handles[self.background_calls]
        self.background_calls += 1
        return handle


def _make_runner(runner_cls, actor: _Actor, rollout: _Rollout, no_wait: bool = False):
    """Build a runner with only the attributes the weight-sync path touches."""
    runner = object.__new__(runner_cls)
    runner.actor = actor
    runner.rollout = rollout
    runner.logger = logging.getLogger("test_async_weight_sync")
    runner.sync_weight_no_wait = no_wait
    runner._pending_rollout_weight_sync = None
    runner._weight_sync_request_total = 0
    runner._weight_sync_coalesced_total = 0
    return runner


class _LifecycleWorker:
    """Worker-group stub for a zero-step runner lifecycle."""

    def interact(self, **kwargs) -> _Handle:
        return _Handle()

    def generate(self, **kwargs) -> _Handle:
        return _Handle()

    def recv_rollout_trajectories(self, **kwargs) -> _Handle:
        return _Handle()

    def stop(self) -> _Handle:
        return _Handle()


def _make_async_rollout_worker(
    apply_gates: list[asyncio.Event],
) -> tuple[AsyncMultiStepRolloutWorker, list[asyncio.Event]]:
    """Build only the rollout-side background-sync state machine."""
    worker = object.__new__(AsyncMultiStepRolloutWorker)
    worker._background_weight_sync_active = True
    worker._weight_sync_requested = False
    worker._weight_sync_work = None
    worker._weight_sync_apply_total = 0
    worker._weight_sync_coalesced_total = 0
    worker._weight_sync_request_total = 0
    worker.version = 0
    apply_started = [asyncio.Event() for _ in apply_gates]

    async def recv_and_apply(worker_self) -> int:
        apply_index = worker_self.version
        apply_started[apply_index].set()
        await apply_gates[apply_index].wait()
        worker_self.version += 1
        return worker_self.version

    worker._recv_and_apply_actor_sync = MethodType(recv_and_apply, worker)
    return worker, apply_started


async def _request_actor_sync_model(worker: AsyncMultiStepRolloutWorker) -> int:
    """Call the undecorated worker method without constructing a Ray worker."""
    request = inspect.unwrap(AsyncMultiStepRolloutWorker.request_actor_sync_model)
    return await request(worker)


def _weight_sync_cfg(no_wait: bool, syncer_type: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "actor": {"sync_weight_no_wait": no_wait},
            "weight_syncer": {"type": syncer_type},
        }
    )


@pytest.mark.parametrize("syncer_type", ["patch", "bucket"])
def test_mixin_reads_the_flag_for_either_syncer(syncer_type) -> None:
    """The mixin only reads the flag; the config layer rejects bad combos."""
    runner = object.__new__(AsyncWeightSyncMixin)
    runner.cfg = _weight_sync_cfg(no_wait=True, syncer_type=syncer_type)

    runner.init_weight_sync_state()

    assert runner.sync_weight_no_wait
    assert runner._pending_rollout_weight_sync is None


def test_nonblocking_weight_sync_requires_patch_syncer() -> None:
    with pytest.raises(AssertionError, match="weight_syncer.type=patch"):
        validate_weight_sync_overlap_cfg(
            _weight_sync_cfg(no_wait=True, syncer_type="bucket")
        )


def test_nonblocking_weight_sync_accepts_patch_syncer() -> None:
    validate_weight_sync_overlap_cfg(
        _weight_sync_cfg(no_wait=True, syncer_type="patch")
    )


def test_blocking_weight_sync_allows_bucket_syncer() -> None:
    validate_weight_sync_overlap_cfg(
        _weight_sync_cfg(no_wait=False, syncer_type="bucket")
    )


def test_async_embodied_runner_startup_sync_is_blocking() -> None:
    runner = object.__new__(AsyncEmbodiedRunner)
    runner.global_step = 0
    runner.max_steps = 0
    runner.actor = _LifecycleWorker()
    runner.rollout = _LifecycleWorker()
    runner.env = _LifecycleWorker()
    runner.reward = None
    runner.env_channel = None
    runner.rollout_channel = None
    runner.reward_channel = None
    runner.actor_channel = None
    runner.env_metric_channel = None
    runner.rollout_metric_channel = None
    runner.update_rollout_weights = MagicMock()
    runner.drain_pending_rollout_weight_sync = MagicMock()

    runner.run()

    runner.update_rollout_weights.assert_called_once_with()
    runner.drain_pending_rollout_weight_sync.assert_called_once_with()


def test_rollout_request_completes_only_after_weights_are_applied() -> None:
    async def run_test() -> None:
        apply_gate = asyncio.Event()
        worker, apply_started = _make_async_rollout_worker([apply_gate])

        request_task = asyncio.create_task(_request_actor_sync_model(worker))
        await asyncio.wait_for(apply_started[0].wait(), timeout=5.0)
        await asyncio.sleep(0)

        assert not request_task.done()
        assert worker._weight_sync_apply_total == 0
        assert worker.version == 0

        apply_gate.set()

        assert await asyncio.wait_for(request_task, timeout=5.0) == 1
        assert worker._weight_sync_apply_total == 1
        assert worker._weight_sync_work is None
        assert worker.version == 1

    asyncio.run(run_test())


def test_rollout_requests_coalesce_without_reordering_apply() -> None:
    async def run_test() -> None:
        first_apply_gate = asyncio.Event()
        second_apply_gate = asyncio.Event()
        worker, apply_started = _make_async_rollout_worker(
            [first_apply_gate, second_apply_gate]
        )

        first_request = asyncio.create_task(_request_actor_sync_model(worker))
        await asyncio.wait_for(apply_started[0].wait(), timeout=5.0)
        second_request = asyncio.create_task(_request_actor_sync_model(worker))
        await asyncio.sleep(0)

        assert worker._weight_sync_coalesced_total == 1
        assert not first_request.done()
        assert not second_request.done()

        first_apply_gate.set()
        assert await asyncio.wait_for(first_request, timeout=5.0) == 1
        await asyncio.wait_for(apply_started[1].wait(), timeout=5.0)
        assert not second_request.done()
        assert worker.version == 1

        second_apply_gate.set()
        assert await asyncio.wait_for(second_request, timeout=5.0) == 2
        assert worker._weight_sync_apply_total == 2
        assert worker._weight_sync_work is None
        assert worker.version == 2

    asyncio.run(run_test())


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_runner_inherits_the_shared_mixin(runner_cls) -> None:
    assert issubclass(runner_cls, AsyncWeightSyncMixin)
    # The overlap must resolve to the mixin, i.e. no runner-local reimplementation.
    assert (
        runner_cls.update_rollout_weights is AsyncWeightSyncMixin.update_rollout_weights
    )


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_blocking_weight_sync_waits_for_both_sides(runner_cls) -> None:
    actor_handle = _Handle()
    rollout_handle = _Handle()
    actor = _Actor([actor_handle])
    rollout = _Rollout([rollout_handle], [])
    runner = _make_runner(runner_cls, actor, rollout)

    runner.update_rollout_weights()

    assert actor.sync_calls == 1
    assert rollout.blocking_calls == 1
    assert rollout.background_calls == 0
    assert actor_handle.wait_calls == 1
    assert rollout_handle.wait_calls == 1
    assert runner._pending_rollout_weight_sync is None


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_nonblocking_weight_sync_does_not_wait(runner_cls) -> None:
    actor_handle = _Handle()
    rollout_handle = _Handle()
    actor = _Actor([actor_handle])
    rollout = _Rollout([], [rollout_handle])
    runner = _make_runner(runner_cls, actor, rollout, no_wait=True)

    runner.update_rollout_weights(no_wait=True)

    assert rollout.background_calls == 1
    assert actor.sync_calls == 1
    assert actor_handle.wait_calls == 0
    assert rollout_handle.wait_calls == 0
    assert runner._pending_rollout_weight_sync == (rollout_handle, actor_handle)


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_nonblocking_weight_sync_coalesces_until_previous_sync_finishes(
    runner_cls,
) -> None:
    first_actor_handle = _Handle()
    first_rollout_handle = _Handle(done=True)
    second_actor_handle = _Handle()
    second_rollout_handle = _Handle(done=True)
    actor = _Actor([first_actor_handle, second_actor_handle])
    rollout = _Rollout([], [first_rollout_handle, second_rollout_handle])
    runner = _make_runner(runner_cls, actor, rollout, no_wait=True)

    runner.update_rollout_weights(no_wait=True)
    runner.update_rollout_weights(no_wait=True)

    # The second request is dropped, not queued: still one sync in flight.
    assert actor.sync_calls == 1
    assert rollout.background_calls == 1
    assert first_actor_handle.wait_calls == 0
    assert first_rollout_handle.wait_calls == 0
    assert runner._weight_sync_coalesced_total == 1
    assert runner._weight_sync_request_total == 2

    first_actor_handle.is_done = True
    runner.update_rollout_weights(no_wait=True)

    assert first_actor_handle.wait_calls == 1
    assert first_rollout_handle.wait_calls == 1
    assert actor.sync_calls == 2
    assert rollout.background_calls == 2
    assert runner._weight_sync_coalesced_total == 1
    assert runner._weight_sync_request_total == 3
    assert runner._pending_rollout_weight_sync == (
        second_rollout_handle,
        second_actor_handle,
    )


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_blocking_weight_sync_drains_an_inflight_background_sync(runner_cls) -> None:
    pending_actor_handle = _Handle()
    pending_rollout_handle = _Handle()
    blocking_actor_handle = _Handle()
    blocking_rollout_handle = _Handle()
    actor = _Actor([blocking_actor_handle])
    rollout = _Rollout([blocking_rollout_handle], [])
    runner = _make_runner(runner_cls, actor, rollout)
    runner._pending_rollout_weight_sync = (
        pending_rollout_handle,
        pending_actor_handle,
    )

    runner.update_rollout_weights(no_wait=False)

    assert pending_actor_handle.wait_calls == 1
    assert pending_rollout_handle.wait_calls == 1
    assert blocking_actor_handle.wait_calls == 1
    assert blocking_rollout_handle.wait_calls == 1
    assert runner._pending_rollout_weight_sync is None


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_teardown_drains_an_inflight_background_sync(runner_cls) -> None:
    actor_handle = _Handle()
    rollout_handle = _Handle()
    runner = _make_runner(runner_cls, _Actor([]), _Rollout([], []))
    runner._pending_rollout_weight_sync = (rollout_handle, actor_handle)

    runner.drain_pending_rollout_weight_sync()

    assert actor_handle.wait_calls == 1
    assert rollout_handle.wait_calls == 1
    assert runner._pending_rollout_weight_sync is None


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_teardown_is_a_noop_without_a_pending_sync(runner_cls) -> None:
    runner = _make_runner(runner_cls, _Actor([]), _Rollout([], []))

    runner.drain_pending_rollout_weight_sync()

    assert runner._pending_rollout_weight_sync is None
