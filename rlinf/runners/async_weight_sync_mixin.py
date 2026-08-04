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

"""Non-blocking actor-to-rollout weight synchronization for async runners."""

from rlinf.scheduler import WorkerGroupFuncResult as Handle


class AsyncWeightSyncMixin:
    """Overlap actor-to-rollout weight sync with the next step's work.

    The blocking path waits for the sync to land before the runner moves on,
    which idles the actor for as long as the transfer takes. Async runners can
    instead issue the sync and carry on, since a rollout worker that is one
    update behind is already expected under off-policy training.

    Mix into a runner that has ``actor`` / ``rollout`` worker groups and a
    ``logger``, ahead of the runner base class in the MRO::

        class AsyncEmbodiedRunner(AsyncWeightSyncMixin, EmbodiedRunner): ...

    Call :meth:`init_weight_sync_state` from ``__init__``, pass
    ``no_wait=self.sync_weight_no_wait`` to :meth:`update_rollout_weights`, and
    call :meth:`drain_pending_rollout_weight_sync` before tearing workers down.

    Only one sync is ever in flight: a request that arrives while the previous
    one is still running is coalesced away rather than queued, so a slow
    transfer cannot build up a backlog. Weights are dropped, never reordered —
    the rollout worker either gets an update or keeps the ones it has.

    ``sync_weight_no_wait`` defaults to ``False``, which routes every call
    through the base class's blocking implementation and leaves behavior
    unchanged.
    """

    def init_weight_sync_state(self) -> None:
        """Set up sync bookkeeping. Call from the runner's ``__init__``."""
        # Requires the rollout worker's background sync path, which is gated on
        # the same config key.
        # validate_embodied_cfg enforces that this requires weight_syncer.type=patch.
        self.sync_weight_no_wait = self.cfg.actor.get("sync_weight_no_wait", False)
        self._pending_rollout_weight_sync: tuple[Handle, Handle] | None = None
        self._weight_sync_request_total = 0
        self._weight_sync_coalesced_total = 0

    def _cleanup_pending_rollout_weight_sync(self, no_wait: bool) -> bool:
        """Retire the in-flight sync if it has landed.

        Args:
            no_wait: Return immediately when the sync is still running, instead
                of blocking until it lands.

        Returns:
            Whether no sync remains in flight.
        """
        if self._pending_rollout_weight_sync is None:
            return True

        rollout_apply_handle, actor_handle = self._pending_rollout_weight_sync
        if no_wait and (not rollout_apply_handle.done() or not actor_handle.done()):
            return False

        # request_actor_sync_model keeps its RPC alive until the rollout worker
        # has applied the weights, not merely acknowledged the request.
        rollout_apply_handle.wait()
        actor_handle.wait()
        self._pending_rollout_weight_sync = None
        return True

    def update_rollout_weights(self, no_wait: bool = False) -> None:
        """Synchronize actor weights to the rollout workers.

        Args:
            no_wait: Issue the sync in the background instead of blocking. When
                a previous sync is still in flight, this one is coalesced away.
        """
        if not no_wait:
            self._cleanup_pending_rollout_weight_sync(no_wait=False)
            return super().update_rollout_weights()

        self._weight_sync_request_total += 1
        if not self._cleanup_pending_rollout_weight_sync(no_wait=True):
            self._weight_sync_coalesced_total += 1
            self.logger.info(
                f"Weight sync still in flight; coalesced this request "
                f"({self._weight_sync_coalesced_total} coalesced / "
                f"{self._weight_sync_request_total} requested)."
            )
            return

        rollout_apply_handle: Handle = self.rollout.request_actor_sync_model()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        self._pending_rollout_weight_sync = (rollout_apply_handle, actor_handle)

    def drain_pending_rollout_weight_sync(self) -> None:
        """Wait out any in-flight sync. Call before stopping the workers."""
        self._cleanup_pending_rollout_weight_sync(no_wait=False)
