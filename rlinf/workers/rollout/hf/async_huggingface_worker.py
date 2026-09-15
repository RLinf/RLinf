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

import asyncio

from omegaconf.omegaconf import DictConfig

from rlinf.scheduler import Channel, Worker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._generate_task: asyncio.Task = None
        self._evaluate_task: asyncio.Task = None
        self.staleness_threshold = cfg.algorithm.get("staleness_threshold", None)
        assert not self.enable_offload, (
            "Offload not supported in AsyncMultiStepRolloutWorker"
        )

        self._background_weight_sync_active = self.cfg.actor.get(
            "sync_weight_no_wait", False
        )
        self._weight_sync_requested = False
        self._weight_sync_work = None
        self._weight_sync_apply_total = 0
        self._weight_sync_coalesced_total = 0
        self._weight_sync_request_total = 0

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
        metric_channel: Channel,
    ):
        assert self._generate_task is None, (
            "generate task is not None but generate function is called."
        )
        self._generate_task = asyncio.create_task(
            self._generate(input_channel, output_channel, actor_channel, metric_channel)
        )
        try:
            await self._generate_task
        except asyncio.CancelledError:
            pass

    async def _generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
        metric_channel: Channel,
    ):
        while True:
            if self._background_weight_sync_active:
                await self._poll_background_weight_sync()
            await self.wait_if_stale()

            for _ in range(self.rollout_epoch):
                await self.generate_one_epoch(
                    input_channel,
                    output_channel,
                    actor_channel,
                )
            if self.finished_episodes is not None:
                self.finished_episodes += self.total_num_train_envs * self.rollout_epoch
            rollout_metrics = {
                f"time/rollout/{key}": value
                for key, value in self.pop_execution_times().items()
            }
            metric_channel.put(
                {"rank": self._rank, "time": rollout_metrics}, async_op=True
            )

    async def wait_if_stale(self) -> None:
        if self.staleness_threshold is None:
            return
        assert self.finished_episodes is not None, (
            "finished_episodes should be initialized."
        )
        while True:
            capacity = (
                (self.staleness_threshold + self.version + 1)
                * self.total_num_train_envs
                * self.rollout_epoch
            )
            if (
                self.finished_episodes + self.total_num_train_envs * self.rollout_epoch
                <= capacity
            ):
                break
            await asyncio.sleep(0.01)

    async def _run_evaluate_service(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ) -> None:
        """Serve decoupled evaluation requests until the worker is stopped."""
        await super().evaluate(input_channel, output_channel)

    async def ensure_evaluate_service(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ) -> None:
        """Start one persistent decoupled evaluation consumer if needed.

        The base decoupled ``evaluate`` loop is intentionally unbounded. Running
        it directly for every validation leaks one consumer per call. Keeping
        the task on the rollout worker makes repeated validation idempotent and
        also lets callers detect if the service exited unexpectedly.
        """
        if not self.env_decoupled_mode:
            raise RuntimeError(
                "The persistent evaluation service is only used in decoupled mode."
            )
        if self._evaluate_task is not None and not self._evaluate_task.done():
            return
        if self._evaluate_task is not None:
            if self._evaluate_task.cancelled():
                self._evaluate_task = None
            else:
                error = self._evaluate_task.exception()
                self._evaluate_task = None
                if error is not None:
                    raise RuntimeError(
                        "The decoupled evaluation service exited unexpectedly."
                    ) from error
                raise RuntimeError(
                    "The decoupled evaluation service returned unexpectedly."
                )

        self._evaluate_task = asyncio.create_task(
            self._run_evaluate_service(input_channel, output_channel)
        )
        # Give the service a chance to enter its first channel receive and
        # surface immediate initialization failures to the runner.
        await asyncio.sleep(0)
        if self._evaluate_task.done():
            error = self._evaluate_task.exception()
            self._evaluate_task = None
            if error is not None:
                raise RuntimeError(
                    "Failed to start the decoupled evaluation service."
                ) from error
            raise RuntimeError(
                "The decoupled evaluation service returned during startup."
            )

    def stop(self):
        if self._generate_task is not None and not self._generate_task.done():
            self._generate_task.cancel()
        if self._evaluate_task is not None and not self._evaluate_task.done():
            self._evaluate_task.cancel()

    async def _recv_and_apply_actor_sync(self) -> int:
        await super().sync_model_from_actor()
        return self.version

    def _start_background_weight_sync_if_needed(self):
        if (
            not self._background_weight_sync_active
            or not self._weight_sync_requested
            or self._weight_sync_work is not None
        ):
            return

        self._weight_sync_requested = False
        self._weight_sync_work = asyncio.create_task(self._recv_and_apply_actor_sync())

    async def _finish_background_weight_sync(self, work: asyncio.Task) -> int:
        """Await one apply task and retire it exactly once.

        Both the poll loop and an outstanding request can await the same task,
        so only the caller that still sees it as the active one retires it.
        """
        applied_version = await work
        if self._weight_sync_work is work:
            self._weight_sync_work = None
            self._weight_sync_apply_total += 1
            self._start_background_weight_sync_if_needed()
        return applied_version

    @Worker.timer("rollout/poll_weight_sync")
    async def _poll_background_weight_sync(self):
        self._start_background_weight_sync_if_needed()
        work = self._weight_sync_work
        if work is None:
            return

        if not work.done():
            return

        await self._finish_background_weight_sync(work)

    @Worker.timer("rollout/request_weight_sync")
    async def request_actor_sync_model(self) -> int:
        """Request a background sync and return once its weights are applied.

        The RPC is still asynchronous to the runner: calling it returns a
        ``WorkerGroupFuncResult`` immediately. Keeping this coroutine alive
        until rollout-side application finishes is what makes that result a
        truthful completion signal, which the runner relies on to coalesce
        requests, to switch back to a blocking sync, and to drain at teardown.

        Returns:
            The model version applied by the time this request finishes.
        """
        assert self._background_weight_sync_active, (
            "Background weight sync requires actor.sync_weight_no_wait=true."
        )
        self._weight_sync_request_total += 1
        if self._weight_sync_requested or self._weight_sync_work is not None:
            self._weight_sync_coalesced_total += 1

        # Wait out the applies already owed: one for a task in flight, one for a
        # request that has not started yet, plus this request when it is new.
        pending_apply_count = int(self._weight_sync_work is not None) + int(
            self._weight_sync_requested
        )
        if not self._weight_sync_requested:
            self._weight_sync_requested = True
            pending_apply_count += 1
        target_apply_total = self._weight_sync_apply_total + pending_apply_count

        self._start_background_weight_sync_if_needed()
        while self._weight_sync_apply_total < target_apply_total:
            work = self._weight_sync_work
            assert work is not None, (
                "A requested background weight sync has no active apply task."
            )
            await self._finish_background_weight_sync(work)

        return self.version
