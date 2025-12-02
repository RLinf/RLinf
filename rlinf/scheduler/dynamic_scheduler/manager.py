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
import math
import time
from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING, Callable

from omegaconf import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler.dynamic_scheduler.utils import (
    RolloutAction,
    RolloutReport,
    RolloutScheduleInfo,
    get_global_scheduer_state,
    get_scheduler_channel,
    get_scheduler_request_queue,
    get_scheduler_response_queue,
)
from rlinf.utils.placement import ModelParallelComponentPlacement

if TYPE_CHECKING:
    from rlinf.data.io_struct import SeqGroupInfo
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker


class ComponentManager(ABC):
    """ComponentManager is an abstract base class for all component managers.

    Specific component managers should inherit this class and implement the following methods:
    - pre_process_impl : Pre-process implementation
    - main_loop_finalize : Process after the last training iteration in main_loop
    - release_resource (Optional) : Release the resource of the component
    - allocate_resource (Optional) : Allocate the resource of the component
    """

    def __init__(
        self,
        component_role: str,
        config: DictConfig,
        component_placement: ModelParallelComponentPlacement,
        use_pre_process_policy: bool,
        use_wait_before_last_iter_policy: bool,
        channel_factory: Callable[[str], Channel],
        _logger: Logger,
    ):
        """Initialize the ComponentManager.

        Args:
            component_role (str): The role of the component.
            config (DictConfig): The configuration of this training task.
            component_placement (ModelParallelComponentPlacement): The component placement.
            use_pre_process_policy (bool): Whether to use the pre-process policy.
            use_wait_before_last_iter_policy (bool): Whether to use the wait before last iter policy.
            channel_factory (Callable[[str], Channel]): The factory for creating channels.
            _logger (Logger): The logger for this training task.
        """
        self.component_role = component_role
        self.cfg = config
        self.component_placement = component_placement
        self.use_pre_process_policy = use_pre_process_policy
        self.use_wait_before_last_iter_policy = use_wait_before_last_iter_policy
        self.channel_factory = channel_factory
        self._logger = _logger
        self.n_minibatches = self.cfg.algorithm.n_minibatches

        assert self.component_role in self.component_placement._components

        self.init_instance_num = getattr(
            component_placement, f"{self.component_role}_dp_size"
        )
        self.init_gpu_num = getattr(
            component_placement, f"{self.component_role}_world_size"
        )
        # Note. mode_parallel_size here represents the number of GPUs, the quantity required for a single instance
        self.model_parallel_size = self.init_gpu_num // self.init_instance_num

        self.reset()

    def create_channels(self, channel_num: int):
        """Create channels and queues for communication, and each channel is for a single instance.

        Args:
            channel_num (int): The number of channel.
        """
        self.channels: list[Channel] = []
        self.request_queue = get_scheduler_request_queue()
        self.response_queue = get_scheduler_response_queue()
        for instance_id in range(channel_num):
            channel = self.channel_factory(
                get_scheduler_channel(self.component_role, instance_id)
            )
            self.channels.append(channel)

    def reset(self):
        """Reset state of ComponentManager."""
        self.current_instance_num = self.init_instance_num
        self.current_gpu_num = self.init_gpu_num
        self.current_instance_offset = 0

    def update(self, released_instance_num: int = 0, incremental_instance_num: int = 0):
        """Update state of ComponentManager.

        Args:
            released_instance_num (int): The number of instances to release.
            incremental_instance_num (int): The number of instances to increment.
        """
        assert released_instance_num == 0 or incremental_instance_num == 0
        if released_instance_num == 0 and incremental_instance_num == 0:
            return

        if released_instance_num != 0:
            assert self.current_instance_num >= released_instance_num
            self.current_gpu_num -= released_instance_num * self.model_parallel_size
            self.current_instance_num -= released_instance_num
            self.current_instance_offset += released_instance_num
        else:
            assert incremental_instance_num > 0
            self.current_instance_num += incremental_instance_num
            self.current_gpu_num = self.current_instance_num * self.model_parallel_size
            assert self.current_gpu_num <= self.component_placement._cluster_num_gpus
            self.current_instance_offset -= incremental_instance_num

    async def pre_process(self, *args, **kwargs):
        """Pre-process. Reset state of ComponentManager and call pre_process_impl."""
        self.reset()
        await self.pre_process_impl(*args, **kwargs)

    async def release_or_allocate(self, train_iter: int) -> tuple[int, int]:
        """Execute release_resource or allocate_resource for this component.

        Args:
            train_iter (int): The current train-iter completed by the actor.

        Returns:
            released_gpu_num (int): The number of released GPU resources.
            incremental_gpu_num (int): The number of incremental GPU resources.
        """
        if train_iter == self.n_minibatches - 1:
            await self.main_loop_finalize()
            return 0, 0

        released_gpu_num = await self.release_resource(train_iter)
        incremental_gpu_num = await self.allocate_resource(train_iter)
        return (released_gpu_num, incremental_gpu_num)

    # ------------------------------------------------- Abstract methods to be implemented by subclass -------------------------------------------------
    @abstractmethod
    async def pre_process_impl(self, *args, **kwargs):
        """Implement of pre_process."""
        ...

    @abstractmethod
    async def main_loop_finalize(self):
        """Processing after the last training iteration in main_loop."""
        ...

    @abstractmethod
    async def release_resource(self, *args, **kwargs) -> int:
        """Release the GPU resources.

        Returns:
            int: The number of released GPU resources.
        """
        ...

    @abstractmethod
    async def allocate_resource(self, *args, **kwargs) -> int:
        """Allocate the GPU resources.

        Returns:
            int: The number of incremental GPU resources.
        """
        ...


class RolloutManager(ComponentManager):
    """Manage resource allocation for rollout.

    There are three core actions for rollout instances:

    - report  : collect the report from all alive rollout instances
    - finish  : send Finish or Wait_For_Finish signal to all alive rollout instances
    - migrate : migrate the rollout instances
        - migrate_policy : return the max number of rollout instances could migrate out
        - find_release_instance_num_needed : find the number of rollout instances needed to release
        - TODO(balance_batches) : balance the batches between the rollout instances
    """

    def __init__(
        self,
        config: DictConfig,
        component_placement: ModelParallelComponentPlacement,
        use_pre_process_policy: bool,
        use_wait_before_last_iter_policy: bool,
        channel_factory: Callable[[str], Channel],
        _logger: Logger,
    ):
        """Initialize the RolloutManager."""
        super().__init__(
            component_role="rollout",
            config=config,
            component_placement=component_placement,
            use_pre_process_policy=use_pre_process_policy,
            use_wait_before_last_iter_policy=use_wait_before_last_iter_policy,
            channel_factory=channel_factory,
            _logger=_logger,
        )
        self.create_channels(self.init_instance_num)

        self.max_running_requests = self.cfg.rollout.max_running_requests
        self.rollout_total_tasks = (
            self.cfg.algorithm.group_size * self.cfg.data.rollout_batch_size
        )

    # ------------------------------------------------- override start -------------------------------------------------

    async def pre_process_impl(self, running_tasks_threshold: int = -1):
        """Pre-process implementation of rollout.

        Args:
            running_tasks_threshold (int): The threshold of running tasks. If -1, use half of rollout_total_tasks.

        At the beginning of each global step, rollout occupies the resources of the actor until running_tasks is less than running_tasks_threshold.
        Then, rollout releases the actor's resources.
        """
        self.running_tasks = self.rollout_total_tasks
        if not self.use_pre_process_policy:
            return

        migrate_out_gpu_num = self.component_placement.actor_world_size
        migrate_out_instance_num = migrate_out_gpu_num // self.model_parallel_size
        assert migrate_out_gpu_num % self.model_parallel_size == 0
        assert migrate_out_instance_num > 0

        if running_tasks_threshold == -1:
            running_tasks_threshold = self.rollout_total_tasks // 2
        assert (
            running_tasks_threshold > 0
            and running_tasks_threshold < self.rollout_total_tasks
        )

        while True:
            report_str = await self.report()
            if (
                self.total_tasks == self.rollout_total_tasks
                and self.running_tasks <= running_tasks_threshold
            ):
                self._logger.info("\npre_process condition satisfied:\n" + report_str)
                await self.migrate(migrate_out_instance_num)
                break
            await asyncio.sleep(1)

    async def main_loop_finalize(self):
        """Processing after the last training iteration in main_loop. Perform RolloutAction.Finish on all surviving instances."""
        if self.current_instance_num == 0:
            return

        await self.finish(action=RolloutAction.Finish)

    async def release_resource(
        self,
        train_iter: int,
    ) -> int:
        """Release the GPU resources.

        Args:
            train_iter (int): The current train-iter completed by the actor.

        Returns:
            int: The number of released GPU resources.
        """
        if self.current_instance_num == 0:
            return 0

        # Report Action
        report_str = await self.report()
        self._logger.info(report_str)

        # Finish Action
        if self.running_tasks == 0:
            return await self.finish(action=RolloutAction.Finish)

        # Wait_For_Finish Action
        if (
            self.use_wait_before_last_iter_policy
            and train_iter == self.n_minibatches - 2
        ):
            return await self.finish(action=RolloutAction.Wait_For_Finish)

        # Migrate Action
        released_instance_num = self.migrate_policy(train_iter)
        released_gpu_num = await self.migrate(released_instance_num)
        return released_gpu_num

    async def allocate_resource(self, *args, **kwargs) -> int:
        """Allocate the GPU resources.

        Returns:
            int: The number of incremental GPU resources.
        """
        return 0

    # ------------------------------------------------- override end -------------------------------------------------

    async def _scatter_requests(
        self,
        requests: RolloutScheduleInfo | list[RolloutScheduleInfo],
        instance_ids: list[int],
    ):
        """Scatter the requests to the rollout instances.

        Args:
            requests (RolloutScheduleInfo | List[RolloutScheduleInfo]): The requests to scatter.
            instance_ids (List[int]): The list of instance ids.
        """
        if isinstance(requests, RolloutScheduleInfo):
            requests = [requests] * len(instance_ids)
        assert len(requests) == len(instance_ids), (
            f"Try to send {len(requests)} requests to {len(instance_ids)} rollout instances."
        )
        tasks = [
            asyncio.create_task(
                self.channels[rollout_instance_id]
                .put(
                    request,
                    key=self.request_queue,
                    async_op=True,
                )
                .async_wait()
            )
            for request, rollout_instance_id in zip(requests, instance_ids)
        ]
        await asyncio.gather(*tasks)

    async def _gather_responses(
        self,
        instance_ids: list[int],
    ) -> list[RolloutScheduleInfo]:
        """Gather the responses from the rollout instances.

        Args:
            instance_ids (List[int]): The list of instance ids.
        """
        tasks = [
            asyncio.create_task(
                self.channels[rollout_instance_id]
                .get(
                    key=self.response_queue,
                    async_op=True,
                )
                .async_wait()
            )
            for rollout_instance_id in instance_ids
        ]
        responses: list[RolloutScheduleInfo] = await asyncio.gather(*tasks)
        assert all(
            response.instance_id == instance_id
            for response, instance_id in zip(responses, instance_ids)
        ), (
            f"Expect to get responses from instance_ids={instance_ids}, "
            f"but got from {[response.instance_id for response in responses]}"
        )
        return responses

    def _get_running_instances(self) -> list[int]:
        return list(range(self.current_instance_offset, self.init_instance_num))

    async def report(self):
        """Check the report of rollout instances."""
        alive_instance_ids = self._get_running_instances()
        await self._scatter_requests(
            RolloutScheduleInfo(action=RolloutAction.Report),
            alive_instance_ids,
        )
        responses = await self._gather_responses(alive_instance_ids)
        self.reports = {response.instance_id: response.report for response in responses}

        self.total_tasks = sum(report.total_tasks for report in self.reports.values())
        self.running_tasks = sum(
            report.running_tasks for report in self.reports.values()
        )

        report_str = f"Rollout Report:\ncurrent_total_tasks={self.total_tasks}, current_running_tasks={self.running_tasks}\n"
        for instance_id, report in self.reports.items():
            report_str += f"rollout{instance_id} : total_tasks={report.total_tasks}, running_tasks={report.running_tasks}, completed_tasks={report.completed_tasks}\n"
        return report_str

    async def finish(
        self, action: RolloutAction, finished_instance_ids: list[int] | None = None
    ) -> int:
        """Finish the rollout instances.

        Args:
            action (RolloutAction): The action to finish.
            finished_instance_ids (List[int]): The list of finished instance ids. If None, finish all alive rollout instances.

        Returns:
            int: The number of released GPU resources.
        """
        if finished_instance_ids is None:
            finished_instance_ids = self._get_running_instances()
        assert action in [RolloutAction.Finish, RolloutAction.Wait_For_Finish]

        await self._scatter_requests(
            RolloutScheduleInfo(action=action), finished_instance_ids
        )

        responses = await self._gather_responses(finished_instance_ids)
        assert all(response.action == RolloutAction.Offloaded for response in responses)

        self.update(released_instance_num=len(finished_instance_ids))
        return len(finished_instance_ids) * self.model_parallel_size

    def _assign_sequences_sequential(
        self,
        migrate_in_instance_ids: list[int],
        migrate_in_instance_reports: list[RolloutReport],
        migrate_out_batches: list["SeqGroupInfo"],
    ) -> list[list["SeqGroupInfo"]]:
        assert len(migrate_in_instance_ids) == len(migrate_in_instance_reports), (
            f"Get {len(migrate_in_instance_ids)} instance ids != {len(migrate_in_instance_reports)} reports."
        )
        instance_running_tasks_expected = max(
            0, self.running_tasks // len(migrate_in_instance_ids)
        )
        migrate_out_batches_index = 0
        migrate_out_batches_len = len(migrate_out_batches)

        batches_assigned: list[list["SeqGroupInfo"]] = []
        for in_id, report in zip(
            migrate_in_instance_ids[:-1], migrate_in_instance_reports[:-1]
        ):
            running_tasks = report.running_tasks
            if running_tasks >= instance_running_tasks_expected:
                self._logger.info(
                    f"Warning : rollout-{in_id} has {running_tasks} running tasks "
                    f"> expected {instance_running_tasks_expected}"
                )
                batches_assigned.append([])
                continue

            migrate_in_batches = []
            while (migrate_out_batches_index < migrate_out_batches_len) and (
                running_tasks < instance_running_tasks_expected
            ):
                migrate_batch = migrate_out_batches[migrate_out_batches_index]
                migrate_in_batches.append(migrate_batch)
                migrate_out_batches_index += 1
                running_tasks += migrate_batch.num_aborted

            batches_assigned.append(migrate_in_batches)

        if migrate_out_batches_index < migrate_out_batches_len:
            migrate_in_batches = migrate_out_batches[migrate_out_batches_index:]
            running_tasks = sum(
                migrate_batch.num_aborted
                for migrate_batch in migrate_out_batches[migrate_out_batches_index:]
            )

            batches_assigned.append(migrate_in_batches)

        return batches_assigned

    def assign_sequences(
        self,
        migrate_in_instance_ids: list[int],
        migrate_in_instance_reports: list[RolloutReport],
        migrate_out_batches: list["SeqGroupInfo"],
        algo: str = "sequential",
    ) -> list[list["SeqGroupInfo"]]:
        """Assign sequences to instances based on the specified algorithm.

        This method handles the assignment of sequences to instances during
        migration. It supports different algorithms for the assignment process.

        Args:
            migrate_in_instance_ids (List[int]): A list of instance IDs that are
                migrating in.
            migrate_in_instance_reports (List[RolloutReport]): A list of rollout
                reports corresponding to the instances that are migrating in.
            migrate_out_batches (List["SeqGroupInfo"]): A list of sequence group
                information for the batches that are migrating out.
            algo (str, optional): The algorithm to use for assigning sequences.
                Defaults to "sequential". Currently, only "sequential" is supported.

        Returns:
            The result of the sequence assignment process, as determined by the
            selected algorithm. It has the same length as migrate_in_instance_ids.
            If an instance is not assigned any batches, the corresponding entry is an
            empty list.
        """
        if algo == "sequential":
            return self._assign_sequences_sequential(
                migrate_in_instance_ids,
                migrate_in_instance_reports,
                migrate_out_batches,
            )
        else:
            raise ValueError(f"Unexpected migration assigning algorithm: {algo}")

    async def migrate_out(self, migrate_out_instance_ids: list[int]):
        """Execute the Migrate_Out action.

        Args:
            migrate_out_instance_ids (List[int]): The list of instance ids to migrate out.

        Returns:
            List["SeqGroupInfo"]: The list of migrate out batches.
        """
        await self._scatter_requests(
            RolloutScheduleInfo(action=RolloutAction.Migrate_Out),
            migrate_out_instance_ids,
        )
        migrate_out_batches: list["SeqGroupInfo"] = []
        responses = await self._gather_responses(migrate_out_instance_ids)
        for response in responses:
            assert response.data is not None
            migrate_out_batches.extend(response.data)
        return migrate_out_batches

    async def migrate_in(
        self,
        migrate_in_instance_ids: list[int],
        migrate_out_batches: list["SeqGroupInfo"],
    ):
        """Execute the Migrate_In action.

        Args:
            migrate_in_instance_ids (List[int]): The list of instance ids to migrate in.
            migrate_out_batches (List["SeqGroupInfo"]): The list of migrate out batches.
        """
        instance_running_tasks_expected = max(
            0, self.running_tasks // len(migrate_in_instance_ids)
        )
        self._logger.info(
            f"[Migrate-Info] "
            f"migrate_out_batches_len={len(migrate_out_batches)}, "
            f"migrate_out_tasks={sum(batch.num_aborted for batch in migrate_out_batches)}, "
            f"{self.running_tasks=}, "
            f"{instance_running_tasks_expected=}"
        )

        migrate_in_instance_reports = [
            self.reports[instance_id] for instance_id in migrate_in_instance_ids
        ]

        assigned_batches = self.assign_sequences(
            migrate_in_instance_ids,
            migrate_in_instance_reports,
            migrate_out_batches,
            algo="sequential",
        )

        migrate_in_ids: list[int] = []
        migrate_in_requests: list[RolloutScheduleInfo] = []

        for instance_id, batches in zip(migrate_in_instance_ids, assigned_batches):
            if len(batches) > 0:
                migrate_in_request = RolloutScheduleInfo(
                    action=RolloutAction.Migrate_In, data=batches
                )
                migrate_in_requests.append(migrate_in_request)
                migrate_in_ids.append(instance_id)

        migrate_out_msg = "[Migrate-Info]:\n"
        for request, instance_id in zip(migrate_in_requests, migrate_in_ids):
            migrate_in_batches: list["SeqGroupInfo"] = request.data
            running_tasks = self.reports[instance_id].running_tasks + sum(
                batch.num_aborted for batch in migrate_in_batches
            )
            migrate_out_msg += (
                f"rollout-{instance_id} : "
                f"migrate_in_batches: {len(request.data)}, "
                f"running_tasks={self.reports[instance_id].running_tasks} "
                f"-> {running_tasks} ~= {instance_running_tasks_expected}\n"
            )
        self._logger.info(migrate_out_msg)

        await self._scatter_requests(migrate_in_requests, migrate_in_ids)

    async def migrate(self, migrate_instance_num: int) -> int:
        """Execute the migration of rollout instances.

        Args:
            migrate_instance_num (int): The number of rollout instances to migrate out.

        Returns:
            int: The number of released GPU resources.
        """
        if migrate_instance_num == 0:
            return 0
        assert migrate_instance_num < self.current_instance_num
        assert len(self.reports) == self.current_instance_num

        running_instance_ids = self._get_running_instances()
        migrate_out_instance_ids = running_instance_ids[:migrate_instance_num]
        migrate_in_instance_ids = running_instance_ids[migrate_instance_num:]

        # Migrate Out
        migrate_out_batches = await self.migrate_out(migrate_out_instance_ids)

        # Migrate In
        await self.migrate_in(migrate_in_instance_ids, migrate_out_batches)

        # Send Finish signal to migrate out instances to finish
        await self.finish(RolloutAction.Finish, migrate_out_instance_ids)

        return migrate_instance_num * self.model_parallel_size

    def migrate_policy(self, train_iter: int) -> int:
        """Return the max number of rollout instances could migrate out.

        Args:
            train_iter (int): current train iter.

        Returns:
            int: the max number of rollout instances could migrate out
        """
        if self.current_instance_num <= 1:
            return 0

        min_instance_num_needed = math.ceil(
            self.running_tasks / self.max_running_requests
        )

        released_instance_num_max = max(
            0, self.current_instance_num - min_instance_num_needed
        )
        released_instance_num_needed = self.find_release_instance_num_needed(
            released_instance_num_max
        )
        self._logger.info(
            f"[Release-Info] rollout migrate info: released_instance_num_max={released_instance_num_max}, released_instance_num_needed={released_instance_num_needed}"
        )
        return released_instance_num_needed

    def find_release_instance_num_needed(
        self,
        released_instance_num_max: int,
    ) -> int:
        """Find the number of rollout instances needed to release.

        Args:
            released_instance_num_max (int): The maximum number of rollout instances to release.

        Returns:
            int: The number of rollout instances needed to release.
        """
        if released_instance_num_max == 0:
            return 0

        scheduler_state = get_global_scheduer_state()
        actor_model_parallel_size = scheduler_state.get_component_model_parallel_size(
            "actor"
        )
        actor_current_instance_num = scheduler_state.get_component_instance_num("actor")
        actor_valid_dp_sizes = scheduler_state.actor_valid_dp_sizes

        assert actor_current_instance_num in actor_valid_dp_sizes
        index = actor_valid_dp_sizes.index(actor_current_instance_num)
        assert index < len(actor_valid_dp_sizes)

        released_gpu_num_max = released_instance_num_max * self.model_parallel_size

        actor_increment_gpu_num = 0
        for actor_dp_size in actor_valid_dp_sizes[index + 1 :]:
            actor_gpu_num_needed = (
                actor_dp_size - actor_current_instance_num
            ) * actor_model_parallel_size
            if actor_gpu_num_needed <= released_gpu_num_max:
                actor_increment_gpu_num = actor_gpu_num_needed
            else:
                break

        released_instance_num_needed = math.ceil(
            actor_increment_gpu_num / self.model_parallel_size
        )
        assert released_instance_num_needed <= released_instance_num_max
        return released_instance_num_needed


class InferenceManager(ComponentManager):
    """Manage resource allocation for inference."""

    def __init__(
        self,
        config: DictConfig,
        component_placement: ModelParallelComponentPlacement,
        use_pre_process_policy: bool,
        use_wait_before_last_iter_policy: bool,
        channel_factory: Callable[[str], Channel],
        _logger: Logger,
    ):
        """Initialize the InferenceManager."""
        super().__init__(
            component_role="inference",
            config=config,
            component_placement=component_placement,
            use_pre_process_policy=use_pre_process_policy,
            use_wait_before_last_iter_policy=use_wait_before_last_iter_policy,
            channel_factory=channel_factory,
            _logger=_logger,
        )
        self.create_channels(1)

    async def wait_for_finish(self) -> int:
        """Last train iter process.

        If use_wait_before_last_iter_policy is True, this function will block training until the inference is finished.
        """
        while not self.main_loop_finished_handler.done():
            await asyncio.sleep(0.1)

        released_instance_num = self.current_instance_num
        self.update(released_instance_num=released_instance_num)
        return released_instance_num * self.model_parallel_size

    async def pre_process_impl(self):
        """Pre-process implementation of inference.

        Initialize the main loop finished handler.
        """
        self.main_loop_finished_handler = self.channels[0].get(
            key=self.response_queue, async_op=True
        )

    async def main_loop_finalize(self):
        """Processing after the last training iteration in main_loop."""
        await self.main_loop_finished_handler.async_wait()
        assert self.main_loop_finished_handler.done()

    async def release_resource(
        self,
        train_iter: int,
    ) -> int:
        """Release the GPU resources.

        Args:
            train_iter (int): The current train-iter completed by the actor.

        Returns:
            int: The number of released GPU resources.
        """
        if self.current_instance_num == 0:
            return 0

        if not self.use_wait_before_last_iter_policy:
            released_instance_num = (
                self.current_instance_num
                if self.main_loop_finished_handler.done()
                else 0
            )
            self.update(released_instance_num=released_instance_num)
            return released_instance_num * self.model_parallel_size

        # Wait for finish
        scheduler_state = get_global_scheduer_state()
        rollout_current_instance_num = scheduler_state.get_component_instance_num(
            "rollout"
        )
        need_wait_for_finish = (train_iter == self.n_minibatches - 2) or (
            rollout_current_instance_num == 0
        )
        if need_wait_for_finish:
            return await self.wait_for_finish()

        return 0

    async def allocate_resource(self, *args, **kwargs) -> int:
        """Allocate the GPU resources.

        Returns:
            int: The number of incremental GPU resources.
        """
        return 0


class ActorManager(ComponentManager):
    """Manage resource allocation for actor."""

    def __init__(
        self,
        config: DictConfig,
        component_placement: ModelParallelComponentPlacement,
        use_pre_process_policy: bool,
        use_wait_before_last_iter_policy: bool,
        channel_factory: Callable[[str], Channel],
        _logger: Logger,
    ):
        """Initialize the ActorManager."""
        super().__init__(
            component_role="actor",
            config=config,
            component_placement=component_placement,
            use_pre_process_policy=use_pre_process_policy,
            use_wait_before_last_iter_policy=use_wait_before_last_iter_policy,
            channel_factory=channel_factory,
            _logger=_logger,
        )
        self.create_channels(1)

        assert hasattr(self, "current_instance_num")

    async def pre_process_impl(self):
        """Pre-process implementation of actor.

        If use_pre_process_policy is True, send a signal to actor to start training.
        """
        if not self.use_pre_process_policy:
            return
        await (
            self.channels[0]
            .put(None, key=self.request_queue, async_op=True)
            .async_wait()
        )

    def try_allocate(
        self, available_gpu_num: int, actor_valid_dp_sizes: list[int]
    ) -> int:
        """Try to allocate the GPU resources.

        Args:
            available_gpu_num (int): The number of available GPU resources.
            actor_valid_dp_sizes (List[int]): The valid data parallel sizes for the actor.

        Returns:
            incremental_gpu_num (int): The number of incremental GPU resources of actor.
        """
        if available_gpu_num < self.model_parallel_size:
            return 0

        incremental_gpu_num = 0
        assert (
            self.current_instance_num in actor_valid_dp_sizes
            and self.current_instance_num != actor_valid_dp_sizes[-1]
        )
        index = actor_valid_dp_sizes.index(self.current_instance_num)
        for next_dp_size in actor_valid_dp_sizes[index + 1 :]:
            needed_gpu_nums = (
                next_dp_size - self.current_instance_num
            ) * self.model_parallel_size
            if needed_gpu_nums <= available_gpu_num:
                incremental_gpu_num = needed_gpu_nums
            else:
                break

        assert incremental_gpu_num <= available_gpu_num
        return incremental_gpu_num

    async def scale(self, new_gpu_num: int):
        """Send scale info to actor."""
        scale_info = {"world_size": new_gpu_num}
        if new_gpu_num == self.current_gpu_num:
            scale_info = None

        await (
            self.channels[0]
            .put(
                scale_info,
                key=self.request_queue,
                async_op=True,
            )
            .async_wait()
        )

        if new_gpu_num > self.current_gpu_num:
            incremental_instance_num = (
                new_gpu_num // self.model_parallel_size - self.current_instance_num
            )
            self.update(incremental_instance_num=incremental_instance_num)
        elif new_gpu_num < self.current_gpu_num:
            released_instance_num = (
                self.current_instance_num - new_gpu_num // self.model_parallel_size
            )
            self.update(released_instance_num=released_instance_num)

    async def main_loop_finalize(self):
        """Processing after the last training iteration in main_loop. GPU resources of actor should be scale-down to init_gpu_num."""
        return await self.scale(self.init_gpu_num)

    async def allocate_resource(
        self,
        train_iter: int,
    ) -> int:
        """Allocate the GPU resources.

        Based on the value of available_gpu_num, try to allocate resources.
        If the allocation result shows that the new_gpu_num != self.current_gpu_num, then send {"world_size": new_gpu_num} to actor, else send None.

        Args:
            train_iter (int): The current train-iter completed by the actor.

        Returns:
            incremental_gpu_num (int): The number of incremental GPU resources of actor.
        """
        scheduler_state = get_global_scheduer_state()
        available_gpu_num = scheduler_state.available_gpu_num
        actor_valid_dp_sizes = scheduler_state.actor_valid_dp_sizes

        incremental_gpu_num = self.try_allocate(available_gpu_num, actor_valid_dp_sizes)
        assert incremental_gpu_num >= 0

        await self.scale(incremental_gpu_num + self.current_gpu_num)

        return incremental_gpu_num

    async def wait_for_actor_update(self):
        """Wait for the actor update."""
        await self.channels[0].get(key=self.response_queue, async_op=True).async_wait()

    async def release_resource(self, *args, **kwargs) -> int:
        """Release the GPU resources.

        Returns:
            int: The number of released GPU resources.
        """
        return 0


def create_component_manager(
    component_role: str, component_manager_kwargs
) -> ComponentManager:
    """Create component manager."""
    if component_role == "rollout":
        return RolloutManager(**component_manager_kwargs)
    elif component_role == "actor":
        return ActorManager(**component_manager_kwargs)
    elif component_role == "inference":
        return InferenceManager(**component_manager_kwargs)
    raise ValueError(f"can't find ComponentManager subclass for {component_role}")


class RolloutScalingScheduler:
    """Manage communication and lifecycle transitions for a rollout instance that participates in a centralized scheduling system.

    This class encapsulates the asynchronous logic required for a rollout instance
    to report progress, accept new workload (migrate in), release workload
    (migrate out), wait for completion, and notify the scheduler when it is
    offloaded. It interfaces with:
    - a scheduler_channel for sending/receiving RolloutScheduleInfo messages,
    - a per-instance request/response queue, and
    - the worker and its status_manager which track and manage locally running
        sequence-group generation tasks.
    """

    def __init__(
        self,
        rank: int,
        scheduler_channel: Channel,
        worker: "SGLangWorker",
    ):
        """Initialize the dynamic scheduler manager.

        Args:
            rank (int): The rank of the rollout instance.
            scheduler_channel (Channel): The channel for communication with the scheduler.
            worker (SGLangWorker): The rollout worker instance.
        """
        self._rank = rank
        self.scheduler_channel = scheduler_channel
        self.scheduler_request_queue = get_scheduler_request_queue()
        self.scheduler_response_queue = get_scheduler_response_queue()
        self.worker = worker
        self.status_manager = self.worker.status_manager

    async def _report(self):
        report = RolloutReport(
            total_requests=self.status_manager.num_seq_group,
            completed_requests=self.status_manager.num_seq_group_done,
            total_tasks=self.status_manager.num_seq,
            completed_tasks=self.status_manager.num_seq_returned,
            running_tasks=self.status_manager.num_seq_running,
            timestamp=time.time(),
        )
        scheduler_response = RolloutScheduleInfo(instance_id=self._rank, report=report)
        await self.scheduler_channel.put(
            scheduler_response,
            key=self.scheduler_response_queue,
            async_op=True,
        ).async_wait()

    async def _migrate_out(self):
        await self.worker.abort_generation()
        await self._wait_until_no_running_task()

        assert self.status_manager.num_seq_group_running == 0
        assert self.status_manager.num_seq_running == 0
        scheduler_response = RolloutScheduleInfo(
            instance_id=self._rank, data=self.status_manager.get_aborted_seq_groups()
        )
        await self.scheduler_channel.put(
            scheduler_response,
            key=self.scheduler_response_queue,
            async_op=True,
        ).async_wait()

    async def _migrate_in(self, scheduler_request: RolloutScheduleInfo):
        seq_groups: list["SeqGroupInfo"] = scheduler_request.data
        if self.status_manager.num_seq_group_running == 0:
            # When migrate_in happens, if there is no running task, rollout() will
            # be waiting for a notification, we need to notify it to continue.
            # Otherwise, rollout() will continue to run until all tasks are done.
            self.status_manager.notify()
        for group in seq_groups:
            task = asyncio.create_task(self.worker._async_generate_group(group))
            self.status_manager.add_task(group, task)

    async def _wait_for_finish(self):
        await self._wait_until_no_running_task()
        self.status_manager.notify()

    async def _wait_until_no_running_task(self):
        # After rollout() launches tasks initially, only migrate_in can increase num_seq_group_running.
        # migrate_in will not be called concurrently with other RolloutScalingScheduler methods,
        # so num_seq_group_running will not increase between the return of this coroutine and when the caller regains control.
        while self.status_manager.num_seq_group_running > 0:
            await asyncio.sleep(0.1)

    async def report_offloaded(self):
        """Report that this rollout instance has been offloaded."""
        scheduler_response = RolloutScheduleInfo(
            instance_id=self._rank, action=RolloutAction.Offloaded
        )
        await self.scheduler_channel.put(
            scheduler_response,
            key=self.scheduler_response_queue,
            async_op=True,
        ).async_wait()

    async def main_loop(self):
        """Asynchronous main loop for processing scheduler requests.

        This coroutine runs an infinite event loop that waits for RolloutScheduleInfo
        requests from the scheduler_channel and dispatches handling based on the
        RolloutAction contained in each request. It is intended to be run as a
        background task and will only terminate if cancelled or if an unhandled
        exception is raised.
        """
        while True:
            request: RolloutScheduleInfo = await self.scheduler_channel.get(
                key=self.scheduler_request_queue, async_op=True
            ).async_wait()

            match request.action:
                case RolloutAction.Report:
                    await self._report()
                case RolloutAction.Migrate_In:
                    await self._migrate_in(request)
                case RolloutAction.Migrate_Out:
                    await self._migrate_out()
                case RolloutAction.Wait_For_Finish | RolloutAction.Finish:
                    await self._wait_for_finish()
                case _:
                    raise ValueError(f"Unknown scheduler action: {request.action}")
