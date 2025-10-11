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
import inspect
import math
from logging import Logger
from typing import Callable, List, Optional, Tuple

from omegaconf import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler.dynamic_scheduler.utils import (
    RolloutAction,
    RolloutScheduleInfo,
    get_scheduler_channel,
    get_scheduler_request_queue,
    get_scheduler_response_queue,
)
from rlinf.utils.placement import ComponentPlacement


class ComponentManager:
    """ComponentManager is a base class for all component managers.

    Specific component managers should inherit this class and override the following methods:
    - pre_process_impl : Pre-process implementation
    - main_loop_finalize : Process after the last training iteration in main_loop
    - release_resource (Optional) : Release the resource of the component
    - allocate_resource (Optional) : Allocate the resource of the component

    NOTE. Subclasses must override exactly one of the release_resource and allocate_resource methods. They cannot override both or neither.
    """

    def __init__(
        self,
        component_role: str,
        config: DictConfig,
        component_placement: ComponentPlacement,
        use_pre_process_policy: bool,
        use_wait_before_last_iter_policy: bool,
        channel_factory: Callable[[str], Channel],
        _logger: Logger,
    ):
        """Initialize the ComponentManager.

        Args:
            component_role (str): The role of the component.
            config (DictConfig): The configuration of this training task.
            component_placement (ComponentPlacement): The component placement.
            use_pre_process_policy (bool): Whether to use the pre-process policy.
            use_wait_before_last_iter_policy (bool): Whether to use the wait before last iter policy.
            channel_factory (Callable[[str], Channel]): The factory for creating channels.
            _logger (Logger): The logger for this training task.
        """
        self._check_method_overrides()
        self.component_role = component_role
        self.cfg = config
        self.component_placement = component_placement
        self.use_pre_process_policy = use_pre_process_policy
        self.use_wait_before_last_iter_policy = use_wait_before_last_iter_policy
        self.channel_factory = channel_factory
        self._logger = _logger
        self.n_minibatches = self.cfg.algorithm.n_minibatches

        if self.component_role not in self.component_placement._components:
            self.component_role = "dummy"
            self.init_instance_num = 0
            self.init_gpu_num = 0
            self.model_parallel_size = 0
            return

        self.init_instance_num = getattr(
            component_placement, f"{self.component_role}_dp_size"
        )
        self.init_gpu_num = getattr(
            component_placement, f"{self.component_role}_world_size"
        )
        # Note. mode_parallel_size here represents the number of GPUs, the quantity required for a single instance
        self.model_parallel_size = self.init_gpu_num // self.init_instance_num

    def _check_method_overrides(self):
        """Check that the subclass overrides exactly one of the required methods.

        subclass should override pre_process_impl() and one of [release_resource(), allocate_resource()]
        """
        pre_process_impl_overridden = (
            type(self).pre_process_impl != ComponentManager.pre_process_impl
        )
        main_loop_finalize_overridden = (
            type(self).main_loop_finalize != ComponentManager.main_loop_finalize
        )
        if not pre_process_impl_overridden:
            raise TypeError(
                f"{self.__class__.__name__} must override pre_process_impl()"
            )
        if not main_loop_finalize_overridden:
            raise TypeError(
                f"{self.__class__.__name__} must override main_loop_finalize()"
            )

        self.release_overridden = (
            type(self).release_resource != ComponentManager.release_resource
        )
        self.allocate_overridden = (
            type(self).allocate_resource != ComponentManager.allocate_resource
        )

        if self.release_overridden and self.allocate_overridden:
            raise TypeError(
                f"{self.__class__.__name__} must override only one of "
                "release_resource() or allocate_resource(), not both."
            )
        elif not self.release_overridden and not self.allocate_overridden:
            raise TypeError(
                f"{self.__class__.__name__} must override either "
                "release_resource() or allocate_resource()."
            )

        self.release_fn_parameters = inspect.signature(
            self.release_resource
        ).parameters.keys()
        self.allocate_fn_parameters = inspect.signature(
            self.allocate_resource
        ).parameters.keys()

    def create_channel(self, communication_instance_num: int):
        """Create channel queues for communication.

        Args:
            communication_instance_num (int): The number of communication instances.
        """
        self.channel = self.channel_factory(get_scheduler_channel(self.component_role))
        for instance_id in range(communication_instance_num):
            self.channel.create_queue(get_scheduler_request_queue(instance_id))
            self.channel.create_queue(get_scheduler_response_queue(instance_id))

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

    async def release_or_allocate(self, *args, **kwargs) -> Tuple[int, int]:
        """Execute release_resource or allocate_resource for this component.

        Returns:
            released_gpu_num (int): The number of released GPU resources.
            incremental_gpu_num (int): The number of incremental GPU resources.
        """
        train_iter = kwargs["train_iter"]
        if train_iter == self.n_minibatches - 1:
            await self.main_loop_finalize()
            return 0, 0

        release_fn_kwargs = {}
        allocate_fn_kwargs = {}
        if self.release_overridden:
            for key in self.release_fn_parameters:
                release_fn_kwargs[key] = kwargs[key]
            allocate_fn_kwargs = kwargs
        else:
            for key in self.allocate_fn_parameters:
                allocate_fn_kwargs[key] = kwargs[key]
            release_fn_kwargs = kwargs

        released_gpu_num = await self.release_resource(**release_fn_kwargs)
        incremental_gpu_num = await self.allocate_resource(**allocate_fn_kwargs)
        return (released_gpu_num, incremental_gpu_num)

    # ------------------------------------------------- Need to be overridden by subclass -------------------------------------------------
    async def pre_process_impl(self, *args, **kwargs):
        """Implement of pre_process."""
        pass

    async def main_loop_finalize(self):
        """Processing after the last training iteration in main_loop."""
        pass

    async def release_resource(self, *args, **kwargs) -> int:
        """Release the GPU resources.

        Returns:
            int: The number of released GPU resources.
        """
        return 0

    async def allocate_resource(self, *args, **kwargs) -> int:
        """Allocate the GPU resources.

        Returns:
            int: The number of incremental GPU resources.
        """
        return 0


class RolloutManager(ComponentManager):
    """Manage resource allocation for rollout.

    There are three core actions for rollout instances:

    - report  : collect the report from all alive rollout instances
    - finish  : send Finish or Wait_For_Finish signal to all alive rollout instances
        - check_offloaded : check if the rollout instances with id in in finished_instance_ids have been finished
    - migrate : migrate the rollout instances
        - migrate_policy : return the max number of rollout instances could migrate out
        - find_release_instance_num_needed : find the number of rollout instances needed to release
        - TODO(balance_batches) : balance the batches between the rollout instances
    """

    def __init__(
        self,
        config: DictConfig,
        component_placement: ComponentPlacement,
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
        self.create_channel(self.init_instance_num)

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
            await asyncio.sleep(1)
            if self.running_tasks <= running_tasks_threshold:
                self._logger.info("\npre_process condition satisfied:\n" + report_str)
                await self.migrate(migrate_out_instance_num)
                break

    async def main_loop_finalize(self):
        """Processing after the last training iteration in main_loop. Perform RolloutAction.Finish on all surviving instances."""
        if self.current_instance_num == 0:
            return

        await self.finish(action=RolloutAction.Finish)

    async def release_resource(
        self,
        train_iter: int,
        actor_current_instance_num: int,
    ) -> int:
        """Release the GPU resources.

        Args:
            train_iter (int): The current train iter.
            actor_current_instance_num (int): The current number of instances of the actor.

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
        released_instance_num = self.migrate_policy(
            train_iter, actor_current_instance_num
        )
        released_gpu_num = await self.migrate(released_instance_num)
        return released_gpu_num

    # ------------------------------------------------- override end -------------------------------------------------

    async def send_request(
        self,
        request: RolloutScheduleInfo,
        rollout_instance_id: int,
        need_feedback: bool = True,
    ) -> Optional[RolloutScheduleInfo]:
        """Send the request to the rollout instance.

        Args:
            request (RolloutScheduleInfo): The request to send.
            rollout_instance_id (int): The id of the rollout instance.
            need_feedback (bool): Whether to wait for the response.

        Returns:
            RolloutScheduleInfo: The response from the rollout instance.
        """
        assert (
            rollout_instance_id >= self.current_instance_offset
            and rollout_instance_id < self.init_instance_num
        ), (
            f"rollout instance id={rollout_instance_id} is not in valid range=[{self.current_instance_offset}, {self.init_instance_num})"
        )

        await self.channel.put(
            request,
            queue_name=get_scheduler_request_queue(rollout_instance_id),
            async_op=True,
        ).async_wait()

        if not need_feedback:
            return None

        response = await self.channel.get(
            queue_name=get_scheduler_response_queue(rollout_instance_id),
            async_op=True,
        ).async_wait()
        assert response.instance_id == rollout_instance_id, (
            f"rollout_instance_id={rollout_instance_id}, response={response}"
        )
        return response

    async def report(self):
        """Check the report of rollout instances."""
        self.reports = [None for i in range(self.current_instance_num)]
        report_request = RolloutScheduleInfo(action=RolloutAction.Report)
        for instance_id in range(self.current_instance_num):
            rollout_instance_id = instance_id + self.current_instance_offset
            response = await self.send_request(report_request, rollout_instance_id)
            assert response.report is not None
            self.reports[instance_id] = response.report

        total_tasks = sum(report.total_tasks for report in self.reports)
        running_tasks = sum(report.running_tasks for report in self.reports)
        completed_tasks = self.running_tasks - running_tasks
        self.running_tasks = running_tasks
        self.pre_iter_speed_per_instance = completed_tasks // self.current_instance_num

        report_str = f"Rollout Report:\ncurrent_total_tasks={total_tasks}, current_running_tasks={running_tasks}, pre-iter completed_tasks={completed_tasks}, pre_iter_speed_per_instance={self.pre_iter_speed_per_instance}\n"
        for i, report in enumerate(self.reports):
            report_str += f"rollout{i + self.current_instance_offset} : total_tasks={report.total_tasks}, running_tasks={report.running_tasks}, completed_tasks={report.completed_tasks}\n"
        return report_str

    async def check_offloaded(self, finished_instance_ids: List[int]):
        """Check if the rollout instances with id in in finished_instance_ids have been finished.

        Args:
            finished_instance_ids (List[int]): The list of finished instance ids.
        """
        for rollout_instance_id in finished_instance_ids:
            response = await self.channel.get(
                queue_name=get_scheduler_response_queue(rollout_instance_id),
                async_op=True,
            ).async_wait()
            assert (
                response.action == RolloutAction.Offloaded
                and response.instance_id == rollout_instance_id
            )

    async def finish(self, action: RolloutAction) -> int:
        """Finish the rollout instances.

        Args:
            action (RolloutAction): The action to finish.

        Returns:
            int: The number of released GPU resources.
        """
        assert action in [RolloutAction.Finish, RolloutAction.Wait_For_Finish]

        # Send action to all rollout instances
        finished_instance_ids = []
        for instance_id in range(self.current_instance_num):
            rollout_instance_id = instance_id + self.current_instance_offset
            await self.send_request(
                RolloutScheduleInfo(action=action),
                rollout_instance_id,
                need_feedback=False,
            )
            finished_instance_ids.append(rollout_instance_id)

        # Wait for all rollout instances to offload
        await self.check_offloaded(finished_instance_ids)

        released_instance_num = self.current_instance_num
        self.update(released_instance_num=released_instance_num)
        return released_instance_num * self.model_parallel_size

    async def migrate(self, migrate_instance_num: int) -> int:
        """Execute the migration of rollout instances.

        Args:
            migrate_instance_num (int): The number of rollout instances to migrate out.

        Returns:
            int: The number of released GPU resources.

        1. Send Migrate_Out signal to [current_instance_offset, current_instance_offset + migrate_instance_num) rollout instances
        2. Collect the migrate batches from those rollout instances
        3. Update the state of the ComponentManager
        4. Send Migrate_In signal and migrate batches to alive rollout instances
        5. Before return, check and wait for all migrate out instances to finish
        """
        if migrate_instance_num == 0:
            return 0
        assert migrate_instance_num < self.current_instance_num
        assert len(self.reports) == self.current_instance_num

        # Send Migrate_Out signal and collect the migrate batches
        finished_instance_ids = []
        migrate_out_batches = []
        migrate_out_request = RolloutScheduleInfo(action=RolloutAction.Migrate_Out)
        for instance_id in range(migrate_instance_num):
            rollout_instance_id = instance_id + self.current_instance_offset
            response = await self.send_request(
                migrate_out_request,
                rollout_instance_id,
            )
            assert response.data is not None
            migrate_out_batches += response.data
            finished_instance_ids.append(rollout_instance_id)

        # Update the state of the ComponentManager
        self.update(released_instance_num=migrate_instance_num)

        # Calculate the expected running tasks for each instance
        instance_running_tasks_expected = (
            self.running_tasks // self.current_instance_num
        )

        migrate_out_batches_index = 0
        migrate_out_batches_len = len(migrate_out_batches)
        migrate_out_tasks = sum(
            batch.get_running_tasks() for batch in migrate_out_batches
        )
        self._logger.info(
            f"[Migrate-Info] migrate_out_batches_len: {migrate_out_batches_len}, migrate_out_tasks={migrate_out_tasks}, self.running_tasks={self.running_tasks}, instance_running_tasks_expected={instance_running_tasks_expected}"
        )

        # Send Migrate_In signal and migrate batches to alive rollout instances
        for instance_id in range(self.current_instance_num):
            rollout_instance_id = instance_id + self.current_instance_offset
            is_last_instance = instance_id == self.current_instance_num - 1

            report = self.reports[instance_id + migrate_instance_num]
            running_tasks = report.running_tasks

            # TODO::Need get all running tasks from rollout instance for this kind of case
            if (
                not is_last_instance
                and running_tasks >= instance_running_tasks_expected
            ):
                self._logger.info(
                    f"Warning : rollout-{rollout_instance_id} has {running_tasks} running tasks >  expected {instance_running_tasks_expected}"
                )
                continue

            migrate_in_batches = []
            while (migrate_out_batches_index < migrate_out_batches_len) and (
                running_tasks < instance_running_tasks_expected
            ):
                migrate_batch = migrate_out_batches[migrate_out_batches_index]
                migrate_in_batches.append(migrate_batch)
                migrate_out_batches_index += 1
                running_tasks += migrate_batch.get_running_tasks()

            self._logger.info(
                f"[Migrate-Info] rollout-{rollout_instance_id} migrate_in_batches: {len(migrate_in_batches)}, running_tasks={report.running_tasks} -> {running_tasks} ~= {instance_running_tasks_expected}"
            )

            if is_last_instance and (
                migrate_out_batches_index != migrate_out_batches_len
            ):
                migrate_in_batches += migrate_out_batches[migrate_out_batches_index:]
                running_tasks += sum(
                    migrate_batch.get_running_tasks()
                    for migrate_batch in migrate_out_batches[migrate_out_batches_index:]
                )
                migrate_out_batches_index = migrate_out_batches_len
                self._logger.info(
                    f"[Migrate-Info] Error: migrate_out_batches split error, last-rollout instance get all data. running_tasks={running_tasks}"
                )

            if len(migrate_in_batches) > 0:
                migrate_in_request = RolloutScheduleInfo(
                    action=RolloutAction.Migrate_In, data=migrate_in_batches
                )
                await self.send_request(
                    migrate_in_request, rollout_instance_id, need_feedback=False
                )
        assert migrate_out_batches_index == migrate_out_batches_len

        # Wait for migrate out instances offloaded
        await self.check_offloaded(finished_instance_ids)

        return migrate_instance_num * self.model_parallel_size

    def migrate_policy(self, train_iter: int, actor_current_instance_num: int) -> int:
        """Return the max number of rollout instances could migrate out.

        Args:
            train_iter (int): current train iter.
            actor_current_instance_num (int): The current number of instances of the actor.

        Returns:
            int: the max number of rollout instances could migrate out
        """
        if self.current_instance_num <= 1:
            return 0

        # expect main_loop_finish rollout before last train iter
        remain_iter_before_last = (self.n_minibatches - 1) - train_iter

        # find minimum instance num to main_loop_finish rollout before last train iter
        min_instance_num_needed = math.ceil(
            self.running_tasks
            / (self.pre_iter_speed_per_instance * remain_iter_before_last)
        )

        released_instance_num_max = max(
            0, self.current_instance_num - min_instance_num_needed
        )
        released_instance_num_needed = self.find_release_instance_num_needed(
            released_instance_num_max, actor_current_instance_num
        )
        self._logger.info(
            f"[Release-Info] rollout migrate info: released_instance_num_max={released_instance_num_max}, released_instance_num_needed={released_instance_num_needed}"
        )
        return released_instance_num_needed

    def find_release_instance_num_needed(
        self,
        released_instance_num_max: int,
        actor_current_instance_num: int,
    ) -> int:
        """Find the number of rollout instances needed to release.

        Args:
            released_instance_num_max (int): The maximum number of rollout instances to release.
            actor_current_instance_num (int): The current number of instances of the actor.

        Returns:
            int: The number of rollout instances needed to release.
        """
        if released_instance_num_max == 0:
            return 0

        actor_model_parallel_size = (
            self.component_placement.actor_world_size
            // self.component_placement.actor_dp_size
        )

        assert actor_current_instance_num in self.actor_valid_dp_sizes
        index = self.actor_valid_dp_sizes.index(actor_current_instance_num)
        assert index < len(self.actor_valid_dp_sizes)

        released_gpu_num_max = released_instance_num_max * self.model_parallel_size

        actor_increment_gpu_num = 0
        for actor_dp_size in self.actor_valid_dp_sizes[index + 1 :]:
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
        component_placement: ComponentPlacement,
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
        self.create_channel(1)

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
        self.main_loop_finished_handler = self.channel.get(
            queue_name=get_scheduler_response_queue(), async_op=True
        )

    async def main_loop_finalize(self):
        """Processing after the last training iteration in main_loop."""
        assert self.main_loop_finished_handler.done()

    async def release_resource(
        self,
        train_iter: int,
        rollout_current_instance_num: int,
    ) -> int:
        """Release the GPU resources.

        Args:
            train_iter (int): The current train iter.
            rollout_current_instance_num (int): rollout current_instance_num.

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
        need_wait_for_finish = (train_iter == self.n_minibatches - 2) or (
            rollout_current_instance_num == 0
        )
        if need_wait_for_finish:
            return await self.wait_for_finish()

        return 0


class ActorManager(ComponentManager):
    """Manage resource allocation for actor."""

    def __init__(
        self,
        config: DictConfig,
        component_placement: ComponentPlacement,
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
        self.create_channel(1)

    async def pre_process_impl(self):
        """Pre-process implementation of actor.

        If use_pre_process_policy is True, send a signal to actor to start training.
        """
        if not self.use_pre_process_policy:
            return
        await self.channel.put(
            None, queue_name=get_scheduler_request_queue(), async_op=True
        ).async_wait()

    def try_allocate(self, available_gpu_num: int) -> int:
        """Try to allocate the GPU resources.

        Args:
            available_gpu_num (int): The number of available GPU resources.

        Returns:
            incremental_gpu_num (int): The number of incremental GPU resources of actor.
        """
        if available_gpu_num < self.model_parallel_size:
            return 0

        incremental_gpu_num = 0
        assert (
            self.current_instance_num in self.actor_valid_dp_sizes
            and self.current_instance_num != self.actor_valid_dp_sizes[-1]
        )
        index = self.actor_valid_dp_sizes.index(self.current_instance_num)
        for next_dp_size in self.actor_valid_dp_sizes[index + 1 :]:
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

        await self.channel.put(
            scale_info,
            queue_name=get_scheduler_request_queue(),
            async_op=True,
        ).async_wait()

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
        available_gpu_num: int,
    ) -> int:
        """Allocate the GPU resources.

        Based on the value of available_gpu_num, try to allocate resources.
        If the allocation result shows that the new_gpu_num != self.current_gpu_num, then send {"world_size": new_gpu_num} to actor, else send None.

        Args:
            train_iter (int): The current train iter.
            available_gpu_num (int): The number of available GPU resources.

        Returns:
            incremental_gpu_num (int): The number of incremental GPU resources of actor.
        """
        incremental_gpu_num = self.try_allocate(available_gpu_num)
        assert incremental_gpu_num >= 0

        await self.scale(incremental_gpu_num + self.current_gpu_num)

        return incremental_gpu_num

    async def wait_for_actor_update(self):
        """Wait for the actor update."""
        await self.channel.get(
            queue_name=get_scheduler_response_queue(), async_op=True
        ).async_wait()


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
    raise ValueError(f"can't find ComponentManager subcalss for {component_role}")
