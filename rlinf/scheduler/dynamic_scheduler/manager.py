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
from logging import Logger
from typing import List

from omegaconf import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler.dynamic_scheduler.utils import (
    RolloutAction,
    RolloutScheduleInfo,
    get_scheduler_request_queue,
    get_scheduler_response_queue,
)


class ComponentManager:
    """ComponentManager is a base class for all component managers."""

    def __init__(
        self,
        component_role: str,
        config: DictConfig,
        channel: Channel,
        model_parallel_size: int,
        instance_num: int,
        communication_instance_num: int,
        _logger: Logger,
        use_pre_process_policy: bool = True,
        use_wait_before_last_iter_policy: bool = True,
    ):
        """Initialize the ComponentManager.

        Args:
            component_role (str): The role of the component.
            config (DictConfig): The configuration of this training task.
            channel (Channel): The channel for inter-worker communication.
            model_parallel_size (int): The model parallel size.
            instance_num (int): The number of instances.
            communication_instance_num (int): The number of communication instances.
            _logger (Logger): The logger for this training task.
            use_pre_process_policy (bool): Whether to use the pre-process policy.
            use_wait_before_last_iter_policy (bool): Whether to use the wait before last iter policy.
        """
        if component_role == "dummy":
            self.component_role = "dummy"
            self.cfg = config
            self._logger = _logger
            self.model_parallel_size = 0
            # self.reset(instance_num)
            return
        self.component_role = component_role
        self.cfg = config
        self.channel = channel
        self._logger = _logger

        self.use_pre_process_policy = use_pre_process_policy
        self.use_wait_before_last_iter_policy = use_wait_before_last_iter_policy

        self.n_minibatches = self.cfg.algorithm.n_minibatches

        for instance_id in range(communication_instance_num):
            self.channel.create_queue(get_scheduler_request_queue(instance_id))
            self.channel.create_queue(get_scheduler_response_queue(instance_id))

        self.model_parallel_size = model_parallel_size

        self.reset(instance_num)

    def release(self, release_gpu_num: int = 0, release_instance_num: int = 0):
        """Update the current GPU and instance num after release.

        Args:
            release_gpu_num (int): The number of GPU resources to release.
            release_instance_num (int): The number of instances to release.
        """
        assert release_gpu_num == 0 or release_instance_num == 0
        if release_gpu_num == 0 and release_instance_num == 0:
            return

        if release_gpu_num != 0:
            assert self.current_gpu_num >= release_gpu_num
            assert release_gpu_num % self.model_parallel_size == 0
            self.current_gpu_num -= release_gpu_num
            self.current_instance_num = self.current_gpu_num // self.model_parallel_size
            self.current_instance_offset += release_gpu_num // self.model_parallel_size
        else:
            assert self.current_instance_num >= release_instance_num
            self.current_gpu_num -= release_instance_num * self.model_parallel_size
            self.current_instance_num -= release_instance_num
            self.current_instance_offset += release_instance_num

    async def pre_process(self, *args, **kwargs):
        """Pre-process the component."""
        pass

    async def post_process(self, *args, **kwargs):
        """Pre-process the component."""
        pass

    # main loop
    def reset(self, instance_num: int):
        """Reset state of ComponentManager.

        Args:
            instance_num (int): The number of instances.
        """
        self.current_instance_num = instance_num
        self.current_gpu_num = self.current_instance_num * self.model_parallel_size
        self.current_instance_offset = 0

        self.init_instance_num = instance_num
        self.init_gpu_num = self.init_instance_num * self.model_parallel_size

    async def main_loop_finish(self, *args, **kwargs):
        """Main loop finish."""
        pass

    async def release_resource(self, *args, **kwargs) -> int:
        """Release the GPU resources.

        Returns:
            int: The number of released GPU resources.
        """
        return 0

    async def allocate_resource(self, *args, **kwargs):
        """Allocate the GPU resources."""
        pass


class RolloutManager(ComponentManager):
    """Manage resource allocation for rollout."""

    def __init__(
        self,
        config: DictConfig,
        channel: Channel,
        model_parallel_size: int,
        instance_num: int,
        communication_instance_num: int,
        _logger: Logger,
        use_pre_process_policy: bool,
        use_wait_before_last_iter_policy: bool,
    ):
        """Initialize the RolloutManager."""
        super().__init__(
            component_role="rollout",
            config=config,
            channel=channel,
            model_parallel_size=model_parallel_size,
            instance_num=instance_num,
            communication_instance_num=communication_instance_num,
            _logger=_logger,
            use_pre_process_policy=use_pre_process_policy,
            use_wait_before_last_iter_policy=use_wait_before_last_iter_policy,
        )

        self.reports = []
        self.rollout_total_tasks = (
            self.cfg.algorithm.group_size * self.cfg.data.rollout_batch_size
        )

    async def pre_process(self, migrate_out_instance_num):
        """Pre-process the rollout instances."""
        assert self.use_pre_process_policy
        while True:
            report_str = await self.check_report()
            await asyncio.sleep(1)
            if self.running_tasks <= (self.rollout_total_tasks // 2):
                self._logger.info("\npre_process condition finished:\n" + report_str)
                await self.execute_migrate(migrate_out_instance_num)
                break

    async def check_report(self):
        """Check the report of rollout instances."""
        self.reports = [None for i in range(self.current_instance_num)]
        for instance_id in range(self.current_instance_num):
            rollout_instance_id = instance_id + self.current_instance_offset
            await self.channel.put(
                RolloutScheduleInfo(action=RolloutAction.Report),
                queue_name=get_scheduler_request_queue(rollout_instance_id),
                async_op=True,
            ).async_wait()

            response = await self.channel.get(
                queue_name=get_scheduler_response_queue(rollout_instance_id),
                async_op=True,
            ).async_wait()
            assert (
                response.instance_id == rollout_instance_id
                and response.report is not None
            ), f"rollout_instance_id={rollout_instance_id}, response={response}"
            self.reports[instance_id] = response.report

        total_tasks = sum(report.total_tasks for report in self.reports)
        running_tasks = sum(report.running_tasks for report in self.reports)

        if not hasattr(self, "running_tasks"):
            self.running_tasks = self.rollout_total_tasks

        completed_tasks = self.running_tasks - running_tasks
        self.running_tasks = running_tasks
        self.pre_iter_speed_per_instance = completed_tasks // self.current_instance_num

        report_str = f"Rollout Report:\ncurrent_total_tasks={total_tasks}, current_running_tasks={running_tasks}, pre-iter completed_tasks={completed_tasks}, pre_iter_speed_per_instance={self.pre_iter_speed_per_instance}\n"
        for i, report in enumerate(self.reports):
            report_str += f"rollout{i + self.current_instance_offset} : total_tasks={report.total_tasks}, running_tasks={report.running_tasks}, completed_tasks={report.completed_tasks}\n"
        return report_str

    async def offload(self, action: RolloutAction) -> int:
        """Offload the rollout instances.

        Args:
            action (RolloutAction): The action to offload.

        Returns:
            int: The number of released instances.
        """
        assert action in [RolloutAction.Finish, RolloutAction.Wait_For_Finish]
        offloaded_instance_ids = []
        for instance_id in range(self.current_instance_num):
            rollout_instance_id = instance_id + self.current_instance_offset
            await self.channel.put(
                RolloutScheduleInfo(action=action),
                queue_name=get_scheduler_request_queue(rollout_instance_id),
                async_op=True,
            ).async_wait()

            offloaded_instance_ids.append(rollout_instance_id)

        released_instance_num = self.current_instance_num
        self.release(release_instance_num=released_instance_num)
        assert self.current_gpu_num == 0

        assert len(offloaded_instance_ids) == released_instance_num
        for rollout_instance_id in offloaded_instance_ids:
            response = await self.channel.get(
                queue_name=get_scheduler_response_queue(rollout_instance_id),
                async_op=True,
            ).async_wait()
            assert (
                response.action == RolloutAction.Offloaded
                and response.instance_id == rollout_instance_id
            )

        del self.running_tasks
        return released_instance_num * self.model_parallel_size

    async def migrate_policy(self, train_iter: int) -> int:
        """Return the max number of rollout instances could migrate out.

        Args:
            train_iter (int): current train iter

        Returns:
            int: the max number of rollout instances could migrate out
        """
        if self.current_instance_num <= 1:
            return 0

        # expect main_loop_finish rollout before last train iter
        remain_iter_before_last = (self.n_minibatches - 1) - (train_iter + 1)

        # find minimum instance num to main_loop_finish rollout before last train iter
        min_instance_num = self.current_instance_num
        for instance_num in range(1, self.current_instance_num):
            expected_completed_tasks = (
                instance_num
                * self.pre_iter_speed_per_instance
                * remain_iter_before_last
            )
            if expected_completed_tasks >= self.running_tasks:
                min_instance_num = instance_num
                break

        return self.current_instance_num - min_instance_num

    async def execute_migrate(self, migrate_instance_num: int):
        """Execute the migration of rollout instances.

        Args:
            migrate_instance_num (int): The number of rollout instances to migrate out.
        """
        if migrate_instance_num == 0:
            return
        assert migrate_instance_num < self.current_instance_num
        assert len(self.reports) == self.current_instance_num

        offloaded_instance_ids = []
        # Migrate_Out
        migrate_out_batches = []
        for instance_id in range(migrate_instance_num):
            rollout_instance_id = instance_id + self.current_instance_offset
            await self.channel.put(
                RolloutScheduleInfo(action=RolloutAction.Migrate_Out),
                queue_name=get_scheduler_request_queue(rollout_instance_id),
                async_op=True,
            ).async_wait()

            response = await self.channel.get(
                queue_name=get_scheduler_response_queue(rollout_instance_id),
                async_op=True,
            ).async_wait()
            assert (
                response.instance_id == rollout_instance_id
                and response.data is not None
            )
            migrate_out_batches += response.data

            offloaded_instance_ids.append(rollout_instance_id)

        self.release(release_instance_num=migrate_instance_num)

        # migrate_out_running_tasks = sum(batch.get_running_tasks() for batch in migrate_out_batches)
        instance_running_tasks_expected = (
            self.running_tasks // self.current_instance_num
        )
        # assert instance_running_tasks_expected > 0

        migrate_out_batches_index = 0
        migrate_out_batches_len = len(migrate_out_batches)
        self._logger.info(
            f"[Migrate-Info] migrate_out_batches_len: {migrate_out_batches_len}, running_tasks={self.running_tasks}, instance_running_tasks_expected={instance_running_tasks_expected}"
        )

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
                await self.channel.put(
                    migrate_in_request,
                    queue_name=get_scheduler_request_queue(rollout_instance_id),
                    async_op=True,
                ).async_wait()
        assert migrate_out_batches_index == migrate_out_batches_len

        # wait before migrate out instance is offloaded
        assert len(offloaded_instance_ids) == migrate_instance_num
        for rollout_instance_id in offloaded_instance_ids:
            response = await self.channel.get(
                queue_name=get_scheduler_response_queue(rollout_instance_id),
                async_op=True,
            ).async_wait()
            assert (
                response.action == RolloutAction.Offloaded
                and response.instance_id == rollout_instance_id
            )

    async def find_release_instance_num_needed(
        self,
        release_instance_num_max: int,
        actor_valid_dp_sizes: List[int],
        actor_model_parallel_size: int,
        actor_current_instance_num: int,
    ) -> int:
        """Find the number of rollout instances needed to release.

        Args:
            release_instance_num_max (int): The maximum number of rollout instances to release.
            actor_valid_dp_sizes (List[int]): The valid DP sizes of the actor.
            actor_model_parallel_size (int): The model parallel size of the actor.
            actor_current_instance_num (int): The current number of instances of the actor.

        Returns:
            int: The number of rollout instances needed to release.
        """
        if release_instance_num_max == 0:
            return 0

        assert actor_current_instance_num in actor_valid_dp_sizes
        index = actor_valid_dp_sizes.index(actor_current_instance_num)
        assert index < len(actor_valid_dp_sizes)

        released_gpu_num_max = release_instance_num_max * self.model_parallel_size

        actor_increment_gpu_num = 0
        for actor_dp_size in actor_valid_dp_sizes[index + 1 :]:
            actor_gpu_num_needed = (
                actor_dp_size - actor_current_instance_num
            ) * actor_model_parallel_size
            if actor_gpu_num_needed <= released_gpu_num_max:
                actor_increment_gpu_num = actor_gpu_num_needed
            else:
                break

        release_instance_num_needed = math.ceil(
            actor_increment_gpu_num / self.model_parallel_size
        )
        if release_instance_num_needed > release_instance_num_max:
            return self.find_release_instance_num_needed(
                release_instance_num_max - 1,
                actor_valid_dp_sizes,
                actor_model_parallel_size,
                actor_current_instance_num,
            )
        return release_instance_num_needed

    async def release_resource(
        self,
        train_iter: int,
        actor_valid_dp_sizes: List[int],
        actor_model_parallel_size: int,
        actor_current_instance_num: int,
    ) -> int:
        """Release the GPU resources.

        Args:
            train_iter (int): The current train iter.
            actor_valid_dp_sizes (List[int]): The valid DP sizes of the actor.
            actor_model_parallel_size (int): The model parallel size of the actor.
            actor_current_instance_num (int): The current number of instances of the actor.

        Returns:
            int: The number of released GPU resources.
        """
        # Step1 : get report
        if self.current_instance_num == 0:
            return 0

        report_str = await self.check_report()
        self._logger.info(report_str)

        if (self.running_tasks == 0) or (train_iter == self.n_minibatches - 1):
            assert self.running_tasks == 0
            return await self.offload(action=RolloutAction.Finish)

        if (
            train_iter == self.n_minibatches - 2
            and self.use_wait_before_last_iter_policy
        ):
            return await self.offload(action=RolloutAction.Wait_For_Finish)

        release_instance_num_max = await self.migrate_policy(train_iter)

        release_instance_num_needed = await self.find_release_instance_num_needed(
            release_instance_num_max,
            actor_valid_dp_sizes,
            actor_model_parallel_size,
            actor_current_instance_num,
        )

        self._logger.info(
            f"[Release-Info] train_iter={train_iter}, release_instance_num_max={release_instance_num_max}, release_instance_num_needed={release_instance_num_needed}"
        )

        await self.execute_migrate(release_instance_num_needed)
        return release_instance_num_needed * self.model_parallel_size


class InferenceManager(ComponentManager):
    """Manage resource allocation for inference."""

    def __init__(
        self,
        config: DictConfig,
        channel: Channel,
        model_parallel_size: int,
        instance_num: int,
        communication_instance_num: int,
        _logger: Logger,
        use_pre_process_policy: bool,
        use_wait_before_last_iter_policy: bool,
    ):
        """Initialize the InferenceManager."""
        super().__init__(
            component_role="inference",
            config=config,
            channel=channel,
            model_parallel_size=model_parallel_size,
            instance_num=instance_num,
            communication_instance_num=communication_instance_num,
            _logger=_logger,
            use_pre_process_policy=use_pre_process_policy,
            use_wait_before_last_iter_policy=use_wait_before_last_iter_policy,
        )

    def reset(self, instance_num):
        """Reset the state of the InferenceManager."""
        super().reset(instance_num)

    async def last_iter_release_policy(self) -> int:
        """Release the GPU resources at the last iter.

        This policy will block training until the inference is finished.
        """
        while not self.main_loop_finished_handler.done():
            await asyncio.sleep(0.1)

        release_gpu_num = self.current_gpu_num
        self.release(release_gpu_num=release_gpu_num)
        return release_gpu_num

    async def main_loop_finish(self):
        """Finish the main loop of the InferenceManager."""
        assert self.main_loop_finished_handler.done()

    async def release_resource(
        self, train_iter: int, all_rollout_offloaded: bool
    ) -> int:
        """Release the GPU resources."""
        if self.current_instance_num == 0:
            return 0

        if train_iter == 0:
            self.main_loop_finished_handler = self.channel.get(
                queue_name=get_scheduler_response_queue(), async_op=True
            )

        if not self.use_wait_before_last_iter_policy:
            release_gpu_num = (
                self.current_gpu_num if self.main_loop_finished_handler.done() else 0
            )
            self.release(release_gpu_num=release_gpu_num)
            return release_gpu_num

        if (train_iter == self.n_minibatches - 2) or all_rollout_offloaded:
            return await self.last_iter_release_policy()
        return 0


class ActorManager(ComponentManager):
    """Manage resource allocation for actor."""

    def __init__(
        self,
        config: DictConfig,
        channel: Channel,
        model_parallel_size: int,
        instance_num: int,
        communication_instance_num: int,
        _logger: Logger,
        use_pre_process_policy: bool,
        use_wait_before_last_iter_policy: bool,
        valid_dp_sizes: List[int],
    ):
        """Initialize the ActorManager."""
        super().__init__(
            component_role="actor",
            config=config,
            channel=channel,
            model_parallel_size=model_parallel_size,
            instance_num=instance_num,
            communication_instance_num=communication_instance_num,
            _logger=_logger,
            use_pre_process_policy=use_pre_process_policy,
            use_wait_before_last_iter_policy=use_wait_before_last_iter_policy,
        )

        self.valid_dp_sizes = valid_dp_sizes

    async def pre_process(self):
        """Pre-process the actor instances."""
        await self.channel.put(
            None, queue_name=get_scheduler_request_queue(), async_op=True
        ).async_wait()

    async def main_loop_finish(self):
        """Finish the main loop of the ActorManager."""
        return self.init_gpu_num

    async def try_allocate(self, available_gpu_num: int) -> int:
        """Try to allocate the GPU resources.

        Args:
            available_gpu_num (int): The number of available GPU resources.

        Returns:
            int: The number of allocated GPU resources.
        """
        if available_gpu_num < self.model_parallel_size:
            return 0

        increment_world_size = 0
        assert (
            self.current_instance_num in self.valid_dp_sizes
            and self.current_instance_num != self.valid_dp_sizes[-1]
        )
        index = self.valid_dp_sizes.index(self.current_instance_num)
        for next_dp_size in self.valid_dp_sizes[index + 1 :]:
            needed_gpu_nums = (
                next_dp_size - self.current_instance_num
            ) * self.model_parallel_size
            if needed_gpu_nums <= available_gpu_num:
                increment_world_size = needed_gpu_nums
            else:
                break

        assert increment_world_size <= available_gpu_num
        return increment_world_size

    async def allocate_resource(self, available_gpu_num: int, train_iter: int):
        """Allocate the GPU resources.

        Args:
            available_gpu_num (int): The number of available GPU resources.
            train_iter (int): The current train iter.
        """
        new_gpu_num = self.current_gpu_num
        if train_iter == self.n_minibatches - 1:
            new_gpu_num = await self.main_loop_finish()
        elif available_gpu_num != 0:
            new_gpu_num += await self.try_allocate(available_gpu_num)

        if new_gpu_num != self.current_gpu_num:
            self.current_gpu_num = new_gpu_num
            self.current_instance_num = new_gpu_num // self.model_parallel_size
            await self.channel.put(
                {"world_size": new_gpu_num},
                queue_name=get_scheduler_request_queue(),
                async_op=True,
            ).async_wait()
        else:
            await self.channel.put(
                None, queue_name=get_scheduler_request_queue(), async_op=True
            ).async_wait()

    async def wait_for_actor_update(self):
        """Wait for the actor update."""
        await self.channel.get(
            queue_name=get_scheduler_response_queue(), async_op=True
        ).async_wait()
