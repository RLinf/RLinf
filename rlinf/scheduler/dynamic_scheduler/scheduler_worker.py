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

from omegaconf import DictConfig

from rlinf.scheduler import Worker
from rlinf.scheduler.dynamic_scheduler.manager import (
    ActorManager,
    ComponentManager,
    InferenceManager,
    RolloutManager,
)
from rlinf.scheduler.dynamic_scheduler.utils import (
    get_scheduler_channel,
    get_valid_dp_sizes,
)
from rlinf.utils.placement import ComponentPlacement


class SchedulerWorker(Worker):
    """Dynamic Scheduler."""

    def __init__(self, config: DictConfig, component_placement: ComponentPlacement):
        """Initialize the SchedulerWorker."""
        super().__init__()
        self.cfg = config
        self.component_placement = component_placement
        self.components = self.component_placement._components
        self.total_gpus = self.component_placement._cluster_num_gpus

        assert self.cfg.rollout.rollout_backend == "sglang", (
            "only sglang is supported for dynamic scheduler"
        )
        assert self.cfg.actor.training_backend == "megatron", (
            "only megatron is supported for dynamic scheduler"
        )

        # Set policies for scheduler
        self.use_pre_process_policy = getattr(
            self.cfg.cluster, "use_pre_process_policy", True
        )
        self.use_wait_before_last_iter_policy = getattr(
            self.cfg.cluster, "use_wait_before_last_iter_policy", True
        )

        assert "rollout" in self.components, "rollout component is required"
        assert "actor" in self.components, "actor component is required"

        self.component_channels = {}
        for component in self.components:
            self.component_channels[component] = self.create_channel(
                get_scheduler_channel(component)
            )

        # Note. mode_parallel_size here represents the number of GPUs, the quantity required for a single instance
        self.init_rollout_instance_num = component_placement.rollout_dp_size
        self.init_rollout_gpu_num = component_placement.rollout_world_size
        self.rollout_model_parallel_size = (
            self.init_rollout_gpu_num // self.init_rollout_instance_num
        )

        self.init_actor_instance_num = component_placement.actor_dp_size
        self.init_actor_gpu_num = component_placement.actor_world_size
        self.actor_model_parallel_size = (
            self.init_actor_gpu_num // self.init_actor_instance_num
        )

        self.init_inference_instance_num = component_placement.inference_world_size
        self.init_inference_gpu_num = component_placement.inference_world_size
        self.inference_model_parallel_size = (
            0
            if self.init_inference_gpu_num == 0
            else (self.init_inference_gpu_num // self.init_inference_instance_num)
        )

        # Get valid dp size list for actor
        self.actor_valid_dp_sizes = get_valid_dp_sizes(
            self.cfg,
            self.component_placement._cluster_num_gpus,
            self.actor_model_parallel_size,
        )

        # Create ComponentManager for each component
        self.rollout_manager = RolloutManager(
            config=config,
            channel=self.component_channels["rollout"],
            model_parallel_size=self.rollout_model_parallel_size,
            instance_num=self.init_rollout_instance_num,
            communication_instance_num=(
                self.total_gpus // self.rollout_model_parallel_size
            ),
            _logger=self._logger,
            use_pre_process_policy=self.use_pre_process_policy,
            use_wait_before_last_iter_policy=self.use_wait_before_last_iter_policy,
        )

        self.actor_manager = ActorManager(
            config=config,
            channel=self.component_channels["actor"],
            model_parallel_size=self.actor_model_parallel_size,
            instance_num=self.init_actor_instance_num,
            communication_instance_num=1,
            _logger=self._logger,
            valid_dp_sizes=self.actor_valid_dp_sizes,
            use_pre_process_policy=self.use_pre_process_policy,
            use_wait_before_last_iter_policy=self.use_wait_before_last_iter_policy,
        )

        self.inference_manager = ComponentManager(
            "dummy", None, None, None, None, None, None
        )
        if self.inference_model_parallel_size > 0:
            self.inference_manager = InferenceManager(
                config=config,
                channel=self.component_channels["inference"],
                model_parallel_size=self.inference_model_parallel_size,
                instance_num=self.init_inference_instance_num,
                communication_instance_num=1,
                _logger=self._logger,
                use_pre_process_policy=self.use_pre_process_policy,
                use_wait_before_last_iter_policy=self.use_wait_before_last_iter_policy,
            )

    async def run(self):
        """Run the scheduler."""
        await self.pre_process()
        await self.main_loop()
        await self.post_process()

    async def pre_process(self):
        """Pre process.

        Policy : Allocate resource to rollout first, then allocate resource to actor.
        """
        self.rollout_manager.reset(self.init_rollout_instance_num)
        self.actor_manager.reset(self.init_actor_instance_num)
        self.inference_manager.reset(self.init_inference_instance_num)

        if not self.use_pre_process_policy:
            return

        assert self.init_actor_gpu_num % self.rollout_model_parallel_size == 0
        migrate_out_instance_num = (
            self.init_actor_gpu_num // self.rollout_model_parallel_size
        )
        await self.rollout_manager.pre_process(migrate_out_instance_num)

        await self.actor_manager.pre_process()
        await self.inference_manager.pre_process()

        # Offset may changed by pre_process()
        assert (
            self.rollout_manager.current_instance_num
            == self.init_rollout_instance_num - migrate_out_instance_num
        )
        assert self.rollout_manager.current_instance_offset == migrate_out_instance_num

    async def post_process(self):
        """Post Process."""
        pass

    async def main_loop(self):
        """Main loop.

        The signal synchronization granularity is consistent with the actor training granularity.
        Resource allocation is performed after the actor has completed one execution of optimizer.step().
        """
        for train_iter in range(self.cfg.algorithm.n_minibatches):
            # Wait for actor ready to update

            await self.actor_manager.wait_for_actor_update()

            # Trying to release the resource of rollout and inference
            rollout_released_gpu_num = await self.rollout_manager.release_resource(
                train_iter,
                self.actor_valid_dp_sizes,
                self.actor_model_parallel_size,
                self.actor_manager.current_instance_num,
            )
            inference_released_gpu_num = await self.inference_manager.release_resource(
                train_iter, self.rollout_manager.current_instance_num == 0
            )
            self.log_info(
                f"[Release-Info] train_iter={train_iter}, rollout_released_gpu_num={rollout_released_gpu_num}, inference_released_gpu_num={inference_released_gpu_num}"
            )

            # Trying to allocate the resource to actor
            await self.actor_manager.allocate_resource(
                rollout_released_gpu_num + inference_released_gpu_num, train_iter
            )
