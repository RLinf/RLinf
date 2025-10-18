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

from typing import Dict, List

from omegaconf import DictConfig

from rlinf.scheduler import Worker
from rlinf.scheduler.dynamic_scheduler.manager import (
    ComponentManager,
    create_component_manager,
)
from rlinf.scheduler.dynamic_scheduler.utils import (
    get_valid_dp_sizes,
)
from rlinf.utils.placement import ComponentPlacement


class SchedulerWorker(Worker):
    """Dynamic Scheduler."""

    def __init__(
        self,
        config: DictConfig,
        component_placement: ComponentPlacement,
        component_order: List[str] = ["rollout", "inference", "actor"],
    ):
        """Initialize the SchedulerWorker."""
        super().__init__()
        self.cfg = config
        self.component_placement = component_placement
        self.components = self.component_placement._components
        self.total_gpus = self.component_placement._cluster_num_gpus
        self.component_order = component_order

        assert self.cfg.rollout.rollout_backend == "sglang", (
            "only sglang is supported for dynamic scheduler"
        )
        assert self.cfg.actor.training_backend == "megatron", (
            "only megatron is supported for dynamic scheduler"
        )
        assert "rollout" in self.components, "rollout component is required"
        assert "actor" in self.components, "actor component is required"

        # Set policies for scheduler
        self.use_pre_process_policy = getattr(
            self.cfg.cluster, "use_pre_process_policy", True
        )
        self.use_wait_before_last_iter_policy = getattr(
            self.cfg.cluster, "use_wait_before_last_iter_policy", True
        )

        # Create ComponentManager
        component_manager_kwargs = {
            "config": config,
            "component_placement": component_placement,
            "use_pre_process_policy": self.use_pre_process_policy,
            "use_wait_before_last_iter_policy": self.use_wait_before_last_iter_policy,
            "_logger": self._logger,
            "channel_factory": self.create_channel,
        }
        self.component_managers: Dict[str, ComponentManager] = {}

        self.rollout_manager: ComponentManager = create_component_manager(
            "rollout", component_manager_kwargs
        )
        self.component_managers["rollout"] = self.rollout_manager

        self.actor_manager: ComponentManager = create_component_manager(
            "actor", component_manager_kwargs
        )
        self.component_managers["actor"] = self.actor_manager

        if "inference" in self.components:
            self.inference_manager: ComponentManager = create_component_manager(
                "inference", component_manager_kwargs
            )
            self.component_managers["inference"] = self.inference_manager

        # Set actor_valid_dp_sizes for rollout/actor manager.
        actor_valid_dp_sizes = get_valid_dp_sizes(
            self.cfg,
            self.component_placement._cluster_num_gpus,
            self.actor_manager.model_parallel_size,
        )
        self.actor_manager.actor_valid_dp_sizes = actor_valid_dp_sizes
        self.rollout_manager.actor_valid_dp_sizes = actor_valid_dp_sizes

    async def run(self):
        """Run the scheduler."""
        await self.pre_process()
        await self.main_loop()

    async def pre_process(self):
        """Pre process.

        Reset component manager states and execute use_pre_process_policy.
        Need to ensure rollout completes first.
        """
        await self.rollout_manager.pre_process()

        for component, manager in self.component_managers.items():
            if component != "rollout":
                await manager.pre_process()

    async def main_loop(self):
        """Main loop.

        The signal synchronization granularity is consistent with the actor training granularity.
        Trying to release or allocate gpu resource for each components by component_order.
        """
        available_gpu_num = 0
        for train_iter in range(self.cfg.algorithm.n_minibatches):
            # Wait for actor ready to update
            await self.actor_manager.wait_for_actor_update()

            # Trying to release or allocate resource for each components by component_order
            resource_info = f"[Release && Allocate Info] After train-iter{train_iter}\n"
            for component in self.component_order:
                if component not in self.component_managers:
                    self.log_warning(f"can't find ComponentManager for {component}")
                    continue

                release_or_allocate_params = {
                    "train_iter": train_iter,
                    "actor_current_instance_num": self.actor_manager.current_instance_num,
                    "rollout_current_instance_num": self.rollout_manager.current_instance_num,
                    "available_gpu_num": available_gpu_num,
                }
                released_gpu_num, incremental_gpu_num = await self.component_managers[
                    component
                ].release_or_allocate(**release_or_allocate_params)
                assert released_gpu_num == 0 or incremental_gpu_num == 0

                available_gpu_num += released_gpu_num - incremental_gpu_num
                resource_info += (
                    f"{component} : released_gpu_num = {released_gpu_num}, "
                    f"incremental_gpu_num={incremental_gpu_num} => "
                    f"available_gpu_num={available_gpu_num}\n"
                )

            self.log_info(resource_info)
