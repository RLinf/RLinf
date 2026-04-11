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
    ComponentManager,
    create_component_manager,
)
from rlinf.scheduler.dynamic_scheduler.utils import (
    get_global_scheduer_state,
    set_global_scheduer_state,
)
from rlinf.utils.placement import ComponentPlacement


class SchedulerWorker(Worker):
    """Dynamic Scheduler."""

    def __init__(
        self,
        config: DictConfig,
        component_placement: ComponentPlacement,
        workflow: list[str] = ["rollout", "inference", "actor"],
    ):
        """Initialize the SchedulerWorker."""
        super().__init__()
        self.cfg = config
        self.component_placement = component_placement
        self.components = self.component_placement.components
        self.workflow = workflow

        assert self.cfg.rollout.rollout_backend in ["sglang", "vllm"], (
            "only sglang and vllm are supported for dynamic scheduler"
        )
        assert self.cfg.actor.training_backend == "megatron", (
            "only megatron is supported for dynamic scheduler"
        )
        assert "rollout" in self.components, "rollout component is required"
        assert "actor" in self.components, "actor component is required"

        # Set policies for dynamic-scheduler
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
        self.component_managers: dict[str, ComponentManager] = {}
        for component in self.components:
            if component == "reward":
                continue
            self.component_managers[component] = create_component_manager(
                component, component_manager_kwargs
            )

        set_global_scheduer_state(
            self.cfg,
            self.component_placement._cluster_num_gpus,
            self.component_managers,
        )
        self.scheduler_state = get_global_scheduer_state()

    async def schedule(self):
        """Run the scheduler."""
        await self.pre_process()
        await self.main_loop()

    async def pre_process(self):
        """Reset component manager states and execute pre_process policy."""
        await self.component_managers["rollout"].pre_process()

        for component, manager in self.component_managers.items():
            if component != "rollout":
                await manager.pre_process()

        self.scheduler_state.reset()

    async def main_loop(self):
        """Main loop. Trying to release or allocate gpu resource for each components by workflow after actor ready to update."""
        for train_iter in range(self.cfg.algorithm.n_minibatches):
            # Wait for actor ready to update
            await self.component_managers["actor"].wait_for_actor_update()

            # Trying to release or allocate resource for each components by workflow
            resource_info = f"[Release && Allocate Info] After train-iter{train_iter}\n"
            for component in self.workflow:
                if component not in self.component_managers:
                    self.log_warning(f"can't find ComponentManager for {component}")
                    continue

                released_gpu_num, incremental_gpu_num = await self.component_managers[
                    component
                ].release_or_allocate(train_iter)

                self.scheduler_state.update(
                    component, released_gpu_num, incremental_gpu_num
                )

                resource_info += (
                    f"{component} : released_gpu_num = {released_gpu_num}, "
                    f"incremental_gpu_num={incremental_gpu_num} => "
                    f"available_gpu_num={self.scheduler_state.available_gpu_num}\n"
                )

            self.log_info(resource_info)
