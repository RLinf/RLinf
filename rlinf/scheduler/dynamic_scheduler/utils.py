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

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from rlinf.data.io_struct import SeqGroupInfo
    from rlinf.scheduler.dynamic_scheduler.manager import ComponentManager


def get_valid_dp_sizes_with_hetero_dp(
    max_dp_size: int,
    global_batch_size: int,
    micro_batch_size: int = 1,
    alignment: int = 1,
) -> list[int]:
    """Get the valid data parallel sizes for the Actor based on the constraints of batch and group size when hetero-dp is applied."""
    all_dp_size_list = list(range(1, max_dp_size + 1))

    assert global_batch_size % (micro_batch_size * alignment) == 0
    num_microbatches_with_alignment_sum = (
        global_batch_size // micro_batch_size // alignment
    )

    valid_dp_sizes = [
        dp_size
        for dp_size in all_dp_size_list
        if num_microbatches_with_alignment_sum >= dp_size
    ]

    return valid_dp_sizes


def get_valid_dp_sizes(
    cfg, total_gpus: int, model_parallel_size_with_cp: int, with_hetero_dp: bool = True
) -> list[int]:
    """This function is used to get the valid data parallel sizes for the Actor based on the constraints of batch and group size.

    Args:
        cfg: The configuration object.
        total_gpus: The total number of GPUs.
        model_parallel_size_with_cp: The model parallel size with context parallel.
        with_hetero_dp: Whether to use hetero-dp.

    Returns:
        List[int]: The valid data parallel sizes for the component.
    """
    group_size = cfg.algorithm.group_size
    n_minibatches = cfg.algorithm.n_minibatches
    rollout_batch_size = cfg.data.rollout_batch_size

    global_step_batch_size = rollout_batch_size * group_size
    assert global_step_batch_size % n_minibatches == 0, (
        f"global_step_batch_size={global_step_batch_size} must be divisible by train_iter={n_minibatches}"
    )
    trainer_iter_batch_size = global_step_batch_size // n_minibatches
    max_dp_size = total_gpus // model_parallel_size_with_cp

    if with_hetero_dp:
        return [1, 2, 3, 4]
        # return [2, 4, 6, 8]
        # return [4, 8, 13, 16]
        # return [8, 16, 26, 32]
        # return [16, 32, 54, 64]
        return get_valid_dp_sizes_with_hetero_dp(
            max_dp_size=max_dp_size,
            global_batch_size=global_step_batch_size,
            alignment=group_size,
        )

    valid_dp_sizes = []

    for dp_size in range(1, max_dp_size + 1):
        if trainer_iter_batch_size % (dp_size * group_size) == 0:
            valid_dp_sizes.append(dp_size)

    return valid_dp_sizes


def get_scheduler_channel(component: str, instance_id: int = 0):
    """Get the scheduler channel name."""
    return f"dynamic_scheduler_channel_for_{component}_{instance_id}"


def get_scheduler_request_queue():
    """Get the scheduler request queue name."""
    return "dynamic_scheduler_request_queue"


def get_scheduler_response_queue():
    """Get the scheduler response queue name."""
    return "dynamic_scheduler_response_queue"


@dataclass
class RolloutReport:
    """Rollout report."""

    total_requests: int = None
    completed_requests: int = None
    total_tasks: int = None
    completed_tasks: int = None
    running_tasks: int = None
    timestamp: float = None


class RolloutAction(Enum):
    """Rollout action."""

    Default = auto()
    Report = auto()  # Check report
    Migrate_In = auto()  # Abort running tasks
    Migrate_Out = auto()  # Recevie running tasks
    Finish = auto()  # Finish running taks => offload weight
    Wait_For_Finish = auto()  # Block by rollout
    Offloaded = auto()  # Rollout offloaded signal


@dataclass
class RolloutScheduleInfo:
    """Rollout schedule info."""

    instance_id: int = -1
    data: list["SeqGroupInfo"] = None
    report: RolloutReport = None
    action: RolloutAction = RolloutAction.Default


class _DynamicSchedulerState:
    """GPU resource state and components' state."""

    def __init__(
        self,
        cfg: DictConfig,
        total_gpus: int,
        component_managers: dict[str, "ComponentManager"],
    ):
        """Initialize the dynamic scheduler state."""
        self.total_gpus = total_gpus
        self.available_gpu_num = 0
        self.component_managers = component_managers

        self.components_instance_num: dict[str, int] = {}
        self.components_model_parallel_size: dict[str, int] = {}
        for component, manager in self.component_managers.items():
            self.components_instance_num[component] = manager.current_instance_num
            self.components_model_parallel_size[component] = manager.model_parallel_size

        self.actor_valid_dp_sizes: list[int] = get_valid_dp_sizes(
            cfg, total_gpus, self.components_model_parallel_size["actor"]
        )

    def reset(self):
        """Reset state."""
        self.available_gpu_num = 0
        for component, manager in self.component_managers.items():
            self.components_instance_num[component] = manager.current_instance_num

    def update(self, component: str, released_gpu_num: int, incremental_gpu_num: int):
        """Update current state."""
        assert released_gpu_num == 0 or incremental_gpu_num == 0
        self.available_gpu_num += released_gpu_num - incremental_gpu_num
        self.components_instance_num[component] = self.component_managers[
            component
        ].current_instance_num

    def get_component_instance_num(self, component: str):
        return self.components_instance_num[component]

    def get_component_model_parallel_size(self, component: str):
        return self.components_model_parallel_size[component]


DynamicSchedulerState = None


def set_global_scheduer_state(
    cfg: DictConfig, total_gpus: int, component_managers: dict[str, "ComponentManager"]
):
    """Set DynamicSchedulerState."""
    global DynamicSchedulerState
    DynamicSchedulerState = _DynamicSchedulerState(cfg, total_gpus, component_managers)


def get_global_scheduer_state() -> _DynamicSchedulerState:
    """Get DynamicSchedulerState."""
    global DynamicSchedulerState
    assert DynamicSchedulerState is not None
    return DynamicSchedulerState
