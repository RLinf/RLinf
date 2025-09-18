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
from typing import List


def get_valid_dp_sizes(cfg, model_parallel_size_with_cp) -> List[int]:
    """This function is used to get the valid data parallel sizes for the Actor based on the constraints of batch and group size.

    Returns:
        List[int]: The valid data parallel sizes for the component.
    """
    total_gpus = cfg.cluster.num_gpus_per_node * cfg.cluster.num_nodes
    group_size = cfg.algorithm.group_size
    n_minibatches = cfg.algorithm.n_minibatches
    rollout_batch_size = cfg.data.rollout_batch_size

    global_step_batch_size = rollout_batch_size * group_size
    assert global_step_batch_size % n_minibatches == 0, (
        f"global_step_batch_size={global_step_batch_size} must be divisible by train_iter={n_minibatches}"
    )
    trainer_iter_batch_size = global_step_batch_size // n_minibatches

    valid_dp_sizes = []

    max_dp_size = total_gpus // model_parallel_size_with_cp

    for dp_size in range(1, max_dp_size + 1):
        if trainer_iter_batch_size % (dp_size * group_size) == 0:
            valid_dp_sizes.append(dp_size)

    return valid_dp_sizes


def get_scheduler_channel(component: str):
    """Get the scheduler channel name."""
    return f"schedule_channel_{component}"


def get_scheduler_request_queue(instance_id: int = 0):
    """Get the scheduler request queue name."""
    return f"schedule_request_{instance_id}"


def get_scheduler_response_queue(instance_id: int = 0):
    """Get the scheduler response queue name."""
    return f"schedule_response_{instance_id}"


# Rollout Communication Related
@dataclass
class RolloutMigrateBatch:
    """Rollout migrate batch."""

    input_ids: list = None
    results: list = None
    abort_results: list = None
    answers: list = None

    def get_running_tasks(self) -> int:
        """Get the number of running tasks."""
        return len(self.abort_results)


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
    data: List[RolloutMigrateBatch] = None
    report: RolloutReport = None
    action: RolloutAction = RolloutAction.Default
