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

"""Task/trial dataset exposing embodied benchmark episodes to RolloutBuffer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from torch.utils.data import Dataset

from rlinf.models.embodiment.qwen_vlm.protocol import mirror_instruction


@dataclass
class EmbodiedTaskItem:
    """Dataset item shaped for ``rlinf.data.datasets.reasoning.collate_fn``."""

    prompt: list[int]
    length: int
    idx: int
    answer: str
    solution: Optional[str] = None
    image_data: Optional[list[Any]] = None
    prompt_text: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
    multi_modal_inputs: Optional[dict[str, Any]] = None


def build_libero_task_specs(
    task_suite_name: str,
    *,
    task_id_filter: Optional[list[int]] = None,
    specific_reset_id: Optional[int] = None,
    embodiment: str = "libero",
) -> list[dict[str, Any]]:
    """Enumerate ``(task_id, trial_id, reset_state_id)`` specs from a LIBERO suite."""
    from rlinf.envs.sim.libero.libero_env import _init_libero
    from rlinf.envs.sim.libero.utils import get_benchmark_overridden

    _init_libero()
    suite = get_benchmark_overridden(task_suite_name)()
    source_suites = getattr(suite, "task_suite_names", None)
    num_tasks = suite.get_num_tasks()
    allowed_tasks = (
        {int(tid) for tid in task_id_filter}
        if task_id_filter is not None
        else set(range(num_tasks))
    )
    for tid in allowed_tasks:
        if tid < 0 or tid >= num_tasks:
            raise ValueError(
                f"task_id {tid} in task_id_filter is out of range [0, {num_tasks - 1}]"
            )

    specs: list[dict[str, Any]] = []
    global_reset_id = 0
    for task_id in range(num_tasks):
        task = suite.get_task(task_id)
        num_trials = len(suite.get_task_init_states(task_id))
        source_suite = (
            str(source_suites[task_id])
            if source_suites is not None and task_id < len(source_suites)
            else str(task_suite_name)
        )
        instruction = mirror_instruction(str(task.language))
        for trial_id in range(num_trials):
            reset_state_id = global_reset_id
            global_reset_id += 1
            if task_id not in allowed_tasks:
                continue
            if specific_reset_id is not None and reset_state_id != int(
                specific_reset_id
            ):
                continue
            specs.append(
                {
                    "embodiment": embodiment,
                    "task_suite_name": str(task_suite_name),
                    "source_suite": source_suite,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "reset_state_id": reset_state_id,
                    "instruction": instruction,
                }
            )
    if not specs:
        raise ValueError(
            f"EmbodiedTaskDataset matched 0 episodes for suite={task_suite_name!r}, "
            f"task_id_filter={task_id_filter!r}, specific_reset_id={specific_reset_id!r}"
        )
    return specs


class EmbodiedTaskDataset(Dataset):
    """PyTorch Dataset of embodied episode specifications for RolloutBuffer.

    Each item encodes the episode specification (suite, task id, trial id,
    reset state id, instruction) as JSON in ``answer`` and carries a 1-token
    placeholder ``prompt`` so ``RolloutBufferWorker`` can issue, filter, and
    admit embodied groups identically to text agentic tasks.
    """

    def __init__(
        self,
        *,
        task_suite_name: str = "libero_gso",
        embodiment: str = "libero",
        task_id_filter: Optional[list[int]] = None,
        specific_reset_id: Optional[int] = None,
        specs: Optional[list[dict[str, Any]]] = None,
        min_size: int = 1,
    ) -> None:
        if specs is not None:
            if not specs:
                raise ValueError("specs must be non-empty when provided explicitly")
            base_specs = [dict(spec) for spec in specs]
        else:
            base_specs = build_libero_task_specs(
                task_suite_name,
                task_id_filter=task_id_filter,
                specific_reset_id=specific_reset_id,
                embodiment=embodiment,
            )

        self._specs: list[dict[str, Any]] = list(base_specs)
        # Ensure StatefulDataLoader(drop_last=True) always has at least one batch,
        # even for single-task smoke configs where len(base_specs) < batch_size.
        target_size = max(int(min_size), 1)
        while len(self._specs) < target_size:
            self._specs.extend(base_specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, idx: int) -> EmbodiedTaskItem:
        spec = self._specs[idx]
        return EmbodiedTaskItem(
            prompt=[0],
            length=1,
            idx=idx,
            answer=json.dumps(spec, sort_keys=True),
            meta=spec,
        )
