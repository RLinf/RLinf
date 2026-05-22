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

import pytest
import torch

from rlinf.envs.robotwin.seed_utils import partition_success_seeds


def _first_eval_seeds(
    *,
    seed_count: int,
    total_num_envs: int,
    total_num_processes: int,
    group_size: int = 1,
    base_seed: int = 0,
) -> list[int]:
    success_seeds = torch.arange(seed_count, dtype=torch.long)
    selected_seeds = []
    for seed_offset in range(total_num_processes):
        num_envs = total_num_envs // total_num_processes
        num_group = num_envs // group_size
        worker_seeds = partition_success_seeds(
            success_seeds,
            base_seed=base_seed,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            num_group=num_group,
        )
        selected_seeds.extend(worker_seeds[:num_group].tolist())
    return selected_seeds


@pytest.mark.parametrize(
    ("seed_count", "total_num_envs", "total_num_processes"),
    [
        (320, 128, 4),
        (320, 128, 8),
        (320, 128, 16),
        (260, 128, 4),
        (200, 128, 8),
        (150, 128, 4),
    ],
)
def test_robotwin_eval_success_seeds_do_not_overlap_across_workers(
    seed_count: int,
    total_num_envs: int,
    total_num_processes: int,
):
    """Regression test for duplicate RoboTwin eval seeds across EnvWorkers."""
    selected_seeds = _first_eval_seeds(
        seed_count=seed_count,
        total_num_envs=total_num_envs,
        total_num_processes=total_num_processes,
    )

    assert len(selected_seeds) == total_num_envs
    assert len(set(selected_seeds)) == total_num_envs


def test_robotwin_eval_success_seed_order_is_controlled_by_base_seed():
    selected_seed_0 = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=0,
    )
    selected_seed_0_again = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=0,
    )
    selected_seed_1 = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=1,
    )

    assert selected_seed_0 == selected_seed_0_again
    assert selected_seed_0 != selected_seed_1
