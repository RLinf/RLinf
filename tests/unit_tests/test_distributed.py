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

"""Distributed rollout redistribution contracts."""

import copy
from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.utils.distributed import balance_rollout_groups


def _balance_groups_process(rank, world_size, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        results = []
        for i in range(2):
            identity = rank * 2 + i
            length = 11 + identity * 7
            results.append(
                DynamicRolloutResult(
                    num_sequence=2,
                    group_size=2,
                    idx_to_traj=[0, 1],
                    input_ids=[[identity] * length, [identity] * (length + 1)],
                    prompt_lengths=[3, 3],
                    response_lengths=[length - 3, length - 2],
                    is_end=[True, True],
                    rewards=[float(identity), -float(identity)],
                    rollout_logprobs=[[-0.5] * (length - 3), [-0.25] * (length - 2)],
                    extra_fields_group={"identity": identity},
                    extra_fields_train={"idx_to_sub_traj": [0, 0]},
                )
            )
        snapshots = copy.deepcopy(results)
        output = balance_rollout_groups(results, dist.group.WORLD)
        assert len(output) == 2
        for result in output:
            identity = result.extra_fields_group["identity"]
            length = 11 + identity * 7
            assert result.idx_to_traj == [0, 1]
            assert result.input_ids == [[identity] * length, [identity] * (length + 1)]
            assert result.rewards == [float(identity), -float(identity)]
            assert result.rollout_logprobs == [
                [-0.5] * (length - 3),
                [-0.25] * (length - 2),
            ]
            assert result.extra_fields_train == {"idx_to_sub_traj": [0, 0]}
        assert results == snapshots
        identities = [None] * world_size
        dist.all_gather_object(
            identities, [res.extra_fields_group["identity"] for res in output]
        )
        assert sorted(sum(identities, [])) == list(range(2 * world_size))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_balance_complete_rollout_groups(tmp_path, world_size):
    mp.spawn(
        _balance_groups_process,
        args=(world_size, str(tmp_path / "rendezvous")),
        nprocs=world_size,
        join=True,
    )
