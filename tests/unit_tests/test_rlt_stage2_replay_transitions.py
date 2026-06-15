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

from types import SimpleNamespace

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.rlt_stage2.replay_buffer import RLTStage2ReplayBuffer
from rlinf.models.embodiment.rlt_stage2.trajectory_adapter import (
    RLTStage2TrajectoryReplayAdapter,
)
from rlinf.models.embodiment.rlt_stage2.transition import TransitionSource


def _build_cfg(*, stride: int) -> SimpleNamespace:
    return SimpleNamespace(
        actor=SimpleNamespace(
            model=SimpleNamespace(
                num_action_chunks=2,
                action_dim=2,
                rlt_stage2={
                    "replay_subsample_stride": stride,
                    "replay_allow_terminal_partial": True,
                    "embedding_dim": 4,
                    "proprio_dim": 0,
                },
            ),
            seed=1234,
        ),
        algorithm=SimpleNamespace(get=lambda key, default=None: default),
        env=SimpleNamespace(train=SimpleNamespace(get=lambda key, default=None: False)),
        rollout=SimpleNamespace(pipeline_stage_num=1),
    )


def _build_test_adapter(
    *,
    stride: int,
) -> tuple[RLTStage2TrajectoryReplayAdapter, RLTStage2ReplayBuffer]:
    replay_buffer = RLTStage2ReplayBuffer(
        capacity=8,
        state_dim=4,
        action_chunk_dim=4,
        chunk_length=2,
        seed=1234,
    )
    adapter = RLTStage2TrajectoryReplayAdapter(
        cfg=_build_cfg(stride=stride),
        replay_buffer=replay_buffer,
    )
    return adapter, replay_buffer


def test_chunk_trajectory_to_transitions_uses_canonical_a_tilde():
    adapter, replay_buffer = _build_test_adapter(stride=0)

    traj = Trajectory(
        actions=torch.tensor(
            [[[1.0, 10.0, 2.0, 20.0]]],
            dtype=torch.float32,
        ),
        rewards=torch.tensor(
            [[[0.1, 0.2]]],
            dtype=torch.float32,
        ),
        dones=torch.tensor(
            [[[False]], [[True]]],
            dtype=torch.bool,
        ),
        forward_inputs={
            "x": torch.tensor(
                [
                    [[1.0, 2.0, 3.0, 4.0]],
                    [[9.0, 8.0, 7.0, 6.0]],
                ],
                dtype=torch.float32,
            ),
            "a_tilde": torch.tensor(
                [[[101.0, 1001.0, 202.0, 2002.0]]],
                dtype=torch.float32,
            ),
        },
    )

    added, completed = adapter.add_trajectory(traj)

    assert added == 1
    assert completed == 1
    np.testing.assert_allclose(
        replay_buffer._ref_chunk[0],
        np.array([101.0, 1001.0, 202.0, 2002.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        replay_buffer._next_ref_chunk[0],
        np.array([101.0, 1001.0, 202.0, 2002.0], dtype=np.float32),
    )


def test_step_trace_to_transitions_keeps_reference_chunks_for_human_windows():
    adapter, replay_buffer = _build_test_adapter(stride=1)

    traj = Trajectory(
        actions=torch.tensor(
            [
                [[1.0, 10.0, 2.0, 20.0]],
                [[3.0, 30.0, 4.0, 40.0]],
            ],
            dtype=torch.float32,
        ),
        rewards=torch.tensor(
            [
                [[0.1, 0.2]],
                [[0.3, 0.4]],
            ],
            dtype=torch.float32,
        ),
        dones=torch.tensor(
            [
                [[False, False]],
                [[False, False]],
                [[False, True]],
            ],
            dtype=torch.bool,
        ),
        intervene_flags=torch.tensor(
            [
                [[True, True, False, False]],
                [[False, False, False, False]],
            ],
            dtype=torch.bool,
        ),
        forward_inputs={
            "x": torch.tensor(
                [
                    [[10.0, 11.0, 12.0, 13.0]],
                    [[20.0, 21.0, 22.0, 23.0]],
                    [[30.0, 31.0, 32.0, 33.0]],
                ],
                dtype=torch.float32,
            ),
            "a_tilde": torch.tensor(
                [
                    [[101.0, 1001.0, 202.0, 2002.0]],
                    [[303.0, 3003.0, 404.0, 4004.0]],
                    [[505.0, 5005.0, 606.0, 6006.0]],
                ],
                dtype=torch.float32,
            ),
            "source_chunk": torch.tensor(
                [
                    [[int(TransitionSource.HUMAN), int(TransitionSource.RL)]],
                    [[int(TransitionSource.HUMAN), int(TransitionSource.RL)]],
                ],
                dtype=torch.uint8,
            ),
        },
        rlt_step_trace={
            "anchor_offsets": torch.tensor(
                [
                    [[1]],
                    [[1]],
                ],
                dtype=torch.int64,
            ),
            "x": torch.tensor(
                [
                    [[[14.0, 15.0, 16.0, 17.0]]],
                    [[[24.0, 25.0, 26.0, 27.0]]],
                ],
                dtype=torch.float32,
            ),
            "a_tilde": torch.tensor(
                [
                    [[[111.0, 1111.0, 222.0, 2222.0]]],
                    [[[333.0, 3333.0, 444.0, 4444.0]]],
                ],
                dtype=torch.float32,
            ),
        },
    )

    added, completed = adapter.add_trajectory(traj)

    assert added == 3
    assert completed == 1

    np.testing.assert_allclose(
        replay_buffer._ref_chunk[0],
        np.array([101.0, 1001.0, 202.0, 2002.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        replay_buffer._next_ref_chunk[0],
        np.array([303.0, 3003.0, 404.0, 4004.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        replay_buffer._ref_chunk[1],
        np.array([111.0, 1111.0, 222.0, 2222.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        replay_buffer._ref_chunk[2],
        np.array([303.0, 3003.0, 404.0, 4004.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        replay_buffer._next_ref_chunk[2],
        np.array([505.0, 5005.0, 606.0, 6006.0], dtype=np.float32),
    )
