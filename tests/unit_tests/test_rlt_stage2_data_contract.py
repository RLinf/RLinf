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

import pytest
import torch

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
)
from rlinf.models.embodiment.rlt_stage2.rollout_result_adapter import (
    update_last_rlt_action_metadata,
)


def _make_obs(start: int, batch_size: int) -> dict:
    return {
        "states": torch.arange(start, start + batch_size * 2, dtype=torch.float32).view(
            batch_size,
            2,
        ),
        "main_images": None,
        "wrist_images": None,
        "extra_view_images": None,
        "task_descriptions": [
            f"task-{idx}" for idx in range(start, start + batch_size)
        ],
    }


def test_merge_env_outputs_supports_step_obs_and_policy_info():
    env_output_0 = EnvOutput(
        obs=_make_obs(0, 2),
        step_obs={
            "states": torch.arange(12, dtype=torch.float32).view(3, 2, 2),
            "_rlt_step_offsets": torch.tensor(
                [[0, 0], [2, 2], [4, 4]], dtype=torch.int64
            ),
        },
        policy_info={
            "expert_takeover": torch.tensor([[0], [1]], dtype=torch.bool),
        },
    ).to_dict()
    env_output_1 = EnvOutput(
        obs=_make_obs(100, 3),
        step_obs={
            "states": torch.arange(18, dtype=torch.float32).view(3, 3, 2),
            "_rlt_step_offsets": torch.tensor(
                [[0, 0, 0], [2, 2, 2], [4, 4, 4]], dtype=torch.int64
            ),
        },
        policy_info={
            "expert_takeover": torch.tensor([[1], [0], [1]], dtype=torch.bool),
        },
    ).to_dict()

    merged = EnvOutput.merge_env_outputs([env_output_0, env_output_1])

    assert merged["step_obs"] is not None
    assert merged["step_obs"]["states"].shape == (3, 5, 2)
    assert merged["step_obs"]["_rlt_step_offsets"].shape == (3, 5)
    assert torch.equal(
        merged["step_obs"]["states"][:, :2],
        env_output_0["step_obs"]["states"],
    )
    assert torch.equal(
        merged["step_obs"]["states"][:, 2:],
        env_output_1["step_obs"]["states"],
    )

    assert merged["policy_info"] is not None
    assert merged["policy_info"]["expert_takeover"].shape == (5, 1)
    assert torch.equal(
        merged["policy_info"]["expert_takeover"][:2],
        env_output_0["policy_info"]["expert_takeover"],
    )
    assert torch.equal(
        merged["policy_info"]["expert_takeover"][2:],
        env_output_1["policy_info"]["expert_takeover"],
    )


def test_merge_env_outputs_rejects_partial_policy_info():
    env_output_0 = EnvOutput(
        obs=_make_obs(0, 2),
        policy_info={
            "expert_takeover": torch.tensor([[0], [1]], dtype=torch.bool),
        },
    ).to_dict()
    env_output_1 = EnvOutput(
        obs=_make_obs(100, 3),
        policy_info=None,
    ).to_dict()

    with pytest.raises(ValueError, match="Inconsistent policy_info"):
        EnvOutput.merge_env_outputs([env_output_0, env_output_1])


def test_merge_env_outputs_rejects_inconsistent_policy_info_keys():
    env_output_0 = EnvOutput(
        obs=_make_obs(0, 2),
        policy_info={
            "expert_takeover": torch.tensor([[0], [1]], dtype=torch.bool),
            "record_transition": torch.tensor([[1], [1]], dtype=torch.bool),
        },
    ).to_dict()
    env_output_1 = EnvOutput(
        obs=_make_obs(100, 3),
        policy_info={
            "expert_takeover": torch.tensor([[1], [0], [1]], dtype=torch.bool),
        },
    ).to_dict()

    with pytest.raises(ValueError, match="Inconsistent policy_info keys"):
        EnvOutput.merge_env_outputs([env_output_0, env_output_1])


def test_update_last_actions_updates_actions_without_rewriting_rlt_refs():
    rollout_result = EmbodiedRolloutResult()
    rollout_result.append_step_result(
        ChunkStepResult(
            actions=torch.tensor([[1.0, 10.0, 2.0, 20.0]], dtype=torch.float32),
            forward_inputs={
                "action": torch.tensor([[1.0, 10.0, 2.0, 20.0]], dtype=torch.float32),
                "action_chunk": torch.tensor(
                    [[1.0, 10.0, 2.0, 20.0]], dtype=torch.float32
                ),
                "ref_chunk": torch.tensor(
                    [[11.0, 110.0, 22.0, 220.0]], dtype=torch.float32
                ),
                "next_ref_chunk": torch.tensor(
                    [[33.0, 330.0, 44.0, 440.0]], dtype=torch.float32
                ),
                "source_chunk": torch.tensor([[1, 0]], dtype=torch.uint8),
                "source": torch.tensor([[3]], dtype=torch.uint8),
                "intervention_flags": torch.tensor([[False, False]], dtype=torch.bool),
                "model_action": torch.tensor(
                    [[101.0, 102.0, 103.0, 104.0]], dtype=torch.float32
                ),
            },
        )
    )

    intervene_actions = torch.tensor(
        [[5.0, 50.0, 6.0, 60.0]],
        dtype=torch.float32,
    )
    intervene_flags = torch.tensor([[True, False]], dtype=torch.bool)

    rollout_result.update_last_actions(intervene_actions, intervene_flags)

    assert torch.equal(
        rollout_result.actions[-1],
        torch.tensor([[5.0, 50.0, 2.0, 20.0]], dtype=torch.float32),
    )
    assert torch.equal(
        rollout_result.intervene_flags[-1],
        torch.tensor([[True, True, False, False]], dtype=torch.bool),
    )

    last_forward_inputs = rollout_result.forward_inputs[-1]
    assert torch.equal(
        last_forward_inputs["action"],
        torch.tensor([[5.0, 50.0, 2.0, 20.0]], dtype=torch.float32),
    )
    assert torch.equal(
        last_forward_inputs["action_chunk"],
        torch.tensor([[5.0, 50.0, 2.0, 20.0]], dtype=torch.float32),
    )
    assert torch.equal(
        last_forward_inputs["ref_chunk"],
        torch.tensor([[11.0, 110.0, 22.0, 220.0]], dtype=torch.float32),
    )
    assert torch.equal(
        last_forward_inputs["next_ref_chunk"],
        torch.tensor([[33.0, 330.0, 44.0, 440.0]], dtype=torch.float32),
    )
    assert torch.equal(
        last_forward_inputs["source_chunk"],
        torch.tensor([[1, 0]], dtype=torch.uint8),
    )
    assert torch.equal(
        last_forward_inputs["source"],
        torch.tensor([[3]], dtype=torch.uint8),
    )
    assert torch.equal(
        last_forward_inputs["intervention_flags"],
        torch.tensor([[True, False]], dtype=torch.bool),
    )
    assert "model_action" in last_forward_inputs

    update_last_rlt_action_metadata(rollout_result, intervene_flags)

    assert torch.equal(
        last_forward_inputs["source_chunk"],
        torch.tensor([[2, 0]], dtype=torch.uint8),
    )
    assert torch.equal(
        last_forward_inputs["source"],
        torch.tensor([[3]], dtype=torch.uint8),
    )
    assert torch.equal(
        last_forward_inputs["intervention_flag"],
        torch.tensor([[True]], dtype=torch.bool),
    )
    assert "model_action" not in last_forward_inputs


def test_rlt_metadata_update_marks_uniform_human_source_as_human():
    rollout_result = EmbodiedRolloutResult()
    rollout_result.append_step_result(
        ChunkStepResult(
            actions=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
            forward_inputs={
                "source_chunk": torch.tensor([[0, 0]], dtype=torch.uint8),
                "source": torch.tensor([[0]], dtype=torch.uint8),
            },
        )
    )

    rollout_result.update_last_actions(
        intervene_actions=torch.tensor([[9.0, 9.0, 8.0, 8.0]], dtype=torch.float32),
        intervene_flags=torch.tensor([[True, True]], dtype=torch.bool),
    )
    update_last_rlt_action_metadata(
        rollout_result,
        intervene_flags=torch.tensor([[True, True]], dtype=torch.bool),
    )

    last_forward_inputs = rollout_result.forward_inputs[-1]
    assert torch.equal(
        last_forward_inputs["source_chunk"],
        torch.tensor([[2, 2]], dtype=torch.uint8),
    )
    assert torch.equal(
        last_forward_inputs["source"],
        torch.tensor([[2]], dtype=torch.uint8),
    )


def test_split_trajectories_by_sizes_splits_rlt_step_trace_on_batch_dim():
    rollout_result = EmbodiedRolloutResult()
    rollout_result.append_step_result(
        ChunkStepResult(
            actions=torch.zeros((3, 4), dtype=torch.float32),
            rlt_step_trace={
                "anchor_offsets": torch.tensor(
                    [
                        [10, 11, 12],
                        [20, 21, 22],
                    ],
                    dtype=torch.int64,
                ),
                "x": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
                "a_tilde": torch.arange(24, 48, dtype=torch.float32).reshape(2, 3, 4),
            },
        )
    )

    trajectories = rollout_result.to_splited_trajectories_by_sizes([1, 2])

    assert trajectories[0].rlt_step_trace["anchor_offsets"].shape == (1, 2, 1)
    assert trajectories[1].rlt_step_trace["anchor_offsets"].shape == (1, 2, 2)
    assert torch.equal(
        trajectories[0].rlt_step_trace["anchor_offsets"],
        torch.tensor([[[10], [20]]], dtype=torch.int64),
    )
    assert torch.equal(
        trajectories[1].rlt_step_trace["anchor_offsets"],
        torch.tensor([[[11, 12], [21, 22]]], dtype=torch.int64),
    )
    assert torch.equal(
        trajectories[0].rlt_step_trace["x"],
        rollout_result.to_trajectory().rlt_step_trace["x"][:, :, :1],
    )
    assert torch.equal(
        trajectories[1].rlt_step_trace["a_tilde"],
        rollout_result.to_trajectory().rlt_step_trace["a_tilde"][:, :, 1:],
    )
