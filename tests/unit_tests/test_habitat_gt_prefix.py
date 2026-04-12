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

import torch

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.habitat.gt_prefix import (
    CMA_CURRENT_GT_ACTION_KEY,
    HABITAT_CURRENT_STEP_KEY,
    HABITAT_GT_ACTION_VALID_KEY,
    HABITAT_GT_CURRENT_ACTION_KEY,
    HABITAT_GT_PREFIX_LENGTH_KEY,
    build_gt_prefix_metadata,
    encode_episode_state_id,
    encode_episode_state_ids,
)


def test_habitat_gt_prefix_metadata_tracks_current_step_lookup():
    metadata = build_gt_prefix_metadata(
        episode_ids=[101, 202],
        elapsed_steps=[2, 5],
        episode_gt_action_sequences=[(1, 2, 3), (3, 2)],
        valid_action_ids={0, 1, 2, 3},
    )

    assert torch.equal(
        metadata[HABITAT_CURRENT_STEP_KEY], torch.tensor([3, 6], dtype=torch.int64)
    )
    assert torch.equal(
        metadata[HABITAT_GT_CURRENT_ACTION_KEY],
        torch.tensor([3, 0], dtype=torch.int64),
    )
    assert torch.equal(
        metadata[CMA_CURRENT_GT_ACTION_KEY], torch.tensor([3, 0], dtype=torch.int64)
    )
    assert torch.equal(
        metadata[HABITAT_GT_ACTION_VALID_KEY], torch.tensor([True, False])
    )
    assert torch.equal(
        metadata[HABITAT_GT_PREFIX_LENGTH_KEY],
        torch.tensor([3, 2], dtype=torch.int64),
    )


def test_habitat_gt_prefix_metadata_exposes_first_decision_gt_at_episode_start():
    metadata = build_gt_prefix_metadata(
        episode_ids=["episode-001"],
        elapsed_steps=[0],
        episode_gt_action_sequences=[(1, 2, 3)],
        valid_action_ids={0, 1, 2, 3},
    )

    assert metadata[HABITAT_CURRENT_STEP_KEY].item() == 1
    assert metadata[CMA_CURRENT_GT_ACTION_KEY].item() == 1
    assert metadata[HABITAT_GT_CURRENT_ACTION_KEY].item() == 1
    assert metadata[HABITAT_GT_ACTION_VALID_KEY].item() is True
    assert metadata[HABITAT_GT_PREFIX_LENGTH_KEY].item() == 1


def test_habitat_gt_prefix_metadata_missing_sequence_uses_zero_prefix():
    metadata = build_gt_prefix_metadata(
        episode_ids=[303],
        elapsed_steps=[0],
        episode_gt_action_sequences=[None],
        valid_action_ids={0, 1, 2, 3},
    )

    assert metadata[HABITAT_CURRENT_STEP_KEY].item() == 1
    assert metadata[HABITAT_GT_CURRENT_ACTION_KEY].item() == 0
    assert metadata[CMA_CURRENT_GT_ACTION_KEY].item() == 0
    assert metadata[HABITAT_GT_PREFIX_LENGTH_KEY].item() == 0
    assert metadata[HABITAT_GT_ACTION_VALID_KEY].item() is False


def test_habitat_gt_prefix_metadata_rejects_invalid_gt_action_ids():
    try:
        build_gt_prefix_metadata(
            episode_ids=[404],
            elapsed_steps=[1],
            episode_gt_action_sequences=[(9,)],
            valid_action_ids={0, 1, 2, 3},
        )
    except ValueError as exc:
        assert "Invalid Habitat GT action id 9" in str(exc)
    else:
        raise AssertionError(
            "Expected invalid Habitat GT action id to raise ValueError"
        )


def test_habitat_episode_state_id_encoding_supports_string_ids_deterministically():
    first = encode_episode_state_id("episode-001")
    second = encode_episode_state_id("episode-001")
    third = encode_episode_state_id("episode-002")
    numeric = encode_episode_state_id(7)
    numeric_string = encode_episode_state_id("7")

    assert isinstance(first, int)
    assert first == second
    assert first != third
    assert numeric == 7
    assert numeric_string != numeric

    state_ids = encode_episode_state_ids(["episode-001", 7])
    assert state_ids.dtype == torch.int64
    assert state_ids.tolist()[0] == first
    assert state_ids.tolist()[1] == 7


def test_habitat_gt_prefix_env_output_preserves_metadata_for_cma_path():
    env_output = EnvOutput(
        obs={
            "main_images": torch.zeros(1, 4, 4, 3),
            "wrist_images": [[1, 2, 3]],
            "extra_view_images": torch.zeros(1, 1, 4, 4, 1),
            "states": encode_episode_state_ids(["episode-001"]),
            "task_descriptions": ["navigate"],
            HABITAT_CURRENT_STEP_KEY: torch.tensor([1], dtype=torch.int64),
            CMA_CURRENT_GT_ACTION_KEY: torch.tensor([2], dtype=torch.int64),
            HABITAT_GT_CURRENT_ACTION_KEY: torch.tensor([2], dtype=torch.int64),
            HABITAT_GT_ACTION_VALID_KEY: torch.tensor([True]),
            HABITAT_GT_PREFIX_LENGTH_KEY: torch.tensor([1], dtype=torch.int64),
        }
    )

    obs = env_output.to_dict()["obs"]

    assert obs[HABITAT_CURRENT_STEP_KEY].item() == 1
    assert obs[CMA_CURRENT_GT_ACTION_KEY].item() == 2
    assert obs[HABITAT_GT_CURRENT_ACTION_KEY].item() == 2
    assert obs[HABITAT_GT_ACTION_VALID_KEY].item() is True
    assert obs[HABITAT_GT_PREFIX_LENGTH_KEY].item() == 1
