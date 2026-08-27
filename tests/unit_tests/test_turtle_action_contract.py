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

import numpy as np
import pytest

from rlinf.envs.action_utils import prepare_actions_for_realworld


def test_turtle_ppo_action_flatten_and_restore_contract() -> None:
    action_chunks = np.arange(2 * 50 * 6, dtype=np.float32).reshape(2, 50, 6)
    forward_input_action = action_chunks.reshape(2, -1)

    assert forward_input_action.shape == (2, 300)
    restored = prepare_actions_for_realworld(
        forward_input_action,
        model_type="openpi",
        num_action_chunks=50,
        action_dim=6,
    )
    assert restored.shape == (2, 50, 6)
    np.testing.assert_array_equal(restored, action_chunks)


def test_turtle_openpi_chunk_actions_are_preserved() -> None:
    action_chunks = np.zeros((4, 50, 6), dtype=np.float32)

    restored = prepare_actions_for_realworld(
        action_chunks,
        model_type="openpi",
        num_action_chunks=50,
        action_dim=6,
    )
    assert restored.shape == (4, 50, 6)


def test_turtle_openpi_rejects_invalid_action_shape() -> None:
    with pytest.raises(ValueError, match="trailing dimension 300"):
        prepare_actions_for_realworld(
            np.zeros((1, 299), dtype=np.float32),
            model_type="openpi",
            num_action_chunks=50,
            action_dim=6,
        )
