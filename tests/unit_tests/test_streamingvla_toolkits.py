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

"""Tests for portable StreamingVLA data-preparation tools."""

import numpy as np

from toolkits.lerobot.convert_libero_data_to_lerobot import (
    _compute_action_states,
)


def test_converter_computes_start_of_step_cumulative_action_states():
    """Converted action states match the former parquet preprocessing rule."""
    actions = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [0.5, -1.0, 0.0, 1.0, 0.0, -2.0, 3.0],
            [-2.0, 1.0, 4.0, 0.0, 1.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    expected = np.stack(
        [np.zeros(7, dtype=np.float32), actions[0], actions[0] + actions[1]]
    )
    actual = _compute_action_states(actions)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_norm_stats_remove_strings_preserves_non_string_fields():
    """The filter removes strings without changing compatible object fields."""
    from toolkits.lerobot.calculate_streamingvla_norm_stats import RemoveStrings

    numeric = np.array([1.0, 2.0], dtype=np.float32)
    metadata = np.array([{"episode": 1}], dtype=object)
    output = RemoveStrings()(
        {"actions": numeric, "prompt": "move forward", "metadata": metadata}
    )

    assert set(output) == {"actions", "metadata"}
    assert output["actions"] is numeric
    assert output["metadata"] is metadata


def test_norm_stats_reuse_the_exact_actions_statistics_object():
    """Action states intentionally share all normalization values with actions."""
    from toolkits.lerobot.calculate_streamingvla_norm_stats import (
        _copy_action_statistics,
    )

    actions_stats = object()
    output = _copy_action_statistics({"state": object(), "actions": actions_stats})
    assert output["action_states"] is actions_stats
