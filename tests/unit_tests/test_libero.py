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

"""LIBERO observation preprocessing contracts without simulator dependencies."""

import numpy as np
import pytest

from rlinf.envs.sim.libero.utils import get_libero_image, get_libero_wrist_image


@pytest.mark.parametrize("extract_image", [get_libero_image, get_libero_wrist_image])
@pytest.mark.parametrize("memory_suite", [False, True])
def test_camera_orientation_matches_demonstrations(extract_image, memory_suite):
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    obs = {"agentview_image": image, "robot0_eye_in_hand_image": image}

    actual = extract_image(obs, memory_suite=memory_suite)

    expected = image[::-1] if memory_suite else image[::-1, ::-1]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(extract_image(obs), image[::-1, ::-1])
