# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from rlinf.utils.dual_franka_actions import hold_right_arm_action
from rlinf.utils.rot6d import quat_xyzw_to_rot6d


@pytest.mark.parametrize(
    ("gripper_open", "expected_gripper"), [(True, 1.0), (False, -1.0)]
)
def test_hold_right_arm_action_preserves_left_and_holds_right(
    gripper_open: bool,
    expected_gripper: float,
):
    action = np.linspace(-0.9, 0.9, 20, dtype=np.float32)
    original_action = action.copy()
    quaternion = R.from_euler("xyz", [0.2, -0.3, 0.4]).as_quat()
    right_tcp_pose = np.concatenate([[0.51, 0.17, 0.32], quaternion])

    held = hold_right_arm_action(action, right_tcp_pose, gripper_open)

    np.testing.assert_array_equal(held[:10], original_action[:10])
    np.testing.assert_array_equal(action, original_action)
    np.testing.assert_allclose(held[10:13], right_tcp_pose[:3], atol=1e-6)
    np.testing.assert_allclose(held[13:19], quat_xyzw_to_rot6d(quaternion), atol=1e-6)
    assert held[19] == expected_gripper


@pytest.mark.parametrize(
    ("action", "tcp_pose"),
    [
        (np.zeros(19), np.zeros(7)),
        (np.zeros(20), np.zeros(6)),
    ],
)
def test_hold_right_arm_action_validates_shapes(action, tcp_pose):
    with pytest.raises(ValueError):
        hold_right_arm_action(action, tcp_pose, right_gripper_open=True)
