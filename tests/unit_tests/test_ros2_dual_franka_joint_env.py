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

from rlinf.envs.realworld.franka.tasks.ros2_dual_franka_joint_env import (
    Ros2DualFrankaJointEnv,
)


def test_ros2_joint_env_uses_placeholder_reward_for_manual_collection():
    env = object.__new__(Ros2DualFrankaJointEnv)

    assert env._calc_step_reward([True, True]) == 0.0
