# Copyright 2026 Ying-Chun Lee
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

"""Unit tests for the YAM follower home/zero-torque helpers."""

from unittest.mock import Mock

import numpy as np

from rlinf.envs.yam.yam_env import YAMEnv


class _FakeRobot:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float32)
        self.use_future = False
        self.future_history = []
        self.zero_torque_calls = 0

    def get_observations(self):
        return {
            "joint_pos": self.position[:6].copy(),
            "gripper_pos": self.position[6:].copy(),
        }

    def get_joint_pos(self):
        return self.position.copy()

    def command_joint_pos(self, target):
        self.position = np.asarray(target, dtype=np.float32)

    def set_use_future(self, value: bool):
        self.future_history.append(value)
        self.use_future = value

    def zero_torque_mode(self):
        self.zero_torque_calls += 1


class _FakeRobotEnv:
    def __init__(self, robots):
        self._robots = robots

    def get_all_robots(self):
        return self._robots

    def robot(self, name):
        return self._robots[name]


def _make_env():
    left_robot = _FakeRobot([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    right_robot = _FakeRobot([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4])
    env = object.__new__(YAMEnv)
    env._is_dummy = False
    env._robot_env = _FakeRobotEnv({"left": left_robot, "right": right_robot})
    env._move_to_reset_pose_on_reset = True
    env._reset_motion_steps = 3
    env._reset_motion_duration_s = 0.0
    env._reset_motion_tolerance = 1e-6
    env._reset_motion_settle_timeout_s = 0.0
    env._reset_motion_poll_interval_s = 0.0
    env._reset_joint_positions = {
        "left": np.full(7, 0.5, dtype=np.float32),
        "right": np.full(7, -0.25, dtype=np.float32),
    }
    env._logger = Mock()
    return env, left_robot, right_robot


def test_move_robots_to_reset_pose_reaches_targets_and_uses_future_mode():
    env, left_robot, right_robot = _make_env()

    env._move_robots_to_reset_pose()

    assert np.allclose(left_robot.position, env._reset_joint_positions["left"])
    assert np.allclose(right_robot.position, env._reset_joint_positions["right"])
    assert left_robot.future_history
    assert right_robot.future_history
    env._logger.info.assert_called_with(
        "[YAMEnv] Startup home pose reached for robots: ['left', 'right']."
    )


def test_enter_zero_torque_mode_calls_all_connected_robots():
    env, left_robot, right_robot = _make_env()

    env.enter_zero_torque_mode()

    assert left_robot.zero_torque_calls == 1
    assert right_robot.zero_torque_calls == 1
    env._logger.info.assert_called_with(
        "[YAMEnv] Entered zero-torque mode for robots: ['left', 'right']."
    )


def test_read_robot_joint_position_combines_joint_and_gripper_state():
    env, left_robot, _ = _make_env()

    joint_state = env._read_robot_joint_position("left")

    assert joint_state.shape == (7,)
    assert np.allclose(joint_state, left_robot.position)
