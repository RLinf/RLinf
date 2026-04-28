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

from __future__ import annotations

import sys
import types

import numpy as np


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakePosCmd:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.gripper = 0.0
        self.mode1 = None
        self.mode2 = None


def _install_controller_import_stubs(monkeypatch):
    rospy = types.ModuleType("rospy")
    rospy.loginfo_throttle = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = object
    monkeypatch.setitem(sys.modules, "cv_bridge", cv_bridge)

    turtle2_basic = types.ModuleType("turtle2_basic")
    turtle2_controller_pkg = types.ModuleType("turtle2_basic.turtle2_controller")
    turtle2_controller_mod = types.ModuleType(
        "turtle2_basic.turtle2_controller.Turtle2Controller"
    )
    turtle2_controller_mod.Turtle2Controller = object
    monkeypatch.setitem(sys.modules, "turtle2_basic", turtle2_basic)
    monkeypatch.setitem(
        sys.modules, "turtle2_basic.turtle2_controller", turtle2_controller_pkg
    )
    monkeypatch.setitem(
        sys.modules,
        "turtle2_basic.turtle2_controller.Turtle2Controller",
        turtle2_controller_mod,
    )


def _load_controller_class(monkeypatch):
    _install_controller_import_stubs(monkeypatch)
    sys.modules.pop("rlinf.envs.realworld.xsquare.turtle2_smooth_controller", None)
    from rlinf.envs.realworld.xsquare.turtle2_smooth_controller import (
        Turtle2SmoothController,
    )

    return Turtle2SmoothController


def test_turtle2_robot_config_default_backend_is_smooth():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from rlinf.envs.realworld.xsquare.turtle2_env import Turtle2RobotConfig

    assert Turtle2RobotConfig().pose_control_backend == "smooth"


def test_turtle2_absolute_pose_step_direct_accepts_raw_action():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from rlinf.envs.realworld.xsquare.tasks.deploy_env import Turtle2DeployEnv

    env = Turtle2DeployEnv(
        {
            "is_dummy": True,
            "pose_control_backend": "direct",
            "use_arm_ids": [0, 1],
            "use_camera_ids": [2],
            "enforce_gripper_close": False,
            "ee_pose_limit_min": [[-1.0] * 6, [-1.0] * 6],
            "ee_pose_limit_max": [[1.0] * 6, [1.0] * 6],
            "gripper_width_limit_min": 0.0,
            "gripper_width_limit_max": 5.0,
        }
    )

    action = np.array([0.1] * 6 + [0.7] + [-0.1] * 6 + [0.8], dtype=np.float32)
    _, _, _, _, info = env.step_absolute_pose(action)

    assert info["pose_control_backend"] == "direct"
    assert info["action_rejected"] is False
    assert info["rejection_reason"] is None
    assert info["action_clipped"] is False
    np.testing.assert_array_equal(info["raw_action"], action)
    np.testing.assert_array_equal(info["executed_action"], action)
    np.testing.assert_array_equal(info["last_published_action"], action)
    np.testing.assert_array_equal(env._turtle2_state.follow1_pos, action[:7])
    np.testing.assert_array_equal(env._turtle2_state.follow2_pos, action[7:])


def test_turtle2_absolute_pose_step_direct_rejects_out_of_bounds_without_publish():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from rlinf.envs.realworld.xsquare.tasks.deploy_env import Turtle2DeployEnv

    env = Turtle2DeployEnv(
        {
            "is_dummy": True,
            "pose_control_backend": "direct",
            "use_arm_ids": [0, 1],
            "use_camera_ids": [2],
            "enforce_gripper_close": False,
            "ee_pose_limit_min": [[-0.1] * 6, [-0.2] * 6],
            "ee_pose_limit_max": [[0.1] * 6, [0.2] * 6],
            "gripper_width_limit_min": 0.0,
            "gripper_width_limit_max": 1.0,
        }
    )
    before = np.stack(
        [env._turtle2_state.follow1_pos, env._turtle2_state.follow2_pos]
    ).reshape(-1)

    action = np.array([0.3] * 6 + [2.0] + [-0.3] * 6 + [-1.0], dtype=np.float32)
    _, _, _, _, info = env.step_absolute_pose(action)

    assert info["action_rejected"] is True
    assert info["rejection_reason"] == "outside_absolute_pose_action_space"
    assert info["action_clipped"] is False
    np.testing.assert_array_equal(info["raw_action"], action)
    np.testing.assert_array_equal(info["executed_action"], before)
    np.testing.assert_array_equal(info["last_published_action"], before)
    np.testing.assert_array_equal(env._turtle2_state.follow1_pos, before[:7])
    np.testing.assert_array_equal(env._turtle2_state.follow2_pos, before[7:])


def test_turtle2_absolute_pose_step_direct_rejects_bad_shape():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from rlinf.envs.realworld.xsquare.tasks.deploy_env import Turtle2DeployEnv

    env = Turtle2DeployEnv(
        {
            "is_dummy": True,
            "pose_control_backend": "direct",
            "use_arm_ids": [0, 1],
            "use_camera_ids": [2],
            "enforce_gripper_close": False,
        }
    )

    _, _, _, _, info = env.step_absolute_pose(np.ones(13, dtype=np.float32))

    assert info["action_rejected"] is True
    assert info["rejection_reason"] == "invalid_shape:(13,)"
    assert info["executed_action"].shape == (14,)


def test_turtle2_reset_next_step_bounds_large_pose_jumps():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from rlinf.envs.realworld.xsquare.turtle2_env import Turtle2Env, Turtle2RobotConfig

    env = Turtle2Env.__new__(Turtle2Env)
    env.config = Turtle2RobotConfig(
        reset_max_xyz_step=0.02,
        reset_max_rpy_step=0.075,
        reset_max_gripper_step=0.25,
    )
    current = np.array(
        [
            [0.0, 0.0, 0.0, -3.13, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -3.13, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [0.1, 0.0, 0.0, 3.13, 0.0, 0.0, 0.5],
            [0.0, 0.2, 0.0, 0.0, 3.13, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    previous = current
    for _ in range(20):
        waypoint = env._reset_next_step(previous, target)
        assert np.max(np.abs(waypoint[:, :3] - previous[:, :3])) <= 0.020001
        assert (
            np.max(
                np.abs(
                    Turtle2Env._shortest_angle_delta(
                        previous[:, 3:6],
                        waypoint[:, 3:6],
                    )
                )
            )
            <= 0.075001
        )
        assert np.max(np.abs(waypoint[:, 6] - previous[:, 6])) <= 0.250001
        previous = waypoint
        if env._reset_pose_reached(previous, target):
            break

    assert env._reset_pose_reached(previous, target)


def test_direct_controller_move_arm_publishes_pos_cmd(monkeypatch):
    controller_cls = _load_controller_class(monkeypatch)
    controller = controller_cls.__new__(controller_cls)
    controller.pose_control_backend = "direct"
    controller._direct_pose_msg_cls = FakePosCmd
    controller._direct_pose_left_pub = FakePublisher()
    controller._direct_pose_right_pub = FakePublisher()
    controller._direct_pose_left_target = None
    controller._direct_pose_right_target = None
    left = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    right = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]

    controller.move_arm(left, right)

    assert controller.left_arm_target == left
    assert controller.right_arm_target == right
    assert len(controller._direct_pose_left_pub.messages) == 1
    assert len(controller._direct_pose_right_pub.messages) == 1
    left_msg = controller._direct_pose_left_pub.messages[0]
    right_msg = controller._direct_pose_right_pub.messages[0]
    assert [
        left_msg.x,
        left_msg.y,
        left_msg.z,
        left_msg.roll,
        left_msg.pitch,
        left_msg.yaw,
        left_msg.gripper,
    ] == left
    assert [
        right_msg.x,
        right_msg.y,
        right_msg.z,
        right_msg.roll,
        right_msg.pitch,
        right_msg.yaw,
        right_msg.gripper,
    ] == right
    assert left_msg.mode1 == 0
    assert left_msg.mode2 == 0
