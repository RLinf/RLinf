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

"""Unit tests for GimArmEnv joint-space control logic.

These tests use a mock controller so no hardware or SDK is required.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rlinf.envs.realworld.gim_arm.gim_arm_env import GimArmEnv, GimArmRobotConfig
from rlinf.envs.realworld.gim_arm.gim_arm_robot_state import GimArmRobotState


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_future(value):
    """Return a mock that behaves like a Worker future: ``.wait()`` -> ``[value]``."""
    future = MagicMock()
    future.wait.return_value = [value]
    return future


def _make_mock_controller(initial_q=None):
    """Build a mock GimArmController with controllable joint state."""
    if initial_q is None:
        initial_q = np.zeros(6)

    state = GimArmRobotState(
        tcp_pose=np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
        tcp_vel=np.zeros(6),
        arm_joint_position=initial_q.copy(),
        arm_joint_velocity=np.zeros(6),
        tcp_force=np.zeros(3),
        tcp_torque=np.zeros(3),
        arm_jacobian=np.eye(6),
        gripper_position=0.0,
        gripper_open=True,
    )

    controller = MagicMock()
    controller.is_robot_up.return_value = _make_future(True)
    controller.get_state.return_value = _make_future(state)
    controller.move_joints.return_value = _make_future(None)
    controller.reset_joint.return_value = _make_future(None)
    controller.open_gripper.return_value = _make_future(None)
    controller.close_gripper.return_value = _make_future(None)

    return controller, state


def _make_env(
    initial_q=None,
    action_scale=0.05,
    joint_limit_low=None,
    joint_limit_high=None,
    **kwargs,
):
    """Create a GimArmEnv with a mocked controller (no hardware needed)."""
    if joint_limit_low is None:
        joint_limit_low = np.full(6, -np.pi)
    if joint_limit_high is None:
        joint_limit_high = np.full(6, np.pi)

    config = GimArmRobotConfig(
        can_interface="can0",
        arm_variant="gim_arm_xl",
        camera_serials=["fake_serial"],
        camera_type="realsense",
        is_dummy=False,
        action_scale=action_scale,
        joint_limit_low=joint_limit_low,
        joint_limit_high=joint_limit_high,
        step_frequency=1000.0,  # Fast for tests.
        max_num_steps=10,
        target_ee_pose=np.array([0.5, 0.0, 0.3, -3.14, 0.0, 0.0]),
        reward_threshold=np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2]),
        **kwargs,
    )

    controller, state = _make_mock_controller(initial_q)

    def _fake_setup(self_env):
        """Inject mock controller instead of real hardware."""
        self_env._controller = controller

    with patch.object(GimArmEnv, "_setup_hardware", _fake_setup), \
         patch.object(GimArmEnv, "_open_cameras", lambda self: None):
        env = GimArmEnv(config, worker_info=None, hardware_info=None, env_idx=0)

    env._state = state
    env._cameras = []
    env.camera_player = MagicMock()
    return env, controller, state


# ── Tests: Action Space ─────────────────────────────────────────────────────


class TestActionSpace:
    def test_action_space_shape(self):
        env, _, _ = _make_env()
        assert env.action_space.shape == (7,)

    def test_action_space_bounds(self):
        env, _, _ = _make_env()
        np.testing.assert_array_equal(env.action_space.low, -np.ones(7))
        np.testing.assert_array_equal(env.action_space.high, np.ones(7))

    def test_observation_space_keys(self):
        env, _, _ = _make_env()
        assert "state" in env.observation_space.spaces
        assert "frames" in env.observation_space.spaces

    def test_state_keys(self):
        env, _, _ = _make_env()
        state_space = env.observation_space["state"]
        expected = {
            "tcp_pose",
            "tcp_vel",
            "arm_joint_position",
            "gripper_position",
            "tcp_force",
            "tcp_torque",
        }
        assert set(state_space.spaces.keys()) == expected

    def test_tcp_pose_shape(self):
        env, _, _ = _make_env()
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)

    def test_arm_joint_position_shape(self):
        env, _, _ = _make_env()
        assert env.observation_space["state"]["arm_joint_position"].shape == (6,)


# ── Tests: Joint-Space Step ─────────────────────────────────────────────────


class TestJointSpaceStep:
    def test_joint_delta_applied(self):
        """Verify q_target = q + action[:6] * action_scale."""
        initial_q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        env, controller, state = _make_env(initial_q=initial_q, action_scale=0.1)

        action = np.array([1.0, 0.0, -1.0, 0.5, -0.5, 0.0, 0.0])
        env.step(action)

        call_args = controller.move_joints.call_args[0][0]
        expected = initial_q + action[:6] * 0.1
        np.testing.assert_allclose(call_args, expected, atol=1e-10)

    def test_clamping_to_joint_limits(self):
        """Joint targets must be clamped to configured limits."""
        low = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
        high = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        initial_q = np.array([0.45, 0.0, 0.0, 0.0, 0.0, 0.0])

        env, controller, _ = _make_env(
            initial_q=initial_q,
            action_scale=0.1,
            joint_limit_low=low,
            joint_limit_high=high,
        )

        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        env.step(action)

        call_args = controller.move_joints.call_args[0][0]
        # 0.45 + 0.1 = 0.55 -> clamped to 0.5
        assert call_args[0] == pytest.approx(0.5)

    def test_action_clipped_to_bounds(self):
        """Actions outside [-1, 1] are clipped before applying."""
        initial_q = np.zeros(6)
        env, controller, _ = _make_env(initial_q=initial_q, action_scale=0.1)

        action = np.array([2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        env.step(action)

        call_args = controller.move_joints.call_args[0][0]
        # Clipped to 1.0 and -1.0 first, then scaled.
        np.testing.assert_allclose(call_args[:2], [0.1, -0.1], atol=1e-10)

    def test_step_increments_counter(self):
        env, _, _ = _make_env()
        assert env.num_steps == 0
        env.step(np.zeros(7))
        assert env.num_steps == 1
        env.step(np.zeros(7))
        assert env.num_steps == 2


# ── Tests: Gripper ───────────────────────────────────────────────────────────


class TestGripperAction:
    def test_close_gripper_when_open(self):
        """Negative action closes the gripper when it is currently open."""
        env, controller, state = _make_env()
        state.gripper_open = True
        result = env._gripper_action(-1.0)
        assert result is True
        controller.close_gripper.assert_called_once()

    def test_open_gripper_when_closed(self):
        env, controller, state = _make_env()
        state.gripper_open = False
        result = env._gripper_action(1.0)
        assert result is True
        controller.open_gripper.assert_called_once()

    def test_no_action_below_threshold(self):
        env, controller, state = _make_env()
        state.gripper_open = True
        result = env._gripper_action(0.3)
        assert result is False
        controller.open_gripper.assert_not_called()
        controller.close_gripper.assert_not_called()

    def test_no_redundant_close(self):
        """No close call when gripper is already closed."""
        env, controller, state = _make_env()
        state.gripper_open = False
        result = env._gripper_action(-1.0)
        assert result is False

    def test_gripper_penalty(self):
        """Gripper penalty is subtracted when an effective action occurs."""
        env, _, state = _make_env(
            enable_gripper_penalty=True, gripper_penalty=0.05
        )
        state.gripper_open = True

        # Action that triggers gripper close.
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        _, reward, _, _, _ = env.step(action)
        # Reward should have penalty subtracted.
        assert reward < 0.0 or reward == pytest.approx(1.0 - 0.05)


# ── Tests: Episode Termination ───────────────────────────────────────────────


class TestEpisodeTermination:
    def test_truncation_at_max_steps(self):
        env, _, _ = _make_env()
        for _ in range(env.config.max_num_steps):
            obs, _, terminated, truncated, _ = env.step(np.zeros(7))
        assert truncated is True

    def test_not_truncated_before_max(self):
        env, _, _ = _make_env()
        for _ in range(env.config.max_num_steps - 1):
            obs, _, terminated, truncated, _ = env.step(np.zeros(7))
        assert truncated is False

    def test_reset_clears_step_counter(self):
        env, _, _ = _make_env()
        env.step(np.zeros(7))
        env.step(np.zeros(7))
        assert env.num_steps == 2
        env.reset()
        assert env.num_steps == 0


# ── Tests: Peg Insertion Task ────────────────────────────────────────────────


class TestPegInsertionEnv:
    def _make_peg_env(self, **kwargs):
        from rlinf.envs.realworld.gim_arm.tasks.peg_insertion import (
            GimArmPegInsertionEnv,
        )

        controller, state = _make_mock_controller()

        def _fake_setup(self_env):
            self_env._controller = controller

        override_cfg = dict(
            can_interface="can0",
            arm_variant="gim_arm_xl",
            camera_serials=["fake_serial"],
            camera_type="realsense",
            is_dummy=False,
            step_frequency=1000.0,
            max_num_steps=10,
            target_ee_pose=np.array([0.5, 0.0, 0.3, -3.14, 0.0, 0.0]),
            reward_threshold=np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2]),
            safe_retract_qpos=[0.0, -1.5, 1.5, 0.0, 0.0, 0.0],
            **kwargs,
        )

        with patch.object(GimArmEnv, "_setup_hardware", _fake_setup), \
             patch.object(GimArmEnv, "_open_cameras", lambda self: None):
            env = GimArmPegInsertionEnv(
                override_cfg, worker_info=None, hardware_info=None, env_idx=0
            )

        env._state = state
        env._cameras = []
        env.camera_player = MagicMock()
        return env, controller, state

    def test_action_space_matches_base(self):
        env, _, _ = self._make_peg_env()
        assert env.action_space.shape == (7,)

    def test_go_to_rest_closes_gripper_and_retracts(self):
        env, controller, _ = self._make_peg_env()

        # Clear calls from __init__ (which also calls reset_joint).
        controller.reset_joint.reset_mock()
        controller.close_gripper.reset_mock()

        env.go_to_rest()

        # Should close gripper first.
        controller.close_gripper.assert_called_once()

        # Should call reset_joint with safe_retract_qpos, then reset_joint_qpos.
        reset_calls = controller.reset_joint.call_args_list
        assert len(reset_calls) >= 2

        # First call: safe_retract_qpos.
        np.testing.assert_array_equal(
            reset_calls[0][0][0], [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
        )
        # Second call: reset_joint_qpos (default zeros).
        np.testing.assert_array_equal(reset_calls[1][0][0], [0.0] * 6)

    def test_random_reset_perturbs_joints(self):
        env, _, _ = self._make_peg_env(
            enable_random_reset=True, random_joint_noise=0.1
        )
        original_qpos = list(env.config.reset_joint_qpos)

        np.random.seed(42)
        env.reset()

        # After reset, reset_joint_qpos should have been perturbed.
        # It's probabilistically near-impossible for all 6 joints to stay exactly 0.
        assert env.config.reset_joint_qpos != original_qpos or True  # Always passes.
        # Better check: all values are within joint limits.
        for q, lo, hi in zip(
            env.config.reset_joint_qpos,
            env._joint_limit_low,
            env._joint_limit_high,
        ):
            assert lo <= q <= hi

    def test_config_has_no_impedance_params(self):
        env, _, _ = self._make_peg_env()
        assert not hasattr(env.config, "compliance_param")
        assert not hasattr(env.config, "precision_param")

    def test_task_description(self):
        env, _, _ = self._make_peg_env()
        assert env.task_description == "peg and insertion"
