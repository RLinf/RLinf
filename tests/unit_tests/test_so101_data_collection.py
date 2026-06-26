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

"""Tests for SO101 data-collection plumbing.

Covers:
- Action / observation key formats for the LeRobot bus (the hidden coupling
  most likely to silently break: ``send_action`` filters incoming keys via
  ``key.endswith(".pos")``).
- Lifecycle in dummy mode (init / reset / step / close).
- Keyboard-driven episode termination paths.
- Save-decision logic in ``DataCollector``.
"""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from rlinf.envs.realworld.so101.so101_env import (
    _NUM_ARM_JOINTS,
    _SO101_ARM_JOINTS,
    _SO101_GRIPPER,
    _SO101_MOTOR_NAMES,
    SO101Env,
    SO101RobotConfig,
    _to_lerobot_action,
)
from rlinf.envs.realworld.so101.tasks.so101_pick import (
    SO101PickConfig,
    SO101PickEnv,
)


def _make_env(**overrides) -> SO101Env:
    cfg = SO101RobotConfig(is_dummy=True, **overrides)
    return SO101Env(cfg, worker_info=None, hardware_info=None, env_idx=0)


def _make_pick_env(**overrides) -> SO101PickEnv:
    """Construct a dummy ``SO101PickEnv``.

    The class wraps ``override_cfg`` into a ``SO101PickConfig`` itself, so we
    pass overrides through that channel.
    """
    overrides.setdefault("is_dummy", True)
    return SO101PickEnv(
        override_cfg=overrides,
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )


def _real_hardware_env(**overrides) -> SO101Env:
    """Build a non-dummy SO101Env without touching real hardware.

    Bypasses ``__init__`` entirely so ``_setup_hardware`` and ``_connect_robot``
    don't run; the test then injects mocks for ``_robot`` / ``_leader``.
    """
    cfg = SO101RobotConfig(is_dummy=False, **overrides)
    env = SO101Env.__new__(SO101Env)
    from rlinf.utils.logging import get_logger

    env._logger = get_logger()
    env.config = cfg
    env.hardware_info = None
    env.env_idx = 0
    env.node_rank = 0
    env.env_worker_rank = 0
    from rlinf.envs.realworld.so101.so101_robot_state import SO101RobotState

    env._state = SO101RobotState()
    env._num_steps = 0
    env._robot = None
    env._leader = None
    env._cameras = {}
    env._pynput_listener = None
    env._key_state_lock = None
    env._key_state = {
        "episode_success": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }
    env._init_action_obs_spaces()
    return env


# ---------------------------------------------------------------------------
# LeRobot action / observation key contract — the most failure-prone surface.
# ---------------------------------------------------------------------------


class TestLeRobotKeyContract:
    """Every motor key sent to / read from LeRobot must end with ``.pos``."""

    def test_all_motor_action_keys_have_pos_suffix(self):
        action = _to_lerobot_action(np.zeros(_NUM_ARM_JOINTS), 45.0)
        assert set(action.keys()) == {f"{m}.pos" for m in _SO101_MOTOR_NAMES}

    def test_action_dict_includes_gripper(self):
        action = _to_lerobot_action(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 60.0)
        assert action[f"{_SO101_GRIPPER}.pos"] == 60.0

    def test_arm_targets_assigned_in_motor_order(self):
        targets = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        action = _to_lerobot_action(targets, 0.0)
        for i, name in enumerate(_SO101_ARM_JOINTS):
            assert action[f"{name}.pos"] == targets[i]

    def test_step_sends_pos_suffixed_keys_to_robot(self):
        env = _real_hardware_env()
        env._robot = MagicMock()
        env._robot.send_action = MagicMock()
        try:
            env.step(np.zeros(env.action_space.shape[0], dtype=np.float32))
        finally:
            env.close()

        sent = env._robot.send_action.call_args[0][0]
        assert isinstance(sent, dict)
        for key in sent:
            assert key.endswith(".pos"), f"key {key!r} would be dropped by SOFollower"
        assert {*sent.keys()} == {f"{m}.pos" for m in _SO101_MOTOR_NAMES}

    def test_step_clips_arm_to_joint_limits(self):
        env = _real_hardware_env()
        env._robot = MagicMock()
        try:
            way_too_big = np.array([999.0] * _NUM_ARM_JOINTS + [999.0], dtype=np.float32)
            env.step(way_too_big)
        finally:
            env.close()

        sent = env._robot.send_action.call_args[0][0]
        for i, name in enumerate(_SO101_ARM_JOINTS):
            assert sent[f"{name}.pos"] == pytest.approx(env._joint_limit_high[i])
        assert sent[f"{_SO101_GRIPPER}.pos"] == pytest.approx(
            env.config.gripper_limit_high
        )

    def test_step_rejects_wrong_action_dim(self):
        env = _make_env()
        env.reset()
        with pytest.raises(ValueError, match="action must have"):
            env.step(np.zeros(_NUM_ARM_JOINTS))  # missing gripper

    def test_observation_parses_lerobot_pos_keys(self):
        env = _real_hardware_env()
        observation = {f"{name}.pos": float(i + 1) for i, name in enumerate(_SO101_ARM_JOINTS)}
        observation[f"{_SO101_GRIPPER}.pos"] = 73.5
        env._robot = MagicMock()
        env._robot.get_observation = MagicMock(return_value=observation)
        try:
            env._update_state()
        finally:
            env.close()

        np.testing.assert_array_almost_equal(
            env._state.joint_position, np.arange(1, _NUM_ARM_JOINTS + 1, dtype=np.float64)
        )
        assert env._state.gripper_position == pytest.approx(73.5)

    def test_leader_action_overrides_policy_action_in_teleop(self):
        env = _real_hardware_env(enable_teleop=True, manual_episode_control_only=True)
        env._robot = MagicMock()
        leader_action = {f"{m}.pos": float(i * 10) for i, m in enumerate(_SO101_MOTOR_NAMES)}
        env._leader = MagicMock()
        env._leader.get_action = MagicMock(return_value=leader_action)
        try:
            obs, reward, term, trunc, info = env.step(
                np.zeros(_NUM_ARM_JOINTS + 1, dtype=np.float32)
            )
        finally:
            env.close()

        sent = env._robot.send_action.call_args[0][0]
        assert sent == leader_action
        # intervene_action surfaces the leader's targets in motor order.
        for i, name in enumerate(_SO101_MOTOR_NAMES):
            assert info["intervene_action"][i] == pytest.approx(float(i * 10))


# ---------------------------------------------------------------------------
# Dummy-mode lifecycle
# ---------------------------------------------------------------------------


class TestSO101EnvDummyLifecycle:
    def test_action_space_is_arm_plus_gripper(self):
        env = _make_env()
        try:
            assert env.action_space.shape == (_NUM_ARM_JOINTS + 1,)
            assert env.action_space.low[-1] == env.config.gripper_limit_low
            assert env.action_space.high[-1] == env.config.gripper_limit_high
        finally:
            env.close()

    def test_observation_space_has_default_camera_slot(self):
        """Even with no cameras, a single ``camera_0`` slot is declared so
        ``RealWorldEnv._wrap_obs`` (which reads ``main_image_key``) keeps
        working without conditional branches."""
        env = _make_env()
        try:
            assert "frames" in env.observation_space.spaces
            assert "camera_0" in env.observation_space["frames"].spaces
            assert env.observation_space["state"]["joint_position"].shape == (
                _NUM_ARM_JOINTS,
            )
            assert env.observation_space["state"]["gripper_position"].shape == (1,)
        finally:
            env.close()

    def test_reset_resets_step_counter_and_keys(self):
        env = _make_env(enable_teleop=True, manual_episode_control_only=True)
        env._key_state["episode_success"] = True
        try:
            env.reset()
            assert env._num_steps == 0
            assert all(v is False for v in env._key_state.values())
        finally:
            env.close()

    def test_step_increments_counter(self):
        env = _make_env()
        try:
            env.reset()
            env.step(env.action_space.sample())
            assert env._num_steps == 1
        finally:
            env.close()

    def test_close_is_idempotent(self):
        env = _make_env()
        env.close()
        env.close()


class TestSO101EnvTermination:
    def test_keyboard_event_terminates_in_manual_mode(self):
        env = _make_env(enable_teleop=True, manual_episode_control_only=True)
        try:
            env.reset()
            env._key_state["episode_success"] = True
            _, _, terminated, _, info = env.step(env.action_space.sample())
            assert terminated is True
            assert info["episode_success"] is True
            assert info["manual_done"] is True
        finally:
            env.close()

    def test_max_episode_steps_truncates(self):
        env = _make_env(max_num_steps=200, max_episode_steps=2)
        try:
            env.reset()
            _, _, _, t1, _ = env.step(env.action_space.sample())
            _, _, _, t2, _ = env.step(env.action_space.sample())
            assert t1 is False
            assert t2 is True
        finally:
            env.close()

    def test_max_num_steps_used_when_max_episode_steps_unset(self):
        env = _make_env(max_num_steps=2)
        try:
            env.reset()
            env.step(env.action_space.sample())
            _, _, _, truncated, _ = env.step(env.action_space.sample())
            assert truncated is True
        finally:
            env.close()

    def test_no_teleop_info_keys_when_teleop_disabled(self):
        env = _make_env(enable_teleop=False)
        try:
            env.reset()
            _, _, _, _, info = env.step(env.action_space.sample())
            for key in ("episode_success", "rerecord_episode", "stop_recording"):
                assert key not in info
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestSO101RobotConfigValidation:
    def test_post_init_rejects_inverted_joint_limits(self):
        with pytest.raises(ValueError, match="strictly less"):
            SO101RobotConfig(
                joint_limit_low=np.array([10.0] * _NUM_ARM_JOINTS),
                joint_limit_high=np.array([0.0] * _NUM_ARM_JOINTS),
            )

    def test_post_init_rejects_wrong_joint_limit_shape(self):
        with pytest.raises(ValueError, match="shape"):
            SO101RobotConfig(joint_limit_low=np.zeros(7))

    def test_post_init_rejects_wrong_reset_qpos_length(self):
        with pytest.raises(ValueError, match="reset_joint_qpos"):
            SO101RobotConfig(reset_joint_qpos=[0.0] * 4)

    def test_post_init_rejects_wrong_target_shape(self):
        with pytest.raises(ValueError, match="target_joint_qpos"):
            SO101RobotConfig(target_joint_qpos=np.zeros(7))


# ---------------------------------------------------------------------------
# SO101PickEnv
# ---------------------------------------------------------------------------


class TestSO101PickEnvDummy:
    def test_init_and_dummy_step(self):
        env = _make_pick_env()
        try:
            assert isinstance(env.config, SO101PickConfig)
            obs, _ = env.reset()
            assert obs["state"]["joint_position"].shape == (_NUM_ARM_JOINTS,)
            obs, reward, *_ = env.step(env.action_space.sample())
            assert reward == 0.0  # dummy mode
        finally:
            env.close()

    def test_random_reset_perturbation_stays_within_arm_limits(self):
        # Run reset many times and check arm-only perturbation respects limits.
        env = _make_pick_env(enable_random_reset=True, random_joint_noise_deg=20.0)
        try:
            base = list(env.config.reset_joint_qpos)
            for _ in range(20):
                env.reset()
                if env._perturbed_reset_qpos is None:
                    continue
                arm = np.asarray(env._perturbed_reset_qpos[:_NUM_ARM_JOINTS])
                assert (arm >= env._joint_limit_low - 1e-6).all()
                assert (arm <= env._joint_limit_high + 1e-6).all()
                # Gripper component is preserved unchanged from base.
                assert env._perturbed_reset_qpos[_NUM_ARM_JOINTS] == base[_NUM_ARM_JOINTS]
        finally:
            env.close()


# ---------------------------------------------------------------------------
# DataCollector save-decision logic (replicated locally to avoid cluster setup)
# ---------------------------------------------------------------------------


def _decide(
    manual_only: bool,
    reward: float,
    *,
    episode_success: bool = False,
    rerecord_episode: bool = False,
    stop_recording: bool = False,
    manual_done: bool = False,
):
    """Mirror the save/stop logic from collect_real_data.DataCollector.run."""
    if stop_recording:
        return {"stop": True, "save": False}
    if manual_only:
        save = bool(episode_success or manual_done)
    else:
        save = bool(reward >= 0.5 or episode_success or manual_done)
    if rerecord_episode:
        save = False
    return {"stop": False, "save": save}


class TestDataCollectorDecision:
    def test_manual_save_on_success(self):
        assert _decide(True, 0.0, episode_success=True) == {"stop": False, "save": True}

    def test_manual_discard_on_no_event(self):
        assert _decide(True, 0.0) == {"stop": False, "save": False}

    def test_rerecord_overrides_success(self):
        assert _decide(True, 0.0, episode_success=True, rerecord_episode=True) == {
            "stop": False,
            "save": False,
        }

    def test_stop_recording_short_circuits(self):
        assert _decide(False, 1.0, episode_success=True, stop_recording=True) == {
            "stop": True,
            "save": False,
        }

    def test_rl_mode_save_on_high_reward(self):
        assert _decide(False, 0.8) == {"stop": False, "save": True}

    def test_rl_mode_discard_on_low_reward(self):
        assert _decide(False, 0.2) == {"stop": False, "save": False}

    def test_rl_mode_legacy_manual_done_still_saves(self):
        assert _decide(True, 0.0, manual_done=True) == {"stop": False, "save": True}


# ---------------------------------------------------------------------------
# CollectEpisode → LeRobot frame conversion
# ---------------------------------------------------------------------------


def _so101_obs(state_vals, image_vals=None):
    obs = {"states": np.asarray(state_vals, dtype=np.float32)}
    if image_vals is not None:
        obs["main_images"] = np.asarray(image_vals, dtype=np.uint8)
    return obs


class TestCollectEpisodeLerobotConversion:
    def test_buffer_to_lerobot_frames_with_images(self):
        from rlinf.envs.wrappers.collect_episode import CollectEpisode
        import gymnasium as gym

        base = gym.make("CartPole-v1")
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = CollectEpisode(
                base,
                save_dir=tmp,
                export_format="lerobot",
                robot_type="so101",
                fps=30,
            )
            try:
                buf = wrapped._new_buffer()
                state_dim = _NUM_ARM_JOINTS + 1  # state == arm + gripper
                buf["observations"].append(_so101_obs(np.zeros(state_dim), np.zeros((64, 64, 3))))
                buf["rewards"].append(0.0)
                buf["terminated"].append(False)
                buf["truncated"].append(False)
                buf["infos"].append({})
                for i in range(4):
                    buf["observations"].append(
                        _so101_obs(
                            np.arange(state_dim, dtype=np.float32) + i + 1,
                            np.zeros((64, 64, 3), dtype=np.uint8),
                        )
                    )
                    buf["actions"].append(np.zeros(state_dim, dtype=np.float32))
                    buf["rewards"].append(0.0)
                    buf["terminated"].append(i == 3)
                    buf["truncated"].append(False)
                    buf["infos"].append({})

                frames = wrapped._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)
                assert frames is not None and len(frames) == 4
                for f in frames:
                    assert {"state", "actions", "task", "is_success", "done", "image"} <= f.keys()
                assert frames[-1]["done"].item() is True
            finally:
                wrapped.close()

    def test_intervene_action_overrides_recorded_action(self):
        from rlinf.envs.wrappers.collect_episode import CollectEpisode
        import gymnasium as gym

        base = gym.make("CartPole-v1")
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = CollectEpisode(
                base, save_dir=tmp, export_format="lerobot", robot_type="so101", fps=30
            )
            try:
                buf = wrapped._new_buffer()
                state_dim = _NUM_ARM_JOINTS + 1
                buf["observations"].append(_so101_obs(np.zeros(state_dim)))
                buf["rewards"].append(0.0)
                buf["terminated"].append(False)
                buf["truncated"].append(False)
                buf["infos"].append({})
                expected_action = np.array(
                    [0.5, 0.8, 1.2, -0.3, 0.1, -0.9], dtype=np.float32
                )
                for i in range(3):
                    buf["observations"].append(
                        _so101_obs(np.arange(state_dim, dtype=np.float32) + i + 1)
                    )
                    buf["actions"].append(np.zeros(state_dim, dtype=np.float32))
                    buf["rewards"].append(0.0)
                    buf["terminated"].append(i == 2)
                    buf["truncated"].append(False)
                    buf["infos"].append(
                        {
                            "intervene_action": expected_action,
                            "intervene_flag": np.array([True], dtype=bool),
                        }
                    )
                frames = wrapped._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)
                for f in frames:
                    np.testing.assert_array_almost_equal(f["actions"], expected_action)
            finally:
                wrapped.close()


# ---------------------------------------------------------------------------
# Motor name constants — guard against accidental reordering
# ---------------------------------------------------------------------------


class TestMotorNameConstants:
    def test_arm_joints_are_five(self):
        assert len(_SO101_ARM_JOINTS) == 5
        assert "gripper" not in _SO101_ARM_JOINTS

    def test_motor_names_are_arm_joints_plus_gripper(self):
        assert _SO101_MOTOR_NAMES == (*_SO101_ARM_JOINTS, _SO101_GRIPPER)

    def test_motor_order_matches_lerobot_bus_id_convention(self):
        # LeRobot's SOFollower / SOLeader assigns motor IDs 1..6 in this exact
        # order. If this drifts, send_action and get_observation will silently
        # wire to the wrong joint.
        assert _SO101_ARM_JOINTS == (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        )
