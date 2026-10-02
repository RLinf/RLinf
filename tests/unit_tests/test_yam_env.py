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

"""Hardware-free Gym contract tests for the dual-YAM environment."""

from __future__ import annotations

import numpy as np
import pytest

from rlinf.envs.real.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.real.yam.types import (
    DualYamState,
    YamArmState,
    YamCommandResult,
    split_dual_action,
)


class _Runtime:
    def __init__(self, state_vector):
        self.state = np.asarray(state_vector, dtype=np.float64).copy()
        self.connect_calls = 0
        self.read_calls = 0
        self.commands = []
        self.moves = []
        self.hold_calls = 0
        self.close_calls = 0

    def connect_followers(self):
        self.connect_calls += 1

    def read_state(self):
        self.read_calls += 1
        left, right = split_dual_action(self.state)
        return DualYamState(
            left=YamArmState(left[:6], left[6], 1.0),
            right=YamArmState(right[:6], right[6], 1.0),
        )

    def command(self, action):
        requested = np.asarray(action, dtype=np.float64).copy()
        self.commands.append(requested)
        self.state = requested
        return YamCommandResult(requested=requested, accepted=requested)

    def move_to(self, target, **kwargs):
        target = np.asarray(target, dtype=np.float64).copy()
        self.moves.append((target, kwargs))
        self.state = target
        return target.copy()

    def hold(self):
        self.hold_calls += 1
        return self.state.copy()

    def emergency_hold(self):
        self.hold_calls += 1

    def close(self):
        self.close_calls += 1


def _camera_must_not_be_created(_camera_info):
    raise AssertionError("dummy YAM env attempted to create a camera")


def test_dummy_reset_is_lazy_and_returns_the_canonical_observation():
    initial = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
        + [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.25]
    )
    runtime = _Runtime(initial)
    env = DualYamJointEnv(
        override_cfg={
            "is_dummy": True,
            "image_height": 4,
            "image_width": 5,
            "dummy_camera_names": ["top_rgb", "left_rgb", "right_rgb"],
            "task_description": "pick the block",
        },
        runtime=runtime,
        camera_factory=_camera_must_not_be_created,
    )

    assert runtime.connect_calls == 0
    assert runtime.read_calls == 0
    assert env.action_space.shape == (14,)
    assert env.task_description == "pick the block"

    observation, info = env.reset()

    assert runtime.connect_calls == 1
    assert info == {"episode_phase": "pre"}
    np.testing.assert_allclose(observation["state"]["joint_position"], initial)
    assert list(observation["frames"]) == ["top_rgb", "left_rgb", "right_rgb"]
    assert all(
        frame.shape == (4, 5, 3) and not frame.any()
        for frame in observation["frames"].values()
    )
    assert env.observation_space.contains(observation)
    np.testing.assert_allclose(env.get_joint_positions(), initial)
    np.testing.assert_allclose(env.get_hold_action(), initial)


def test_dummy_step_preserves_14d_order_and_close_is_idempotent(monkeypatch):
    runtime = _Runtime(np.zeros(14))
    env = DualYamJointEnv(
        override_cfg={
            "is_dummy": True,
            "max_num_steps": 1,
            "dummy_camera_names": ["top_rgb"],
        },
        runtime=runtime,
        camera_factory=_camera_must_not_be_created,
    )
    monkeypatch.setattr(env, "_pace", lambda: None)
    env.reset()
    action = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] + [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.2],
        dtype=np.float32,
    )

    observation, reward, terminated, truncated, info = env.step(action)

    np.testing.assert_allclose(runtime.commands, [action])
    np.testing.assert_allclose(observation["state"]["joint_position"], action)
    np.testing.assert_allclose(info["accepted_action"], action)
    assert reward == 0.0
    assert not terminated
    assert truncated

    env.close()
    env.close()
    assert runtime.close_calls == 1


def test_park_on_close_disabled_by_default():
    runtime = _Runtime(np.zeros(14))
    env = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
        camera_factory=_camera_must_not_be_created,
    )
    env.reset()
    env.close()
    assert runtime.moves == []
    assert runtime.close_calls == 1


def test_action_parts_describe_the_14d_vector_in_step_order():
    from rlinf.envs.real.wrappers.teleop.layout import action_spec
    from rlinf.robotics.actions import ActionKind

    runtime = _Runtime(np.zeros(14))
    env = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
        camera_factory=_camera_must_not_be_created,
    )

    spec = action_spec(env)

    assert spec.kinds == {
        "left.arm": ActionKind.JOINT_POSITION,
        "left.end_effector": ActionKind.GRIPPER,
        "right.arm": ActionKind.JOINT_POSITION,
        "right.end_effector": ActionKind.GRIPPER,
    }
    assert spec.layout == {
        "left.arm": slice(0, 6),
        "left.end_effector": slice(6, 7),
        "right.arm": slice(7, 13),
        "right.end_effector": slice(13, 14),
    }


def test_the_shared_teleop_stack_takes_only_yam_picos_own_mapping():
    from rlinf.envs.real.wrappers.teleop.config import resolve_teleop_devices

    # ``pico`` produces Cartesian deltas or poses, while YAM's arm slot means
    # absolute joint angles. Accepting the shared name would write motion into
    # the wrong slot, so YAM reaches the controllers through ``yam_pico``.
    with pytest.raises(ValueError, match="Unsupported teleop device"):
        resolve_teleop_devices({"teleop": ["pico"]}, supported=DualYamJointEnv.TELEOP)
    assert resolve_teleop_devices(
        {"teleop": "yam_pico"}, supported=DualYamJointEnv.TELEOP
    ) == ["yam_pico"]


def test_dummy_yam_import_does_not_require_i2rt():
    import subprocess
    import sys

    code = """
import builtins
import psutil
psutil.process_iter = lambda: ()
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'i2rt' or name.startswith('i2rt.'):
        raise AssertionError(f'unexpected i2rt import: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import rlinf.envs.real.yam
from rlinf.robotics.robots.dual_yam import DualYamRobot
assert DualYamRobot.ROBOT_TYPE == 'DualYam'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_pico_collection_recipe_composes_with_site_values(monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    for key in (
        "YAM_LEFT_FOLLOWER_CAN",
        "YAM_RIGHT_FOLLOWER_CAN",
        "YAM_TOP_CAMERA_SERIAL",
        "YAM_LEFT_CAMERA_SERIAL",
        "YAM_RIGHT_CAMERA_SERIAL",
        "YAM_PICO_ZMQ_ADDR",
        "YAM_LEFT_OPERATOR_TO_ROBOT_YAW",
        "YAM_RIGHT_OPERATOR_TO_ROBOT_YAW",
    ):
        monkeypatch.setenv(key, "0" if key.endswith("YAW") else f"test-{key}")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(config_name="realworld_dual_yam_collect_data_pico")
    resolved = OmegaConf.to_container(config, resolve=True)
    assert resolved["env"]["eval"]["teleop"] == "yam_pico"
    assert resolved["cluster"]["num_nodes"] == 1
