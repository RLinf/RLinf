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

"""Tests for real-world tasks, teleoperation, configuration, and layout."""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import io
import os
import pickle
import re
import subprocess
import sys
import textwrap
import threading
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import rlinf.scheduler.hardware.accelerators.nvidia_gpu as nvidia_gpu
import rlinf.utils.robosuite_compat as robosuite_compat
from rlinf.envs.real import load_tasks
from rlinf.envs.real.dosw1.base import DOSW1Env, DOSW1EnvConfig
from rlinf.envs.real.franka.base import FrankaEnv
from rlinf.envs.real.franka.dual_franka_joint import (
    DualFrankaJointEnv,
)
from rlinf.envs.real.gim_arm.base import GimArmEnv, GimArmEnvConfig
from rlinf.envs.real.task_env import RobotTask, RobotTaskEnv
from rlinf.envs.real.wrappers.teleop.config import (  # noqa: E402
    NO_DEVICE,
    resolve_teleop_device,
    resolve_teleop_devices,
)
from rlinf.envs.real.wrappers.teleop.intervention import (  # noqa: E402
    TeleopDevice,
    TeleopIntervention,
    TeleopSample,
)
from rlinf.envs.real.xsquare.base import Turtle2Env, Turtle2EnvConfig
from rlinf.envs.sim.robotwin.seed_utils import partition_success_seeds
from rlinf.robotics import (
    ControllablePart,
    DualFrankaConfig,
    FrankaConfig,
    PartGroup,
    PiperConfig,
    Robot,
    SO101Config,
)
from rlinf.robotics.discovery import RobotDiscovery
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import (
    EGL_DEVICE_ID_ENV_VARS,
    NvidiaGPUManager,
)
from rlinf.scheduler.manager.net_emulation import (
    CrossDCPair,
    NetEmulationConfig,
    NetEmulationManager,
)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _robot_info(config):
    """Describe test hardware without probing a physical device."""
    if config is None:
        return None
    from rlinf.robotics.discovery import RobotDiscovery, RobotInfo

    robot_type = next(
        name
        for name, registration in RobotDiscovery.registry.items()
        if isinstance(config, registration.config_cls)
    )
    return RobotInfo(
        type=robot_type, model=config.hardware_model(robot_type), config=config
    )


class DummyDriver(ControllablePart):
    def __init__(self) -> None:
        self.connected = False
        self.last_action: dict[str, Any] | None = None

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def observation_features(self) -> dict[str, Any]:
        return {"position": {"shape": (1,)}}

    @property
    def action_features(self) -> dict[str, Any]:
        return {"target": {"shape": (1,)}}

    def connect(self) -> None:
        self.connected = True

    def reset(self) -> None:
        self.last_action = None

    def get_observation(self) -> dict[str, Any]:
        return {"position": np.zeros(1)}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.last_action = action
        return action

    def disconnect(self) -> None:
        self.connected = False


class DummyTask(RobotTask):
    @property
    def description(self) -> str:
        return "Move the test arm."

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {"position": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)}
        )

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "arms": gym.spaces.Dict(
                    {
                        "arm": gym.spaces.Dict(
                            {
                                "arm": gym.spaces.Dict(
                                    {
                                        "target": gym.spaces.Box(
                                            -1.0,
                                            1.0,
                                            shape=(1,),
                                            dtype=np.float32,
                                        )
                                    }
                                )
                            }
                        )
                    }
                )
            }
        )

    def reset(
        self,
        robot: Robot,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed, options
        robot.reset()
        return {"position": np.zeros(1, dtype=np.float32)}, {}

    def step(
        self,
        robot: Robot,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        robot.send_action(action)
        return {"position": np.ones(1, dtype=np.float32)}, 1.0, True, False, {}


def test_robot_task_env_composes_task_and_robot_lifecycles():
    driver = DummyDriver()
    robot = Robot(arm=PartGroup(arm=driver))
    env = RobotTaskEnv(robot, DummyTask())
    action = {"arm": {"arm": {"target": np.array([0.5])}}}

    observation, _ = env.reset(seed=3)
    transition = env.step(action)

    assert env.task_description == "Move the test arm."
    assert observation["position"].tolist() == [0.0]
    assert transition[0]["position"].tolist() == [1.0]
    assert driver.last_action is not None
    assert driver.last_action["target"].tolist() == [0.5]
    env.close()
    assert not driver.is_connected


def _assert_legacy_transition(env) -> None:
    observation, _ = env.reset()
    transition = env.step(env.action_space.sample())

    assert set(observation) == {"state", "frames"}
    assert set(transition[0]) == {"state", "frames"}
    assert len(transition) == 5
    env.close()


@pytest.mark.parametrize(
    ("module_name", "class_name", "override"),
    [
        (
            "rlinf.envs.real.franka",
            "FrankaEnv",
            {"step_frequency": 10000.0},
        ),
        ("rlinf.envs.real.so101", "SO101ReachEnv", {}),
    ],
)
def test_a_hardware_free_env_repeats_with_a_seed(module_name, class_name, override):
    """reset(seed=...) must pin what a dummy env reports.

    With no hardware the env samples its declared space, and a Gymnasium
    space seeds itself from entropy. Two runs of one config then disagree on
    every value, so a dummy end-to-end run can only catch a change of shape.
    That is how an observation built from two different moments went
    unnoticed: it was correctly shaped every time.
    """
    import importlib

    from robot_mocks import mocked_sdks

    def observe(seed):
        with mocked_sdks():
            env_cls = getattr(importlib.import_module(module_name), class_name)
            cfg = {"is_dummy": True, "enable_camera_player": False, **override}
            try:
                env = env_cls(
                    override_cfg=cfg,
                    worker_info=None,
                    env_idx=0,
                    robot_info=_robot_info(
                        FrankaConfig(node_rank=0, camera_serials=["dummy"])
                        if class_name == "FrankaEnv"
                        else None
                    ),
                )
            except TypeError:
                env = env_cls(cfg)
            observation, _ = env.reset(seed=seed)
            return np.concatenate(
                [
                    np.asarray(v).reshape(-1)
                    for _, v in sorted(observation["state"].items())
                ]
            )

    assert np.array_equal(observe(7), observe(7)), "the same seed must repeat"
    assert not np.array_equal(observe(7), observe(8)), "a different seed must differ"


def test_a_franka_observation_comes_from_one_snapshot():
    """Every field a policy sees must describe the same instant.

    _read_robot takes one snapshot per step and the observation is built from
    it. Reading the gripper live instead would mix two moments in one
    recorded transition, by up to a control period.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(FrankaEnv._get_observation))
    tree = ast.parse(source)

    reads = {
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and ast.unparse(node).startswith(("self._franka_state", "self._end_effector"))
    }
    live = sorted(r for r in reads if r.startswith("self._end_effector"))

    assert live == [], (
        f"the observation reads hardware directly: {live}. "
        "Take the value from self._franka_state, which _read_robot fills once."
    )


def test_franka_depth_reaches_the_observation_only_when_asked_for():
    """A rig without a depth camera keeps the schema a policy already reads.

    Depth arrives as its own key rather than a fourth channel, so an existing
    policy sees the frames it always saw and one that wants depth reads
    metres without knowing which camera produced them.
    """
    from robot_mocks import mocked_sdks
    from robot_mocks.cameras import DEPTH_FAR, DEPTH_NEAR, DEPTH_SCALE, SERIAL

    with mocked_sdks():
        from rlinf.envs.real.franka.base import FrankaEnv

        def build(enable_camera_depth):
            return FrankaEnv(
                override_cfg={
                    "enable_camera_depth": enable_camera_depth,
                    "enable_camera_player": False,
                    "step_frequency": 10000.0,
                },
                worker_info=None,
                env_idx=0,
                robot_info=_robot_info(
                    FrankaConfig(
                        node_rank=0,
                        robot_ip="0.0.0.0",
                        camera_serials=[SERIAL],
                        disable_validate=True,
                    )
                ),
            )

        env = build(False)
        try:
            observation, _ = env.reset()
            assert set(env.observation_space.spaces) == {"state", "frames"}
            assert set(observation) == {"state", "frames"}
        finally:
            env.close()

        env = build(True)
        try:
            observation, _ = env.reset()
            assert set(observation) == {"state", "frames", "depths"}
            depth = observation["depths"]["wrist_1"]
            frame = observation["frames"]["wrist_1"]
            # Cropped and resized to the same view as the frame beside it.
            assert depth.shape == frame.shape[:2]
            assert depth.dtype == np.float32
            # Resampling by nearest keeps every pixel at a distance something
            # was actually measured at; averaging would invent readings between
            # the near and far halves, where nothing is.
            distances = np.unique(depth)
            assert len(distances) == 2
            assert np.allclose(
                distances, [DEPTH_NEAR * DEPTH_SCALE, DEPTH_FAR * DEPTH_SCALE]
            )
        finally:
            env.close()


def test_depth_reaches_the_policy_split_like_the_frames_beside_it():
    """A policy reads depth the way it reads images: main view, then the rest.

    The runner never sees the per-camera dict, so depth has to follow the same
    main/extra split as the frames, keyed by the same camera.
    """
    from rlinf.envs.real.env import RealWorldEnv

    def wrap(raw_observation):
        env = RealWorldEnv.__new__(RealWorldEnv)
        env.main_image_key = "wrist_1"
        env.task_descriptions = ["pick up the cube"]
        return env._wrap_obs(raw_observation)

    frames = {
        "wrist_1": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "wrist_2": np.ones((1, 4, 4, 3), dtype=np.uint8),
    }
    state = {"tcp_pose": np.zeros((1, 7), dtype=np.float32)}

    observation = wrap({"state": state, "frames": frames})
    assert "main_depths" not in observation
    assert "extra_view_depths" not in observation

    observation = wrap(
        {
            "state": state,
            "frames": frames,
            "depths": {
                "wrist_1": np.full((1, 4, 4), 0.5, dtype=np.float32),
                "wrist_2": np.full((1, 4, 4), 1.5, dtype=np.float32),
            },
        }
    )
    assert observation["main_depths"].shape == (1, 4, 4)
    assert torch.allclose(observation["main_depths"], torch.tensor(0.5))
    # The extra views stack on axis 1, as the extra images do.
    assert observation["extra_view_depths"].shape == (1, 1, 4, 4)
    assert torch.allclose(observation["extra_view_depths"], torch.tensor(1.5))


def test_franka_dummy_preserves_legacy_policy_schema():
    env = FrankaEnv(
        override_cfg={
            "is_dummy": True,
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        env_idx=0,
        robot_info=_robot_info(FrankaConfig(node_rank=0, camera_serials=["dummy"])),
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_dual_franka_dummy_preserves_legacy_policy_schema():
    env = DualFrankaJointEnv(
        override_cfg={
            "is_dummy": True,
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        env_idx=0,
        robot_info=_robot_info(
            DualFrankaConfig(node_rank=0, base_camera_serials=["dummy"])
        ),
    )

    assert env.action_space.shape == (16,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_gim_arm_dummy_preserves_legacy_policy_schema():
    env = GimArmEnv(
        config=GimArmEnvConfig(
            is_dummy=True,
            enable_camera_player=False,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_dosw1_dummy_preserves_legacy_policy_schema():
    env = DOSW1Env(
        config=DOSW1EnvConfig(
            is_dummy=True,
            camera_names=[],
            enable_camera_player=False,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (14,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_turtle2_dummy_preserves_legacy_policy_schema():
    env = Turtle2Env(
        config=Turtle2EnvConfig(
            is_dummy=True,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


class _TerminatingEnv(gym.Env):
    """Terminates on its second step and counts how often it is reset."""

    observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.step_count = 0
        self.resets = 0

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self.step_count = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return (
            np.array([self.step_count], dtype=np.float32),
            float(self.step_count),
            self.step_count >= 2,
            False,
            {},
        )


def test_the_vector_env_keeps_stepping_a_terminated_env():
    """The runner resets on its own schedule, so the batch must not.

    gymnasium renamed the batch arrays in 1.x and added an autoreset mode
    whose DISABLED setting refuses to step a terminated env rather than
    carrying on, so both halves are worth pinning: nothing is reset, and the
    observation is the one the env produced after it terminated.
    """
    from rlinf.envs.real.venv import NoAutoResetSyncVectorEnv

    env = NoAutoResetSyncVectorEnv([_TerminatingEnv, _TerminatingEnv])
    try:
        env.reset(seed=0)
        after_reset = [inner.resets for inner in env.envs]
        for _ in range(4):
            observation, reward, terminated, truncated, _ = env.step(
                np.zeros((2, 1), dtype=np.float32)
            )

        assert [inner.resets for inner in env.envs] == after_reset
        assert terminated.tolist() == [True, True]
        assert observation.ravel().tolist() == [4.0, 4.0]
        assert reward.tolist() == [4.0, 4.0]
    finally:
        env.close()


def _turtle2_camera_check(camera_ids, ready):
    """Run _check_cameras against a rig with the given cameras."""
    env = Turtle2Env.__new__(Turtle2Env)
    env.config = SimpleNamespace(is_dummy=False)
    env.hardware = SimpleNamespace(camera_ids=list(camera_ids))
    env._camera_parts = lambda: [
        SimpleNamespace(is_ready=lambda state=state: state) for state in ready
    ]
    env._check_cameras()


def test_turtle2_accepts_a_camera_selected_by_a_nonzero_id():
    """``camera_ids`` selects hardware; the parts are named by position.

    The shipped default is ``[2]``, one camera, so reading the id as a slot
    rejected a healthy rig.
    """
    _turtle2_camera_check([2], [True])
    _turtle2_camera_check([1], [True])
    _turtle2_camera_check([0, 1, 2], [True, True, True])


def test_turtle2_refuses_a_camera_that_is_not_delivering():
    """A stalled camera is still named by the id that selected it."""
    with pytest.raises(ValueError, match="Camera 3 not available"):
        _turtle2_camera_check([2], [False])
    # Built short: the robot has fewer cameras than the config asked for.
    with pytest.raises(ValueError, match="Camera 3 not available"):
        _turtle2_camera_check([2], [])
    with pytest.raises(ValueError, match="Camera 2 not available"):
        _turtle2_camera_check([0, 1, 2], [True, False, True])


def test_franka_builds_cameras_after_applying_hardware_info(monkeypatch):
    from rlinf.envs.real.franka.base import FrankaEnvConfig
    from rlinf.robotics import FrankaConfig, RobotInfo
    from rlinf.robotics.robots.franka import FrankaRobot

    captured = {}

    class BuiltRobot:
        def connect(self):
            pass

        def child(self, name, part_type=None):
            # The env reaches for the arm and, beside it, the end effector,
            # naming the class it expects each to be.
            assert name in ("arm", "end_effector")
            assert part_type is not None, "the env should say what it expects"
            # is_hand is part of the end-effector contract: the env asks the
            # part which kind it is rather than trusting the config alone.
            return SimpleNamespace(owner=object(), is_hand=False, is_gripper=True)

    def build(**kwargs):
        captured.update(kwargs)
        return BuiltRobot()

    monkeypatch.setattr(FrankaRobot, "build", build)
    env = FrankaEnv.__new__(FrankaEnv)
    env.config = FrankaEnvConfig()
    env.robot_info = RobotInfo(
        type="Robot",
        model="Franka",
        config=FrankaConfig(
            node_rank=0,
            robot_ip="10.0.0.1",
            camera_serials=["hardware-camera"],
        ),
    )
    env.env_idx = 0
    env.node_rank = 0
    env.env_worker_rank = 3
    env.hardware = env.robot_info.config

    env._setup_hardware()

    assert [info.serial_number for info in env._camera_infos] == ["hardware-camera"]
    assert list(captured["cameras"]) == ["wrist_1"]


def test_gim_arm_reopens_the_existing_camera_after_a_stall(monkeypatch):
    from rlinf.robotics.parts.cameras import CameraInfo

    class Camera:
        def __init__(self):
            self._camera_info = CameraInfo("wrist_1", "camera")
            self.reads = 0
            self.reopens = 0

        @property
        def name(self) -> str:
            """As BaseCamera exposes it, so the env need not reach inside."""
            return self._camera_info.name

        def get_frame(self):
            self.reads += 1
            if self.reads == 1:
                raise __import__("queue").Empty
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def reopen(self):
            self.reopens += 1

    env = GimArmEnv.__new__(GimArmEnv)
    camera = Camera()
    env._cameras = [camera]
    env._logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    env.camera_player = SimpleNamespace(put_frame=lambda frames: None)
    env.observation_space = gym.spaces.Dict(
        {
            "frames": gym.spaces.Dict(
                {"wrist_1": gym.spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8)}
            )
        }
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    frames = env._get_camera_frames()

    assert camera.reopens == 1
    assert frames["wrist_1"].shape == (4, 4, 3)


def test_dual_franka_reads_depth_beside_each_frame():
    """Both arms' cameras report depth through the same reading as the frame.

    The dual env reads each camera on its own so a stalled one cannot stall
    the control loop, which is why depth has to arrive on that path too.
    """
    near, far = 0.5, 1.5

    class _Camera:
        def __init__(self, with_depth):
            self.timeouts = []
            self._with_depth = with_depth

        def get_observation(self, timeout=5, attempts=1, wait=0.0):
            self.timeouts.append(timeout)
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            if not self._with_depth:
                return {"frame": frame}
            # The near/far step sits off the resize grid so an averaging
            # resample would show up as a distance nothing measured.
            depth = np.full((8, 8), far, dtype=np.float32)
            depth[:, :3] = near
            return {"frame": frame, "depth": depth}

    def read(with_depth):
        env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
        camera = _Camera(with_depth)
        env._cameras = {"left_wrist_0_rgb": camera}
        env._last_camera_frame = {}
        env._logger = SimpleNamespace(error=lambda *args, **kwargs: None)
        env.camera_player = SimpleNamespace(put_frame=lambda frames: None)
        env.observation_space = gym.spaces.Dict(
            {
                "frames": gym.spaces.Dict(
                    {
                        "left_wrist_0_rgb": gym.spaces.Box(
                            0, 255, shape=(4, 4, 3), dtype=np.uint8
                        )
                    }
                )
            }
        )
        return camera, env._get_camera_observation()

    camera, (frames, depths) = read(with_depth=False)
    assert frames["left_wrist_0_rgb"].shape == (4, 4, 3)
    assert depths == {}
    # Read on the control period, not the default timeout, so a stalled
    # camera falls back to its last reading instead of holding the loop.
    assert camera.timeouts == [0.5]

    _, (frames, depths) = read(with_depth=True)
    depth = depths["left_wrist_0_rgb"]
    assert depth.shape == (4, 4)
    # Nearest resampling keeps every pixel at a measured distance.
    distances = np.unique(depth)
    assert len(distances) == 2
    assert np.allclose(distances, [near, far])


def test_dual_franka_declares_depth_only_where_a_camera_captures_it():
    """The depth space follows the cameras, so a rig without one is unchanged."""

    def camera_spaces(enable_camera_depth):
        env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
        env.hardware = DualFrankaConfig(
            node_rank=0, base_camera_serials=["dummy"], camera_type="realsense"
        )
        env.config = SimpleNamespace(enable_camera_depth=enable_camera_depth)
        return env._build_camera_spaces()

    assert set(camera_spaces(False)) == {"frames"}

    spaces = camera_spaces(True)
    assert set(spaces) == {"frames", "depths"}
    depth_space = spaces["depths"]["base_0_rgb"]
    assert depth_space.shape == spaces["frames"]["base_0_rgb"].shape[:2]
    assert depth_space.dtype == np.float32


def test_dual_franka_runs_independent_arm_calls_concurrently():
    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )
    rendezvous = threading.Barrier(2)

    def call(side):
        rendezvous.wait(timeout=1.0)
        return side

    try:
        assert env._run_arm_calls(lambda: call("left"), lambda: call("right")) == (
            "left",
            "right",
        )
    finally:
        for executor in env._arm_executors:
            executor.shutdown(wait=True)


def test_dual_franka_applies_a_tasks_compliance_to_both_arms():
    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env.config = SimpleNamespace(compliance_param={"translational_stiffness": 800})
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )

    class Arm:
        def __init__(self):
            self.applied = None

        def reconfigure_compliance_params(self, params):
            self.applied = params

    env._left_arm, env._right_arm = Arm(), Arm()
    try:
        env._reconfigure_compliance()
    finally:
        for executor in env._arm_executors:
            executor.shutdown(wait=True)

    assert env._left_arm.applied == {"translational_stiffness": 800}
    assert env._right_arm.applied == {"translational_stiffness": 800}


def test_dual_franka_does_not_wait_for_gripper_motion():
    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env.config = SimpleNamespace(binary_gripper_threshold=0.5)
    env._logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )
    entered = threading.Event()
    release = threading.Event()

    class Hand:
        is_open = True

        def close(self):
            entered.set()
            assert release.wait(timeout=1.0)

    try:
        changed = env._gripper_action(0, Hand(), -1.0)
        assert changed
        assert entered.wait(timeout=1.0)
        assert not release.is_set(), "the control loop must not wait for the gripper"
    finally:
        release.set()
        for executor in env._arm_executors:
            executor.shutdown(wait=True)


def test_franka_reward_model_waits_for_the_worker_result():
    class Work:
        def wait(self):
            return [np.array([0.75], dtype=np.float32)]

    env = FrankaEnv.__new__(FrankaEnv)
    env.config = SimpleNamespace(reward_image_key=None)
    env._reward_worker = SimpleNamespace(compute_reward=lambda _batch: Work())

    reward = env._compute_reward_model(
        {"frames": {"wrist_1": np.zeros((4, 4, 3), dtype=np.uint8)}}
    )

    assert reward == pytest.approx(0.75)


def test_direct_gello_stream_keeps_both_arm_commands_concurrent():
    from rlinf.envs.real.wrappers.teleop.adapters import DualGelloJointStream

    rendezvous = threading.Barrier(2)

    class Controller:
        def move_joints(self, _target):
            rendezvous.wait(timeout=1.0)

    class Leader:
        ready = True

        def get_observation(self):
            return {"joint_position": np.zeros(7), "grip": np.zeros(1)}

    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env._left_ctrl = Controller()
    env._right_ctrl = Controller()
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )
    streamer = DualGelloJointStream(
        Leader(), Leader(), gripper_enabled=False, direct_stream=False
    )

    try:
        streamer.stream_once(env)
    finally:
        for executor in env._arm_executors:
            executor.shutdown(wait=True)


class FakeEnv(gym.Env):
    """Record actions passed to ``step``."""

    def __init__(self) -> None:
        self.stepped: list[np.ndarray] = []
        self.reset_calls = 0
        self.closed = False

    def step(self, action):
        self.stepped.append(np.asarray(action))
        return {"obs": 1}, 0.0, False, False, {}

    def reset(self, **kwargs):
        self.reset_calls += 1
        return {"obs": 1}, {}

    def close(self):
        self.closed = True


class ScriptedDevice(TeleopDevice):
    """Return one scripted sample per read."""

    def __init__(self, samples: list[TeleopSample]) -> None:
        self.samples = list(samples)
        self.reads = 0
        self.resets = 0
        self.closed = False
        self.before_steps = 0
        self.fallback_action: Optional[np.ndarray] = None

    def read(self, env: Any, policy_action: np.ndarray) -> TeleopSample:
        sample = self.samples[min(self.reads, len(self.samples) - 1)]
        self.reads += 1
        return sample

    def reset(self, env: Any) -> None:
        self.resets += 1

    def before_step(self, env: Any) -> None:
        self.before_steps += 1

    def fallback(self, env: Any, policy_action: np.ndarray) -> np.ndarray:
        if self.fallback_action is not None:
            return self.fallback_action
        return policy_action

    def close(self) -> None:
        self.closed = True


POLICY = np.array([0.0, 0.0, 0.0])


EXPERT = np.array([1.0, 1.0, 1.0])


def test_active_sample_replaces_the_policy_action():
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env, ScriptedDevice([TeleopSample(action=EXPERT, active=True)])
    )

    _, _, _, _, info = wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], EXPERT)
    assert np.array_equal(info["intervene_action"], EXPERT)


def test_inactive_device_leaves_the_policy_action_alone():
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env, ScriptedDevice([TeleopSample(action=None, active=False)])
    )

    _, _, _, _, info = wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], POLICY)
    assert "intervene_action" not in info


def test_control_is_held_between_samples_then_released():
    env = FakeEnv()
    device = ScriptedDevice(
        [
            TeleopSample(action=EXPERT, active=True),
            TeleopSample(action=EXPERT, active=False),
        ]
    )
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)  # Operator moves.
    wrapper.step(POLICY)  # Quiet sample within the hold window.
    assert np.array_equal(env.stepped[1], EXPERT)

    device.timeout = 0.0  # Hold window expires.
    wrapper.step(POLICY)
    assert np.array_equal(env.stepped[2], POLICY)


class _StubDevice:
    """A teleop device that reads nothing, for arbitration tests.

    Stands in for the parts of the device contract the group touches, so a
    test can supply only the mapping it wants to exercise.
    """

    NEEDS = ()
    CLIPS_TO_ACTION_SPACE = False
    APPLIES_WHILE_IDLE = False
    HOLD_WINDOW = None
    is_connected = True

    def get_observation(self):
        return {}

    def publish(self, reading):
        return {}

    def hold(self, context):
        return {}

    def on_action_chunk_begin(self):
        pass

    def on_reset(self, context=None):
        pass

    def connect(self):
        pass

    def disconnect(self):
        pass

    def drive(self, context, reading=None):
        if reading is None:
            reading = self.get_observation()
        return self.action(reading, context)


def test_an_unfilled_part_keeps_the_policy_action():
    import numpy as np

    from rlinf.envs.real.wrappers.teleop.composed import ComposedTeleop
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup

    class Fixed(_StubDevice):
        PRODUCES = {"hand": ActionKind.HAND}

        def action(self, reading, context):
            from rlinf.robotics.parts.teleop import TeleopAction

            return TeleopAction(parts={"hand": np.full(6, 0.5)}, driving=True)

    layout = {"arm": slice(0, 6), "hand": slice(6, 12)}
    group = TeleopGroup(
        [TeleopEntry(Fixed())],
        available={"arm": ActionKind.CARTESIAN_DELTA, "hand": ActionKind.HAND},
    )
    device = ComposedTeleop(group, layout)

    policy = np.arange(12, dtype=np.float64)
    sample = device.read(_FakeLayoutEnv(), policy)

    assert np.allclose(sample.action[:6], policy[:6])  # Arm remains policy-driven.
    assert np.allclose(sample.action[6:], 0.5)  # Glove controls the hand.


def test_an_idle_glove_keeps_its_hand_pose_without_claiming_the_arm():
    from rlinf.envs.real.wrappers.teleop.composed import ComposedTeleop
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import TeleopAction, TeleopEntry, TeleopGroup

    class HeldHand(_StubDevice):
        PRODUCES = {"hand": ActionKind.HAND}
        APPLIES_WHILE_IDLE = True

        def action(self, reading, context):
            return TeleopAction(parts={"hand": np.full(6, 0.5)}, driving=False)

    layout = {"arm": slice(0, 6), "hand": slice(6, 12)}
    group = TeleopGroup(
        [TeleopEntry(HeldHand())],
        available={"arm": ActionKind.CARTESIAN_DELTA, "hand": ActionKind.HAND},
    )
    policy = np.arange(12, dtype=np.float64)
    sample = ComposedTeleop(group, layout).read(_FakeLayoutEnv(), policy)
    env = FakeEnv()

    _, _, _, _, info = TeleopIntervention(env, ScriptedDevice([sample])).step(policy)

    assert np.allclose(env.stepped[0][:6], policy[:6])
    assert np.allclose(env.stepped[0][6:], 0.5)
    assert "intervene_action" not in info


class _FakeLayoutEnv:
    """Provide the attributes required by teleoperation layout tests."""

    unwrapped = None

    def get_wrapper_attr(self, name):
        raise AttributeError(name)


def test_mark_flag_is_opt_in():
    sample = TeleopSample(action=EXPERT, active=True)

    plain = TeleopIntervention(FakeEnv(), ScriptedDevice([sample]))
    flagged = TeleopIntervention(FakeEnv(), ScriptedDevice([sample]), mark_flag=True)

    assert "intervene_flag" not in plain.step(POLICY)[4]
    assert flagged.step(POLICY)[4]["intervene_flag"] == np.ones(1)


def test_device_info_reaches_the_step_info():
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env,
        ScriptedDevice([TeleopSample(action=EXPERT, active=True, info={"left": True})]),
    )

    assert wrapper.step(POLICY)[4]["left"] is True


def test_reset_resyncs_the_device_and_drops_the_hold():
    env = FakeEnv()
    device = ScriptedDevice(
        [
            TeleopSample(action=EXPERT, active=True),
            TeleopSample(action=EXPERT, active=False),
        ]
    )
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)
    assert wrapper.intervening

    wrapper.reset()

    assert device.resets == 1
    assert env.reset_calls == 1
    assert not wrapper.intervening


def test_close_releases_the_device_before_the_env():
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=None, active=False)])
    wrapper = TeleopIntervention(env, device)

    wrapper.close()

    assert device.closed
    assert env.closed


def test_before_step_runs_ahead_of_the_env():
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=None, active=False)])
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)

    assert device.before_steps == 1


def test_read_is_abstract():
    with pytest.raises(TypeError):
        TeleopDevice()


SINGLE_ARM = ("spacemouse", "gello", "pico")


def test_retired_single_key_still_selects_its_device_and_warns():
    with pytest.warns(DeprecationWarning, match="'teleop_device' is retired"):
        device = resolve_teleop_device({"teleop_device": "gello"}, supported=SINGLE_ARM)

    assert device == "gello"


def test_missing_config_falls_back_to_the_default():
    assert resolve_teleop_device({}, supported=SINGLE_ARM) == NO_DEVICE
    assert (
        resolve_teleop_device({}, supported=SINGLE_ARM, default="spacemouse")
        == "spacemouse"
    )


def test_retired_boolean_still_selects_its_device_and_warns():
    with pytest.warns(DeprecationWarning, match="use_pico"):
        device = resolve_teleop_device({"use_pico": True}, supported=SINGLE_ARM)

    assert device == "pico"


def test_all_retired_booleans_off_means_no_device():
    with pytest.warns(DeprecationWarning):
        device = resolve_teleop_device(
            {"use_spacemouse": False, "use_gello": False},
            supported=SINGLE_ARM,
            default="spacemouse",
        )

    assert device == NO_DEVICE


def test_two_retired_booleans_on_is_an_error():
    with pytest.raises(ValueError, match="Only one teleop device"):
        resolve_teleop_device(
            {"use_spacemouse": True, "use_pico": True}, supported=SINGLE_ARM
        )


def test_disagreeing_old_and_new_keys_are_refused():
    with pytest.raises(ValueError, match="cannot be reconciled"):
        resolve_teleop_device(
            {"teleop_device": "pico", "use_spacemouse": True},
            supported=SINGLE_ARM,
        )


def test_agreeing_old_and_new_keys_only_warn():
    with pytest.warns(DeprecationWarning, match="redundant"):
        device = resolve_teleop_device(
            {"teleop_device": "pico", "use_pico": True}, supported=SINGLE_ARM
        )

    assert device == "pico"


def test_device_the_env_cannot_drive_is_refused():
    with pytest.raises(ValueError, match="Unsupported teleop device"):
        resolve_teleop_device(
            {"teleop_device": "spacemouse"}, supported=("gello_joint", "pico")
        )


def test_none_is_always_allowed():
    with pytest.warns(DeprecationWarning):
        device = resolve_teleop_device({"teleop_device": "none"}, supported=("pico",))

    assert device == NO_DEVICE


def test_shipped_configs_use_the_new_key():
    roots = [_ROOT / "examples", _ROOT / "evaluations", _ROOT / "tests"]
    offenders = []
    for root in roots:
        for path in root.rglob("*.yaml"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if any(
                    stripped.startswith(f"{flag}:")
                    for flag in (
                        "teleop_device",
                        "use_spacemouse",
                        "use_gello",
                        "use_gello_joint",
                        "use_pico",
                    )
                ):
                    offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == []


_REAL = _ROOT / "rlinf" / "envs" / "real"


_ROBOTS = ("franka", "dosw1", "gim_arm", "xsquare")


EXPECTED_IDS = {
    "FrankaEnv-v1",
    "PegInsertionEnv-v1",
    "FrankaBinRelocationEnv-v1",
    "BottleEnv-v1",
    "DexpnpEnv-v1",
    "DualFrankaJointEnv-v1",
    "DualFrankaTCPEnv-v1",
    "DOSW1PickEnv-v1",
    "ButtonEnv-v1",
    "GimArmPegInsertionEnv-v1",
}


def test_no_robot_keeps_a_tasks_subpackage():
    leftovers = [name for name in _ROBOTS if (_REAL / name / "tasks").exists()]

    assert leftovers == []


def test_every_robot_folder_has_a_base():
    missing = [name for name in _ROBOTS if not (_REAL / name / "base.py").exists()]

    assert missing == []


def test_all_task_ids_are_registered():
    from gymnasium.envs.registration import registry

    from rlinf.envs.real import RealWorldEnv

    assert RealWorldEnv is not None
    assert EXPECTED_IDS <= set(registry)


def test_every_entry_point_resolves():
    from gymnasium.envs.registration import registry

    from rlinf.envs.real import RealWorldEnv

    assert RealWorldEnv is not None
    unresolved = []
    for env_id in sorted(EXPECTED_IDS):
        entry_point = registry[env_id].entry_point
        module_name, _, attribute = str(entry_point).partition(":")
        module = importlib.import_module(module_name)
        if getattr(module, attribute, None) is None:
            unresolved.append(f"{env_id} -> {entry_point}")

    assert unresolved == []


def test_task_tables_cover_the_wrapped_robots():
    from rlinf.envs.real import dosw1, franka, xsquare

    declared = set(franka.TASKS) | set(dosw1.TASKS) | set(xsquare.TASKS)

    # GimArm registers its environment class without a wrapper factory.
    assert declared == EXPECTED_IDS - {"GimArmPegInsertionEnv-v1"}


def test_pose_math_is_not_filed_under_a_robot():
    from rlinf.envs.real.utils import pose

    assert hasattr(pose, "construct_adjoint_matrix")
    assert not (_REAL / "franka" / "utils.py").exists()


def test_task_configs_state_only_their_compliance_deltas():
    from rlinf.envs.real.franka.base import COMPLIANCE_DEFAULTS
    from rlinf.envs.real.franka.bin_relocation import BinEnvConfig
    from rlinf.envs.real.franka.bottle import BottleConfig
    from rlinf.envs.real.franka.dex_pnp import DexpnpConfig
    from rlinf.envs.real.franka.peg_insertion import PegInsertionConfig

    deltas = {
        cls.__name__: {
            key
            for key, value in cls().compliance_param.items()
            if COMPLIANCE_DEFAULTS[key] != value
        }
        for cls in (PegInsertionConfig, BottleConfig, BinEnvConfig, DexpnpConfig)
    }

    # Every task receives the complete gain set after defaults are applied.
    for cls in (PegInsertionConfig, BottleConfig, BinEnvConfig, DexpnpConfig):
        assert set(cls().compliance_param) == set(COMPLIANCE_DEFAULTS)
    assert {name: len(keys) for name, keys in deltas.items()} == {
        "PegInsertionConfig": 1,
        "BottleConfig": 8,
        "BinEnvConfig": 11,
        "DexpnpConfig": 6,
    }


def test_unknown_compliance_gain_is_refused():
    import pytest

    from rlinf.envs.real.franka.base import compliance

    with pytest.raises(KeyError, match="Unknown compliance gains"):
        compliance(translational_stifness=1000)


# Wrapper families


def test_wrappers_are_split_by_what_they_change():
    real = _ROOT / "rlinf" / "envs" / "real"
    wrappers = real / "wrappers"

    assert wrappers.is_dir(), "the three families live under one parent"
    for family in ("teleop", "transforms", "episode"):
        assert (wrappers / family / "__init__.py").exists(), family

    # Top-level modules contain robot packages and shared environment machinery.
    loose = sorted(
        path.stem for path in real.glob("*.py") if path.name != "__init__.py"
    )
    assert loose == ["env", "registry", "task_env", "venv"], loose


def test_no_teleop_wrapper_is_left_outside_teleop():
    real = _ROOT / "rlinf" / "envs" / "real"
    strays = sorted(
        path.name
        for family in ("transforms", "episode")
        for path in (real / "wrappers" / family).glob("*.py")
        if "intervention" in path.name and "leader_follower" not in path.name
    )

    assert strays == []


def test_a_held_button_device_does_not_keep_control_after_release():
    from rlinf.robotics.parts.teleop import Pico, PicoTcp

    assert Pico.HOLD_WINDOW == 0.0
    assert PicoTcp.HOLD_WINDOW == 0.0


def test_streaming_device_lifecycle_without_hardware():
    from rlinf.envs.real.wrappers.teleop.intervention import TeleopSample
    from rlinf.envs.real.wrappers.teleop.streaming import TeleopStreamer

    ticks = []

    class Fake(TeleopStreamer):
        def read(self, env, policy_action):
            return TeleopSample(action=None, active=False)

        def stream_once(self, env):
            ticks.append(1)

    device = Fake(period=0.001, enabled=True)
    device.before_reset(None, {})
    device.reset(None)
    device.after_reset(None)
    deadline = time.monotonic() + 1.0
    while not ticks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ticks, "stream thread never ran"
    assert device.streaming

    device.close()

    assert not device.streaming


# Keyboard sessions


def _keyboard_session(monkeypatch, queued):
    """Build a keyboard session that replays queued key batches."""
    from rlinf.envs.real.wrappers.episode import session as session_module

    class FakeListener:
        def __init__(self):
            self.batches = list(queued)

        def pop_pressed_keys(self):
            return self.batches.pop(0) if self.batches else []

        def get_key(self):
            batch = self.pop_pressed_keys()
            return batch[0] if batch else None

    monkeypatch.setattr(session_module, "KeyboardListener", FakeListener)

    class Env(gym.Env):
        def __init__(self):
            self.resets = 0

        def reset(self, seed=None, options=None):
            self.resets += 1
            return {}, {}

        def step(self, action):
            return {}, 0.0, False, False, {}

    return session_module.KeyboardSession(Env())


def test_repeat_presses_within_the_debounce_window_are_dropped(monkeypatch):
    session = _keyboard_session(monkeypatch, [["a"], ["a"], ["b"]])

    assert list(session.presses()) == ["a"]
    assert list(session.presses()) == []  # Same key within the debounce window.
    assert list(session.presses()) == ["b"]  # A different key is accepted.


def test_presses_queued_between_episodes_do_not_leak(monkeypatch):
    session = _keyboard_session(monkeypatch, [["c"], ["a"]])

    session.reset()

    assert session.env.resets == 1
    assert list(session.presses()) == ["a"]  # The queued key was drained.


def test_every_keyboard_wrapper_shares_the_session(monkeypatch):
    from rlinf.envs.real.wrappers.episode import (
        KeyboardEvalControlWrapper,
        KeyboardRewardDoneMultiStageWrapper,
        KeyboardRewardDoneWrapper,
        KeyboardRLTPolicySwitchWrapper,
        KeyboardStartEndWrapper,
    )
    from rlinf.envs.real.wrappers.episode.session import KeyboardSession

    for wrapper in (
        KeyboardEvalControlWrapper,
        KeyboardRLTPolicySwitchWrapper,
        KeyboardStartEndWrapper,
        KeyboardRewardDoneWrapper,
        KeyboardRewardDoneMultiStageWrapper,
    ):
        assert issubclass(wrapper, KeyboardSession), wrapper.__name__


def test_episode_wrappers_report_through_the_logger():
    episode_dir = _ROOT / "rlinf" / "envs" / "real" / "episode"
    offenders = sorted(
        path.name
        for path in episode_dir.glob("*.py")
        if re.search(r"^\s*print\(", path.read_text(), re.M)
    )

    assert offenders == []


def test_euler_conversion_is_one_wrapper_for_any_arm_count():
    import numpy as np
    from gymnasium import spaces

    from rlinf.envs.real.wrappers.transforms import (
        DualQuat2EulerWrapper,
        Quat2EulerWrapper,
    )

    class Env(gym.Env):
        def __init__(self, dim):
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Dict(
                        {"tcp_pose": spaces.Box(-np.inf, np.inf, (dim,))}
                    )
                }
            )

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
    one = np.concatenate([np.array([1.0, 2.0, 3.0]), identity_quat])

    single = Quat2EulerWrapper(Env(7))
    dual = DualQuat2EulerWrapper(Env(14))

    assert single.observation_space["state"]["tcp_pose"].shape == (6,)
    assert dual.observation_space["state"]["tcp_pose"].shape == (12,)

    got = single.observation({"state": {"tcp_pose": one.copy()}})["state"]["tcp_pose"]
    assert np.allclose(got, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

    both = dual.observation({"state": {"tcp_pose": np.concatenate([one, one])}})
    assert np.allclose(both["state"]["tcp_pose"], [1.0, 2.0, 3.0, 0, 0, 0] * 2)


# Full wrapper stacks built through the production builders


def _dummy_franka(env_cls=None, **overrides):
    from rlinf.envs.real.franka.base import FrankaEnv

    cfg = {
        "is_dummy": True,
        "enable_camera_player": False,
        "step_frequency": 10000.0,
    }
    robot_info = overrides.pop(
        "robot_info", _robot_info(FrankaConfig(node_rank=0, camera_serials=["dummy"]))
    )
    cfg.update(overrides)
    return (env_cls or FrankaEnv)(
        override_cfg=cfg,
        worker_info=None,
        env_idx=0,
        robot_info=robot_info,
    )


def _chain(env):
    """Return wrapper class names from outermost to innermost."""
    names = []
    while hasattr(env, "env"):
        names.append(type(env).__name__)
        env = env.env
    return names


def test_wrapper_stack_converts_the_pose_it_hands_the_policy():
    from rlinf.envs.real.wrappers import build_stack

    env = _dummy_franka()
    raw, _ = env.reset()
    assert raw["state"]["tcp_pose"].shape == (7,)

    wrapped = build_stack(
        env, {"teleop": "none", "no_gripper": False, "use_relative_frame": True}
    )
    observation, _ = wrapped.reset()

    assert _chain(wrapped) == ["Quat2EulerWrapper", "RelativeFrame"]
    assert observation["state"]["tcp_pose"].shape == (6,)
    wrapped.close()


def test_no_teleop_device_leaves_no_intervention_in_the_stack():
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(),
        {"teleop": "none", "no_gripper": False, "use_relative_frame": False},
    )

    assert not any("Intervention" in name for name in _chain(wrapped))
    wrapped.close()


def test_no_gripper_narrows_the_action_the_policy_must_produce():
    from rlinf.envs.real.wrappers import build_stack

    env = _dummy_franka()
    assert env.action_space.shape == (7,)

    wrapped = build_stack(
        env,
        {"teleop": "none", "no_gripper": True, "use_relative_frame": False},
    )

    assert "GripperCloseEnv" in _chain(wrapped)
    assert wrapped.action_space.shape == (6,)
    wrapped.close()


def test_the_no_gripper_default_does_not_wrap_a_dexterous_hand():
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(
            robot_info=_robot_info(
                FrankaConfig(
                    node_rank=0,
                    camera_serials=["dummy"],
                    end_effector_type="ruiyan_hand",
                )
            ),
            hand_target_state=np.zeros(6),
            hand_reset_state=np.zeros(6),
        ),
        {"teleop": "none", "use_relative_frame": False},
    )

    assert "GripperCloseEnv" not in _chain(wrapped)
    assert wrapped.action_space.shape == (12,)
    wrapped.close()


def test_gim_arm_keeps_the_unwrapped_legacy_action_and_observation_schema():
    from rlinf.envs.real.gim_arm.base import GimArmEnv, GimArmEnvConfig
    from rlinf.envs.real.wrappers import build_stack

    env = GimArmEnv(
        config=GimArmEnvConfig(is_dummy=True),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )
    wrapped = build_stack(env, {})

    assert wrapped is env
    assert wrapped.action_space.shape == (7,)
    assert wrapped.observation_space["state"]["tcp_pose"].shape == (7,)
    wrapped.close()


def test_dual_franka_keeps_the_legacy_no_gripper_default():
    from rlinf.envs.real.franka.dual_franka_joint import DualFrankaJointEnv
    from rlinf.envs.real.wrappers import build_stack

    env = DualFrankaJointEnv(
        override_cfg={
            "is_dummy": True,
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        env_idx=0,
        robot_info=_robot_info(
            DualFrankaConfig(
                node_rank=0,
                base_camera_serials=["dummy"],
                left_camera_serials=[],
                right_camera_serials=[],
            )
        ),
    )

    with pytest.raises(NotImplementedError, match="no_gripper"):
        build_stack(env, {"teleop": "none"})
    env.close()


def test_a_task_env_runs_with_its_own_config():
    from rlinf.envs.real.franka.base import COMPLIANCE_DEFAULTS
    from rlinf.envs.real.franka.peg_insertion import PegInsertionEnv

    env = _dummy_franka(
        PegInsertionEnv, target_ee_pose=[0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
    )

    assert env.config.task_description == "peg and insertion"
    # Task-specific gains override shared defaults.
    assert set(env.config.compliance_param) == set(COMPLIANCE_DEFAULTS)
    assert env.config.compliance_param["translational_stiffness"] == 2000

    env.reset()
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    assert set(observation) == {"state", "frames"}
    assert isinstance(bool(terminated), bool)
    env.close()


def test_every_registered_task_builds_through_its_entry_point():
    import gymnasium as gym

    from rlinf.envs.real import RealWorldEnv  # noqa: F401  (registers the tasks)

    cfg = {
        "teleop": "none",
        "no_gripper": False,
        "use_relative_frame": False,
    }
    built = []
    for env_id in ("FrankaEnv-v1", "PegInsertionEnv-v1", "BottleEnv-v1"):
        env = gym.make(
            env_id,
            override_cfg={
                "is_dummy": True,
                "enable_camera_player": False,
                "step_frequency": 10000.0,
            },
            worker_info=None,
            env_idx=0,
            env_cfg=cfg,
            robot_info=_robot_info(FrankaConfig(node_rank=0, camera_serials=["dummy"])),
        )
        env.reset()
        env.close()
        built.append(env_id)

    assert built == ["FrankaEnv-v1", "PegInsertionEnv-v1", "BottleEnv-v1"]


def test_converted_pose_stays_inside_the_observation_space():
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(),
        {"teleop": "none", "no_gripper": False, "use_relative_frame": False},
    )
    observation, _ = wrapped.reset()

    assert (
        observation["state"]["tcp_pose"].dtype
        == wrapped.observation_space["state"]["tcp_pose"].dtype
    )
    assert wrapped.observation_space.contains(observation)
    wrapped.close()


# Multiple teleoperation devices


class _FakeInner:
    """Provide the environment attributes read by teleoperation builders."""

    def __init__(self, **config: Any) -> None:
        self.config = SimpleNamespace(**config)


def test_one_named_device_resolves_to_one_entry():
    assert resolve_teleop_devices({"teleop": "gello"}, supported=SINGLE_ARM) == [
        "gello"
    ]


def test_saying_nothing_resolves_to_no_entries():
    assert resolve_teleop_devices({}, supported=SINGLE_ARM) == []
    assert resolve_teleop_devices({"teleop": "none"}, supported=SINGLE_ARM) == []


def test_a_list_keeps_every_device_it_names():
    assert resolve_teleop_devices(
        {"teleop": ["spacemouse", "glove"]}, supported=("spacemouse", "glove")
    ) == ["spacemouse", "glove"]


def test_an_entry_carries_its_own_options():
    entries = resolve_teleop_devices(
        {"teleop": [{"gello_joint": {"port": "/dev/left", "drives": "left"}}]},
        supported=("gello_joint",),
    )

    assert entries == [{"gello_joint": {"port": "/dev/left", "drives": "left"}}]


def test_one_device_may_appear_twice_on_different_branches():
    entries = resolve_teleop_devices(
        {
            "teleop": [
                {"gello_joint": {"drives": "left"}},
                {"gello_joint": {"drives": "right"}},
            ]
        },
        supported=("gello_joint",),
    )

    assert [entry["gello_joint"]["drives"] for entry in entries] == ["left", "right"]


def test_a_listed_device_the_env_cannot_drive_is_refused():
    with pytest.raises(ValueError, match="Unsupported teleop device"):
        resolve_teleop_devices(
            {"teleop": ["spacemouse", "glove"]}, supported=("gello_joint", "pico")
        )


def test_a_list_supersedes_a_retired_key_underneath_it():
    with pytest.warns(DeprecationWarning, match="supersedes"):
        entries = resolve_teleop_devices(
            {"teleop": ["spacemouse"], "teleop_device": "none"},
            supported=SINGLE_ARM,
        )

    assert entries == ["spacemouse"]


def test_a_list_supersedes_a_retired_boolean():
    with pytest.warns(DeprecationWarning, match="supersedes"):
        entries = resolve_teleop_devices(
            {"teleop": ["spacemouse"], "use_pico": True}, supported=SINGLE_ARM
        )

    assert entries == ["spacemouse"]


def test_an_empty_list_is_refused():
    with pytest.raises(ValueError, match="'teleop' is empty"):
        resolve_teleop_devices({"teleop": []}, supported=SINGLE_ARM)


def test_none_cannot_share_the_list():
    with pytest.raises(ValueError, match="cannot share the list"):
        resolve_teleop_devices({"teleop": ["none", "spacemouse"]}, supported=SINGLE_ARM)


def test_none_alone_in_a_list_means_nobody_takes_over():
    assert resolve_teleop_devices({"teleop": ["none"]}, supported=SINGLE_ARM) == []


def test_a_two_key_entry_is_refused():
    with pytest.raises(ValueError, match="mapping of one name"):
        resolve_teleop_devices(
            {"teleop": [{"spacemouse": {}, "glove": {}}]},
            supported=("spacemouse", "glove"),
        )


def test_no_device_is_named_in_the_wrapper_stack():
    import inspect

    from rlinf.envs.real import wrappers

    source = inspect.getsource(wrappers)
    for device in ("spacemouse", "glove", "gello_joint", "pico"):
        assert device not in source, f"the wrapper stack still names {device!r}"


def test_a_leader_arm_reads_the_joint_convention_from_the_env():
    from rlinf.envs.real.wrappers.teleop.builder import EnvFacts
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import TeleopDevice

    facts = EnvFacts(
        layout={"left.arm": slice(0, 7), "left.end_effector": slice(7, 8)},
        kinds={
            "left.arm": ActionKind.JOINT_DELTA,
            "left.end_effector": ActionKind.GRIPPER,
        },
        joint_action_scale=0.25,
    )
    entry = TeleopDevice.named("gello_joint").from_config(
        {"left_gello_port": "/dev/left"}, {"drives": "left"}, facts
    )

    assert entry.device.use_delta is True
    assert entry.device.action_scale == 0.25


def test_an_entry_option_wins_over_the_env_default():
    from rlinf.envs.real.wrappers.teleop.builder import EnvFacts
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import TeleopDevice

    facts = EnvFacts(
        layout={"left.arm": slice(0, 7), "left.end_effector": slice(7, 8)},
        kinds={
            "left.arm": ActionKind.JOINT_DELTA,
            "left.end_effector": ActionKind.GRIPPER,
        },
        joint_action_scale=0.25,
    )
    entry = TeleopDevice.named("gello_joint").from_config(
        {"left_gello_port": "/dev/left"},
        {"drives": "left", "action_scale": 0.5},
        facts,
    )

    assert entry.device.action_scale == 0.5
    assert entry.device.use_delta is True


def test_a_failed_teleop_connect_leaves_no_device_open():
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup, TeleopPart

    log: list[str] = []

    class Device(TeleopPart):
        def __init__(self, tag, fail=False):
            self.tag, self.fail = tag, fail

        def _open(self):
            if self.fail:
                raise RuntimeError("cable unplugged")
            log.append(f"open:{self.tag}")
            return self.tag

        def _release(self, device):
            log.append(f"close:{self.tag}")

        PRODUCES = {
            "arm": ActionKind.CARTESIAN_DELTA,
            "end_effector": ActionKind.GRIPPER,
        }

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

        def action(self, reading, context):
            from rlinf.robotics.parts.teleop import TeleopAction

            return TeleopAction()

    kinds = {
        f"{side}.{part}": kind
        for side in ("left", "right")
        for part, kind in (
            ("arm", ActionKind.CARTESIAN_DELTA),
            ("end_effector", ActionKind.GRIPPER),
        )
    }
    group = TeleopGroup(
        [
            TeleopEntry(Device("first"), drives="left"),
            TeleopEntry(Device("second", fail=True), drives="right"),
        ],
        available=kinds,
    )

    with pytest.raises(RuntimeError, match="cable unplugged"):
        group.connect()

    assert log == ["open:first", "close:first"], (
        f"the device opened before the failure was left open: {log}"
    )


def test_the_glove_reads_the_key_the_shipped_configs_set():
    from rlinf.robotics.parts.teleop import TeleopDevice

    glove = TeleopDevice.named("glove")
    cfg = {
        "glove_config": {
            "left_port": "/dev/ttyACM7",
            "right_port": "/dev/ttyACM8",
            "frequency": 90,
            "config_file": "/etc/glove.json",
        }
    }

    device = glove.from_config(cfg, {}, None).device
    assert device._left_port == "/dev/ttyACM7"
    assert device._right_port == "/dev/ttyACM8"
    assert device._frequency == 90
    assert device._config_file == "/etc/glove.json"

    # Per-entry options override shared device configuration.
    overridden = glove.from_config(cfg, {"left_port": "/dev/override"}, None).device
    assert overridden._left_port == "/dev/override"

    # The documented default applies when the option is omitted.
    assert glove.from_config({}, {}, None).device._left_port == "/dev/ttyACM0"


def test_every_shipped_teleop_config_key_is_one_a_device_reads():
    import pathlib
    import re

    from rlinf.robotics.parts.teleop import TeleopDevice

    # Every device reads its own config, so scan the whole devices package.
    source = "".join(
        path.read_text()
        for path in pathlib.Path("rlinf/robotics/parts/teleop").glob("*.py")
    )
    read = set(re.findall(r'cfg\.get\(\s*f?"([a-z_{}]+)"', source))
    # A key built per side, like "{drives}_gello_port", covers both sides.
    read |= {
        f"{side}_{key.split('}_', 1)[1]}"
        for key in read
        if key.startswith("{")
        for side in ("left", "right")
    }

    configured: set[str] = set()
    for path in pathlib.Path("examples/embodiment/config").glob("*.yaml"):
        text = path.read_text()
        if "teleop:" not in text:
            continue
        configured |= {
            key
            for key in re.findall(r"^\s{4}([a-z_]+):", text, re.MULTILINE)
            if key.split("_")[0] in TeleopDevice.names()
        }

    unread = sorted(configured - read)
    assert not unread, f"configs set {unread}, which no teleop device reads"


def test_a_teleop_device_is_one_class_that_registers_itself():
    from rlinf.robotics.parts.teleop import TeleopDevice

    # The shipped devices are all registered. Asserting a superset rather than
    # an exact list keeps adding one from editing this test.
    assert set(TeleopDevice.names()) >= {
        "gello",
        "gello_joint",
        "glove",
        "pico",
        "so101_leader",
        "spacemouse",
    }

    # One class answers all three questions: hardware, mapping, and config.
    for name in TeleopDevice.names():
        device_cls = TeleopDevice.named(name)
        assert issubclass(device_cls, TeleopDevice)
        assert callable(device_cls._open)
        assert callable(device_cls.get_observation)
        assert callable(device_cls.action)
        assert callable(device_cls.from_config)
        assert callable(device_cls.streamer)
        assert device_cls.PRODUCES, f"{name} fills no action part"

    # Only a device that bypasses step overrides the streamer hook.
    quiet = [
        name
        for name in TeleopDevice.names()
        if "streamer" not in vars(TeleopDevice.named(name))
    ]
    assert "gello_joint" not in quiet, "the one device that streams must say so"
    assert set(quiet) >= {"gello", "glove", "pico", "so101_leader", "spacemouse"}

    with pytest.raises(ValueError, match="Unknown teleop device"):
        TeleopDevice.named("no_such_device")

    # Duplicate names fail deterministically instead of depending on import order.
    with pytest.raises(ValueError, match="already registered"):

        @TeleopDevice.register("pico")
        class Second(TeleopDevice):
            @classmethod
            def entry(cls, cfg, options, facts):
                raise AssertionError("never built")


def test_every_env_only_offers_teleop_devices_that_exist():
    from rlinf.envs.real.dosw1.base import DOSW1Env
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.franka.dual_base import DualFrankaEnv
    from rlinf.envs.real.xsquare.base import Turtle2Env
    from rlinf.robotics.parts.teleop import TeleopDevice

    known = set(TeleopDevice.names())
    for env_cls in (FrankaEnv, DualFrankaEnv, Turtle2Env, DOSW1Env):
        offered = set(getattr(env_cls, "TELEOP", ()))
        unknown = sorted(offered - known)
        assert not unknown, f"{env_cls.__name__} offers {unknown}, which do not exist"


def test_the_streamer_comes_from_the_registry_not_the_stack():
    from rlinf.envs.real.wrappers.teleop.builder import EnvFacts
    from rlinf.robotics.parts.teleop import TeleopDevice

    quiet = EnvFacts(layout={}, kinds={}, direct_stream=False)
    assert TeleopDevice.named("gello_joint").streamer({}, quiet, []) is None


# Environment action declarations


def _declared(cls, **attrs):
    """Return the action parts declared by an environment class."""
    return cls.action_parts(SimpleNamespace(**attrs))


def test_every_env_declares_parts_that_tile_its_action():
    from rlinf.envs.real.dosw1.base import DOSW1Env
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.gim_arm.base import GimArmEnv
    from rlinf.envs.real.xsquare.base import Turtle2Env

    cases = [
        (7, _declared(FrankaEnv, _is_hand=False)),
        (
            12,
            _declared(
                FrankaEnv, _is_hand=True, _ee_interface=SimpleNamespace(action_dim=6)
            ),
        ),
        (7, _declared(GimArmEnv)),
        (7, _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[1]))),
        (14, _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[0, 1]))),
        (14, _declared(DOSW1Env)),
    ]
    for width, parts in cases:
        assert sum(part.width for part in parts) == width


def test_a_two_armed_robot_names_both_arms():
    from rlinf.envs.real.xsquare.base import Turtle2Env

    parts = _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[0, 1]))

    assert [part.name for part in parts] == [
        "left.arm",
        "left.end_effector",
        "right.arm",
        "right.end_effector",
    ]


def test_two_arms_of_the_same_width_can_mean_different_things():
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.gim_arm.base import GimArmEnv
    from rlinf.robotics.actions import ActionKind

    franka_arm = _declared(FrankaEnv, _is_hand=False)[0]
    gim_arm = _declared(GimArmEnv)[0]

    assert franka_arm.width == gim_arm.width == 6
    assert franka_arm.kind is ActionKind.CARTESIAN_DELTA
    assert gim_arm.kind is ActionKind.JOINT_POSITION


def test_an_env_that_declares_nothing_cannot_be_teleoperated():
    from rlinf.envs.real.wrappers.teleop.layout import action_spec

    class Bare:
        unwrapped = None

        def get_wrapper_attr(self, name):
            raise AttributeError(name)

    Bare.unwrapped = Bare()
    with pytest.raises(AttributeError, match="does not declare action_parts"):
        action_spec(Bare())


def test_a_declaration_that_does_not_tile_the_action_is_refused():
    import gymnasium as gym

    from rlinf.envs.real.wrappers.teleop.layout import action_spec
    from rlinf.robotics.actions import ActionKind, ActionPart

    class Wrong:
        action_space = gym.spaces.Box(-1, 1, (7,), np.float32)

        def action_parts(self):
            return (ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),)

        def get_wrapper_attr(self, name):
            return getattr(self, name)

    env = Wrong()
    env.unwrapped = env
    with pytest.raises(ValueError, match="declares parts covering 6"):
        action_spec(env)


def test_a_device_that_means_something_else_is_refused():
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import SpaceMouse, TeleopEntry, TeleopGroup

    joint_arm = {
        "arm": ActionKind.JOINT_POSITION,
        "end_effector": ActionKind.GRIPPER,
    }
    # A SpaceMouse drives Cartesian deltas, so it cannot fill a joint arm.
    with pytest.raises(ValueError, match="mean different things"):
        TeleopGroup([TeleopEntry(SpaceMouse())], available=joint_arm)


# Teleoperation compatibility of shipped configurations


def _task_classes():
    """Return registered real-world task classes keyed by Gym ID.

    The robot packages are read from the list the env package itself loads,
    so a newly added robot is covered here without editing this helper.
    """
    import importlib

    import rlinf.envs.real as real

    classes = {}
    for name in real._ROBOT_PACKAGES:
        module = importlib.import_module(name, real.__name__)
        classes.update(getattr(module, "TASKS", {}))
    return classes


def _env_configs():
    """Return each shipped environment config and its Gym ID.

    A run config layers an ``env/<name>`` file into ``env.train`` or
    ``env.eval`` and may override the device there, so both halves are read.
    """
    import yaml

    roots = [_ROOT / "examples", _ROOT / "evaluations"]
    env_files = {}
    for root in roots:
        for path in root.rglob("env/*.yaml"):
            try:
                doc = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:
                continue
            env_id = (doc.get("init_params") or {}).get("id")
            # Simulated environments have no real-world teleoperation contract.
            if env_id and str(doc.get("env_type", "")) in ("real", "realworld"):
                env_files[path.stem] = (path, doc, str(env_id))

    for path, doc, env_id in env_files.values():
        yield path, doc, env_id

    pattern = re.compile(r"env/([\w-]+)@env\.(train|eval)")
    for root in roots:
        for path in root.rglob("*.yaml"):
            if path.parent.name == "env":
                continue
            try:
                doc = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:
                continue
            for entry in doc.get("defaults") or []:
                match = pattern.search(str(entry))
                if not match:
                    continue
                base = env_files.get(match.group(1))
                if base is None:
                    continue
                section = ((doc.get("env") or {}).get(match.group(2))) or {}
                if any(key in section for key in ("teleop", "teleop_device")):
                    yield path, section, base[2]


def test_shipped_configs_name_devices_their_env_can_drive():
    classes = _task_classes()
    offenders = []
    for path, section, env_id in _env_configs():
        env_cls = classes.get(env_id)
        if env_cls is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                resolve_teleop_devices(
                    section,
                    supported=getattr(env_cls, "TELEOP", ()),
                    default=getattr(env_cls, "TELEOP_DEFAULT", NO_DEVICE),
                )
        except ValueError as error:
            offenders.append(f"{path.relative_to(_ROOT)} ({env_id}): {error}")

    assert offenders == []


def test_every_shipped_config_names_a_registered_task():
    classes = _task_classes()
    unknown = sorted(
        {
            env_id
            for _, _, env_id in _env_configs()
            if env_id not in classes and env_id.endswith("-v1")
        }
    )

    assert unknown == []


def test_the_retired_env_type_still_resolves_and_warns():
    from rlinf.envs import SupportedEnvType

    assert SupportedEnvType("real") is SupportedEnvType.REAL
    with pytest.warns(DeprecationWarning, match="'realworld' is retired"):
        assert SupportedEnvType("realworld") is SupportedEnvType.REAL


def test_no_worker_compares_the_env_type_to_a_bare_string():
    offenders = []
    for path in (_ROOT / "rlinf" / "workers").rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'env_type\s*[!=]=\s*["\']', line):
                offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == []


def _resolved(doc, value):
    """Resolve a ``${a.b.c}`` interpolation within one document."""
    if not isinstance(value, str) or not value.startswith("${"):
        return value
    node = doc
    for key in value[2:-1].split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return _resolved(doc, node)


def _merged_sections(path, doc):
    """Yield each environment section after Hydra-style composition.

    A section layers its own keys over the ``env/<name>`` file its ``defaults``
    names, and ``override_cfg`` merges key by key rather than wholesale. What
    the section leaves out it inherits, which is how a task ends up on a
    different robot than the one it trained on.
    """
    import yaml

    pattern = re.compile(r"env/([\w-]+)@env\.(train|eval)")
    for entry in doc.get("defaults") or []:
        match = pattern.search(str(entry))
        if not match:
            continue
        base_path = path.parent / "env" / f"{match.group(1)}.yaml"
        if not base_path.exists():
            continue
        try:
            base = yaml.safe_load(base_path.read_text()) or {}
        except yaml.YAMLError:
            continue
        section = ((doc.get("env") or {}).get(match.group(2))) or {}
        merged = {**base, **section}
        merged["override_cfg"] = {
            **(base.get("override_cfg") or {}),
            **(section.get("override_cfg") or {}),
        }
        yield match.group(2), merged, str((base.get("init_params") or {}).get("id"))


def test_shipped_configs_give_the_policy_the_action_width_it_expects():
    import yaml

    from rlinf.envs.real.franka.base import FrankaEnv

    classes = _task_classes()
    offenders = []
    for path in (_ROOT / "examples").rglob("*.yaml"):
        if path.parent.name == "env":
            continue
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        action_dim = _resolved(
            doc, ((doc.get("rollout") or {}).get("model") or {}).get("action_dim")
        )
        if not isinstance(action_dim, int):
            continue
        for name, section, env_id in _merged_sections(path, doc):
            # This legacy action layout applies only to single-arm Franka.
            env_cls = classes.get(env_id)
            if env_cls is None or not issubclass(env_cls, FrankaEnv):
                continue
            hardware_configs = [
                config
                for group in doc.get("cluster", {}).get("node_groups", [])
                if group.get("hardware", {}).get("type") == "Franka"
                for config in group["hardware"].get("configs", [])
            ]
            end_effector = str(
                hardware_configs[0].get("end_effector_type", "franka_gripper")
                if hardware_configs
                else "franka_gripper"
            )
            from rlinf.robotics.robots.franka import FrankaRobot

            hardware = hardware_configs[0] if hardware_configs else {}
            driver = FrankaRobot.end_effector_class(
                backend=hardware.get("backend"),
                gripper_type=hardware.get("gripper_type"),
                end_effector_type=hardware.get("end_effector_type"),
            )
            parts = FrankaEnv.action_parts(
                SimpleNamespace(_is_hand=driver.is_hand, _ee_interface=driver)
            )
            width = sum(part.width for part in parts)
            if width != action_dim:
                offenders.append(
                    f"{path.relative_to(_ROOT)} env.{name}: {end_effector} gives "
                    f"{width}, model wants {action_dim}"
                )

    assert offenders == []


def test_direct_stream_gello_opens_one_reader_per_port(monkeypatch):
    """The streamer reuses the group's devices instead of reopening ports."""
    import numpy as np

    from rlinf.envs.real.wrappers.teleop import builder as builder_module
    from rlinf.robotics.parts.teleop import (
        GelloJoint,
        TeleopDevice,
        TeleopEntry,
        gello_joint,
    )

    opened = []

    class FakeExpert:
        def __init__(self, port):
            opened.append(port)
            self.ready = True

        def get_action(self):
            return np.zeros(7), np.zeros(1)

        def close(self):
            pass

    # The reader lives in the device module now, so the class is what a test
    # replaces rather than a separate module.
    monkeypatch.setattr(gello_joint, "GelloJointExpert", FakeExpert)

    arms = {side: GelloJoint(port=f"/dev/{side}") for side in ("left", "right")}
    entries = [TeleopEntry(arm, drives=side) for side, arm in arms.items()]
    try:
        for arm in arms.values():
            arm.connect()
        assert opened == ["/dev/left", "/dev/right"]

        facts = builder_module.EnvFacts(layout={}, kinds={}, direct_stream=True)
        streamer = TeleopDevice.named("gello_joint").streamer({}, facts, entries)

        assert opened == ["/dev/left", "/dev/right"], (
            f"the streamer opened more readers: {opened}"
        )
        assert streamer.left_arm is arms["left"]
        assert streamer.right_arm is arms["right"]
    finally:
        for arm in arms.values():
            arm.disconnect()


def _so101_env(robot_info=None, **overrides):
    """Build an SO-101 reach env against the faked lerobot SDK."""
    from rlinf.envs.real.so101 import SO101ReachEnv

    settings = {
        "target_joint_qpos": [0.0] * 5,
        # Do not pace the test at the real 10 Hz control rate.
        "step_frequency": 1000.0,
        # Headless: there is no display to show frames on.
        "enable_camera_player": False,
    }
    settings.update(overrides)
    return SO101ReachEnv(
        settings,
        env_idx=0,
        robot_info=robot_info
        or _robot_info(
            SO101Config(
                node_rank=0, serial_port="/dev/mock-so101", calibration_id="bench"
            )
        ),
    )


def test_so101_env_runs_a_whole_episode_against_a_faked_arm():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env(max_num_steps=3)
        try:
            observation, _ = env.reset()
            assert set(observation["state"]) == {
                "arm_joint_position",
                "gripper_position",
            }
            assert observation["state"]["arm_joint_position"].shape == (5,)

            # Five joints plus one gripper opening, which is 0..1 not -1..1.
            assert env.action_space.shape == (6,)
            assert env.action_space.low[5] == pytest.approx(0.0)
            assert env.action_space.high[5] == pytest.approx(1.0)
            assert [(part.name, part.width) for part in env.action_parts()] == [
                ("arm", 5),
                ("end_effector", 1),
            ]

            for _ in range(3):
                _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            assert truncated, "the episode should truncate at max_num_steps"
        finally:
            env.close()


def test_so101_env_keeps_its_action_in_radians_across_a_degree_driver():
    """The env speaks radians; only the driver may speak lerobot's units.

    The arm runs in a scheduler worker, so the fake servo bus is not reachable
    from here. Read the action back instead: the driver converts on the way
    out and again on the way in, so a stray conversion anywhere on the env's
    side of the boundary would come back scaled.
    ``test_so101_commands_are_converted_back_to_degrees`` pins the degrees
    themselves, against an arm declared in this process.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env()
        try:
            env.reset()
            observation, *_ = env.step(
                np.array([np.pi / 2, 0, 0, 0, 0, 0.25], dtype=np.float32)
            )

            state = observation["state"]
            assert state["arm_joint_position"][0] == pytest.approx(np.pi / 2)
            assert state["gripper_position"][0] == pytest.approx(0.25)
        finally:
            env.close()


def test_so101_env_scores_the_distance_to_the_target_configuration():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env(target_joint_qpos=[0.0] * 5, reward_threshold=0.05)
        try:
            env.reset()
            _, reward, terminated, _, _ = env.step(np.zeros(6, dtype=np.float32))
            assert reward == pytest.approx(1.0)
            assert terminated

            # The fake arm holds whatever was last written, so a command away
            # from the target is measured as a miss on the next read.
            _, reward, terminated, _, _ = env.step(
                np.array([1.0, 0, 0, 0, 0, 0.0], dtype=np.float32)
            )
            assert reward == pytest.approx(0.0)
            assert not terminated
        finally:
            env.close()


def test_so101_env_clips_an_action_to_the_joint_limits():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env(joint_limit_low=[-0.5] * 5, joint_limit_high=[0.5] * 5)
        try:
            env.reset()
            observation, *_ = env.step(
                np.array([10.0, -10.0, 0, 0, 0, 2.0], dtype=np.float32)
            )

            state = observation["state"]
            assert state["arm_joint_position"][0] == pytest.approx(0.5)
            assert state["arm_joint_position"][1] == pytest.approx(-0.5)
            # The gripper is clipped into 0..1 before it is scaled.
            assert state["gripper_position"][0] == pytest.approx(1.0)
        finally:
            env.close()


def test_so101_env_runs_without_hardware_when_dummy():
    """A dummy env samples its own space, so it needs no lerobot at all."""
    from rlinf.envs.real.so101 import SO101ReachEnv

    env = SO101ReachEnv({"is_dummy": True, "target_joint_qpos": [0.0] * 5})
    try:
        observation, _ = env.reset()
        assert observation in env.observation_space
        assert env.robot is None
        env.step(np.zeros(6, dtype=np.float32))
    finally:
        env.close()


def test_so101_reach_refuses_a_target_that_is_not_five_joints():
    from rlinf.envs.real.so101 import SO101ReachEnv

    with pytest.raises(ValueError, match="5 arm joints"):
        SO101ReachEnv({"is_dummy": True, "target_joint_qpos": [0.0] * 6})


def test_so101_task_is_registered_with_gymnasium():
    import gymnasium as gym

    import rlinf.envs.real as real

    real.load_tasks()
    assert "SO101ReachEnv-v1" in gym.registry


def test_so101_env_resizes_camera_frames_to_the_declared_shape():
    """A camera delivers its native resolution; the space fixes one size."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env(
            robot_info=_robot_info(
                SO101Config(
                    node_rank=0,
                    serial_port="/dev/mock-so101",
                    camera_serials=["MOCK0001"],
                    camera_type="realsense",
                )
            )
        )
        try:
            observation, _ = env.reset()
            frame = observation["frames"]["wrist_1"]
            assert frame.shape == (128, 128, 3)
            assert frame.dtype == np.uint8
            # An observation outside its own space fails Gymnasium's checker.
            assert observation in env.observation_space
        finally:
            env.close()


def test_so101_env_omits_frames_entirely_when_no_camera_is_configured():
    """Gymnasium rejects an empty Dict space, so the key is dropped instead."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env()
        try:
            observation, _ = env.reset()
            assert "frames" not in env.observation_space.spaces
            assert "frames" not in observation
            assert observation in env.observation_space
        finally:
            env.close()


def _piper_env(robot_info=None, **overrides):
    """Build a Piper reach env against the faked pyAgxArm SDK."""
    from rlinf.envs.real.piper import PiperReachEnv

    settings = {
        "target_joint_qpos": [0.0] * 6,
        # Do not pace the test at the real 10 Hz control rate.
        "step_frequency": 1000.0,
        # Headless: there is no display to show frames on.
        "enable_camera_player": False,
    }
    settings.update(overrides)
    return PiperReachEnv(
        settings,
        env_idx=0,
        robot_info=robot_info or _robot_info(PiperConfig(node_rank=0)),
    )


def test_a_hardware_type_resolves_without_importing_the_robot_package():
    """The lookup loads the modules that register hardware configs.

    In a subprocess, because this one has already imported them.
    """
    import pathlib
    import subprocess
    import sys

    program = (
        "import sys\n"
        "from rlinf.scheduler.cluster.config import ClusterConfig\n"
        "from rlinf.scheduler.hardware import NodeHardwareConfig\n"
        "assert not [m for m in sys.modules if m.startswith('rlinf.robotics')]\n"
        "assert not NodeHardwareConfig._hardware_config_registry\n"
        "from omegaconf import OmegaConf\n"
        "cfg = OmegaConf.create({'num_nodes': 1, 'component_placement':"
        " {'env': {'node_group': 'f', 'placement': 0}}, 'node_groups':"
        " [{'label': 'f', 'node_ranks': 0, 'hardware': {'type': 'Franka',"
        " 'configs': [{'robot_ip': '0.0.0.0', 'node_rank': 0}]}}]})\n"
        "hw = ClusterConfig.from_dict_cfg(cfg).node_groups[0].hardware\n"
        "print(type(hw.configs[0]).__name__)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[2]),
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "FrankaConfig", done.stdout


def test_entry_points_reach_realworldenv_through_its_package():
    """``rlinf.envs.real`` registers its Gymnasium tasks from ``__getattr__``.

    Reaching ``RealWorldEnv`` at ``rlinf.envs.real.env`` bypasses that.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]
    offenders = []
    for path in list((root / "examples").rglob("*.py")) + list(
        (root / "toolkits").rglob("*.py")
    ):
        if re.search(r"^from rlinf\.envs\.real\.env import", path.read_text(), re.M):
            offenders.append(str(path.relative_to(root)))

    assert offenders == [], (
        "import RealWorldEnv from rlinf.envs.real, not rlinf.envs.real.env: "
        f"{offenders}"
    )


def test_piper_env_runs_a_whole_episode_against_a_faked_arm():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env(max_num_steps=3)
        try:
            observation, _ = env.reset()
            assert set(observation["state"]) == {
                "arm_joint_position",
                "tcp_pose",
                "gripper_position",
            }
            assert observation["state"]["arm_joint_position"].shape == (6,)
            assert observation["state"]["tcp_pose"].shape == (7,)

            # The gripper axis is 0..1, not -1..1.
            assert env.action_space.shape == (7,)
            assert env.action_space.low[6] == pytest.approx(0.0)
            assert env.action_space.high[6] == pytest.approx(1.0)
            assert [(part.name, part.width) for part in env.action_parts()] == [
                ("arm", 6),
                ("end_effector", 1),
            ]

            for _ in range(3):
                _, _, _, truncated, _ = env.step(np.zeros(7, dtype=np.float32))
            assert truncated, "the episode should truncate at max_num_steps"
        finally:
            env.close()


def test_piper_env_commands_reach_the_arm_in_radians():
    """pyAgxArm takes radians, so nothing on this path rescales them.

    The arm runs in a scheduler worker, so the fake CAN session is not
    reachable from here; the action is read back instead.
    ``test_piper_commands_go_out_in_radians_and_are_held_to_the_travel`` and
    ``test_piper_gripper_rides_the_arm_connection`` pin what reaches the SDK,
    against an arm declared in this process.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env()
        try:
            env.reset()
            observation, *_ = env.step(
                np.array([0.5, 1.0, -1.0, 0, 0, 0, 0.25], dtype=np.float32)
            )

            state = observation["state"]
            assert state["arm_joint_position"] == pytest.approx(
                [0.5, 1.0, -1.0, 0.0, 0.0, 0.0]
            )
            assert state["gripper_position"][0] == pytest.approx(0.25)
        finally:
            env.close()


def test_piper_env_clips_an_action_to_the_joint_limits():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env(joint_limit_low=[-0.5] * 6, joint_limit_high=[0.5] * 6)
        try:
            env.reset()
            observation, *_ = env.step(
                np.array([10.0, -10.0, 0, 0, 0, 0, 2.0], dtype=np.float32)
            )

            joints = observation["state"]["arm_joint_position"]
            assert joints[0] == pytest.approx(0.5)
            # The env bound is looser than the arm's, so the arm holds joint 2
            # at the travel its firmware accepts.
            assert joints[1] == pytest.approx(0.0)
            assert observation["state"]["gripper_position"][0] == pytest.approx(1.0)
        finally:
            env.close()


def test_piper_env_scores_the_distance_to_the_target_configuration():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env(target_joint_qpos=[0.0] * 6, reward_threshold=0.05)
        try:
            env.reset()
            _, reward, terminated, _, _ = env.step(np.zeros(7, dtype=np.float32))
            assert reward == pytest.approx(1.0)
            assert terminated

            # The fake arm holds whatever was last written, so a command away
            # from the target is measured as a miss on the next read.
            _, reward, terminated, _, _ = env.step(
                np.array([1.0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)
            )
            assert reward == pytest.approx(0.0)
            assert not terminated
        finally:
            env.close()


def test_piper_env_runs_without_hardware_when_dummy():
    """A dummy env samples its own space, so it needs no pyAgxArm at all."""
    from rlinf.envs.real.piper import PiperReachEnv

    env = PiperReachEnv({"is_dummy": True, "target_joint_qpos": [0.0] * 6})
    try:
        observation, _ = env.reset()
        assert observation in env.observation_space
        assert env.robot is None
        env.step(np.zeros(7, dtype=np.float32))
    finally:
        env.close()


def test_piper_env_without_a_gripper_has_a_six_wide_action():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env(
            robot_info=_robot_info(PiperConfig(node_rank=0, with_gripper=False))
        )
        try:
            observation, _ = env.reset()
            assert "gripper_position" not in observation["state"]
            assert [part.name for part in env.action_parts()] == ["arm"]
            assert env.action_space.shape == (6,)
            observation, *_ = env.step(np.zeros(6, dtype=np.float32))
            assert observation in env.observation_space
        finally:
            env.close()


def test_piper_reach_refuses_a_target_that_is_not_six_joints():
    from rlinf.envs.real.piper import PiperReachEnv

    with pytest.raises(ValueError, match="6 arm joints"):
        PiperReachEnv({"is_dummy": True, "target_joint_qpos": [0.0] * 5})


def test_piper_task_is_registered_with_gymnasium():
    import gymnasium as gym

    import rlinf.envs.real as real

    real.load_tasks()
    assert "PiperReachEnv-v1" in gym.registry


def test_piper_env_resizes_camera_frames_to_the_declared_shape():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env(
            robot_info=_robot_info(
                PiperConfig(
                    node_rank=0, camera_serials=["MOCK0001"], camera_type="realsense"
                )
            )
        )
        try:
            observation, _ = env.reset()
            frame = observation["frames"]["wrist_1"]
            assert frame.shape == (128, 128, 3)
            assert frame.dtype == np.uint8
            assert observation in env.observation_space
        finally:
            env.close()


def test_so101_leader_reports_radians_and_a_zero_to_one_grip():
    """The leader is the same servos as the follower, so it converts alike."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.teleop import SO101Leader

        leader = SO101Leader(port="/dev/mock-leader", calibration_id="lead")
        leader.connect()
        try:
            leader._device.positions.update(
                {"shoulder_pan.pos": 90.0, "gripper.pos": 40.0}
            )
            reading = leader.get_observation()
            assert reading["joint_position"][0] == pytest.approx(np.pi / 2)
            assert reading["grip"][0] == pytest.approx(0.4)
            # It never reaches lerobot's interactive calibration.
            assert leader._device.calibrate_calls == 0
        finally:
            leader.disconnect()


def test_so101_leader_releases_the_servo_bus():
    """lerobot spells release ``disconnect``, which the base class does not try.

    The device overrides ``_release`` for exactly that reason. Without this the
    serial port would leak on every disconnect and nothing would say so.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.teleop import SO101Leader

        leader = SO101Leader(port="/dev/mock-leader")
        leader.connect()
        handle = leader._device
        assert handle.is_connected
        leader.disconnect()
        assert not handle.is_connected


def test_so101_leader_refuses_an_uncalibrated_arm():
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.teleop import SO101Leader

        fake = made["lerobot.teleoperators.so_leader"].SO101Leader
        fake.calibrated = False
        try:
            leader = SO101Leader(port="/dev/mock-leader")
            with pytest.raises(RuntimeError, match="no calibration") as raised:
                leader.connect()
            # The refusal has to be actionable: the operator should be able to
            # copy a command out of it rather than go looking for one.
            assert "--calibrate" in str(raised.value)
        finally:
            fake.calibrated = True


def test_so101_leader_calibrates_when_the_caller_asks():
    """The standalone entry point has a terminal, so it may opt in."""
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.teleop import SO101Leader

        fake = made["lerobot.teleoperators.so_leader"].SO101Leader
        fake.calibrated = False
        try:
            leader = SO101Leader(port="/dev/mock-leader", calibrate=True)
            # The same arm that refuses above now opens, because calibration
            # is allowed to run.
            leader.connect()
            assert leader.is_connected
            leader.disconnect()
        finally:
            fake.calibrated = True


def test_so101_leader_only_drives_once_the_operator_moves_it():
    """A leader resting in its holder must not take control from the policy."""
    from rlinf.robotics.parts.teleop import SO101Leader

    binding = SO101Leader(port="/dev/unused", movement_epsilon=0.01)
    at_rest = {"joint_position": np.zeros(5), "grip": np.array([0.0])}
    context = {"joint_positions": np.zeros((1, 5))}

    assert not binding.action(at_rest, context).driving

    moved = {"joint_position": np.array([0.5, 0, 0, 0, 0]), "grip": np.array([0.7])}
    sample = binding.action(moved, context)
    assert sample.driving
    # The leader's pose is the follower's target, joint for joint.
    assert sample.parts["arm"] == pytest.approx(moved["joint_position"])
    # And the grip stays on the 0..1 axis the SO-101 env opens over.
    assert sample.parts["end_effector"][0] == pytest.approx(0.7)


def test_so101_env_is_driven_by_its_leader():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        import gymnasium as gym

        import rlinf.envs.real as real

        real.load_tasks()
        env = gym.make(
            "SO101ReachEnv-v1",
            override_cfg={
                "step_frequency": 1000.0,
                "enable_camera_player": False,
            },
            worker_info=None,
            env_idx=0,
            env_cfg={
                "teleop": [{"so101_leader": {"port": "/dev/mock-leader"}}],
                "no_gripper": False,
                "use_relative_frame": False,
            },
            robot_info=_robot_info(
                SO101Config(
                    node_rank=0, serial_port="/dev/mock-so101", calibration_id="bench"
                )
            ),
        )
        try:
            env.reset()
            policy_action = np.zeros(6, dtype=np.float32)

            observation, *_ = env.step(policy_action)
            assert observation["state"]["arm_joint_position"] == pytest.approx(
                np.zeros(5)
            )

            leader = list(env.get_wrapper_attr("device").group.devices)[0]
            leader._device.positions.update(
                {"shoulder_pan.pos": 45.0, "gripper.pos": 80.0}
            )
            observation, *_ = env.step(policy_action)
            assert observation["state"]["arm_joint_position"][0] == pytest.approx(
                np.deg2rad(45.0), abs=1e-4
            )
            assert observation["state"]["gripper_position"][0] == pytest.approx(0.8)
        finally:
            env.close()


@pytest.mark.parametrize(
    "path",
    [
        "evaluations/realworld/realworld_pnp_eval.yaml",
        "evaluations/realworld/realworld_pnp_eval_dreamzero.yaml",
        "evaluations/realworld/realworld_pnp_eval_pi05_sft_RTC.yaml",
        "examples/embodiment/config/realworld_pnp_dagger_openpi.yaml",
        "examples/embodiment/config/realworld_pnp_rlpd_cnn_async.yaml",
        "examples/reward/config/realworld_teleop.yaml",
    ],
)
def test_pnp_examples_discover_cameras_without_serial_placeholders(path):
    import yaml
    from robot_mocks import mocked_sdks

    from rlinf.robotics import FrankaConfig
    from rlinf.robotics.parts.cameras import BaseCamera

    root = Path(__file__).resolve().parents[2]
    doc = yaml.safe_load((root / path).read_text())
    entries = [
        config
        for group in doc["cluster"]["node_groups"]
        if group.get("hardware", {}).get("type") == "Franka"
        for config in group["hardware"]["configs"]
    ]
    assert entries
    with mocked_sdks():
        discovered = sorted(BaseCamera.backend("realsense").discover())
        assert discovered
        for entry in entries:
            assert "camera_serials" not in entry
            config = FrankaConfig(**entry)
            resources = RobotDiscovery.registry["Franka"].discovery_cls.enumerate(
                config.node_rank, [config]
            )
            assert resources.infos[0].config.camera_serials == discovered


def test_shipped_realworld_task_overrides_contain_no_hardware_fields():
    import yaml

    load_tasks()
    hardware_fields = {
        field.name
        for registration in RobotDiscovery.registry.values()
        for field in dataclasses.fields(registration.config_cls)
    }
    root = Path(__file__).resolve().parents[2]
    offenders = []

    def check(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "override_cfg" and isinstance(value, dict):
                    overlap = hardware_fields.intersection(value)
                    if overlap:
                        offenders.append((str(path.relative_to(root)), sorted(overlap)))
                check(value, path)
        elif isinstance(node, list):
            for value in node:
                check(value, path)

    for directory in (
        "examples",
        "evaluations",
        "tests",
    ):
        for path in (root / directory).rglob("*.yaml"):
            check(yaml.safe_load(path.read_text()), path)
    assert offenders == []


@pytest.mark.parametrize(
    "robot_type,env_id,hardware,frames,action_width",
    [
        (
            "Franka",
            "FrankaEnv-v1",
            {"camera_serials": ["MOCK0001"], "end_effector_type": "ruiyan_hand"},
            ["wrist_1"],
            12,
        ),
        (
            "DualFranka",
            "DualFrankaJointEnv-v1",
            {
                "base_camera_serials": ["MOCK0001"],
                "left_camera_serials": ["MOCK0002"],
                "right_camera_serials": [],
            },
            ["base_0_rgb", "left_wrist_0_rgb"],
            16,
        ),
        (
            "SO101",
            "SO101ReachEnv-v1",
            {
                "serial_port": "/dev/bench",
                "calibration_id": "bench",
                "camera_serials": ["MOCK0001", "MOCK0002"],
            },
            ["wrist_1", "wrist_2"],
            6,
        ),
        (
            "Piper",
            "PiperReachEnv-v1",
            {"model": "piper_h", "with_gripper": False, "camera_serials": []},
            [],
            6,
        ),
        (
            "GimArm",
            "GimArmPegInsertionEnv-v1",
            {
                "arm_variant": "gim_arm",
                "enable_gripper": False,
                "camera_serials": ["MOCK0001"],
            },
            ["wrist_1"],
            7,
        ),
        (
            "DOSW1",
            "DOSW1PickEnv-v1",
            {"robot_url": "bench", "camera_serials": ["MOCK0001"]},
            ["cam_front"],
            14,
        ),
        ("Turtle2", "ButtonEnv-v1", {"camera_ids": [0, 2]}, ["wrist_1", "wrist_2"], 7),
    ],
)
def test_task_schema_uses_enumerated_hardware_without_changing_it(
    robot_type, env_id, hardware, frames, action_width
):
    from robot_mocks import mocked_sdks

    load_tasks()
    registration = RobotDiscovery.registry[robot_type]
    config = registration.config_cls(node_rank=0, **hardware)
    with mocked_sdks():
        resources = registration.discovery_cls.enumerate(0, [config])
        info = resources.infos[0]
        before = pickle.dumps(info)
        env = gym.make(
            env_id,
            override_cfg={"is_dummy": True, "enable_camera_player": False}
            if robot_type != "Turtle2"
            else {"is_dummy": True},
            worker_info=None,
            robot_info=info,
            env_idx=0,
            env_cfg={
                "teleop": "none",
                "no_gripper": False,
                "use_relative_frame": False,
            },
        )
        try:
            observation, _ = env.reset(seed=3)
            assert sorted(observation.get("frames", {})) == sorted(frames)
            assert env.action_space.shape == (action_width,)
            assert pickle.dumps(info) == before
            assert not set(hardware) & {
                field.name for field in dataclasses.fields(env.unwrapped.config)
            }
            if robot_type == "Piper":
                assert info.model == "Piper"
                assert info.config.model == "piper_h"
        finally:
            env.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("port", "/dev/wrong-arm"),
        ("serial_port", "/dev/wrong-arm"),
        ("camera_serials", ["wrong-camera"]),
        ("calibration_id", "wrong-id"),
    ],
)
def test_task_overrides_cannot_redirect_the_allocated_robot(field, value):
    from rlinf.envs.real.so101 import SO101ReachEnv

    with pytest.raises(TypeError, match=field):
        SO101ReachEnv({"is_dummy": True, field: value})


def test_env_rejects_wrong_hardware_before_opening_an_arm(monkeypatch):
    from rlinf.envs.real.so101 import SO101ReachEnv
    from rlinf.robotics import PiperConfig, RobotInfo, SO101Robot

    build = Mock()
    monkeypatch.setattr(SO101Robot, "build", build)
    info = RobotInfo(type="Piper", model="Piper", config=PiperConfig(node_rank=0))
    with pytest.raises(TypeError, match="Expected SO101Config"):
        SO101ReachEnv({}, robot_info=info)
    build.assert_not_called()


def test_real_env_requires_a_robot_descriptor(monkeypatch):
    from rlinf.envs.real.so101 import SO101ReachEnv
    from rlinf.robotics import SO101Robot

    build = Mock()
    monkeypatch.setattr(SO101Robot, "build", build)
    with pytest.raises(ValueError, match="Supply robot_info"):
        SO101ReachEnv({})
    build.assert_not_called()


@pytest.mark.parametrize("controller_node_rank", [None, 7])
def test_franka_preserves_hardware_and_placement_at_construction(
    monkeypatch, controller_node_rank
):
    from rlinf.envs.real.franka import FrankaEnv
    from rlinf.robotics import FrankaConfig, FrankaRobot, RobotInfo

    config = FrankaConfig(
        node_rank=3,
        robot_ip="bench",
        controller_node_rank=controller_node_rank,
        camera_node_rank=5,
        camera_type="zed",
        camera_serials=["camera"],
        end_effector_type="ruiyan_hand",
        end_effector_config={"port": "/dev/hand"},
    )
    info = RobotInfo(type="Franka", model="Franka", config=config)
    before = pickle.dumps(info)
    build = Mock(side_effect=RuntimeError("stop before opening hardware"))
    monkeypatch.setattr(FrankaRobot, "build", build)
    with pytest.raises(RuntimeError, match="stop before opening hardware"):
        FrankaEnv(
            {},
            robot_info=info,
            worker_info=SimpleNamespace(cluster_node_rank=3, rank=2),
            env_idx=4,
        )
    kwargs = build.call_args.kwargs
    assert kwargs["robot_ip"] == "bench"
    assert kwargs["node_rank"] == (3 if controller_node_rank is None else 7)
    assert kwargs["camera_node_rank"] == 5
    assert kwargs["worker_rank"] == 2
    assert kwargs["env_idx"] == 4
    assert kwargs["end_effector_type"] == "ruiyan_hand"
    assert kwargs["end_effector_config"] == {"port": "/dev/hand"}
    camera = kwargs["cameras"]["wrist_1"]
    assert camera.serial_number == "camera"
    assert camera.camera_type == "zed"
    assert pickle.dumps(info) == before


@pytest.mark.parametrize("has_robot", [False, True])
def test_worker_passes_robot_descriptors_and_allows_dummy_cpu_placement(has_robot):
    from omegaconf import OmegaConf

    from rlinf.envs.real import RealWorldEnv
    from rlinf.robotics import RobotInfo, SO101Config
    from rlinf.scheduler.hardware import HardwareInfo

    allocated = (
        RobotInfo(
            type="SO101",
            model="SO101",
            config=SO101Config(node_rank=0, camera_serials=["camera"]),
        )
        if has_robot
        else HardwareInfo(type="CPU", model="CPU")
    )
    wrapper = RealWorldEnv.__new__(RealWorldEnv)
    wrapper.worker_info = SimpleNamespace(
        hardware_infos=[allocated], cluster_node_rank=0, rank=0
    )
    wrapper.override_cfg = {"is_dummy": True, "enable_camera_player": False}
    wrapper.cfg = OmegaConf.create(
        {"init_params": {"id": "SO101ReachEnv-v1"}, "teleop": "none"}
    )
    env = wrapper._create_env(0)
    try:
        observation, _ = env.reset()
        assert bool(observation.get("frames")) is has_robot
        assert env.unwrapped.robot_info is (allocated if has_robot else None)
    finally:
        env.close()


@pytest.fixture
def so101_tool(monkeypatch):
    from robot_mocks import mocked_sdks

    from rlinf.envs.real import so101
    from rlinf.robotics import SO101Config
    from toolkits.realworld_check import test_so101_env as tool

    for field in dataclasses.fields(SO101Config):
        monkeypatch.delenv(field.name.upper(), raising=False)
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr(so101, "SO101ReachEnv", constructor)
    monkeypatch.setattr(tool, "drive", Mock())

    def run(*flags):
        monkeypatch.setattr(sys, "argv", ["test_so101_env", *flags])
        with mocked_sdks():
            tool.main()

    return run, constructor


@pytest.mark.parametrize(
    "flags,port,calibration,player",
    [
        ([], "/dev/from-env", "from-env", False),
        (
            ["--port", "/dev/from-env", "--id", "from-cli", "--enable-camera-player"],
            "/dev/from-env",
            "from-cli",
            True,
        ),
    ],
)
def test_so101_tool_enumerates_environment_and_cli_settings(
    monkeypatch, so101_tool, flags, port, calibration, player
):
    monkeypatch.setenv("SERIAL_PORT", "/dev/from-env")
    monkeypatch.setenv("CALIBRATION_ID", "from-env")
    monkeypatch.setenv("CAMERA_SERIALS", "MOCK0001,MOCK0002")
    monkeypatch.setenv("MAX_RELATIVE_TARGET", "12")
    run, constructor = so101_tool
    run(*flags)
    config = constructor.call_args.kwargs["robot_info"].config
    assert config.serial_port == port
    assert config.calibration_id == calibration
    assert config.camera_serials == ["MOCK0001", "MOCK0002"]
    assert config.max_relative_target == 12
    assert constructor.call_args.args[0]["enable_camera_player"] is player
    constructor.return_value.close.assert_called_once_with()


@pytest.mark.parametrize("explicit_port", [False, True])
def test_so101_tool_ignores_unrelated_process_environment(
    monkeypatch, so101_tool, explicit_port
):
    from rlinf.robotics import SO101Config

    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("ID", "unrelated-service-id")
    run, constructor = so101_tool
    run(*(["--port", "/dev/ttyACM1", "--id", "follower"] if explicit_port else []))
    config = constructor.call_args.kwargs["robot_info"].config
    assert config.serial_port == (
        "/dev/ttyACM1" if explicit_port else SO101Config(node_rank=0).serial_port
    )
    assert config.calibration_id == ("follower" if explicit_port else None)


@pytest.mark.parametrize("ports", [None, "/dev/ttyACM0", "/dev/ttyACM0,/dev/ttyACM1"])
def test_so101_tool_rejects_a_port_not_in_the_configured_rig(
    monkeypatch, so101_tool, ports, capsys
):
    monkeypatch.setenv(
        "CALIBRATION_ID",
        "leader,follower" if ports and "," in ports else "leader",
    )
    if ports is not None:
        monkeypatch.setenv("SERIAL_PORT", ports)
    run, constructor = so101_tool
    with pytest.raises(SystemExit) as error:
        run("--port", "/dev/typo")
    assert error.value.code == 2
    assert "No configured SO-101 matches --port" in capsys.readouterr().err
    constructor.assert_not_called()


def test_so101_tool_selects_a_port_with_its_own_calibration(monkeypatch, so101_tool):
    monkeypatch.setenv("SERIAL_PORT", "/dev/ttyACM0,/dev/ttyACM1")
    monkeypatch.setenv("CALIBRATION_ID", "leader,follower")
    monkeypatch.setenv("MAX_RELATIVE_TARGET", "5,12")
    run, constructor = so101_tool
    run("--port", "/dev/ttyACM1")
    config = constructor.call_args.kwargs["robot_info"].config
    assert (config.serial_port, config.calibration_id, config.max_relative_target) == (
        "/dev/ttyACM1",
        "follower",
        12,
    )


def test_so101_tool_requires_selection_with_multiple_arms(monkeypatch, so101_tool):
    monkeypatch.setenv("SERIAL_PORT", "/dev/ttyACM0,/dev/ttyACM1")
    monkeypatch.setenv("CALIBRATION_ID", "leader,follower")
    run, constructor = so101_tool
    with pytest.raises(SystemExit):
        run()
    constructor.assert_not_called()


def test_dummy_franka_requires_an_explicit_camera_layout():
    from rlinf.envs.real.franka import FrankaEnv
    from rlinf.robotics import FrankaConfig, RobotInfo

    with pytest.raises(ValueError, match="including in dummy mode"):
        FrankaEnv({"is_dummy": True})
    info = RobotInfo(
        type="Franka",
        model="Franka",
        config=FrankaConfig(node_rank=0, camera_serials=["dummy"]),
    )
    env = FrankaEnv({"is_dummy": True}, robot_info=info)
    try:
        observation, _ = env.reset()
        assert observation in env.observation_space
        assert list(observation["frames"]) == ["wrist_1"]
        assert env.robot is None
    finally:
        env.close()


def test_dual_franka_hardware_defaults_build_supported_franky_grippers():
    from rlinf.robotics import DualFrankaConfig, DualFrankaRobot
    from rlinf.robotics.parts.end_effectors import EndEffector
    from rlinf.robotics.parts.end_effectors.grippers.franky import FrankyGripper

    config = DualFrankaConfig(node_rank=0)
    assert config.left_gripper_type == config.right_gripper_type == "franka"
    robot = DualFrankaRobot.build(
        left_robot_ip="192.0.2.1",
        right_robot_ip="192.0.2.2",
        left_gripper_type=config.left_gripper_type,
        right_gripper_type=config.right_gripper_type,
    )
    grippers = robot.parts_of_type(EndEffector)
    assert len(grippers) == 2
    assert all(isinstance(gripper, FrankyGripper) for gripper in grippers.values())


def test_scheduler_hardware_passes_plain_nested_settings_to_the_hand(monkeypatch):
    from omegaconf import OmegaConf
    from robot_mocks import mocked_sdks

    from rlinf.robotics import FrankaRobot
    from rlinf.scheduler.hardware import NodeHardwareConfig

    source = OmegaConf.create(
        {
            "type": "Franka",
            "configs": [
                {
                    "node_rank": 0,
                    "end_effector_type": "ruiyan_hand",
                    "end_effector_config": {
                        "port": "/dev/hand",
                        "motor_ids": [1, 2, 3, 4, 5, 6],
                        "default_state": [0.0] * 6,
                    },
                }
            ],
        }
    )
    hardware = NodeHardwareConfig(**source).configs[0]
    settings = hardware.end_effector_config
    assert type(settings) is dict
    assert type(settings["motor_ids"]) is list
    assert type(settings["default_state"]) is list
    with mocked_sdks():
        import rlinf_dexhand.ruiyan as sdk

        driver = Mock()
        monkeypatch.setattr(sdk, "RuiyanHandDriver", driver)
        hand = FrankaRobot.declare_end_effector(
            robot_ip="192.0.2.1",
            node_rank=None,
            name="test-hand",
            end_effector_type=hardware.end_effector_type,
            end_effector_config=settings,
        )
        hand.connect()
        try:
            for name, value in settings.items():
                assert driver.call_args.kwargs[name] == value
            assert type(driver.call_args.kwargs["motor_ids"]) is list
            assert type(driver.call_args.kwargs["default_state"]) is list
        finally:
            hand.disconnect()


MUJOCO_EGL_DEVICE_ID, EGL_DEVICE_ID = EGL_DEVICE_ID_ENV_VARS

_MANAGED_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", *EGL_DEVICE_ID_ENV_VARS)

# The node from the bug report: nine EGL devices, of which four are GPUs, and an
# EGL enumeration order that does not follow the CUDA one.
_EGL_INDEX_BY_CUDA_ORDINAL = {0: 2, 1: 3, 2: 0, 3: 1}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _MANAGED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    nvidia_gpu._egl_index_by_cuda_device.cache_clear()
    yield
    nvidia_gpu._egl_index_by_cuda_device.cache_clear()


@pytest.fixture(autouse=True)
def _restore_meta_path():
    saved = list(sys.meta_path)
    yield
    sys.meta_path[:] = saved


@pytest.fixture
def egl_devices(monkeypatch):
    """Enumerate the bug report's EGL topology instead of the real driver."""
    monkeypatch.setattr(
        nvidia_gpu,
        "_query_egl_index_by_cuda_ordinal",
        lambda: dict(_EGL_INDEX_BY_CUDA_ORDINAL),
    )


@pytest.fixture
def no_egl(monkeypatch):
    """A node whose driver cannot be asked about EGL devices."""

    def unavailable():
        raise OSError("libEGL.so.1: cannot open shared object file")

    monkeypatch.setattr(nvidia_gpu, "_query_egl_index_by_cuda_ordinal", unavailable)


def _become_a_rendering_worker(monkeypatch, cuda_device_id="3", egl_device_id="1"):
    """Reproduce a worker whose EGL index and CUDA device id disagree."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_device_id)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv(MUJOCO_EGL_DEVICE_ID, egl_device_id)


def _accelerator_env_var(monkeypatch, visible_accelerators: list[str]) -> dict:
    monkeypatch.setattr(nvidia_gpu, "_torch_needs_avoid_record_streams", lambda: False)
    return NvidiaGPUManager.get_accelerator_env_var(visible_accelerators)


# ---------------------------------------------------------------------------
# CUDA -> EGL mapping
# ---------------------------------------------------------------------------


def test_every_cuda_device_maps_to_its_own_egl_index(egl_devices):
    resolved = {
        cuda_id: NvidiaGPUManager.get_egl_device_id(cuda_id) for cuda_id in range(4)
    }

    assert resolved == _EGL_INDEX_BY_CUDA_ORDINAL


def test_the_mapping_accepts_the_string_ids_placement_speaks_in(egl_devices):
    assert NvidiaGPUManager.get_egl_device_id("2") == 0


def test_ordinals_are_translated_through_cuda_visible_devices(monkeypatch, egl_devices):
    # EGL_CUDA_DEVICE_NV reports a device's position in CUDA_VISIBLE_DEVICES, so
    # in a process that only sees some GPUs those ordinals are not device ids.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert NvidiaGPUManager.get_egl_device_id(4) == 2
    assert NvidiaGPUManager.get_egl_device_id(7) == 1
    # Devices this process cannot see have no readable mapping.
    assert NvidiaGPUManager.get_egl_device_id(0) is None


def test_an_unmapped_device_has_no_egl_index(egl_devices):
    assert NvidiaGPUManager.get_egl_device_id(8) is None


def test_a_uuid_device_has_no_egl_index(egl_devices):
    assert NvidiaGPUManager.get_egl_device_id("GPU-05d35c06-da01") is None


def test_the_mapping_is_empty_when_the_driver_cannot_be_queried(no_egl):
    assert NvidiaGPUManager.get_egl_device_id(0) is None


def test_the_driver_is_queried_once(monkeypatch):
    calls = []

    def counting_query():
        calls.append(None)
        return dict(_EGL_INDEX_BY_CUDA_ORDINAL)

    monkeypatch.setattr(nvidia_gpu, "_query_egl_index_by_cuda_ordinal", counting_query)

    NvidiaGPUManager.get_egl_device_id(0)
    NvidiaGPUManager.get_egl_device_id(1)

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Worker env vars
# ---------------------------------------------------------------------------


def test_the_worker_gets_the_egl_index_of_its_own_gpu(monkeypatch, egl_devices):
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert env_vars["CUDA_VISIBLE_DEVICES"] == "3"
    # A CUDA device id is not an EGL index, so it must not be passed through.
    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "1"
    assert env_vars[EGL_DEVICE_ID] == "1"


def test_a_multi_gpu_worker_renders_on_its_first_gpu(monkeypatch, egl_devices):
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, ["2", "3"])

    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "0"


@pytest.mark.parametrize("backend", ["osmesa", "glx"])
def test_cpu_rendering_sets_no_egl_device(monkeypatch, egl_devices, backend):
    monkeypatch.setenv("MUJOCO_GL", backend)

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert not [name for name in EGL_DEVICE_ID_ENV_VARS if name in env_vars]


@pytest.mark.parametrize("backend", ["", "glfw", "OSMesa"])
def test_backends_robosuite_rewrites_to_egl_get_a_device(
    monkeypatch, egl_devices, backend
):
    # robosuite 1.4.1 forces GPU rendering to EGL for every value that is not
    # literally "osmesa" or "glx", down to the casing, so anything else has to
    # be treated as EGL here too.
    monkeypatch.setenv("MUJOCO_GL", backend)

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "1"


def test_a_worker_without_gpus_gets_no_egl_device(monkeypatch, egl_devices):
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, [])

    assert not [name for name in EGL_DEVICE_ID_ENV_VARS if name in env_vars]


def test_an_unmappable_device_falls_back_to_the_cuda_id(monkeypatch, no_egl):
    # The pre-existing behaviour: right whenever the two namespaces happen to
    # agree, which is the common single-node case.
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "3"
    assert env_vars[EGL_DEVICE_ID] == "3"


# ---------------------------------------------------------------------------
# robosuite import shim
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_robosuite(tmp_path, monkeypatch):
    """A stand-in for robosuite that records the environment it was imported with."""
    bindings = tmp_path / "robosuite" / "utils"
    bindings.mkdir(parents=True)
    (tmp_path / "robosuite" / "__init__.py").write_text("")
    (bindings / "__init__.py").write_text("")
    (bindings / "binding_utils.py").write_text(
        textwrap.dedent(
            """
            import os

            SAW_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [n for n in sys.modules if n.split(".")[0] == "robosuite"]:
        del sys.modules[name]


def test_the_device_is_hidden_from_the_robosuite_import_check(
    monkeypatch, fake_robosuite
):
    # robosuite 1.4.1 asserts MUJOCO_EGL_DEVICE_ID occurs in
    # CUDA_VISIBLE_DEVICES, which a correct EGL index generally does not.
    _become_a_rendering_worker(monkeypatch)
    robosuite_compat.install_robosuite_egl_device_shim()

    bindings = importlib.import_module("robosuite.utils.binding_utils")

    assert bindings.SAW_DEVICE_ID is None
    assert os.environ[MUJOCO_EGL_DEVICE_ID] == "1"


def test_other_modules_import_with_the_device_visible(monkeypatch, fake_robosuite):
    _become_a_rendering_worker(monkeypatch)
    robosuite_compat.install_robosuite_egl_device_shim()

    assert importlib.import_module("robosuite") is not None
    assert os.environ[MUJOCO_EGL_DEVICE_ID] == "1"


def test_a_failed_robosuite_import_still_restores_the_device(
    monkeypatch, fake_robosuite
):
    _become_a_rendering_worker(monkeypatch)
    (fake_robosuite / "robosuite" / "utils" / "binding_utils.py").write_text(
        "raise RuntimeError('import failed')"
    )
    robosuite_compat.install_robosuite_egl_device_shim()

    with pytest.raises(RuntimeError, match="import failed"):
        importlib.import_module("robosuite.utils.binding_utils")

    assert os.environ[MUJOCO_EGL_DEVICE_ID] == "1"


def test_installing_the_shim_repeatedly_leaves_one_finder():
    # Importing rlinf.envs has already installed it once.
    robosuite_compat.install_robosuite_egl_device_shim()
    robosuite_compat.install_robosuite_egl_device_shim()

    finders = [
        finder
        for finder in sys.meta_path
        if isinstance(finder, robosuite_compat._RobosuiteBindingsFinder)
    ]
    assert len(finders) == 1


def test_importing_the_env_package_installs_the_shim():
    # Simulator subprocesses re-import rlinf.envs in a fresh interpreter and rely
    # on it to install the shim before any simulator is imported.
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                import rlinf.envs  # noqa: F401
                from rlinf.utils.robosuite_compat import _RobosuiteBindingsFinder

                assert any(
                    isinstance(f, _RobosuiteBindingsFinder) for f in sys.meta_path
                )
                assert "rlinf.scheduler" not in sys.modules, (
                    "the env package must not drag the scheduler into every "
                    "simulator subprocess"
                )
                """
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert child.returncode == 0, child.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("robosuite") is None, reason="robosuite is not installed"
)
def test_real_robosuite_imports_with_a_mismatched_egl_index():
    # The EGL index and the CUDA ordinal disagree here, which is what robosuite
    # 1.4.1 refuses to import with.
    env = os.environ.copy()
    env.update(
        {"CUDA_VISIBLE_DEVICES": "3", "MUJOCO_GL": "osmesa", MUJOCO_EGL_DEVICE_ID: "1"}
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import os

                import rlinf.envs  # installs the shim
                import robosuite.utils.binding_utils  # noqa: F401

                assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"
                """
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert child.returncode == 0, child.stderr


class _FakeCudaTensor:
    def torch(self):
        return torch.tensor([1.0])


class _FakePx:
    def __init__(self):
        self.cuda_articulation_link_data = _FakeCudaTensor()
        self.cuda_articulation_qacc = _FakeCudaTensor()
        self.cuda_articulation_qf = _FakeCudaTensor()
        self.cuda_articulation_qpos = _FakeCudaTensor()
        self.cuda_articulation_qvel = _FakeCudaTensor()
        self.cuda_articulation_target_qpos = _FakeCudaTensor()
        self.cuda_articulation_target_qvel = _FakeCudaTensor()
        self.cuda_rigid_body_data = _FakeCudaTensor()
        self.cuda_rigid_dynamic_data = _FakeCudaTensor()

    def gpu_update_articulation_kinematics(self):
        return None


class _FakeScene:
    def __init__(self):
        self.px = _FakePx()
        self.timestep = None

    def get_timestep(self):
        return 0.02

    def set_timestep(self, timestep):
        self.timestep = timestep

    def _gpu_apply_all(self):
        return None

    def _gpu_fetch_all(self):
        return None


class _FakeBatchedRng:
    rngs = ["rng"]


class _FakeController:
    def __init__(self):
        self.loaded_state = None

    def get_state(self):
        return {"controller": torch.tensor([1.0])}

    def set_state(self, state):
        self.loaded_state = state


class _FakeAgent:
    def __init__(self):
        self.controller = _FakeController()


class _FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.device = "cpu"
        self.scene = _FakeScene()
        self._main_rng = "main_rng"
        self._batched_main_rng = _FakeBatchedRng()
        self._main_seed = 123
        self._episode_rng = "episode_rng"
        self._batched_episode_rng = _FakeBatchedRng()
        self._episode_seed = 456
        self.action_space = "action_space"
        self.single_action_space = "single_action_space"
        self._orig_single_action_space = "orig_single_action_space"
        self._elapsed_steps = torch.tensor([3])
        self._init_raw_obs = {"obs": torch.tensor([1.0])}
        self.agent = _FakeAgent()
        self.task_reset_states = {}
        self.task_metric_states = {}
        self.reset_seed = None
        self.reset_options = None
        self.loaded_state = None

    def get_state(self):
        return {"sim": torch.tensor([2.0])}

    def reset(self, seed=None, options=None):
        self.reset_seed = seed
        self.reset_options = options

    def set_state(self, state):
        self.loaded_state = state


class _FakeManiskillEnv:
    pass


def _load_maniskill_offload_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root / "rlinf" / "envs" / "sim" / "maniskill" / "maniskill_offload_env.py"
    )

    fake_package = types.ModuleType("rlinf.envs.sim.maniskill")
    fake_package.__path__ = [str(module_path.parent)]
    fake_env_module = types.ModuleType("rlinf.envs.sim.maniskill.maniskill_env")
    fake_env_module.ManiskillEnv = _FakeManiskillEnv

    monkeypatch.setitem(sys.modules, "rlinf.envs.sim.maniskill", fake_package)
    monkeypatch.setitem(
        sys.modules, "rlinf.envs.sim.maniskill.maniskill_env", fake_env_module
    )
    monkeypatch.delitem(
        sys.modules, "rlinf.envs.sim.maniskill.maniskill_offload_env", raising=False
    )

    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.sim.maniskill.maniskill_offload_env", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _make_core(module):
    core = object.__new__(module._ManiskillEnvCore)
    core.env = _FakeEnv()
    core.seed = 10
    core.device = "cpu"
    core.prev_step_reward = torch.tensor([0.5])
    core.reset_state_ids = torch.tensor([1])
    core._generator = torch.Generator()
    core._generator.manual_seed(0)
    core.is_start = True
    core.record_metrics = False
    return core


def test_maniskill_offload_state_does_not_require_record_video_counter(monkeypatch):
    module = _load_maniskill_offload_module(monkeypatch)
    core = _make_core(module)

    state_buffer = core.get_state()
    state = torch.load(io.BytesIO(state_buffer), map_location="cpu", weights_only=False)

    assert "video_cnt" not in state


def test_maniskill_offload_load_state_accepts_state_without_video_counter(
    monkeypatch,
):
    module = _load_maniskill_offload_module(monkeypatch)
    monkeypatch.setattr(
        module, "set_batch_rng_state", lambda rng_state: _FakeBatchedRng()
    )
    core = _make_core(module)

    source_state = torch.load(
        io.BytesIO(core.get_state()),
        map_location="cpu",
        weights_only=False,
    )
    assert "video_cnt" not in source_state

    core.load_state(_serialize_state(source_state))

    assert core.env.reset_seed == core.seed
    assert core.env.reset_options == {"reconfigure": False}
    torch.testing.assert_close(core.env.loaded_state["sim"], torch.tensor([2.0]))
    assert core.env.scene.timestep == 0.02
    assert not hasattr(core, "video_cnt")


def test_maniskill_offload_load_state_ignores_legacy_video_counter(monkeypatch):
    module = _load_maniskill_offload_module(monkeypatch)
    monkeypatch.setattr(
        module, "set_batch_rng_state", lambda rng_state: _FakeBatchedRng()
    )
    core = _make_core(module)

    source_state = torch.load(
        io.BytesIO(core.get_state()),
        map_location="cpu",
        weights_only=False,
    )
    source_state["video_cnt"] = 7

    core.load_state(_serialize_state(source_state))

    assert not hasattr(core, "video_cnt")


def _serialize_state(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def test_net_emulation_config_parses_legacy_crossdc_pairs():
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "symmetric": True,
            "crossdc_pairs": [
                {"src": "Env:0", "dst": "Actor:0", "delay_ms": 10},
            ],
            "bandwidth_groups": [],
        }
    )

    net_cfg = NetEmulationConfig.from_cfg(cfg)

    assert net_cfg is not None
    assert net_cfg.crossdc_pairs == (
        CrossDCPair(src="Env:0", dst="Actor:0", delay_ms=10.0),
    )


def test_net_emulation_config_endpoint_ranges_equal_explicit_lists():
    def build_cfg(src, dst, members):
        return OmegaConf.create(
            {
                "enabled": True,
                "symmetric": True,
                "crossdc_pairs": [
                    {"src": src, "dst": dst, "delay_ms": 10},
                ],
                "bandwidth_groups": [
                    {"members": members, "bandwidth_mbps": 1000},
                ],
            }
        )

    explicit_cfg = build_cfg(
        ["Env:0", "Env:1"],
        ["Actor:0", "Actor:1"],
        ["Env:0", "Env:1", "Actor:0", "Actor:1"],
    )
    range_cfg = build_cfg(["Env:0-1"], ["Actor:0-1"], ["Env:0-1", "Actor:0-1"])

    assert NetEmulationConfig.from_cfg(range_cfg) == NetEmulationConfig.from_cfg(
        explicit_cfg
    )


@pytest.mark.parametrize("field_name", ["src", "dst"])
def test_net_emulation_config_rejects_empty_crossdc_pair_endpoint_lists(field_name):
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "symmetric": True,
            "crossdc_pairs": [
                {
                    "src": ["Env:0"],
                    "dst": ["Actor:0"],
                    "delay_ms": 10,
                },
            ],
            "bandwidth_groups": [],
        }
    )
    cfg.crossdc_pairs[0][field_name] = []

    with pytest.raises(ValueError, match=field_name):
        NetEmulationConfig.from_cfg(cfg)


def test_net_emulation_config_disabled_returns_none():
    cfg = OmegaConf.create({"enabled": False, "crossdc_pairs": []})

    assert NetEmulationConfig.from_cfg(cfg) is None
    assert NetEmulationConfig.from_cfg(None) is None


def _manager(**overrides):
    """Build a NetEmulationManager directly, bypassing the Ray actor launch."""
    cfg = {
        "enabled": True,
        "symmetric": True,
        "crossdc_pairs": [
            {"src": "EnvGroup:0", "dst": "ActorGroup:0", "delay_ms": 100}
        ],
        "bandwidth_groups": [],
    }
    cfg.update(overrides)
    return NetEmulationManager(cfg)


def test_reserve_returns_zero_for_unemulated_links():
    manager = _manager()

    assert manager.reserve("EnvGroup:0", "RolloutGroup:0", 1024) == 0.0


def test_reserve_applies_link_delay_in_both_directions():
    manager = _manager()

    assert manager.reserve("EnvGroup:0", "ActorGroup:0", 0) == pytest.approx(
        0.1, abs=0.02
    )
    # symmetric: true mirrors every configured pair.
    assert manager.reserve("ActorGroup:0", "EnvGroup:0", 0) == pytest.approx(
        0.1, abs=0.02
    )


def test_reserve_ignores_group_suffix_in_endpoint_names():
    """Endpoints may be written with or without the trailing ``Group``."""
    manager = _manager(
        crossdc_pairs=[{"src": "Env:0", "dst": "Actor:0", "delay_ms": 100}]
    )

    assert manager.reserve("EnvGroup:0", "ActorGroup:0", 0) == pytest.approx(
        0.1, abs=0.02
    )


def test_reserve_charges_transfer_time_against_the_bandwidth_budget():
    # 8 Mbps == 1 MB/s, so a 1 MB payload occupies the link for one second.
    manager = _manager(
        bandwidth_groups=[
            {"members": ["EnvGroup:0"], "bandwidth_mbps": 8},
            {"members": ["ActorGroup:0"], "bandwidth_mbps": 8},
        ]
    )

    one_mb = 1_000_000
    assert manager.reserve("EnvGroup:0", "ActorGroup:0", one_mb) == pytest.approx(
        1.1, abs=0.02
    )
    # The uplink is still busy, so a second send queues behind the first.
    assert manager.reserve("EnvGroup:0", "ActorGroup:0", one_mb) == pytest.approx(
        2.1, abs=0.02
    )


def test_estimate_payload_size_counts_tensor_storage():
    tensor = torch.zeros(256, dtype=torch.float32)  # 1024 bytes of data

    assert NetEmulationManager.estimate_payload_size_bytes(tensor) == 1024
    assert NetEmulationManager.estimate_payload_size_bytes(None) == 0


def test_estimate_payload_size_walks_nested_containers():
    payload = {"a": torch.zeros(256, dtype=torch.float32), "b": [torch.zeros(256)]}

    size = NetEmulationManager.estimate_payload_size_bytes(payload)

    # Two tensors of 1024 bytes each, plus per-tensor overhead and pickled keys.
    assert size > 2048
    assert size < 2048 + 4 * 256


def test_estimate_payload_size_falls_back_to_pickle_for_plain_objects():
    payload = {"task": "pick up the cube", "step": 7}

    size = NetEmulationManager.estimate_payload_size_bytes(payload)

    assert size == len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def test_reserve_broadcast_charges_the_uplink_once():
    # 8 Mbps == 1 MB/s on each side.
    manager = _manager(
        crossdc_pairs=[
            {"src": "Env:0", "dst": ["Actor:0", "Actor:1"], "delay_ms": 100}
        ],
        bandwidth_groups=[
            {"members": ["Env:0"], "bandwidth_mbps": 8},
            {"members": ["Actor:0", "Actor:1"], "bandwidth_mbps": 8},
        ],
    )

    one_mb = 1_000_000
    # Both receivers share one bandwidth group, so the payload crosses the link
    # once: 1s uplink + 0.1s delay, not 2s.
    assert manager.reserve_broadcast(
        "EnvGroup:0", ["ActorGroup:0", "ActorGroup:1"], one_mb
    ) == pytest.approx(1.1, abs=0.02)


def test_reserve_broadcast_waits_for_the_slowest_destination():
    manager = _manager(
        crossdc_pairs=[
            {"src": "Env:0", "dst": "Actor:0", "delay_ms": 50},
            {"src": "Env:0", "dst": "Actor:1", "delay_ms": 200},
        ],
    )

    assert manager.reserve_broadcast(
        "EnvGroup:0", ["ActorGroup:0", "ActorGroup:1"], 0
    ) == pytest.approx(0.2, abs=0.02)


def test_reserve_broadcast_ignores_unemulated_destinations():
    manager = _manager()

    assert (
        manager.reserve_broadcast("EnvGroup:0", ["RolloutGroup:0", "EnvGroup:1"], 4096)
        == 0.0
    )


def _first_eval_seeds(
    *,
    seed_count: int,
    total_num_envs: int,
    total_num_processes: int,
    group_size: int = 1,
    base_seed: int = 0,
) -> list[int]:
    success_seeds = torch.arange(seed_count, dtype=torch.long)
    selected_seeds = []
    for seed_offset in range(total_num_processes):
        num_envs = total_num_envs // total_num_processes
        num_group = num_envs // group_size
        worker_seeds = partition_success_seeds(
            success_seeds,
            base_seed=base_seed,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            num_group=num_group,
        )
        selected_seeds.extend(worker_seeds[:num_group].tolist())
    return selected_seeds


@pytest.mark.parametrize(
    ("seed_count", "total_num_envs", "total_num_processes"),
    [
        (320, 128, 4),
        (320, 128, 8),
        (320, 128, 16),
        (260, 128, 4),
        (200, 128, 8),
        (150, 128, 4),
    ],
)
def test_robotwin_eval_success_seeds_do_not_overlap_across_workers(
    seed_count: int,
    total_num_envs: int,
    total_num_processes: int,
):
    """Regression test for duplicate RoboTwin eval seeds across EnvWorkers."""
    selected_seeds = _first_eval_seeds(
        seed_count=seed_count,
        total_num_envs=total_num_envs,
        total_num_processes=total_num_processes,
    )

    assert len(selected_seeds) == total_num_envs
    assert len(set(selected_seeds)) == total_num_envs


def test_robotwin_eval_success_seed_order_is_controlled_by_base_seed():
    selected_seed_0 = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=0,
    )
    selected_seed_0_again = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=0,
    )
    selected_seed_1 = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=1,
    )

    assert selected_seed_0 == selected_seed_0_again
    assert selected_seed_0 != selected_seed_1
