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

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

import rlinf.envs.realworld.franka.tasks  # noqa: F401
from rlinf.envs.realworld.common.camera import CameraInfo
from rlinf.envs.realworld.franka.franka_env import FrankaEnv
from rlinf.envs.realworld.franka.tasks.physical_agent_env import (
    PhysicalAgentFrankaConfig,
)
from rlinf.envs.realworld.realworld_env import RealWorldEnv


def test_franka_projection_metadata_matches_center_crop_and_resize():
    env = object.__new__(FrankaEnv)
    env.config = SimpleNamespace(enable_camera_depth=True, camera_resize=True)
    metadata = env._camera_projection_metadata(
        camera_info=CameraInfo("wrist", "camera-1", resolution=(640, 480)),
        raw_intrinsics={
            "width": 640,
            "height": 480,
            "fx": 600.0,
            "fy": 600.0,
            "ppx": 320.0,
            "ppy": 240.0,
        },
        output_size=(128, 128),
        depth_scale=0.001,
    )

    assert metadata["crop_bounds_xyxy"] == [80, 0, 560, 480]
    np.testing.assert_allclose(
        metadata["intrinsic_K"],
        [[160.0, 0.0, 64.0], [0.0, 160.0, 64.0], [0.0, 0.0, 1.0]],
    )


def test_franka_projection_metadata_preserves_raw_resolution_without_resize():
    env = object.__new__(FrankaEnv)
    env.config = SimpleNamespace(enable_camera_depth=True, camera_resize=False)
    metadata = env._camera_projection_metadata(
        camera_info=CameraInfo("wrist", "camera-1", resolution=(640, 480)),
        raw_intrinsics={
            "width": 640,
            "height": 480,
            "fx": 600.0,
            "fy": 610.0,
            "ppx": 320.0,
            "ppy": 240.0,
        },
        output_size=(640, 480),
        depth_scale=0.001,
    )

    assert metadata["crop_bounds_xyxy"] == [0, 0, 640, 480]
    np.testing.assert_allclose(
        metadata["intrinsic_K"],
        [[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]],
    )


def test_realworld_wrap_obs_preserves_depth_camera_order():
    env = object.__new__(RealWorldEnv)
    env.main_image_key = "wrist"
    env.task_descriptions = ["test task"]
    raw_obs = {
        "state": {"tcp_pose": np.zeros((1, 7), dtype=np.float32)},
        "frames": {
            "wrist": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "external": np.ones((1, 2, 2, 3), dtype=np.uint8),
        },
        "depths": {
            "wrist": np.full((1, 2, 2), 0.5, dtype=np.float32),
            "external": np.full((1, 2, 2), 1.5, dtype=np.float32),
        },
    }

    obs = env._wrap_obs(raw_obs)

    assert tuple(obs["main_depths"].shape) == (1, 2, 2)
    assert tuple(obs["extra_view_depths"].shape) == (1, 1, 2, 2)
    np.testing.assert_allclose(obs["main_depths"].cpu().numpy(), 0.5)
    np.testing.assert_allclose(obs["extra_view_depths"].cpu().numpy(), 1.5)


def test_physical_agent_config_derives_reset_and_safety_bounds():
    target = np.array([0.5, 0.1, 0.2, 3.0, 0.0, 0.25])
    config = PhysicalAgentFrankaConfig(
        target_ee_pose=target,
        clip_x_range=0.2,
        clip_y_range=0.3,
        clip_z_range_low=0.04,
        clip_z_range_high=0.1,
        clip_roll_pitch_range=0.05,
        clip_rz_range=0.4,
        compliance_param={"translational_clip_x": 0.005},
    )

    np.testing.assert_allclose(config.reset_ee_pose, target + [0, 0, 0.1, 0, 0, 0])
    np.testing.assert_allclose(
        config.ee_pose_limit_min,
        [0.3, -0.2, 0.16, 2.95, -0.05, -0.15],
    )
    np.testing.assert_allclose(
        config.ee_pose_limit_max,
        [0.7, 0.4, 0.3, 3.05, 0.05, 0.65],
    )
    assert config.compliance_param["translational_clip_x"] == 0.005


def test_rpent_franka_registrations_exist():
    assert gym.spec("PhysicalAgentFrankaEnv-v1") is not None
    assert gym.spec("DualFrankaTcpEnv-v1") is not None
    assert gym.spec("DualFrankaTCPEnv-v1") is not None
