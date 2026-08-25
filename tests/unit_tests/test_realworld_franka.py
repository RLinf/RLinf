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
from rlinf.envs.realworld.franka.tasks.dual_franka_tcp_env import DualFrankaTCPEnv
from rlinf.envs.realworld.realworld_env import RealWorldEnv


class _FakeDualCamera:
    def __init__(self) -> None:
        self._camera_info = CameraInfo(
            "base_0_rgb",
            "base-camera",
            resolution=(8, 6),
            enable_depth=True,
        )
        self.depth_scale = 0.001
        self.read_count = 0

    def get_frame(self, timeout: float) -> np.ndarray:
        del timeout
        self.read_count += 1
        color = np.zeros((6, 8, 3), dtype=np.uint16)
        color[..., 0] = 10
        color[..., 1] = 20
        color[..., 2] = 30
        depth = np.full((6, 8, 1), 500, dtype=np.uint16)
        return np.concatenate([color, depth], axis=-1)

    def get_color_intrinsics(self) -> dict[str, float]:
        return {"fx": 100.0, "fy": 101.0, "ppx": 4.0, "ppy": 3.0}


def test_dual_franka_caches_raw_rgbd_with_policy_frame():
    env = object.__new__(DualFrankaTCPEnv)
    camera = _FakeDualCamera()
    env._cameras = [camera]
    env._last_camera_frame = {}
    env._raw_camera_frames = {}
    env._raw_camera_depths = {}
    env._raw_camera_meta = {}
    env.camera_player = SimpleNamespace(put_frame=lambda frames: None)
    env.observation_space = gym.spaces.Dict(
        {
            "frames": gym.spaces.Dict(
                {
                    "base_0_rgb": gym.spaces.Box(
                        0, 255, shape=(4, 4, 3), dtype=np.uint8
                    )
                }
            )
        }
    )

    policy_frames = env._get_camera_frames()
    snapshot = env.get_raw_camera_snapshot()

    assert policy_frames["base_0_rgb"].shape == (4, 4, 3)
    assert snapshot["raw_frames"]["base_0_rgb"].shape == (6, 8, 3)
    np.testing.assert_array_equal(
        snapshot["raw_frames"]["base_0_rgb"][0, 0], [30, 20, 10]
    )
    np.testing.assert_allclose(snapshot["raw_depths"]["base_0_rgb"], 0.5)
    assert snapshot["camera_meta"]["base_0_rgb"]["color_intrinsics"]["fx"] == 100
    assert camera.read_count == 1


def test_dual_franka_enables_depth_only_for_supported_cameras():
    env = object.__new__(DualFrankaTCPEnv)
    env.config = SimpleNamespace(
        enable_camera_depth=True,
        camera_type=None,
        base_camera_serials=["base"],
        base_camera_type="realsense",
        left_camera_serials=["left"],
        left_camera_type="lumos",
        right_camera_serials=["right"],
        right_camera_type="lumos",
    )

    infos = env._camera_infos()

    assert [(info.name, info.enable_depth) for info in infos] == [
        ("base_0_rgb", True),
        ("left_wrist_0_rgb", False),
        ("right_wrist_0_rgb", False),
    ]


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


def test_generic_franka_registrations_exist():
    assert gym.spec("FrankaEnv-v1") is not None
    assert gym.spec("DualFrankaTcpEnv-v1") is not None
    assert gym.spec("DualFrankaTCPEnv-v1") is not None
