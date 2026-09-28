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

"""Import and Gym-registration regression tests for optional YAM support."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scheduler_and_dummy_gym_env_do_not_import_i2rt():
    code = textwrap.dedent(
        """
        import builtins
        import sys

        import psutil

        # Importing rlinf.envs.real performs an existing ROS cleanup. Keep
        # this import test process-only and side-effect free.
        psutil.process_iter = lambda: ()

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "i2rt" or name.startswith("i2rt."):
                raise AssertionError(f"unexpected optional SDK import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        from rlinf.robotics.robots.dual_yam import DualYamConfig, DualYamRobot
        from rlinf.scheduler.hardware import NodeHardwareConfig
        import rlinf.envs.real.yam
        import rlinf.envs.real.yam.i2rt_backend
        import gymnasium as gym

        # A cluster config naming DualYam must resolve without the robot SDK.
        assert DualYamRobot.ROBOT_TYPE == "DualYam"
        assert NodeHardwareConfig.config_class("DualYam") is DualYamConfig
        spec = gym.spec("DualYamJointEnv-v1")
        assert spec.entry_point == (
            "rlinf.envs.real.yam:create_DualYamJointEnv"
        )
        assert not any(
            name == "i2rt" or name.startswith("i2rt.") for name in sys.modules
        )

        env = gym.make(
            "DualYamJointEnv-v1",
            override_cfg={
                "is_dummy": True,
                "image_height": 2,
                "image_width": 3,
                "dummy_camera_names": ["top_rgb"],
            },
            worker_info=None,
            robot_info=None,
            env_idx=0,
            env_cfg={"main_image_key": "top_rgb"},
        )
        observation, _ = env.reset()
        assert env.action_space.shape == (14,)
        assert observation["state"]["joint_position"].shape == (14,)
        assert observation["frames"]["top_rgb"].shape == (2, 3, 3)
        env.close()
        assert not any(
            name == "i2rt" or name.startswith("i2rt.") for name in sys.modules
        )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"YAM import subprocess failed\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
