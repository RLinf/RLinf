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

import os

if os.environ.get("RLINF_TIANJI_STAGE1_ONLY") == "1":
    # Keep the standalone Stage1 service importable in LoopInfra's CPU
    # environment, which intentionally does not install RLinf's full
    # Gymnasium/Torch real-world stack.
    __all__ = []
elif os.environ.get("RLINF_REALWORLD_PLATFORM") == "tianji":
    from .realworld_env import RealWorldEnv
    from .tianji import TianjiConfig, TianjiEnv
    from .tianji import tasks as tianji_tasks
    __all__ = ["RealWorldEnv", "TianjiConfig", "TianjiEnv", "tianji_tasks"]
else:
    from .a1 import A1DualEnv, A1DualRobotConfig
    from .a1 import tasks as a1_tasks
    from .dosw1 import DOSW1Config, DOSW1Env
    from .dosw1 import tasks as dosw1_tasks
    from .franka import FrankaEnv, FrankaRobotConfig, FrankaRobotState
    from .franka import tasks as franka_tasks
    from .franka.dual_franka_env import DualFrankaEnv, DualFrankaRobotConfig
    from .franka.tasks.dual_franka_joint_env import (
        DualFrankaJointEnv,
        DualFrankaJointRobotConfig,
    )
    from .franka.tasks.dual_franka_tcp_env import (
        DualFrankaTCPEnv,
        DualFrankaTCPRobotConfig,
    )
    from .gim_arm import GimArmEnv, GimArmRobotConfig, GimArmRobotState
    from .gim_arm import tasks as gim_arm_tasks
    from .realworld_env import RealWorldEnv
    from .tianji import TianjiConfig, TianjiEnv
    from .tianji import tasks as tianji_tasks
    from .xsquare import Turtle2Env, Turtle2RobotConfig, Turtle2RobotState
    from .xsquare import tasks as xsquare_tasks

    RealWorldEnv.realworld_setup()

    __all__ = [
        "A1DualEnv",
        "A1DualRobotConfig",
        "a1_tasks",
        "DualFrankaEnv",
        "DualFrankaJointEnv",
        "DualFrankaJointRobotConfig",
        "DualFrankaTCPEnv",
        "DualFrankaTCPRobotConfig",
        "DualFrankaRobotConfig",
        "DOSW1Config",
        "DOSW1Env",
        "dosw1_tasks",
        "FrankaEnv",
        "FrankaRobotConfig",
        "FrankaRobotState",
        "franka_tasks",
        "GimArmEnv",
        "GimArmRobotConfig",
        "GimArmRobotState",
        "gim_arm_tasks",
        "Turtle2Env",
        "Turtle2RobotConfig",
        "Turtle2RobotState",
        "xsquare_tasks",
        "RealWorldEnv",
        "TianjiConfig",
        "TianjiEnv",
        "tianji_tasks",
    ]
