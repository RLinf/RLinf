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

import importlib
import os
from typing import Any

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
    # Lazy-load platform submodules so importing e.g. `rlinf.envs.realworld.tianji.*`
    # does not eagerly pull legacy `dosw1`/`franka`/`gim_arm`/`xsquare` platforms
    # whose `*HWInfo` classes moved to `rlinf.envs.real` in origin/main.
    _EXPORTS: dict[str, tuple[str, str | None]] = {
        "A1DualEnv": (".a1", "A1DualEnv"),
        "A1DualRobotConfig": (".a1", "A1DualRobotConfig"),
        "a1_tasks": (".a1.tasks", None),
        "DualFrankaEnv": (".franka.dual_franka_env", "DualFrankaEnv"),
        "DualFrankaJointEnv": (
            ".franka.tasks.dual_franka_joint_env",
            "DualFrankaJointEnv",
        ),
        "DualFrankaJointRobotConfig": (
            ".franka.tasks.dual_franka_joint_env",
            "DualFrankaJointRobotConfig",
        ),
        "DualFrankaTCPEnv": (
            ".franka.tasks.dual_franka_tcp_env",
            "DualFrankaTCPEnv",
        ),
        "DualFrankaTCPRobotConfig": (
            ".franka.tasks.dual_franka_tcp_env",
            "DualFrankaTCPRobotConfig",
        ),
        "DualFrankaRobotConfig": (".franka.dual_franka_env", "DualFrankaRobotConfig"),
        "DOSW1Config": (".dosw1", "DOSW1Config"),
        "DOSW1Env": (".dosw1", "DOSW1Env"),
        "dosw1_tasks": (".dosw1.tasks", None),
        "FrankaEnv": (".franka", "FrankaEnv"),
        "FrankaRobotConfig": (".franka", "FrankaRobotConfig"),
        "FrankaRobotState": (".franka", "FrankaRobotState"),
        "franka_tasks": (".franka.tasks", None),
        "GimArmEnv": (".gim_arm", "GimArmEnv"),
        "GimArmRobotConfig": (".gim_arm", "GimArmRobotConfig"),
        "GimArmRobotState": (".gim_arm", "GimArmRobotState"),
        "gim_arm_tasks": (".gim_arm.tasks", None),
        "Turtle2Env": (".xsquare", "Turtle2Env"),
        "Turtle2RobotConfig": (".xsquare", "Turtle2RobotConfig"),
        "Turtle2RobotState": (".xsquare", "Turtle2RobotState"),
        "xsquare_tasks": (".xsquare.tasks", None),
        "RealWorldEnv": (".realworld_env", "RealWorldEnv"),
        "TianjiConfig": (".tianji", "TianjiConfig"),
        "TianjiEnv": (".tianji", "TianjiEnv"),
        "tianji_tasks": (".tianji.tasks", None),
    }
    _setup_done = False

    def __getattr__(name: str) -> Any:
        global _setup_done
        target = _EXPORTS.get(name)
        if target is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        if name == "RealWorldEnv" and not _setup_done:
            _setup_done = True
            env_mod = importlib.import_module(".realworld_env", __name__)
            env_mod.RealWorldEnv.realworld_setup()
        mod_name, attr = target
        mod = importlib.import_module(mod_name, __name__)
        value = mod if attr is None else getattr(mod, attr)
        globals()[name] = value
        return value

    __all__ = list(_EXPORTS.keys())
