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

"""Real-world environments, and the tasks each robot can be asked to do.

Names load on first use rather than at import. Pulling in every robot means
Gymnasium, OpenCV, SciPy, and the model stack, which a bench script driving one
teleop device over a serial port has no need for -- and a machine holding only
that device may not even have installed. Touching any name here loads
everything and registers every task, so ``from rlinf.envs.real import
RealWorldEnv`` behaves exactly as it did.
"""

import importlib
import typing

#: Exported name -> the module that defines it.
_EXPORTS: dict[str, str] = {
    name: module
    for module, names in {
        ".dosw1": ("DOSW1Config", "DOSW1Env", "PickEnv"),
        ".franka": (
            "BottleEnv",
            "DexpnpEnv",
            "DualFrankaEnv",
            "DualFrankaJointEnv",
            "DualFrankaJointRobotConfig",
            "DualFrankaRobotConfig",
            "DualFrankaTCPEnv",
            "DualFrankaTCPRobotConfig",
            "FrankaBinRelocationEnv",
            "FrankaEnv",
            "FrankaRobotConfig",
            "FrankaRobotState",
            "PegInsertionEnv",
        ),
        ".gim_arm": (
            "GimArmEnv",
            "GimArmPegInsertionEnv",
            "GimArmRobotConfig",
            "GimArmRobotState",
        ),
        ".xsquare": (
            "ButtonEnv",
            "Turtle2Env",
            "Turtle2RobotConfig",
            "Turtle2RobotState",
        ),
        ".env": ("RealWorldEnv",),
        ".task_env": ("RobotTask", "RobotTaskEnv"),
    }.items()
    for name in names
}

__all__ = sorted(_EXPORTS)

_loaded = False


def _load_all() -> None:
    """Import every robot package, registering its tasks with Gymnasium.

    Registration is a side effect of importing a robot package, so a caller
    that reaches for one env must get all of them registered -- otherwise
    ``gym.make`` of a task nobody imported would fail.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in (".dosw1", ".franka", ".gim_arm", ".xsquare", ".task_env"):
        importlib.import_module(module, __name__)
    env_module = importlib.import_module(".env", __name__)
    env_module.RealWorldEnv.realworld_setup()


def __getattr__(name: str) -> "typing.Any":
    """Load the real-world envs the first time one of their names is used."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    _load_all()
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value
