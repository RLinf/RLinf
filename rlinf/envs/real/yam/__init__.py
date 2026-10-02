# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf-native YAM environment without a yam-abc-reproduce dependency."""

from rlinf.envs.real.registry import register_tasks
from rlinf.envs.real.yam.config import (
    DualYamJointEnvConfig,
    YamPicoConfig,
    YamResetConfig,
)
from rlinf.envs.real.yam.control_runtime import YamControlRuntime
from rlinf.envs.real.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.real.yam.pico_episode import YamPicoEpisode
from rlinf.envs.real.yam.types import (
    DualYamState,
    YamArmState,
    YamCommandResult,
)

#: Gymnasium IDs mapped to the YAM environments that build them. Registering
#: the class, rather than an entry point of its own, is what lets the shared
#: wrapper stack apply the teleop device and its episode control.
TASKS: dict[str, type] = {"DualYamJointEnv-v1": DualYamJointEnv}

_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

__all__ = [
    "DualYamJointEnv",
    "DualYamJointEnvConfig",
    "DualYamState",
    "YamArmState",
    "YamCommandResult",
    "YamControlRuntime",
    "YamPicoConfig",
    "YamPicoEpisode",
    "YamResetConfig",
    *_ENTRY_POINTS,
]
