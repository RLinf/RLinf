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

from dataclasses import dataclass, field

from rlinf.envs.realworld.xsquare.turtle2_env import Turtle2Env, Turtle2RobotConfig


@dataclass
class Turtle2DeployEnvConfig(Turtle2RobotConfig):
    use_camera_ids: list[int] = field(default_factory=lambda: [0, 1, 2])
    use_arm_ids: list[int] = field(default_factory=lambda: [0, 1])
    enforce_gripper_close: bool = False
    obs_mode: str = "x2robot_raw"
    action_mode: str = "absolute_pose"


class Turtle2DeployEnv(Turtle2Env):
    CONFIG_CLS = Turtle2DeployEnvConfig

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = self.CONFIG_CLS(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    def _calc_step_reward(self, observation) -> float:
        return 0.0
