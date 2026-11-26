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

"""Registered Go2 high-level drive task."""

from typing import Any, Mapping

from rlinf.envs.real.wrappers.teleop.config import resolve_teleop_devices
from rlinf.robotics import Go2Config, RobotInfo
from rlinf.scheduler import WorkerInfo

from .base import Go2Env, Go2EnvConfig


class Go2DriveEnv(Go2Env):
    """Exercise state reception and body velocity commands without training."""

    @classmethod
    def validate_env_config(
        cls, override_cfg: Mapping[str, Any], env_cfg: Mapping[str, Any]
    ) -> list[Any]:
        """Check the stop boundary and teleop selection before opening DDS."""
        outer = env_cfg.get("max_episode_steps")
        inner = override_cfg.get("max_num_steps", Go2EnvConfig.max_num_steps)
        if outer is not None and isinstance(inner, int) and outer < inner:
            raise ValueError(
                f"Go2 max_episode_steps={outer} ends before the stop at "
                f"max_num_steps={inner}. Set max_episode_steps to null or at "
                "least max_num_steps."
            )
        return resolve_teleop_devices(
            env_cfg, supported=cls.TELEOP, default=cls.TELEOP_DEFAULT
        )

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: WorkerInfo | None = None,
        robot_info: RobotInfo[Go2Config] | None = None,
        env_idx: int = 0,
    ) -> None:
        super().__init__(Go2EnvConfig(**override_cfg), worker_info, robot_info, env_idx)

    @property
    def task_description(self) -> str:
        """Return the task name used by language-conditioned policies."""
        return "drive the Go2 mobile base"
